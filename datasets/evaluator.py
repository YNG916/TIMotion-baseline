from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
import copy
import random
import numpy as np
import torch
from tqdm import tqdm

from datasets import InterHumanDataset
from datasets.evaluator_models import InterCLIP
from utils.utils import MotionNormalizer


class GroundTruthTargetDataset(Dataset):
    """
    Wrap editing dataset:
      base __getitem__ returns:
        name, text, src1, src2, tgt1, tgt2, src_len, tgt_len
      here we output only target as ground truth:
        name, text, tgt1, tgt2, tgt_len
    """
    def __init__(self, base_dataset: InterHumanDataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        name, text, src1, src2, tgt1, tgt2, src_len, tgt_len = self.base[idx]
        return name, text, tgt1, tgt2, tgt_len


class EvaluationDataset(Dataset):
    """
    Generate edited motions conditioned on (sources, text).
    Output format to EvaluatorModelWrapper:
      ("generated", text, gen1, gen2, tgt_len)
    """
    def __init__(self, model, dataset, device, mm_num_samples, mm_num_repeats, shuffle=True):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()

        self.dataset = dataset
        self.max_length = dataset.max_gt_length if hasattr(dataset, "max_gt_length") else dataset.max_length

        dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=shuffle)

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = set(idxs[:mm_num_samples])

        generated_motions = []
        mm_generated_motions = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                # editing dataset item:
                # name, text, src1, src2, tgt1, tgt2, src_len, tgt_len
                name, text, src1, src2, tgt1, tgt2, src_len, tgt_len = data

                text0 = text[0]
                rep = mm_num_repeats if i in mm_idxs else 1

                # src1/src2: (1, maxT, D)  D should be 262 per person after process_motion_np
                src1_t = src1.float()
                src2_t = src2.float()
                sources = torch.cat([src1_t, src2_t], dim=-1)  # (1, maxT, 2D)

                tgt_len_i = int(tgt_len[0].item())
                src_len_i = int(src_len[0].item())
                T = min(self.max_length, tgt_len_i)

                sources = sources[:, :T, :]

                if rep > 1:
                    sources = sources.repeat(rep, 1, 1)
                    texts = [text0] * rep
                    motion_lens = torch.LongTensor([T] * rep)
                    source_lens = torch.LongTensor([min(T, src_len_i)] * rep)
                else:
                    texts = [text0]
                    motion_lens = torch.LongTensor([T])
                    source_lens = torch.LongTensor([min(T, src_len_i)])

                batch = {
                    "text": texts,
                    "sources": sources.to(device),
                    "motion_lens": motion_lens.to(device),   # generation length
                    "source_lens": source_lens.to(device),
                }

                out_batch = self.model.forward_test(batch)
                output = out_batch["output"]  # (rep, T, 2*D)

                motions_output = output.reshape(output.shape[0], output.shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())  # (rep,T,2,D)

                # pad to max_length for evaluator
                B, Tcur = motions_output.shape[0], motions_output.shape[1]
                if Tcur < self.max_length:
                    pad = np.zeros((B, self.max_length - Tcur, 2, motions_output.shape[-1]),
                                   dtype=motions_output.dtype)
                    motions_output = np.concatenate([motions_output, pad], axis=1)

                assert motions_output.shape[1] == self.max_length

                sub_dict = {
                    'motion1': motions_output[0, :, 0],
                    'motion2': motions_output[0, :, 1],
                    'motion_lens': np.array(T, dtype=np.int64),
                    'text': text0
                }
                generated_motions.append(sub_dict)

                if rep > 1:
                    mm_sub_dict = {
                        'mm_motions': motions_output,  # (rep, maxT, 2, D)
                        'motion_lens': np.array(T, dtype=np.int64),
                        'text': text0
                    }
                    mm_generated_motions.append(mm_sub_dict)

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset: EvaluationDataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']  # (rep, maxT, 2, D)
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:, :, 0]
        mm_motions2 = mm_motions[:, :, 1]
        text = data['text']
        motion_lens = np.array([motion_lens] * mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size, shuffle=True):
    """
    Returns:
      gt_loader: yields (name, text, tgt1, tgt2, tgt_len)
      base_dataset: editing dataset (provides src/tgt for generation)
    """
    opt = copy.deepcopy(opt)
    if opt.NAME == 'interhuman':
        print('Loading dataset %s ...' % opt.NAME)
        base_dataset = InterHumanDataset(opt)
        gt_dataset = GroundTruthTargetDataset(base_dataset)
        dataloader = DataLoader(gt_dataset, batch_size=batch_size, num_workers=0,
                                drop_last=True, shuffle=shuffle)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth (TARGET) Dataset Loading Completed!!!')
    return dataloader, base_dataset


def get_motion_loader(batch_size, model, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, shuffle=True):
    """
    ground_truth_dataset here is the BASE editing dataset (InterHumanDataset).
    """
    dataset = EvaluationDataset(model, ground_truth_dataset, device,
                               mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                               shuffle=shuffle)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=shuffle)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print('Generated (EDITED) Dataset Loading Completed!!!')
    return motion_loader, mm_motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin('eval_model/interclip.ckpt'), map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model


class EvaluatorModelWrapper(object):
    def __init__(self, cfg, device):
        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1).detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {
                "text": text,
                "motions": motions.reshape(B, T, -1)[:, :padded_len],
                "motion_lens": motion_lens.to(self.device),
            }

            motion_embedding = self.model.encode_motion(batch)['motion_emb']
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()
            motion2 = motion2.detach().float()
            motions = torch.cat([motion1, motion2], dim=-1).detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, int(m_len)) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {
                "text": text,
                "motions": motions.reshape(B, T, -1)[:, :padded_len],
                "motion_lens": motion_lens.to(self.device),
            }

            motion_embedding = self.model.encode_motion(batch)['motion_emb']
        return motion_embedding
