import sys
sys.path.append(sys.path[0] + r"/../")

import os
import numpy as np
import torch
from datetime import datetime
from collections import OrderedDict
from tqdm import tqdm
import argparse

from datasets import get_dataset_motion_loader, get_motion_loader
from models import *
from utils.metrics import *
from configs import get_config

os.environ['WORLD_SIZE'] = '1'
os.environ['RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
torch.multiprocessing.set_sharing_strategy('file_system')


def build_models(cfg):
    if cfg.NAME == "TIMotion":
        model = TIMotion(cfg)
    return model


def evaluate_matching_score(motion_loaders, file, eval_wrapper, top_k=3):
    match_score_dict = OrderedDict({})
    R_precision_dict = OrderedDict({})
    activation_dict = OrderedDict({})

    print('========== Evaluating MM Distance / R@k ==========')
    print('========== Evaluating MM Distance / R@k ==========', file=file, flush=True)

    for motion_loader_name, motion_loader in motion_loaders.items():
        all_motion_embeddings = []
        mm_dist_sum = 0.0
        all_size = 0
        top_k_count = np.zeros(top_k, dtype=np.float64)

        with torch.no_grad():
            for _, batch in tqdm(enumerate(motion_loader), total=len(motion_loader)):
                text_embeddings, motion_embeddings = eval_wrapper.get_co_embeddings(batch)

                dist_mat = euclidean_distance_matrix(
                    text_embeddings.cpu().numpy(),
                    motion_embeddings.cpu().numpy()
                )
                mm_dist_sum += dist_mat.trace()

                argsmax = np.argsort(dist_mat, axis=1)
                top_k_mat = calculate_top_k(argsmax, top_k=top_k)
                top_k_count += top_k_mat.sum(axis=0)

                all_size += text_embeddings.shape[0]
                all_motion_embeddings.append(motion_embeddings.cpu().numpy())

        all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)
        mm_dist = mm_dist_sum / max(1, all_size)
        R_precision = top_k_count / max(1, all_size)

        match_score_dict[motion_loader_name] = mm_dist
        R_precision_dict[motion_loader_name] = R_precision
        activation_dict[motion_loader_name] = all_motion_embeddings

        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}')
        print(f'---> [{motion_loader_name}] MM Distance: {mm_dist:.4f}', file=file, flush=True)

        line = f'---> [{motion_loader_name}] R_precision: '
        for i in range(len(R_precision)):
            line += '(top %d): %.4f ' % (i + 1, R_precision[i])
        print(line)
        print(line, file=file, flush=True)

    return match_score_dict, R_precision_dict, activation_dict


def evaluate_fid(groundtruth_loader, activation_dict, file, eval_wrapper):
    eval_dict = OrderedDict({})
    gt_motion_embeddings = []

    print('========== Evaluating FID ==========')
    print('========== Evaluating FID ==========', file=file, flush=True)

    with torch.no_grad():
        for _, batch in tqdm(enumerate(groundtruth_loader), total=len(groundtruth_loader)):
            motion_embeddings = eval_wrapper.get_motion_embeddings(batch)
            gt_motion_embeddings.append(motion_embeddings.cpu().numpy())

    gt_motion_embeddings = np.concatenate(gt_motion_embeddings, axis=0)
    gt_mu, gt_cov = calculate_activation_statistics(gt_motion_embeddings)

    for model_name, motion_embeddings in activation_dict.items():
        mu, cov = calculate_activation_statistics(motion_embeddings)
        fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
        eval_dict[model_name] = fid

        print(f'---> [{model_name}] FID: {fid:.4f}')
        print(f'---> [{model_name}] FID: {fid:.4f}', file=file, flush=True)

    return eval_dict


def evaluate_diversity(activation_dict, file, diversity_times=300):
    eval_dict = OrderedDict({})
    print('========== Evaluating Diversity ==========')
    print('========== Evaluating Diversity ==========', file=file, flush=True)

    for model_name, motion_embeddings in activation_dict.items():
        diversity = calculate_diversity(motion_embeddings, diversity_times)
        eval_dict[model_name] = diversity
        print(f'---> [{model_name}] Diversity: {diversity:.4f}')
        print(f'---> [{model_name}] Diversity: {diversity:.4f}', file=file, flush=True)

    return eval_dict


def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval


def evaluation(log_file, gt_loader, eval_motion_loaders, eval_wrapper,
               replication_times=20, diversity_times=300, top_k=3):
    with open(log_file, 'w') as f:
        all_metrics = OrderedDict({
            'MM Distance': OrderedDict({}),
            'R_precision': OrderedDict({}),
            'FID': OrderedDict({}),
            'Diversity': OrderedDict({}),
        })

        for replication in range(replication_times):
            motion_loaders = {}
            motion_loaders['ground truth'] = gt_loader

            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader, _ = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)

            mat_score_dict, R_precision_dict, acti_dict = evaluate_matching_score(
                motion_loaders, f, eval_wrapper, top_k=top_k
            )

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            fid_score_dict = evaluate_fid(gt_loader, acti_dict, f, eval_wrapper)

            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            div_score_dict = evaluate_diversity(acti_dict, f, diversity_times=diversity_times)

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

            for key, item in mat_score_dict.items():
                all_metrics['MM Distance'].setdefault(key, []).append(item)
            for key, item in R_precision_dict.items():
                all_metrics['R_precision'].setdefault(key, []).append(item)
            for key, item in fid_score_dict.items():
                all_metrics['FID'].setdefault(key, []).append(item)
            for key, item in div_score_dict.items():
                all_metrics['Diversity'].setdefault(key, []).append(item)

        for metric_name, metric_dict in all_metrics.items():
            print('========== %s Summary ==========' % metric_name)
            print('========== %s Summary ==========' % metric_name, file=f, flush=True)

            for model_name, values in metric_dict.items():
                mean, conf_interval = get_metric_statistics(np.array(values), replication_times)
                if isinstance(mean, (np.float64, np.float32, float)):
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}')
                    print(f'---> [{model_name}] Mean: {mean:.4f} CInterval: {conf_interval:.4f}', file=f, flush=True)
                elif isinstance(mean, np.ndarray):
                    line = f'---> [{model_name}] '
                    for i in range(len(mean)):
                        line += '(top %d) Mean: %.4f CInt: %.4f; ' % (i + 1, mean[i], conf_interval[i])
                    print(line)
                    print(line, file=f, flush=True)


def get_args_parser():
    parser = argparse.ArgumentParser(description='TIMotion Editing Evaluation',
                                     add_help=True,
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--exp-name', default='TIMotion_edit', type=str)
    parser.add_argument("--pth", type=str, default=None, help='checkpoint ckpt')
    parser.add_argument('--n-repeat', default=20, type=int)
    parser.add_argument('--diversity-times', default=300, type=int)
    parser.add_argument('--top-k', default=3, type=int)

    # keep these to match original style; they affect model cfg, not eval_model
    parser.add_argument('--n-head', default=8, type=int)
    parser.add_argument('--n-layer', default=8, type=int)
    parser.add_argument('--LPA', action='store_true')
    parser.add_argument('--conv-layers', default=1, type=int)
    parser.add_argument('--dilation-rate', default=1, type=int)
    parser.add_argument("--norm", type=str, default='AdaLN', choices=['AdaLN', 'LN', 'BN', 'GN'])
    parser.add_argument('--latent-dim', default=1024, type=int)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args_parser()

    mm_num_samples = 100
    mm_num_repeats = 30

    replication_times = args.n_repeat
    diversity_times = args.diversity_times
    top_k = args.top_k

    batch_size = 96
    device = torch.device("cuda")

    data_cfg = get_config("configs/datasets.yaml").interhuman_test

    # GT loader = target motions
    gt_loader, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size)

    # evaluator model wrapper uses eval_model.yaml (INPUT_DIM=258)
    evalmodel_cfg = get_config("configs/eval_model.yaml")
    from datasets import EvaluatorModelWrapper
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    # load TIMotion editing model
    model_cfg = get_config("configs/model.yaml")
    model_cfg.CHECKPOINT = args.pth
    model_cfg.NUM_HEADS = args.n_head
    model_cfg.NUM_LAYERS = args.n_layer
    model_cfg.LATENT_DIM = args.latent_dim
    model_cfg.LPA = args.LPA
    model_cfg.conv_layers = args.conv_layers
    model_cfg.dilation_rate = args.dilation_rate
    model_cfg.norm = args.norm

    model = build_models(model_cfg).to(device)
    checkpoint = torch.load(model_cfg.CHECKPOINT, map_location=torch.device("cpu"))
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()

    eval_motion_loaders = {
        model_cfg.NAME: (lambda m=model: get_motion_loader(
            batch_size=batch_size,
            model=m,
            ground_truth_dataset=gt_dataset,  # base editing dataset
            device=device,
            mm_num_samples=mm_num_samples,
            mm_num_repeats=mm_num_repeats
        ))
    }

    os.makedirs("./eval_log", exist_ok=True)
    log_file = f'./eval_log/evaluation_{args.exp_name}.log'

    evaluation(
        log_file=log_file,
        gt_loader=gt_loader,
        eval_motion_loaders=eval_motion_loaders,
        eval_wrapper=eval_wrapper,
        replication_times=replication_times,
        diversity_times=diversity_times,
        top_k=top_k
    )
