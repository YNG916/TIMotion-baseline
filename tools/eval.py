# tools/eval.py
import sys
sys.path.append(sys.path[0] + r"/../")

import os
import argparse
import numpy as np
import torch
from datetime import datetime
from tqdm import tqdm

from configs import get_config
from datasets import get_dataset_motion_loader
from datasets.evaluator import get_edit_motion_loader, EvaluatorModelWrapper
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity

from models import TIMotion

"""
cd TIMotion-baseline

python tools/eval.py \
  --pth ./checkpoints/TIMotion/model/epoch=1499-step=87000.ckpt \
  --exp-name 01-28-2026-eval-timotion \
  --LPA

"""

def build_models(cfg):
    if cfg.NAME == "TIMotion":
        return TIMotion(cfg)
    raise KeyError(f"Unknown model NAME={cfg.NAME}")


def _r_at_k(sim_mat: np.ndarray, k: int) -> float:
    idx_sorted = np.argsort(-sim_mat, axis=1)
    gt = np.arange(sim_mat.shape[0])[:, None]
    hit = (idx_sorted[:, :k] == gt).any(axis=1)
    return float(hit.mean())


@torch.no_grad()
def _collect_motion_embeds(eval_wrapper, loader):
    all_e = []
    for batch in tqdm(loader, desc="Collect embeddings"):
        e = eval_wrapper.get_motion_embeddings(batch)  # (B,512)
        e = torch.nn.functional.normalize(e, dim=-1)
        all_e.append(e.cpu().numpy())
    return np.concatenate(all_e, axis=0)


def eval_once(model, gt_dataset, eval_wrapper, batch_size, diversity_times):
    gen_loader, tgt_loader, src_loader = get_edit_motion_loader(
        batch_size=batch_size,
        model=model,
        ground_truth_dataset=gt_dataset,
        device=next(model.parameters()).device,
    )

    E_g = _collect_motion_embeds(eval_wrapper, gen_loader)
    E_t = _collect_motion_embeds(eval_wrapper, tgt_loader)
    E_s = _collect_motion_embeds(eval_wrapper, src_loader)

    # Retrieval
    S_g2t = E_g @ E_t.T
    S_g2s = E_g @ E_s.T

    g2t = np.array([_r_at_k(S_g2t, 1), _r_at_k(S_g2t, 2), _r_at_k(S_g2t, 3)])
    g2s = np.array([_r_at_k(S_g2s, 1), _r_at_k(S_g2s, 2), _r_at_k(S_g2s, 3)])

    # FID: gen vs target
    mu_t, cov_t = calculate_activation_statistics(E_t)
    mu_g, cov_g = calculate_activation_statistics(E_g)
    fid = float(calculate_frechet_distance(mu_t, cov_t, mu_g, cov_g))

    # Diversity
    div_gen = float(calculate_diversity(E_g, diversity_times))
    div_tgt = float(calculate_diversity(E_t, diversity_times))

    return {
        "g2t_R": g2t,
        "g2s_R": g2s,
        "FID": fid,
        "Diversity_gen": div_gen,
        "Diversity_tgt": div_tgt,
        "N": int(E_g.shape[0]),
    }


def get_args():
    p = argparse.ArgumentParser("TIMotion Editing Evaluation (g2t/g2s R@1/2/3, FID, Diversity)")

    p.add_argument("--pth", type=str, required=True, help="checkpoint .ckpt path")
    p.add_argument("--exp-name", type=str, default="editlab_timotion_eval")
    p.add_argument("--batch-size", type=int, default=96)
    p.add_argument("--diversity-times", type=int, default=300)
    p.add_argument("--n-repeat", type=int, default=20, help="repeat times (re-generate motions each repeat)")

    # MUST match training model config
    p.add_argument("--latent-dim", type=int, default=512)
    p.add_argument("--n-layer", type=int, default=5)
    p.add_argument("--n-head", type=int, default=16)
    p.add_argument("--LPA", action="store_true")
    p.add_argument("--conv-layers", type=int, default=1)
    p.add_argument("--dilation-rate", type=int, default=1)
    p.add_argument("--norm", type=str, default="AdaLN", choices=["AdaLN", "LN", "BN", "GN"])

    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset cfg (test split)
    data_cfg = get_config("configs/datasets.yaml").interhuman_test
    _, gt_dataset = get_dataset_motion_loader(data_cfg, batch_size=args.batch_size)

    # evaluator model (InterCLIP)
    evalmodel_cfg = get_config("configs/eval_model.yaml")
    eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, device)

    # model cfg
    model_cfg = get_config("configs/model.yaml")
    model_cfg.CHECKPOINT = args.pth
    model_cfg.LATENT_DIM = args.latent_dim
    model_cfg.NUM_LAYERS = args.n_layer
    model_cfg.NUM_HEADS = args.n_head
    model_cfg.LPA = args.LPA
    model_cfg.conv_layers = args.conv_layers
    model_cfg.dilation_rate = args.dilation_rate
    model_cfg.norm = args.norm

    model = build_models(model_cfg).to(device)

    ckpt = torch.load(args.pth, map_location="cpu")
    state = ckpt.get("state_dict", ckpt)

    # strip "model." prefix (Lightning)
    new_state = {}
    for k, v in state.items():
        if k.startswith("model."):
            new_state[k.replace("model.", "", 1)] = v
        else:
            new_state[k] = v

    model.load_state_dict(new_state, strict=True)
    model.eval()

    os.makedirs("./eval_log", exist_ok=True)
    log_path = f"./eval_log/eval_edit_{args.exp_name}.log"

    all_scores = []
    with open(log_path, "w") as f:
        for r in range(args.n_repeat):
            print(f"\n===== Repeat {r}  Time: {datetime.now()} =====")
            print(f"\n===== Repeat {r}  Time: {datetime.now()} =====", file=f, flush=True)

            scores = eval_once(
                model=model,
                gt_dataset=gt_dataset,
                eval_wrapper=eval_wrapper,
                batch_size=args.batch_size,
                diversity_times=args.diversity_times,
            )
            all_scores.append(scores)

            line1 = f"N={scores['N']}  g2t R@1/2/3: {scores['g2t_R'][0]:.4f} {scores['g2t_R'][1]:.4f} {scores['g2t_R'][2]:.4f}"
            line2 = f"     g2s R@1/2/3: {scores['g2s_R'][0]:.4f} {scores['g2s_R'][1]:.4f} {scores['g2s_R'][2]:.4f}"
            line3 = f"     FID(gen||tgt): {scores['FID']:.4f}"
            line4 = f"     Diversity(gen): {scores['Diversity_gen']:.4f}   Diversity(tgt): {scores['Diversity_tgt']:.4f}"

            print(line1); print(line2); print(line3); print(line4)
            print(line1, file=f, flush=True)
            print(line2, file=f, flush=True)
            print(line3, file=f, flush=True)
            print(line4, file=f, flush=True)
        
        def mean_ci(arr, n):
            arr = np.asarray(arr)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            ci = 1.96 * std / np.sqrt(n)
            return mean, ci

        # ----- after the for r in range(args.n_repeat): loop -----
        # stack metrics
        g2t = np.stack([s["g2t_R"] for s in all_scores], axis=0)          # (R,3)
        g2s = np.stack([s["g2s_R"] for s in all_scores], axis=0)          # (R,3)
        fid = np.array([s["FID"] for s in all_scores], dtype=np.float64)  # (R,)
        dgen = np.array([s["Diversity_gen"] for s in all_scores], dtype=np.float64)
        dtgt = np.array([s["Diversity_tgt"] for s in all_scores], dtype=np.float64)

        m_g2t, ci_g2t = mean_ci(g2t, args.n_repeat)
        m_g2s, ci_g2s = mean_ci(g2s, args.n_repeat)
        m_fid, ci_fid = mean_ci(fid, args.n_repeat)
        m_dgen, ci_dgen = mean_ci(dgen, args.n_repeat)
        m_dtgt, ci_dtgt = mean_ci(dtgt, args.n_repeat)

        summary = []
        summary.append("\n========== Summary (Mean ± 95% CI) ==========")
        summary.append(f"Repeats: {args.n_repeat}")
        summary.append(
            "g2t R@1/2/3: "
            f"{m_g2t[0]:.4f}±{ci_g2t[0]:.4f}  {m_g2t[1]:.4f}±{ci_g2t[1]:.4f}  {m_g2t[2]:.4f}±{ci_g2t[2]:.4f}"
        )
        summary.append(
            "g2s R@1/2/3: "
            f"{m_g2s[0]:.4f}±{ci_g2s[0]:.4f}  {m_g2s[1]:.4f}±{ci_g2s[1]:.4f}  {m_g2s[2]:.4f}±{ci_g2s[2]:.4f}"
        )
        summary.append(f"FID(gen||tgt): {float(m_fid):.4f}±{float(ci_fid):.4f}")
        summary.append(f"Diversity(gen): {float(m_dgen):.4f}±{float(ci_dgen):.4f}")
        summary.append(f"Diversity(tgt): {float(m_dtgt):.4f}±{float(ci_dtgt):.4f}")

        for line in summary:
            print(line)
            print(line, file=f, flush=True)
            
    print(f"\nSaved log to: {log_path}")


if __name__ == "__main__":
    main()
