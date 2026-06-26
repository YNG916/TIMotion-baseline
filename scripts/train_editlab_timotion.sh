#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

EXP_NAME="${EXP_NAME:-editlab_timotion_baseline}"
EPOCHS="${EPOCHS:-1500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-123}"

N_HEAD="${N_HEAD:-16}"
N_LAYER="${N_LAYER:-5}"
LATENT_DIM="${LATENT_DIM:-512}"
DROP_OUT="${DROP_OUT:-0.1}"
CONV_LAYERS="${CONV_LAYERS:-1}"
DILATION_RATE="${DILATION_RATE:-1}"
NORM="${NORM:-AdaLN}"
USE_LPA="${USE_LPA:-1}"
RESUME="${RESUME:-}"

mkdir -p checkpoints logs/train

echo "TIMotion EditLab training"
echo "repo=${REPO_ROOT}"
echo "exp=${EXP_NAME} epochs=${EPOCHS} batch_size=${BATCH_SIZE} lr=${LR}"

cmd=(
  python tools/train.py
  --exp-name "${EXP_NAME}"
  --epoch "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --lr "${LR}"
  --num-workers "${NUM_WORKERS}"
  --seed "${SEED}"
  --n-head "${N_HEAD}"
  --n-layer "${N_LAYER}"
  --latent-dim "${LATENT_DIM}"
  --drop-out "${DROP_OUT}"
  --conv-layers "${CONV_LAYERS}"
  --dilation-rate "${DILATION_RATE}"
  --norm "${NORM}"
)

if [[ "${USE_LPA}" == "1" || "${USE_LPA}" == "true" || "${USE_LPA}" == "True" ]]; then
  cmd+=(--LPA)
fi

if [[ -n "${RESUME}" ]]; then
  cmd+=(--resume "${RESUME}")
fi

"${cmd[@]}" 2>&1 | tee -a "logs/train/${EXP_NAME}.log"
