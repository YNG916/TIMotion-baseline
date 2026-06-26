#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PTH="${1:-${PTH:-}}"
if [[ -z "${PTH}" ]]; then
  echo "Usage: $0 /path/to/checkpoint.ckpt"
  echo "Or set PTH=/path/to/checkpoint.ckpt"
  exit 2
fi

EXP_NAME="${EXP_NAME:-editlab_timotion_eval}"
BATCH_SIZE="${BATCH_SIZE:-96}"
DIVERSITY_TIMES="${DIVERSITY_TIMES:-300}"
N_REPEAT="${N_REPEAT:-20}"

N_HEAD="${N_HEAD:-16}"
N_LAYER="${N_LAYER:-5}"
LATENT_DIM="${LATENT_DIM:-512}"
CONV_LAYERS="${CONV_LAYERS:-1}"
DILATION_RATE="${DILATION_RATE:-1}"
NORM="${NORM:-AdaLN}"
USE_LPA="${USE_LPA:-1}"

mkdir -p eval_log logs/eval

echo "TIMotion EditLab evaluation"
echo "repo=${REPO_ROOT}"
echo "checkpoint=${PTH}"
echo "exp=${EXP_NAME} repeats=${N_REPEAT} batch_size=${BATCH_SIZE}"

cmd=(
  python tools/eval.py
  --pth "${PTH}"
  --exp-name "${EXP_NAME}"
  --batch-size "${BATCH_SIZE}"
  --diversity-times "${DIVERSITY_TIMES}"
  --n-repeat "${N_REPEAT}"
  --n-head "${N_HEAD}"
  --n-layer "${N_LAYER}"
  --latent-dim "${LATENT_DIM}"
  --conv-layers "${CONV_LAYERS}"
  --dilation-rate "${DILATION_RATE}"
  --norm "${NORM}"
)

if [[ "${USE_LPA}" == "1" || "${USE_LPA}" == "true" || "${USE_LPA}" == "True" ]]; then
  cmd+=(--LPA)
fi

"${cmd[@]}" 2>&1 | tee -a "logs/eval/${EXP_NAME}.log"
