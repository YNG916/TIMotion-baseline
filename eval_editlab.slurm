#!/bin/bash
#SBATCH --account=hk-project-pai00111
#SBATCH -J eval-editlab-timotion
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH --mem=128000M
#SBATCH -t 04:00:00
#SBATCH -o logs/eval/%x_%j.out
#SBATCH -e logs/eval/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yebinyang916@gmail.com

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

if [[ -z "${PTH:-}" ]]; then
  echo "Set PTH=/path/to/checkpoint.ckpt before sbatch."
  exit 2
fi

mkdir -p eval_log logs/eval

echo "JobID=$SLURM_JOB_ID Node=$SLURM_JOB_NODELIST Start=$(date)"
echo "PWD=$(pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-intergen_test}"

python -V
nvidia-smi

export EXP_NAME="${EXP_NAME:-editlab_timotion_eval}"
export BATCH_SIZE="${BATCH_SIZE:-96}"
export DIVERSITY_TIMES="${DIVERSITY_TIMES:-300}"
export N_REPEAT="${N_REPEAT:-20}"
export USE_LPA="${USE_LPA:-1}"

srun bash scripts/eval_editlab_timotion.sh "${PTH}"

echo "Done: $(date)"
