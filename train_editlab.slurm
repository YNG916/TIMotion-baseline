#!/bin/bash
#SBATCH --account=hk-project-test-p0028273
#SBATCH -J train-editlab-timotion
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH -c 152
#SBATCH --mem=497500M
#SBATCH -t 2-00:00:00
#SBATCH -o logs/train/%x_%j.out
#SBATCH -e logs/train/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yebinyang916@gmail.com

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

mkdir -p checkpoints logs/train

echo "JobID=$SLURM_JOB_ID Node=$SLURM_JOB_NODELIST Start=$(date)"
echo "PWD=$(pwd)"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-intergen_test}"

python -V
nvidia-smi

export EXP_NAME="${EXP_NAME:-editlab_timotion_baseline}"
export EPOCHS="${EPOCHS:-1500}"
export BATCH_SIZE="${BATCH_SIZE:-32}"
export LR="${LR:-1e-4}"
export NUM_WORKERS="${NUM_WORKERS:-2}"
export USE_LPA="${USE_LPA:-1}"

srun bash scripts/train_editlab_timotion.sh

echo "Done: $(date)"
