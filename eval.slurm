#!/bin/bash
#SBATCH --account=hk-project-pai00111
#SBATCH -J 01-28-2026-eval-timotion
#SBATCH -p accelerated
#SBATCH --gres=gpu:1
#SBATCH -c 12
#SBATCH -t 05:00:00
#SBATCH -o logs/eval/%x_%j.out
#SBATCH -e logs/eval/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yebinyang916@gmail.com


set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

echo "JobID=$SLURM_JOB_ID  Node=$SLURM_JOB_NODELIST  Start=$(date)"
echo "PWD=$(pwd)"

# 减少CPU线程抢占
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# 确保在repo根目录跑
mkdir -p checkpoints logs/eval

source ~/miniconda3/etc/profile.d/conda.sh
conda activate intergen_test

# env info
python -V
nvidia-smi

# start job（Lightning在Slurm下推荐用srun）
srun python tools/eval.py \
  --pth ./checkpoints/TIMotion/model/epoch=1499-step=87000.ckpt \
  --exp-name 01-28-2026-eval-timotion \
  --LPA

echo "Done: $(date)"
