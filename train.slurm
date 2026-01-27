#!/bin/bash
#SBATCH --account=hk-project-pai00111
#SBATCH -J 01-27-2026-train-timotion
#SBATCH -p accelerated
#SBATCH --gres=gpu:4
#SBATCH -c 152
#SBATCH --mem=497500M
#SBATCH -t 1-00:00:00
#SBATCH -o logs/train/%x_%j.out
#SBATCH -e logs/train/%x_%j.err
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
mkdir -p checkpoints logs/train

source ~/miniconda3/etc/profile.d/conda.sh
conda activate intergen_test

# env info
python -V
nvidia-smi

# start job（Lightning在Slurm下推荐用srun）
srun python tools/train.py --LPA --epoch 1500 --exp-name 01-27-2026-timotion 2>&1 | tee -a logs/train/train_${SLURM_JOB_ID}.log

echo "Done: $(date)"
