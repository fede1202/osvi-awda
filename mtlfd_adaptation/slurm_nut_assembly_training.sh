#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode14
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_nut_train
#SBATCH --output=mtlfd_adaptation/slurm_logs/nut_training_%j.out
#SBATCH --export=ALL

set -euo pipefail

ROOT="/mnt/beegfs/f.demaio28/osvi_awda_finetune"
REPO="$ROOT/repos/osvi-awda"
FRAMEWORK="$ROOT/repos/Multi-Task-LFD-Training-Framework"

CONFIG="mtlfd_adaptation/experiments/nut_assembly_human_only_heldout_0_4_8.yaml"
SAVE_PARENT="$ROOT/checkpoints/nut_assembly_human_training"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="$ROOT/cuda117_toolkit"

export PYTHONPATH="$REPO:$FRAMEWORK/training:$FRAMEWORK/training/multi_task_il"
export PYTHONUNBUFFERED=1

cd "$REPO"

echo "===== OSVI-AWDA NUT ASSEMBLY TRAINING ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Config: $CONFIG"
echo "Checkpoints: $SAVE_PARENT"

nvidia-smi

python -u mtlfd_adaptation/train_mtlfd.py \
    "$CONFIG" \
    --save-parent "$SAVE_PARENT" \
    --workers 8 \
    --device 0
