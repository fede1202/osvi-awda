#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_human_10wp
#SBATCH --output=mtlfd_adaptation/slurm_logs/human_10wp_%j.out

set -euo pipefail

REPO="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda"
BASE="/mnt/beegfs/f.demaio28/osvi_awda_finetune/checkpoints"

CONFIG="mtlfd_adaptation/experiments/pick_place_human_only_10wp_latent220_scratch_heldout_0_5_10_15.yaml"

SAVE_PARENT="$BASE/human_only_10wp_latent220_scratch"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

cd "$REPO"

export PYTHONPATH="$REPO:$REPO/training/multi_task_il:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="/mnt/beegfs/f.demaio28/osvi_awda_finetune/cuda117_toolkit"

test -f "$CONFIG"
mkdir -p "$SAVE_PARENT" mtlfd_adaptation/slurm_logs

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Config: $CONFIG"
echo "Save parent: $SAVE_PARENT"
echo "Pretrained ResNet18: yes"
echo "Previous OSVI-AWDA checkpoint: NONE"

nvidia-smi

srun python -u mtlfd_adaptation/train_mtlfd_10waypoints.py \
    "$CONFIG" \
    --save-parent "$SAVE_PARENT" \
    --seed 0 \
    --workers 8 \
    --device 0
