#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=osvi_offline
#SBATCH --output=offline_debug_%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

REPO="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda"
cd "$REPO"

export PYTHONPATH="$REPO:$REPO/training/multi_task_il:${PYTHONPATH:-}"

# CUDA/Numba compatibility
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="/mnt/beegfs/f.demaio28/osvi_awda_finetune/cuda117_toolkit"

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"

srun python -u offline_debug_firstframe_osvi.py \
    --config /mnt/beegfs/f.demaio28/osvi_awda_finetune/checkpoints/human_only_training/pick_place_human_only_heldout_0_5_10_15/osvi_mtlfd_ckpt-1793356468/config.yaml \
    --checkpoint /mnt/beegfs/f.demaio28/osvi_awda_finetune/checkpoints/human_only_training/pick_place_human_only_heldout_0_5_10_15/osvi_mtlfd_ckpt-1793356468/model_save-68000.pt \
    --robot-root /mnt/beegfs/f.demaio28/osvi_awda_finetune/data/opt_dataset/pick_place/real_eye_in_hand_ur5e_pick_place \
    --demo-root /mnt/beegfs/f.demaio28/osvi_awda_finetune/data/opt_dataset/pick_place/human_rgb_pick_place \
    --output /mnt/beegfs/f.demaio28/osvi_awda_finetune/offline_debug_robot_5wp_68_crop_correct \
    --task-ids 1 4 8 12 \
    --max-files 200 \
    --demo-index 0 \
    --seed 0 \
    --device cuda:0