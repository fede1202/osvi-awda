#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_10wp_resume
#SBATCH --output=mtlfd_adaptation/slurm_logs/human_10wp_resume_%j.out

set -euo pipefail

REPO="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda"

CKPT_DIR="/mnt/beegfs/f.demaio28/osvi_awda_finetune/checkpoints/human_only_10wp_latent220_scratch/pick_place_human_only_10wp_latent220_scratch_heldout_0_5_10_15/osvi_mtlfd_ckpt-1793464241"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

cd "$REPO"

export PYTHONPATH="$REPO:$REPO/training/multi_task_il:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="/mnt/beegfs/f.demaio28/osvi_awda_finetune/cuda117_toolkit"

# Select the most recent checkpoint with its optimizer
CKPT=""

while IFS= read -r candidate; do
    filename=$(basename "$candidate")
    step="${filename#model_save-}"
    step="${step%.pt}"

    optim="$CKPT_DIR/model_save-optim-${step}.pt"

    if [[ -s "$candidate" && -s "$optim" ]]; then
        CKPT="$candidate"
        break
    fi
done < <(
    find "$CKPT_DIR" -maxdepth 1 \
      -regextype posix-extended \
      -regex '.*/model_save-[0-9]+\.pt' | sort -V -r
)

if [[ -z "$CKPT" ]]; then
    echo "ERROR: No matching model and optimizer checkpoint found."
    exit 1
fi

test -f "$CKPT_DIR/config.yaml"

echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "Resuming checkpoint: $CKPT"

nvidia-smi

srun python -u mtlfd_adaptation/train_mtlfd_10waypoints.py \
    "$CKPT" \
    --resume \
    --seed 0 \
    --workers 8 \
    --device 0
