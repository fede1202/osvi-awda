#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_human_only
#SBATCH --output=mtlfd_adaptation/slurm_logs/human_only_%j.out
#SBATCH --export=ALL

set -euo pipefail

REPO="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda"
CONFIG="mtlfd_adaptation/experiments/pick_place_human_only_heldout_0_5_10_15.yaml"
BASE="/mnt/beegfs/f.demaio28/osvi_awda_finetune/checkpoints"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

# CUDA 11.7 / Numba compatibility
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="/mnt/beegfs/f.demaio28/osvi_awda_finetune/cuda117_toolkit"

cd "$REPO"

export PYTHONPATH="$REPO:$REPO/training/multi_task_il:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "===== OSVI-AWDA HUMAN-ONLY ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Mode: ${RUN_MODE:-train}"
echo "Configuration: $CONFIG"

nvidia-smi

if [[ "${RUN_MODE:-train}" == "smoke" ]]; then
    SAVE_PARENT="$BASE/human_only_smoke"

    echo "Starting limited-duration training verification"

    set +e
    timeout --signal=TERM --kill-after=30s 600s \
        srun python -u mtlfd_adaptation/train_mtlfd.py \
            "$CONFIG" \
            --save-parent "$SAVE_PARENT" \
            --workers 8 \
            --device 0

    status=$?
    set -e

    if [[ "$status" -eq 124 ]]; then
        echo "SMOKE TEST: time limit reached (expected). Inspect losses above."
        exit 0
    fi

    exit "$status"
else
    SAVE_PARENT="$BASE/human_only_training"

    srun python -u mtlfd_adaptation/train_mtlfd.py \
        "$CONFIG" \
        --save-parent "$SAVE_PARENT" \
        --workers 8 \
        --device 0
fi
