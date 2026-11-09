#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode14
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --job-name=osvi_nut_offline
#SBATCH --output=mtlfd_adaptation/slurm_logs/nut_offline_%j.out
#SBATCH --export=ALL

set -euo pipefail

ROOT="/mnt/beegfs/f.demaio28/osvi_awda_finetune"
REPO="$ROOT/repos/osvi-awda"
FRAMEWORK="$ROOT/repos/Multi-Task-LFD-Training-Framework"

DATA="$ROOT/data/nut_assembly"
CKPT_ROOT="$ROOT/checkpoints/nut_assembly_human_training"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

export PYTHONPATH="$REPO:$FRAMEWORK/training:$FRAMEWORK/training/multi_task_il"
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="$ROOT/cuda117_toolkit"
export PYTHONUNBUFFERED=1

cd "$REPO"

if [[ -z "${CHECKPOINT:-}" ]]; then
    CHECKPOINT="$(find "$CKPT_ROOT" -type f -name 'model_save-[0-9]*.pt' ! -name '*optim*' | sort -V | tail -1)"
fi

if [[ -z "$CHECKPOINT" || ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint non trovato"
    exit 1
fi

CONFIG="$(dirname "$CHECKPOINT")/config.yaml"

if [[ ! -f "$CONFIG" ]]; then
    CONFIG="$REPO/mtlfd_adaptation/experiments/nut_assembly_human_only_heldout_0_4_8.yaml"
fi

OUTPUT="${OUTPUT:-$ROOT/offline_debug_nut_assembly_5wp}"
TASK_IDS="${TASK_IDS:-0 4 8}"
MAX_FILES="${MAX_FILES:-0}"
DEMO_INDEX="${DEMO_INDEX:-0}"

read -r -a TASK_LIST <<< "$TASK_IDS"

mkdir -p "$OUTPUT"

echo "===== NUT ASSEMBLY OFFLINE INFERENCE ====="
echo "Node: $(hostname)"
echo "Checkpoint: $CHECKPOINT"
echo "Config: $CONFIG"
echo "Task IDs: $TASK_IDS"
echo "Max files: $MAX_FILES"
echo "Output: $OUTPUT"

nvidia-smi

python -u utils/offline_debug_firstframe_nut.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --robot-root "${ROBOT_ROOT:-$DATA/nut_assembly_agent}" \
    --demo-root "$DATA/nut_assembly_demo" \
    --output "$OUTPUT" \
    --task-ids "${TASK_LIST[@]}" \
    --max-files "$MAX_FILES" \
    --flat-task-id "${FLAT_TASK_ID:-0}" \
    --demo-index "$DEMO_INDEX" \
    --seed 0 \
    --device cuda:0
