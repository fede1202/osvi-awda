#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode14
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:20:00
#SBATCH --job-name=osvi_nut_smoke
#SBATCH --output=mtlfd_adaptation/slurm_logs/nut_smoke_%j.out
#SBATCH --export=ALL

set -euo pipefail

ROOT="/mnt/beegfs/f.demaio28/osvi_awda_finetune"
REPO="$ROOT/repos/osvi-awda"
FRAMEWORK="$ROOT/repos/Multi-Task-LFD-Training-Framework"

CONFIG="mtlfd_adaptation/experiments/nut_assembly_human_only_heldout_0_4_8.yaml"
SAVE_PARENT="$ROOT/checkpoints/nut_assembly_smoke"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

# CUDA 11.7 / Numba compatibility
export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="$ROOT/cuda117_toolkit"

cd "$REPO"

# OSVI-AWDA prima del framework per risolvere utils correttamente
export PYTHONPATH="$REPO:$FRAMEWORK/training:$FRAMEWORK/training/multi_task_il"

export PYTHONUNBUFFERED=1

echo "=========================================="
echo "OSVI-AWDA NUT ASSEMBLY SMOKE TEST"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Config: $CONFIG"
echo "Save parent: $SAVE_PARENT"
echo "=========================================="

nvidia-smi

# Verifica configurazione e batch bilanciato
python - <<'PY'
import yaml

path = "mtlfd_adaptation/experiments/nut_assembly_human_only_heldout_0_4_8.yaml"

with open(path) as f:
    cfg = yaml.safe_load(f)

tasks = cfg["dataset"]["train_tasks"]
samples_per_task = cfg["samples_per_task"]

assert len(tasks) == 6
assert samples_per_task == 4
assert cfg["batch_size"] == 24

print("Training tasks:", tasks)
print("Samples per task:", samples_per_task)
print("Effective batch size:", len(tasks) * samples_per_task)
print("Person augmentation:", cfg["dataset"]["person_aug"])
print("Joystick augmentation:", cfg["dataset"]["joystick_aug"])
print("CONFIGURATION OK")
PY

echo ""
echo "===== START SMOKE TRAINING ====="

mkdir -p "$SAVE_PARENT"

set +e

timeout --signal=TERM --kill-after=30s 600s \
    srun python -u mtlfd_adaptation/train_mtlfd.py \
        "$CONFIG" \
        --save-parent "$SAVE_PARENT" \
        --workers 4 \
        --device 0

STATUS=$?

set -e

echo ""
echo "===== SMOKE TEST RESULT ====="
echo "Exit code: $STATUS"

if [[ "$STATUS" -eq 124 ]]; then
    echo "SMOKE TEST: 10-minute limit reached."
    echo "Inspect the log to verify that optimizer steps and losses were produced."
    exit 0
elif [[ "$STATUS" -eq 0 ]]; then
    echo "SMOKE TEST: training process terminated normally."
    exit 0
else
    echo "SMOKE TEST FAILED: training error."
    exit "$STATUS"
fi
