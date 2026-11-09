#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09,gnode14
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --job-name=osvi_nut_resume
#SBATCH --output=/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda/mtlfd_adaptation/slurm_logs/nut_resume_%j.out
#SBATCH --export=ALL

set -euo pipefail

ROOT="/mnt/beegfs/f.demaio28/osvi_awda_finetune"
REPO="$ROOT/repos/osvi-awda"
FRAMEWORK="$ROOT/repos/Multi-Task-LFD-Training-Framework"

# Directory del training Nut Assembly originale
RUN_DIR="${RUN_DIR:-$ROOT/checkpoints/nut_assembly_human_training/nut_assembly_human_only_heldout_0_4_8/osvi_mtlfd_ckpt-1793893283}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

export NUMBA_CUDA_USE_NVIDIA_BINDING=1
export CUDA_HOME="$ROOT/cuda117_toolkit"
export PYTHONPATH="$REPO:$FRAMEWORK/training:$FRAMEWORK/training/multi_task_il"
export PYTHONUNBUFFERED=1

cd "$REPO"

# Trova l'ultimo checkpoint con modello e ottimizzatore.
# Non seleziona model_save-optim-N.pt come modello.
CHECKPOINT="$(python - "$RUN_DIR" <<'PY'
import re
import sys
from pathlib import Path

run = Path(sys.argv[1])

if not run.is_dir():
    raise SystemExit("Directory training non trovata: {}".format(run))

if not (run / "config.yaml").is_file():
    raise SystemExit("config.yaml mancante in {}".format(run))

candidates = []

for p in run.glob("model_save-*.pt"):
    match = re.fullmatch(r"model_save-(\d+)\.pt", p.name)
    if match is None:
        continue

    step = int(match.group(1))
    optim = run / "model_save-optim-{}.pt".format(step)

    if (
        optim.is_file()
        and p.stat().st_size > 0
        and optim.stat().st_size > 0
    ):
        candidates.append((step, p))

if not candidates:
    raise SystemExit(
        "Nessuna coppia checkpoint modello/ottimizzatore trovata"
    )

step, checkpoint = max(candidates, key=lambda item: item[0])
print(checkpoint)
PY
)"

test -f "$CHECKPOINT"

STEP="$(basename "$CHECKPOINT" | sed -E 's/model_save-([0-9]+)\.pt/\1/')"
OPTIM="$RUN_DIR/model_save-optim-${STEP}.pt"

echo "=========================================="
echo "OSVI-AWDA NUT ASSEMBLY - RESUME TRAINING"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Python: $(which python)"
echo "Run directory: $RUN_DIR"
echo "Checkpoint: $CHECKPOINT"
echo "Optimizer: $OPTIM"
echo "Resume step: $STEP"
echo "Config: $RUN_DIR/config.yaml"
echo "=========================================="

nvidia-smi

echo ""
echo "===== START RESUME ====="

python -u mtlfd_adaptation/train_mtlfd.py \
    "$CHECKPOINT" \
    --resume \
    --workers 8 \
    --device 0

echo "===== RESUME COMPLETED ====="
