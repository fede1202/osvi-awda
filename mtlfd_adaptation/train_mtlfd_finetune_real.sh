#!/bin/bash
# Fine-tuning OSVI-AWDA on the new real UR5e dataset.

#SBATCH -A did_robot_learning_359
#SBATCH --qos=did_robot_learning_359_gpuq_qos
#SBATCH --partition=gpuq
#SBATCH --exclude=gnode09
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --export=ALL
#SBATCH --job-name=osvi_mtlfd_finetune
#SBATCH --output=mtlfd_adaptation/slurm_logs/train_finetune_%j.out

set -e

CONDA_ENV=osvi_mtlfd_fdemaio
REPO_ROOT="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/osvi-awda"

SAVE_PATH="${SAVE_PATH:-/mnt/beegfs/f.demaio28/osvi_awda_finetune/outputs/osvi_mtlfd}"

EXPERIMENT_FILE="${EXPERIMENT_FILE:-mtlfd_adaptation/experiments/pick_place_mtlfd_finetune_real.yaml}"

WORKERS=8

export EXPERT_DATA=/mnt/beegfs/f.demaio28/osvi_awda_finetune/data/opt_dataset

export PYTHONPATH="/mnt/beegfs/f.demaio28/osvi_awda_finetune/repos/Multi-Task-LFD-Training-Framework/training:${PYTHONPATH:-}"

export CUDA117_PREFIX="/mnt/beegfs/f.demaio28/osvi_awda_finetune/tools/cuda-11.7"
export CUDA_HOME="$CUDA117_PREFIX"
export CUDA_PATH="$CUDA117_PREFIX"
export LD_LIBRARY_PATH="$CUDA117_PREFIX/lib:${LD_LIBRARY_PATH:-}"

export NUMBA_CUDA_USE_NVIDIA_BINDING=1

mkdir -p "$REPO_ROOT/mtlfd_adaptation/slurm_logs" "$SAVE_PATH"
cd "$REPO_ROOT"

module load anaconda/3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

srun python -u mtlfd_adaptation/train_mtlfd.py \
    "$EXPERIMENT_FILE" \
    --save-parent "$SAVE_PATH" \
    --workers $WORKERS \
    "$@"
