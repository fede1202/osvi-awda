#!/bin/bash
#SBATCH -A did_robot_learning_359
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:10:00
#SBATCH --job-name=test_cuda
#SBATCH --output=test_cuda_%j.out

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate osvi_mtlfd_fdemaio

echo "===== SLURM ====="
echo "Node: $(hostname)"
echo "Job: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-UNSET}"
echo "SLURM_JOB_GPUS: ${SLURM_JOB_GPUS:-UNSET}"
echo "SLURM_STEP_GPUS: ${SLURM_STEP_GPUS:-UNSET}"

echo "===== NVIDIA ====="
nvidia-smi

echo "===== PYTORCH CUDA TEST ====="

srun python -u - <<'PY'
import torch

print("PyTorch:", torch.__version__, flush=True)
print("Compiled CUDA:", torch.version.cuda, flush=True)
print("CUDA available:", torch.cuda.is_available(), flush=True)
print("Device count:", torch.cuda.device_count(), flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable inside SLURM allocation")

print("GPU:", torch.cuda.get_device_name(0), flush=True)

x = torch.randn(1024, 1024, device="cuda:0")
y = x @ x
torch.cuda.synchronize()

print("GPU matrix multiplication OK:", y.shape, flush=True)
PY
