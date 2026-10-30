import sys
import traceback

import torch
from numba import cuda
import numba

print("Python:", sys.version, flush=True)
print("PyTorch:", torch.__version__, flush=True)
print("Numba:", numba.__version__, flush=True)
print("Torch CUDA:", torch.version.cuda, flush=True)

def check(label, fn):
    print("\n=== {} ===".format(label), flush=True)
    try:
        result = fn()
        print("OK:", result, flush=True)
        return True
    except Exception:
        traceback.print_exc()
        return False

check("Torch CUDA available", lambda: torch.cuda.is_available())
check("Numba CUDA available", lambda: cuda.is_available())
check("Numba context", lambda: cuda.current_context())

def torch_allocation():
    x = torch.ones((10, 10), device="cuda:0")
    torch.cuda.synchronize()
    return x

x = None
try:
    x = torch_allocation()
    print("\nTorch allocation OK:", x.device, flush=True)
except Exception:
    traceback.print_exc()

if x is not None:
    check(
        "Numba reads Torch tensor",
        lambda: cuda.as_cuda_array(x)
    )

@cuda.jit
def add_one(arr):
    i = cuda.grid(1)
    if i < arr.size:
        arr[i] += 1

def run_kernel():
    arr = cuda.to_device(torch.ones(10).numpy())
    add_one[1, 32](arr)
    cuda.synchronize()
    return arr.copy_to_host()

check("Numba CUDA kernel", run_kernel)
