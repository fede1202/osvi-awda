import os
import ctypes.util
import traceback

import torch
import numba
from numba import cuda
from numba.cuda.cudadrv import driver, nvvm

print("=== ENVIRONMENT ===", flush=True)
print("Torch:", torch.__version__, flush=True)
print("Numba:", numba.__version__, flush=True)

for name in (
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    "CUDA_PATH",
    "LD_LIBRARY_PATH",
    "NUMBA_CUDA_DRIVER",
    "NUMBA_CUDA_DEFAULT_PTX_CC",
):
    print("{}: {}".format(name, os.environ.get(name)), flush=True)

print("\n=== LIBRARY DISCOVERY ===", flush=True)
print("libcuda:", ctypes.util.find_library("cuda"), flush=True)
print("libnvvm:", ctypes.util.find_library("nvvm"), flush=True)

def test(name, fn):
    print("\n=== {} ===".format(name), flush=True)
    try:
        result = fn()
        print("OK:", result, flush=True)
    except Exception:
        traceback.print_exc()

test("Torch CUDA", lambda: torch.cuda.is_available())
test("Numba driver available", lambda: driver.driver.is_available)
test("Numba NVVM available", lambda: nvvm.is_available())
test("Numba CUDA available", lambda: cuda.is_available())
test("Numba detect", lambda: cuda.detect())

def driver_info():
    lib = driver.driver.lib
    return getattr(lib, "_name", str(lib))

test("Loaded driver library", driver_info)
