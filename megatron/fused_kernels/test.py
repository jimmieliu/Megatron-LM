import torch
import pathlib
import os
from torch.utils import cpp_extension
import time

srcpath = pathlib.Path(os.path.abspath("."))
buildpath = srcpath / 'testbuild'
os.environ["TORCH_CUDA_ARCH_LIST"] = ""
def _cpp_extention_load_helper(name, sources, extra_cuda_flags):
    return cpp_extension.load(
        name=name,
        sources=sources,
        build_directory=buildpath,
        extra_cflags=['-O3',],
        extra_cuda_cflags=['-O3',
                           '--use_fast_math'] + extra_cuda_flags,

        verbose=True
    )

extra_cuda_flags = ['-U__CUDA_NO_HALF_OPERATORS__',
                    '-U__CUDA_NO_HALF_CONVERSIONS__',
                    '--expt-relaxed-constexpr',
                    '--expt-extended-lambda',
                    ]

# Upper triangular softmax.
sources=[srcpath / 'scaled_upper_triang_masked_softmax.cpp',
         srcpath / 'scaled_upper_triang_masked_softmax_cuda.cu']

st = time.time()
print(st)
scaled_upper_triang_masked_softmax_cuda = _cpp_extention_load_helper(
    "scaled_upper_triang_masked_softmax_cuda",
    sources, extra_cuda_flags)
print("Compile used", time.time() - st)
