# setup.py for DCN (v1)
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")

setup(
    name="deform_conv_cuda",
    ext_modules=[
        CUDAExtension(
            name="deform_conv_cuda",
            sources=[
                os.path.join(src_dir, "deform_conv_cuda.cpp"),
                os.path.join(src_dir, "deform_conv_cuda_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "--expt-relaxed-constexpr"],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension}
)
