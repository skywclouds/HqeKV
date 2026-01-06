from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


extra_compile_args = {
    "cxx": [
        "-g", # 这是一个调试选项
        "-O3", # 这是一个优化选项
        "-fopenmp", # 启用OpenMP支持
        "-lgomp", # 链接到OpenMP库
        "-std=c++17",# 指定C++语言版本
        "-DENABLE_BF16"
    ],
    "nvcc": [
        "-O3", # 这是一个优化选项
        "-std=c++17", # C++语言版本
        "-DENABLE_BF16",
        "-U__CUDA_NO_HALF_OPERATORS__",# 允许 nvcc 使用半精度浮点数的操作
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT162_OPERATORS__",
        "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
        "--expt-relaxed-constexpr",# 允许在设备代码中使用更灵活的常量表达式
        "--expt-extended-lambda",# 允许在CUDA内核中使用扩展的lambda表达式
        "--use_fast_math",# 启用了快速数学库
        "--threads=8"# 指定了每个CUDA块（block）中的线程数
    ],
}

setup(
    name="kvhq_gemv",
    packages=find_packages(),
    # 定义了一个列表，包含了扩展模块的定义
    ext_modules=[
        CUDAExtension(
            name="kvhq_gemv",
            sources=[
                "csrc/pybind.cpp", 
                "csrc/gemv_cuda_uniform.cu",
                "csrc/gemv_cuda_uniform_outlier.cu",
                "csrc/gemv_cuda_wv_uniform.cu",
                "csrc/gemv_cuda_wv_uniform_outlier.cu",
                "csrc/gemv_cuda_normal.cu",
                "csrc/gemv_cuda_normal_outlier.cu",
                "csrc/gemv_cuda_wv_normal.cu",
                "csrc/gemv_cuda_wv_normal_outlier.cu",
            ],
            extra_compile_args=extra_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch"],
)