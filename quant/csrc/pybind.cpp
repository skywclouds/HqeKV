#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "gemv_cuda.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  // 将名为gemv_forward_cuda的C++函数暴露给Python
  // 这意味着在Python中可以通过模块名调用这个函数
  m.def("gemv_forward_cuda_uniform", &gemv_forward_cuda_uniform);
  m.def("gemv_forward_cuda_uniform_outlier", &gemv_forward_cuda_uniform_outlier);
  m.def("gemv_forward_cuda_normalize", &gemv_forward_cuda_normalize);
  m.def("gemv_forward_cuda_normalize_outlier", &gemv_forward_cuda_normalize_outlier);
  m.def("gemv_forward_cuda_uniform_group", &gemv_forward_cuda_uniform_group);
  m.def("gemv_forward_cuda_uniform_group_outlier", &gemv_forward_cuda_uniform_group_outlier);
  m.def("gemv_forward_cuda_qk_uniform_group_token", &gemv_forward_cuda_qk_uniform_group_token);
  m.def("gemv_forward_cuda_qk_group_hq", &gemv_forward_cuda_qk_group_hq);
  m.def("gemv_forward_cuda_qk_group_hq_seq", &gemv_forward_cuda_qk_group_hq_seq);
  m.def("gemv_forward_cuda_wv_group_hq", &gemv_forward_cuda_wv_group_hq);
  m.def("gemv_forward_cuda_wv_group_hq_seq", &gemv_forward_cuda_wv_group_hq_seq);
  m.def("gemv_forward_cuda_wv_group_outlier_hq", &gemv_forward_cuda_wv_group_outlier_hq);
  m.def("gemv_forward_cuda_wv_uniform_group", &gemv_forward_cuda_wv_uniform_group);
  m.def("gemv_forward_cuda_wv_uniform_group_outlier", &gemv_forward_cuda_wv_uniform_group_outlier);
  m.def("gemv_forward_cuda_normalize_group", &gemv_forward_cuda_normalize_group);
  m.def("gemv_forward_cuda_normalize_group_outlier", &gemv_forward_cuda_normalize_group_outlier);
  m.def("gemv_forward_cuda_qk_normalize_group_token", &gemv_forward_cuda_qk_normalize_group_token);
  m.def("gemv_forward_cuda_wv_normalize_group", &gemv_forward_cuda_wv_normalize_group);
  m.def("gemv_forward_cuda_wv_normalize_group_outlier", &gemv_forward_cuda_wv_normalize_group_outlier);
}