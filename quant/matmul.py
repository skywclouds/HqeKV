import torch
import kvhq_gemv
from pack_unpack import (
	uniform_group_quantize_pack_last_dim,
    uniform_group_dequantize_unpack_last_dim,
	normal_group_quantize_pack_last_dim,
    normal_group_dequantize_unpack_last_dim, 
    normal_quantiles_center,)
from transformers.models.llama.modeling_llama import repeat_kv
import time

def cuda_bmm_fA_qB_normal(
				group_size: int,
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	Compute the matrix multiplication C = query x key.

	group_size不用设置
	fA is of shape (B, nh_q, M, K) float16
	qB is of shape (B, nh_kv, K, N // feat_per_int) int32
	scales is of shape (B, nh_kv, K, 1) float16
	zeros is of shape (B, nh_kv, K, 1) float16

	Returns C of shape (B, nh_q, M, N) float16
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_normalize(fA, qB, stds, means, center, bits, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_normal_outlier(
				group_size: int,
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.IntTensor, 
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	带outlier的normal量化
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_normalize_outlier(fA, qB, stds, means, center, outliers, outliers_idx, bits, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_normal_group(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	'''
	用全精度的fA矩阵乘以量化后的qB矩阵
	normal量化
	'''
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_normalize_group(fA, qB, stds, means, center, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_forward_qk_normal_group_token(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, qB.shape[-2], qB.shape[-1]).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).contiguous()
	c = kvhq_gemv.gemv_forward_cuda_qk_normalize_group_token(fA, qB, stds, means, center, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_normal_group_outlier(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.IntTensor, 
				bits: int) -> torch.FloatTensor:
	'''
	用全精度的fA矩阵乘以量化后的qB矩阵
	normal量化
	'''
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_normalize_group_outlier(fA, qB, stds, means, center, outliers, outliers_idx, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c


def cuda_bmm_fA_qB_wv_normal_group(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	'''
	用全精度的fA矩阵乘以量化后的qB矩阵
	normal量化
	'''
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_normalize_group(fA, qB, stds, means, center, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_wv_normal_group_outlier(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				stds: torch.FloatTensor, 
				means: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.IntTensor, 
				bits: int) -> torch.FloatTensor:
	'''
	用全精度的fA矩阵乘以量化后的qB矩阵
	normal量化
	'''
	assert len(fA.shape) == 4 and len(qB.shape) == 4
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	center = normal_quantiles_center[bits].to(fA.device)
	stds = stds.view(-1, stds.shape[-2], stds.shape[-1]).transpose(1, 2).contiguous()
	means = means.view(-1, means.shape[-2], means.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_normalize_group_outlier(fA, qB, stds, means, center, outliers, outliers_idx, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_uniform(
				group_size: int,
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	Compute the matrix multiplication C = query x key.

	group_size不用设置
	fA is of shape (B, nh_q, M, K) float16
	qB is of shape (B, nh_kv, K, N // feat_per_int) int32
	scales is of shape (B, nh_kv, K, 1) float16
	zeros is of shape (B, nh_kv, K, 1) float16

	Returns C of shape (B, nh_q, M, N) float16
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	'''
	因为qB是int32,而实际上数字是量化后低位的,所以一个int32能存储多个
	量化后的数,所以N要除以feat_per_int
	'''
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_uniform(fA, qB, scales, zeros, bits, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_uniform_outlier(
				group_size: int,
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.LongTensor, 
				bits: int) -> torch.FloatTensor:
	'''
	用全精度的fA矩阵乘以量化后的qB矩阵
	带有outlier的均匀量化
	'''
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_uniform_outlier(fA, qB, scales, zeros, outliers, outliers_idx, bits, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_uniform_group(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	Compute the matrix multiplication C = query x key.
	Where key is quantized into 2-bit values.

	fA is of shape (B, nh_q, M, K) float16
	qB is of shape (B, nh_kv, K, N // feat_per_int) int32
	scales is of shape (B, nh_kv, K, G) float16
	zeros is of shape (B, nh_kv, K, G) float16

	groupsize is the number of outer dimensions in each group.
	G = N // groupsize

	Returns C of shape (B, nh_q, M, N) float16
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	'''
	因为qB是int32,而实际上数字是int2或int4,所以一个int32能存储多个
	量化后的数,所以N要除以feat_per_int
	'''
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_uniform_group(fA, qB, scales, zeros, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_fA_qB_uniform_group_outlier(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.LongTensor, 
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	带有outlier的分组均匀量化
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	'''
	因为qB是int32,而实际上数字是int2或int4,所以一个int32能存储多个
	量化后的数,所以N要除以feat_per_int
	'''
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c = kvhq_gemv.gemv_forward_cuda_uniform_group_outlier(fA, qB, scales, zeros, outliers, outliers_idx, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_forward_qk_uniform_group_token(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, qB.shape[-2], qB.shape[-1]).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).contiguous()
	c = kvhq_gemv.gemv_forward_cuda_qk_uniform_group_token(fA, qB, scales, zeros, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_forward_qk_group_hq(
				group_size: int, 
				nh_kv: int,
				in_vector: torch.FloatTensor, 
				matrix_16: torch.IntTensor, 
				matrix_16_idx: torch.IntTensor, 
				matrix_8: torch.IntTensor, 
				matrix_8_idx: torch.IntTensor, 
				scales_8: torch.FloatTensor, 
				zeros_8: torch.FloatTensor,
				matrix_4: torch.IntTensor, 
				matrix_4_idx: torch.IntTensor, 
				scales_4: torch.FloatTensor, 
				zeros_4: torch.FloatTensor,
				matrix_2: torch.IntTensor, 
				matrix_2_idx: torch.IntTensor, 
				scales_2: torch.FloatTensor, 
				zeros_2: torch.FloatTensor,
				matrix_1: torch.IntTensor, 
				matrix_1_idx: torch.IntTensor, 
				std_1: torch.FloatTensor, 
				means_1: torch.FloatTensor,
				new_K: torch.FloatTensor
				) -> torch.FloatTensor:
	'''8421 位的向量乘以量化的矩阵'''
	B, nh_q, M, K = in_vector.shape 

	num_out_channels = 0
	if matrix_16_idx is not None:
		num_out_channels += matrix_16.shape[-1]
	if matrix_8_idx is not None:
		num_out_channels += matrix_8.shape[-1] * 4
	if matrix_4_idx is not None:
		num_out_channels += matrix_4.shape[-1] * 8
	if matrix_2_idx is not None:
		num_out_channels += matrix_2.shape[-1] * 16
	if matrix_1_idx is not None:
		num_out_channels += matrix_1.shape[-1] * 32
	if new_K is not None:
		num_out_channels += new_K.shape[-2]
	if num_out_channels == 0:
		raise ValueError('invalid matrix.')
	num_16 = matrix_16_idx.shape[0] if matrix_16_idx is not None else 0
	num_8 = matrix_8_idx.shape[0] if matrix_8_idx is not None else 0
	num_4 = matrix_4_idx.shape[0] if matrix_4_idx is not None else 0
	num_2 = matrix_2_idx.shape[0] if matrix_2_idx is not None else 0
	num_1 = matrix_1_idx.shape[0] if matrix_1_idx is not None else 0
	new_len = new_K.shape[-2] if new_K is not None else 0

	# print('func_in_start', time.perf_counter())
	in_vector = in_vector.view(-1, M, K).contiguous()
	matrix_16 = matrix_16.reshape(-1, matrix_16.shape[-2], matrix_16.shape[-1]).transpose(1, 2).contiguous() if matrix_16 is not None else None
	# print('func_in_transpose_16', time.perf_counter())
	matrix_8 = matrix_8.reshape(-1, matrix_8.shape[-2], matrix_8.shape[-1]).transpose(1, 2).contiguous() if matrix_8 is not None else None
	scales_8 = scales_8.view(-1, scales_8.shape[-2], scales_8.shape[-1]).transpose(1, 2).contiguous() if scales_8 is not None else None
	zeros_8 = zeros_8.view(-1, zeros_8.shape[-2], zeros_8.shape[-1]).transpose(1, 2).contiguous() if zeros_8 is not None else None
	# print('func_in_transpose_8', time.perf_counter())
	matrix_4 = matrix_4.reshape(-1, matrix_4.shape[-2], matrix_4.shape[-1]).transpose(1, 2).contiguous() if matrix_4 is not None else None
	scales_4 = scales_4.view(-1, scales_4.shape[-2], scales_4.shape[-1]).transpose(1, 2).contiguous() if scales_4 is not None else None
	zeros_4 = zeros_4.view(-1, zeros_4.shape[-2], zeros_4.shape[-1]).transpose(1, 2).contiguous() if zeros_4 is not None else None
	# print('func_in_transpose_4', time.perf_counter())
	matrix_2 = matrix_2.reshape(-1, matrix_2.shape[-2], matrix_2.shape[-1]).transpose(1, 2).contiguous() if matrix_2 is not None else None
	scales_2 = scales_2.view(-1, scales_2.shape[-2], scales_2.shape[-1]).transpose(1, 2).contiguous() if scales_2 is not None else None
	zeros_2 = zeros_2.view(-1, zeros_2.shape[-2], zeros_2.shape[-1]).transpose(1, 2).contiguous() if zeros_2 is not None else None
	# print('func_in_transpose_2', time.perf_counter())
	matrix_1 = matrix_1.reshape(-1, matrix_1.shape[-2], matrix_1.shape[-1]).transpose(1, 2).contiguous() if matrix_1 is not None else None
	std_1 = std_1.view(-1, std_1.shape[-2], std_1.shape[-1]).transpose(1, 2).contiguous() if std_1 is not None else None
	means_1 = means_1.view(-1, means_1.shape[-2], means_1.shape[-1]).transpose(1, 2).contiguous() if means_1 is not None else None
	# print('func_in_transpose_1', time.perf_counter())
	new_K = new_K.view(-1, new_K.shape[-2], new_K.shape[-1]).contiguous() if new_K is not None else None
	# print('func_in_transpose_new', time.perf_counter())
	center = normal_quantiles_center[1].to(in_vector.device)
	# print('func_in_center', time.perf_counter())

	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_qk_group_hq(
		in_vector, matrix_16, matrix_16_idx, matrix_8, matrix_8_idx, scales_8, zeros_8,
		matrix_4, matrix_4_idx, scales_4, zeros_4, matrix_2, matrix_2_idx, scales_2, zeros_2,
		matrix_1, matrix_1_idx, std_1, means_1, center, new_K,
		num_out_channels, num_16, num_8, num_4, num_2, num_1, new_len, group_size, nh_q, nh_kv
		)
	# print('func_in_cuda', time.perf_counter())
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	# print('func_in_end', time.perf_counter())
	return c

def cuda_bmm_forward_qk_group_hq_seq(
				group_size: int, 
				nh_kv: int,
				in_vector: torch.FloatTensor, 
				matrix_16: torch.IntTensor, 
				matrix_8: torch.IntTensor, 
				scales_8: torch.FloatTensor, 
				zeros_8: torch.FloatTensor,
				matrix_4: torch.IntTensor, 
				scales_4: torch.FloatTensor, 
				zeros_4: torch.FloatTensor,
				matrix_2: torch.IntTensor,  
				scales_2: torch.FloatTensor, 
				zeros_2: torch.FloatTensor,
				matrix_1: torch.IntTensor, 
				std_1: torch.FloatTensor, 
				means_1: torch.FloatTensor,
				new_K: torch.FloatTensor
				) -> torch.FloatTensor:
	B, nh_q, M, K = in_vector.shape 

	num_out_channels = 0
	if matrix_16 is not None:
		num_out_channels += matrix_16.shape[-1]
	if matrix_8 is not None:
		num_out_channels += matrix_8.shape[-1] * 4
	if matrix_4 is not None:
		num_out_channels += matrix_4.shape[-1] * 8
	if matrix_2 is not None:
		num_out_channels += matrix_2.shape[-1] * 16
	if matrix_1 is not None:
		num_out_channels += matrix_1.shape[-1] * 32
	if new_K is not None:
		num_out_channels += new_K.shape[-2]
	if num_out_channels == 0:
		raise ValueError('invalid matrix.')
	num_16 = matrix_16.shape[-1] if matrix_16 is not None else 0
	num_8 = matrix_8.shape[-1] * 4 if matrix_8 is not None else 0
	num_4 = matrix_4.shape[-1] * 8 if matrix_4 is not None else 0
	num_2 = matrix_2.shape[-1] * 16 if matrix_2 is not None else 0
	num_1 = matrix_1.shape[-1] * 32 if matrix_1 is not None else 0
	new_len = new_K.shape[-2] if new_K is not None else 0

	in_vector = in_vector.view(-1, M, K).contiguous()
	matrix_16 = matrix_16.reshape(-1, matrix_16.shape[-2], matrix_16.shape[-1]).transpose(1, 2).contiguous() if matrix_16 is not None else None
	matrix_8 = matrix_8.reshape(-1, matrix_8.shape[-2], matrix_8.shape[-1]).transpose(1, 2).contiguous() if matrix_8 is not None else None
	scales_8 = scales_8.view(-1, scales_8.shape[-2], scales_8.shape[-1]).transpose(1, 2).contiguous() if scales_8 is not None else None
	zeros_8 = zeros_8.view(-1, zeros_8.shape[-2], zeros_8.shape[-1]).transpose(1, 2).contiguous() if zeros_8 is not None else None
	matrix_4 = matrix_4.reshape(-1, matrix_4.shape[-2], matrix_4.shape[-1]).transpose(1, 2).contiguous() if matrix_4 is not None else None
	scales_4 = scales_4.view(-1, scales_4.shape[-2], scales_4.shape[-1]).transpose(1, 2).contiguous() if scales_4 is not None else None
	zeros_4 = zeros_4.view(-1, zeros_4.shape[-2], zeros_4.shape[-1]).transpose(1, 2).contiguous() if zeros_4 is not None else None
	matrix_2 = matrix_2.reshape(-1, matrix_2.shape[-2], matrix_2.shape[-1]).transpose(1, 2).contiguous() if matrix_2 is not None else None
	scales_2 = scales_2.view(-1, scales_2.shape[-2], scales_2.shape[-1]).transpose(1, 2).contiguous() if scales_2 is not None else None
	zeros_2 = zeros_2.view(-1, zeros_2.shape[-2], zeros_2.shape[-1]).transpose(1, 2).contiguous() if zeros_2 is not None else None
	matrix_1 = matrix_1.reshape(-1, matrix_1.shape[-2], matrix_1.shape[-1]).transpose(1, 2).contiguous() if matrix_1 is not None else None
	std_1 = std_1.view(-1, std_1.shape[-2], std_1.shape[-1]).transpose(1, 2).contiguous() if std_1 is not None else None
	means_1 = means_1.view(-1, means_1.shape[-2], means_1.shape[-1]).transpose(1, 2).contiguous() if means_1 is not None else None
	new_K = new_K.view(-1, new_K.shape[-2], new_K.shape[-1]).contiguous() if new_K is not None else None
	center = normal_quantiles_center[1].to(in_vector.device)

	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_qk_group_hq_seq(
		in_vector, matrix_16, matrix_8, scales_8, zeros_8,
		matrix_4, scales_4, zeros_4, matrix_2, scales_2, zeros_2,
		matrix_1, std_1, means_1, center, new_K,
		num_out_channels, num_16, num_8, num_4, num_2, num_1, new_len, group_size, nh_q, nh_kv
		)
	# print('func_in_cuda', time.perf_counter())
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	# print('func_in_end', time.perf_counter())
	return c

def cuda_bmm_forward_wv_group_hq(
				group_size: int, 
				nh_kv: int,
				in_vector: torch.FloatTensor, 
				matrix_16: torch.IntTensor, 
				matrix_16_idx: torch.IntTensor, 
				matrix_8: torch.IntTensor, 
				matrix_8_idx: torch.IntTensor, 
				scales_8: torch.FloatTensor, 
				zeros_8: torch.FloatTensor,
				matrix_4: torch.IntTensor, 
				matrix_4_idx: torch.IntTensor, 
				scales_4: torch.FloatTensor, 
				zeros_4: torch.FloatTensor,
				matrix_2: torch.IntTensor, 
				matrix_2_idx: torch.IntTensor, 
				scales_2: torch.FloatTensor, 
				zeros_2: torch.FloatTensor,
				matrix_1: torch.IntTensor, 
				matrix_1_idx: torch.IntTensor, 
				std_1: torch.FloatTensor, 
				means_1: torch.FloatTensor,
				new_V:  torch.FloatTensor
				) -> torch.FloatTensor:
	'''8421 位的向量乘以量化的矩阵'''
	B, nh_q, M, K = in_vector.shape 
	num_out_channels = 0
	if matrix_16 is not None:
		num_out_channels = matrix_16.shape[-1]
	elif matrix_8 is not None:
		num_out_channels = matrix_8.shape[-1] * 4
	elif matrix_4 is not None:
		num_out_channels = matrix_4.shape[-1] * 8
	elif matrix_2 is not None:
		num_out_channels = matrix_2.shape[-1] * 16
	elif matrix_1 is not None:
		num_out_channels = matrix_1.shape[-1] * 32
	elif new_V is not None:
		num_out_channels = new_V.shape[-1]
	else:
		raise ValueError('invalid matrix.')
	num_16 = matrix_16_idx.shape[0] if matrix_16_idx is not None else 0
	num_8 = matrix_8_idx.shape[0] if matrix_8_idx is not None else 0
	num_4 = matrix_4_idx.shape[0] if matrix_4_idx is not None else 0
	num_2 = matrix_2_idx.shape[0] if matrix_2_idx is not None else 0
	num_1 = matrix_1_idx.shape[0] if matrix_1_idx is not None else 0
	new_len = new_V.shape[-2] if new_V is not None else 0

	in_vector = in_vector.view(-1, M, K).contiguous()
	matrix_16 = matrix_16.reshape(-1, matrix_16.shape[-2], matrix_16.shape[-1]).transpose(1, 2).contiguous() if matrix_16 is not None else None
	matrix_8 = matrix_8.reshape(-1, matrix_8.shape[-2], matrix_8.shape[-1]).transpose(1, 2).contiguous() if matrix_8 is not None else None
	scales_8 = scales_8.view(-1, scales_8.shape[-2], scales_8.shape[-1]).transpose(1, 2).contiguous() if scales_8 is not None else None
	zeros_8 = zeros_8.view(-1, zeros_8.shape[-2], zeros_8.shape[-1]).transpose(1, 2).contiguous() if zeros_8 is not None else None
	matrix_4 = matrix_4.reshape(-1, matrix_4.shape[-2], matrix_4.shape[-1]).transpose(1, 2).contiguous() if matrix_4 is not None else None
	scales_4 = scales_4.view(-1, scales_4.shape[-2], scales_4.shape[-1]).transpose(1, 2).contiguous() if scales_4 is not None else None
	zeros_4 = zeros_4.view(-1, zeros_4.shape[-2], zeros_4.shape[-1]).transpose(1, 2).contiguous() if zeros_4 is not None else None
	matrix_2 = matrix_2.reshape(-1, matrix_2.shape[-2], matrix_2.shape[-1]).transpose(1, 2).contiguous() if matrix_2 is not None else None
	scales_2 = scales_2.view(-1, scales_2.shape[-2], scales_2.shape[-1]).transpose(1, 2).contiguous() if scales_2 is not None else None
	zeros_2 = zeros_2.view(-1, zeros_2.shape[-2], zeros_2.shape[-1]).transpose(1, 2).contiguous() if zeros_2 is not None else None
	matrix_1 = matrix_1.reshape(-1, matrix_1.shape[-2], matrix_1.shape[-1]).transpose(1, 2).contiguous() if matrix_1 is not None else None
	std_1 = std_1.view(-1, std_1.shape[-2], std_1.shape[-1]).transpose(1, 2).contiguous() if std_1 is not None else None
	means_1 = means_1.view(-1, means_1.shape[-2], means_1.shape[-1]).transpose(1, 2).contiguous() if means_1 is not None else None
	new_V = new_V.view(-1, new_V.shape[-2], new_V.shape[-1]).transpose(1, 2).contiguous() if new_V is not None else None
	center = normal_quantiles_center[1].to(in_vector.device)
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_group_hq(
		in_vector, matrix_16, matrix_16_idx, matrix_8, matrix_8_idx, scales_8, zeros_8,
		matrix_4, matrix_4_idx, scales_4, zeros_4,
		matrix_2, matrix_2_idx, scales_2, zeros_2,
		matrix_1, matrix_1_idx, std_1, means_1, center, new_V, num_out_channels, 
		num_16, num_8, num_4, num_2, num_1, new_len, group_size, nh_q, nh_kv
		)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

def cuda_bmm_forward_wv_group_hq_seq(
				group_size: int, 
				nh_kv: int,
				in_vector: torch.FloatTensor, 
				matrix_16: torch.IntTensor, 
				matrix_8: torch.IntTensor, 
				scales_8: torch.FloatTensor, 
				zeros_8: torch.FloatTensor,
				matrix_4: torch.IntTensor, 
				scales_4: torch.FloatTensor, 
				zeros_4: torch.FloatTensor,
				matrix_2: torch.IntTensor, 
				scales_2: torch.FloatTensor, 
				zeros_2: torch.FloatTensor,
				matrix_1: torch.IntTensor, 
				std_1: torch.FloatTensor, 
				means_1: torch.FloatTensor,
				new_V:  torch.FloatTensor
				) -> torch.FloatTensor:
	'''8421 位的向量乘以量化的矩阵'''
	B, nh_q, M, K = in_vector.shape 
	num_out_channels = 0
	if matrix_16 is not None:
		num_out_channels = matrix_16.shape[-1]
	elif matrix_8 is not None:
		num_out_channels = matrix_8.shape[-1] * 4
	elif matrix_4 is not None:
		num_out_channels = matrix_4.shape[-1] * 8
	elif matrix_2 is not None:
		num_out_channels = matrix_2.shape[-1] * 16
	elif matrix_1 is not None:
		num_out_channels = matrix_1.shape[-1] * 32
	elif new_V is not None:
		num_out_channels = new_V.shape[-1]
	else:
		raise ValueError('invalid matrix.')
	num_16 = matrix_16.shape[-2] if matrix_16 is not None else 0
	num_8 = matrix_8.shape[-2] if matrix_8 is not None else 0
	num_4 = matrix_4.shape[-2] if matrix_4 is not None else 0
	num_2 = matrix_2.shape[-2] if matrix_2 is not None else 0
	num_1 = matrix_1.shape[-2] if matrix_1 is not None else 0
	new_len = new_V.shape[-2] if new_V is not None else 0

	in_vector = in_vector.view(-1, M, K).contiguous()
	matrix_16 = matrix_16.reshape(-1, matrix_16.shape[-2], matrix_16.shape[-1]).transpose(1, 2).contiguous() if matrix_16 is not None else None
	matrix_8 = matrix_8.reshape(-1, matrix_8.shape[-2], matrix_8.shape[-1]).transpose(1, 2).contiguous() if matrix_8 is not None else None
	scales_8 = scales_8.view(-1, scales_8.shape[-2], scales_8.shape[-1]).transpose(1, 2).contiguous() if scales_8 is not None else None
	zeros_8 = zeros_8.view(-1, zeros_8.shape[-2], zeros_8.shape[-1]).transpose(1, 2).contiguous() if zeros_8 is not None else None
	matrix_4 = matrix_4.reshape(-1, matrix_4.shape[-2], matrix_4.shape[-1]).transpose(1, 2).contiguous() if matrix_4 is not None else None
	scales_4 = scales_4.view(-1, scales_4.shape[-2], scales_4.shape[-1]).transpose(1, 2).contiguous() if scales_4 is not None else None
	zeros_4 = zeros_4.view(-1, zeros_4.shape[-2], zeros_4.shape[-1]).transpose(1, 2).contiguous() if zeros_4 is not None else None
	matrix_2 = matrix_2.reshape(-1, matrix_2.shape[-2], matrix_2.shape[-1]).transpose(1, 2).contiguous() if matrix_2 is not None else None
	scales_2 = scales_2.view(-1, scales_2.shape[-2], scales_2.shape[-1]).transpose(1, 2).contiguous() if scales_2 is not None else None
	zeros_2 = zeros_2.view(-1, zeros_2.shape[-2], zeros_2.shape[-1]).transpose(1, 2).contiguous() if zeros_2 is not None else None
	matrix_1 = matrix_1.reshape(-1, matrix_1.shape[-2], matrix_1.shape[-1]).transpose(1, 2).contiguous() if matrix_1 is not None else None
	std_1 = std_1.view(-1, std_1.shape[-2], std_1.shape[-1]).transpose(1, 2).contiguous() if std_1 is not None else None
	means_1 = means_1.view(-1, means_1.shape[-2], means_1.shape[-1]).transpose(1, 2).contiguous() if means_1 is not None else None
	new_V = new_V.view(-1, new_V.shape[-2], new_V.shape[-1]).transpose(1, 2).contiguous() if new_V is not None else None
	center = normal_quantiles_center[1].to(in_vector.device)
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_group_hq_seq(
		in_vector, matrix_16, matrix_8, scales_8, zeros_8,
		matrix_4, scales_4, zeros_4,
		matrix_2, scales_2, zeros_2,
		matrix_1, std_1, means_1, center, new_V, num_out_channels, 
		num_16, num_8, num_4, num_2, num_1, new_len, group_size, nh_q, nh_kv
		)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c


def cuda_bmm_forward_wv_group_outlier_hq(
				group_size: int, 
				nh_kv: int,
				in_vector: torch.FloatTensor, 
				matrix_16: torch.IntTensor, 
				matrix_16_idx: torch.IntTensor, 
				matrix_8: torch.IntTensor, 
				matrix_8_idx: torch.IntTensor, 
				scales_8: torch.FloatTensor, 
				zeros_8: torch.FloatTensor,
				matrix_4: torch.IntTensor, 
				matrix_4_idx: torch.IntTensor, 
				scales_4: torch.FloatTensor, 
				zeros_4: torch.FloatTensor,
				matrix_2: torch.IntTensor, 
				matrix_2_idx: torch.IntTensor, 
				scales_2: torch.FloatTensor, 
				zeros_2: torch.FloatTensor,
				matrix_1: torch.IntTensor, 
				matrix_1_idx: torch.IntTensor, 
				std_1: torch.FloatTensor, 
				means_1: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.LongTensor,
				new_V:  torch.FloatTensor
				) -> torch.FloatTensor:
	'''8421 位的向量乘以量化的矩阵'''
	B, nh_q, M, K = in_vector.shape 
	num_out_channels = 0
	if matrix_16 is not None:
		num_out_channels = matrix_16.shape[-1]
	elif matrix_8 is not None:
		num_out_channels = matrix_8.shape[-1] * 4
	elif matrix_4 is not None:
		num_out_channels = matrix_4.shape[-1] * 8
	elif matrix_2 is not None:
		num_out_channels = matrix_2.shape[-1] * 16
	elif matrix_1 is not None:
		num_out_channels = matrix_1.shape[-1] * 32
	elif new_V is not None:
		num_out_channels = new_V.shape[-1]
	else:
		raise ValueError('invalid matrix.')
	num_16 = matrix_16_idx.shape[0] if matrix_16_idx is not None else 0
	num_8 = matrix_8_idx.shape[0] if matrix_8_idx is not None else 0
	num_4 = matrix_4_idx.shape[0] if matrix_4_idx is not None else 0
	num_2 = matrix_2_idx.shape[0] if matrix_2_idx is not None else 0
	num_1 = matrix_1_idx.shape[0] if matrix_1_idx is not None else 0
	new_len = new_V.shape[-2] if new_V is not None else 0

	print('func_in_start', time.perf_counter())
	in_vector = in_vector.view(-1, M, K).contiguous()
	matrix_16 = matrix_16.reshape(-1, matrix_16.shape[-2], matrix_16.shape[-1]).transpose(1, 2).contiguous() if matrix_16 is not None else None
	matrix_8 = matrix_8.reshape(-1, matrix_8.shape[-2], matrix_8.shape[-1]).transpose(1, 2).contiguous() if matrix_8 is not None else None
	scales_8 = scales_8.view(-1, scales_8.shape[-2], scales_8.shape[-1]).transpose(1, 2).contiguous() if scales_8 is not None else None
	zeros_8 = zeros_8.view(-1, zeros_8.shape[-2], zeros_8.shape[-1]).transpose(1, 2).contiguous() if zeros_8 is not None else None
	matrix_4 = matrix_4.reshape(-1, matrix_4.shape[-2], matrix_4.shape[-1]).transpose(1, 2).contiguous() if matrix_4 is not None else None
	scales_4 = scales_4.view(-1, scales_4.shape[-2], scales_4.shape[-1]).transpose(1, 2).contiguous() if scales_4 is not None else None
	zeros_4 = zeros_4.view(-1, zeros_4.shape[-2], zeros_4.shape[-1]).transpose(1, 2).contiguous() if zeros_4 is not None else None
	matrix_2 = matrix_2.reshape(-1, matrix_2.shape[-2], matrix_2.shape[-1]).transpose(1, 2).contiguous() if matrix_2 is not None else None
	scales_2 = scales_2.view(-1, scales_2.shape[-2], scales_2.shape[-1]).transpose(1, 2).contiguous() if scales_2 is not None else None
	zeros_2 = zeros_2.view(-1, zeros_2.shape[-2], zeros_2.shape[-1]).transpose(1, 2).contiguous() if zeros_2 is not None else None
	matrix_1 = matrix_1.reshape(-1, matrix_1.shape[-2], matrix_1.shape[-1]).transpose(1, 2).contiguous() if matrix_1 is not None else None
	std_1 = std_1.view(-1, std_1.shape[-2], std_1.shape[-1]).transpose(1, 2).contiguous() if std_1 is not None else None
	means_1 = means_1.view(-1, means_1.shape[-2], means_1.shape[-1]).transpose(1, 2).contiguous() if means_1 is not None else None
	center = normal_quantiles_center[1].to(in_vector.device)
	new_V = new_V.view(-1, new_V.shape[-2], new_V.shape[-1]).transpose(1, 2).contiguous() if new_V is not None else None
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()

	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_group_outlier_hq(
		in_vector, matrix_16, matrix_16_idx, matrix_8, matrix_8_idx, scales_8, zeros_8,
		matrix_4, matrix_4_idx, scales_4, zeros_4,
		matrix_2, matrix_2_idx, scales_2, zeros_2,
		matrix_1, matrix_1_idx, std_1, means_1, center, new_V, outliers, outliers_idx,
		num_out_channels, num_16, num_8, num_4, num_2, num_1, new_len, group_size, nh_q, nh_kv
		)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])

	print('func_in_end', time.perf_counter())
	return c

def cuda_bmm_fA_qB_wv_uniform_group(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	带有outlier的分组均匀量化
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	'''
	因为qB是int32,而实际上数字是int2或int4,所以一个int32能存储多个
	量化后的数,所以N要除以feat_per_int
	'''
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_uniform_group(fA, qB, scales, zeros, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c


def cuda_bmm_fA_qB_wv_uniform_group_outlier(
				group_size: int, 
				fA: torch.FloatTensor, 
				qB: torch.IntTensor, 
				scales: torch.FloatTensor, 
				zeros: torch.FloatTensor,
				outliers: torch.FloatTensor, 
				outliers_idx: torch.LongTensor, 
				bits: int) -> torch.FloatTensor:
	"""
	用全精度的fA矩阵乘以量化后的qB矩阵
	带有outlier的分组均匀量化
	"""    
	B, nh_q, M, K = fA.shape 
	nh_kv = qB.shape[1]
	'''
	因为qB是int32,而实际上数字是int2或int4,所以一个int32能存储多个
	量化后的数,所以N要除以feat_per_int
	'''
	# flatten to a 3D tensor
	fA = fA.view(-1, M, K).contiguous()
	qB = qB.reshape(-1, K, qB.shape[-1]).transpose(1, 2).contiguous()
	scales = scales.view(-1, scales.shape[-2], scales.shape[-1]).transpose(1, 2).contiguous()
	zeros = zeros.view(-1, zeros.shape[-2], zeros.shape[-1]).transpose(1, 2).contiguous()
	outliers = outliers.view(-1, outliers.shape[-2], outliers.shape[-1]).transpose(1, 2).contiguous()
	outliers_idx = outliers_idx.view(-1, outliers_idx.shape[-2], outliers_idx.shape[-1]).transpose(1, 2).contiguous()
	# 用cuda代码执行的矩阵乘法
	c: torch.Tensor = kvhq_gemv.gemv_forward_cuda_wv_uniform_group_outlier(fA, qB, scales, zeros, outliers, outliers_idx, bits, group_size, nh_q, nh_kv)
	c = c.view(B, nh_q, c.shape[-2], c.shape[-1])
	return c

if __name__ == '__main__':

	seed = 42
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	torch.cuda.manual_seed_all(seed)
	B, M, K, N, bit = 1, 1, 1024, 128, 4
	feat_per_int = 32 // bit
	group_size = 32

	device = 'cuda:6'
	fA = torch.randn(B, 32, M, K, dtype=torch.float16, device=device)
	data = torch.randn(B, 8, K, N, dtype=torch.float16, device=device)
	# data_outliers = torch.zeros(data.shape[0], data.shape[1], data.shape[2], 2, dtype=data.dtype, device=data.device)
	# data_outliers_idx = torch.zeros(data.shape[0], data.shape[1], data.shape[2], 2, dtype=torch.int64, device=data.device)
	# data_median, _ = torch.median(data, dim=-1, keepdim=True)
	# data_up_outliers, data_up_outliers_idx = torch.topk(data, k=1, dim=-1)
	# data_low_outliers, data_low_outliers_idx = torch.topk(data, k=1, dim=-1, largest=False)
	# data_outliers[:, :, :, 0] = data_up_outliers[:, :, :, 0]
	# data_outliers[:, :, :, 1] = data_low_outliers[:, :, :, 0]
	# data_outliers_idx[:, :, :, 0] = data_up_outliers_idx[:, :, :, 0]
	# data_outliers_idx[:, :, :, 1] = data_low_outliers_idx[:, :, :, 0]
	# data.scatter_(dim=-1, index=data_outliers_idx, src=data_median.expand_as(data_outliers_idx))
	qB, scales, zeros = normal_group_quantize_pack_last_dim(data.clone(), group_size, bit)

	dq_data = normal_group_dequantize_unpack_last_dim(qB, scales, zeros, group_size, bit)
	# dq_data.scatter_(dim=-1, index=data_outliers_idx, src=data_outliers)
	print(torch.norm(dq_data - data))

	# print(dq_data[0, 0, 0, :])
	# c = cuda_bmm_fA_qB_wv_uniform_group_outlier(group_size, fA, qB, scales, zeros, data_outliers, data_outliers_idx, bit)
	c = cuda_bmm_fA_qB_wv_normal_group(group_size, fA, qB, scales, zeros, bit)
	
	d = torch.matmul(fA, repeat_kv(dq_data, 4))

	# print(c[0, 0, 0, :])
	# print(d[0, 0, 0, :])
	# print(c - d)
	print(torch.norm(c - d, p=2))
