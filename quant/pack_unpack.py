from scipy.stats import norm
import triton
import triton.language as tl
import torch

n_bits_range = [8, 4, 2, 1]
normal_quantiles_upper_bound = {
                n: torch.tensor(norm.ppf(torch.arange(0, 1, 1/(2**n)) + 1/(2**n)), dtype=torch.float32)
                for n in n_bits_range
            }
normal_quantiles_center = {
	n: torch.tensor(norm.ppf(torch.arange(0, 1, 1/(2**n)) + 0.5/(2**n)), dtype=torch.float32)
	for n in n_bits_range
}

@triton.jit
def _pack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,# 形状为(-1, T)
	code_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	# 打包后的数组最后一个维度的大小 =
	# 打包前的数组最后一个维度的大小 / 一个int32存储的量化后的数的个数
	num_int_per_y_dim = num_feats // feat_per_int
	# 线程块是2维的
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	'''
	以bid=0为例,offs_N * num_feats会得到[0, T, 2T, ..., 127T],也就是intensor_ptr的每一行的开头
	yid * feat_per_int会让每一行的开头再加上一个偏移量
	每个线程会把feat_per_int个数打包到一个int32,所以列的偏移量是yid * feat_per_int
	这也是线程块的第一个维度是triton.cdiv(data.shape[0], BLOCK_SIZE_N)
	第二个维度是data.shape[1] // feat_per_int的原因
	'''
	block_start = intensor_ptr + offs_N * num_feats + yid * feat_per_int # offset of the first element at current tile
	packed = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	for i in range(feat_per_int):
		ptr = block_start + i
		element = tl.load(ptr, mask=offs_N<N, other=0.)
		# 把量化的数左移
		element = element << (i * bits)
		# 之后按位或
		packed = packed | element
	# offs_N * num_int_per_y_dim是定位到每一行的开头
	# + yid是定位到每一列
	tl.store(code_ptr + offs_N * num_int_per_y_dim + yid, packed, mask=offs_N < N)

def pack_along_last_dim(data: torch.Tensor, bit: int):
	'''打包数据'''
	shape = data.shape
	B, nh, D, T = shape
	data = data.view(-1, T)
	feat_per_int = 32 // bit# 一个int32能存储的量化后的数的个数
	# 由于一个int32能存储多个量化后的数,所以要除以feat_per_int
	packshape = (B * nh * D, T // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 计算线程数，这里的线程块是2维的
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1)

def normal_quantize_pack_last_dim(data: torch.Tensor, group_size, bit: int):
	'''normal量化后进行打包, group_size不用设置'''
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	# Quantize
	std, mean = torch.std_mean(data, dim=-1, keepdim=True)
	# 减去均值
	data.sub_(mean)
	# 除以标准差
	data.div_(std)
	# 标准正态分布对应位数的边界和中心
	upper_bound = normal_quantiles_upper_bound[bit].to(data.device)
	# 量化
	data = torch.searchsorted(upper_bound, data.contiguous(), out_int32=True).contiguous()
	data = data.view(-1, T)
	feat_per_int = 32 // bit# 一个int32能存储的量化后的数的个数
	# 由于一个int32能存储多个量化后的数,所以要除以feat_per_int
	packshape = (B * nh * D, T // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 计算线程数，这里的线程块是2维的
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), std, mean


def uniform_quantize_pack_last_dim(data: torch.Tensor, group_size, bit: int):
	'''均匀量化后进行打包, group_size不用设置'''
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	# Quantize
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	# 计算缩放系数
	scale = (mx - mn) / (2 ** bit - 1)
	# 减去最小值
	data.sub_(mn)
	# 除以缩放系数
	data.div_(scale)
	# 取整
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32).contiguous()
	data = data.view(-1, T)
	feat_per_int = 32 // bit# 一个int32能存储的量化后的数的个数
	# 由于一个int32能存储多个量化后的数,所以要除以feat_per_int
	packshape = (B * nh * D, T // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 计算线程数，这里的线程块是2维的
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale, mn

def uniform_group_fake_quant(data: torch.Tensor, group_size: int, bit: int):
	if bit == 0:
		return 0
	
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	data = data.reshape(new_shape)

	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	scale: torch.Tensor = (mx - mn) / (2 ** bit - 1)
	q_data = (data - mn) / scale

	q_data = q_data.clamp_(0, 2 ** bit - 1).round_()
	q_data = q_data * scale + mn
	q_data = q_data.reshape(shape)

	return q_data

def uniform_group_quantize_pack_last_dim(data: torch.Tensor, group_size: int, bit: int):
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	scale_mn_shape = B, nh, D, num_groups # 每个group都有scale和mn
	# Quantize
	data = data.reshape(new_shape)
	# 计算张量最后一个维度的最大值和最小值
	mn = torch.min(data, dim=-1, keepdim=True)[0]
	mx = torch.max(data, dim=-1, keepdim=True)[0]
	# 计算缩放系数
	scale: torch.Tensor = (mx - mn) / (2 ** bit - 1)
	# 减去最小值
	data.sub_(mn)
	# 除以缩放系数
	data.div_(scale)
	# 取整
	data = data.clamp_(0, 2 ** bit - 1).round_().to(torch.int32).contiguous()
	# 把num_groups和group_size合并成一个维度
	data = data.view(-1, T)
	feat_per_int = 32 // bit# 一个int32能存储的量化后的数的个数
	'''
	np.prod(shape[:-1])就是B * nh * D, shape[-1]就是T
	由于一个int32能存储多个量化后的数,所以要除以feat_per_int
	'''
	packshape = (B, nh, D, T // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 计算线程数，这里的线程块是2维的
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), scale.reshape(scale_mn_shape), mn.reshape(scale_mn_shape)

def normal_group_fake_quant(data: torch.Tensor, group_size: int, bit: int):
	if bit == 0:
		return 0
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)

	data = data.reshape(new_shape)
	std, mean = torch.std_mean(data, dim=-1, keepdim=True)
	q_data = (data - mean) / std

	upper_bound = normal_quantiles_upper_bound[bit].to(data.device)
	q_data = torch.searchsorted(upper_bound, q_data.contiguous(), out_int32=True)

	center = normal_quantiles_center[bit].to(data.device)
	q_data = center[q_data].to(std.dtype)
	q_data = q_data * std + mean
	q_data = q_data.reshape(shape)

	return q_data

def normal_group_quantize_pack_last_dim(data: torch.Tensor, group_size: int, bit: int):
	shape = data.shape
	B, nh, D, T = shape
	# ================== Get Scale & Zeros ===============
	num_groups = T // group_size
	new_shape = (B * nh * D, num_groups, group_size)
	std_mean_shape = B, nh, D, num_groups 
	# Quantize
	data = data.reshape(new_shape)
	std, mean = torch.std_mean(data, dim=-1, keepdim=True)
	# 减去均值
	data.sub_(mean)
	# 除以标准差
	data.div_(std)
	# 标准正态分布对应位数的边界和中心
	upper_bound = normal_quantiles_upper_bound[bit].to(data.device)
	# 量化
	data = torch.searchsorted(upper_bound, data.contiguous(), out_int32=True).contiguous()
	# 把num_groups和group_size合并成一个维度
	data = data.view(-1, T)
	feat_per_int = 32 // bit# 一个int32能存储的量化后的数的个数
	'''
	np.prod(shape[:-1])就是B * nh * D, shape[-1]就是T
	由于一个int32能存储多个量化后的数,所以要除以feat_per_int
	'''
	packshape = (B, nh, D, T // feat_per_int,)
	code = torch.zeros(*packshape, device=data.device, dtype=torch.int32)
	# 计算线程数，这里的线程块是2维的
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(data.shape[0], BLOCK_SIZE_N), data.shape[1] // feat_per_int,)
	with torch.cuda.device(data.device):
		_pack_along_last_dim[grid](bit, data, code, data.shape[0], 
								data.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	return code.view(B, nh, D, -1), std.reshape(std_mean_shape), mean.reshape(std_mean_shape)

@triton.jit
def _unpack_along_last_dim(
	bits: tl.constexpr,
	intensor_ptr,# 形状为(-1, T)
	data_ptr,
	N,
	num_feats: tl.constexpr,
	feat_per_int: tl.constexpr,
	BLOCK_SIZE_N: tl.constexpr
):
	# 解包后的数组最后一个维度的大小 =
	# 解包前的数组最后一个维度的大小 * 一个int32存储的量化后的数的个数
	num_int_per_y_dim = num_feats * feat_per_int
	# 线程块是2维的
	bid = tl.program_id(axis=0)
	yid = tl.program_id(axis=1)
	offs_N = bid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
	unpacked = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int32)
	element = tl.load(intensor_ptr + offs_N * num_feats + yid, mask=offs_N<N, other=0.)
	mask = 0xFF >> (8 - bits)
	for i in range(feat_per_int):
		unpacked = element & mask
		element = element >> bits
		tl.store(data_ptr + offs_N * num_int_per_y_dim + yid * feat_per_int + i, unpacked, mask=offs_N < N)
	
# unpack数据
def unpack_along_last_dim(code: torch.Tensor, bit: int):
	B, nh, D, T_d = code.shape
	feat_per_int = 32 // bit
	T = T_d * feat_per_int
	code = code.view(-1, T_d)
	data = torch.zeros(B * nh * D, T, dtype=torch.float16, device=code.device)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(code.shape[0], BLOCK_SIZE_N), code.shape[1],)
	with torch.cuda.device(data.device):
		_unpack_along_last_dim[grid](bit, code, data, code.shape[0], 
								code.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	data = data.view(B, nh, D, T)
	return data

def uniform_dequantize_unpack_last_dim(code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int,bit: int):
	'''均匀量化解包后进行反量化, group_size不用设置'''
	B, nh, D, T_d = code.shape
	feat_per_int = 32 // bit
	T = T_d * feat_per_int
	code = code.view(-1, T_d)
	data = torch.zeros(B * nh * D, T, dtype=torch.float16, device=code.device)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(code.shape[0], BLOCK_SIZE_N), code.shape[1],)
	with torch.cuda.device(data.device):
		_unpack_along_last_dim[grid](bit, code, data, code.shape[0], 
								code.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	data = data.view(B, nh, D, T)
	data.mul_(scale)
	data.add_(mn)
	return data

def uniform_group_dequantize_unpack_last_dim(code: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, group_size: int, bit: int):
	B, nh, D, T_d = code.shape
	feat_per_int = 32 // bit
	T = T_d * feat_per_int
	num_groups = T // group_size
	code = code.view(-1, T_d)
	data = torch.zeros(B * nh * D, T, dtype=torch.float16, device=code.device)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(code.shape[0], BLOCK_SIZE_N), code.shape[1],)
	with torch.cuda.device(data.device):
		_unpack_along_last_dim[grid](bit, code, data, code.shape[0], 
								code.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	data = data.view(B, nh, D, num_groups, group_size)
	data.mul_(scale.unsqueeze(-1))
	data.add_(mn.unsqueeze(-1))
	return data.reshape(B, nh, D, T)

def normal_dequantize_unpack_last_dim(code: torch.Tensor, std: torch.Tensor, mean: torch.Tensor, group_size, bit: int):
	'''normal量化解包后进行反量化, group_size不用设置'''
	B, nh, D, T_d = code.shape
	feat_per_int = 32 // bit
	T = T_d * feat_per_int
	code = code.view(-1, T_d)
	data = torch.zeros(B * nh * D, T, dtype=torch.float16, device=code.device)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(code.shape[0], BLOCK_SIZE_N), code.shape[1],)
	with torch.cuda.device(data.device):
		_unpack_along_last_dim[grid](bit, code, data, code.shape[0], 
								code.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	data = data.view(B, nh, D, T).to(torch.int32)
	center = normal_quantiles_center[bit].to(data.device)
	data = center[data].to(std.dtype)
	data.mul_(std)
	data.add_(mean)
	return data

def normal_group_dequantize_unpack_last_dim(code: torch.Tensor, std: torch.Tensor, mean: torch.Tensor, group_size: int, bit: int):
	B, nh, D, T_d = code.shape
	feat_per_int = 32 // bit
	T = T_d * feat_per_int
	num_groups = T // group_size
	code = code.view(-1, T_d)
	data = torch.zeros(B * nh * D, T, dtype=torch.float16, device=code.device)
	BLOCK_SIZE_N = 128
	grid = lambda meta: (triton.cdiv(code.shape[0], BLOCK_SIZE_N), code.shape[1],)
	with torch.cuda.device(data.device):
		_unpack_along_last_dim[grid](bit, code, data, code.shape[0], 
								code.shape[1], feat_per_int, 
								BLOCK_SIZE_N=BLOCK_SIZE_N, 
								num_warps=8)
	data = data.view(B, nh, D, num_groups, group_size).to(torch.int32)

	center = normal_quantiles_center[bit].to(data.device)
	data = center[data].to(std.dtype)
	data.mul_(std.unsqueeze(-1))
	data.add_(mean.unsqueeze(-1))
	return data.reshape(B, nh, D, T)

if __name__ == "__main__":
	device = 'cuda:7'
	bit = 2
	group_size = 32
	data = torch.arange(0, 128, dtype=torch.float16, device=device)
	data = data.reshape(1, 1, 1, -1).expand(-1, 8, 4096, -1)
	q_data = normal_group_fake_quant(data, group_size, bit)
	if bit > 0:
		print(q_data.shape)
		print(q_data[0, 0, 0, :])