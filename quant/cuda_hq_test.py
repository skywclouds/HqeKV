import torch
import kvhq_gemv
from flash_attn import flash_attn_func
from matmul import (cuda_bmm_fA_qB_uniform_group,
                    cuda_bmm_forward_qk_group_hq,
                    cuda_bmm_forward_qk_group_hq_seq,
                    cuda_bmm_forward_wv_group_hq, 
                    cuda_bmm_forward_wv_group_hq_seq, 
                    cuda_bmm_forward_wv_group_outlier_hq)
from pack_unpack import (
	uniform_group_quantize_pack_last_dim,
    uniform_group_dequantize_unpack_last_dim,
	normal_group_quantize_pack_last_dim,
    normal_group_dequantize_unpack_last_dim)
import time
from transformers.models.llama.modeling_llama import repeat_kv
import pickle

def test_qk(q: torch.Tensor, k: torch.Tensor, quant_len: int, new_len: int, device: torch.device):
    
    idx = torch.arange(0, quant_len, dtype=torch.int64, device=device)
    shuffled_idx = idx[torch.randperm(len(idx))]

    # bit_idx = torch.randint(low=1, high=seq_len//32, size=(4,), dtype=torch.int64, device=device)
    # bit_idx = 32 * torch.sort(bit_idx)[0] 
    bit_idx = torch.tensor([64, 128, 256, 7168, quant_len], dtype=torch.int64, device=device)

    sixteen = k[:, :, :bit_idx[0], :].clone().contiguous()
    sixteen_idx = shuffled_idx[:bit_idx[0]]
    q_sixteen = sixteen.transpose(2, 3)
    
    # eight = k[:, :, bit_idx[0]:bit_idx[1], :].clone().contiguous()
    eight_idx = shuffled_idx[bit_idx[0]:bit_idx[1]]
    # q_eight, eight_scale, eight_zero = uniform_group_quantize_pack_last_dim(eight.clone().transpose(2, 3), 32, 8)
    q_eight, eight_scale, eight_zero = None, None, None
    # dq_eight = uniform_group_dequantize_unpack_last_dim(q_eight.clone(), eight_scale, eight_zero, 32, 8)
    dq_eight = None
    eight_idx = None

    four = k[:, :, bit_idx[0]:bit_idx[2], :].clone().contiguous()
    four_idx = shuffled_idx[bit_idx[0]:bit_idx[2]]
    q_four, four_scale, four_zero = uniform_group_quantize_pack_last_dim(four.clone().transpose(2, 3), 32, 4)
    dq_four = uniform_group_dequantize_unpack_last_dim(q_four.clone(), four_scale, four_zero, 32, 4)
    # four_idx = None

    two = k[:, :, bit_idx[2]:bit_idx[4], :].clone().contiguous()
    two_idx = shuffled_idx[bit_idx[2]:bit_idx[4]]
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone().transpose(2, 3), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)
    # two_idx = None

    # one = k[:, :, bit_idx[3]:bit_idx[4], :].clone().contiguous()
    one_idx = shuffled_idx[bit_idx[3]:bit_idx[4]]
    # q_one, one_std, one_mean = normal_group_quantize_pack_last_dim(one.clone().transpose(2, 3), 32, 1)
    q_one, one_std, one_mean = None, None, None
    # dq_one = normal_group_dequantize_unpack_last_dim(q_one.clone(), one_std, one_mean, 32, 1)
    dq_one = None
    one_idx = None

    new_K = k[:, :, bit_idx[4]:, :].clone().contiguous()
    # new_K = None

    dq_k = torch.zeros_like(k)
    dq_k[:, :, sixteen_idx, :] = sixteen
    # dq_k[:, :, eight_idx, :] = dq_eight.transpose(2, 3)
    dq_k[:, :, four_idx, :] = dq_four.transpose(2, 3)
    dq_k[:, :, two_idx, :] = dq_two.transpose(2, 3)
    # dq_k[:, :, one_idx, :] = dq_one.transpose(2, 3)
    dq_k[:, :, -new_len:, :] = new_K

    start_time = time.perf_counter()
    # print('func_out_start', start_time)
    c = cuda_bmm_forward_qk_group_hq(
        32, 8, q, q_sixteen, sixteen_idx, q_eight, eight_idx, eight_scale, eight_zero,
        q_four, four_idx, four_scale, four_zero, 
        q_two, two_idx, two_scale, two_zero,
        q_one, one_idx, one_std, one_mean, new_K
    )
    end_time = time.perf_counter()
    # print('func_out_end', end_time)
    # print('time:', end_time - start_time)
    d = torch.matmul(q, repeat_kv(dq_k.transpose(2, 3), 4))

    error = torch.abs(c - d).flatten().tolist()
    print('qk max error:', max(error))
    print('qk avg error:', sum(error) / len(error))

    return c, dq_k

def test_qk_kivi(q: torch.Tensor, k: torch.Tensor, quant_len: int, new_len: int, device: torch.device):
    nh = q.shape[1]
    B, _, seq_len, _ = k.shape
    two = k[:, :, :quant_len, :].clone().contiguous()
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone().transpose(2, 3), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)

    new_K = k[:, :, -new_len:, :]

    dq_k = torch.zeros_like(k)
    dq_k[:, :, :quant_len, :] = dq_two.transpose(2, 3)
    dq_k[:, :, -new_len:, :] = new_K

    c = torch.empty(B, nh, 1, seq_len, dtype=q.dtype, device=q.device)
    c[:, :, :, :quant_len] = cuda_bmm_fA_qB_uniform_group(32, q, q_two, two_scale, two_zero, 2)
    c[:, :, :, -new_len:] = torch.matmul(q, repeat_kv(new_K.transpose(2, 3), 4))

    d = torch.matmul(q, repeat_kv(dq_k.transpose(2, 3), 4))

    error = torch.abs(c - d).flatten().tolist()
    print('qk max error:', max(error))
    print('qk avg error:', sum(error) / len(error))

    return c, dq_k

def test_qk_seq(q: torch.Tensor, k: torch.Tensor, quant_len: int, new_len: int, device: torch.device):
    
    idx = torch.arange(0, quant_len, dtype=torch.int64, device=device)

    # bit_idx = torch.randint(low=1, high=seq_len//32, size=(4,), dtype=torch.int64, device=device)
    # bit_idx = 32 * torch.sort(bit_idx)[0] 
    bit_idx = torch.tensor([64, 128, 256, 7168, quant_len], dtype=torch.int64, device=device)

    sixteen = k[:, :, :bit_idx[0], :].clone().contiguous()
    q_sixteen = sixteen.transpose(2, 3)
    
    # eight = k[:, :, bit_idx[0]:bit_idx[1], :].clone().contiguous()
    # q_eight, eight_scale, eight_zero = uniform_group_quantize_pack_last_dim(eight.clone().transpose(2, 3), 32, 8)
    q_eight, eight_scale, eight_zero = None, None, None
    # dq_eight = uniform_group_dequantize_unpack_last_dim(q_eight.clone(), eight_scale, eight_zero, 32, 8)
    dq_eight = None
    eight_idx = None

    four = k[:, :, bit_idx[0]:bit_idx[2], :].clone().contiguous()
    q_four, four_scale, four_zero = uniform_group_quantize_pack_last_dim(four.clone().transpose(2, 3), 32, 4)
    dq_four = uniform_group_dequantize_unpack_last_dim(q_four.clone(), four_scale, four_zero, 32, 4)
    # four_idx = None

    two = k[:, :, bit_idx[2]:bit_idx[4], :].clone().contiguous()
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone().transpose(2, 3), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)
    # two_idx = None

    # one = k[:, :, bit_idx[3]:bit_idx[4], :].clone().contiguous()
    # q_one, one_std, one_mean = normal_group_quantize_pack_last_dim(one.clone().transpose(2, 3), 32, 1)
    q_one, one_std, one_mean = None, None, None
    # dq_one = normal_group_dequantize_unpack_last_dim(q_one.clone(), one_std, one_mean, 32, 1)
    dq_one = None
    one_idx = None

    new_K = k[:, :, bit_idx[4]:, :].clone().contiguous()
    # new_K = None

    dq_k = torch.zeros_like(k)
    dq_k[:, :, :bit_idx[0], :] = sixteen
    # dq_k[:, :, bit_idx[0]:bit_idx[1], :] = dq_eight.transpose(2, 3)
    dq_k[:, :, bit_idx[0]:bit_idx[2], :] = dq_four.transpose(2, 3)
    dq_k[:, :, bit_idx[2]:bit_idx[4], :] = dq_two.transpose(2, 3)
    # dq_k[:, :, bit_idx[3]:bit_idx[4], :] = dq_one.transpose(2, 3)
    dq_k[:, :, -new_len:, :] = new_K

    start_time = time.perf_counter()
    # print('func_out_start', start_time)
    c = cuda_bmm_forward_qk_group_hq_seq(
        32, 8, q, q_sixteen, q_eight, eight_scale, eight_zero,
        q_four, four_scale, four_zero, 
        q_two, two_scale, two_zero,
        q_one, one_std, one_mean, new_K
    )
    end_time = time.perf_counter()
    # print('func_out_end', end_time)
    # print('time:', end_time - start_time)
    d = torch.matmul(q, repeat_kv(dq_k.transpose(2, 3), 4))

    error = torch.abs(c - d).flatten().tolist()
    print('qk max error:', max(error))
    print('qk avg error:', sum(error) / len(error))

    return c, dq_k

def test_wv(w: torch.Tensor, v: torch.Tensor, quant_len: int, new_len: int, device: torch.device):

    idx = torch.arange(0, quant_len, dtype=torch.int64, device=device)
    shuffled_idx = idx[torch.randperm(len(idx))]

    # bit_idx = torch.randint(low=1, high=seq_len, size=(4,), dtype=torch.int64, device=device)
    # bit_idx = torch.sort(bit_idx)[0]
    bit_idx = torch.tensor([64, 128, 256, 7168, quant_len], dtype=torch.int64, device=device)

    sixteen = v[:, :, :bit_idx[0], :].clone().contiguous()
    sixteen_idx = shuffled_idx[:bit_idx[0]]
    q_sixteen = sixteen.clone()
    
    # eight = v[:, :, bit_idx[0]:bit_idx[1], :].clone().contiguous()
    # eight_idx = shuffled_idx[bit_idx[0]:bit_idx[1]]
    # q_eight, eight_scale, eight_zero = uniform_group_quantize_pack_last_dim(eight.clone(), 32, 8)
    q_eight, eight_scale, eight_zero = None, None, None
    # dq_eight = uniform_group_dequantize_unpack_last_dim(q_eight.clone(), eight_scale, eight_zero, 32, 8)
    dq_eight = None
    eight_idx = None

    four = v[:, :, bit_idx[0]:bit_idx[2], :].clone().contiguous()
    four_idx = shuffled_idx[bit_idx[0]:bit_idx[2]]
    q_four, four_scale, four_zero = uniform_group_quantize_pack_last_dim(four.clone(), 32, 4)
    dq_four = uniform_group_dequantize_unpack_last_dim(q_four.clone(), four_scale, four_zero, 32, 4)
    # four_idx = None

    two = v[:, :, bit_idx[2]:bit_idx[3], :].clone().contiguous()
    two_idx = shuffled_idx[bit_idx[2]:bit_idx[3]]
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone(), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)
    # two_idx = None

    one = v[:, :, bit_idx[3]:bit_idx[4], :].clone().contiguous()
    one_idx = shuffled_idx[bit_idx[3]:bit_idx[4]]
    # one_idx = None
    q_one, one_std, one_mean = normal_group_quantize_pack_last_dim(one.clone(), 32, 1)
    dq_one = normal_group_dequantize_unpack_last_dim(q_one.clone(), one_std, one_mean, 32, 1)

    new_V = v[:, :, bit_idx[4]:, :].clone().contiguous()
    # new_V = None

    dq_v = torch.zeros(1, 8, quant_len+new_len, 128, dtype=torch.float16, device=device)
    dq_v[:, :, sixteen_idx, :] = sixteen
    # dq_v[:, :, eight_idx, :] = dq_eight
    dq_v[:, :, four_idx, :] = dq_four
    dq_v[:, :, two_idx, :] = dq_two
    dq_v[:, :, one_idx, :] = dq_one
    dq_v[:, :, -new_len:, :] = new_V
    # dq_v.scatter_(dim=-1, index=data_outliers_idx, src=data_outliers)

    start_time = time.perf_counter()
    # print('func_out_start', start_time)
    c = cuda_bmm_forward_wv_group_hq(
        32, 8, w, q_sixteen, sixteen_idx, q_eight, eight_idx, eight_scale, eight_zero,
        q_four, four_idx, four_scale, four_zero, 
        q_two, two_idx, two_scale, two_zero,
        q_one, one_idx, one_std, one_mean, new_V
    )
    end_time = time.perf_counter()
    # print('func_out_end', end_time)
    # print('time:', end_time - start_time)
    d = torch.matmul(w, repeat_kv(dq_v, 4))

    error = torch.abs(c - d).flatten().tolist()
    print('qk max error:', max(error))
    print('qk avg error:', sum(error) / len(error))

    return c, dq_v

def test_wv_kivi(w: torch.Tensor, v: torch.Tensor, quant_len: int, new_len: int, device: torch.device):
    nh = w.shape[1]
    B, _, seq_len, head_dim = v.shape
    two = v[:, :, :quant_len, :].clone().contiguous()
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone(), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)

    new_V = v[:, :, -new_len:, :]

    dq_v = torch.zeros_like(k)
    dq_v[:, :, :quant_len, :] = dq_two
    dq_v[:, :, -new_len:, :] = new_V

    c = torch.zeros(B, nh, seq_len, head_dim, dtype=w.dtype, device=w.device)
    c += cuda_bmm_fA_qB_uniform_group(32, w[:, :, :, :quant_len], q_two, two_scale, two_zero, 2)
    c += torch.matmul(w[:, :, :, -new_len:], repeat_kv(new_V, 4))

    d = torch.matmul(w, repeat_kv(dq_v, 4))

    error = torch.abs(c - d).flatten().tolist()
    print('qk max error:', max(error))
    print('qk avg error:', sum(error) / len(error))

    return c, dq_v

def test_wv_seq(w: torch.Tensor, v: torch.Tensor, quant_len: int, new_len: int, device: torch.device):

    idx = torch.arange(0, quant_len, dtype=torch.int64, device=device)

    # bit_idx = torch.randint(low=1, high=seq_len, size=(4,), dtype=torch.int64, device=device)
    # bit_idx = torch.sort(bit_idx)[0]
    bit_idx = torch.tensor([64, 128, 256, 7168, quant_len], dtype=torch.int64, device=device)

    sixteen = v[:, :, :bit_idx[0], :].clone().contiguous()
    q_sixteen = sixteen.clone()
    
    # eight = v[:, :, bit_idx[0]:bit_idx[1], :].clone().contiguous()
    # eight_idx = shuffled_idx[bit_idx[0]:bit_idx[1]]
    # q_eight, eight_scale, eight_zero = uniform_group_quantize_pack_last_dim(eight.clone(), 32, 8)
    q_eight, eight_scale, eight_zero = None, None, None
    # dq_eight = uniform_group_dequantize_unpack_last_dim(q_eight.clone(), eight_scale, eight_zero, 32, 8)
    dq_eight = None
    eight_idx = None

    four = v[:, :, bit_idx[0]:bit_idx[2], :].clone().contiguous()
    q_four, four_scale, four_zero = uniform_group_quantize_pack_last_dim(four.clone(), 32, 4)
    dq_four = uniform_group_dequantize_unpack_last_dim(q_four.clone(), four_scale, four_zero, 32, 4)
    # four_idx = None

    two = v[:, :, bit_idx[2]:bit_idx[3], :].clone().contiguous()
    q_two, two_scale, two_zero = uniform_group_quantize_pack_last_dim(two.clone(), 32, 2)
    dq_two = uniform_group_dequantize_unpack_last_dim(q_two.clone(), two_scale, two_zero, 32, 2)
    # two_idx = None

    one = v[:, :, bit_idx[3]:bit_idx[4], :].clone().contiguous()
    # one_idx = None
    q_one, one_std, one_mean = normal_group_quantize_pack_last_dim(one.clone(), 32, 1)
    dq_one = normal_group_dequantize_unpack_last_dim(q_one.clone(), one_std, one_mean, 32, 1)

    new_V = v[:, :, bit_idx[4]:, :].clone().contiguous()
    # new_V = None

    dq_v = torch.zeros(1, 8, quant_len+new_len, 128, dtype=torch.float16, device=device)
    dq_v[:, :, :bit_idx[0], :] = sixteen
    # dq_v[:, :, eight_idx, :] = dq_eight
    dq_v[:, :, bit_idx[0]:bit_idx[2], :] = dq_four
    dq_v[:, :, bit_idx[2]:bit_idx[3], :] = dq_two
    dq_v[:, :, bit_idx[3]:bit_idx[4], :] = dq_one
    dq_v[:, :, -new_len:, :] = new_V

    start_time = time.perf_counter()
    # print('func_out_start', start_time)
    c = cuda_bmm_forward_wv_group_hq_seq(
        32, 8, w, q_sixteen, q_eight, eight_scale, eight_zero,
        q_four, four_scale, four_zero, 
        q_two, two_scale, two_zero,
        q_one, one_std, one_mean, new_V
    )
    end_time = time.perf_counter()
    # print('func_out_end', end_time)
    # print('time:', end_time - start_time)
    d = torch.matmul(w, repeat_kv(dq_v, 4))

    error = torch.abs(c - d).flatten().tolist()
    print('wv max error:', max(error))
    print('wv avg error:', sum(error) / len(error))

    return c, dq_v

if __name__ == '__main__':
    new_len = 1023
    quant_len = 8 * 1024
    device = torch.device('cuda:5')
    
    for _ in range(1):
        q = torch.rand(1, 32, 1, 128, dtype=torch.float16, device=device)
        k = torch.rand(1, 8, quant_len+new_len, 128, dtype=torch.float16, device=device)
        w, dq_k = test_qk_seq(q, k, quant_len, new_len, device)
    w /= 128 ** 0.5
    w = torch.softmax(w, dim=-1)
    for _ in range(1):
        # w = torch.rand(1, 32, 1, quant_len+new_len, dtype=torch.float16, device=device)
        # w = torch.softmax(w, dim=-1)
        v = torch.rand(1, 8, quant_len+new_len, 128, dtype=torch.float16, device=device)
        out, dq_v = test_wv_seq(w, v, quant_len, new_len, device)
        if torch.isnan(out).any():
            print('nan in attn_output.')
            break
    q.transpose_(1, 2)
    dq_k.transpose_(1, 2)
    dq_v.transpose_(1, 2)
    out_2 = flash_attn_func(q, dq_k, dq_v)
    error = torch.abs(out - out_2).flatten().tolist()
    print('max error:', max(error))
    print('avg error:', sum(error) / len(error))