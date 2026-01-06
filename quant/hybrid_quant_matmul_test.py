import pickle
import torch
from quant_matmul_hq import qkv_matmul, qkv_matmul_outlier
from hybrid_quant import KV_cache_compress, KV_cache_reconstruct
from hybrid_quant_outlier import KV_cache_compress_outlier, KV_cache_reconstruct_outlier
from transformers.models.llama.modeling_llama import repeat_kv

dtype=torch.float16
device = 'cuda:3'
head_dim = 128
seq_len = 4474
histroy_length = 0
compress_length = 4032
torch.manual_seed(0)
q = torch.randn(1, 32, 1, 128, dtype=dtype, device=device)
# with open('KV_caches_longbench.pkl', 'rb') as f:
#     KV_caches = pickle.load(f)

# K = KV_caches[0][0].to(device).detach()
# V = KV_caches[0][1].to(device).detach()
# del KV_caches

K = torch.randn(1, 8, 4096, 128, dtype=torch.float16, device=device)
V = torch.randn(1, 8, 4096, 128, dtype=torch.float16, device=device)

# quant_strategy = 'uniform_group'
quant_strategy = 'high_uniform_group_low_normal_group'
print(f'*****{quant_strategy}*****')

rep_k = repeat_kv(K.transpose(2, 3), 4)
w = torch.matmul(q, rep_k)
w /= (128 ** 0.5)
w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(q.dtype)
o = torch.matmul(w, repeat_kv(V, 4))

new_K, new_V = K.clone(), V.clone()

past_key_value = (None,) * 39 + (new_K, new_V, seq_len)

token_importance = torch.rand(histroy_length + compress_length)

block_size = 32
block_importance = token_importance.clone()
block_importance = block_importance.reshape(-1, block_size)
block_importance /= block_importance.shape[1]
block_importance = torch.sum(block_importance, dim=-1)
_, block_idx = torch.sort(block_importance, descending=True)
block_idx *= block_size
block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
block_idx += add
block_sorted_idx = block_idx.flatten()
_, token_sorted_idx = torch.sort(token_importance, descending=True)

block_bit_num = torch.tensor([0, 0, 0, 4032, 0], dtype=torch.int32, device=device)
token_bit_num = torch.tensor([0, 0, 0, 4032, 0], dtype=torch.int32, device=device)

past_key_value = KV_cache_compress_outlier(past_key_value, compress_length, histroy_length, block_sorted_idx, token_sorted_idx, block_bit_num, token_bit_num, quant_strategy)
(sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, new_K, new_V, kv_seq_len) = past_key_value

reconstuct_K, reconstuct_V = KV_cache_reconstruct_outlier(past_key_value, quant_strategy)

print(torch.norm(K - reconstuct_K, p=2))
print(torch.norm(V - reconstuct_V, p=2))

attn_weights, attn_output = qkv_matmul_outlier(q, past_key_value, 0, head_dim, quant_strategy)

print(torch.norm(w - attn_weights, p=2))
print(torch.norm(o - attn_output, p=2))

new_K = torch.randn(1, 8, 64, 128, dtype=dtype, device=device)
new_V = torch.randn(1, 8, 64, 128, dtype=dtype, device=device)
new_K = torch.cat((past_key_value[-3], new_K), dim=2) if past_key_value[-3] is not None else new_K
new_V = torch.cat((past_key_value[-2], new_V), dim=2) if past_key_value[-2] is not None else new_V
seq_len += 64
histroy_length += compress_length
compress_length = 64
past_key_value = past_key_value[:-3] + (new_K, new_V, seq_len)

token_importance = torch.rand(histroy_length + compress_length)
block_importance = token_importance.clone()
block_importance = block_importance.reshape(-1, block_size)
block_importance /= block_importance.shape[1]
block_importance = torch.sum(block_importance, dim=-1)
_, block_idx = torch.sort(block_importance, descending=True)
block_idx *= block_size
block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
block_idx += add
block_sorted_idx = block_idx.flatten()
_, token_sorted_idx = torch.sort(token_importance, descending=True)

block_bit_num = torch.tensor([0, 0, 0, 4096, 0], dtype=torch.int32, device=device)
token_bit_num = torch.tensor([0, 0, 0, 4096, 0], dtype=torch.int32, device=device)

past_key_value = KV_cache_compress_outlier(past_key_value, compress_length, histroy_length, block_sorted_idx, token_sorted_idx, block_bit_num, token_bit_num, quant_strategy)
reconstruct_K, reconstruct_V = KV_cache_reconstruct_outlier(past_key_value, quant_strategy)

K_idx = torch.tensor([-1])
K_idx = torch.cat((K_idx, past_key_value[1])) if past_key_value[1] is not None else K_idx
K_idx = torch.cat((K_idx, past_key_value[5])) if past_key_value[5] is not None else K_idx
K_idx = torch.cat((K_idx, past_key_value[13])) if past_key_value[13] is not None else K_idx
K_idx = torch.cat((K_idx, past_key_value[21])) if past_key_value[21] is not None else K_idx
K_idx = torch.cat((K_idx, past_key_value[29])) if past_key_value[29] is not None else K_idx

K_idx = torch.sort(K_idx[1:])[0].tolist()
for i, idx in enumerate(K_idx):
    if i != idx:
        print(idx)