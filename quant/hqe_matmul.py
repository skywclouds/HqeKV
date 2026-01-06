import torch
from torch import nn
from typing import Tuple
from flash_attn import flash_attn_func
from model.public_func import last_attn_weights

from hqe import KV_cache_hqe
from hqe_1 import KV_cache_hqe_1, KV_cache_reconstruct_hqe

from matmul import (cuda_bmm_fA_qB_uniform_group,
                    cuda_bmm_forward_qk_group_hq,
                    cuda_bmm_forward_qk_group_hq_seq,
                    cuda_bmm_forward_qk_uniform_group_token,
                    cuda_bmm_forward_wv_group_hq,
                    cuda_bmm_forward_wv_group_hq_seq,
                    cuda_bmm_forward_wv_group_outlier_hq,
                    cuda_bmm_fA_qB_uniform_group_outlier,
                    cuda_bmm_fA_qB_wv_uniform_group,
                    cuda_bmm_fA_qB_wv_uniform_group_outlier,
                    cuda_bmm_fA_qB_normal_group,
                    cuda_bmm_fA_qB_normal_group_outlier,
                    cuda_bmm_forward_qk_normal_group_token,
                    cuda_bmm_fA_qB_wv_normal_group_outlier)

from transformers.models.llama.modeling_llama import repeat_kv

def qkv_matmul_hqe(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    if quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_forward_qk_uniform_group_token
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_forward_qk_normal_group_token
    else:
        raise ValueError('invalid quant_strategy')
    
    real_len = 0
    if sixteen_V is not None:
        _, nh_kv, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
    if eight_V is not None:
        nh_kv = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
    if four_V is not None:
        nh_kv = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
    if two_V is not None:
        nh_kv = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
    if one_V is not None:
        nh_kv = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
    if new_V is not None:
        _, nh_kv, new_len, head_dim = new_V.shape
        real_len += new_len
    if real_len == 0:
        raise ValueError('invalid past_key_value')

    # 计算softmax(qK^T)
    attn_weights = cuda_bmm_forward_qk_group_hq(block_size, nh_kv, q, sixteen_K, sixteen_K_idx,
                        eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean,
                        four_K, four_K_idx, four_K_scale_std, four_K_mn_mean,
                        two_K, two_K_idx, two_K_scale_std, two_K_mn_mean,
                        one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, new_K)
    attn_weights /= (head_dim ** 0.5)
    # attn_weights += attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = cuda_bmm_forward_wv_group_hq(group_size, nh_kv, attn_weights, sixteen_V, sixteen_V_idx,
                        eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean,
                        four_V, four_V_idx, four_V_scale_std, four_V_mn_mean,
                        two_V, two_V_idx, two_V_scale_std, two_V_mn_mean,
                        one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, new_V)
    
    return attn_weights, attn_output

def qkv_matmul_hqe_1(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    real_len = 0
    if sixteen_V is not None:
        _, nh_kv, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
    if eight_V is not None:
        nh_kv = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
    if four_V is not None:
        nh_kv = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
    if two_V is not None:
        nh_kv = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
    if one_V is not None:
        nh_kv = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
    if new_V is not None:
        _, nh_kv, new_len, head_dim = new_V.shape
        real_len += new_len
    if real_len == 0:
        raise ValueError('invalid past_key_value')
    
    attn_weights = cuda_bmm_forward_qk_group_hq_seq(block_size, nh_kv, q, sixteen_K,
                        eight_K,eight_K_scale_std, eight_K_mn_mean, four_K, four_K_scale_std, four_K_mn_mean, 
                        two_K, two_K_scale_std, two_K_mn_mean, one_K, one_K_scale_std, one_K_mn_mean, new_K)
    attn_weights /= (head_dim ** 0.5)
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    attn_output = cuda_bmm_forward_wv_group_hq_seq(group_size, nh_kv, attn_weights, sixteen_V,
                        eight_V, eight_V_scale_std, eight_V_mn_mean, four_V, four_V_scale_std, four_V_mn_mean, 
                        two_V, two_V_scale_std, two_V_mn_mean, one_V, one_V_scale_std, one_V_mn_mean, new_V)

    return attn_weights, attn_output

def qkv_matmul_hqe_V_serial(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''对 qk 采用并行计算, 对 wv 采用串行计算, 因为对 wv 采用并行计算后会出现乱码, 原因未知'''
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    if quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
    else:
        raise ValueError('invalid quant_strategy')
    
    real_len = 0
    nh_q = q.shape[1]
    if sixteen_V is not None:
        _, nh_kv, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
    if eight_V is not None:
        nh_kv = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
    if four_V is not None:
        nh_kv = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
    if two_V is not None:
        nh_kv = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
    if one_V is not None:
        nh_kv = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
    if new_V is not None:
        _, nh_kv, new_len, head_dim = new_V.shape
        real_len += new_len
    if real_len == 0:
        raise ValueError('invalid past_key_value')
    
    sixteen_num = sixteen_K.shape[-1] if sixteen_K is not None else 0
    eight_num = eight_K.shape[-1] * 4 if eight_K is not None else 0
    four_num = four_K.shape[-1] * 8 if four_K is not None else 0
    two_num = two_K.shape[-1] * 16 if two_K is not None else 0
    one_num = one_K.shape[-1] * 32 if one_K is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)

    n_rep = nh_q // nh_kv
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    attn_weights = cuda_bmm_forward_qk_group_hq_seq(block_size, nh_kv, q, sixteen_K,
                        eight_K,eight_K_scale_std, eight_K_mn_mean, four_K, four_K_scale_std, four_K_mn_mean, 
                        two_K, two_K_scale_std, two_K_mn_mean, one_K, one_K_scale_std, one_K_mn_mean, new_K)
    attn_weights /= (head_dim ** 0.5)
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    
    sixteen_num = sixteen_V.shape[-2] if sixteen_V is not None else 0
    eight_num = eight_V.shape[-2] if eight_V is not None else 0
    four_num = four_V.shape[-2] if four_V is not None else 0
    two_num = two_V.shape[-2] if two_V is not None else 0
    one_num = one_V.shape[-2] if one_V is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)
    
    if sixteen_V is not None:
        sixteen_V_rep =  repeat_kv(sixteen_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, 0:bit_nums[0]], sixteen_V_rep)
    if eight_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[0]:bit_nums[1]], eight_V, eight_V_scale_std, eight_V_mn_mean, 8)
    if four_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[1]:bit_nums[2]], four_V, four_V_scale_std, four_V_mn_mean, 4)
    if two_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[2]:bit_nums[3]], two_V, two_V_scale_std, two_V_mn_mean, 2)
    if one_V is not None:
        attn_output += V_cuda_fA_qB_low(group_size, attn_weights[:, :, :, bit_nums[3]:bit_nums[4]], one_V, one_V_scale_std, one_V_mn_mean, 1)
    if new_V is not None:
        new_V_rep = repeat_kv(new_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, -new_len:], new_V_rep)
    
    return attn_weights, attn_output

def qkv_matmul_hqe_1_serial(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    if quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
    else:
        raise ValueError('invalid quant_strategy')
    
    real_len = 0
    nh_q = q.shape[1]
    if sixteen_V is not None:
        _, nh_kv, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
    if eight_V is not None:
        nh_kv = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
    if four_V is not None:
        nh_kv = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
    if two_V is not None:
        nh_kv = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
    if one_V is not None:
        nh_kv = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
    if new_V is not None:
        _, nh_kv, new_len, head_dim = new_V.shape
        real_len += new_len
    if real_len == 0:
        raise ValueError('invalid past_key_value')
    
    sixteen_num = sixteen_K.shape[-1] if sixteen_K is not None else 0
    eight_num = eight_K.shape[-1] * 4 if eight_K is not None else 0
    four_num = four_K.shape[-1] * 8 if four_K is not None else 0
    two_num = two_K.shape[-1] * 16 if two_K is not None else 0
    one_num = one_K.shape[-1] * 32 if one_K is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)

    n_rep = nh_q // nh_kv
    attn_weights = torch.empty(q.shape[0], q.shape[1], 1, real_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    # 计算softmax(qK^T)
    if sixteen_K is not None:
        sixteen_K_rep = repeat_kv(sixteen_K, n_rep)
        sixteen_attn_weight = torch.matmul(q, sixteen_K_rep)
        attn_weights[:, :, :, 0:bit_nums[0]] = sixteen_attn_weight
    if eight_K is not None:
        eight_attn_weight = K_cuda_fA_qB_high(block_size, q, eight_K, eight_K_scale_std, eight_K_mn_mean, 8)
        attn_weights[:, :, :, bit_nums[0]:bit_nums[1]] = eight_attn_weight
    if four_K is not None:
        four_attn_weight = K_cuda_fA_qB_high(block_size, q, four_K, four_K_scale_std, four_K_mn_mean, 4)
        attn_weights[:, :, :, bit_nums[1]:bit_nums[2]] = four_attn_weight
    if two_K is not None:
        two_attn_weight = K_cuda_fA_qB_high(block_size, q, two_K, two_K_scale_std, two_K_mn_mean, 2)
        attn_weights[:, :, :, bit_nums[2]:bit_nums[3]] = two_attn_weight
    if one_K is not None:
        one_attn_weight = K_cuda_fA_qB_low(block_size, q, one_K, one_K_scale_std, one_K_mn_mean, 1)
        attn_weights[:, :, :, bit_nums[3]:bit_nums[4]] = one_attn_weight
    if new_K is not None:
        new_K_rep = repeat_kv(new_K, n_rep)
        new_attn_weight = torch.matmul(q, new_K_rep.transpose(2, 3))
        attn_weights[:, :, :, -new_len:] = new_attn_weight
    attn_weights /= (head_dim ** 0.5)
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    sixteen_num = sixteen_V.shape[-2] if sixteen_V is not None else 0
    eight_num = eight_V.shape[-2] if eight_V is not None else 0
    four_num = four_V.shape[-2] if four_V is not None else 0
    two_num = two_V.shape[-2] if two_V is not None else 0
    one_num = one_V.shape[-2] if one_V is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)
    
    if sixteen_V is not None:
        sixteen_V_rep =  repeat_kv(sixteen_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, 0:bit_nums[0]], sixteen_V_rep)
    if eight_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[0]:bit_nums[1]], eight_V, eight_V_scale_std, eight_V_mn_mean, 8)
    if four_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[1]:bit_nums[2]], four_V, four_V_scale_std, four_V_mn_mean, 4)
    if two_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, bit_nums[2]:bit_nums[3]], two_V, two_V_scale_std, two_V_mn_mean, 2)
    if one_V is not None:
        attn_output += V_cuda_fA_qB_low(group_size, attn_weights[:, :, :, bit_nums[3]:bit_nums[4]], one_V, one_V_scale_std, one_V_mn_mean, 1)
    if new_V is not None:
        new_V_rep = repeat_kv(new_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, -new_len:], new_V_rep)
    
    return attn_weights, attn_output


def qkv_matmul_hqe_2(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    if quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
    elif quant_strategy == 'uniform_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group
    else:
        raise ValueError('invalid quant_strategy')
    
    real_len = 0
    nh_q = q.shape[1]
    if sixteen_V is not None:
        _, nh_kv, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
    if eight_V is not None:
        nh_kv = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
    if four_V is not None:
        nh_kv = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
    if two_V is not None:
        nh_kv = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
    if one_V is not None:
        nh_kv = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
    if new_V is not None:
        _, nh_kv, new_len, head_dim = new_V.shape
        real_len += new_len
    if real_len == 0:
        raise ValueError('invalid past_key_value')
    n_rep = nh_q // nh_kv
    attn_weights = torch.empty(q.shape[0], q.shape[1], 1, real_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    # 计算softmax(qK^T)
    if sixteen_K is not None:
        sixteen_K_rep = repeat_kv(sixteen_K, n_rep)
        sixteen_attn_weight = torch.matmul(q, sixteen_K_rep)
        attn_weights[:, :, :, sixteen_K_idx] = sixteen_attn_weight
    if eight_K is not None:
        eight_attn_weight = K_cuda_fA_qB_high(block_size, q, eight_K, eight_K_scale_std, eight_K_mn_mean, 8)
        attn_weights[:, :, :, eight_K_idx] = eight_attn_weight
    if four_K is not None:
        four_attn_weight = K_cuda_fA_qB_high(block_size, q, four_K, four_K_scale_std, four_K_mn_mean, 4)
        attn_weights[:, :, :, four_K_idx] = four_attn_weight
    if two_K is not None:
        two_attn_weight = K_cuda_fA_qB_high(block_size, q, two_K, two_K_scale_std, two_K_mn_mean, 2)
        attn_weights[:, :, :, two_K_idx] = two_attn_weight
    if one_K is not None:
        one_attn_weight = K_cuda_fA_qB_low(block_size, q, one_K, one_K_scale_std, one_K_mn_mean, 1)
        attn_weights[:, :, :, one_K_idx] = one_attn_weight
    if new_K is not None:
        new_K_rep = repeat_kv(new_K, n_rep)
        new_attn_weight = torch.matmul(q, new_K_rep.transpose(2, 3))
        attn_weights[:, :, :, -new_len:] = new_attn_weight
    attn_weights /= (head_dim ** 0.5)
    # attn_weights += attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    
    if sixteen_V is not None:
        sixteen_V_rep =  repeat_kv(sixteen_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, sixteen_V_idx], sixteen_V_rep)
    if eight_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, eight_V_idx], eight_V, eight_V_scale_std, eight_V_mn_mean, 8)
    if four_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, four_V_idx], four_V, four_V_scale_std, four_V_mn_mean, 4)
    if two_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, two_V_idx], two_V, two_V_scale_std, two_V_mn_mean, 2)
    if one_V is not None:
        attn_output += V_cuda_fA_qB_low(group_size, attn_weights[:, :, :, one_V_idx], one_V, one_V_scale_std, one_V_mn_mean, 1)
    if new_V is not None:
        new_V_rep = repeat_kv(new_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, -new_len:], new_V_rep)
    
    return attn_weights, attn_output

if __name__ == '__main__':
    quant_strategy = 'high_uniform_group_low_normal_group'
    dtype = torch.float16
    device = torch.device('cuda:1')
    last_weights = last_attn_weights()
    histroy_length = 0
    seq_len = 9216
    compress_length = 8192
    q = torch.rand(1, 8, 1, 128, dtype=dtype, device=device)
    new_K = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)
    new_V = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)

    past_key_value = (None,) * 39 + (new_K, new_V, seq_len)
    token_importance = torch.rand(histroy_length + compress_length, device=device)
    block_size = 32
    K_bit_num = torch.tensor([64, 128, 256, 7520, 128, 96], dtype=torch.int32, device=device)
    V_bit_num = K_bit_num.clone()

    past_key_value = KV_cache_hqe_1(past_key_value, compress_length, histroy_length, K_bit_num, V_bit_num, token_importance, quant_strategy)

    q = torch.rand(1, 32, 1, 128, dtype=torch.float16, device=device)
    _, out_1 = qkv_matmul_hqe_1(q, past_key_value, 0, 128, quant_strategy)
    
    dq_K, dq_V = KV_cache_reconstruct_hqe(past_key_value, quant_strategy)
    q.transpose_(1, 2)
    dq_K.transpose_(1, 2)
    dq_V.transpose_(1, 2)
    out_2 = flash_attn_func(q, dq_K, dq_K)

    error = torch.abs(out_1 - out_2).flatten().tolist()
    print('max error:', max(error))
    print('avg error:', sum(error) / len(error))