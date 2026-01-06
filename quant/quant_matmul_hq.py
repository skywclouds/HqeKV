import torch
from torch import nn
from typing import Tuple
from matmul import (cuda_bmm_fA_qB_uniform, 
                    cuda_bmm_fA_qB_uniform_outlier,
                    cuda_bmm_fA_qB_normal,
                    cuda_bmm_fA_qB_normal_outlier,
                    cuda_bmm_fA_qB_uniform_group,
                    cuda_bmm_forward_qk_group_hq,
                    cuda_bmm_forward_wv_group_hq,
                    cuda_bmm_forward_wv_group_outlier_hq,
                    cuda_bmm_fA_qB_uniform_group_outlier,
                    cuda_bmm_fA_qB_wv_uniform_group,
                    cuda_bmm_fA_qB_wv_uniform_group_outlier,
                    cuda_bmm_fA_qB_normal_group,
                    cuda_bmm_fA_qB_normal_group_outlier,
                    cuda_bmm_fA_qB_wv_normal_group_outlier)

from transformers.models.llama.modeling_llama import repeat_kv
import pickle
import time

def qkv_matmul(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''计算q与量化的kv相乘'''
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value
    
    new_len = new_K.shape[2] if new_K is not None else 0
    nh_q = q.shape[1]
    nh_kv = 0
    if sixteen_K is not None:
        nh_kv = sixteen_K.shape[1]
    elif eight_K is not None:
        nh_kv = eight_K.shape[1]
    elif four_K is not None:
        nh_kv = four_K.shape[1]
    elif two_K is not None:
        nh_kv = two_K.shape[1]
    elif one_K is not None:
        nh_kv = one_K.shape[1]
    elif new_K is not None:
        nh_kv = new_K.shape[1]
    else:
        raise ValueError('invalid past_key_value')
    n_rep = nh_q // nh_kv
    attn_weights = torch.zeros(q.shape[0], q.shape[1], 1, kv_seq_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    if quant_strategy == 'uniform':
        K_cuda_fA_qB_high = K_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group
        V_cuda_fA_qB_high = V_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform
    elif quant_strategy == 'normal':
        K_cuda_fA_qB_high = K_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
        V_cuda_fA_qB_high = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal
    elif quant_strategy == 'uniform_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group
    elif quant_strategy == 'normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_normal_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
    elif quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = V_cuda_fA_qB_high = cuda_bmm_fA_qB_wv_uniform_group
        K_cuda_fA_qB_low = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
    else:
        raise ValueError('invalid quant_strategy')

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
    attn_weights += attention_mask
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

def qkv_matmul_hq(
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
    
    nh_kv = 0
    if sixteen_K is not None:
        nh_kv = sixteen_K.shape[1]
    elif eight_K is not None:
        nh_kv = eight_K.shape[1]
    elif four_K is not None:
        nh_kv = four_K.shape[1]
    elif two_K is not None:
        nh_kv = two_K.shape[1]
    elif one_K is not None:
        nh_kv = one_K.shape[1]
    elif new_K is not None:
        nh_kv = new_K.shape[1]
    else:
        raise ValueError('invalid past_key_value')
    
    attn_weights = torch.zeros(q.shape[0], q.shape[1], 1, kv_seq_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    # 计算softmax(qK^T)
    attn_weights = cuda_bmm_forward_qk_group_hq(block_size, nh_kv, q, sixteen_K, sixteen_K_idx,
                        eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean,
                        four_K, four_K_idx, four_K_scale_std, four_K_mn_mean,
                        two_K, two_K_idx, two_K_scale_std, two_K_mn_mean,
                        one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, new_K)
    attn_weights /= (head_dim ** 0.5)
    attn_weights += attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    
    attn_output = cuda_bmm_forward_wv_group_hq(group_size, nh_kv, attn_weights, sixteen_V, sixteen_V_idx,
                        eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean,
                        four_V, four_V_idx, four_V_scale_std, four_V_mn_mean,
                        two_V, two_V_idx, two_V_scale_std, two_V_mn_mean,
                        one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, new_V)
    
    return attn_weights, attn_output

def qkv_matmul_outlier_hq(
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

    nh_kv = 0
    if sixteen_K is not None:
        nh_kv = sixteen_K.shape[1]
    elif eight_K is not None:
        nh_kv = eight_K.shape[1]
    elif four_K is not None:
        nh_kv = four_K.shape[1]
    elif two_K is not None:
        nh_kv = two_K.shape[1]
    elif one_K is not None:
        nh_kv = one_K.shape[1]
    elif new_K is not None:
        nh_kv = new_K.shape[1]
    else:
        raise ValueError('invalid past_key_value')

    attn_weights = torch.zeros(q.shape[0], q.shape[1], 1, kv_seq_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    # 计算softmax(qK^T)
    attn_weights = cuda_bmm_forward_qk_group_hq(block_size, nh_kv, q, sixteen_K, sixteen_K_idx,
                        eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean,
                        four_K, four_K_idx, four_K_scale_std, four_K_mn_mean,
                        two_K, two_K_idx, two_K_scale_std, two_K_mn_mean,
                        one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, new_K)
    attn_weights /= (head_dim ** 0.5)
    attn_weights += attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)

    # if torch.isnan(hq_attn_output).any():
    #     print('nan in hq_attn_output.')
        # cuda_wv_arguments = [group_size, attn_weights, sixteen_V, sixteen_V_idx,
        #                 eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean,
        #                 four_V, four_V_idx, four_V_scale_std, four_V_mn_mean,
        #                 two_V, two_V_idx, two_V_scale_std, two_V_mn_mean,
        #                 one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx]
        # cuda_wv_arguments = [elem.cpu() if isinstance(elem, torch.Tensor) else elem for elem in cuda_wv_arguments]
        # with open('cuda_wv_arguments.pkl', 'wb') as f:
        #     pickle.dump(cuda_wv_arguments, f)
        # time.sleep(3)
        # exit(0)
    attn_output = cuda_bmm_forward_wv_group_outlier_hq(group_size, nh_kv, attn_weights, sixteen_V, sixteen_V_idx,
                        eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean,
                        four_V, four_V_idx, four_V_scale_std, four_V_mn_mean,
                        two_V, two_V_idx, two_V_scale_std, two_V_mn_mean,
                        one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, new_V)
    
    return attn_weights, attn_output

def qkv_matmul_outlier(
        q: torch.Tensor, 
        past_key_value: Tuple,
        attention_mask: torch.Tensor,
        head_dim: int,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''计算q与量化的kv相乘'''
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value
    
    new_len = new_K.shape[2] if new_K is not None else 0
    nh_q = q.shape[1]
    nh_kv = 0
    if sixteen_K is not None:
        nh_kv = sixteen_K.shape[1]
    elif eight_K is not None:
        nh_kv = eight_K.shape[1]
    elif four_K is not None:
        nh_kv = four_K.shape[1]
    elif two_K is not None:
        nh_kv = two_K.shape[1]
    elif one_K is not None:
        nh_kv = one_K.shape[1]
    elif new_K is not None:
        nh_kv = new_K.shape[1]
    else:
        raise ValueError('invalid past_key_value')
    n_rep = nh_q // nh_kv
    attn_weights = torch.zeros(q.shape[0], q.shape[1], 1, kv_seq_len, dtype=q.dtype, device=q.device)
    attn_output = torch.zeros(q.shape[0], q.shape[1], 1, q.shape[3], dtype=q.dtype, device=q.device)

    if quant_strategy == 'uniform':
        K_cuda_fA_qB_high = K_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group
        V_cuda_fA_qB_high = V_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_outlier
    elif quant_strategy == 'normal':
        K_cuda_fA_qB_high = K_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
        V_cuda_fA_qB_high = V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_outlier
    elif quant_strategy == 'uniform_group':
        K_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        V_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group_outlier
        K_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group
        V_cuda_fA_qB_low = cuda_bmm_fA_qB_uniform_group_outlier
    elif quant_strategy == 'normal_group':
        K_cuda_fA_qB_high = cuda_bmm_fA_qB_normal_group
        V_cuda_fA_qB_high = cuda_bmm_fA_qB_normal_group_outlier
        K_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
        V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group_outlier
    elif quant_strategy == 'high_uniform_group_low_normal_group':
        K_cuda_fA_qB_high = cuda_bmm_fA_qB_uniform_group
        V_cuda_fA_qB_high = cuda_bmm_fA_qB_wv_uniform_group_outlier
        K_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group
        V_cuda_fA_qB_low = cuda_bmm_fA_qB_normal_group_outlier
    else:
        raise ValueError('invalid quant_strategy')

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
    attn_weights += attention_mask
    attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
    
    if sixteen_V is not None:
        sixteen_V_outliers = torch.scatter(sixteen_V, dim=-1, index=outliers_idx[:, :, sixteen_V_idx, :], src = outliers[:, :, sixteen_V_idx, :])
        sixteen_V_rep =  repeat_kv(sixteen_V_outliers, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, sixteen_V_idx], sixteen_V_rep)
    if eight_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, eight_V_idx], eight_V, eight_V_scale_std, eight_V_mn_mean, outliers[:, :, eight_V_idx, :], outliers_idx[:, :, eight_V_idx, :], 8)
    if four_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, four_V_idx], four_V, four_V_scale_std, four_V_mn_mean, outliers[:, :, four_V_idx, :], outliers_idx[:, :, four_V_idx, :], 4)
    if two_V is not None:
        attn_output += V_cuda_fA_qB_high(group_size, attn_weights[:, :, :, two_V_idx], two_V, two_V_scale_std, two_V_mn_mean, outliers[:, :, two_V_idx, :], outliers_idx[:, :, two_V_idx, :], 2)
    if one_V is not None:
        attn_output += V_cuda_fA_qB_low(group_size, attn_weights[:, :, :, one_V_idx], one_V, one_V_scale_std, one_V_mn_mean, outliers[:, :, one_V_idx, :], outliers_idx[:, :, one_V_idx, :], 1)
    if new_V is not None:
        new_V_rep = repeat_kv(new_V, n_rep)
        attn_output += torch.matmul(attn_weights[:, :, :, -new_len:], new_V_rep)
    
    return attn_weights, attn_output