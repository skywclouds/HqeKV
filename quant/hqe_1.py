from model.public_func import (calculate_token_norm, calculate_new_token_range,
    calculate_token_range, get_block_token_idx, last_attn_weights)
import torch
from typing import Callable
import json
from hqe_2 import (
    quant_new_K, K_high_2_eviction, V_high_2_eviction, 
    K_high_2_low, V_high_2_low)
from pack_unpack import (
    uniform_quantize_pack_last_dim, 
    uniform_dequantize_unpack_last_dim, 
    normal_dequantize_unpack_last_dim,
    normal_quantize_pack_last_dim,
    uniform_group_quantize_pack_last_dim,
    uniform_group_dequantize_unpack_last_dim,
    normal_group_quantize_pack_last_dim,
    normal_group_dequantize_unpack_last_dim)


def compress_K(sixteen_K, eight_K, eight_K_scale_std, eight_K_mn_mean,
               four_K, four_K_scale_std, four_K_mn_mean, two_K, two_K_scale_std, two_K_mn_mean,
               one_K, one_K_scale_std, one_K_mn_mean, block_size, 
               new_sixteen_K_idx, new_eight_K_idx, new_four_K_idx, new_two_K_idx, new_one_K_idx, new_e_K_idx, 
               new_K, new_token_idx, K_quantize_high, K_dequantize_high, K_quantize_low, K_dequantize_low):
    sixteen_num = sixteen_K.shape[-1] if sixteen_K is not None else 0
    eight_num = eight_K.shape[-1] * 4 if eight_K is not None else 0
    four_num = four_K.shape[-1] * 8 if four_K is not None else 0
    two_num = two_K.shape[-1] * 16 if two_K is not None else 0
    one_num = one_K.shape[-1] * 32 if one_K is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)
    sixteen_K_idx = eight_K_idx = four_K_idx = two_K_idx = one_K_idx = None
    if sixteen_num > 0:
        sixteen_K_idx = torch.arange(0, bit_nums[0], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if eight_num > 0:
        eight_K_idx = torch.arange(bit_nums[0], bit_nums[1], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if four_num > 0:
        four_K_idx = torch.arange(bit_nums[1], bit_nums[2], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if two_num > 0:
        two_K_idx = torch.arange(bit_nums[2], bit_nums[3], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if one_num > 0:
        one_K_idx = torch.arange(bit_nums[3], bit_nums[4], dtype=new_token_idx.dtype, device=new_token_idx.device)

    # 将各部分 K 应丢弃的丢弃
    if new_e_K_idx is not None:
        # 新生成的 K 应丢弃的丢弃
        new_2_e = torch.isin(new_token_idx, new_e_K_idx)
        if new_2_e.any():
            # 因为 new_K 是用token维度放置的，所以用 V 的丢弃函数
            new_K, new_token_idx, _, _ = V_high_2_eviction(new_K, new_token_idx, None, None, new_2_e)
        # 以前生成的16位 K 应丢弃的丢弃
        if sixteen_K is not None:
            sixteen_2_e = torch.isin(sixteen_K_idx, new_e_K_idx)
            if sixteen_2_e.any():
                sixteen_K, sixteen_K_idx, _, _ = K_high_2_eviction(sixteen_K, sixteen_K_idx, None, None, 16, sixteen_2_e, block_size)
        # 以前生成的8位 K 应丢弃的丢弃
        if eight_K_idx is not None:
            eight_2_e = torch.isin(eight_K_idx, new_e_K_idx)
            if eight_2_e.any():
                eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean = K_high_2_eviction(eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, eight_2_e, block_size)
        # 以前生成的4位 K 应丢弃的丢弃
        if four_K_idx is not None:
            four_2_e = torch.isin(four_K_idx, new_e_K_idx)
            if four_2_e.any():
                four_K, four_K_idx, four_K_scale_std, four_K_mn_mean = K_high_2_eviction(four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, four_2_e, block_size)
        # 以前生成的2位 K 应丢弃的丢弃
        if two_K_idx is not None:
            two_2_e = torch.isin(two_K_idx, new_e_K_idx)
            if two_2_e.any():
                two_K, two_K_idx, two_K_scale_std, two_K_mn_mean = K_high_2_eviction(two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, two_2_e, block_size)
        # 以前生成的1位 K 应丢弃的丢弃
        if one_K_idx is not None:
            one_2_e = torch.isin(one_K_idx, new_e_K_idx)
            if one_2_e.any():
                one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = K_high_2_eviction(one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, one_2_e, block_size)
    # 将各部分 K 应量化到1位的部分量化到1位
    if new_one_K_idx is not None:
        # 新生成的 K 应量化到1位的部分量化到1位
        new_2_one = torch.isin(new_token_idx, new_one_K_idx)
        if new_2_one.any():
            new_K, new_token_idx, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = quant_new_K(new_K, new_token_idx, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, new_2_one, K_quantize_low, block_size)
        # 以前生成的16位 K 应量化到1位的部分量化到1位
        if sixteen_K_idx is not None:
            sixteen_2_one = torch.isin(sixteen_K_idx, new_one_K_idx)
            if sixteen_2_one.any():
                sixteen_K, sixteen_K_idx, _, _, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = K_high_2_low(sixteen_K, sixteen_K_idx, None, None, 16, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, sixteen_2_one, K_quantize_low, None, block_size)
        # 以前生成的8位 K 应量化到1位的部分量化到1位
        if eight_K_idx is not None:
            eight_2_one = torch.isin(eight_K_idx, new_one_K_idx)
            if eight_2_one.any():
                eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = K_high_2_low(eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, eight_2_one, K_quantize_low, K_dequantize_high, block_size)
        # 以前生成的4位 K 应量化到1位的部分量化到1位
        if four_K_idx is not None:
            four_2_one = torch.isin(four_K_idx, new_one_K_idx)
            if four_2_one.any():
                four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = K_high_2_low(four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, four_2_one, K_quantize_low, K_dequantize_high, block_size)
        # 以前生成的2位 K 应量化到1位的部分量化到1位
        if two_K_idx is not None:
            two_2_one = torch.isin(two_K_idx, new_one_K_idx)
            if two_2_one.any():
                two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean = K_high_2_low(two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 1, two_2_one, K_quantize_low, K_dequantize_high, block_size)
    # 将各部分 K 应量化到2位的部分量化到2位
    if new_two_K_idx is not None:
        # 新生成的 K 应量化到2位的部分量化到2位
        new_2_two = torch.isin(new_token_idx, new_two_K_idx)
        if new_2_two.any():
            new_K, new_token_idx, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean = quant_new_K(new_K, new_token_idx, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, new_2_two, K_quantize_high, block_size)
        # 以前生成的16位 K 应量化到2位的部分量化到2位
        if sixteen_K_idx is not None:
            sixteen_2_two = torch.isin(sixteen_K_idx, new_two_K_idx)
            if sixteen_2_two.any():
                sixteen_K, sixteen_K_idx, _, _, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean = K_high_2_low(sixteen_K, sixteen_K_idx, None, None, 16, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, sixteen_2_two, K_quantize_high, None, block_size)
        # 以前生成的8位 K 应量化到2位的部分量化到2位
        if eight_K_idx is not None:
            eight_2_two = torch.isin(eight_K_idx, new_two_K_idx)
            if eight_2_two.any():
                eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean = K_high_2_low(eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, eight_2_two, K_quantize_high, K_dequantize_high, block_size)
        # 以前生成的4位 K 应量化到2位的部分量化到2位
        if four_K_idx is not None:
            four_2_two = torch.isin(four_K_idx, new_two_K_idx)
            if four_2_two.any():
                four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean = K_high_2_low(four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 2, four_2_two, K_quantize_high, K_dequantize_high, block_size)
    # 将各部分 K 应量化到4位的部分量化到4位
    if new_four_K_idx is not None:
        new_2_four = torch.isin(new_token_idx, new_four_K_idx)
        if new_2_four.any():
            new_K, new_token_idx, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean = quant_new_K(new_K, new_token_idx, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, new_2_four, K_quantize_high, block_size)
        # 以前生成的16位 K 应量化到4位的部分量化到4位
        if sixteen_K_idx is not None:
            sixteen_2_four = torch.isin(sixteen_K_idx, new_four_K_idx)
            if sixteen_2_four.any():
                sixteen_K, sixteen_K_idx, _, _, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean = K_high_2_low(sixteen_K, sixteen_K_idx, None, None, 16, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, sixteen_2_four, K_quantize_high, None, block_size)
        # 以前生成的8位 K 应量化到4位的部分量化到4位
        if eight_K_idx is not None:
            eight_2_four = torch.isin(eight_K_idx, new_four_K_idx)
            if eight_2_four.any():
                eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean = K_high_2_low(eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 4, eight_2_four, K_quantize_high, K_dequantize_high, block_size)
    # 将各部分 K 应量化到8位的部分量化到8位
    if new_eight_K_idx is not None:
        new_2_eight = torch.isin(new_token_idx, new_eight_K_idx)
        if new_2_eight.any():
            new_K, new_token_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean = quant_new_K(new_K, new_token_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, new_2_eight, K_quantize_high, block_size)
        # 以前生成的16位 K 应量化到8位的部分量化到8位
        if sixteen_K_idx is not None:
            sixteen_2_eight = torch.isin(sixteen_K_idx, new_eight_K_idx)
            if sixteen_2_eight.any():
                sixteen_K, sixteen_K_idx, _, _, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean = K_high_2_low(sixteen_K, sixteen_K_idx, None, None, 16, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 8, sixteen_2_eight, K_quantize_high, None, block_size)
    # 将各部分 K 应量化到16位的部分量化到16位
    if new_sixteen_K_idx is not None:
        new_2_sixteen = torch.isin(new_token_idx, new_sixteen_K_idx)
        if new_2_sixteen.any():
            _, _, sixteen_K, sixteen_K_idx, _, _, = quant_new_K(new_K, new_token_idx, sixteen_K, sixteen_K_idx, None, None, 16, new_2_sixteen, None, block_size)
    
    return (sixteen_K, eight_K, eight_K_scale_std, eight_K_mn_mean,
               four_K, four_K_scale_std, four_K_mn_mean, two_K, two_K_scale_std, two_K_mn_mean,
               one_K, one_K_scale_std, one_K_mn_mean, new_K)

def compress_V(sixteen_V, eight_V, eight_V_scale_std, eight_V_mn_mean,
               four_V, four_V_scale_std, four_V_mn_mean, two_V, two_V_scale_std, two_V_mn_mean,
               one_V, one_V_scale_std, one_V_mn_mean, group_size, 
               new_sixteen_V_idx, new_eight_V_idx, new_four_V_idx, new_two_V_idx, new_one_V_idx, new_e_V_idx, 
               new_V, new_token_idx, V_token_norm, V_quantize_high, V_dequantize_high, V_quantize_low, V_dequantize_low):
    sixteen_num = sixteen_V.shape[-2] if sixteen_V is not None else 0
    eight_num = eight_V.shape[-2] if eight_V is not None else 0
    four_num = four_V.shape[-2] if four_V is not None else 0
    two_num = two_V.shape[-2] if two_V is not None else 0
    one_num = one_V.shape[-2] if one_V is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)
    sixteen_V_idx = eight_V_idx = four_V_idx = two_V_idx = one_V_idx = None
    if sixteen_num > 0:
        sixteen_V_idx = torch.arange(0, bit_nums[0], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if eight_num > 0:
        eight_V_idx = torch.arange(bit_nums[0], bit_nums[1], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if four_num > 0:
        four_V_idx = torch.arange(bit_nums[1], bit_nums[2], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if two_num > 0:
        two_V_idx = torch.arange(bit_nums[2], bit_nums[3], dtype=new_token_idx.dtype, device=new_token_idx.device)
    if one_num > 0:
        one_V_idx = torch.arange(bit_nums[3], bit_nums[4], dtype=new_token_idx.dtype, device=new_token_idx.device)
    # 将各部分 V 应丢弃的丢弃
    if new_e_V_idx is not None:
        # 新生成的 V 应丢弃的丢弃
        new_2_e = torch.isin(new_token_idx, new_e_V_idx)
        if new_2_e.any():
            new_V, new_token_idx, _, _ = V_high_2_eviction(new_V, new_token_idx, None, None, new_2_e)
        # 以前生成的16位 V 应丢弃的丢弃
        if sixteen_V_idx is not None:
            sixteen_2_e = torch.isin(sixteen_V_idx, new_e_V_idx)
            if sixteen_2_e.any():
                sixteen_V, sixteen_V_idx, _, _ = V_high_2_eviction(sixteen_V, sixteen_V_idx, None, None, sixteen_2_e)
        # 以前生成的8位 V 应丢弃的丢弃
        if eight_V_idx is not None:
            eight_2_e = torch.isin(eight_V_idx, new_e_V_idx)
            if eight_2_e.any():
                eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean = V_high_2_eviction(eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, eight_2_e)
        # 以前生成的4位 V 应丢弃的丢弃
        if four_V_idx is not None:
            four_2_e = torch.isin(four_V_idx, new_e_V_idx)
            if four_2_e.any():
                four_V, four_V_idx, four_V_scale_std, four_V_mn_mean = V_high_2_eviction(four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, four_2_e)
        # 以前生成的2位 V 应丢弃的丢弃
        if two_V_idx is not None:
            two_2_e = torch.isin(two_V_idx, new_e_V_idx)
            if two_2_e.any():
                two_V, two_V_idx, two_V_scale_std, two_V_mn_mean = V_high_2_eviction(two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, two_2_e)
        # 以前生成的1位 V 应丢弃的丢弃
        if one_V_idx is not None:
            one_2_e = torch.isin(one_V_idx, new_e_V_idx)
            if one_2_e.any():
                one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_eviction(one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, one_2_e)
    # 各部分 V 应量化到1位的部分量化到1位
    if new_one_V_idx is not None:
        # 把新生成的 V 应量化到1位的部分量化到1位
        new_2_one = torch.isin(new_token_idx, new_one_V_idx)
        if new_2_one.any():
            new_V, new_token_idx, _, _, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(new_V, new_token_idx, None, None, 16, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, new_2_one, V_quantize_low, None, group_size)
        # 把16位的 V 应量化到1位的部分量化到1位
        if sixteen_V_idx is not None:
            sixteen_2_one = torch.isin(sixteen_V_idx, new_one_V_idx)
            if sixteen_2_one.any():
                sixteen_V, sixteen_V_idx, _, _, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(sixteen_V, sixteen_V_idx, None, None, 16, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, sixteen_2_one, V_quantize_low, None, group_size)
        # 把8位的 V 应量化到1位的部分量化到1位
        if eight_V_idx is not None:
            eight_2_one = torch.isin(eight_V_idx, new_one_V_idx)
            if eight_2_one.any():
                eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, 8, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, eight_2_one, V_quantize_low, V_dequantize_high, group_size)
        # 把4位的 V 应量化到1位的部分量化到1位
        if four_V_idx is not None:
            four_2_one = torch.isin(four_V_idx, new_one_V_idx)
            if four_2_one.any():
                four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, 4, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, four_2_one, V_quantize_low, V_dequantize_high, group_size)
        # 把2位的 V 应量化到1位的部分量化到1位
        if two_V_idx is not None:
            two_2_one = torch.isin(two_V_idx, new_one_V_idx)
            if two_2_one.any():
                two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, 2, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, two_2_one, V_quantize_low, V_dequantize_high, group_size)
    # 各部分 V 应量化到2位的部分量化到2位
    if new_two_V_idx is not None:
        # 把新生成的 V 应量化到2位的部分量化到2位
        if new_token_idx is not None:
            new_2_two = torch.isin(new_token_idx, new_two_V_idx)
            if new_2_two.any():
                new_V, new_token_idx, _, _, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean = V_high_2_low(new_V, new_token_idx, None, None, 16, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, 2, new_2_two, V_quantize_high, None, group_size)
        # 把16位的 V 应量化到2位的部分量化到2位
        if sixteen_V_idx is not None:
            sixteen_2_two = torch.isin(sixteen_V_idx, new_two_V_idx)
            if sixteen_2_two.any():
                sixteen_V, sixteen_V_idx, _, _, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean = V_high_2_low(sixteen_V, sixteen_V_idx, None, None, 16, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, 2, sixteen_2_two, V_quantize_high, None, group_size)
        # 把8位的 V 应量化到2位的部分量化到2位
        if eight_V_idx is not None:
            eight_2_two = torch.isin(eight_V_idx, new_two_V_idx)
            if eight_2_two.any():
                eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean = V_high_2_low(eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, 8, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, 2, eight_2_two, V_quantize_high, V_dequantize_high, group_size)
        # 把4位的 V 应量化到2位的部分量化到2位
        if four_V_idx is not None:
            four_2_two = torch.isin(four_V_idx, new_two_V_idx)
            if four_2_two.any():
                four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean = V_high_2_low(four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, 4, two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, 2, four_2_two, V_quantize_high, V_dequantize_high, group_size)
    # 各部分 V 应量化到4位的部分量化到4位
    if new_four_V_idx is not None:
        # 把新生成的 V 应量化到4位的部分量化到4位
        if new_token_idx is not None:
            new_2_four = torch.isin(new_token_idx, new_four_V_idx)
            if new_2_four.any():
                new_V, new_token_idx, _, _, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean = V_high_2_low(new_V, new_token_idx, None, None, 16, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, 4, new_2_four, V_quantize_high, None, group_size)
        # 把16位的 V 应量化到4位的部分量化到4位
        if sixteen_V_idx is not None:
            sixteen_2_four = torch.isin(sixteen_V_idx, new_four_V_idx)
            if sixteen_2_four.any():
                sixteen_V, sixteen_V_idx, _, _, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean = V_high_2_low(sixteen_V, sixteen_V_idx, None, None, 16, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, 4, sixteen_2_four, V_quantize_high, None, group_size)
        # 把8位的 V 应量化到4位的部分量化到4位
        if eight_V_idx is not None:
            eight_2_four = torch.isin(eight_V_idx, new_four_V_idx)
            if eight_2_four.any():
                eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean = V_high_2_low(eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, 8, four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, 4, eight_2_four, V_quantize_high, V_dequantize_high, group_size)
    # 各部分 V 应量化到8位的部分量化到8位
    if new_eight_V_idx is not None:
        # 把新生成的 V 应量化到8位的部分量化到8位
        if new_token_idx is not None:
            new_2_eight = torch.isin(new_token_idx, new_eight_V_idx)
            if new_2_eight.any():
                new_V, new_token_idx, _, _, eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean = V_high_2_low(new_V, new_token_idx, None, None, 16, eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, 8, new_2_eight, V_quantize_high, None, group_size)
        # 把16位的 V 应量化到8位的部分量化到8位
        if sixteen_V_idx is not None:
            sixteen_2_eight = torch.isin(sixteen_V_idx, new_eight_V_idx)
            if sixteen_2_eight.any():
                sixteen_V, sixteen_V_idx, _, _, eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean = V_high_2_low(sixteen_V, sixteen_V_idx, None, None, 16, eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, 8, sixteen_2_eight, V_quantize_high, None, group_size)
    # 各部分 V 应量化到16位的部分量化到16位
    if new_sixteen_V_idx is not None:
        # 把新生成的 V 应量化到16位的部分量化到16位
        if new_token_idx is not None:
            new_2_sixteen = torch.isin(new_token_idx, new_sixteen_V_idx)
            if new_2_sixteen.any():
                bsz, num_heads, _, num_channel = new_V.shape
                new_2_sixteen_V_idx = new_token_idx[new_2_sixteen]
                new_token_idx = new_token_idx[~new_2_sixteen]
                new_token_idx = None if new_token_idx.numel() == 0 else new_token_idx
                new_2_sixteen = new_2_sixteen.view(1, 1, -1, 1).expand_as(new_V)
                new_2_sixteen_V = new_V[new_2_sixteen].view(bsz, num_heads, -1, num_channel)
                sixteen_V = torch.cat((sixteen_V, new_2_sixteen_V), dim=2) if sixteen_V is not None else new_2_sixteen_V
                sixteen_V_idx = torch.cat((sixteen_V_idx, new_2_sixteen_V_idx)) if sixteen_V_idx is not None else new_2_sixteen_V_idx
        # 丢弃 V_token_norm 中应丢弃的部分, 并调整顺序
    reserve_V_idx = [sixteen_V_idx, eight_V_idx, four_V_idx, two_V_idx, one_V_idx, new_token_idx]
    reserve_V_idx = [elem for elem in reserve_V_idx if elem is not None]
    if len(reserve_V_idx) > 0:
        reserve_V_idx = torch.cat(reserve_V_idx)
        V_token_norm = V_token_norm[reserve_V_idx]
    
    return (sixteen_V, eight_V, eight_V_scale_std, eight_V_mn_mean,
               four_V, four_V_scale_std, four_V_mn_mean, two_V, two_V_scale_std, two_V_mn_mean,
               one_V, one_V_scale_std, one_V_mn_mean, new_V, V_token_norm)

def KV_cache_hqe_1(
        past_key_value: tuple, 
        compress_length: int, 
        history_length: int, 
        K_bit_num: torch.Tensor, 
        V_bit_num: torch.Tensor,
        token_importance: torch.Tensor,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''对KV cache进行量化和丢弃 K 沿channel维度 V 沿token维度'''
    # 选择量化函数
    if quant_strategy == 'uniform':
        K_quantize_high = K_quantize_low = uniform_group_quantize_pack_last_dim
        V_quantize_high = V_quantize_low = uniform_quantize_pack_last_dim
        K_dequantize_high = K_dequantize_low = uniform_group_dequantize_unpack_last_dim
        V_dequantize_high = V_dequantize_low = uniform_dequantize_unpack_last_dim
    elif quant_strategy == 'normal':
        K_quantize_high = K_quantize_low = normal_group_quantize_pack_last_dim
        V_quantize_high = V_quantize_low = normal_quantize_pack_last_dim
        K_dequantize_high = K_dequantize_low = normal_group_dequantize_unpack_last_dim
        V_dequantize_high = V_dequantize_low = normal_dequantize_unpack_last_dim
    elif quant_strategy == 'uniform_group':
        K_quantize_high = V_quantize_high = K_quantize_low = V_quantize_low = uniform_group_quantize_pack_last_dim
        K_dequantize_high = V_dequantize_high = K_dequantize_low = V_dequantize_low = uniform_group_dequantize_unpack_last_dim
    elif quant_strategy == 'normal_group':
        K_quantize_high = V_quantize_high = K_quantize_low = V_quantize_low = normal_group_quantize_pack_last_dim
        K_dequantize_high = V_dequantize_high = K_dequantize_low = V_dequantize_low = normal_group_dequantize_unpack_last_dim
    elif quant_strategy == 'high_uniform_group_low_normal_group':
        K_quantize_high = V_quantize_high = uniform_group_quantize_pack_last_dim
        K_quantize_low = V_quantize_low = normal_group_quantize_pack_last_dim
        K_dequantize_high = V_dequantize_high = uniform_group_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = normal_group_dequantize_unpack_last_dim
    else:
        raise ValueError('invalid quant_strategy')
    # 取出各个项
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value
    # 不能被32整除的部分不作处理
    residual_new_K = new_K[:, :, compress_length:, :] if new_K is not None else None
    residual_new_V = new_V[:, :, compress_length:, :] if new_V is not None else None
    if residual_new_K is not None:
        residual_new_K = None if residual_new_K.numel() == 0 else residual_new_K
        residual_new_V = None if residual_new_V.numel() == 0 else residual_new_V
    new_K = new_K[:, :, :compress_length, :]
    new_V = new_V[:, :, :compress_length, :]

    total_length = history_length + compress_length
    block_num = total_length // block_size
    
    V_token_norm = calculate_token_range(new_V, V_token_norm)

    V_token_norm_importance = token_importance * V_token_norm
    # V_token_norm_importance = V_token_norm
    K_sorted_idx = get_block_token_idx(block_size, block_num, V_token_norm_importance)
    V_sorted_idx = K_sorted_idx.clone()
    
    # 对各个位数量化的channel和token的位数的个数做预处理
    K_bit_num = torch.cumsum(K_bit_num, dim = 0)
    V_bit_num = torch.cumsum(V_bit_num, dim = 0)

    # 统计 1 -> 2, 2 -> 4, 4 ->8, 8 -> 16 的个数
    # jsonl_path = '/home/users/wanghe/projects/KIVI-main/tests/low_2_high.jsonl'
    # low_2_high_dict = {
    #     'K' : {'1 -> 2' : 0, '2 -> 4' : 0, '4 -> 8' : 0, '8 -> 16' : 0},
    #     'V' : {'1 -> 2' : 0, '2 -> 4' : 0, '4 -> 8' : 0, '8 -> 16' : 0},
    #     'compress_length' : compress_length
    #     }
    
    new_sixteen_K_idx = K_sorted_idx[0:K_bit_num[0]] if K_bit_num[0] > 0 else None
    new_eight_K_idx = K_sorted_idx[K_bit_num[0]:K_bit_num[1]] if K_bit_num[1] > K_bit_num[0] else None
    new_four_K_idx = K_sorted_idx[K_bit_num[1]:K_bit_num[2]] if K_bit_num[2] > K_bit_num[1] else None
    new_two_K_idx = K_sorted_idx[K_bit_num[2]:K_bit_num[3]] if K_bit_num[3] > K_bit_num[2] else None
    new_one_K_idx = K_sorted_idx[K_bit_num[3]:K_bit_num[4]] if K_bit_num[4] > K_bit_num[3] else None
    new_e_K_idx = K_sorted_idx[K_bit_num[4]:K_bit_num[5]] if K_bit_num[5] > K_bit_num[4] else None
    new_token_idx = torch.arange(history_length, history_length + compress_length, device=V_sorted_idx.device, dtype=V_sorted_idx.dtype)

    (sixteen_K, eight_K, eight_K_scale_std, eight_K_mn_mean,
        four_K, four_K_scale_std, four_K_mn_mean, two_K, two_K_scale_std, two_K_mn_mean,
        one_K, one_K_scale_std, one_K_mn_mean, new_K) = compress_K(sixteen_K, eight_K, eight_K_scale_std, eight_K_mn_mean,
               four_K, four_K_scale_std, four_K_mn_mean, two_K, two_K_scale_std, two_K_mn_mean,
               one_K, one_K_scale_std, one_K_mn_mean, block_size, 
               new_sixteen_K_idx, new_eight_K_idx, new_four_K_idx, new_two_K_idx, new_one_K_idx, new_e_K_idx, 
               new_K, new_token_idx, K_quantize_high, K_dequantize_high, K_quantize_low, K_dequantize_low)
    
    # 提取各个位数量化的 V 的 token
    new_sixteen_V_idx = V_sorted_idx[0:V_bit_num[0]] if V_bit_num[0] > 0 else None
    new_eight_V_idx = V_sorted_idx[V_bit_num[0]:V_bit_num[1]] if V_bit_num[1] > V_bit_num[0] else None
    new_four_V_idx = V_sorted_idx[V_bit_num[1]:V_bit_num[2]] if V_bit_num[2] > V_bit_num[1] else None
    new_two_V_idx = V_sorted_idx[V_bit_num[2]:V_bit_num[3]] if V_bit_num[3] > V_bit_num[2] else None
    new_one_V_idx = V_sorted_idx[V_bit_num[3]:V_bit_num[4]] if V_bit_num[4] > V_bit_num[3] else None
    new_e_V_idx = V_sorted_idx[V_bit_num[4]:V_bit_num[5]] if V_bit_num[5] > V_bit_num[4] else None
    new_token_idx = torch.arange(history_length, history_length + compress_length, device=V_sorted_idx.device, dtype=V_sorted_idx.dtype)

    (sixteen_V, eight_V, eight_V_scale_std, eight_V_mn_mean,
        four_V, four_V_scale_std, four_V_mn_mean, two_V, two_V_scale_std, two_V_mn_mean,
        one_V, one_V_scale_std, one_V_mn_mean, new_V, V_token_norm) = compress_V(sixteen_V, eight_V, eight_V_scale_std, eight_V_mn_mean,
               four_V, four_V_scale_std, four_V_mn_mean, two_V, two_V_scale_std, two_V_mn_mean,
               one_V, one_V_scale_std, one_V_mn_mean, group_size, 
               new_sixteen_V_idx, new_eight_V_idx, new_four_V_idx, new_two_V_idx, new_one_V_idx, new_e_V_idx, 
               new_V, new_token_idx, V_token_norm, V_quantize_high, V_dequantize_high, V_quantize_low, V_dequantize_low)
    
    # 重新把各个部分打包
    past_key_value = (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
                    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
                    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
                    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
                    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, residual_new_K, residual_new_V, kv_seq_len)
    return past_key_value

def KV_cache_reconstruct_hqe(
        past_key_value: tuple, 
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''对压缩后的KV cache进行重建'''
    # 选择反量化函数
    if quant_strategy == 'uniform':
        K_dequantize_high = V_dequantize_high = uniform_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = uniform_dequantize_unpack_last_dim
    elif quant_strategy == 'normal':
        K_dequantize_high = V_dequantize_high = normal_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = normal_dequantize_unpack_last_dim
    elif quant_strategy == 'uniform_group':
        K_dequantize_high = V_dequantize_high = uniform_group_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = uniform_group_dequantize_unpack_last_dim
    elif quant_strategy == 'normal_group':
        K_dequantize_high = V_dequantize_high = normal_group_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = normal_group_dequantize_unpack_last_dim
    elif quant_strategy == 'high_uniform_group_low_normal_group':
        K_dequantize_high = V_dequantize_high = uniform_group_dequantize_unpack_last_dim
        K_dequantize_low = V_dequantize_low = normal_group_dequantize_unpack_last_dim
    else:
        raise ValueError('invalid quant_strategy')
    
    (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, new_K, new_V, kv_seq_len) = past_key_value

    real_len = 0
    dtype = torch.float16
    sixteen_num = sixteen_V.shape[-2] if sixteen_V is not None else 0
    eight_num = eight_V.shape[-2] if eight_V is not None else 0
    four_num = four_V.shape[-2] if four_V is not None else 0
    two_num = two_V.shape[-2] if two_V is not None else 0
    one_num = one_V.shape[-2] if one_V is not None else 0
    bit_nums = torch.tensor([sixteen_num, eight_num, four_num, two_num, one_num])
    bit_nums = torch.cumsum(bit_nums, dim=0)
    # 获取batch_size, num_heads, head_dim, dtype, device
    if sixteen_V is not None:
        batch_size, num_heads, sixteen_len, head_dim = sixteen_V.shape
        real_len += sixteen_len
        device = sixteen_V.device
    if eight_V is not None:
        batch_size = eight_V.shape[0]
        num_heads = eight_V.shape[1]
        real_len += eight_V.shape[2]
        head_dim = eight_V.shape[3] * (32 // 8)
        device = eight_V.device
    if four_V is not None:
        batch_size = four_V.shape[0]
        num_heads = four_V.shape[1]
        real_len += four_V.shape[2]
        head_dim = four_V.shape[3] * (32 // 4)
        device = four_V.device
    if two_V is not None:
        batch_size = two_V.shape[0]
        num_heads = two_V.shape[1]
        real_len += two_V.shape[2]
        head_dim = two_V.shape[3] * (32 // 2)
        device = two_V.device
    if one_V is not None:
        batch_size = one_V.shape[0]
        num_heads = one_V.shape[1]
        real_len += one_V.shape[2]
        head_dim = one_V.shape[3] * (32 // 1)
        device = one_V.device
    if new_V is not None:
        batch_size, num_heads, new_len, head_dim = new_V.shape
        real_len += new_len
        device = new_V.device
    if real_len == 0:
        raise ValueError('invalid past_key_value')
    
    # 重建的 K, V
    reconstuct_K = torch.zeros(batch_size, num_heads, real_len, head_dim, dtype=dtype, device=device)
    reconstuct_V = torch.zeros(batch_size, num_heads, real_len, head_dim, dtype=dtype, device=device)

    # 把 16 bit 的 K, V 放回对应的位置
    if sixteen_K is not None:
        reconstuct_K[:, :, 0:bit_nums[0], :] = sixteen_K.transpose(2, 3)
    if sixteen_V is not None:
        reconstuct_V[:, :, 0:bit_nums[0], :] = sixteen_V

    # 把 8 bit 的 K, V 放回对应的位置
    if eight_K is not None:
        reconstuct_K_eight = K_dequantize_high(eight_K, eight_K_scale_std, eight_K_mn_mean, block_size, 8).transpose(2, 3)
        reconstuct_K[:, :, bit_nums[0]:bit_nums[1], :] = reconstuct_K_eight
    if eight_V is not None:
        reconstuct_V[:, :, bit_nums[0]:bit_nums[1], :] = V_dequantize_high(eight_V, eight_V_scale_std, eight_V_mn_mean, group_size, 8)

    # 把 4 bit 的 K, V 放回对应的位置
    if four_K is not None:
        reconstuct_K_four = K_dequantize_high(four_K, four_K_scale_std, four_K_mn_mean, block_size, 4).transpose(2, 3)
        reconstuct_K[:, :, bit_nums[1]:bit_nums[2], :] = reconstuct_K_four
    if eight_V is not None:
        reconstuct_V_eight = V_dequantize_high(four_V, four_V_scale_std, four_V_mn_mean, group_size, 4)
        reconstuct_V[:, :, bit_nums[1]:bit_nums[2], :] = reconstuct_V_eight
    
    # 把 2 bit 的 K, V 放回对应的位置
    if two_K is not None:
        reconstuct_K_two = K_dequantize_high(two_K, two_K_scale_std, two_K_mn_mean, block_size, 2).transpose(2, 3)
        reconstuct_K[:, :, bit_nums[2]:bit_nums[3], :] = reconstuct_K_two
    if two_V is not None:
        reconstuct_V_two = V_dequantize_high(two_V, two_V_scale_std, two_V_mn_mean, group_size, 2)
        reconstuct_V[:, :, bit_nums[2]:bit_nums[3], :] = reconstuct_V_two

    # 把 1 bit 的 K, V 放回对应的位置
    if one_K is not None:
        reconstuct_K_one = K_dequantize_low(one_K, one_K_scale_std, one_K_mn_mean, block_size, 1).transpose(2, 3)
        reconstuct_K[:, :, bit_nums[3]:bit_nums[4], :] =reconstuct_K_one
    if one_V is not None:
        reconstuct_V_one = V_dequantize_low(one_V, one_V_scale_std, one_V_mn_mean, group_size, 1)
        reconstuct_V[:, :, bit_nums[3]:bit_nums[4], :] = reconstuct_V_one
    
    # 把 新的 bit 的 K, V 放回对应的位置
    if new_K is not None:
        reconstuct_K[:, :, -new_len:, :] = new_K
    if new_V is not None:
        reconstuct_V[:, :, -new_len:, :] = new_V
    
    return reconstuct_K, reconstuct_V

if __name__ == '__main__':
    quant_strategy = 'uniform_group'
    dtype = torch.float16
    device = torch.device('cuda:3')
    histroy_length = 0
    seq_len = 1026
    compress_length = 1024
    last_weights = last_attn_weights()
    new_K = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)
    new_V = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)

    past_key_value = (None,) * 39 + (new_K, new_V, seq_len)
    token_importance = torch.rand(histroy_length + compress_length, device=device)
    block_size = 32
    block_importance = token_importance.clone()
    block_importance = block_importance.reshape(-1, block_size)
    block_importance /= block_importance.shape[1]
    block_importance = torch.sum(block_importance, dim=-1)
    _, block_idx = torch.sort(block_importance, descending=True)
    _, token_idx = torch.sort(token_importance, descending=True)
    block_idx *= block_size
    block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
    add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
    block_idx += add
    block_idx = block_idx.flatten()
    K_bit_num = torch.tensor([864, 32, 32, 32, 32, 32], dtype=torch.int32, device=device)
    V_bit_num = torch.tensor([864, 32, 32, 32, 32, 32], dtype=torch.int32, device=device)

    past_key_value = KV_cache_hqe_1(past_key_value, compress_length, histroy_length, block_idx, K_bit_num, V_bit_num, token_importance, quant_strategy, last_weights)

    new_seq_len = 1024
    key_states = torch.randn(1, 8, new_seq_len, 128, dtype=dtype, device=device)
    value_states = torch.randn(1, 8, new_seq_len, 128, dtype=dtype, device=device)
    new_K = torch.cat((past_key_value[-3], key_states), dim=2) if past_key_value[-3] is not None else key_states
    new_V = torch.cat((past_key_value[-2], value_states), dim=2) if past_key_value[-2]is not None else value_states
    seq_len += new_seq_len
    past_key_value = past_key_value[:-3] + (new_K, new_V, seq_len)
    histroy_length = histroy_length + compress_length - 32
    compress_length = new_seq_len

    token_importance = torch.rand(histroy_length + compress_length, device=device)
    block_size = 32
    block_importance = token_importance.clone()
    block_importance = block_importance.reshape(-1, block_size)
    block_importance /= block_importance.shape[1]
    block_importance = torch.sum(block_importance, dim=-1)
    _, block_idx = torch.sort(block_importance, descending=True)
    _, token_idx = torch.sort(token_importance, descending=True)
    block_idx *= block_size
    block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
    add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
    block_idx += add
    block_idx = block_idx.flatten()
    K_bit_num = torch.tensor([64, 0, 0, 384, 1248, 320], dtype=torch.int32, device=device)
    V_bit_num = torch.tensor([64, 0, 0, 384, 1248, 320], dtype=torch.int32, device=device)

    past_key_value = KV_cache_hqe_1(past_key_value, compress_length, histroy_length, block_idx, K_bit_num, V_bit_num, token_importance, quant_strategy, last_weights)

    new_seq_len = 1024
    key_states = torch.randn(1, 8, new_seq_len, 128, dtype=dtype, device=device)
    value_states = torch.randn(1, 8, new_seq_len, 128, dtype=dtype, device=device)
    new_K = torch.cat((past_key_value[-3], key_states), dim=2) if past_key_value[-3] is not None else key_states
    new_V = torch.cat((past_key_value[-2], value_states), dim=2) if past_key_value[-2]is not None else value_states
    seq_len += new_seq_len
    past_key_value = past_key_value[:-3] + (new_K, new_V, seq_len)
    histroy_length = histroy_length + compress_length - 320
    compress_length = new_seq_len

    token_importance = torch.rand(histroy_length + compress_length, device=device)
    block_size = 32
    block_importance = token_importance.clone()
    block_importance = block_importance.reshape(-1, block_size)
    block_importance /= block_importance.shape[1]
    block_importance = torch.sum(block_importance, dim=-1)
    _, block_idx = torch.sort(block_importance, descending=True)
    _, token_idx = torch.sort(token_importance, descending=True)
    block_idx *= block_size
    block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
    add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
    block_idx += add
    block_idx = block_idx.flatten()
    K_bit_num = torch.tensor([64, 0, 0, 384, 1952, 320], dtype=torch.int32, device=device)
    V_bit_num = torch.tensor([64, 0, 0, 384, 1952, 320], dtype=torch.int32, device=device)

    past_key_value = KV_cache_hqe_1(past_key_value, compress_length, histroy_length, block_idx, K_bit_num, V_bit_num, token_importance, quant_strategy, last_weights)
    reconstruct_K, reconstruct_V = KV_cache_reconstruct_hqe(past_key_value, quant_strategy)
    print(reconstruct_K.shape, reconstruct_V.shape)
