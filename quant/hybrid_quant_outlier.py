from model.public_func import calculate_token_norm, calculate_token_range, get_block_token_idx
import torch
from typing import Callable
import json
from pack_unpack import (
    uniform_quantize_pack_last_dim, 
    uniform_dequantize_unpack_last_dim, 
    normal_dequantize_unpack_last_dim,
    normal_quantize_pack_last_dim,
    uniform_group_quantize_pack_last_dim,
    uniform_group_dequantize_unpack_last_dim,
    normal_group_quantize_pack_last_dim,
    normal_group_dequantize_unpack_last_dim)

def insert_V_outliers(V: torch.Tensor,
                      V_idx: torch.Tensor,
                      outliers: torch.Tensor,
                      outliers_idx: torch.Tensor,
                      ):
    index = outliers_idx[:, :, V_idx, :]
    src = outliers[:, :, V_idx, :]
    V.scatter_(dim=-1, index=index, src=src)

def quant_new_K(new_K: torch.Tensor,
                new_K_idx: torch.Tensor,
                quantized_K: torch.Tensor,
                quantized_K_idx: torch.Tensor, 
                quantized_K_scale_std: torch.Tensor,
                quantized_K_mn_mean: torch.Tensor,
                bit: int,
                mask: torch.Tensor, 
                quantize: Callable, 
                block_size: int = 32):
    '''把新生成的 K 量化到指定位数并和已经量化的 K 拼接起来'''
    bsz, num_heads, _, last_dim = new_K.shape
    new_2_quant_idx = new_K_idx[mask]
    new_K_idx = new_K_idx[~mask]
    quantized_K_idx = torch.cat((quantized_K_idx, new_2_quant_idx)) if quantized_K_idx is not None else new_2_quant_idx
    mask = mask.view(1, 1, -1, 1).repeat(bsz, num_heads, 1, last_dim)
    new_2_quant = new_K[mask].view(bsz, num_heads, -1, last_dim).transpose(2, 3)
    new_K = new_K[~mask].view(bsz, num_heads, -1, last_dim)
    if bit < 16:
        new_2_quant, new_2_quant_scale_std, new_2_quant_mn_mean = quantize(new_2_quant, block_size, bit)
        quantized_K = torch.cat((quantized_K, new_2_quant), dim=3) if quantized_K is not None else new_2_quant
        quantized_K_scale_std = torch.cat((quantized_K_scale_std, new_2_quant_scale_std), dim=3) if quantized_K_scale_std is not None else new_2_quant_scale_std
        quantized_K_mn_mean = torch.cat((quantized_K_mn_mean, new_2_quant_mn_mean), dim=3) if quantized_K_mn_mean is not None else new_2_quant_mn_mean
    else:
        quantized_K = torch.cat((quantized_K, new_2_quant), dim=3) if quantized_K is not None else new_2_quant

    return new_K, new_K_idx, quantized_K, quantized_K_idx, quantized_K_scale_std, quantized_K_mn_mean

def K_high_2_low(high: torch.Tensor, high_idx: torch.Tensor, 
               high_scale_std: torch.Tensor | None, high_mn_mean: torch.Tensor | None, 
               high_bit: int,
               low: torch.Tensor, low_idx: torch.Tensor, 
               low_scale_std: torch.Tensor, low_mn_mean: torch.Tensor, 
               low_bit: int, 
               mask: torch.Tensor, 
               quantize: Callable, dequantize: Callable,
               block_size: int = 32):
    '''把高位的部分量化到低位的部分'''
    
    bsz, num_heads, head_dim, _ = high.shape
    high_2_low_idx = high_idx[mask]
    high_idx = high_idx[~mask]
    mask = mask.view(-1, block_size)
    if high_scale_std is not None:
        scale_std_mn_mean_mask = mask.clone()
        scale_std_mn_mean_mask = scale_std_mn_mean_mask[:, 0].contiguous()
        scale_std_mn_mean_mask = scale_std_mn_mean_mask.view(1, 1, 1, -1).repeat(bsz, num_heads, head_dim, 1)
        high_2_low_scale_std = high_scale_std[scale_std_mn_mean_mask].view(bsz, num_heads, head_dim, -1)
        high_2_low_mn_mean = high_mn_mean[scale_std_mn_mean_mask].view(bsz, num_heads, head_dim, -1)
        high_scale_std = high_scale_std[~scale_std_mn_mean_mask].view(bsz, num_heads, head_dim, -1)
        high_mn_mean = high_mn_mean[~scale_std_mn_mean_mask].view(bsz, num_heads, head_dim, -1)
    if high_bit < 16:
        pack_factor = 32 // high_bit
        mask = mask[:, :block_size//pack_factor].contiguous()
    mask = mask.view(1, 1, 1, -1).repeat(bsz, num_heads, head_dim, 1)
    high_2_low = high[mask].view(bsz, num_heads, head_dim, -1)
    high = high[~mask].view(bsz, num_heads, head_dim, -1)
    if high_bit < 16:
        high_2_low = dequantize(high_2_low, high_2_low_scale_std, high_2_low_mn_mean, block_size, high_bit)
    high_2_low, high_2_low_scale_std, high_2_low_mn_mean = quantize(high_2_low, block_size, low_bit)
    low = torch.cat((low, high_2_low), dim=3) if low is not None else high_2_low
    low_scale_std = torch.cat((low_scale_std, high_2_low_scale_std), dim=3) if low_scale_std is not None else high_2_low_scale_std
    low_mn_mean = torch.cat((low_mn_mean, high_2_low_mn_mean), dim=3) if low_mn_mean is not None else high_2_low_mn_mean
    low_idx = torch.cat((low_idx, high_2_low_idx)) if low_idx is not None else high_2_low_idx
    # 如果high全部转移到了low, 就把high置为None
    high = None if high.numel() == 0 else high
    high_idx = None if high_idx.numel() == 0 else high_idx
    if high_scale_std is not None:
        high_scale_std = None if high_scale_std.numel() == 0 else high_scale_std
        high_mn_mean = None if high_mn_mean.numel() == 0 else high_mn_mean
    return high, high_idx, high_scale_std, high_mn_mean, low, low_idx, low_scale_std, low_mn_mean

def V_high_2_low(high: torch.Tensor, high_idx: torch.Tensor, 
               high_scale_std: torch.Tensor | None, high_mn_mean: torch.Tensor | None, 
               high_bit: int,
               low: torch.Tensor, low_idx: torch.Tensor, 
               low_scale_std: torch.Tensor, low_mn_mean: torch.Tensor, 
               low_bit: int, 
               mask: torch.Tensor, 
               quantize: Callable, dequantize: Callable,
               group_size: int = 32):
    '''把高位的部分量化到低位的部分'''
    
    bsz, num_heads, _, last_dim = high.shape
    high_2_low_idx = high_idx[mask]
    high_idx = high_idx[~mask]
    mask = mask.view(1, 1, -1, 1).repeat(bsz, num_heads, 1, 1)
    if high_scale_std is not None:
        num_group = high_scale_std.shape[-1]
        mask = mask.repeat(1, 1, 1, num_group)
        high_2_low_scale_std = high_scale_std[mask].view(bsz, num_heads, -1, num_group)
        high_2_low_mn_mean = high_mn_mean[mask].view(bsz, num_heads, -1, num_group)
        high_scale_std = high_scale_std[~mask].view(bsz, num_heads, -1, num_group)
        high_mn_mean = high_mn_mean[~mask].view(bsz, num_heads, -1, num_group)
    mask = mask.repeat(1, 1, 1, last_dim // mask.shape[-1])
    high_2_low = high[mask].view(bsz, num_heads, -1, last_dim)
    high = high[~mask].view(bsz, num_heads, -1, last_dim)
    if high_bit < 16:
        high_2_low = dequantize(high_2_low, high_2_low_scale_std, high_2_low_mn_mean, group_size, high_bit)
    high_2_low, high_2_low_scale_std, high_2_low_mn_mean = quantize(high_2_low, group_size, low_bit)
    low = torch.cat((low, high_2_low), dim=2) if low is not None else high_2_low
    low_scale_std = torch.cat((low_scale_std, high_2_low_scale_std), dim=2) if low_scale_std is not None else high_2_low_scale_std
    low_mn_mean = torch.cat((low_mn_mean, high_2_low_mn_mean), dim=2) if low_mn_mean is not None else high_2_low_mn_mean
    low_idx = torch.cat((low_idx, high_2_low_idx)) if low_idx is not None else high_2_low_idx
    # 如果high全部转移到了low, 就把high置为None
    high = None if high.numel() == 0 else high
    high_idx = None if high_idx.numel() == 0 else high_idx
    if high_scale_std is not None:
        high_scale_std = None if high_scale_std.numel() == 0 else high_scale_std
        high_mn_mean = None if high_mn_mean.numel() == 0 else high_mn_mean
    return high, high_idx, high_scale_std, high_mn_mean, low, low_idx, low_scale_std, low_mn_mean

def KV_cache_compress_outlier(
        past_key_value: tuple, 
        compress_length: int, 
        history_length: int, 
        K_sorted_idx: torch.Tensor, 
        K_bit_num: torch.Tensor, 
        V_bit_num: torch.Tensor,
        token_importance: torch.Tensor,
        quant_strategy: str,
        block_size: int = 32,
        group_size: int = 32
        ):
    '''对KV cache进行量化'''
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
    V_sorted_idx = get_block_token_idx(block_size, block_num, V_token_norm_importance)
    # 提取 new_V 的 每个 token 的离群值
    new_V_outliers = torch.zeros(new_V.shape[0], new_V.shape[1], new_V.shape[2], 2, dtype=new_V.dtype, device=new_V.device)
    new_V_outliers_idx = torch.zeros(new_V.shape[0], new_V.shape[1], new_V.shape[2], 2, dtype=torch.int64, device=new_V.device)
    new_V_median, _ = torch.median(new_V, dim=-1, keepdim=True)
    new_V_up_outliers, new_V_up_outliers_idx = torch.topk(new_V, k=1, dim=-1)
    new_V_low_outliers, new_V_low_outliers_idx = torch.topk(new_V, k=1, dim=-1, largest=False)
    new_V_outliers[:, :, :, 0] = new_V_up_outliers[:, :, :, 0]
    new_V_outliers[:, :, :, 1] = new_V_low_outliers[:, :, :, 0]
    new_V_outliers_idx[:, :, :, 0] = new_V_up_outliers_idx[:, :, :, 0]
    new_V_outliers_idx[:, :, :, 1] = new_V_low_outliers_idx[:, :, :, 0]
    new_V.scatter_(dim=-1, index=new_V_outliers_idx, src=new_V_median.expand_as(new_V_outliers_idx))
    # 把离群值和索引拼接到以往的离群值和索引
    outliers = torch.cat((outliers, new_V_outliers), dim=2) if outliers is not None else new_V_outliers
    outliers_idx = torch.cat((outliers_idx, new_V_outliers_idx), dim=2) if outliers_idx is not None else new_V_outliers_idx
    
    # V_token_norm = calculate_token_range(new_V, V_token_norm)
    # V_token_norm_importance = token_importance * V_token_norm
    # V_sorted_idx = get_block_token_idx(block_size, block_num, V_token_norm_importance)
    # 对各个位数量化的channel和token的位数的个数做预处理
    K_bit_num = torch.cumsum(K_bit_num, dim = 0)
    V_bit_num = torch.cumsum(V_bit_num, dim = 0)
    
    # 统计 1 -> 2, 2 -> 4, 4 ->8, 8 -> 16 的个数
    # jsonl_path = 'low_2_high.jsonl'
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
    new_token_idx = torch.arange(history_length, history_length + compress_length, device=V_sorted_idx.device, dtype=V_sorted_idx.dtype)

    # 统计 K 中 1 -> 2, 2 -> 4, 4 ->8, 8 -> 16 的个数
    # if new_two_K_idx is not None and one_K_idx is not None:
    #     one_2_two = torch.isin(one_K_idx, new_two_K_idx)
    #     low_2_high_dict['K']['1 -> 2'] = one_2_two.sum().item()
    # if new_four_K_idx is not None and two_K_idx is not None:
    #     two_2_four = torch.isin(two_K_idx, new_four_K_idx)
    #     low_2_high_dict['K']['2 -> 4'] = two_2_four.sum().item()
    # if new_eight_K_idx is not None and four_K_idx is not None:
    #     four_2_eight = torch.isin(four_K_idx, new_eight_K_idx)
    #     low_2_high_dict['K']['4 -> 8'] = four_2_eight.sum().item()
    # if new_sixteen_K_idx is not None and eight_K_idx is not None:
    #     eight_2_sixteen = torch.isin(eight_K_idx, new_sixteen_K_idx)
    #     low_2_high_dict['K']['8 -> 16'] = eight_2_sixteen.sum().item()

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
    
    # 提取各个位数量化的 V 的 token
    new_sixteen_V_idx = V_sorted_idx[0:V_bit_num[0]] if V_bit_num[0] > 0 else None
    new_eight_V_idx = V_sorted_idx[V_bit_num[0]:V_bit_num[1]] if V_bit_num[1] > V_bit_num[0] else None
    new_four_V_idx = V_sorted_idx[V_bit_num[1]:V_bit_num[2]] if V_bit_num[2] > V_bit_num[1] else None
    new_two_V_idx = V_sorted_idx[V_bit_num[2]:V_bit_num[3]] if V_bit_num[3] > V_bit_num[2] else None
    new_one_V_idx = V_sorted_idx[V_bit_num[3]:V_bit_num[4]] if V_bit_num[4] > V_bit_num[3] else None
    new_token_idx = torch.arange(history_length, history_length + compress_length, device=V_sorted_idx.device, dtype=V_sorted_idx.dtype)

    # 统计 V 中 1 -> 2, 2 -> 4, 4 ->8, 8 -> 16 的个数
    # if new_two_V_idx is not None and one_V_idx is not None:
    #     one_2_two = torch.isin(one_V_idx, new_two_V_idx)
    #     low_2_high_dict['V']['1 -> 2'] = one_2_two.sum().item()
    # if new_four_V_idx is not None and two_V_idx is not None:
    #     two_2_four = torch.isin(two_V_idx, new_four_V_idx)
    #     low_2_high_dict['V']['2 -> 4'] = two_2_four.sum().item()
    # if new_eight_V_idx is not None and four_V_idx is not None:
    #     four_2_eight = torch.isin(four_V_idx, new_eight_V_idx)
    #     low_2_high_dict['V']['4 -> 8'] = four_2_eight.sum().item()
    # if new_sixteen_V_idx is not None and eight_V_idx is not None:
    #     eight_2_sixteen = torch.isin(eight_V_idx, new_sixteen_V_idx)
    #     low_2_high_dict['V']['8 -> 16'] = eight_2_sixteen.sum().item()
    
    # 把量化比例写入jsonl文件
    # with open(jsonl_path, "a", encoding="utf-8") as f:
    #     new_line = json.dumps(low_2_high_dict) + "\n"
    #     f.write(new_line)

    # 各部分 V 应量化到1位的部分量化到1位
    if new_one_V_idx is not None:
        # 把新生成的 V 应量化到1位的部分量化到1位
        new_2_one = torch.isin(new_token_idx, new_one_V_idx)
        if new_2_one.any():
            new_V, new_token_idx, _, _, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean = V_high_2_low(new_V, new_token_idx, None, None, 16, one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, 1, new_2_one, V_quantize_low, None, group_size)
            #print(new_token_idx.shape)
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
                new_2_sixteen = new_2_sixteen.view(1, 1, -1, 1).expand_as(new_V)
                new_2_sixteen_V = new_V[new_2_sixteen].view(bsz, num_heads, -1, num_channel)
                sixteen_V = torch.cat((sixteen_V, new_2_sixteen_V), dim=2) if sixteen_V is not None else new_2_sixteen_V
                sixteen_V_idx = torch.cat((sixteen_V_idx, new_2_sixteen_V_idx)) if sixteen_V_idx is not None else new_2_sixteen_V_idx
    # 重新把各个部分打包
    past_key_value = (sixteen_K, sixteen_K_idx, sixteen_V, sixteen_V_idx, eight_K, eight_K_idx, eight_K_scale_std, eight_K_mn_mean, 
                    eight_V, eight_V_idx, eight_V_scale_std, eight_V_mn_mean, four_K, four_K_idx, four_K_scale_std, four_K_mn_mean, 
                    four_V, four_V_idx, four_V_scale_std, four_V_mn_mean, two_K, two_K_idx, two_K_scale_std, two_K_mn_mean, 
                    two_V, two_V_idx, two_V_scale_std, two_V_mn_mean, one_K, one_K_idx, one_K_scale_std, one_K_mn_mean, 
                    one_V, one_V_idx, one_V_scale_std, one_V_mn_mean, outliers, outliers_idx, V_token_norm, residual_new_K, residual_new_V, kv_seq_len)
    return past_key_value

def KV_cache_reconstruct_outlier(
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

    new_len = new_K.shape[2] if new_K is not None else 0
    batch_size = 0
    num_heads = 0
    head_dim = 0
    dtype = torch.float16
    device = None
    # 获取batch_size, num_heads, head_dim, dtype, device
    if sixteen_V is not None:
        batch_size, num_heads, _, head_dim = sixteen_V.shape
        device = sixteen_V.device
    elif eight_V is not None:
        batch_size = eight_V.shape[0]
        num_heads = eight_V.shape[1]
        head_dim = eight_V.shape[3] * (32 // 8)
        device = eight_V.device
    elif four_V is not None:
        batch_size = four_V.shape[0]
        num_heads = four_V.shape[1]
        head_dim = four_V.shape[3] * (32 // 4)
        device = four_V.device
    elif two_V is not None:
        batch_size = two_V.shape[0]
        num_heads = two_V.shape[1]
        head_dim = two_V.shape[3] * (32 // 2)
        device = two_V.device
    elif one_V is not None:
        batch_size = one_V.shape[0]
        num_heads = one_V.shape[1]
        head_dim = one_V.shape[3] * (32 // 1)
        device = one_V.device
    elif new_V is not None:
        batch_size, num_heads, _, head_dim = new_V.shape
    else:
        raise ValueError('invalid past_key_value')
    
    # 重建的 K, V
    reconstuct_K = torch.zeros(batch_size, num_heads, kv_seq_len, head_dim, dtype=dtype, device=device)
    reconstuct_V = torch.zeros(batch_size, num_heads, kv_seq_len, head_dim, dtype=dtype, device=device)

    # 把 16 bit 的 K, V 放回对应的位置
    if sixteen_K is not None:
        reconstuct_K[:, :, sixteen_K_idx, :] = sixteen_K.transpose(2, 3)
    if sixteen_V is not None:
        insert_V_outliers(sixteen_V, sixteen_V_idx, outliers, outliers_idx)
        reconstuct_V[:, :, sixteen_V_idx, :] = sixteen_V

    # 把 8 bit 的 K, V 放回对应的位置
    if eight_K is not None:
        reconstuct_K_eight = K_dequantize_high(eight_K, eight_K_scale_std, eight_K_mn_mean, block_size, 8).transpose(2, 3)
        reconstuct_K[:, :, eight_K_idx, :] = reconstuct_K_eight
    if eight_V is not None:
        reconstuct_V_eight = V_dequantize_high(eight_V, eight_V_scale_std, eight_V_mn_mean, group_size, 8)
        insert_V_outliers(reconstuct_V_eight, eight_V_idx, outliers, outliers_idx)
        reconstuct_V[:, :, eight_V_idx, :] = reconstuct_V_eight

    # 把 4 bit 的 K, V 放回对应的位置
    if four_K is not None:
        reconstuct_K_four = K_dequantize_high(four_K, four_K_scale_std, four_K_mn_mean, block_size, 4).transpose(2, 3)
        reconstuct_K[:, :, four_K_idx, :] = reconstuct_K_four
    if eight_V is not None:
        reconstuct_V_four = V_dequantize_high(four_V, four_V_scale_std, four_V_mn_mean, group_size, 4)
        insert_V_outliers(reconstuct_V_four, four_V_idx, outliers, outliers_idx)
        reconstuct_V[:, :, four_V_idx, :] = reconstuct_V_four
    
    # 把 2 bit 的 K, V 放回对应的位置
    if two_K is not None:
        reconstuct_K_two = K_dequantize_high(two_K, two_K_scale_std, two_K_mn_mean, block_size, 2).transpose(2, 3)
        reconstuct_K[:, :, two_K_idx, :] = reconstuct_K_two
    if two_V is not None:
        reconstuct_V_two = V_dequantize_high(two_V, two_V_scale_std, two_V_mn_mean, group_size, 2)
        insert_V_outliers(reconstuct_V_two, two_V_idx, outliers, outliers_idx)
        reconstuct_V[:, :, two_V_idx, :] = reconstuct_V_two

    # 把 1 bit 的 K, V 放回对应的位置
    if one_K is not None:
        reconstuct_K_one = K_dequantize_low(one_K, one_K_scale_std, one_K_mn_mean, block_size, 1).transpose(2, 3)
        reconstuct_K[:, :, one_K_idx, :] =reconstuct_K_one
    if one_V is not None:
        reconstuct_V_one = V_dequantize_low(one_V, one_V_scale_std, one_V_mn_mean, group_size, 1)
        insert_V_outliers(reconstuct_V_two, two_V_idx, outliers, outliers_idx)
        reconstuct_V[:, :, one_V_idx, :] = reconstuct_V_one
    
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
    q = torch.rand(1, 8, 1, 128, dtype=dtype, device=device)
    new_K = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)
    new_V = torch.rand(1, 8, seq_len, 128, dtype=dtype, device=device)

    past_key_value = (None,) * 38 + (new_K, new_V, seq_len)
    token_importance = torch.rand(histroy_length + compress_length)
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
    K_bit_num = torch.tensor([1024, 0, 0, 0, 0], dtype=torch.int32, device=device)
    V_bit_num = torch.tensor([1024, 0, 0, 0, 0], dtype=torch.int32, device=device)

    past_key_value = KV_cache_compress_outlier(past_key_value, compress_length, histroy_length, block_idx, token_idx, K_bit_num, V_bit_num, 'uniform_group')
    reconstruct_K, reconstruct_V = KV_cache_reconstruct_outlier(past_key_value, quant_strategy)