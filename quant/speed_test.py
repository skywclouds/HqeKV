import torch
from pack_unpack import (
	uniform_group_quantize_pack_last_dim,
	normal_group_quantize_pack_last_dim)
from matmul import (cuda_bmm_fA_qB_uniform_group,
                    cuda_bmm_fA_qB_uniform_group_outlier,
                    cuda_bmm_fA_qB_wv_uniform_group,
                    cuda_bmm_fA_qB_wv_uniform_group_outlier,
                    cuda_bmm_fA_qB_normal_group,
                    cuda_bmm_fA_qB_normal_group_outlier,
                    cuda_bmm_fA_qB_wv_normal_group_outlier)
import time

def test_qk(device: torch.device):
    B, seq_len, bit = 1, 60 * 1024, 4
    fA = torch.rand(B, 32, 1, 128, dtype=torch.float16, device=device)
    k = torch.rand(B, 8, seq_len, 128, dtype=torch.float16, device=device)
    qB, scale, zero = uniform_group_quantize_pack_last_dim(k.clone().transpose(2, 3), 32, bit)
    
    start_time = time.time()
    c = cuda_bmm_fA_qB_uniform_group(32, fA, qB, scale, zero, bit)
    end_time = time.time()
    print(end_time - start_time)

def test_wv(device: torch.device):
    B, M, K, N, bit = 1, 1, 60 * 1024, 128, 2
    feat_per_int = 32 // bit
    group_size = 32

    fA = torch.randn(B, 32, M, K, dtype=torch.float16, device=device)
    data = torch.randn(B, 8, K, N, dtype=torch.float16, device=device)
    data_outliers = torch.zeros(data.shape[0], data.shape[1], data.shape[2], 2, dtype=data.dtype, device=data.device)
    data_outliers_idx = torch.zeros(data.shape[0], data.shape[1], data.shape[2], 2, dtype=torch.int64, device=data.device)
    data_median, _ = torch.median(data, dim=-1, keepdim=True)
    data_up_outliers, data_up_outliers_idx = torch.topk(data, k=1, dim=-1)
    data_low_outliers, data_low_outliers_idx = torch.topk(data, k=1, dim=-1, largest=False)
    data_outliers[:, :, :, 0] = data_up_outliers[:, :, :, 0]
    data_outliers[:, :, :, 1] = data_low_outliers[:, :, :, 0]
    data_outliers_idx[:, :, :, 0] = data_up_outliers_idx[:, :, :, 0]
    data_outliers_idx[:, :, :, 1] = data_low_outliers_idx[:, :, :, 0]
    data.scatter_(dim=-1, index=data_outliers_idx, src=data_median.expand_as(data_outliers_idx))
    qB, scales, zeros = uniform_group_quantize_pack_last_dim(data.clone(), group_size, bit)
    
    # start_time = time.time()
    # c = cuda_bmm_fA_qB_wv_uniform_group_outlier(group_size, fA, qB, scales, zeros, data_outliers, data_outliers_idx, bit)
    # end_time = time.time()
    # print(end_time - start_time)
    
    # start_time = time.time()
    # c = cuda_bmm_fA_qB_uniform_group_outlier(group_size, fA, qB, scales, zeros, data_outliers, data_outliers_idx, bit)
    # end_time = time.time()
    # print(end_time - start_time)

    # start_time = time.time()
    # for _ in range(200):
    #     c = cuda_bmm_fA_qB_wv_uniform_group(group_size, fA, qB, scales, zeros, bit)
    # end_time = time.time()
    # print(end_time - start_time)

    start_time = time.time()
    c = cuda_bmm_fA_qB_uniform_group(group_size, fA, qB, scales, zeros, bit)
    end_time = time.time()
    print(end_time - start_time)

if __name__ == '__main__':

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

    device = 'cuda:6'
    test_qk(device)
    # test_wv(device)
    