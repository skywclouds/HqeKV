import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds
import torch
import fast_hadamard_transform, math

def approximate_compress_ratio(layer_memory, batch_size, hidden_size, total_seq_len):
    n_bits = torch.tensor([1, 2, 4, 8, 16], dtype=torch.int32)
    full_length = 64
    if total_seq_len <= full_length :
        compress_ratio = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        return compress_ratio
    else:
        full_memory = batch_size * 2 * hidden_size * full_length  * 2
        residual_memory = batch_size * 2 * hidden_size * (total_seq_len - full_length ) * 2
        bit = 16 * (layer_memory - full_memory) / residual_memory
        bit_idx = torch.searchsorted(n_bits, bit)
        if bit_idx == 0:
            compress_ratio = torch.tensor([full_length  / total_seq_len, 0, 0, 0, (total_seq_len - full_length ) / total_seq_len], dtype=torch.float32)
            return compress_ratio
        if bit_idx == 4:
            bit = 16 * layer_memory / (batch_size * 2 * hidden_size * total_seq_len * 2)
            high_bit = 16
            low_bit = 8
            high_idx = 0
            low_idx = 1
            high_ratio = (bit - low_bit) / (high_bit - low_bit)
            low_ratio = 1 - high_ratio
            compress_ratio = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
            compress_ratio[low_idx] = low_ratio
            compress_ratio[high_idx] = high_ratio
            return compress_ratio
        if bit_idx == 5:
            compress_ratio = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
            return compress_ratio
        high_bit = n_bits[bit_idx]
        low_bit = n_bits[bit_idx - 1]
        high_idx = 4 - bit_idx
        low_idx = high_idx + 1
        residual_ratio = 1 - full_length  / total_seq_len
        high_ratio = (bit - low_bit) / (high_bit - low_bit) * residual_ratio
        low_ratio = residual_ratio - high_ratio
        compress_ratio = torch.tensor([full_length  / total_seq_len, 0, 0, 0, 0], dtype=torch.float32)
        compress_ratio[low_idx] = low_ratio
        compress_ratio[high_idx] = high_ratio
        return compress_ratio
    
def calculate_token_norm(new_cache: torch.Tensor, token_norm: torch.Tensor):
        if new_cache is not None:
            if token_norm is None:
                token_norm = new_cache.mul(new_cache).sum(0).sum(0).sum(-1)
            else:
                new_norm = new_cache.mul(new_cache).sum(0).sum(0).sum(-1)
                token_norm = torch.cat((token_norm, new_norm), dim=-1)
            return token_norm
        else:
            return None
        
def calculate_token_range(new_cache: torch.Tensor, token_range: torch.Tensor):
        if new_cache is not None:
            if token_range is None:
                max_value = torch.max(new_cache, dim=-1)[0]
                min_value = torch.min(new_cache, dim=-1)[0]
                token_range = max_value - min_value
                token_range = token_range.sum(0).sum(0)
            else:
                max_value = torch.max(new_cache, dim=-1)[0]
                min_value = torch.min(new_cache, dim=-1)[0]
                new_range = max_value - min_value
                new_range = new_range.sum(0).sum(0)
                token_range = torch.cat((token_range, new_range), dim=-1)
            return token_range
        else:
            return None

def calculate_new_token_range(new_cache: torch.Tensor):
    if new_cache is not None:
        max_value = torch.max(new_cache, dim=-1)[0]
        min_value = torch.min(new_cache, dim=-1)[0]
        token_range = max_value - min_value
        token_range = token_range.sum(0).sum(0)
        return token_range
    else:
        return None

def get_block_importance(block_size: int, block_num: int, importance: torch.Tensor):
    block_importance = importance.reshape(block_num, -1)
    # 均值的情况
    block_importance /= block_importance.shape[1]
    block_importance = torch.sum(block_importance, dim=-1)
    return block_importance

def get_block_token_idx(block_size: int, block_num: int, importance: torch.Tensor):
    '''获取分块的各个token的索引'''
    block_importance = importance.reshape(block_num, -1)
    # 均值的情况
    block_importance /= block_importance.shape[1]
    block_importance = torch.sum(block_importance, dim=-1)
    # 最大值的情况
    # block_importance = torch.max(block_importance, dim=-1)[0]
    _, block_idx = torch.sort(block_importance, descending=True)
    block_idx *= block_size
    block_idx = block_idx.reshape(-1, 1).repeat(1, block_size)
    add = torch.arange(0, block_size, dtype=block_idx.dtype, device=block_idx.device).reshape(1, -1)
    block_idx += add
    block_idx = block_idx.flatten()
    return block_idx

def get_compress_ratio(layer_memory, batch_size, hidden_size, total_seq_len):
    '''根据层大小求解该层各个量化位数的比例'''
    # 目标函数
    def objective(x):
        mean = np.mean(x)
        return np.sum((mean - x)**2)  # 计算方差

    # 如果全部 16 bit 空间仍然够用, 则全部 16 bit
    if batch_size * 2 * hidden_size * total_seq_len * 2 <= layer_memory:
        compress_ratio = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
        return compress_ratio
    full_length = 64
    # 如果全部 1 bit 空间仍然不够用, 则全部 1 bit
    if batch_size * 2 * hidden_size * ((total_seq_len - full_length) * 2 / 16 + full_length * 2) >= layer_memory:
        compress_ratio = torch.tensor([full_length / total_seq_len, 0, 0, 0, (total_seq_len - full_length) / total_seq_len], dtype=torch.float32)
        return compress_ratio

    # 定义变量的边界
    bounds = Bounds([0, 0, 0, 0, 0], [1, 1, 1, 1, 1])

    # 定义线性等式约束
    A_eq = np.array([[16, 8, 4, 2, 1], [1, 1, 1, 1, 1]])
    b_eq = np.array([layer_memory * 8 / (batch_size * hidden_size * total_seq_len * 2), 1])
    linear_constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)  

    # 定义线性不等式约束
    A_ineq = np.array([[total_seq_len, 0, 0, 0, 0]])
    b_ineq = np.array([full_length])
    linear_constraint_ineq0 = LinearConstraint(A_ineq, b_ineq, np.inf)

    A_ineq = np.array([[0, 0, 0, 0, 1]])
    b_ineq = np.array([0.1])
    linear_constraint_ineq1 = LinearConstraint(A_ineq, -np.inf, b_ineq)

    x0 = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

    # 使用差分进化的结果作为初始猜测进行局部优化
    result = minimize(objective, x0, bounds=bounds, constraints=[linear_constraint_eq, linear_constraint_ineq0, linear_constraint_ineq1])
    compress_ratio = torch.tensor(result.x).to(torch.float32)

    #compress_ratio = torch.tensor([64 / total_seq_len, 0, 1, 0, 0], dtype=torch.float32)
    return compress_ratio

def get_compress_ratio_2(layer_memory, batch_size, hidden_size, total_seq_len):

    full_length = 64
    if total_seq_len <= full_length:
        compress_ratio = torch.tensor([1, 0, 0, 0, 0], dtype=torch.float32)
    else:
        # 目标函数
        def objective(x):
            return -x[2]

        # 定义变量的边界
        bounds = Bounds([0, 0, 0], [1, 1, 1])

        # 定义线性等式约束
        A_eq = np.array([[16, 2, 1], [1, 1, 1]])
        b_eq = np.array([layer_memory * 8 / (batch_size * hidden_size * total_seq_len * 2), 1])
        linear_constraint_eq = LinearConstraint(A_eq, b_eq, b_eq)  

        A_ineq = np.array([[0, 0, 1]])
        b_ineq = np.array([0.1])
        linear_constraint_ineq = LinearConstraint(A_ineq, 0, b_ineq)

        x0 = np.array([0.33, 0.33, 0.34])

        # 使用差分进化的结果作为初始猜测进行局部优化
        result = minimize(objective, x0, bounds=bounds, constraints=[linear_constraint_eq, linear_constraint_ineq])
        compress_ratio = torch.tensor([result.x[0], 0, 0, result.x[1], result.x[2]], dtype=torch.float32)
    
    return compress_ratio

def hadamard_transform(v_proj, o_proj, head_dim):
    
    # 对 v_proj 的权重进行 hadamard_transform
    W_ = v_proj.weight.data
    W_dtype = W_.dtype
    W_device = W_.device
    W_ = W_.t()
    transposed_shape = W_.shape
    W_ = W_.reshape(-1, transposed_shape[-1]//head_dim, head_dim)
    W_ = fast_hadamard_transform.hadamard_transform(
        W_, 
        scale=1/math.sqrt(head_dim)
        ).reshape(transposed_shape).t()
    v_proj.weight.data = W_.to(device=W_device, dtype=W_dtype)

    # 对 o_proj 的权重进行 hadamard_transform
    W_ = o_proj.weight.data
    W_dtype = W_.dtype
    W_device = W_.device
    W_shape = W_.shape
    W_ = W_.reshape(-1, W_shape[-1]//head_dim, head_dim)
    W_ = fast_hadamard_transform.hadamard_transform(
        W_, 
        scale=1/math.sqrt(head_dim)
        ).reshape(W_shape)
    o_proj.weight.data = W_.to(device=W_device, dtype=W_dtype)

class last_attn_weights:

    def __init__(self, last_length: int = 20):
        self.last_length = last_length
        self.max_length = 0
        self.kv_seq_len = 0
        self.next_line = 0
        self.weights = None
    
    def calculate_token_importance(self):
        count = torch.arange(self.kv_seq_len, 0, -1, dtype=self.weights.dtype, device=self.weights.device)
        count[:-self.last_length] = self.last_length
        token_importance = self.weights[:, :self.kv_seq_len].sum(0) / count
        return token_importance
    
    def clear(self):
        '''将self.weighs置为0'''
        self.next_line = 0
        self.weights.zero_()

    def eviction(self, reserve_idx):
        if self.weights is not None:
            reserve_weight = self.weights[:, reserve_idx]
            self.weights[:, :reserve_weight.shape[1]] = reserve_weight
    
    def reset(self):
        '''重置所有成员变量'''
        self.max_length = 0
        self.kv_seq_len = 0
        self.next_line = 0
        self.weights = None

    def set_max_length(self, max_length: int):
        self.max_length = max_length

    def update(self, attn_weights: torch.Tensor):
        if self.weights is None:
            self.weights = torch.zeros(self.last_length, self.max_length, dtype=attn_weights.dtype, device=attn_weights.device)
        
        if attn_weights.shape[2] > 1:
            if attn_weights.shape[2] < self.last_length:
                new_line = attn_weights.sum(0).sum(0)
            else:
                new_line = attn_weights[:, :, -self.last_length:, :].sum(0).sum(0)
            self.weights[:new_line.shape[0], :new_line.shape[1]] = new_line
        else:
            new_line = attn_weights.sum(0).sum(0)
            self.weights[self.next_line:self.next_line+1, :new_line.shape[1]] = new_line
        self.kv_seq_len = new_line.shape[1]
        self.next_line = (self.next_line + new_line.shape[0]) % self.last_length

def make_sum_equal(array: torch.Tensor, array_sum: int, start_idx: int = 4):
        sub = array.sum() - array_sum
        for i in range(start_idx, -1, -1):
            if array[i] - sub < 0:
                sub -= array[i]
                array[i] = 0
            else:
                array[i] -= sub
                break
        return array

def manuset_ratio(ratio:torch.Tensor, full_length:int, total_seq_len:int):
    compress_ratio = torch.tensor([0, 0, 0, 0, 0], dtype=torch.float32)
    compress_ratio[0] = full_length / total_seq_len
    residual_ratio = 1 - compress_ratio[0]
    compress_ratio[1:] = residual_ratio * ratio
    return compress_ratio

if __name__ == '__main__':
    device='cuda:2'
    seq_len = 30 * 1024
    import time
    start_time = time.time()
    last_weights = last_attn_weights(last_length=20)
    last_weights.max_length = seq_len + 128
    attn_weights = torch.rand(1, 8, 20, seq_len, dtype=torch.float16, device=device)
    last_weights.update(attn_weights)
    importance = last_weights.calculate_token_importance()
    end_time = time.time()
    print(end_time - start_time)

    start_time = time.time()
    attn_weights = torch.rand(1, 8, 1, seq_len, dtype=torch.float16, device=device)
    last_weights.update(attn_weights)
    importance = last_weights.calculate_token_importance()
    end_time = time.time()
    print(end_time - start_time)