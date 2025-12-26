// Inspired by https://github.com/ankan-ban/llama_cu_awq 
// and the official implementation of AWQ
/*

@article{lin2023awq,
  title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration},
  author={Lin, Ji and Tang, Jiaming and Tang, Haotian and Yang, Shang and Dang, Xingyu and Han, Song},
  journal={arXiv},
  year={2023}
}

*/


#include <cuda_fp16.h>
#include <stdio.h>
#include <torch/extension.h>
#include "gemv_cuda.h"
# include "reduce_utils.h"
#define VECTORIZE_FACTOR 8
#define Q_VECTORIZE_FACTOR 8
#define PACK_FACTOR 8
#define WARP_SIZE 32

// Reduce sum within the warp using the tree reduction algorithm.
__device__ __forceinline__ float warp_reduce_sum(float sum) {
  #pragma unroll
  // 这个for循环实现了32个线程的树形求和
  for(int i = 4; i >= 0; i--){
    // 0xffffffff是一个掩码参数，用于同步所有线程
    // 当前线程的 sum 值将被发送到下移 1<<i 个位置的线程
    sum += __shfl_down_sync(0xffffffff, sum, 1<<i);
  }
  return sum;
}

__global__ void bgemv_kernel_wv_normalize_group_outlier_8(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, const int OL,
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * OC * IC / group_size;//同理
    const half* means = _means + _batch_idx * OC * IC / group_size;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc

    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;//乘以4的原因是每个线程算4个数
    int std_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
    inputs_ptr_delta = outliers_offset = tile_idx * tile_dim + tid * 32;
    
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float std[32]{};
    float mean[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      std[i] = __half2float(*(stds + std_mn_offset + i));
      mean[i] = __half2float(*(means + std_mn_offset + i));
    }
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_std = std[i_num];
      float cur_mean = mean[i_num];
      // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // 判断当前位置是否是离群值           
          bool is_outlier = false;
          int row;
          if (outliers_offset + i_num < ICR){// 检查列是否越界
            for (row = 0; row < 2; row++){
              if(oc_idx == *(outliers_idx + outliers_offset + i_num + row * ICR)){
                is_outlier = true;
                break;
              }
            }
          }
          if (is_outlier){
            // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
            float outlier = __half2float(*(outliers + outliers_offset + i_num + row * ICR));
            res[ic_1] += outlier * cur_inp;
          }else{
            // unpack数据
            int cur_single_weight_fp = cur_packed_weight & num;
            float center = *(_normal_quantiles_center + cur_single_weight_fp);
            // 进行反量化
            float dequantized_weight = cur_std * center + cur_mean;
            // 进行右移,unpack下一个数
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的pack_factor个量化的数的相乘
            res[ic_1] += dequantized_weight * cur_inp;
          }
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        res[i] = warp_reduce_sum(res[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (tid == 0) 
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
      }
    }
}

__global__ void bgemv_kernel_wv_normalize_group_outlier_4(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, const int OL,
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * OC * IC / group_size;//同理
    const half* means = _means + _batch_idx * OC * IC / group_size;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc

    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;//乘以4的原因是每个线程算4个数
    int std_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
    inputs_ptr_delta = outliers_offset = tile_idx * tile_dim + tid * 32;
    
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float std[32]{};
    float mean[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      std[i] = __half2float(*(stds + std_mn_offset + i));
      mean[i] = __half2float(*(means + std_mn_offset + i));
    }
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_std = std[i_num];
      float cur_mean = mean[i_num];
      // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // 判断当前位置是否是离群值           
          bool is_outlier = false;
          int row;
          if (outliers_offset + i_num < ICR){// 检查列是否越界
            for (row = 0; row < 2; row++){
              if(oc_idx == *(outliers_idx + outliers_offset + i_num + row * ICR)){
                is_outlier = true;
                break;
              }
            }
          }
          if (is_outlier){
            // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
            float outlier = __half2float(*(outliers + outliers_offset + i_num + row * ICR));
            res[ic_1] += outlier * cur_inp;
          }else{
            // unpack数据
            int cur_single_weight_fp = cur_packed_weight & num;
            float center = *(_normal_quantiles_center + cur_single_weight_fp);
            // 进行反量化
            float dequantized_weight = cur_std * center + cur_mean;
            // 进行右移,unpack下一个数
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的pack_factor个量化的数的相乘
            res[ic_1] += dequantized_weight * cur_inp;
          }
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        res[i] = warp_reduce_sum(res[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (tid == 0) 
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
      }
    }
}

__global__ void bgemv_kernel_wv_normalize_group_outlier_2(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, const int OL,
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * OC * IC / group_size;//同理
    const half* means = _means + _batch_idx * OC * IC / group_size;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc

    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;//乘以4的原因是每个线程算4个数
    int std_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
    inputs_ptr_delta = outliers_offset = tile_idx * tile_dim + tid * 32;
    
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float std[32]{};
    float mean[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      std[i] = __half2float(*(stds + std_mn_offset + i));
      mean[i] = __half2float(*(means + std_mn_offset + i));
    }
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_std = std[i_num];
      float cur_mean = mean[i_num];
      // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // 判断当前位置是否是离群值           
          bool is_outlier = false;
          int row;
          if (outliers_offset + i_num < ICR){// 检查列是否越界
            for (row = 0; row < 2; row++){
              if(oc_idx == *(outliers_idx + outliers_offset + i_num + row * ICR)){
                is_outlier = true;
                break;
              }
            }
          }
          if (is_outlier){
            // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
            float outlier = __half2float(*(outliers + outliers_offset + i_num + row * ICR));
            res[ic_1] += outlier * cur_inp;
          }else{
            // unpack数据
            int cur_single_weight_fp = cur_packed_weight & num;
            float center = *(_normal_quantiles_center + cur_single_weight_fp);
            // 进行反量化
            float dequantized_weight = cur_std * center + cur_mean;
            // 进行右移,unpack下一个数
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的pack_factor个量化的数的相乘
            res[ic_1] += dequantized_weight * cur_inp;
          }
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        res[i] = warp_reduce_sum(res[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (tid == 0) 
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
      }
    }
}

__global__ void bgemv_kernel_wv_normalize_group_outlier_1(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, const int OL,
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y;
    const int tile_idx = blockIdx.z;
    const int tid = threadIdx.x;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * OC * IC / group_size;//同理
    const half* means = _means + _batch_idx * OC * IC / group_size;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc

    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;//乘以4的原因是每个线程算4个数
    int std_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
    inputs_ptr_delta = outliers_offset = tile_idx * tile_dim + tid * 32;
    
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float std[32]{};
    float mean[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      std[i] = __half2float(*(stds + std_mn_offset + i));
      mean[i] = __half2float(*(means + std_mn_offset + i));
    }
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_std = std[i_num];
      float cur_mean = mean[i_num];
      // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // 判断当前位置是否是离群值           
          bool is_outlier = false;
          int row;
          if (outliers_offset + i_num < ICR){// 检查列是否越界
            for (row = 0; row < 2; row++){
              if(oc_idx == *(outliers_idx + outliers_offset + i_num + row * ICR)){
                is_outlier = true;
                break;
              }
            }
          }
          if (is_outlier){
            // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
            float outlier = __half2float(*(outliers + outliers_offset + i_num + row * ICR));
            res[ic_1] += outlier * cur_inp;
          }else{
            // unpack数据
            int cur_single_weight_fp = cur_packed_weight & num;
            float center = *(_normal_quantiles_center + cur_single_weight_fp);
            // 进行反量化
            float dequantized_weight = cur_std * center + cur_mean;
            // 进行右移,unpack下一个数
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的pack_factor个量化的数的相乘
            res[ic_1] += dequantized_weight * cur_inp;
          }
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        res[i] = warp_reduce_sum(res[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (tid == 0) 
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
      }
    }
}

/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC // PACK_Factor, IC];
  _means: int tensor of shape [OC // G, IC];
  _stds: tensor of shape [OC // G, IC];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;
Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda_wv_normalize_group_outlier(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _stds,
    torch::Tensor _means,
    torch::Tensor _normal_quantiles_center,
    torch::Tensor _outliers,
    torch::Tensor _outliers_idx,
    const int bit,
    const int group_size,
    const int nh_q,
    const int nh_kv)
{
    int BS = _in_feats.size(0);
    int num_in_feats = _in_feats.size(1);
    int num_in_channels = _in_feats.size(2);
    int num_out_channels = _means.size(1) * group_size;
    // 将 _in_feats 转换为类型指针
    // at 是 PyTorch 中的一个命名空间
    auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
    auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
    auto means = reinterpret_cast<half*>(_means.data_ptr<at::Half>());
    auto stds = reinterpret_cast<half*>(_stds.data_ptr<at::Half>());
    auto normal_quantiles_center = reinterpret_cast<float*>(_normal_quantiles_center.data_ptr<float>());
    auto outliers = reinterpret_cast<half*>(_outliers.data_ptr<at::Half>());
    auto outliers_idx = reinterpret_cast<int64_t*>(_outliers_idx.data_ptr<int64_t>());
    // auto out_in_map = _out_in_map.data_ptr<int>();
    auto options =
    torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
    // kernel is [OC, IC]
    // 存储计算结果的tensor
    const int num_out_lines = (num_in_channels + 1023) / 1024;
    at::Tensor _out_feats = torch::zeros({BS, num_in_feats, num_out_channels}, options);
    int num_out_feats = _out_feats.size(-2);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int pack_factor = 32 / bit;
    // 定义网格尺寸
    dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4, num_out_lines);
    // 定义线程块尺寸
    dim3 num_threads(32, 4);
    // 进行核函数计算
    if (bit == 8){
      bgemv_kernel_wv_normalize_group_outlier_8<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
        // constants
        num_in_channels, num_out_channels, num_out_lines, group_size, bit, nh_q, nh_kv
      ); 
    }else if (bit == 4){
      bgemv_kernel_wv_normalize_group_outlier_4<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
        // constants
        num_in_channels, num_out_channels, num_out_lines, group_size, bit, nh_q, nh_kv
      ); 
    }else if (bit == 2){
      bgemv_kernel_wv_normalize_group_outlier_2<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
        // constants
        num_in_channels, num_out_channels, num_out_lines, group_size, bit, nh_q, nh_kv
      ); 
    }else if (bit == 1){
      bgemv_kernel_wv_normalize_group_outlier_1<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
        // constants
        num_in_channels, num_out_channels, num_out_lines, group_size, bit, nh_q, nh_kv
      ); 
    }
    return _out_feats;
;}