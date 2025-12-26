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
// #include <torch/cuda.h>
// #include <c10/cuda/CUDAGuard.h>
#include "gemv_cuda.h"
// # include "reduce_utils.h"
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

__global__ void bgemv_kernel_wv_uniform_group_8(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, 
  half* _outputs, const int IC, const int OC, const int OL,
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
    // float* outputs = _outputs + batch_idx * OL * OC + tile_idx * OC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc
    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;
    int scale_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta = tile_idx * tile_dim + tid * 32; 
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float scale[32]{};
    float zero[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      scale[i] = __half2float(*(scaling_factors + scale_mn_offset + i));
      zero[i] = __half2float(*(zeros + scale_mn_offset + i));
    }
    #pragma unroll // 用于指示编译器展开循环
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_scale = scale[i_num];
      float cur_zero = zero[i_num];
      // 一个int32中有pack_factor个数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的第ic_1个数相乘
          // 这样一个循环结束之后实现了
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的pack_factor个数相乘
          res[ic_1] += dequantized_weight * cur_inp;
          // 进行右移,unpack下一个数
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
        if (tid == 0) {
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
        }
      }
    }
}

__global__ void bgemv_kernel_wv_uniform_group_4(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, 
  half* _outputs, const int IC, const int OC, const int OL,
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
    // float* outputs = _outputs + batch_idx * OL * OC + tile_idx * OC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc
    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;
    int scale_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta = tile_idx * tile_dim + tid * 32; 
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float scale[32]{};
    float zero[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      scale[i] = __half2float(*(scaling_factors + scale_mn_offset + i));
      zero[i] = __half2float(*(zeros + scale_mn_offset + i));
    }
    #pragma unroll // 用于指示编译器展开循环
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_scale = scale[i_num];
      float cur_zero = zero[i_num];
      // 一个int32中有pack_factor个数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的第ic_1个数相乘
          // 这样一个循环结束之后实现了
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的pack_factor个数相乘
          res[ic_1] += dequantized_weight * cur_inp;
          // 进行右移,unpack下一个数
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
        if (tid == 0) {
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
        }
      }
    }
}

__global__ void bgemv_kernel_wv_uniform_group_2(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, 
 half* _outputs, const int IC, const int OC, const int OL,
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
    // float* outputs = _outputs + batch_idx * OL * OC + tile_idx * OC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc
    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;
    int scale_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta = tile_idx * tile_dim + tid * 32; 
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float scale[32]{};
    float zero[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      scale[i] = __half2float(*(scaling_factors + scale_mn_offset + i));
      zero[i] = __half2float(*(zeros + scale_mn_offset + i));
    }
    #pragma unroll // 用于指示编译器展开循环
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_scale = scale[i_num];
      float cur_zero = zero[i_num];
      // 一个int32中有pack_factor个数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的第ic_1个数相乘
          // 这样一个循环结束之后实现了
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的pack_factor个数相乘
          res[ic_1] += dequantized_weight * cur_inp;
          // 进行右移,unpack下一个数
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
        if (tid == 0) {
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
        }
      }
    }
}

__global__ void bgemv_kernel_wv_uniform_group_1(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, 
  half* _outputs, const int IC, const int OC, const int OL,
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
    // float* outputs = _outputs + batch_idx * OL * OC + tile_idx * OC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    const int tile_dim = 1024;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float res[pack_factor]{}; // 这里必须用这种初始化方式, 不能用malloc
    int weight_offset = packed_oc_idx * ICR + tile_idx * tile_dim + tid * 32;
    int scale_mn_offset = group_idx * ICR + tile_idx * tile_dim + tid * 32;
    int inputs_ptr_delta = tile_idx * tile_dim + tid * 32; 
    uint32_t packed_weight[32]{};
    float inp[32]{};
    float scale[32]{};
    float zero[32]{};
    // 防止行越界
    for (int i = 0; i < 32 && inputs_ptr_delta + i < ICR; i++){
      packed_weight[i] =  *(weight + weight_offset + i);
      inp[i] = __half2float(*(inputs + inputs_ptr_delta + i));
      scale[i] = __half2float(*(scaling_factors + scale_mn_offset + i));
      zero[i] = __half2float(*(zeros + scale_mn_offset + i));
    }
    #pragma unroll // 用于指示编译器展开循环
    // 每个线程算 32 个数, 这样 32 个线程算 1024 个数
    for (int i_num = 0; i_num < 32; i_num ++){
      uint32_t cur_packed_weight = packed_weight[i_num];
      float cur_inp = inp[i_num];
      float cur_scale = scale[i_num];
      float cur_zero = zero[i_num];
      // 一个int32中有pack_factor个数,所以循环pack_factor次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int oc_idx = oc_start_idx + ic_1;
        if (oc_idx < OC){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的第ic_1个数相乘
          // 这样一个循环结束之后实现了
          // 向量中的第ic_0个数与矩阵中的第ic_0个int32中的pack_factor个数相乘
          res[ic_1] += dequantized_weight * cur_inp;
          // 进行右移,unpack下一个数
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
        if (tid == 0) {
          atomicAdd((outputs + oc_idx), __float2half(res[i]));
        }
      }
    }
}

__global__ void bgemv_kernel_wv_group_hq(
  const half* _in_feats,
  half* _outputs_16,
  half* _outputs_8,
  half* _outputs_4,
  half* _outputs_2,
  half* _outputs_1,
  half* _outputs_new,
  const half* _weight_16,
  const int64_t* _weight_16_idx,
  const uint32_t* _weight_8,
  const int64_t* _weight_8_idx,
  const half* _scale_8,
  const half* _zeros_8,
  const uint32_t* _weight_4,
  const int64_t* _weight_4_idx,
  const half* _scale_4,
  const half* _zeros_4,
  const uint32_t* _weight_2,
  const int64_t* _weight_2_idx,
  const half* _scale_2,
  const half* _zeros_2,
  const uint32_t* _weight_1,
  const int64_t* _weight_1_idx,
  const half* _std_1,
  const half* _means_1,
  const float* _normal_quantiles_center,
  const half* _weight_new,
  const int IC,// num_in_channels
  const int QuantLen,
  const int OC,// num_out_channels
  const int* bit_num,
  const int* bit_packed_oc_idx,
  const int group_size,
  const int head_ratio)
{
  const int batch_idx = blockIdx.x;// 当前线程在第几个batch
  // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
  const half* inputs = _in_feats + batch_idx * IC;
  int _batch_idx = batch_idx / head_ratio;// q 和 kv 的头数不同
  const int row_idx = blockDim.y * blockIdx.y + threadIdx.y;// 当前线程在第几行
  // 获取当前线程处理哪个位数
  int cur_bit = 0;
  int bits[6] = {16, 8, 4, 2, 1, 100};
  int bit_idx = 0;
  for (bit_idx = 0; bit_idx < 6; bit_idx++)
    if(row_idx >= bit_packed_oc_idx[bit_idx] && row_idx < bit_packed_oc_idx[bit_idx+1])
      break;
  cur_bit = bits[bit_idx];
  const int num = 0xFF >> (8-cur_bit);
  const int packed_oc_idx = row_idx - bit_packed_oc_idx[bit_idx];// 当前线程在该位数的第几行
  const int TILE_DIM = 128;// 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
  const int WC = bit_num[bit_idx]; // 当前位数的数量, weight channel

  if (cur_bit == 16){// 16 bit 的情况
    half* outputs_16 = _outputs_16 + batch_idx * OC;
    const int oc_idx = packed_oc_idx;
    const half* weight = _weight_16 + _batch_idx * WC * OC;
    float psum = 0;
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      half w[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + _weight_16_idx[inputs_ptr_delta + ic_0]);
      }
      if(inputs_ptr_delta < WC){
        half2 pack[2];
        memcpy(&pack[0], &weight[weight_offset], sizeof(half2));
        memcpy(&pack[1], &weight[weight_offset + 2], sizeof(half2));
        w[0] = __low2half(pack[0]);
        w[1] = __high2half(pack[0]);
        w[2] = __low2half(pack[1]);
        w[3] = __high2half(pack[1]);
      }
      #pragma unroll // 用于指示编译器展开循环
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        float cur_weight =  __half2float(w[ic_0]); 
        psum += cur_weight * cur_inp;
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) 
      outputs_16[oc_idx] = __float2half(psum);
    
  }else if (cur_bit == 8){// 8 bit 的情况
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    half* outputs_8 = _outputs_8 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_8 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_8 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_8 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + _weight_8_idx[inputs_ptr_delta + ic_0]);
      }
      if (inputs_ptr_delta < WC){
        half2 pack[4];
        memcpy(&pack[0], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[1], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[0]);
        cscale[1] = __high2half(pack[0]);
        cscale[2] = __low2half(pack[1]);
        cscale[3] = __high2half(pack[1]);
        memcpy(&pack[2], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[2]);
        czero[1] = __high2half(pack[2]);
        czero[2] = __low2half(pack[3]);
        czero[3] = __high2half(pack[3]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight = qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有4个int8,所以循环4次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 进行右移,unpack下一个数
          cur_packed_weight = cur_packed_weight >> cur_bit;
          // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
          // 这样一个小循环结束之后实现了
          // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;     
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_8[oc_idx] = __float2half(psum[i]); 
    }

  }else if (cur_bit == 4){// 4 bit 的情况
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    half* outputs_4 = _outputs_4 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_4 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_4 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_4 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + _weight_4_idx[inputs_ptr_delta + ic_0]);
      }
      if (inputs_ptr_delta < WC){
        half2 pack[4];
        memcpy(&pack[0], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[1], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[0]);
        cscale[1] = __high2half(pack[0]);
        cscale[2] = __low2half(pack[1]);
        cscale[3] = __high2half(pack[1]);
        memcpy(&pack[2], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[2]);
        czero[1] = __high2half(pack[2]);
        czero[2] = __low2half(pack[3]);
        czero[3] = __high2half(pack[3]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
       for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight = qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){        
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;          
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_4[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 2){// 2 bit 的情况
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    half* outputs_2 = _outputs_2 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_2 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_2 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_2 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + _weight_2_idx[inputs_ptr_delta + ic_0]);
      }
      if (inputs_ptr_delta < WC){
        half2 pack[4];
        memcpy(&pack[0], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[1], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[0]);
        cscale[1] = __high2half(pack[0]);
        cscale[2] = __low2half(pack[1]);
        cscale[3] = __high2half(pack[1]);
        memcpy(&pack[2], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[2]);
        czero[1] = __high2half(pack[2]);
        czero[2] = __low2half(pack[3]);
        czero[3] = __high2half(pack[3]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight = qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;      
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) {
        outputs_2[oc_idx] = __float2half(psum[i]); 
      }
    }
  }else if (cur_bit == 1){// 1 bit 的情况
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    half* outputs_1 = _outputs_1 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 一个batch占WC * IC / pack_factor的空间,所以要偏移_batch_idx * WC * IC / pack_factor
    const uint32_t*  weight = _weight_1 + _batch_idx * WC * OC / pack_factor;
    const half* stds = _std_1 + _batch_idx * WC * OC / pack_factor;
    const half* means = _means_1 + _batch_idx * WC * OC / pack_factor;
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + _weight_1_idx[inputs_ptr_delta + ic_0]);
      }
      if (inputs_ptr_delta < WC){
        (uint4&) qw = (uint4&)weight[weight_offset];
        (half2&) cstd[0] = (half2&)stds[std_mn_offset];
        (half2&) cstd[2] = (half2&)stds[std_mn_offset + 2];
        (half2&) cmean[0] = (half2&)means[std_mn_offset];
        (half2&) cmean[2] = (half2&)means[std_mn_offset + 2];
        half2 pack[4];
        memcpy(&pack[0], &stds[std_mn_offset], sizeof(half2));
        memcpy(&pack[1], &stds[std_mn_offset + 2], sizeof(half2));
        cstd[0] = __low2half(pack[0]);
        cstd[1] = __high2half(pack[0]);
        cstd[2] = __low2half(pack[1]);
        cstd[3] = __high2half(pack[1]);
        memcpy(&pack[2], &means[std_mn_offset], sizeof(half2));
        memcpy(&pack[3], &means[std_mn_offset + 2], sizeof(half2));
        cmean[0] = __low2half(pack[2]);
        cmean[1] = __high2half(pack[2]);
        cmean[2] = __low2half(pack[3]);
        cmean[3] = __high2half(pack[3]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight = qw[ic_0];
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){     
          int cur_single_weight_fp = cur_packed_weight & num;
          float center = *(_normal_quantiles_center + cur_single_weight_fp);
          float dequantized_weight = cur_std * center + cur_mean;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_1[oc_idx] = __float2half(psum[i]);
    }
  }else if (cur_bit == 100){// new_V 的情况
    half* outputs_new = _outputs_new + batch_idx * OC;
    const int oc_idx = packed_oc_idx;// 输出指针的偏移量
    inputs += QuantLen;
    const half*  weight = _weight_new + _batch_idx * WC * OC;
    float psum = 0;
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      half w[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + inputs_ptr_delta + ic_0);
        w[ic_0] = *(weight + weight_offset + ic_0);
      }
      #pragma unroll // 用于指示编译器展开循环
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        float cur_weight =  __half2float(w[ic_0]);
        psum += cur_weight * cur_inp;
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
      outputs_new[oc_idx] = __float2half(psum);
    }
  }
}

__global__ void bgemv_kernel_wv_group_hq_seq(
  const half* _in_feats,
  half* _outputs_16,
  half* _outputs_8,
  half* _outputs_4,
  half* _outputs_2,
  half* _outputs_1,
  half* _outputs_new,
  const half* _weight_16,
  const uint32_t* _weight_8,
  const half* _scale_8,
  const half* _zeros_8,
  const uint32_t* _weight_4,
  const half* _scale_4,
  const half* _zeros_4,
  const uint32_t* _weight_2,
  const half* _scale_2,
  const half* _zeros_2,
  const uint32_t* _weight_1,
  const half* _std_1,
  const half* _means_1,
  const float* _normal_quantiles_center,
  const half* _weight_new,
  const int IC,// num_in_channels
  const int OC,// num_out_channels
  const int* bit_num,
  const int* bit_packed_oc_idx,
  const int group_size,
  const int head_ratio)
{
  const int batch_idx = blockIdx.x;// 当前线程在第几个batch
  // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
  const half* inputs = _in_feats + batch_idx * IC;
  int _batch_idx = batch_idx / head_ratio;// q 和 kv 的头数不同
  const int row_idx = blockDim.y * blockIdx.y + threadIdx.y;// 当前线程在第几行
  // 获取当前线程处理哪个位数
  int cur_bit = 0;
  int bits[6] = {16, 8, 4, 2, 1, 100};
  int bit_idx = 0;
  for (bit_idx = 0; bit_idx < 6; bit_idx++)
    if(row_idx >= bit_packed_oc_idx[bit_idx] && row_idx < bit_packed_oc_idx[bit_idx+1])
      break;
  cur_bit = bits[bit_idx];
  const int num = 0xFF >> (8-cur_bit);
  const int packed_oc_idx = row_idx - bit_packed_oc_idx[bit_idx];// 当前线程在该位数的第几行
  const int TILE_DIM = 128;// 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
  const int WC = bit_num[bit_idx]; // 当前位数的数量, weight channel

  if (cur_bit == 16){// 16 bit 的情况
    half* outputs_16 = _outputs_16 + batch_idx * OC;
    const int oc_idx = packed_oc_idx;
    const half* weight = _weight_16 + _batch_idx * WC * OC;
    float psum = 0;
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      half w[4]{};
      if(inputs_ptr_delta < WC){
        half2 pack[4];
        memcpy(&pack[0], &inputs[inputs_ptr_delta], sizeof(half2));
        memcpy(&pack[1], &inputs[inputs_ptr_delta + 2], sizeof(half2));
        inp[0] = __low2half(pack[0]);
        inp[1] = __high2half(pack[0]);
        inp[2] = __low2half(pack[1]);
        inp[3] = __high2half(pack[1]);
        memcpy(&pack[2], &weight[weight_offset], sizeof(half2));
        memcpy(&pack[3], &weight[weight_offset + 2], sizeof(half2));
        w[0] = __low2half(pack[2]);
        w[1] = __high2half(pack[2]);
        w[2] = __low2half(pack[3]);
        w[3] = __high2half(pack[4]);
      }
      #pragma unroll // 用于指示编译器展开循环
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        float cur_weight =  __half2float(w[ic_0]);
        psum += cur_weight * cur_inp;
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) 
      outputs_16[oc_idx] = __float2half(psum);
    
  }else if (cur_bit == 8){// 8 bit 的情况
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    half* outputs_8 = _outputs_8 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    inputs += bit_num[0];
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_8 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_8 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_8 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      if (inputs_ptr_delta < WC){
        half2 pack[6];
        memcpy(&pack[0], &inputs[inputs_ptr_delta], sizeof(half2));
        memcpy(&pack[1], &inputs[inputs_ptr_delta + 2], sizeof(half2));
        inp[0] = __low2half(pack[0]);
        inp[1] = __high2half(pack[0]);
        inp[2] = __low2half(pack[1]);
        inp[3] = __high2half(pack[1]);
        memcpy(&pack[2], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[2]);
        cscale[1] = __high2half(pack[2]);
        cscale[2] = __low2half(pack[3]);
        cscale[3] = __high2half(pack[3]);
        memcpy(&pack[4], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[5], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[4]);
        czero[1] = __high2half(pack[4]);
        czero[2] = __low2half(pack[5]);
        czero[3] = __high2half(pack[5]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          // unpack数据
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          // 进行反量化
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          // 进行右移,unpack下一个数
          cur_packed_weight = cur_packed_weight >> cur_bit;
          // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
          // 这样一个小循环结束之后实现了
          // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;     
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_8[oc_idx] = __float2half(psum[i]); 
    }

  }else if (cur_bit == 4){// 4 bit 的情况
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    half* outputs_4 = _outputs_4 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    inputs += bit_num[0] + bit_num[1];
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_4 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_4 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_4 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      if (inputs_ptr_delta < WC){
        half2 pack[6];
        memcpy(&pack[0], &inputs[inputs_ptr_delta], sizeof(half2));
        memcpy(&pack[1], &inputs[inputs_ptr_delta + 2], sizeof(half2));
        inp[0] = __low2half(pack[0]);
        inp[1] = __high2half(pack[0]);
        inp[2] = __low2half(pack[1]);
        inp[3] = __high2half(pack[1]);
        memcpy(&pack[2], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[2]);
        cscale[1] = __high2half(pack[2]);
        cscale[2] = __low2half(pack[3]);
        cscale[3] = __high2half(pack[3]);
        memcpy(&pack[4], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[5], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[4]);
        czero[1] = __high2half(pack[4]);
        czero[2] = __low2half(pack[5]);
        czero[3] = __high2half(pack[5]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){        
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;          
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_4[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 2){// 2 bit 的情况
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    half* outputs_2 = _outputs_2 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;// 输出指针的偏移量
    const int group_idx = oc_start_idx / group_size; // 当前线程在第几组
    inputs += bit_num[0] + bit_num[1] + bit_num[2];
    // 一个batch占 WC * OC / pack_factor的空间,所以要偏移_batch_idx * WC * OC / pack_factor
    const uint32_t*  weight = _weight_2 + _batch_idx * WC * OC / pack_factor;
    const half* scaling_factors = _scale_2 + _batch_idx * WC * OC / group_size;//同理
    const half* zeros = _zeros_2 + _batch_idx * WC * OC / group_size;//同理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      if (inputs_ptr_delta < WC){
        half2 pack[6];
        memcpy(&pack[0], &inputs[inputs_ptr_delta], sizeof(half2));
        memcpy(&pack[1], &inputs[inputs_ptr_delta + 2], sizeof(half2));
        inp[0] = __low2half(pack[0]);
        inp[1] = __high2half(pack[0]);
        inp[2] = __low2half(pack[1]);
        inp[3] = __high2half(pack[1]);
        memcpy(&pack[2], &scaling_factors[scale_mn_offset], sizeof(half2));
        memcpy(&pack[3], &scaling_factors[scale_mn_offset + 2], sizeof(half2));
        cscale[0] = __low2half(pack[2]);
        cscale[1] = __high2half(pack[2]);
        cscale[2] = __low2half(pack[3]);
        cscale[3] = __high2half(pack[3]);
        memcpy(&pack[4], &zeros[scale_mn_offset], sizeof(half2));
        memcpy(&pack[5], &zeros[scale_mn_offset + 2], sizeof(half2));
        czero[0] = __low2half(pack[4]);
        czero[1] = __high2half(pack[4]);
        czero[2] = __low2half(pack[5]);
        czero[3] = __high2half(pack[5]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          float cur_single_weight_fp = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    // 求和
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;      
      // 实现32个线程结果的求和
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) {
        outputs_2[oc_idx] = __float2half(psum[i]); 
      }
    }
  }else if (cur_bit == 1){// 1 bit 的情况
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    half* outputs_1 = _outputs_1 + batch_idx * OC;
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    inputs += bit_num[0] + bit_num[1] + bit_num[2] + bit_num[3];
    // 一个batch占WC * IC / pack_factor的空间,所以要偏移_batch_idx * WC * IC / pack_factor
    const uint32_t*  weight = _weight_1 + _batch_idx * WC * OC / pack_factor;
    const half* stds = _std_1 + _batch_idx * WC * OC / pack_factor;
    const half* means = _means_1 + _batch_idx * WC * OC / pack_factor;
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    float psum[pack_factor]{};
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = group_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      half inp[4]{};
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      if (inputs_ptr_delta < WC){
        half2 pack[6];
        memcpy(&pack[0], &inputs[inputs_ptr_delta], sizeof(half2));
        memcpy(&pack[1], &inputs[inputs_ptr_delta + 2], sizeof(half2));
        inp[0] = __low2half(pack[0]);
        inp[1] = __high2half(pack[0]);
        inp[2] = __low2half(pack[1]);
        inp[3] = __high2half(pack[1]);
        memcpy(&pack[2], &stds[std_mn_offset], sizeof(half2));
        memcpy(&pack[3], &stds[std_mn_offset + 2], sizeof(half2));
        cstd[0] = __low2half(pack[2]);
        cstd[1] = __high2half(pack[2]);
        cstd[2] = __low2half(pack[3]);
        cstd[3] = __high2half(pack[3]);
        memcpy(&pack[4], &means[std_mn_offset], sizeof(half2));
        memcpy(&pack[5], &means[std_mn_offset + 2], sizeof(half2));
        cmean[0] = __low2half(pack[4]);
        cmean[1] = __high2half(pack[4]);
        cmean[2] = __low2half(pack[5]);
        cmean[3] = __high2half(pack[5]);
        uint4 pack_qw;
        memcpy(&pack_qw, &weight[weight_offset], sizeof(uint4));
        qw[0] = pack_qw.x;
        qw[1] = pack_qw.y;
        qw[2] = pack_qw.z;
        qw[3] = pack_qw.w;
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){     
          int cur_single_weight_fp = cur_packed_weight & num;
          float center = *(_normal_quantiles_center + cur_single_weight_fp);
          float dequantized_weight = cur_std * center + cur_mean;
          cur_packed_weight = cur_packed_weight >> cur_bit;
          psum[ic_1] += dequantized_weight * cur_inp;
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs_1[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 100){// new_V 的情况
    half* outputs_new = _outputs_new + batch_idx * OC;
    const int oc_idx = packed_oc_idx;// 输出指针的偏移量
    inputs += bit_num[0] + bit_num[1] + bit_num[2] + bit_num[3] + bit_num[4];
    const half* weight = _weight_new + _batch_idx * WC * OC;
    float psum = 0;
    for (int k=0; k < (WC + TILE_DIM - 1) / TILE_DIM; k++){
      int weight_offset = packed_oc_idx * WC + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x*4; 
      half inp[4]{};
      half w[4]{};
      for (int ic_0 = 0; ic_0 < 4 && inputs_ptr_delta + ic_0 < WC; ic_0++){
        inp[ic_0] = *(inputs + inputs_ptr_delta + ic_0);
        w[ic_0] = *(weight + weight_offset + ic_0);
      }
      #pragma unroll // 用于指示编译器展开循环
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        float cur_inp = __half2float(inp[ic_0]);
        float cur_weight =  __half2float(w[ic_0]);
        psum += cur_weight * cur_inp;
      }
    }
    psum = warp_reduce_sum(psum);
    if (threadIdx.x == 0) {
      outputs_new[oc_idx] = __float2half(psum);
    }
  }
}

torch::Tensor gemv_forward_cuda_wv_uniform_group(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _scaling_factors,
  torch::Tensor _zeros,
  const int bit,
  const int group_size,
  const int nh_q,
  const int nh_kv)
{
  int BS = _in_feats.size(0);
  int num_in_feats = _in_feats.size(1);
  int num_in_channels = _in_feats.size(2);
  int num_out_channels = _zeros.size(1) * group_size;
  // int kernel_volume = _out_in_map.size(1);
  // 将tensor转换为指针
  // at 是 PyTorch 中的一个命名空间
  auto in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
  auto kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
  auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
  auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
  // auto out_in_map = _out_in_map.data_ptr<int>();
  auto options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  // kernel is [OC, IC]
  // 存储计算结果的tensor
  const int num_out_rows = (num_in_channels + 1023) / 1024;
  at::Tensor _out_feats = torch::zeros({BS, num_in_feats, num_out_channels}, options);
  int num_out_feats = _out_feats.size(-2);
  auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
  int pack_factor = 32 / bit;
  // c10::cuda::CUDAStream current_stream = c10::cuda::getCurrentCUDAStream(); 
  // 定义网格尺寸
  dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4, num_out_rows);
  // 定义线程块尺寸
  dim3 num_threads(32, 4);
  // 进行核函数计算
  if (bit == 8){
    bgemv_kernel_wv_uniform_group_8<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, num_out_rows, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 4){
    bgemv_kernel_wv_uniform_group_4<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, num_out_rows, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 2){
    bgemv_kernel_wv_uniform_group_2<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, num_out_rows, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 1){
    bgemv_kernel_wv_uniform_group_1<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, num_out_rows, group_size, bit, nh_q, nh_kv
    );
  }
  
  return _out_feats;
}

torch::Tensor gemv_forward_cuda_wv_group_hq(
  torch::Tensor _in_feats,
  c10::optional<torch::Tensor> _kernel_16,
  c10::optional<torch::Tensor> _kernel_16_idx,
  c10::optional<torch::Tensor> _kernel_8,
  c10::optional<torch::Tensor> _kernel_8_idx,
  c10::optional<torch::Tensor> _scaling_factors_8,
  c10::optional<torch::Tensor> _zeros_8,
  c10::optional<torch::Tensor> _kernel_4,
  c10::optional<torch::Tensor> _kernel_4_idx,
  c10::optional<torch::Tensor> _scaling_factors_4,
  c10::optional<torch::Tensor> _zeros_4,
  c10::optional<torch::Tensor> _kernel_2,
  c10::optional<torch::Tensor> _kernel_2_idx,
  c10::optional<torch::Tensor> _scaling_factors_2,
  c10::optional<torch::Tensor> _zeros_2,
  c10::optional<torch::Tensor> _kernel_1,
  c10::optional<torch::Tensor> _kernel_1_idx,
  c10::optional<torch::Tensor> _std_1,
  c10::optional<torch::Tensor> _means_1,
  torch::Tensor _normal_quantiles_center,
  c10::optional<torch::Tensor> _kernel_new,
  int num_out_channels,
  int num_16,
  int num_8,
  int num_4,
  int num_2,
  int num_1,
  int new_len,
  const int group_size,
  const int nh_q,
  const int nh_kv)
{
  int BS = _in_feats.size(0);
  int num_in_channels = _in_feats.size(2);
  half* in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());

  half* kernel_16 = _kernel_16.has_value() ? reinterpret_cast<half*>(_kernel_16.value().data_ptr<at::Half>()) : nullptr;
  int64_t* kernel_16_idx = _kernel_16_idx.has_value() ? reinterpret_cast<int64_t*>(_kernel_16_idx.value().data_ptr<int64_t>()) : nullptr;
  
  uint32_t* kernel_8 = _kernel_8.has_value() ? reinterpret_cast<uint32_t*>(_kernel_8.value().data_ptr<int>()) : nullptr;
  int64_t* kernel_8_idx = _kernel_8_idx.has_value() ? reinterpret_cast<int64_t*>(_kernel_8_idx.value().data_ptr<int64_t>()) : nullptr;
  half* scaling_factors_8 = _scaling_factors_8.has_value() ? reinterpret_cast<half*>(_scaling_factors_8.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_8 = _zeros_8.has_value() ? reinterpret_cast<half*>(_zeros_8.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_4 = _kernel_4.has_value() ? reinterpret_cast<uint32_t*>(_kernel_4.value().data_ptr<int>()) : nullptr;
  int64_t* kernel_4_idx = _kernel_4_idx.has_value() ? reinterpret_cast<int64_t*>(_kernel_4_idx.value().data_ptr<int64_t>()) : nullptr;
  half* scaling_factors_4 = _scaling_factors_4.has_value() ? reinterpret_cast<half*>(_scaling_factors_4.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_4 = _zeros_4.has_value() ? reinterpret_cast<half*>(_zeros_4.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_2 = _kernel_2.has_value() ? reinterpret_cast<uint32_t*>(_kernel_2.value().data_ptr<int>()) : nullptr;
  int64_t* kernel_2_idx = _kernel_2_idx.has_value() ? reinterpret_cast<int64_t*>(_kernel_2_idx.value().data_ptr<int64_t>()) : nullptr;
  half* scaling_factors_2 = _scaling_factors_2.has_value() ? reinterpret_cast<half*>(_scaling_factors_2.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_2 = _zeros_2.has_value() ? reinterpret_cast<half*>(_zeros_2.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_1 = _kernel_1.has_value() ? reinterpret_cast<uint32_t*>(_kernel_1.value().data_ptr<int>()) : nullptr;
  int64_t* kernel_1_idx = _kernel_1_idx.has_value() ? reinterpret_cast<int64_t*>(_kernel_1_idx.value().data_ptr<int64_t>()) : nullptr;
  half* std_1 = _std_1.has_value() ? reinterpret_cast<half*>(_std_1.value().data_ptr<at::Half>()) : nullptr;
  half* means_1 = _means_1.has_value() ? reinterpret_cast<half*>(_means_1.value().data_ptr<at::Half>()) : nullptr;

  half* kernel_new = _kernel_new.has_value() ? reinterpret_cast<half*>(_kernel_new.value().data_ptr<at::Half>()) : nullptr;
  float* normal_quantiles_center = reinterpret_cast<float*>(_normal_quantiles_center.data_ptr<float>());

  torch::TensorOptions options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats_16 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_16 = reinterpret_cast<half*>(_out_feats_16.data_ptr<at::Half>());
  at::Tensor _out_feats_8 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_8 = reinterpret_cast<half*>(_out_feats_8.data_ptr<at::Half>());
  at::Tensor _out_feats_4 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_4 = reinterpret_cast<half*>(_out_feats_4.data_ptr<at::Half>());
  at::Tensor _out_feats_2 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_2 = reinterpret_cast<half*>(_out_feats_2.data_ptr<at::Half>());
  at::Tensor _out_feats_1 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_1 = reinterpret_cast<half*>(_out_feats_1.data_ptr<at::Half>());
  at::Tensor _out_feats_new = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_new = reinterpret_cast<half*>(_out_feats_new.data_ptr<at::Half>());
  
  // 计算网格的第二个维度
  int grid_dim_y = 0;
  int bit_packed_oc_idx[7] = {0, 0, 0, 0, 0, 0, 0};

  if (num_16){
    grid_dim_y += 128;
    bit_packed_oc_idx[1] = bit_packed_oc_idx[0] + 128;
  }else
    bit_packed_oc_idx[1] = bit_packed_oc_idx[0];
  if (num_8){
    grid_dim_y += 32;
    bit_packed_oc_idx[2] = bit_packed_oc_idx[1] + 32;
  }else
    bit_packed_oc_idx[2] = bit_packed_oc_idx[1];
  if (num_4){
    grid_dim_y += 16;
    bit_packed_oc_idx[3] = bit_packed_oc_idx[2] + 16;
  }else
    bit_packed_oc_idx[3] = bit_packed_oc_idx[2];
  if (num_2){
    grid_dim_y += 8;
    bit_packed_oc_idx[4] = bit_packed_oc_idx[3] + 8;
  }else
    bit_packed_oc_idx[4] = bit_packed_oc_idx[3];
  if (num_1){
    grid_dim_y += 4;
    bit_packed_oc_idx[5] = bit_packed_oc_idx[4] + 4;
  }else
    bit_packed_oc_idx[5] = bit_packed_oc_idx[4];
  if (new_len){
    grid_dim_y += 128;
    bit_packed_oc_idx[6] = bit_packed_oc_idx[5] + 128;
  }else
    bit_packed_oc_idx[6] = bit_packed_oc_idx[5];
  grid_dim_y = (grid_dim_y + 3) / 4;

  cudaSetDevice(_in_feats.device().index());
  int bit_num[6] = {num_16, num_8, num_4, num_2, num_1, new_len};
  int quant_len = num_16 + num_8 + num_4 + num_2 + num_1;
  int *d_bit_num;
  int *d_bit_packed_oc_idx;
  cudaMalloc(&d_bit_num, 6 * sizeof(int));
  cudaMalloc(&d_bit_packed_oc_idx, 7 * sizeof(int));
  cudaMemcpy(d_bit_num, bit_num, 6 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bit_packed_oc_idx, bit_packed_oc_idx, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
  const int head_ratio = nh_q / nh_kv;

  // 定义网格尺寸
  dim3 num_blocks(BS, grid_dim_y);
  // 定义线程块尺寸, 这样划分的原因是一个warp有32个线程, 一个线程块有128个线程
  dim3 num_threads(32, 4);

  bgemv_kernel_wv_group_hq<<<num_blocks, num_threads>>>(
    in_feats, out_feats_16, out_feats_8, out_feats_4, out_feats_2, out_feats_1, out_feats_new, 
    kernel_16, kernel_16_idx, kernel_8, kernel_8_idx, scaling_factors_8, zeros_8,
    kernel_4, kernel_4_idx, scaling_factors_4, zeros_4, kernel_2, kernel_2_idx,
    scaling_factors_2, zeros_2, kernel_1, kernel_1_idx, std_1, means_1, 
    normal_quantiles_center, kernel_new, num_in_channels, quant_len, num_out_channels, 
    d_bit_num, d_bit_packed_oc_idx, group_size, head_ratio
  );

  cudaFree(d_bit_num);
  cudaFree(d_bit_packed_oc_idx);

  at::Tensor _out_feats = _out_feats_16 + _out_feats_8 + _out_feats_4 + _out_feats_2 + _out_feats_1 + _out_feats_new;
  return _out_feats;
}

torch::Tensor gemv_forward_cuda_wv_group_hq_seq(
  torch::Tensor _in_feats,
  c10::optional<torch::Tensor> _kernel_16,
  c10::optional<torch::Tensor> _kernel_8,
  c10::optional<torch::Tensor> _scaling_factors_8,
  c10::optional<torch::Tensor> _zeros_8,
  c10::optional<torch::Tensor> _kernel_4,
  c10::optional<torch::Tensor> _scaling_factors_4,
  c10::optional<torch::Tensor> _zeros_4,
  c10::optional<torch::Tensor> _kernel_2,
  c10::optional<torch::Tensor> _scaling_factors_2,
  c10::optional<torch::Tensor> _zeros_2,
  c10::optional<torch::Tensor> _kernel_1,
  c10::optional<torch::Tensor> _std_1,
  c10::optional<torch::Tensor> _means_1,
  torch::Tensor _normal_quantiles_center,
  c10::optional<torch::Tensor> _kernel_new,
  int num_out_channels,
  int num_16,
  int num_8,
  int num_4,
  int num_2,
  int num_1,
  int new_len,
  const int group_size,
  const int nh_q,
  const int nh_kv)
{
  int BS = _in_feats.size(0);
  int num_in_channels = _in_feats.size(2);
  half* in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());

  half* kernel_16 = _kernel_16.has_value() ? reinterpret_cast<half*>(_kernel_16.value().data_ptr<at::Half>()) : nullptr;
  
  uint32_t* kernel_8 = _kernel_8.has_value() ? reinterpret_cast<uint32_t*>(_kernel_8.value().data_ptr<int>()) : nullptr;
  half* scaling_factors_8 = _scaling_factors_8.has_value() ? reinterpret_cast<half*>(_scaling_factors_8.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_8 = _zeros_8.has_value() ? reinterpret_cast<half*>(_zeros_8.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_4 = _kernel_4.has_value() ? reinterpret_cast<uint32_t*>(_kernel_4.value().data_ptr<int>()) : nullptr;
  half* scaling_factors_4 = _scaling_factors_4.has_value() ? reinterpret_cast<half*>(_scaling_factors_4.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_4 = _zeros_4.has_value() ? reinterpret_cast<half*>(_zeros_4.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_2 = _kernel_2.has_value() ? reinterpret_cast<uint32_t*>(_kernel_2.value().data_ptr<int>()) : nullptr;
  half* scaling_factors_2 = _scaling_factors_2.has_value() ? reinterpret_cast<half*>(_scaling_factors_2.value().data_ptr<at::Half>()) : nullptr;
  half* zeros_2 = _zeros_2.has_value() ? reinterpret_cast<half*>(_zeros_2.value().data_ptr<at::Half>()) : nullptr;

  uint32_t* kernel_1 = _kernel_1.has_value() ? reinterpret_cast<uint32_t*>(_kernel_1.value().data_ptr<int>()) : nullptr;
  half* std_1 = _std_1.has_value() ? reinterpret_cast<half*>(_std_1.value().data_ptr<at::Half>()) : nullptr;
  half* means_1 = _means_1.has_value() ? reinterpret_cast<half*>(_means_1.value().data_ptr<at::Half>()) : nullptr;

  half* kernel_new = _kernel_new.has_value() ? reinterpret_cast<half*>(_kernel_new.value().data_ptr<at::Half>()) : nullptr;
  float* normal_quantiles_center = reinterpret_cast<float*>(_normal_quantiles_center.data_ptr<float>());

  torch::TensorOptions options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats_16 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_16 = reinterpret_cast<half*>(_out_feats_16.data_ptr<at::Half>());
  at::Tensor _out_feats_8 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_8 = reinterpret_cast<half*>(_out_feats_8.data_ptr<at::Half>());
  at::Tensor _out_feats_4 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_4 = reinterpret_cast<half*>(_out_feats_4.data_ptr<at::Half>());
  at::Tensor _out_feats_2 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_2 = reinterpret_cast<half*>(_out_feats_2.data_ptr<at::Half>());
  at::Tensor _out_feats_1 = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_1 = reinterpret_cast<half*>(_out_feats_1.data_ptr<at::Half>());
  at::Tensor _out_feats_new = torch::zeros({BS, 1, num_out_channels}, options);
  half* out_feats_new = reinterpret_cast<half*>(_out_feats_new.data_ptr<at::Half>());
  
  // 计算网格的第二个维度
  int grid_dim_y = 0;
  int bit_packed_oc_idx[7] = {0, 0, 0, 0, 0, 0, 0};

  if (num_16){
    grid_dim_y += 128;
    bit_packed_oc_idx[1] = bit_packed_oc_idx[0] + 128;
  }else
    bit_packed_oc_idx[1] = bit_packed_oc_idx[0];
  if (num_8){
    grid_dim_y += 32;
    bit_packed_oc_idx[2] = bit_packed_oc_idx[1] + 32;
  }else
    bit_packed_oc_idx[2] = bit_packed_oc_idx[1];
  if (num_4){
    grid_dim_y += 16;
    bit_packed_oc_idx[3] = bit_packed_oc_idx[2] + 16;
  }else
    bit_packed_oc_idx[3] = bit_packed_oc_idx[2];
  if (num_2){
    grid_dim_y += 8;
    bit_packed_oc_idx[4] = bit_packed_oc_idx[3] + 8;
  }else
    bit_packed_oc_idx[4] = bit_packed_oc_idx[3];
  if (num_1){
    grid_dim_y += 4;
    bit_packed_oc_idx[5] = bit_packed_oc_idx[4] + 4;
  }else
    bit_packed_oc_idx[5] = bit_packed_oc_idx[4];
  if (new_len){
    grid_dim_y += 128;
    bit_packed_oc_idx[6] = bit_packed_oc_idx[5] + 128;
  }else
    bit_packed_oc_idx[6] = bit_packed_oc_idx[5];
  grid_dim_y = (grid_dim_y + 3) / 4;

  cudaSetDevice(_in_feats.device().index());
  int bit_num[6] = {num_16, num_8, num_4, num_2, num_1, new_len};
  int *d_bit_num;
  int *d_bit_packed_oc_idx;
  cudaMalloc(&d_bit_num, 6 * sizeof(int));
  cudaMalloc(&d_bit_packed_oc_idx, 7 * sizeof(int));
  cudaMemcpy(d_bit_num, bit_num, 6 * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bit_packed_oc_idx, bit_packed_oc_idx, 7 * sizeof(int), cudaMemcpyHostToDevice);
  
  const int head_ratio = nh_q / nh_kv;

  // 定义网格尺寸
  dim3 num_blocks(BS, grid_dim_y);
  // 定义线程块尺寸, 这样划分的原因是一个warp有32个线程, 一个线程块有128个线程
  dim3 num_threads(32, 4);

  bgemv_kernel_wv_group_hq_seq<<<num_blocks, num_threads>>>(
    in_feats, out_feats_16, out_feats_8, out_feats_4, out_feats_2, out_feats_1, out_feats_new, 
    kernel_16, kernel_8, scaling_factors_8, zeros_8,
    kernel_4, scaling_factors_4, zeros_4, kernel_2, 
    scaling_factors_2, zeros_2, kernel_1, std_1, means_1, 
    normal_quantiles_center, kernel_new, num_in_channels, num_out_channels, 
    d_bit_num, d_bit_packed_oc_idx, group_size, head_ratio
  );

  cudaFree(d_bit_num);
  cudaFree(d_bit_packed_oc_idx);

  at::Tensor _out_feats = _out_feats_16 + _out_feats_8 + _out_feats_4 + _out_feats_2 + _out_feats_1 + _out_feats_new;
  return _out_feats;
}