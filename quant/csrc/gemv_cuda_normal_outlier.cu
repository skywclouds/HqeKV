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

/*
Computes Batched 8-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, 1, IC];
  scaling_factors: matrix of shape [BS, 1, IC];

Notes:
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_normalize_outlier_8(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * IC;//同理
    const half* means = _means + _batch_idx * IC;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          // oc_idx 就是当前对应的 V 的列号
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

/*
Computes Batched 4-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, 1, IC];
  scaling_factors: matrix of shape [BS, 1, IC];

Notes:
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_normalize_outlier_4(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * IC;//同理
    const half* means = _means + _batch_idx * IC;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

/*
Computes Batched 2-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, 1, IC];
  scaling_factors: matrix of shape [BS, 1, IC];

Notes:
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_normalize_outlier_2(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * IC;//同理
    const half* means = _means + _batch_idx * IC;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

/*
Computes Batched 1-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, 1, IC];
  scaling_factors: matrix of shape [BS, 1, IC];

Notes:
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_normalize_outlier_1(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * IC;//同理
    const half* means = _means + _batch_idx * IC;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

__global__ void bgemv_kernel_normalize_group_outlier_8(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
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
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

/*
Computes Batched 4-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  means: matrix of shape [BS, 1, IC];
  stds: matrix of shape [BS, 1, IC];

Notes:
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_normalize_group_outlier_4(
    const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
    half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
    const int group_size, const int bit, const int nh_q, const int nh_kv){
      const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
      const int batch_idx = blockIdx.x;// 当前线程在第几个batch
      const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
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
      // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
      const int TILE_DIM = 128;
      const int num = 0xFF >> (8-bit);
      const int ICR = IC;
      float psum[pack_factor]{};
      for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
        uint32_t qw[4]{};
        half cstd[4]{};
        half cmean[4]{};
        half inp[4]{};
        // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
        // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
        int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
        int std_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
        int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
        inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
        for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
          inp[i] = *(inputs + inputs_ptr_delta + i);
          qw[i] = *(weight + weight_offset + i);
          cstd[i] = *(stds + std_mn_offset + i);
          cmean[i] = *(means + std_mn_offset + i);
        }
        #pragma unroll // 用于指示编译器展开循环
        // 取q中的4个数和K中的4个int32相乘相加
        for (int ic_0 = 0; ic_0 < 4; ic_0++){
          uint32_t cur_packed_weight =  qw[ic_0];
          float cur_inp = __half2float(inp[ic_0]);
          float cur_std = __half2float(cstd[ic_0]);
          float cur_mean = __half2float(cmean[ic_0]);
          // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
          for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
            int oc_idx = oc_start_idx + ic_1;
            if (oc_idx < OC){
              // 判断当前位置是否是离群值           
              bool is_outlier = false;
              int row;
              if (outliers_offset + ic_0 < ICR){// 检查列是否越界
                for (row = 0; row < 2; row++){
                  if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                    is_outlier = true;
                    break;
                  }
                }
              }
              if (is_outlier){
                // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
                float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
                psum[ic_1] += outlier * cur_inp;
              }else{
                // unpack数据
                int cur_single_weight_fp = cur_packed_weight & num;
                float center = *(_normal_quantiles_center + cur_single_weight_fp);
                // 进行反量化
                float dequantized_weight = cur_std * center + cur_mean;
                // 进行右移,unpack下一个数
                
                // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
                // 这样一个小循环结束之后实现了
                // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
                psum[ic_1] += dequantized_weight * cur_inp;
              }
              cur_packed_weight = cur_packed_weight >> bit;
            }
          }
        }
      }
      for (int i=0; i < pack_factor; i++){
        int oc_idx = oc_start_idx + i;
        if (oc_idx < OC){
          // 实现32个线程结果的求和
          // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
          psum[i] = warp_reduce_sum(psum[i]);
          // 选择这32个线程中的第一个作为最终的结果
          if (threadIdx.x == 0) 
            outputs[oc_idx] = __float2half(psum[i]); 
        }
      }
  }

__global__ void bgemv_kernel_normalize_group_outlier_2(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
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
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // 进行右移,unpack下一个数
              
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

__global__ void bgemv_kernel_normalize_group_outlier_1(
  const half* _inputs, const uint32_t* _weight, const half* _means, const half* _std, const float* _normal_quantiles_center, 
  half* _outliers, int64_t* _outliers_idx, half* _outputs, const int IC, const int OC, 
  const int group_size, const int bit, const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
    const half* inputs = _inputs + batch_idx * IC;
    // 同理这里也省略了_inputs的行数
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx  = batch_idx / (nh_q / nh_kv);
    // 定位到当前batch
    // 一个batch占OC * IC / pack_factor的空间,所以要偏移_batch_idx * OC * IC / pack_factor
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* stds = _std + _batch_idx * OC * IC / group_size;//同理
    const half* means = _means + _batch_idx * OC * IC / group_size;//同理
    const half* outliers = _outliers + _batch_idx * IC * 2;//同理
    const int64_t* outliers_idx = _outliers_idx + _batch_idx * IC * 2;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cstd[4]{};
      half cmean[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int std_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta, outliers_offset;// 因为q只有一行,outlier只有两行,所以他俩的偏移量一样
      inputs_ptr_delta = outliers_offset = k * TILE_DIM + threadIdx.x*4; 
      for (int i=0; i<4 && inputs_ptr_delta + i < ICR; i++){
        inp[i] = *(inputs + inputs_ptr_delta + i);
        qw[i] = *(weight + weight_offset + i);
        cstd[i] = *(stds + std_mn_offset + i);
        cmean[i] = *(means + std_mn_offset + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_std = __half2float(cstd[ic_0]);
        float cur_mean = __half2float(cmean[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // 判断当前位置是否是离群值           
            bool is_outlier = false;
            int row;
            if (outliers_offset + ic_0 < ICR){// 检查列是否越界
              for (row = 0; row < 2; row++){
                if(oc_idx == *(outliers_idx + outliers_offset + ic_0 + row * ICR)){
                  is_outlier = true;
                  break;
                }
              }
            }
            if (is_outlier){
              // 如果列越界了那么is_outlier肯定是false,所以如何执行到这里列肯定没越界
              float outlier = __half2float(*(outliers + outliers_offset + ic_0 + row * ICR));
              psum[ic_1] += outlier * cur_inp;
            }else{
              // unpack数据
              int cur_single_weight_fp = cur_packed_weight & num;
              float center = *(_normal_quantiles_center + cur_single_weight_fp);
              // 进行反量化
              float dequantized_weight = cur_std * center + cur_mean;
              // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
              // 这样一个小循环结束之后实现了
              // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
              psum[ic_1] += dequantized_weight * cur_inp;
            }
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
          }
        }
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      if (oc_idx < OC){
        // 实现32个线程结果的求和
        // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
        psum[i] = warp_reduce_sum(psum[i]);
        // 选择这32个线程中的第一个作为最终的结果
        if (threadIdx.x == 0) 
          outputs[oc_idx] = __float2half(psum[i]); 
      }
    }
}

torch::Tensor gemv_forward_cuda_normalize_outlier(
  torch::Tensor _in_feats,
  torch::Tensor _kernel,
  torch::Tensor _stds,
  torch::Tensor _means,
  torch::Tensor _normal_quantiles_center,
  torch::Tensor _outliers,
  torch::Tensor _outliers_idx,
  const int bit,
  const int nh_q,
  const int nh_kv)
{
    int pack_factor = 32 / bit;
    int BS = _in_feats.size(0);
    int num_in_feats = _in_feats.size(1);
    int num_in_channels = _in_feats.size(2);
    int num_out_channels = _kernel.size(1) * pack_factor;
    // int kernel_volume = _out_in_map.size(1);
    // 将tensor转换为指针
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
    at::Tensor _out_feats = torch::empty({BS, num_in_feats, num_out_channels}, options);
    int num_out_feats = _out_feats.size(-2);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    // 定义网格尺寸
    dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4);
    // 定义线程块尺寸
    dim3 num_threads(32, 4);
      // note: in this case, pack factor == 8
    if(bit == 8){
      bgemv_kernel_normalize_outlier_8<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 4){
      bgemv_kernel_normalize_outlier_4<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 2){
      bgemv_kernel_normalize_outlier_2<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 1){
      bgemv_kernel_normalize_outlier_1<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }
    
    
    return _out_feats;
;}


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
torch::Tensor gemv_forward_cuda_normalize_group_outlier(
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
    at::Tensor _out_feats = torch::empty({BS, num_in_feats, num_out_channels}, options);
    int num_out_feats = _out_feats.size(-2);
    auto out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
    int pack_factor = 32 / bit;
    // 定义网格尺寸
    dim3 num_blocks(BS, (num_out_channels / pack_factor + 3) / 4);
    // 定义线程块尺寸, 这样划分的原因是一个warp有32个线程, 一个线程块有128个线程
    dim3 num_threads(32, 4);
    // 进行核函数计算
    if (bit == 8){
      bgemv_kernel_normalize_group_outlier_8<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
      );
    }
    else if (bit == 4){
        bgemv_kernel_normalize_group_outlier_4<<<num_blocks, num_threads>>>(
        // pointers
        in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
        // constants
        num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
        );
    }
    else if (bit == 2){
      bgemv_kernel_normalize_group_outlier_2<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
      );
    }
    else if (bit == 1){
      bgemv_kernel_normalize_group_outlier_1<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, means, stds, normal_quantiles_center, outliers, outliers_idx, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
      );
    }
        
    return _out_feats;
;}