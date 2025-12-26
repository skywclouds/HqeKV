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
__global__ void bgemv_kernel_uniform_8(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * IC;//同理
    const half* zeros = _zeros + _batch_idx * IC;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int scale_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < ICR){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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
__global__ void bgemv_kernel_uniform_4(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * IC;//同理
    const half* zeros = _zeros + _batch_idx * IC;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int scale_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < ICR){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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
__global__ void bgemv_kernel_uniform_2(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * IC;//同理
    const half* zeros = _zeros + _batch_idx * IC;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int scale_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < ICR){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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

__global__ void bgemv_kernel_uniform_1(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * IC;//同理
    const half* zeros = _zeros + _batch_idx * IC;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;//乘以4的原因是每个线程算4个数
      int scale_mn_offset = k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < ICR){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有pack_factor个量化的数,所以循环pack_factor次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个量化的数的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个量化的数的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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
Computes Batched 8-bit GEMV

Args:
  这个函数只在decoding阶段用到,所以inputs的行数是1
  inputs: vector of shape [BS, 1, IC];
  weight: matrix of shape [BS, OC // PACK_FACTOR, IC];
  output: vector of shape [BS, 1, OC];
  zeros: matrix of shape [BS, OC // group_size, IC];
  scaling_factors: matrix of shape [BS, OC // group_size, IC];

Notes:
  One cannot infer group_size from the shape of scaling factors.
  the second dimension is rounded up to a multiple of PACK_FACTOR.
*/
__global__ void bgemv_kernel_uniform_group_8(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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

__global__ void bgemv_kernel_uniform_group_4(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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

__global__ void bgemv_kernel_uniform_group_2(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int nh_q, const int nh_kv){
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
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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

__global__ void bgemv_kernel_uniform_group_1(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit,const int nh_q, const int nh_kv){
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
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
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;//同理
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;//同理
    // 把每128个数分为一组,如果IC超过了128就以128为一个组进行串行处理
    const int TILE_DIM = 128;
    const int num = 0xFF >> (8-bit);
    const int ICR = IC;
    float psum[pack_factor]{};
    for (int k=0; k < (IC + TILE_DIM - 1) / TILE_DIM; k++){
      uint32_t qw[4]{};
      half cscale[4]{};
      half czero[4]{};
      half inp[4]{};
      // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
      // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
      int weight_offset = packed_oc_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int scale_mn_offset = group_idx * ICR + k * TILE_DIM + threadIdx.x*4;
      int inputs_ptr_delta = k * TILE_DIM + threadIdx.x * 4; 
      for (int i=0; i<4; i++){
        if (weight_offset + i < OC * ICR / pack_factor)
          qw[i] = *(weight + weight_offset + i);
        if (scale_mn_offset + i < OC * ICR / group_size){
          cscale[i] = *(scaling_factors + scale_mn_offset + i);
          czero[i] = *(zeros + scale_mn_offset + i);}
        if (inputs_ptr_delta + i < ICR)
          inp[i] = *(inputs + inputs_ptr_delta + i);
      }
      #pragma unroll // 用于指示编译器展开循环
      // 取q中的4个数和K中的4个int32相乘相加
      for (int ic_0 = 0; ic_0 < 4; ic_0++){
        uint32_t cur_packed_weight =  qw[ic_0];
        float cur_inp = __half2float(inp[ic_0]);
        float cur_scale = __half2float(cscale[ic_0]);
        float cur_zero = __half2float(czero[ic_0]);
        // 一个int32中有8个int4,所以循环8次
        for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
          int oc_idx = oc_start_idx + ic_1;
          if (oc_idx < OC){
            // unpack数据
            float cur_single_weight_fp = (float)(cur_packed_weight & num);
            // 进行反量化
            float dequantized_weight = cur_scale * cur_single_weight_fp + cur_zero;
            // if(blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0 && threadIdx.y == 0 && k == 1) printf("%d %d %d %f %f %f %f %f\n", k, ic_0, ic_1, dequantized_weight, cur_single_weight_fp, cur_scale, cur_zero, cur_inp);
            // 进行右移,unpack下一个数
            cur_packed_weight = cur_packed_weight >> bit;
            // q中的第ic_0个数与K中的第ic_0个int32中的第ic_1个int4的相乘
            // 这样一个小循环结束之后实现了
            // q中的第ic_0个数与K中的第ic_0个int32中的8个int4的相乘
            psum[ic_1] += dequantized_weight * cur_inp;
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

__global__ void bgemv_kernel_qk_group_token_8(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int head_ratio){
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_IC = IC / pack_factor;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const half* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / head_ratio;
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int num = 0xFF >> (8-bit);
    float res = 0;
    if (oc_idx < OC){ // 检查行是否越界
      for (int k=0; k < (packed_IC + 3) / 4; k++){
        int inputs_start_offset = (k*4+threadIdx.x) * pack_factor;
        int group_idx = inputs_start_offset / group_size;
        int weight_offset = oc_idx * packed_IC + k * 4 + threadIdx.x;
        int scale_mn_offset = oc_idx * IC / group_size + group_idx;
        
        uint32_t cur_packed_weight = *(weight + weight_offset);
        float cur_scale = __half2float(*(scaling_factors+scale_mn_offset));
        float cur_zero = __half2float(*(zeros+scale_mn_offset));
        
        #pragma unroll
        for (int i=0; i < pack_factor; i++){
          float cur_inp = __half2float(*(inputs+inputs_start_offset+i));
          float cur_weight = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_weight + cur_zero;
          res += dequantized_weight * cur_inp;
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }

      res += __shfl_down_sync(0xffffffff, res, 2, 4);
      res += __shfl_down_sync(0xffffffff, res, 1, 4);
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(res); 
    }

}

__global__ void bgemv_kernel_qk_group_token_4(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int head_ratio){
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_IC = IC / pack_factor;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const half* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / head_ratio;
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int num = 0xFF >> (8-bit);
    float res = 0;
    if (oc_idx < OC){ // 检查行是否越界
      for (int k=0; k < (packed_IC + 3) / 4; k++){
        int inputs_start_offset = (k*4+threadIdx.x) * pack_factor;
        int group_idx = inputs_start_offset / group_size;
        int weight_offset = oc_idx * packed_IC + k * 4 + threadIdx.x;
        int scale_mn_offset = oc_idx * IC / group_size + group_idx;
        
        uint32_t cur_packed_weight = *(weight + weight_offset);
        float cur_scale = __half2float(*(scaling_factors+scale_mn_offset));
        float cur_zero = __half2float(*(zeros+scale_mn_offset));
        
        #pragma unroll
        for (int i=0; i < pack_factor; i++){
          float cur_inp = __half2float(*(inputs+inputs_start_offset+i));
          float cur_weight = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_weight + cur_zero;
          res += dequantized_weight * cur_inp;
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }

      res += __shfl_down_sync(0xffffffff, res, 2, 4);
      res += __shfl_down_sync(0xffffffff, res, 1, 4);
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(res); 
    }
}

__global__ void bgemv_kernel_qk_group_token_2(
  const half* _inputs, const uint32_t* _weight, const half* _zeros, const half* _scale, half* _outputs, 
  const int IC, const int OC, const int group_size, const int bit, const int head_ratio){
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int batch_idx = blockIdx.x;// 当前线程在第几个batch
    const int packed_IC = IC / pack_factor;
    const int oc_idx = blockIdx.y * blockDim.y + threadIdx.y; 
    const half* inputs = _inputs + batch_idx * IC;
    half* outputs = _outputs + batch_idx * OC;
    int _batch_idx = batch_idx / head_ratio;
    const uint32_t*  weight = _weight + _batch_idx * OC * IC / pack_factor;
    const half* scaling_factors = _scale + _batch_idx * OC * IC / group_size;
    const half* zeros = _zeros + _batch_idx * OC * IC / group_size;
    const int num = 0xFF >> (8-bit);
    float res = 0;
    if (oc_idx < OC){ // 检查行是否越界
      for (int k=0; k < (packed_IC + 3) / 4; k++){
        int inputs_start_offset = (k*4+threadIdx.x) * pack_factor;
        int group_idx = inputs_start_offset / group_size;
        int weight_offset = oc_idx * packed_IC + k * 4 + threadIdx.x;
        int scale_mn_offset = oc_idx * IC / group_size + group_idx;
        
        uint32_t cur_packed_weight = *(weight + weight_offset);
        float cur_scale = __half2float(*(scaling_factors+scale_mn_offset));
        float cur_zero = __half2float(*(zeros+scale_mn_offset));
        
        #pragma unroll
        for (int i=0; i < pack_factor; i++){
          float cur_inp = __half2float(*(inputs+inputs_start_offset+i));
          float cur_weight = (float)(cur_packed_weight & num);
          float dequantized_weight = cur_scale * cur_weight + cur_zero;
          res += dequantized_weight * cur_inp;
          cur_packed_weight = cur_packed_weight >> bit;
        }
      }

      res += __shfl_down_sync(0xffffffff, res, 2, 4);
      res += __shfl_down_sync(0xffffffff, res, 1, 4);
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(res); 
    }

}

/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC // PACK_Factor, IC];
  _zeros: int tensor of shape [OC // G, IC];
  _scaling_factors: tensor of shape [OC // G, IC];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;
Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda_uniform(
    torch::Tensor _in_feats,
    torch::Tensor _kernel,
    torch::Tensor _scaling_factors,
    torch::Tensor _zeros,
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
    auto zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
    auto scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());
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
      bgemv_kernel_uniform_8<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 4){
      bgemv_kernel_uniform_4<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 2){
      bgemv_kernel_uniform_2<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }else if(bit == 1){
      bgemv_kernel_uniform_1<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, bit, nh_q, nh_kv);
    }
    return _out_feats;
;}

__global__ void bgemv_kernel_qk_group_hq(
  const half* _in_feats,
  half* _outputs,
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
  const int OC,// num_out_channels
  const int* bit_num,
  const int* bit_packed_oc_idx,
  const int group_size,
  const int head_ratio)
{
  const int batch_idx = blockIdx.x;// 当前线程在第几个batch
  // 这里应该是batch_idx * _inputs的行数 * IC,但是_inputs只有一行,所以是batch_idx * IC
  const half* inputs = _in_feats + batch_idx * IC;
  half* outputs = _outputs + batch_idx * OC;// 同理这里也省略了_inputs的行数
  int _batch_idx = batch_idx / head_ratio;// q 和 kv 的头数不同
  const int row_idx = blockDim.y * blockIdx.y + threadIdx.y;// 当前线程在第几行
  // 获取当前线程处理哪个位数
  int cur_bit = 0;
  int bits[6] = {16, 8, 4, 2, 1, 100}; // 用100代表new_K
  int bit_idx = 0;
  for (bit_idx = 0; bit_idx < 6; bit_idx++)
    if(row_idx >= bit_packed_oc_idx[bit_idx] && row_idx < bit_packed_oc_idx[bit_idx+1])
      break;
  if (bit_idx < 6)
    cur_bit = bits[bit_idx];
  const int num = 0xFF >> (8-cur_bit);
  const int packed_oc_idx = row_idx - bit_packed_oc_idx[bit_idx];// 当前线程在该位数的第几行
  
  if (cur_bit == 16){// 16 bit 的情况
    const int oc_idx = packed_oc_idx;
    const half* weight = _weight_16 + _batch_idx * bit_num[0] * IC;
    float psum = 0;
    int weight_offset = oc_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x*4; 
    half inp[4]{};
    half w[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (half2&) w[0] = (half2&)weight[weight_offset];
    (half2&) w[2] = (half2&)weight[weight_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    for (int ic_0=0; ic_0<4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      float cur_weight =  __half2float(w[ic_0]);
      psum += cur_weight * cur_inp;
    }
    psum = warp_reduce_sum(psum);
    // 选择这32个线程中的第一个作为最终的结果
    if (threadIdx.x == 0) 
      outputs[_weight_16_idx[oc_idx]] = __float2half(psum); 
    
  }else if (cur_bit == 8){// 8 bit 的情况
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_8 + _batch_idx * bit_num[1] * IC / pack_factor;
    const half* scaling_factors = _scale_8 + _batch_idx * bit_num[1] * IC / group_size;//同理
    const half* zeros = _zeros_8 + _batch_idx * bit_num[1] * IC / group_size;//同理
    float psum[pack_factor]{};
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[_weight_8_idx[oc_idx]] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 4){// 4 bit 的情况
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_4 + _batch_idx * bit_num[2] * IC / pack_factor;
    const half* scaling_factors = _scale_4 + _batch_idx * bit_num[2] * IC / group_size;//同理
    const half* zeros = _zeros_4 + _batch_idx * bit_num[2] * IC / group_size;//同理
    float psum[pack_factor]{};
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    // 取q中的4个数和K中的4个int32相乘相加
    for (int ic_0 = 0; ic_0 < 4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      uint32_t cur_packed_weight = qw[ic_0];
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[_weight_4_idx[oc_idx]] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 2){// 2 bit 的情况
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_2 + _batch_idx * bit_num[3] * IC / pack_factor;
    const half* scaling_factors = _scale_2 + _batch_idx * bit_num[3] * IC / group_size;//同理
    const half* zeros = _zeros_2 + _batch_idx * bit_num[3] * IC / group_size;//同理
    float psum[pack_factor]{};
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs[_weight_2_idx[oc_idx]] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 1){// 1 bit 的情况
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_1 + _batch_idx * bit_num[4] * IC / pack_factor;
    const half* stds = _std_1 + _batch_idx * bit_num[4] * IC / group_size;//同理
    const half* means = _means_1 + _batch_idx * bit_num[4] * IC / group_size;//同理
    float psum[pack_factor]{};
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int std_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cstd[4]{};
    half cmean[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cstd[0] = (half2&)stds[std_mn_offset];
    (half2&) cstd[2] = (half2&)stds[std_mn_offset + 2];
    (half2&) cmean[0] = (half2&)means[std_mn_offset];
    (half2&) cmean[2] = (half2&)means[std_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    // 取q中的4个数和K中的4个int32相乘相加
    for (int ic_0 = 0; ic_0 < 4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      uint32_t cur_packed_weight = qw[ic_0];
      float cur_std = __half2float(cstd[ic_0]);
      float cur_mean = __half2float(cmean[ic_0]);
      // 一个int32中有4个int8,所以循环4次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int cur_single_weight_fp = cur_packed_weight & num;
        float center = *(_normal_quantiles_center + cur_single_weight_fp);
        float dequantized_weight = cur_std * center + cur_mean;
        cur_packed_weight = cur_packed_weight >> cur_bit;
        psum[ic_1] += dequantized_weight * cur_inp;
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[_weight_1_idx[oc_idx]] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 100){// new_K 的情况
    const half* weight = _weight_new + _batch_idx * bit_num[5] * IC;
    float psum = 0;
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x*4; 
    #pragma unroll // 用于指示编译器展开循环
    for (int ic_0=0; ic_0<4 && inputs_ptr_delta + ic_0 < IC; ic_0++){
      float cur_inp = __half2float(*(inputs + inputs_ptr_delta + ic_0));
      float cur_weight =  __half2float(*(weight + weight_offset + ic_0));
      psum += cur_weight * cur_inp;
    }
    const int oc_idx = bit_num[0]+bit_num[1]+bit_num[2]+bit_num[3]+bit_num[4]+packed_oc_idx;
    psum = warp_reduce_sum(psum);
    // 选择这32个线程中的第一个作为最终的结果
    if (threadIdx.x == 0) 
      outputs[oc_idx] = __float2half(psum);
  }
  
}

__global__ void bgemv_kernel_qk_group_hq_seq(
  const half* _in_feats,
  half* _outputs,
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
  half* outputs = _outputs + batch_idx * OC;// 同理这里也省略了_inputs的行数
  int _batch_idx = batch_idx / head_ratio;// q 和 kv 的头数不同
  const int row_idx = blockDim.y * blockIdx.y + threadIdx.y;// 当前线程在第几行
  // 获取当前线程处理哪个位数
  int cur_bit = 0;
  int bits[6] = {16, 8, 4, 2, 1, 100}; // 用100代表new_K
  int bit_idx = 0;
  for (bit_idx = 0; bit_idx < 6; bit_idx++)
    if(row_idx >= bit_packed_oc_idx[bit_idx] && row_idx < bit_packed_oc_idx[bit_idx+1])
      break;
  if (bit_idx < 6)
    cur_bit = bits[bit_idx];
  const int num = 0xFF >> (8-cur_bit);
  const int packed_oc_idx = row_idx - bit_packed_oc_idx[bit_idx];// 当前线程在该位数的第几行

  if (cur_bit == 16){// 16 bit 的情况
    const int oc_idx = packed_oc_idx;
    const half* weight = _weight_16 + _batch_idx * bit_num[0] * IC;
    float psum = 0;
    int weight_offset = oc_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x*4; 
    half inp[4]{};
    half w[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (half2&) w[0] = (half2&)weight[weight_offset];
    (half2&) w[2] = (half2&)weight[weight_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    for (int ic_0=0; ic_0<4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      float cur_weight =  __half2float(w[ic_0]);
      psum += cur_weight * cur_inp;
    }
    psum = warp_reduce_sum(psum);
    // 选择这32个线程中的第一个作为最终的结果
    if (threadIdx.x == 0) 
      outputs[oc_idx] = __float2half(psum); 
    
  }else if (cur_bit == 8){// 8 bit 的情况
    const int pack_factor = 32 / 8;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_8 + _batch_idx * bit_num[1] * IC / pack_factor;
    const half* scaling_factors = _scale_8 + _batch_idx * bit_num[1] * IC / group_size;//同理
    const half* zeros = _zeros_8 + _batch_idx * bit_num[1] * IC / group_size;//同理
    float psum[pack_factor]{};
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    // 取q中的4个数和K中的4个int32相乘相加
    for (int ic_0 = 0; ic_0 < 4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      uint32_t cur_packed_weight =  qw[ic_0];
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = bit_num[0] + oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 4){// 4 bit 的情况
    const int pack_factor = 32 / 4;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_4 + _batch_idx * bit_num[2] * IC / pack_factor;
    const half* scaling_factors = _scale_4 + _batch_idx * bit_num[2] * IC / group_size;//同理
    const half* zeros = _zeros_4 + _batch_idx * bit_num[2] * IC / group_size;//同理
    float psum[pack_factor]{};
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    // 取q中的4个数和K中的4个int32相乘相加
    for (int ic_0 = 0; ic_0 < 4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      uint32_t cur_packed_weight =  qw[ic_0];
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = bit_num[0] + bit_num[1] + oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 2){// 2 bit 的情况
    const int pack_factor = 32 / 2;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_2 + _batch_idx * bit_num[3] * IC / pack_factor;
    const half* scaling_factors = _scale_2 + _batch_idx * bit_num[3] * IC / group_size;//同理
    const half* zeros = _zeros_2 + _batch_idx * bit_num[3] * IC / group_size;//同理
    float psum[pack_factor]{};
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int scale_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cscale[4]{};
    half czero[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cscale[0] = (half2&)scaling_factors[scale_mn_offset];
    (half2&) cscale[2] = (half2&)scaling_factors[scale_mn_offset + 2];
    (half2&) czero[0] = (half2&)zeros[scale_mn_offset];
    (half2&) czero[2] = (half2&)zeros[scale_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
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
    for (int i=0; i < pack_factor; i++){
      int oc_idx = bit_num[0]+bit_num[1]+bit_num[2]+oc_start_idx + i;
      psum[i] = warp_reduce_sum(psum[i]);
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 1){// 1 bit 的情况
    const int pack_factor = 32 / 1;// 一个int32存储pack_factor个量化后的数
    const int oc_start_idx = packed_oc_idx * pack_factor;
    const int group_idx = oc_start_idx / group_size; //当前线程在第几组
    const uint32_t*  weight = _weight_1 + _batch_idx * bit_num[4] * IC / pack_factor;
    const half* stds = _std_1 + _batch_idx * bit_num[4] * IC / group_size;//同理
    const half* means = _means_1 + _batch_idx * bit_num[4] * IC / group_size;//同理
    float psum[pack_factor]{};
    
    // 因为每个线程算4个数,所以要*4,因为128个数为一组,同时128/4=32正好是线程块的
    // 第一个维度也就是32个线程,又因为每个线程算每组的4个数,这样32个线程就能恰好算每组的128个数
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int std_mn_offset = group_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x * 4; 
    half inp[4]{};
    uint32_t qw[4]{};
    half cstd[4]{};
    half cmean[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (uint4&) qw = (uint4&)weight[weight_offset];
    (half2&) cstd[0] = (half2&)stds[std_mn_offset];
    (half2&) cstd[2] = (half2&)stds[std_mn_offset + 2];
    (half2&) cmean[0] = (half2&)means[std_mn_offset];
    (half2&) cmean[2] = (half2&)means[std_mn_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    // 取q中的4个数和K中的4个int32相乘相加
    for (int ic_0 = 0; ic_0 < 4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      uint32_t cur_packed_weight =  qw[ic_0];
      float cur_std = __half2float(cstd[ic_0]);
      float cur_mean = __half2float(cmean[ic_0]);
      // 一个int32中有4个int8,所以循环4次
      for (int ic_1 = 0; ic_1 < pack_factor; ic_1++){
        int cur_single_weight_fp = cur_packed_weight & num;
        float center = *(_normal_quantiles_center + cur_single_weight_fp);
        float dequantized_weight = cur_std * center + cur_mean;
        cur_packed_weight = cur_packed_weight >> cur_bit;
        psum[ic_1] += dequantized_weight * cur_inp;
      }
    }
    for (int i=0; i < pack_factor; i++){
      int oc_idx = bit_num[0]+bit_num[1]+bit_num[2]+bit_num[3]+oc_start_idx + i;
      // 实现32个线程结果的求和
      // 因为128个数为一组,每次小循环算4个数,128/4刚好为32
      psum[i] = warp_reduce_sum(psum[i]);
      // 选择这32个线程中的第一个作为最终的结果
      if (threadIdx.x == 0) 
        outputs[oc_idx] = __float2half(psum[i]); 
    }
  }else if (cur_bit == 100){// new_K 的情况
    const half* weight = _weight_new + _batch_idx * bit_num[5] * IC;
    float psum = 0;
    int weight_offset = packed_oc_idx * IC + threadIdx.x*4;
    int inputs_ptr_delta = threadIdx.x*4; 
    half inp[4]{};
    half w[4]{};
    (half2&) inp[0] = (half2&)inputs[inputs_ptr_delta];
    (half2&) inp[2] = (half2&)inputs[inputs_ptr_delta + 2];
    (half2&) w[0] = (half2&)weight[weight_offset];
    (half2&) w[2] = (half2&)weight[weight_offset + 2];
    #pragma unroll // 用于指示编译器展开循环
    for (int ic_0=0; ic_0<4; ic_0++){
      float cur_inp = __half2float(inp[ic_0]);
      float cur_weight =  __half2float(w[ic_0]);
      psum += cur_weight * cur_inp;
    }
    const int oc_idx = bit_num[0]+bit_num[1]+bit_num[2]+bit_num[3]+bit_num[4]+packed_oc_idx;
    psum = warp_reduce_sum(psum);
    // 选择这32个线程中的第一个作为最终的结果
    if (threadIdx.x == 0) 
      outputs[oc_idx] = __float2half(psum);
  }
}

/*
Computes GEMV (PyTorch interface).

Args:
  _in_feats: tensor of shape [B, IC];
  _kernel: int tensor of shape [OC // PACK_Factor, IC];
  _zeros: int tensor of shape [OC // G, IC];
  _scaling_factors: tensor of shape [OC // G, IC];
  blockDim_x: size of thread block, dimension x, where blockDim_x * workload_per_thread = IC;
  blockDim_y: size of thread block, dimension y, where blockDim_y * gridDim_y = OC;
Returns:
  out_feats: tensor of shape [B, OC];
*/
torch::Tensor gemv_forward_cuda_uniform_group(
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
    bgemv_kernel_uniform_group_8<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 4){
    bgemv_kernel_uniform_group_4<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 2){
    bgemv_kernel_uniform_group_2<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
    );
  }else if (bit == 1){
    bgemv_kernel_uniform_group_1<<<num_blocks, num_threads>>>(
      // pointers
      in_feats, kernel, zeros, scaling_factors, out_feats,
      // constants
      num_in_channels, num_out_channels, group_size, bit, nh_q, nh_kv
    );
  }
    
  return _out_feats;
;}

torch::Tensor gemv_forward_cuda_qk_uniform_group_token(
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
  int num_in_channels = _in_feats.size(2);
  int num_out_channels = _kernel.size(1);
  const int head_ratio = nh_q / nh_kv;

  half* in_feats = reinterpret_cast<half*>(_in_feats.data_ptr<at::Half>());
  uint32_t* kernel = reinterpret_cast<uint32_t*>(_kernel.data_ptr<int>());
  half* zeros = reinterpret_cast<half*>(_zeros.data_ptr<at::Half>());
  half* scaling_factors = reinterpret_cast<half*>(_scaling_factors.data_ptr<at::Half>());

  torch::TensorOptions options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({BS, 1, num_out_channels}, options);
  half* out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

  dim3 num_blocks(BS, (num_out_channels + 31) / 32);
  dim3 num_threads(4, 32);

  if (bit == 8){
    bgemv_kernel_qk_group_token_8<<<num_blocks, num_threads>>>(
      in_feats, kernel, zeros, scaling_factors, out_feats,
      num_in_channels, num_out_channels, group_size, bit, head_ratio
    );
  }else if (bit == 4){
    bgemv_kernel_qk_group_token_4<<<num_blocks, num_threads>>>(
      in_feats, kernel, zeros, scaling_factors, out_feats,
      num_in_channels, num_out_channels, group_size, bit, head_ratio
    );
  }else if(bit == 2){
    bgemv_kernel_qk_group_token_2<<<num_blocks, num_threads>>>(
      in_feats, kernel, zeros, scaling_factors, out_feats,
      num_in_channels, num_out_channels, group_size, bit, head_ratio
    );
  }

  return _out_feats;

}

torch::Tensor gemv_forward_cuda_qk_group_hq(
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

  float* normal_quantiles_center = reinterpret_cast<float*>(_normal_quantiles_center.data_ptr<float>());
  half* kernel_new = _kernel_new.has_value() ? reinterpret_cast<half*>(_kernel_new.value().data_ptr<at::Half>()) : nullptr;

  torch::TensorOptions options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({BS, 1, num_out_channels}, options);
  half* out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());
  
  // 计算网格的第二个维度
  int grid_dim_y = 0;
  int bit_packed_oc_idx[7] = {0, 0, 0, 0, 0, 0, 0};

  int dim_y_add = num_16;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[1] = bit_packed_oc_idx[0] + dim_y_add;

  dim_y_add = num_8 / 4;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[2] = bit_packed_oc_idx[1] + dim_y_add;

  dim_y_add = num_4 / 8;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[3] = bit_packed_oc_idx[2] + dim_y_add;

  dim_y_add = num_2 / 16;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[4] = bit_packed_oc_idx[3] + dim_y_add;

  dim_y_add = num_1 / 32;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[5] = bit_packed_oc_idx[4] + dim_y_add;

  dim_y_add = new_len;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[6] = bit_packed_oc_idx[5] + dim_y_add;

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

  bgemv_kernel_qk_group_hq<<<num_blocks, num_threads>>>(
    in_feats, out_feats, kernel_16, kernel_16_idx, kernel_8, kernel_8_idx, 
    scaling_factors_8, zeros_8, kernel_4, kernel_4_idx, scaling_factors_4, 
    zeros_4, kernel_2, kernel_2_idx, scaling_factors_2, zeros_2, kernel_1, 
    kernel_1_idx, std_1, means_1, normal_quantiles_center, kernel_new, num_in_channels, 
    num_out_channels, d_bit_num, d_bit_packed_oc_idx, group_size, head_ratio
  );

  cudaFree(d_bit_num);
  cudaFree(d_bit_packed_oc_idx);

  return _out_feats;
}

torch::Tensor gemv_forward_cuda_qk_group_hq_seq(
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

  float* normal_quantiles_center = reinterpret_cast<float*>(_normal_quantiles_center.data_ptr<float>());
  half* kernel_new = _kernel_new.has_value() ? reinterpret_cast<half*>(_kernel_new.value().data_ptr<at::Half>()) : nullptr;

  torch::TensorOptions options =
  torch::TensorOptions().dtype(_in_feats.dtype()).device(_in_feats.device());
  at::Tensor _out_feats = torch::empty({BS, 1, num_out_channels}, options);
  half* out_feats = reinterpret_cast<half*>(_out_feats.data_ptr<at::Half>());

  // 计算网格的第二个维度
  int grid_dim_y = 0;
  int bit_packed_oc_idx[7] = {0, 0, 0, 0, 0, 0, 0};

  int dim_y_add = num_16;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[1] = bit_packed_oc_idx[0] + dim_y_add;

  dim_y_add = num_8 / 4;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[2] = bit_packed_oc_idx[1] + dim_y_add;

  dim_y_add = num_4 / 8;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[3] = bit_packed_oc_idx[2] + dim_y_add;

  dim_y_add = num_2 / 16;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[4] = bit_packed_oc_idx[3] + dim_y_add;

  dim_y_add = num_1 / 32;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[5] = bit_packed_oc_idx[4] + dim_y_add;

  dim_y_add = new_len;
  grid_dim_y += dim_y_add;
  bit_packed_oc_idx[6] = bit_packed_oc_idx[5] + dim_y_add;

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

  bgemv_kernel_qk_group_hq_seq<<<num_blocks, num_threads>>>(
    in_feats, out_feats, kernel_16, kernel_8,
    scaling_factors_8, zeros_8, kernel_4, scaling_factors_4, 
    zeros_4, kernel_2, scaling_factors_2, zeros_2, kernel_1, 
    std_1, means_1, normal_quantiles_center, kernel_new, num_in_channels, 
    num_out_channels, d_bit_num, d_bit_packed_oc_idx, group_size, head_ratio
  );

  cudaFree(d_bit_num);
  cudaFree(d_bit_packed_oc_idx);

  return _out_feats;

}