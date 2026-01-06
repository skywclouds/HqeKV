inline __device__ void reduceBlock(float *sData) {
    unsigned int tid = threadIdx.x;
    unsigned int s = blockDim.x / 2;
    while (s > 0) {
        if (tid < s) {
            sData[tid] += sData[tid + s];
        }
        __syncthreads();
        s >>= 1;
    }
  }