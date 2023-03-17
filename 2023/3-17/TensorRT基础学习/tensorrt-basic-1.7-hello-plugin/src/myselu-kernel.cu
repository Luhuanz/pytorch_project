
#include <cuda_runtime.h>
#include <math.h>

static __device__ float sigmoid(float x){
    return 1 / (1 + expf(-x));
}

static __global__ void myselu_kernel(const float* x, float* output, int n){

    int position = threadIdx.x + blockDim.x * blockIdx.x;
    if(position >= n) return;

    output[position] = x[position] * sigmoid(x[position]);
}

void myselu_inference(const float* x, float* output, int n, cudaStream_t stream){

    const int nthreads = 512;
    int block_size = n < nthreads ? n : nthreads;
    int grid_size = (n + block_size - 1) / block_size;
    myselu_kernel<<<grid_size, block_size, 0, stream>>>(x, output, n);
}