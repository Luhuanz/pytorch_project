#include <iostream>

__global__ void vectorAdd(int* a, int* b, int* c, int len_vector){
    int i = threadIdx.x;
    if(i < len_vector){
        c[i] = a[i] + b[i];
    }
}


int main(){
    int len_vector = 32;
    int vector_1[len_vector], vector_2[len_vector], vector_3[len_vector];
    int *vector_1_gpu, *vector_2_gpu, *vector_3_gpu;

    // step 1: init vector
    for(int i =0; i <len_vector; ++i){
        vector_1[i] = 1;
        vector_2[i] = 2;
    }

    //print vector_1
    printf("vector_1:\n");
    for(int j=0; j<len_vector; ++j){
        printf("%d ", vector_1[j]);
    }

    //print vector_2
    printf("\nvector_2:\n");
    for(int k=0; k<len_vector; ++k){
        printf("%d ", vector_2[k]);
    }

    // step 2: copy cpu data to device
    cudaMalloc((void **)&vector_1_gpu, len_vector * sizeof(int));
    cudaMalloc((void **)&vector_2_gpu, len_vector * sizeof(int));
    cudaMalloc((void **)&vector_3_gpu, len_vector * sizeof(int));

    cudaMemcpy(vector_1_gpu, vector_1, len_vector * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_2_gpu, vector_2, len_vector * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vector_3_gpu, vector_3, len_vector * sizeof(int), cudaMemcpyHostToDevice);

    //step 3: run the kernel function
    int threadNum = len_vector;    // 设置核函数的 thread 数
    int blockNum = 1;    // 设置核函数的 block 数量
    vectorAdd<<<blockNum, threadNum>>>(vector_1_gpu, vector_2_gpu, vector_3_gpu, len_vector);

    // step 4: download result from device
    cudaMemcpy(vector_3, vector_3_gpu, len_vector * sizeof(int), cudaMemcpyDeviceToHost);

    //print vector_3
    printf("\nvector_3:\n");
    for(int z=0; z<len_vector; ++z){
        printf("%d ", vector_3[z]);
    }
    printf("\n");
    return 0;

}
