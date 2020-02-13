#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void Foo() {
    int tid_x = threadIdx.x + blockIdx.x * blockDim.x; 
    int tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    printf("Hello world, thread:(%d, %d) \n", tid_y, tid_x);
}

/// same code 
__global__ void Fuck(int width) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    int tid_x = tid % width;
    int tid_y = tid / width;
    printf("fuck the world, thread id: %d = (%d, %d) \n", tid, tid_y, tid_x);
}



int main(int argc, char* argv[]) {

    cudaSetDevice(0);
    dim3 block (3, 3);
    dim3 grid (2, 2);
    
    Foo<<<grid, block>>>();
    Fuck<<<4, 9>>>(6);

    //cudaDeviceReset();
    cudaDeviceSynchronize();

    return 0;
}
