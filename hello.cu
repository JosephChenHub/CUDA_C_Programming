#include <cuda_runtime_api.h>
#include <stdio.h>

__global__ void hello() {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    printf("Hello world, thread:%d\n", tid);
}




int main(int argc, char* argv[]) {

    cudaSetDevice(0);

    hello<<<3, 10>>>();

    //cudaDeviceReset();
    cudaDeviceSynchronize();

    return 0;
}
