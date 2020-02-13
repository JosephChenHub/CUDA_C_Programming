#include <stdio.h>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <time.h>


#define DIMX 512

#define cudaCheck(e) do { \
    if (cudaSuccess != (e)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);


template <typename DType>
__global__ void reduceGmem(DType* out, DType* in, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx >= n) return;
    

    int tid = threadIdx.x; 
    DType* idata = in + blockIdx.x * blockDim.x; 

    if(blockDim.x >= 1024 && tid < 512 ) idata[tid] += idata[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256 ) idata[tid] += idata[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128 ) idata[tid] += idata[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64 ) idata[tid] += idata[tid + 64];
    __syncthreads();

    if(tid < 32) {
        volatile DType* vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if(tid == 0) {
        out[blockIdx.x] = idata[0];
        //printf("ID:%d, sum:%5f\n", blockIdx.x, idata[0]);
    }
}


template <typename DType>
__global__ void reduceSmem(DType* out, DType* in, size_t n) {
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ DType smem[DIMX];
    int tid = threadIdx.x; 
    DType* idata = in + blockIdx.x * blockDim.x; 
    /// global mem. -> shared mem.
    if(idx < n) smem[tid] = idata[tid];
    else smem[tid] = 0;
    __syncthreads();


    if(blockDim.x >= 1024 && tid < 512 ) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256 ) smem[tid] += smem[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128 ) smem[tid] += smem[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64 ) smem[tid] += smem[tid + 64];
    __syncthreads();

    if(tid < 32) {
        volatile DType* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if(tid == 0) {
        out[blockIdx.x] = smem[0];
        //printf("ID:%d, sum:%5f\n", blockIdx.x, idata[0]);
    }
}

template <typename DType>
__global__ void reduceSmemUnroll(DType* out, DType* in, size_t n) {

    __shared__ DType smem[DIMX];
    int tid = threadIdx.x; 
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    /// global mem. -> shared mem.
    DType tmp_sum = 0;
    if(idx + 3 * blockDim.x  < n) {
        DType a1 = in[idx];
        DType a2 = in[idx + blockDim.x];
        DType a3 = in[idx + blockDim.x*2];
        DType a4 = in[idx + blockDim.x*3];
        tmp_sum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmp_sum;
    __syncthreads();

    if(blockDim.x >= 1024 && tid < 512 ) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256 ) smem[tid] += smem[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128 ) smem[tid] += smem[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64 ) smem[tid] += smem[tid + 64];
    __syncthreads();

    if(tid < 32) {
        volatile DType* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if(tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}


template <typename DType>
__global__ void reduceSmemUnrollDynamic(DType* out, DType* in, size_t n) {

    extern __shared__ DType smem[]; //! dynamic shared memory 
    int tid = threadIdx.x; 
    size_t idx = threadIdx.x + blockIdx.x * blockDim.x * 4;
    /// global mem. -> shared mem.
    DType tmp_sum = 0;
    if(idx + 3 * blockDim.x  < n) {
        DType a1 = in[idx];
        DType a2 = in[idx + blockDim.x];
        DType a3 = in[idx + blockDim.x*2];
        DType a4 = in[idx + blockDim.x*3];
        tmp_sum = a1 + a2 + a3 + a4;
    }
    smem[tid] = tmp_sum;
    __syncthreads();

    if(blockDim.x >= 1024 && tid < 512 ) smem[tid] += smem[tid+512];
    __syncthreads();
    if(blockDim.x >= 512 && tid < 256 ) smem[tid] += smem[tid + 256];
    __syncthreads();
    if(blockDim.x >= 256 && tid < 128 ) smem[tid] += smem[tid + 128];
    __syncthreads();
    if(blockDim.x >= 128 && tid < 64 ) smem[tid] += smem[tid + 64];
    __syncthreads();

    if(tid < 32) {
        volatile DType* vsmem = smem;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if(tid == 0) {
        out[blockIdx.x] = smem[0];
    }
}

int main(int argc, char* agv[]) {

    srand(time(NULL));
    cudaStream_t stream[2]; 
    cudaCheck(cudaSetDevice(0)); //! CUDA Streams
    for(int i = 0; i < 2; ++i) cudaCheck(cudaStreamCreate(&stream[i]));

    cudaProfilerStart();
    void * buffers[7];
    const size_t N = 1 << 24;
    float * pdata = new float[N];
    float res = 0;
    double res_check = 0;
    for(size_t i = 0; i < N; ++i) {
        //pdata[i] = 1;
        pdata[i] = rand() / double(RAND_MAX) * 0.5;
        res_check += pdata[i];
    }

    const int threads_per_block = DIMX;
    const int num_blocks = (N + threads_per_block - 1) / threads_per_block;
    const int num_blocks2 = (num_blocks + threads_per_block - 1) / threads_per_block;
    printf("threads_per_block:%d, num_blocks:%d, %d\n", threads_per_block, num_blocks, num_blocks2);

    /// allocate gpu mem.
    cudaCheck(cudaMalloc(&buffers[0], sizeof(float)*num_blocks*threads_per_block));
    cudaCheck(cudaMalloc(&buffers[1], sizeof(float)*num_blocks2 * threads_per_block));
    cudaCheck(cudaMalloc(&buffers[2], sizeof(float)*num_blocks2));
    cudaCheck(cudaMalloc(&buffers[3], sizeof(float)*num_blocks*threads_per_block));
    cudaCheck(cudaMalloc(&buffers[4], sizeof(float)*num_blocks2*threads_per_block));
    cudaCheck(cudaMalloc(&buffers[5], sizeof(float)*num_blocks2));
    cudaCheck(cudaMalloc(&buffers[6], sizeof(float)*4));
    
    /// pinned memory
    float * c_buffer;
    cudaCheck(cudaMallocHost(&c_buffer,  sizeof(float)*N));
    double cpu_res = 0.;
    for(size_t i = 0 ; i < N; ++i) {
        c_buffer[i] = rand() / double(RAND_MAX) * 0.1;
        cpu_res += c_buffer[i];
    }
    printf("Starting reduction ...");

    /// cpu mem. -> gpu mem. 
    cudaCheck(cudaMemcpyAsync(buffers[0], pdata, sizeof(float)*N, cudaMemcpyHostToDevice, stream[0]));
    /// reduceGmem
    reduceGmem<float><<<num_blocks, threads_per_block, 0, stream[0]>>>((float*)buffers[1], (float*)buffers[0], N);
    reduceGmem<float><<<num_blocks2, threads_per_block, 0, stream[0]>>>((float*)buffers[2], (float*)buffers[1], num_blocks);
    reduceGmem<float><<<1, threads_per_block, 0, stream[0]>>>((float*)buffers[6], (float*)buffers[2], num_blocks2);

    cudaCheck(cudaMemsetAsync(buffers[1], 0, sizeof(float)*num_blocks2*threads_per_block, stream[0]));
    cudaCheck(cudaMemsetAsync(buffers[2], 0, sizeof(float)*num_blocks2, stream[0]));
    cudaCheck(cudaMemcpyAsync(buffers[0], pdata, sizeof(float)*N, cudaMemcpyHostToDevice, stream[0]));
    /// reduceSmem
    reduceSmem<float><<<num_blocks, threads_per_block, 0, stream[0]>>>((float*)buffers[1], (float*)buffers[0], N);
    reduceSmem<float><<<num_blocks2, threads_per_block, 0, stream[0]>>>((float*)buffers[2], (float*)buffers[1], num_blocks);
    reduceSmem<float><<<1, threads_per_block, 0, stream[0]>>>((float*)buffers[6]+1, (float*)buffers[2], num_blocks2);


    /// stream[1]
    cudaCheck(cudaMemcpyAsync(buffers[3], c_buffer, sizeof(float)*N, cudaMemcpyHostToDevice, stream[1]));
    /// reduceSmemUnroll
    reduceSmemUnroll<float><<<num_blocks / 4, threads_per_block, 0, stream[1]>>>((float*)buffers[4], (float*)buffers[3], N);
    reduceSmemUnroll<float><<<num_blocks2 / 16, threads_per_block, 0, stream[1]>>>((float*)buffers[5], (float*)buffers[4], num_blocks / 4);
    reduceSmem<float><<<1, threads_per_block, 0, stream[1]>>>((float*)buffers[6]+2, (float*)buffers[5], num_blocks2 / 16);


    /// reduceSmemUnrollDynamic
    cudaCheck(cudaMemsetAsync(buffers[4], 0, sizeof(float)*num_blocks2*threads_per_block, stream[1]));
    cudaCheck(cudaMemsetAsync(buffers[5], 0, sizeof(float)*num_blocks2, stream[1]));
    cudaCheck(cudaMemcpyAsync(buffers[3], c_buffer, sizeof(float)*N, cudaMemcpyHostToDevice, stream[1]));

    reduceSmemUnrollDynamic<float><<<num_blocks / 4, threads_per_block, sizeof(float)*threads_per_block, stream[1]>>>((float*)buffers[4], (float*)buffers[3], N);
    reduceSmemUnrollDynamic<float><<<num_blocks2 / 16, threads_per_block, sizeof(float)*threads_per_block, stream[1]>>>((float*)buffers[5], (float*)buffers[4], num_blocks / 4);
    reduceSmem<float><<<1, threads_per_block, 0, stream[1]>>>((float*)buffers[6]+3, (float*)buffers[5], num_blocks2 / 16);

    /// compare results
    cudaCheck(cudaMemcpy(&res, buffers[6], sizeof(float), cudaMemcpyDeviceToHost));
    printf("Global memory GPU Sum:%5f CPU Sum:%5f\n", res, res_check);

    cudaCheck(cudaMemcpy(&res, (float*)buffers[6]+1, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Shared memory GPU Sum:%5f CPU Sum:%5f\n", res, res_check);

    cudaCheck(cudaMemcpy(&res, (float*)buffers[6]+2, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Shared memory Unroll GPU Sum:%5f CPU Sum:%5f\n", res, cpu_res);

    cudaCheck(cudaMemcpy(&res, (float*)buffers[6]+3, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Shared memory Unroll Dynamic GPU Sum:%5f CPU Sum:%5f\n", res, cpu_res);

    /// free 
    cudaDeviceSynchronize();
    cudaProfilerStop();
    for(auto & e: buffers) cudaCheck(cudaFree(e));
    cudaCheck(cudaFreeHost(c_buffer));
    delete [] pdata;


    return 0;
}
