#include <stdio.h>

#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <time.h>

#include<sys/time.h>
#include<unistd.h>


#define cudaCheck(e) do { \
    if (cudaSuccess != (e)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);


/// res = A (row_A, col_A) * B (col_A, col_B) 
template <typename DType>
__global__ void matmul(DType* A, const int row_A, const int col_A, 
            DType* B, const int col_B, DType* res) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid = xIndex + yIndex * col_B;
    const size_t numel = row_A * col_B;
    if(tid >= numel) return;

    DType tmp = 0;
    for(int i = 0; i < col_A; ++i) {
        tmp += A[yIndex * col_A + i] * B[i*col_B + xIndex];
    }
    res[tid] = tmp;
}

#define DIM 32
#define IPAD 0

template <typename DType>
__global__ void matmul_smem(DType* A, const int row_A, const int col_A, 
            DType* B, const int col_B, DType* res) {
    int xIndex = threadIdx.x + blockIdx.x * blockDim.x;
    int yIndex = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid = xIndex + yIndex * col_B;
    if(tid >= row_A * col_B) return;

    DType tmp = 0;
    for(int i = 0; i < col_A / DIM; ++i) {
        __shared__ DType sA[DIM][DIM+IPAD];
        __shared__ DType sB[DIM][DIM+IPAD];

        sA[threadIdx.y][threadIdx.x] = A[yIndex * col_A + DIM * i + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = B[xIndex + (threadIdx.y + i*DIM) * col_B];
        __syncthreads();

        for(int j = 0; j < DIM; ++j) {
            tmp += sA[threadIdx.y][j] * sB[j][threadIdx.x];
        }
        __syncthreads();
    }
    if(col_A % DIM) {
        for(int i = col_A % DIM ; i > 0; --i) tmp += A[yIndex * col_A + col_A - i] * B[xIndex + (col_A - i)*col_B];
    } 

    res[tid] = tmp;
}


template <typename DType>
void cpu_matmul(DType* A, const int row_A, const int col_A,
        DType* B, const int col_B, DType* res) {
    for(int i = 0; i < row_A; ++i) {
        for(int j = 0 ; j < col_B; ++j) {
            DType tmp = 0;
            for(int k = 0; k < col_A; ++k) tmp += A[i*col_A + k] * B[k*col_B + j];
            res[i*col_B + j] = tmp;
        }
    }
}


__inline__ int divUp(int a, int b) {
    return (a + b - 1) / b;
}


template <typename DType>
DType diff(DType* A, DType* B, size_t numel) {
    DType res = 0;
    for(size_t i = 0; i < numel; ++i) res += fabs(A[i] - B[i]);
    return res;
}


int main(int argc, char* argv[]) {
    srand(time(NULL));
    cudaStream_t stream[2]; 
    cudaCheck(cudaSetDevice(0)); //! CUDA Streams
    for(int i = 0; i < 2; ++i) cudaCheck(cudaStreamCreate(&stream[i]));

    cudaProfilerStart();
    const int M = 1024, N = 1000, K = 1024;
    float * c_buffers[5];
    void* g_buffers[3];
    cudaCheck(cudaMallocHost(&c_buffers[0], sizeof(float)*M*N));
    cudaCheck(cudaMallocHost(&c_buffers[1], sizeof(float)*N*K));
    cudaCheck(cudaMallocHost(&c_buffers[2], sizeof(float)*M*K));
    cudaCheck(cudaMallocHost(&c_buffers[3], sizeof(float)*M*K));
    cudaCheck(cudaMallocHost(&c_buffers[4], sizeof(float)*M*K));

    cudaCheck(cudaMalloc(&g_buffers[0], sizeof(float)*M*N));
    cudaCheck(cudaMalloc(&g_buffers[1], sizeof(float)*N*K));
    cudaCheck(cudaMalloc(&g_buffers[2], sizeof(float)*M*K));


    for(int i = 0; i < M; ++i) {
        for(int j = 0; j < N; ++j) {
            c_buffers[0][i * N + j] = rand() % 3; // / double(RAND_MAX);
        }
    }

    for(int i = 0; i < N; ++i) {
        for(int j = 0; j < K; ++j) {
            c_buffers[1][i * K + j] = rand() % 2; // / double(RAND_MAX);
        }
    }
    /// cpu matmul
    struct timeval start;
    struct timeval end;
    gettimeofday(&start, NULL);
    cpu_matmul(c_buffers[0], M, N, c_buffers[1], K, c_buffers[2]);
    gettimeofday(&end, NULL);
    double cost = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0; 
    printf("cpu matmul cost time:%5f ms\n", cost);


    /// gpu matmul using global memory
    cudaCheck(cudaMemcpy(g_buffers[0], c_buffers[0], sizeof(float)*M*N, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(g_buffers[1], c_buffers[1], sizeof(float)*N*K, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemset(g_buffers[2], 0, sizeof(float)*M*K));

    dim3 block(32, 32);
    dim3 grid(divUp(M, 32), divUp(K, 32));
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    matmul<float><<<grid, block>>>((float*)g_buffers[0], M, N, (float*)g_buffers[1], K,
            (float*)g_buffers[2]);

    cudaCheck(cudaMemcpy(c_buffers[3], g_buffers[2], sizeof(float)*M*K, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cost = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0; 
    printf("gpu matmul cost time:%5f ms\n", cost);


    float diff_val = diff<float>(c_buffers[2], c_buffers[3], M*K);
    printf("diff_val:%5f\n", diff_val);
    /// gpu matmul using shared memory
    cudaCheck(cudaMemset(g_buffers[2], 0, sizeof(float)*M*K));
    dim3 block2(DIM, DIM);
    dim3 grid2(divUp(M, DIM), divUp(K, DIM));
    cudaDeviceSynchronize();
    gettimeofday(&start, NULL);
    matmul_smem<float><<<grid2, block2>>>((float*)g_buffers[0], M, N,
            (float*)g_buffers[1], K, (float*)g_buffers[2]);
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);
    cost = (end.tv_sec - start.tv_sec) * 1000.0 + (end.tv_usec - start.tv_usec) / 1000.0; 
    printf("gpu matmul cost time:%5f ms\n", cost);

    cudaCheck(cudaMemcpy(c_buffers[4], g_buffers[2], sizeof(float)*M*K, cudaMemcpyDeviceToHost));
    diff_val = diff<float>(c_buffers[2], c_buffers[4], M*K);
    printf("diff_val:%5f\n", diff_val);


    cudaProfilerStop();

    for(int i = 0; i < 3; ++i) {
        cudaCheck(cudaFreeHost(c_buffers[i]));
        cudaCheck(cudaFree(g_buffers[i]));
    }
            
    return 0;
}
