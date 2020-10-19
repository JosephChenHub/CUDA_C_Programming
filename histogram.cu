#include <stdlib.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <assert.h>
#include <string.h>
#include <cooperative_groups.h>
#include <chrono>

namespace cg = cooperative_groups;
using namespace std;

#define CUDA_CHECK(e) do { \
    if (cudaSuccess != (e)) { \
    fprintf(stderr, "Cuda runtime error in line %d of file %s \
    : %s \n", __LINE__, __FILE__, cudaGetErrorString(cudaGetLastError()) ); \
    exit(EXIT_FAILURE); \
    } \
    } while(0);

#define BIN_COUNT 256



inline size_t divUp(const size_t a, const size_t b) {
    return (a + b -1) / b;
}

inline __device__ __host__ uint8_t get_val(const uint8_t* data, 
        const size_t offset, 
        const int img_w, const int img_h,
        const int x, const int y, 
        const uint8_t ignore_label) {
    if (x < 0 || x >= img_w || y < 0 || y >= img_h) {
        return ignore_label;
    }
    else {
        return data[offset + x + y*img_w];
    }
}

void cpu_roi_hist(const uint8_t* data, 
        const int img_h, const int img_w,
        const int start_x, const int start_y, 
        const int roi_h, const int roi_w,
        const uint8_t ignore_label, 
        uint32_t* hist) {
    for(int y = start_y; y < start_y+roi_h; ++y) {
        for(int x = start_x; x < start_x+roi_w; ++x) {
            uint8_t val = get_val(data, 0, img_w, img_h, x, y, ignore_label);
            hist[val] += 1;
        }
    }
}


/// naive implementation 
__global__ void naive_roi_hist(const uint8_t* data, 
        const int img_h, const int img_w, 
        const int start_x, const int start_y,
        const int roi_h, const int roi_w, 
        const uint8_t ignore_label, uint32_t* hist) {
    // threads layout 2D
    const int threadX = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadY = threadIdx.y + blockIdx.y * blockDim.y;
//    const int batch_id = static_cast<int>(threadY / roi_h);
    if (threadX >= roi_w || threadY >= roi_h) return;

    uint8_t val = get_val(data, 0, img_w, img_h, threadX+start_x, threadY+start_y, ignore_label);

    atomicAdd(&hist[val], 1);
}

/// fast

__global__ void fast_roihist_kernel(const uint8_t *data, 
        const int img_h, const int img_w, 
        const int start_x, const int start_y,
        const int roi_h, const int roi_w, 
        const uint8_t ignore_label, 
        uint32_t* partial_hist) {
    // pixel coordinates
    const int threadX = threadIdx.x + blockIdx.x * blockDim.x;
    const int threadY = threadIdx.y + blockIdx.y * blockDim.y;
    // grid dimensions 
    const int nx = blockDim.x * gridDim.x;
    const int ny = blockDim.y * gridDim.y;
    // total threads in 2D block
    const int nt = blockDim.x * blockDim.y; 
    // linear block index within 2D grid
    const int g = blockIdx.x + blockIdx.y * gridDim.x; 
    __shared__ uint32_t smem[BIN_COUNT];
    #pragma unroll
    for (int i = threadIdx.x + threadIdx.y * blockDim.x; i < BIN_COUNT; i += nt) {
        smem[i] = 0;
    }
    __syncthreads();

    for(int col = threadX; col < roi_w; col += nx) {
        for(int row = threadY; row < roi_h; row += ny) {
            uint8_t val = get_val(data, 0, img_w, img_h, start_x + col, start_y + row, ignore_label);
            atomicAdd(&smem[val], 1);
        }
    }
    __syncthreads();
    // write 
    partial_hist += g * BIN_COUNT;
    for(int i = threadIdx.x + threadIdx.y * blockDim.x; i < BIN_COUNT; i += nt) {
        partial_hist[i] = smem[i];        
    }
}

__global__ void merge_histogram(
    const uint32_t * partial_hist, 
    const uint32_t hist_num, 
    uint32_t* hist) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < BIN_COUNT) {
        uint32_t sum(0);
        for(int j = 0; j < hist_num; ++j) {
            sum += partial_hist[tid + BIN_COUNT*j];
        }
        hist[tid] = sum;
    }
}



int main(int argc, char* argv[]) {
    cudaSetDevice(0);
    cudaEvent_t _start, _end;
    CUDA_CHECK(cudaEventCreateWithFlags(&_start, cudaEventBlockingSync));
    CUDA_CHECK(cudaEventCreateWithFlags(&_end, cudaEventBlockingSync));


    const int img_h = 1024, img_w = 2048;
    const int roi_h = atoi(argv[1]), roi_w = atoi(argv[2]);
    assert(roi_h > 0 && roi_w > 0);
    std::cout << "ROI: " << roi_h << "," << roi_w << std::endl;

    dim3 block(16, 16);
    dim3 grid(divUp(roi_w, 16), divUp(roi_h, 16));
    cout << "block:" << block.x << "," << block.y
        << " grid:" << grid.x << "," << grid.y << endl;

    size_t total_blocks = grid.x * grid.y;
    

    uint8_t* dummy_data = new uint8_t[img_h*img_w];
    uint32_t cpu_hist[BIN_COUNT], gpu_hist[BIN_COUNT];
    for(size_t i = 0; i < img_h * img_w; ++i) {
        dummy_data[i] = static_cast<uint8_t>(rand() % 255);
    }


    uint8_t* gpu_data;
    uint32_t* gpu_res, *d_partial_hist;

    CUDA_CHECK(cudaMalloc(&gpu_data, sizeof(uint8_t) * img_h * img_w));
    CUDA_CHECK(cudaMalloc(&gpu_res, sizeof(uint32_t) * BIN_COUNT));
    CUDA_CHECK(cudaMalloc(&d_partial_hist, sizeof(uint32_t) * total_blocks * BIN_COUNT));
    CUDA_CHECK(cudaMemset(gpu_res, 0, sizeof(uint32_t)*BIN_COUNT));
    CUDA_CHECK(cudaMemcpy(gpu_data, dummy_data, sizeof(uint8_t) * img_h * img_w, cudaMemcpyHostToDevice));

    float cpu_avg_sec (0.f), gpu_avg_sec(0.f), gpu_avg_sec2(0.f);
    const int N = 10;
    for(int nTest = 0; nTest < N; ++nTest) {
        int start_x = -100 + rand() % (img_w - roi_w + 1);
        int start_y = -100 + rand() % (img_h - roi_h + 1);
        cout << "Testing on roi start:" << start_x 
            << "," << start_y << endl;
        // cpu
        memset(cpu_hist, 0, sizeof(uint32_t)*BIN_COUNT);
        auto t0 = std::chrono::high_resolution_clock::now();
        cpu_roi_hist(dummy_data, img_h,  img_w,
            start_x, start_y, roi_h, roi_w,
            255, cpu_hist);
        auto t1 = std::chrono::high_resolution_clock::now();
        cpu_avg_sec += std::chrono::duration<float, std::milli>(t1 - t0).count();
        // gpu

        float cost_time(0.f);
        CUDA_CHECK(cudaMemset(gpu_res, 0, sizeof(uint32_t)*BIN_COUNT));
        cudaEventRecord(_start);
        naive_roi_hist<<<grid, block>>>(gpu_data, img_h, img_w,
                start_x, start_y, 
                roi_h, roi_w, 255, gpu_res);
        CUDA_CHECK(cudaMemcpy(gpu_hist, gpu_res, sizeof(uint32_t) * BIN_COUNT, cudaMemcpyDeviceToHost));
        cudaEventRecord(_end);
        cudaEventSynchronize(_end);
        cudaEventElapsedTime(&cost_time, _start, _end);
        gpu_avg_sec += cost_time;


        for(int i = 0; i < BIN_COUNT; ++i) {
            float err = gpu_hist[i] - cpu_hist[i];
            if (err > 1e-5) {
                std::cout << "err:" << err << " cpu_hist[" << i << "]:"
                    << cpu_hist[i] << " gpu_hist:" << gpu_hist[i]
                    << std::endl;
                throw std::runtime_error("naive kernel: cpu_hist != gpu_hist !");
            }
        }
        CUDA_CHECK(cudaMemset(gpu_res, 0, sizeof(uint32_t)*BIN_COUNT));
        cudaEventRecord(_start);
        fast_roihist_kernel<<<grid, block>>>(gpu_data, img_h, img_w,
                start_x, start_y, 
                roi_h, roi_w, 255, d_partial_hist);
        merge_histogram<<<4, 64>>>(d_partial_hist, total_blocks, gpu_res);
        CUDA_CHECK(cudaMemcpy(gpu_hist, gpu_res, sizeof(uint32_t) * BIN_COUNT, cudaMemcpyDeviceToHost));
        cudaEventRecord(_end);
        cudaEventSynchronize(_end);
        cudaEventElapsedTime(&cost_time, _start, _end);
        gpu_avg_sec2 += cost_time;

        for(int i = 0; i < BIN_COUNT; ++i) {
            float err = gpu_hist[i] - cpu_hist[i];
            if (err > 1e-5) {
                std::cout << "err:" << err << " cpu_hist[" << i << "]:"
                    << cpu_hist[i] << " gpu_hist:" << gpu_hist[i]
                    << std::endl;
                throw std::runtime_error("fast kernel: cpu_hist != gpu_hist !");
            }
        }
    }
    cpu_avg_sec /= N;
    gpu_avg_sec /= N;
    gpu_avg_sec2 /= N;
    cout << "CPU's throughput is " << (1e-6 * roi_h * roi_w / (cpu_avg_sec/1000.0f + 1e-7)) << " MB/s"
        << " naive GPU's throughput is " << (1e-6 * roi_h * roi_w / (gpu_avg_sec/1000.0f + 1e-7)) << " MB/s"
        << " fast GPU's throughput is " << (1e-6 * roi_h * roi_w / (gpu_avg_sec2/1000.0f + 1e-7)) << " MB/s"
        << std::endl;
    cout << " Speedup naive/cpu:" << cpu_avg_sec / (1e-7 + gpu_avg_sec)
        << ", fast/cpu:" << cpu_avg_sec / (1e-7 + gpu_avg_sec2)
        << endl;





    delete [] dummy_data;

    CUDA_CHECK(cudaEventDestroy(_start));
    CUDA_CHECK(cudaEventDestroy(_end));
    CUDA_CHECK(cudaFree(gpu_data));
    CUDA_CHECK(cudaFree(gpu_res));
    CUDA_CHECK(cudaFree(d_partial_hist));


    return 0;
}







