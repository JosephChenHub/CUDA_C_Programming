# CUDA C Programming demos


## Dependencies
- CUDA

## Items 
- *hello.cu*: hello world from GPU!
- *hello2.cu*: understanding the thread index (1D, 2D, 3D).
- *share_mem.cu*: understanding the memory hierachy, specifically, the power of shared memory compared with the global memory!
- *matrix.cu*: an example of matrix multiplication.

to **compile** these files, run this command
```
  nvcc hello.cu -o hello 
```
or 
```
  nvcc share_mem.cu -o share -std=c++11
```

## Performance analysis
to eval the performance of different kernels, run the command like
```
  nvprof ./share
```
or 
```
  nvprof --metrics gld_throughput  --metrics gst_throughput./share  
```
where `gld_throughput` means global memory load throughput, `gst_througput` refers to global memory store throughput. 
More metrics can be found via `nvprof --query-metrics`.

