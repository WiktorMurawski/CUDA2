#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#ifdef __INTELLISENSE__
    #define KERNEL_LAUNCH(...)
#else
    #define KERNEL_LAUNCH(...) <<< __VA_ARGS__ >>>
#endif

__global__ void dummyKernel() {}
