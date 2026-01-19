#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void dummyKernel() {}

// CUDA kernel: each thread handles one vector and counts pairs with all vectors after it
__global__ void hammingDistanceKernel(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n) return;

    uint64_t count = 0;

    // Compare vector i with all vectors j > i
    for (uint64_t j = i + 1; j < n; j++) {
        size_t diff_count = 0;

        // Check Hamming distance
        for (uint64_t k = 0; k < l; k++) {
            if (bits[i * l + k] != bits[j * l + k]) {
                diff_count++;
                if (diff_count > 1) break;
            }
        }

        if (diff_count == 1) {
            count++;
        }
    }

    results[i] = count;
}