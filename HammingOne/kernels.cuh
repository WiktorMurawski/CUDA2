#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void hammingSearchKernel3(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results, const RadixNode* nodes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // Dynamic per-thread buffer in shared mem
    // Host passed total_size = blockDim.x * l, so stride = l bytes/thread
    extern __shared__ uint8_t shared_buf[];
    uint8_t* queryVector = shared_buf + threadIdx.x * l;  // my slice!

    // Coalesced copy (all threads copy in parallel → very fast)
    memcpy(queryVector, bits + i * l, l);

    uint64_t result = 0;

    for (uint64_t k = 0; k < l; ++k) {
        queryVector[k] ^= 1;

        //uint64_t currentIdx = rootIndex;
        uint64_t currentIdx = 0;
        for (uint64_t j = 0; j < l; ++j) {
            uint8_t bit = queryVector[j];
            if (nodes[currentIdx].children[bit] == -1) {
                break;
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        result += nodes[currentIdx].indicesCount;

        queryVector[k] ^= 1;
    }

    results[i] = result;
}

__global__ void hammingSearchKernel(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results, const RadixNode* nodes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t result = 0;
    uint8_t queryVector[16*4096];
    memcpy(queryVector, &bits[i * l], l * sizeof(uint8_t));

    for (uint64_t k = 0; k < l; ++k) {
        queryVector[k] ^= 1;

        //uint64_t currentIdx = rootIndex;
        uint64_t currentIdx = 0;
        for (uint64_t j = 0; j < l; ++j) {
            uint8_t bit = queryVector[j];
            if (nodes[currentIdx].children[bit] == -1) {
                break;
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        result += nodes[currentIdx].indicesCount;

        queryVector[k] ^= 1;
    }

    results[i] = result;
}



__global__ void hammingSearchKernel1(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results, const RadixNode* nodes) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint64_t result = 0;
    uint8_t* queryVector = (uint8_t*)&bits[i * l];

    for (uint64_t k = 0; k < l; ++k) {
        queryVector[k] = 1 - queryVector[k];

        //uint64_t currentIdx = rootIndex;
        uint64_t currentIdx = 0;
        for (uint64_t j = 0; j < l; ++j) {
            uint8_t bit = queryVector[j];
            if (nodes[currentIdx].children[bit] == -1) {
                break;
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        result += nodes[currentIdx].indicesCount;

        queryVector[k] = 1 - queryVector[k];
    }

    results[i] = result;
}
