#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "RadixTree.cuh"

#define MAX_LOCAL_VECTOR_LENGTH 8192

__global__ void hammingSearchKernelFast(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results, const RadixNode* nodes, const uint64_t rootIndex) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint8_t queryVector[MAX_LOCAL_VECTOR_LENGTH];
    memcpy(queryVector, &bits[i * l], l * sizeof(uint8_t));

    uint64_t result = 0;

    for (uint64_t k = 0; k < l; ++k) {
        queryVector[k] ^= 1;

        uint64_t currentIdx = rootIndex;
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

__global__ void hammingSearchKernelFallback(const uint8_t* bits, uint64_t n, uint64_t l, uint64_t* results, const RadixNode* nodes, const uint64_t rootIndex) {
    uint64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    uint8_t* queryVector = (uint8_t*)&bits[i * l];

    uint64_t result = 0;

    for (uint64_t k = 0; k < l; ++k) {
        queryVector[k] ^= 1;

        uint64_t currentIdx = rootIndex;
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