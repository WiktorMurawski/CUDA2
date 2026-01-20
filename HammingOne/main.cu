#include <vector>
#include <chrono>
#include <stdexcept>
#include "common.h"
#include "RadixTree.cuh"
#include "kernels.cuh"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            char error_msg[256]; \
            snprintf(error_msg, sizeof(error_msg), \
                    "CUDA error at %s:%d: %s", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            throw std::runtime_error(error_msg); \
        } \
    } while(0)

RadixTree buildRadixTree(const Data& data);
uint64_t computeWithRadixTreeCPU(const RadixTree& tree, const Data& data);
uint64_t computeWithRadixTreeGPU(const RadixTree& tree, const Data& data);

int main(const int argc, const char** argv)
{
    Arguments args = parseArguments(argc, argv);
    printArguments(args);
    Data data = prepareData(args);
    if (!data.valid) return 1;

    RadixTree tree = buildRadixTree(data);

    {
        try {
            printf("\nGPU computation:\n");
            uint64_t result = computeWithRadixTreeGPU(tree, data);
            printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
        }
        catch (const std::exception& e) {
            fprintf(stderr, "GPU computation failed: %s\n", e.what());
            return 1;
        }
        //printf("\nGPU computation:\n");
        //uint64_t result = computeWithRadixTreeGPU(tree, data);
        //printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    {
        printf("\nCPU computation:\n");
        uint64_t result = computeWithRadixTreeCPU(tree, data);
        printf("On CPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    return 0;
}

RadixTree buildRadixTree(const Data& data) {
    auto start = std::chrono::high_resolution_clock::now();

    RadixTree tree(data.l);
    for (uint64_t i = 0; i < data.n; ++i) {
        tree.insert(&data.bits[i * data.l], i);
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration_cast<std::chrono::microseconds>(build_end - start).count() / 1000.0;
    printf("Building radix tree took: %.3f ms\n", build_ms);
    printf("Tree has %lld nodes\n", tree.getNodeCount());

    return tree;
}

uint64_t computeWithRadixTreeCPU(const RadixTree& tree, const Data& data) {
    auto start = std::chrono::high_resolution_clock::now();
    uint64_t result = 0;

    for (uint64_t i = 0; i < data.n; ++i) {
        std::vector<uint64_t> neighbors;
        tree.findHammingDistanceOne(&data.bits[i * data.l], i, neighbors);

        for (uint64_t j : neighbors) {
            if (i < j) {
                result++;
            }
        }

        if (i % 1000 == 0) {
            printf(".");
        }
    }
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    printf("Searching for pairs took: %.3f ms\n", search_ms);

    return result;
}

uint64_t computeWithRadixTreeGPU(const RadixTree& tree, const Data& data) {
    uint8_t* d_bits = nullptr;
    RadixNode* d_nodes = nullptr;
    uint64_t* d_results = nullptr;

    uint64_t* h_results = nullptr;

    uint64_t result = 0;

    try {
        CUDA_CHECK(cudaMalloc(&d_bits, data.n * data.l * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_nodes, tree.getNodeCount() * sizeof(RadixNode)));
        CUDA_CHECK(cudaMalloc(&d_results, data.n * sizeof(uint64_t)));

        h_results = new uint64_t[data.n];

        CUDA_CHECK(cudaMemcpy(d_bits, data.bits.data(), data.n * data.l * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes, tree.getNodes().data(), tree.getNodeCount() * sizeof(RadixNode), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocksPerGrid = (data.n + threadsPerBlock - 1) / threadsPerBlock;
        
        dummyKernel << <blocksPerGrid, threadsPerBlock >> > ();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        hammingSearchKernel<<<blocksPerGrid, threadsPerBlock >>>(d_bits, data.n, data.l, d_results, d_nodes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_results, d_results, data.n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        for(uint64_t i = 0; i < data.n; ++i) {
            result += h_results[i];
        }
        result /= 2;

        cudaFree(d_bits);
        cudaFree(d_results);
        cudaFree(d_nodes);
        delete[] h_results;
    }
    catch (...) {
        cudaFree(d_bits);
        cudaFree(d_results);
        cudaFree(d_nodes);
        delete[] h_results;

        throw;
    }

    return result;
}

//uint64_t benchmarkKernel(const char* name, /* kernel params */) {
//    // Warmup run (don't time this)
//    myKernel << <... >> > (...);
//    cudaDeviceSynchronize();
//
//    // Multiple timed runs
//    const int numRuns = 5;
//    float totalTime = 0;
//
//    cudaEvent_t start, stop;
//    cudaEventCreate(&start);
//    cudaEventCreate(&stop);
//
//    for (int run = 0; run < numRuns; run++) {
//        // Optional: reset L2 cache between runs
//        // cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);
//
//        cudaEventRecord(start);
//        myKernel << <... >> > (...);
//        cudaEventRecord(stop);
//        cudaEventSynchronize(stop);
//
//        float ms;
//        cudaEventElapsedTime(&ms, start, stop);
//        totalTime += ms;
//    }
//
//    printf("%s: avg %.3f ms\n", name, totalTime / numRuns);
//
//    cudaEventDestroy(start);
//    cudaEventDestroy(stop);
//}