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
uint64_t computeWithRadixTreeGPU(const RadixTree& tree, const Data& data, const uint16_t user_threads);

int main(const int argc, const char** argv)
{
    Arguments args = parseArguments(argc, argv);
    printArguments(args);
    Data data = prepareData(args);
    if (!data.valid) return 1;

    RadixTree tree = buildRadixTree(data);

    {
        printf("\nGPU computation:\n");
        uint64_t result = computeWithRadixTreeGPU(tree, data, args.threads);
        printf("On GPU: Found %lld pairs with Hamming distance of 1.\n\n", result);
    }

    if (args.cpu) { // CPU
        printf("CPU computation:\n");
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

uint64_t computeWithRadixTreeGPU(const RadixTree& tree, const Data& data, const uint16_t user_threads) {
    uint8_t* d_bits = nullptr;
    RadixNode* d_nodes = nullptr;
    uint64_t* d_results = nullptr;

    uint64_t* h_results = nullptr;

    uint64_t result = 0;

    cudaEvent_t start_copy = nullptr, stop_copy = nullptr;
    cudaEvent_t start_compute = nullptr, stop_compute = nullptr;

    try {
        // Timery
        CUDA_CHECK(cudaEventCreate(&start_copy));
        CUDA_CHECK(cudaEventCreate(&stop_copy));
        CUDA_CHECK(cudaEventCreate(&start_compute));
        CUDA_CHECK(cudaEventCreate(&stop_compute));
        auto chrono_start = std::chrono::high_resolution_clock::now();

        // Alokacja pamięci
        CUDA_CHECK(cudaMalloc(&d_bits, data.n * data.l * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_nodes, tree.getNodeCount() * sizeof(RadixNode)));
        CUDA_CHECK(cudaMalloc(&d_results, data.n * sizeof(uint64_t)));
        h_results = new uint64_t[data.n];

        // Kopiowanie drzewa i danych na GPU
        CUDA_CHECK(cudaEventRecord(start_copy));
        CUDA_CHECK(cudaMemcpy(d_bits, data.bits.data(), data.n * data.l * sizeof(uint8_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_nodes, tree.getNodes().data(), tree.getNodeCount() * sizeof(RadixNode), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(stop_copy));
        CUDA_CHECK(cudaEventSynchronize(stop_copy));

        int threadsPerBlock = 1024;
        if (user_threads > 0) {
            threadsPerBlock = user_threads;
        }
        int blocksPerGrid = (data.n + threadsPerBlock - 1) / threadsPerBlock;
        
        //printf("Kernel launch - %d blocks per grid, % threads per block", blocksPerGrid, threadsPerBlock);

        // Uruchomienie kernela
        CUDA_CHECK(cudaEventRecord(start_compute));
        hammingSearchKernel<<<blocksPerGrid, threadsPerBlock >>>(d_bits, data.n, data.l, d_results, d_nodes);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_compute));
        CUDA_CHECK(cudaEventSynchronize(stop_compute));

        // Kopiowanie wyników z powrotem na CPU
        CUDA_CHECK(cudaMemcpy(h_results, d_results, data.n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        auto chrono_mid = std::chrono::high_resolution_clock::now();

        for(uint64_t i = 0; i < data.n; ++i) {
            result += h_results[i];
        }
        result /= 2;

        auto chrono_end = std::chrono::high_resolution_clock::now();
        double cpu_sum_ms = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_mid).count() / 1000.0;
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_start).count() / 1000.0;

        float copy_ms = 0, compute_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start_copy, stop_copy));
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms, start_compute, stop_compute));

        printf("Data copy time: %.3f ms\n", copy_ms);
        printf("Searching for pairs took: %.3f ms\n", compute_ms);
        //printf("Summing results from threads took: %.3f ms\n", cpu_sum_ms);
        printf("Total operation time: %.3f ms\n", total_ms);

        if (d_bits) CUDA_CHECK(cudaFree(d_bits)); d_bits = nullptr;
        if (d_results) CUDA_CHECK(cudaFree(d_results)); d_results = nullptr;
        if (d_nodes) CUDA_CHECK(cudaFree(d_nodes)); d_nodes = nullptr;
        delete[] h_results; h_results = nullptr;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "GPU computation failed: %s\n", e.what());

        if (d_bits) cudaFree(d_bits); d_bits = nullptr;
        if (d_results) cudaFree(d_results); d_results = nullptr;
        if (d_nodes) cudaFree(d_nodes); d_nodes = nullptr;
        delete[] h_results; h_results = nullptr;

        throw;
    }

    return result;
}