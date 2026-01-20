#include <vector>
#include <chrono>
#include <stdexcept>
#include "common.h"
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

uint64_t computeNaiveCPU(const Data& data);
uint64_t computeNaiveGPU(const Data& data);
//bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data);

int main(const int argc, const char** argv)
{
    Arguments args = parseArguments(argc, argv);
    printArguments(args);
    Data data = prepareData(args);
    if (!data.valid) return 1;

    { // GPU
        printf("\nGPU computation:\n");
        uint64_t result = computeNaiveGPU(data);
        printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    if (args.cpu) { // CPU
        printf("\nCPU computation:\n");
        uint64_t result = computeNaiveCPU(data);
        printf("On CPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    return 0;
}

//bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data) {
//    uint8_t diff_count = 0;
//    for (uint64_t i = 0; i < data.l; ++i) {
//        if (data.bits[a_idx * data.l + i] != data.bits[b_idx * data.l + i]) {
//            diff_count++;
//            if (diff_count > 1) {
//                return false;
//            }
//        }
//    }
//    return diff_count == 1;
//}

uint64_t computeNaiveCPU(const Data& data) {
    uint64_t result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < data.n; ++i) {
        for (uint64_t j = i + 1; j < data.n; ++j) {
            //if (isHammingDistanceOne(i, j, data)) {
            //    result++;
            //}
            uint8_t diff_count = 0;
            for (uint64_t k = 0; k < data.l; ++k) {
                if (data.bits[i * data.l + k] != data.bits[j * data.l + k]) {
                    diff_count++;
                    if (diff_count > 1) {
                        break;
                    }
                }
            }
            if (diff_count == 1) {
                ++result;
            }
        }

        //if (i % 1000 == 0) {
        //    printf(".");
        //}
    }
    printf("\n");

    auto end = std::chrono::high_resolution_clock::now();
    double search_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    printf("Searching for pairs took: %.3f ms\n", search_ms);
    return result;
}

uint64_t computeNaiveGPU(const Data& data) {
    uint8_t* d_bits = nullptr;
    uint64_t* d_results = nullptr;
    uint64_t* h_results = nullptr;

    // Create CUDA events for timing
    cudaEvent_t start_copy = nullptr, stop_copy = nullptr;
    cudaEvent_t start_compute = nullptr, stop_compute = nullptr;

    try {
        CUDA_CHECK(cudaEventCreate(&start_copy));
        CUDA_CHECK(cudaEventCreate(&stop_copy));
        CUDA_CHECK(cudaEventCreate(&start_compute));
        CUDA_CHECK(cudaEventCreate(&stop_compute));

        // Start overall timing with chrono
        auto chrono_start = std::chrono::high_resolution_clock::now();

        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_bits, data.n * data.l * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_results, data.n * sizeof(uint64_t)));

        // Allocate host memory for results
        h_results = new uint64_t[data.n];

        // Start timing data copy
        CUDA_CHECK(cudaEventRecord(start_copy));

        // Copy data to device
        CUDA_CHECK(cudaMemcpy(d_bits, data.bits.data(), data.n * data.l * sizeof(uint8_t), cudaMemcpyHostToDevice));

        // Initialize results to zero
        CUDA_CHECK(cudaMemset(d_results, 0, data.n * sizeof(uint64_t)));

        // Stop timing data copy
        CUDA_CHECK(cudaEventRecord(stop_copy));
        CUDA_CHECK(cudaEventSynchronize(stop_copy));

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (data.n + threadsPerBlock - 1) / threadsPerBlock;

        // Start timing computation
        CUDA_CHECK(cudaEventRecord(start_compute));
        hammingDistanceKernel << <blocksPerGrid, threadsPerBlock >> > (d_bits, data.n, data.l, d_results);
        CUDA_CHECK(cudaGetLastError());

        // Stop timing computation
        CUDA_CHECK(cudaEventRecord(stop_compute));
        CUDA_CHECK(cudaEventSynchronize(stop_compute));

        // Copy results back
        CUDA_CHECK(cudaMemcpy(h_results, d_results, data.n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        // End overall timing with chrono
        auto chrono_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_start).count() / 1000.0;

        // Calculate CUDA event timings
        float copy_ms = 0, compute_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&copy_ms, start_copy, stop_copy));
        CUDA_CHECK(cudaEventElapsedTime(&compute_ms, start_compute, stop_compute));

        // Sum up all results
        uint64_t total = 0;
        for (uint64_t i = 0; i < data.n; i++) {
            total += h_results[i];
        }

        printf("Data copy time: %.3f ms\n", copy_ms);
        printf("Searching for pairs took: %.3f ms\n", compute_ms);
        printf("Total GPU operation time (chrono): %.3f ms\n", total_ms);

        if (start_copy) cudaEventDestroy(start_copy);
        if (stop_copy) cudaEventDestroy(stop_copy);
        if (start_compute) cudaEventDestroy(start_compute);
        if (stop_compute) cudaEventDestroy(stop_compute);
        if (d_bits) cudaFree(d_bits);
        if (d_results) cudaFree(d_results);
        delete[] h_results;

        return total;
    }
    catch (...) {
        if (start_copy) cudaEventDestroy(start_copy);
        if (stop_copy) cudaEventDestroy(stop_copy);
        if (start_compute) cudaEventDestroy(start_compute);
        if (stop_compute) cudaEventDestroy(stop_compute);
        if (d_bits) cudaFree(d_bits);
        if (d_results) cudaFree(d_results);
        delete[] h_results;

        return 0;
        //throw;
    }
}