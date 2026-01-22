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
uint64_t computeNaiveGPU(const Data& data, const uint16_t user_threads);
//bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data);

int main(const int argc, const char** argv)
{
    Arguments args = parseArguments(argc, argv);
    printArguments(args);
    Data data = prepareData(args);
    if (!data.valid) return 1;

    { // GPU
        printf("\nGPU computation:\n");
        uint64_t result = computeNaiveGPU(data, args.threads);
        printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    if (args.cpu) { // CPU
        printf("\nCPU computation:\n");
        uint64_t result = computeNaiveCPU(data);
        printf("On CPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    return 0;
}

uint64_t computeNaiveCPU(const Data& data) {
    uint64_t result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (uint64_t i = 0; i < data.n; ++i) {
        for (uint64_t j = i + 1; j < data.n; ++j) {
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

uint64_t computeNaiveGPU(const Data& data, const uint16_t user_threads) {
    uint8_t* d_bits = nullptr;
    uint64_t* d_results = nullptr;
    uint64_t* h_results = nullptr;

    cudaEvent_t start_copy = nullptr, stop_copy = nullptr;
    cudaEvent_t start_compute = nullptr, stop_compute = nullptr;

    try {
        CUDA_CHECK(cudaEventCreate(&start_copy));
        CUDA_CHECK(cudaEventCreate(&stop_copy));
        CUDA_CHECK(cudaEventCreate(&start_compute));
        CUDA_CHECK(cudaEventCreate(&stop_compute));

        auto chrono_start = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMalloc(&d_bits, data.n * data.l * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&d_results, data.n * sizeof(uint64_t)));

        h_results = new uint64_t[data.n];

        CUDA_CHECK(cudaEventRecord(start_copy));

        CUDA_CHECK(cudaMemcpy(d_bits, data.bits.data(), data.n * data.l * sizeof(uint8_t), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMemset(d_results, 0, data.n * sizeof(uint64_t)));

        CUDA_CHECK(cudaEventRecord(stop_copy));
        CUDA_CHECK(cudaEventSynchronize(stop_copy));

        int threadsPerBlock = 1024;
        if (user_threads > 0) {
            threadsPerBlock = user_threads;
        }
        int blocksPerGrid = (data.n + threadsPerBlock - 1) / threadsPerBlock;

        CUDA_CHECK(cudaEventRecord(start_compute));
        hammingDistanceKernel << <blocksPerGrid, threadsPerBlock >> > (d_bits, data.n, data.l, d_results);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop_compute));
        CUDA_CHECK(cudaEventSynchronize(stop_compute));

        CUDA_CHECK(cudaMemcpy(h_results, d_results, data.n * sizeof(uint64_t), cudaMemcpyDeviceToHost));

        auto chrono_end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_start).count() / 1000.0;

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

        if (start_copy) cudaEventDestroy(start_copy); start_copy = nullptr;
        if (stop_copy) cudaEventDestroy(stop_copy); stop_copy = nullptr;
        if (start_compute) cudaEventDestroy(start_compute); start_compute = nullptr;
        if (stop_compute) cudaEventDestroy(stop_compute); stop_compute = nullptr;
        if (d_bits) cudaFree(d_bits); d_bits = nullptr;
        if (d_results) cudaFree(d_results); d_results = nullptr;
        delete[] h_results; h_results = nullptr;

        return total;
    }
    catch (const std::exception& e) {
        fprintf(stderr, "GPU computation failed: %s\n", e.what());

        if (start_copy) cudaEventDestroy(start_copy); start_copy = nullptr;
        if (stop_copy) cudaEventDestroy(stop_copy); stop_copy = nullptr;
        if (start_compute) cudaEventDestroy(start_compute); start_compute = nullptr;
        if (stop_compute) cudaEventDestroy(stop_compute); stop_compute = nullptr;
        if (d_bits) cudaFree(d_bits); d_bits = nullptr;
        if (d_results) cudaFree(d_results); d_results = nullptr;
        delete[] h_results; h_results = nullptr;

        throw;
    }
}