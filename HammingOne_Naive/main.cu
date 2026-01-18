#include <vector>
#include <chrono>
#include "common.h"
#include "kernels.cuh"

uint64_t computeNaiveCPU(const Data& data);
uint64_t computeNaiveGPU(const Data& data);
bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data);

int main(const int argc, const char** argv)
{
    Data data = prepareData(argc, argv);
    if (!data.valid) return 1;

    {
        uint64_t result = computeNaiveCPU(data);
        printf("On CPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    {
        uint64_t result = computeNaiveGPU(data);
        printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    return 0;
}

bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data) {
    size_t diff_count = 0;
    for (int i = 0; i < data.l; ++i) {
        if (data.bits[a_idx * data.l + i] != data.bits[b_idx * data.l + i]) {
            diff_count++;
            if (diff_count > 1) {
                return false;
            }
        }
    }
    return diff_count == 1;
}

uint64_t computeNaiveCPU(const Data& data) {
    uint64_t result = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < data.n; ++i) {
        for (int j = i + 1; j < data.n; ++j) {
            if (isHammingDistanceOne(i, j, data)) {
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

uint64_t computeNaiveGPU(const Data& data) {
    return 0;
}

