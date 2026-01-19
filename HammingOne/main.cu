#include <vector>
#include <chrono>
#include "common.h"
#include "RadixTree.cuh"
#include "kernels.cuh"

RadixTree buildRadixTree(const Data& data);
uint64_t computeWithRadixTreeCPU(const RadixTree tree, const Data& data);
uint64_t computeWithRadixTreeGPU(const RadixTree tree, const Data& data);

int main(const int argc, const char** argv)
{
    Arguments args = parseArguments(argc, argv);
    printArguments(args);
    Data data = prepareData(args);
    if (!data.valid) return 1;

    RadixTree tree = buildRadixTree(data);

    {
        printf("GPU computation:\n");
        uint64_t result = computeWithRadixTreeGPU(tree, data);
        printf("On GPU: Found %lld pairs with Hamming distance of 1.\n", result);
    }

    {
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

uint64_t computeWithRadixTreeCPU(const RadixTree tree, const Data& data) {
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

uint64_t computeWithRadixTreeGPU(const RadixTree tree, const Data& data) {
    return 0;
}