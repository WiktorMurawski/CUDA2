#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include "RadixTree.cuh"
#include "kernels.cuh"

#ifdef _WIN32
    #define strdup _strdup
#endif

#define _CRT_SECURE_NO_WARNINGS

#define VERBOSE_LIMIT 10

struct Arguments
{
    char* inputFile = nullptr;
    bool cpu = false;
    bool verbose = false;
};

struct Data {
    long long n = -1;
    long long l = -1;
    std::vector<uint8_t> bits;
};

void parseArguments(const int argc, const char** argv, Arguments& args);
void printArguments(const Arguments& args);
bool readTestFile(const std::string& filename, Data& data);
bool timed_readTestFile(const std::string& filename, Data& data);
RadixTree buildRadixTree(const Data& data);
void computeWithRadixTree(const RadixTree tree, const Data& data, std::vector<std::pair<int, int>>& results);
void printPairs(const std::vector<std::pair<int, int>> results, const Data& data);

int main(const int argc, const char** argv)
{
    dummyKernel <<<1, 1>>> ();

    Arguments args;
    parseArguments(argc, argv, args);
    printArguments(args);

    Data data;
    if (!timed_readTestFile(args.inputFile, data)) {
        printf("Error reading input file.\n");
        return 1;
    }
    printf("Data size: n = %lld, l = %lld, total bits = %zu\n\n", data.n, data.l, data.bits.size());

    RadixTree tree = buildRadixTree(data);

    std::vector<std::pair<int, int>> radix_results;
    computeWithRadixTree(tree, data, radix_results);

    printf("Found %zu pairs with Hamming distance of 1.\n", radix_results.size());
    if (args.verbose) {
        printPairs(radix_results, data);
    }

    return 0;
}

void parseArguments(const int argc, const char** argv, Arguments& args)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s <input file> [-c] [-v]", argv[0]);
        exit(1);
    }

    args.inputFile = strdup(argv[1]);

    for (int i = 2; i < argc; i++)
    {
        std::string arg = argv[i];
        if (arg == "-c")
        {
            args.cpu = true;
        }
        else if (arg == "-v")
        {
            args.verbose = true;
        }
    }
}

void printArguments(const Arguments& args)
{
    printf("Input File: %s\n", args.inputFile);
    printf("CPU Mode: %s\n", args.cpu ? "Enabled" : "Disabled");
    printf("Verbose Mode: %s\n", args.verbose ? "Enabled" : "Disabled");
}

bool readTestFile(const std::string& filename, Data& data) {
    FILE* file = fopen(filename.c_str(), "rb");
    if (!file) {
        return false;
    }

    char firstLine[256];
    if (!fgets(firstLine, sizeof(firstLine), file)) {
        fclose(file);
        return false;
    }

    if (sscanf(firstLine, "%lld,%lld", &data.n, &data.l) != 2) {
        fclose(file);
        return false;
    }

    data.bits.resize(data.n * data.l);

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    std::vector<char> buffer(fileSize);
    fread(buffer.data(), 1, fileSize, file);
    fclose(file);

    size_t dataStart = 0;
    while (dataStart < buffer.size() && buffer[dataStart] != '\n') {
        dataStart++;
    }
    dataStart++;

    size_t bitIdx = 0;
    for (size_t i = dataStart; i < buffer.size() && bitIdx < data.bits.size(); ++i) {
        if (buffer[i] == '0') {
            data.bits[bitIdx++] = 0;
        }
        else if (buffer[i] == '1') {
            data.bits[bitIdx++] = 1;
        }
    }

    return true;
}

bool timed_readTestFile(const std::string& filename, Data& data) {
    auto start = std::chrono::high_resolution_clock::now();
    bool result = readTestFile(filename, data);
    auto end = std::chrono::high_resolution_clock::now();
    double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
    if (result) printf("Reading file took: %.3f ms\n", duration_ms);
    return result;
}

RadixTree buildRadixTree(const Data& data) {
    auto start = std::chrono::high_resolution_clock::now();

    RadixTree tree(data.l);
    for (int i = 0; i < data.n; ++i) {
        tree.insert(&data.bits[i * data.l], i);
    }

    auto build_end = std::chrono::high_resolution_clock::now();
    double build_ms = std::chrono::duration_cast<std::chrono::microseconds>(build_end - start).count() / 1000.0;
    printf("Building radix tree took: %.3f ms\n", build_ms);
    printf("Tree has %d nodes\n", tree.getNodeCount());

    return tree;
}

void computeWithRadixTree(const RadixTree tree, const Data& data, std::vector<std::pair<int, int>>& results) {
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < data.n; ++i) {
        std::vector<int> neighbors;
        tree.findHammingDistanceOne(&data.bits[i * data.l], i, neighbors);

        for (int j : neighbors) {
            if (i < j) {
                results.push_back(std::make_pair(i, j));
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
}

void printPairs(const std::vector<std::pair<int, int>> results, const Data& data) {
    int limit = std::min((int)results.size(), VERBOSE_LIMIT);
    for (int i = 0; i < limit; ++i) {
        std::pair<int, int> pair = results[i];

        printf("Pair %d:\n", i + 1);
        for (int j = 0; j < data.l; ++j) {
            printf("%d", data.bits[pair.first * data.l + j]);
        }
        printf("\n");
        for (int j = 0; j < data.l; ++j) {
            printf("%d", data.bits[pair.second * data.l + j]);
        }
        printf("\n");
    }
}
