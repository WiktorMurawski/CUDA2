#define _CRT_SECURE_NO_WARNINGS

#ifdef _WIN32
    #define strdup _strdup
#endif

#include <iostream>
#include <vector>
#include <string>
#include <set>
#include <fstream>
#include <chrono>
#include "kernels.cuh"

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
void printPairs(const std::vector<std::pair<int, int>> results, const Data& data);
void computeNaive(const Data& data, long long& result);
bool isHammingDistanceOne(size_t a_idx, size_t b_idx, const Data& data);
void countUniqueVectors(const Data& data);

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

    countUniqueVectors(data);

    long long result = 0;
    computeNaive(data, result);

    printf("Found %lld pairs with Hamming distance of 1.\n", result);
    //if (args.verbose) {
    //    printPairs(results, data);
    //}

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

void computeNaive(const Data& data, long long& result) {
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

void countUniqueVectors(const Data& data) {
    std::set<std::vector<uint8_t>> unique_vectors;
    for (int i = 0; i < data.n; ++i) {
        std::vector<uint8_t> vec(data.bits.begin() + i * data.l,
            data.bits.begin() + (i + 1) * data.l);
        unique_vectors.insert(vec);
    }
    printf("Unique vectors: %zu out of %lld\n", unique_vectors.size(), data.n);
}