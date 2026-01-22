#include <unordered_map>
#include <vector>

struct RadixNode {
    int64_t children[2];      // indeksy dzieci, -1 jeśli dziecko nie istnieje
    int64_t indicesOffset;    // offset indeksów w tablicy indeksów w drzewie
    int64_t indicesCount;     // liczba indeksów wektorów w liściu (0 jeśli węzeł nie jest liściem)

    RadixNode() {
        children[0] = -1;
        children[1] = -1;
        indicesOffset = -1;
        indicesCount = 0;
    }
};

class RadixTree {
private:
    mutable std::vector<RadixNode> nodes;
    std::unordered_map<uint64_t, std::vector<uint64_t>> leafIndicesMap;
    mutable std::vector<uint64_t> leafIndices;
    uint64_t vectorLength;
    uint64_t rootIndex;
    mutable bool isDirty = false;

    uint64_t allocateNode() {
        nodes.push_back(RadixNode());
        return (uint64_t)nodes.size() - 1;
    }

    // Funkcja do spłaszczania mapy indeksów liści do jednej tablicy
    void ensureFlattened() const {
        if (!isDirty) return;

        leafIndices.clear();

        for (auto& [nodeIdx, indices] : leafIndicesMap) {
            nodes[nodeIdx].indicesOffset = leafIndices.size();
            nodes[nodeIdx].indicesCount = indices.size();

            for (uint64_t idx : indices) {
                leafIndices.push_back(idx);
            }
        }

        isDirty = false;
    }

public:
    RadixTree(uint64_t l) : vectorLength(l), rootIndex(0) {
        allocateNode();
    }

    // Funkcja do wstawiania wektora bitowego wraz z jego indeksem
    void insert(const uint8_t* bits, uint64_t vectorIndex) {
        uint64_t currentIdx = rootIndex;

        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];
            if (nodes[currentIdx].children[bit] == -1) {
                uint64_t newNodeIdx = allocateNode();
                nodes[currentIdx].children[bit] = newNodeIdx;
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        leafIndicesMap[currentIdx].push_back(vectorIndex);
        isDirty = true;
    }

    // Funkcja do wyszukiwania wszystkich indeksów wektorów dokładnie pasujących do podanego wektora bitowego
    std::vector<uint64_t> searchAll(const uint8_t* bits) const {
        ensureFlattened();

        uint64_t currentIdx = rootIndex;

        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];
            if (nodes[currentIdx].children[bit] == -1) {
                return std::vector<uint64_t>();
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        std::vector<uint64_t> result;
        if (nodes[currentIdx].indicesCount > 0) {
            int64_t offset = nodes[currentIdx].indicesOffset;
            int64_t count = nodes[currentIdx].indicesCount;
            result.reserve(count);
            for (int64_t i = 0; i < count; ++i) {
                result.push_back(leafIndices[offset + i]);
            }
        }
        return result;
    }

    // Funkcja do znajdowania wszystkich wektorów różniących się o dokładnie jeden bit
    void findHammingDistanceOne(const uint8_t* bits, uint64_t queryIndex, std::vector<uint64_t>& results) const {
        ensureFlattened();

        std::vector<uint8_t> modified(vectorLength);
        for (uint64_t i = 0; i < vectorLength; ++i) {
            modified[i] = bits[i];
        }

        for (uint64_t i = 0; i < vectorLength; ++i) {
            modified[i] ^= 1;

            std::vector<uint64_t> foundIndices = searchAll(modified.data());
            for (uint64_t foundIndex : foundIndices) {
                if (foundIndex != queryIndex) {
                    results.push_back(foundIndex);
                }
            }

            modified[i] ^= 1;
        }
    }

    // Getter dla płaskiego wektora węzłów
    const std::vector<RadixNode>& getNodes() const {
        ensureFlattened();
        return nodes;
    }

    // Getter dla płaskiego wektora indeksów liści
    const std::vector<uint64_t>& getLeafIndices() const {
        ensureFlattened();
        return leafIndices;
    }

    // Getter dla liczby węzłów
    uint64_t getNodeCount() const {
        return nodes.size();
    }

    // Getter dla długości wektorów bitowych
    uint64_t getBitLength() const {
        return vectorLength;
    }

    // Getter dla indeksu korzenia (ustawianego na 0 w konstruktorze)
    uint64_t getRootIndex() const {
        return rootIndex;
    }
};