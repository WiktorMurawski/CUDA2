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
    std::vector<RadixNode> nodes;       // vector wszystkich węzłów drzewa
    std::vector<uint64_t> leafIndices;  // vector przechowujący indeksy wektorów w danych wejściowych
    uint64_t vectorLength;
    uint64_t rootIndex;

    uint64_t allocateNode() {
        nodes.push_back(RadixNode());
        return (uint64_t)nodes.size() - 1;
    }

public:
    RadixTree(uint64_t l) : vectorLength(l), rootIndex(0) {
        allocateNode();
    }

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

        if (nodes[currentIdx].indicesCount == 0) {
            nodes[currentIdx].indicesOffset = leafIndices.size();
        }
        leafIndices.push_back(vectorIndex);
        nodes[currentIdx].indicesCount++;
    }

    std::vector<uint64_t> searchAll(const uint8_t* bits) const {
        uint64_t currentIdx = rootIndex;

        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];
            if (nodes[currentIdx].children[bit] == -1) {
                return std::vector<uint64_t>(); // Not found
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

    void findHammingDistanceOne(const uint8_t* bits, uint64_t queryIndex, std::vector<uint64_t>& results) const {
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

    const std::vector<RadixNode>& getNodes() const {
        return nodes;
    }

    const std::vector<uint64_t>& getLeafIndices() const {
        return leafIndices;
    }

    uint64_t getNodeCount() const {
        return nodes.size();
    }

    uint64_t getBitLength() const {
        return vectorLength;
    }

    uint64_t getRootIndex() const {
        return rootIndex;
    }
};