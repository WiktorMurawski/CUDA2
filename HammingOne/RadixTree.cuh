struct RadixNode {
    int64_t children[2];     // indeksy węzłów-dzieci, -1 jeśli dziecko nie istnieje
    int64_t vectorIndex;     // index wektora jeśli węzeł jest liściem, -1 jeśli wpp
    int64_t vectorCount;     // liczba wektorów przechowywanych w poddrzewie tego węzła

    RadixNode() {
        children[0] = -1;
        children[1] = -1;
        vectorIndex = -1;
        vectorCount = 0;
    }
};

class RadixTree {
private:
    std::vector<RadixNode> nodes;
    uint64_t vectorLength;
    uint64_t rootIndex;

    uint64_t allocateNode() {
        nodes.push_back(RadixNode());
        return (uint64_t)nodes.size() - 1;
    }

public:
    RadixTree(uint64_t l) : vectorLength(l), rootIndex(0) {
        allocateNode(); // Root
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

        nodes[currentIdx].vectorIndex = vectorIndex;
    }

    int64_t search(const uint8_t* bits) const {
        uint64_t currentIdx = rootIndex;

        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];

            if (nodes[currentIdx].children[bit] == -1) {
                return -1;
            }

            currentIdx = nodes[currentIdx].children[bit];
        }

        return nodes[currentIdx].vectorIndex;
    }

    void findHammingDistanceOne(const uint8_t* bits, uint64_t queryIndex, std::vector<uint64_t>& results) const {
        std::vector<uint8_t> modified(vectorLength);

        for (uint64_t i = 0; i < vectorLength; ++i) {
            modified[i] = bits[i];
        }

        for (uint64_t i = 0; i < vectorLength; ++i) {
            modified[i] = 1 - modified[i];

            int64_t foundIndex = search(modified.data());

            if (foundIndex != -1 && foundIndex != queryIndex) {
                results.push_back(foundIndex);
            }

            modified[i] = 1 - modified[i];
        }
    }

    const std::vector<RadixNode>& getNodes() const {
        return nodes;
    }

    uint64_t getNodeCount() const {
        return nodes.size();
    }

    uint64_t getBitLength() const {
        return vectorLength;
    }
};