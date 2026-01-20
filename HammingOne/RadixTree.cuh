struct RadixNode {
    int64_t children[2];      // indices of child nodes, -1 if child doesn't exist
    int64_t indicesOffset;    // offset into the flat indices array for this leaf
    int64_t indicesCount;     // number of vector indices at this leaf (0 if not a leaf)

    RadixNode() {
        children[0] = -1;
        children[1] = -1;
        indicesOffset = -1;
        indicesCount = 0;
    }
};

class RadixTree {
private:
    std::vector<RadixNode> nodes;
    std::vector<uint64_t> leafIndices;  // flat array of all vector indices at leaves
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

        // Navigate to the leaf
        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];
            if (nodes[currentIdx].children[bit] == -1) {
                uint64_t newNodeIdx = allocateNode();
                nodes[currentIdx].children[bit] = newNodeIdx;
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        // Add this vector index to the leaf
        if (nodes[currentIdx].indicesCount == 0) {
            // First vector at this leaf
            nodes[currentIdx].indicesOffset = leafIndices.size();
        }
        leafIndices.push_back(vectorIndex);
        nodes[currentIdx].indicesCount++;
    }

    // Search and return all vector indices at the matching leaf (empty vector if not found)
    std::vector<uint64_t> searchAll(const uint8_t* bits) const {
        uint64_t currentIdx = rootIndex;

        for (uint64_t i = 0; i < vectorLength; ++i) {
            uint8_t bit = bits[i];
            if (nodes[currentIdx].children[bit] == -1) {
                return std::vector<uint64_t>(); // Not found
            }
            currentIdx = nodes[currentIdx].children[bit];
        }

        // Return all indices at this leaf
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

        // Flip each bit and search
        for (uint64_t i = 0; i < vectorLength; ++i) {
            modified[i] = 1 - modified[i];

            std::vector<uint64_t> foundIndices = searchAll(modified.data());
            for (uint64_t foundIndex : foundIndices) {
                if (foundIndex != queryIndex) {  // Don't match with itself
                    results.push_back(foundIndex);
                }
            }

            modified[i] = 1 - modified[i]; // Flip back
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
};