struct RadixNode {
    int children[2];     // Indices into the node array (-1 if null)
    int vectorIndex;     // -1 if not a leaf, vector index otherwise

    RadixNode() : vectorIndex(-1) {
        children[0] = -1;
        children[1] = -1;
    }
};

class RadixTree {
private:
    std::vector<RadixNode> nodes;
    int bitLength;
    int rootIndex;

    int allocateNode() {
        nodes.push_back(RadixNode());
        return (int)nodes.size() - 1;
    }

public:
    RadixTree(int l) : bitLength(l), rootIndex(0) {
        allocateNode(); // Root node creation
    }

    void insert(const uint8_t* bits, int vectorIndex) {
        int currentIdx = rootIndex;

        for (int i = 0; i < bitLength; ++i) {
            uint8_t bit = bits[i];

            if (nodes[currentIdx].children[bit] == -1) {
                int newNodeIdx = allocateNode();
                nodes[currentIdx].children[bit] = newNodeIdx;
            }

            currentIdx = nodes[currentIdx].children[bit];
        }

        nodes[currentIdx].vectorIndex = vectorIndex;
    }

    int search(const uint8_t* bits) const {
        int currentIdx = rootIndex;

        for (int i = 0; i < bitLength; ++i) {
            uint8_t bit = bits[i];

            if (nodes[currentIdx].children[bit] == -1) {
                return -1;
            }

            currentIdx = nodes[currentIdx].children[bit];
        }

        return nodes[currentIdx].vectorIndex;
    }

    void findHammingDistanceOne(const uint8_t* bits, int queryIndex, std::vector<int>& results) const {
        std::vector<uint8_t> modified(bitLength);

        for (int i = 0; i < bitLength; ++i) {
            modified[i] = bits[i];
        }

        // Iterate through each bit
        for (int i = 0; i < bitLength; ++i) {
            modified[i] = 1 - modified[i];

            int foundIndex = search(modified.data());

            if (foundIndex != -1 && foundIndex != queryIndex) {
                results.push_back(foundIndex);
            }

            modified[i] = 1 - modified[i];
        }
    }

    const std::vector<RadixNode>& getNodes() const {
        return nodes;
    }

    int getNodeCount() const {
        return nodes.size();
    }

    int getBitLength() const {
        return bitLength;
    }
};