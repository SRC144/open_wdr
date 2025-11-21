#include <gtest/gtest.h>
#include "wdr_compressor.hpp"
#include <vector>

// ============================================================================
//  Differential Coding Unit Tests
// ============================================================================

// Test: DifferentialEncode_BasicSequence
TEST(WDRCompressorTest, DifferentialEncode_BasicSequence) {
    // Initialize with default passes (required by new constructor)
    WDRCompressor comp(16);
    
    std::vector<int> indices = {1, 2, 5, 36, 42};
    std::vector<int> diff = comp.differential_encode(indices);
    
    ASSERT_EQ(diff.size(), 5);
    EXPECT_EQ(diff[0], 1);   // First element preserved
    EXPECT_EQ(diff[1], 1);   // 2 - 1 = 1
    EXPECT_EQ(diff[2], 3);   // 5 - 2 = 3
    EXPECT_EQ(diff[3], 31);  // 36 - 5 = 31
    EXPECT_EQ(diff[4], 6);   // 42 - 36 = 6
}

// Test: DifferentialEncode_SingleElement
TEST(WDRCompressorTest, DifferentialEncode_SingleElement) {
    WDRCompressor comp(16);
    std::vector<int> indices = {5};
    std::vector<int> diff = comp.differential_encode(indices);
    
    ASSERT_EQ(diff.size(), 1);
    EXPECT_EQ(diff[0], 5);
}

// Test: DifferentialEncode_ConsecutiveIndices
TEST(WDRCompressorTest, DifferentialEncode_ConsecutiveIndices) {
    WDRCompressor comp(16);
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> diff = comp.differential_encode(indices);
    
    ASSERT_EQ(diff.size(), 5);
    EXPECT_EQ(diff[0], 0);
    for (size_t i = 1; i < diff.size(); i++) {
        EXPECT_EQ(diff[i], 1);
    }
}

// Test: DifferentialEncode_LargeGaps
TEST(WDRCompressorTest, DifferentialEncode_LargeGaps) {
    WDRCompressor comp(16);
    std::vector<int> indices = {0, 100, 200};
    std::vector<int> diff = comp.differential_encode(indices);
    
    ASSERT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 0);
    EXPECT_EQ(diff[1], 100);
    EXPECT_EQ(diff[2], 100);
}

// Test: DifferentialDecode_BasicSequence
TEST(WDRCompressorTest, DifferentialDecode_BasicSequence) {
    WDRCompressor comp(16);
    std::vector<int> diff = {1, 1, 3, 31, 6};
    std::vector<int> indices = comp.differential_decode(diff);
    
    ASSERT_EQ(indices.size(), 5);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);   // 1 + 1
    EXPECT_EQ(indices[2], 5);   // 2 + 3
    EXPECT_EQ(indices[3], 36);  // 5 + 31
    EXPECT_EQ(indices[4], 42);  // 36 + 6
}

// Test: DifferentialEncodeDecode_RoundTrip
TEST(WDRCompressorTest, DifferentialEncodeDecode_RoundTrip) {
    WDRCompressor comp(16);
    
    // Test various sequences
    std::vector<std::vector<int>> test_sequences = {
        {1, 2, 5, 36, 42},
        {0, 1, 2, 3, 4},
        {0, 100, 200},
        {5},
        {10, 20, 30, 40, 50}
    };
    
    for (const auto& indices : test_sequences) {
        std::vector<int> diff = comp.differential_encode(indices);
        std::vector<int> decoded = comp.differential_decode(diff);
        EXPECT_EQ(decoded, indices);
    }
}

// Test: DifferentialEncode_EmptyInput
TEST(WDRCompressorTest, DifferentialEncode_EmptyInput) {
    WDRCompressor comp(16);
    std::vector<int> empty;
    std::vector<int> diff = comp.differential_encode(empty);
    EXPECT_TRUE(diff.empty());
}

// Test: DifferentialEncode_WithZeros
TEST(WDRCompressorTest, DifferentialEncode_WithZeros) {
    WDRCompressor comp(16);
    std::vector<int> indices = {0, 5, 10};
    std::vector<int> diff = comp.differential_encode(indices);
    
    ASSERT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 0);
    EXPECT_EQ(diff[1], 5);
    EXPECT_EQ(diff[2], 5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}