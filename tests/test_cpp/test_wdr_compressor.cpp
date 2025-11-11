#include <gtest/gtest.h>
#include "wdr_compressor.hpp"
#include <cmath>
#include <vector>
#include <cstdio>

// Test: CalculateInitialT_BasicCase
TEST(WDRCompressorTest, CalculateInitialT_BasicCase) {
    WDRCompressor comp;
    std::vector<double> coeffs = {100.0, 42.0, 10.0, 3.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // Expected: T should be 64 (largest power of 2 where 100 >= T and 100 < 2*T)
    // 100 is in [64, 128), so T = 64
    EXPECT_DOUBLE_EQ(T, 64.0);
    EXPECT_GE(100.0, T);
    EXPECT_LT(100.0, 2.0 * T);
}

// Test: CalculateInitialT_PowerOfTwo
TEST(WDRCompressorTest, CalculateInitialT_PowerOfTwo) {
    WDRCompressor comp;
    std::vector<double> coeffs = {64.0, 32.0, 16.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // 64 is exactly 2*32, so we need T such that 64 < 2*T, so T > 32
    // T should be 32 (64 is in [32, 64), wait no - 64 is exactly 2*32
    // Actually, the condition is: max_abs >= T and max_abs < 2*T
    // 64 >= 32 and 64 < 64 is false, so T should be 32, and we check 64 < 128
    // Let's verify: if T=32, then 64 >= 32 (true) and 64 < 64 (false)
    // So we need T such that 64 < 2*T, i.e., T > 32, so T = 64
    // But then 64 >= 64 (true) and 64 < 128 (true), so T = 64
    EXPECT_GE(T, 32.0);
    EXPECT_LT(64.0, 2.0 * T);
}

// Test: CalculateInitialT_AllZeros
TEST(WDRCompressorTest, CalculateInitialT_AllZeros) {
    WDRCompressor comp;
    std::vector<double> coeffs = {0.0, 0.0, 0.0};
    
    double T = comp.calculate_initial_T(coeffs);
    EXPECT_EQ(T, 1.0);  // Default threshold
}

// Test: CalculateInitialT_NegativeValues
TEST(WDRCompressorTest, CalculateInitialT_NegativeValues) {
    WDRCompressor comp;
    std::vector<double> coeffs = {-100.0, 50.0, -25.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // Should use absolute values
    EXPECT_DOUBLE_EQ(T, 64.0);
    EXPECT_GE(100.0, T);
    EXPECT_LT(100.0, 2.0 * T);
}

// Test: CalculateInitialT_SingleElement
TEST(WDRCompressorTest, CalculateInitialT_SingleElement) {
    WDRCompressor comp;
    std::vector<double> coeffs = {42.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // 42 is in [32, 64), so T = 32
    EXPECT_DOUBLE_EQ(T, 32.0);
}

// Test: CalculateInitialT_VeryLargeValues
TEST(WDRCompressorTest, CalculateInitialT_VeryLargeValues) {
    WDRCompressor comp;
    std::vector<double> coeffs = {1000.0, 500.0, 250.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // 1000 is in [512, 1024), so T = 512
    EXPECT_DOUBLE_EQ(T, 512.0);
    EXPECT_GE(1000.0, T);
    EXPECT_LT(1000.0, 2.0 * T);
}

// Test: CalculateInitialT_UniformValues
TEST(WDRCompressorTest, CalculateInitialT_UniformValues) {
    WDRCompressor comp;
    std::vector<double> coeffs = {42.0, 42.0, 42.0};
    
    double T = comp.calculate_initial_T(coeffs);
    // All same value, should still calculate correctly
    EXPECT_DOUBLE_EQ(T, 32.0);
}

// Test: BinaryReduce_BasicValues
TEST(WDRCompressorTest, BinaryReduce_BasicValues) {
    WDRCompressor comp;
    
    // Test value 1: should reduce to empty vector
    std::vector<bool> reduced_1 = comp.binary_reduce(1);
    EXPECT_TRUE(reduced_1.empty());
    
    // Test value 2: binary 10, MSB removed -> 0
    std::vector<bool> reduced_2 = comp.binary_reduce(2);
    EXPECT_EQ(reduced_2.size(), 1);
    EXPECT_EQ(reduced_2[0], false);
    
    // Test value 3: binary 11, MSB removed -> 1
    std::vector<bool> reduced_3 = comp.binary_reduce(3);
    EXPECT_EQ(reduced_3.size(), 1);
    EXPECT_EQ(reduced_3[0], true);
    
    // Test value 19: binary 10011, MSB removed -> 0011 (LSB first: 1,1,0,0)
    std::vector<bool> reduced_19 = comp.binary_reduce(19);
    EXPECT_EQ(reduced_19.size(), 4);
    EXPECT_EQ(reduced_19[0], true);   // LSB first
    EXPECT_EQ(reduced_19[1], true);
    EXPECT_EQ(reduced_19[2], false);
    EXPECT_EQ(reduced_19[3], false);
}

// Test: BinaryReduce_PowersOfTwo
TEST(WDRCompressorTest, BinaryReduce_PowersOfTwo) {
    WDRCompressor comp;
    
    // Test powers of 2
    std::vector<bool> reduced_2 = comp.binary_reduce(2);   // 10 -> 0
    EXPECT_EQ(reduced_2.size(), 1);
    EXPECT_EQ(reduced_2[0], false);
    
    std::vector<bool> reduced_4 = comp.binary_reduce(4);   // 100 -> 00
    EXPECT_EQ(reduced_4.size(), 2);
    EXPECT_EQ(reduced_4[0], false);
    EXPECT_EQ(reduced_4[1], false);
    
    std::vector<bool> reduced_8 = comp.binary_reduce(8);   // 1000 -> 000
    EXPECT_EQ(reduced_8.size(), 3);
    for (bool bit : reduced_8) {
        EXPECT_EQ(bit, false);
    }
}

// Test: BinaryExpand_BasicValues
TEST(WDRCompressorTest, BinaryExpand_BasicValues) {
    WDRCompressor comp;
    
    // Test empty vector -> 1
    std::vector<bool> empty;
    EXPECT_EQ(comp.binary_expand(empty), 1);
    
    // Test [0] -> 2 (binary 10)
    std::vector<bool> bits_2 = {false};
    EXPECT_EQ(comp.binary_expand(bits_2), 2);
    
    // Test [1] -> 3 (binary 11)
    std::vector<bool> bits_3 = {true};
    EXPECT_EQ(comp.binary_expand(bits_3), 3);
    
    // Test [1,1,0,0] -> 19 (binary 10011)
    std::vector<bool> bits_19 = {true, true, false, false};
    EXPECT_EQ(comp.binary_expand(bits_19), 19);
}

// Test: BinaryReduceExpand_RoundTrip
TEST(WDRCompressorTest, BinaryReduceExpand_RoundTrip) {
    WDRCompressor comp;
    
    // Test values from 1 to 100
    for (int value = 1; value <= 100; value++) {
        std::vector<bool> reduced = comp.binary_reduce(value);
        int expanded = comp.binary_expand(reduced);
        EXPECT_EQ(expanded, value) << "Round-trip failed for value " << value;
    }
}

// Test: BinaryReduce_ValueOne
TEST(WDRCompressorTest, BinaryReduce_ValueOne) {
    WDRCompressor comp;
    std::vector<bool> reduced = comp.binary_reduce(1);
    EXPECT_TRUE(reduced.empty());
}

// Test: BinaryExpand_EmptyVector
TEST(WDRCompressorTest, BinaryExpand_EmptyVector) {
    WDRCompressor comp;
    std::vector<bool> empty;
    EXPECT_EQ(comp.binary_expand(empty), 1);
}

// Test: BinaryReduce_InvalidInput
TEST(WDRCompressorTest, BinaryReduce_InvalidInput) {
    WDRCompressor comp;
    EXPECT_THROW(comp.binary_reduce(0), std::invalid_argument);
    EXPECT_THROW(comp.binary_reduce(-1), std::invalid_argument);
}

// Test: DifferentialEncode_BasicSequence
TEST(WDRCompressorTest, DifferentialEncode_BasicSequence) {
    WDRCompressor comp;
    std::vector<int> indices = {1, 2, 5, 36, 42};
    std::vector<int> diff = comp.differential_encode(indices);
    
    EXPECT_EQ(diff.size(), 5);
    EXPECT_EQ(diff[0], 1);   // First element preserved
    EXPECT_EQ(diff[1], 1);   // 2 - 1 = 1
    EXPECT_EQ(diff[2], 3);   // 5 - 2 = 3
    EXPECT_EQ(diff[3], 31);  // 36 - 5 = 31
    EXPECT_EQ(diff[4], 6);   // 42 - 36 = 6
}

// Test: DifferentialEncode_SingleElement
TEST(WDRCompressorTest, DifferentialEncode_SingleElement) {
    WDRCompressor comp;
    std::vector<int> indices = {5};
    std::vector<int> diff = comp.differential_encode(indices);
    
    EXPECT_EQ(diff.size(), 1);
    EXPECT_EQ(diff[0], 5);
}

// Test: DifferentialEncode_ConsecutiveIndices
TEST(WDRCompressorTest, DifferentialEncode_ConsecutiveIndices) {
    WDRCompressor comp;
    std::vector<int> indices = {0, 1, 2, 3, 4};
    std::vector<int> diff = comp.differential_encode(indices);
    
    EXPECT_EQ(diff.size(), 5);
    EXPECT_EQ(diff[0], 0);
    for (size_t i = 1; i < diff.size(); i++) {
        EXPECT_EQ(diff[i], 1);
    }
}

// Test: DifferentialEncode_LargeGaps
TEST(WDRCompressorTest, DifferentialEncode_LargeGaps) {
    WDRCompressor comp;
    std::vector<int> indices = {0, 100, 200};
    std::vector<int> diff = comp.differential_encode(indices);
    
    EXPECT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 0);
    EXPECT_EQ(diff[1], 100);
    EXPECT_EQ(diff[2], 100);
}

// Test: DifferentialDecode_BasicSequence
TEST(WDRCompressorTest, DifferentialDecode_BasicSequence) {
    WDRCompressor comp;
    std::vector<int> diff = {1, 1, 3, 31, 6};
    std::vector<int> indices = comp.differential_decode(diff);
    
    EXPECT_EQ(indices.size(), 5);
    EXPECT_EQ(indices[0], 1);
    EXPECT_EQ(indices[1], 2);   // 1 + 1
    EXPECT_EQ(indices[2], 5);   // 2 + 3
    EXPECT_EQ(indices[3], 36);  // 5 + 31
    EXPECT_EQ(indices[4], 42);  // 36 + 6
}

// Test: DifferentialEncodeDecode_RoundTrip
TEST(WDRCompressorTest, DifferentialEncodeDecode_RoundTrip) {
    WDRCompressor comp;
    
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
    WDRCompressor comp;
    std::vector<int> empty;
    std::vector<int> diff = comp.differential_encode(empty);
    EXPECT_TRUE(diff.empty());
}

// Test: DifferentialEncode_WithZeros
TEST(WDRCompressorTest, DifferentialEncode_WithZeros) {
    WDRCompressor comp;
    std::vector<int> indices = {0, 5, 10};
    std::vector<int> diff = comp.differential_encode(indices);
    
    EXPECT_EQ(diff.size(), 3);
    EXPECT_EQ(diff[0], 0);
    EXPECT_EQ(diff[1], 5);
    EXPECT_EQ(diff[2], 5);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

