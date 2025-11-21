#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include "wdr_compressor.hpp"

// Helper to simulate "Pass 1" Global Threshold Calculation
double calculate_test_T(const std::vector<double>& coeffs) {
    double max_val = 0.0;
    for (double c : coeffs) {
        max_val = std::max(max_val, std::abs(c));
    }
    if (max_val == 0.0) return 1.0;
    return std::pow(2.0, std::floor(std::log2(max_val)));
}

// Test: RoundTrip_SimpleArray
TEST(WDRCompressorRoundTripTest, SimpleArray) {
    // Setup
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, -42.0, 10.0, 0.0, 3.0};
    double T = calculate_test_T(coeffs);

    // 1. Compress (Memory)
    std::vector<uint8_t> compressed = comp.compress(coeffs, T);
    ASSERT_FALSE(compressed.empty());

    // 2. Decompress (Memory)
    std::vector<double> decompressed = comp.decompress(compressed, T, coeffs.size());

    // 3. Verify
    ASSERT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6) 
            << "Mismatch at index " << i << " Input: " << coeffs[i] << " Output: " << decompressed[i];
    }
}

// Test: RoundTrip_SingleElement
TEST(WDRCompressorRoundTripTest, SingleElement) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {42.0};
    double T = calculate_test_T(coeffs); // T should be 32

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    ASSERT_EQ(decompressed.size(), 1);
    EXPECT_NEAR(decompressed[0], coeffs[0], 1e-6);
}

// Test: RoundTrip_AllZeros
TEST(WDRCompressorRoundTripTest, AllZeros) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {0.0, 0.0, 0.0, 0.0};
    double T = calculate_test_T(coeffs); // T should be 1.0

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    ASSERT_EQ(decompressed.size(), coeffs.size());
    for (double val : decompressed) {
        EXPECT_NEAR(val, 0.0, 1e-6);
    }
}

// Test: RoundTrip_AllPositive
TEST(WDRCompressorRoundTripTest, AllPositive) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_AllNegative
TEST(WDRCompressorRoundTripTest, AllNegative) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {-100.0, -50.0, -25.0, -12.5};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_MixedSigns
TEST(WDRCompressorRoundTripTest, MixedSigns) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, -50.0, 25.0, -12.5, 0.0};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_LargeValues
TEST(WDRCompressorRoundTripTest, LargeValues) {
    // Need more passes for high precision on large numbers if we want < 1e-6 error
    // 1000 -> T=512. 29 passes gets us down to ~1e-6 resolution.
    WDRCompressor comp(29); 
    std::vector<double> coeffs = {1000.0, 500.0, 250.0};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-5);
    }
}

// Test: RoundTrip_SmallValues
TEST(WDRCompressorRoundTripTest, SmallValues) {
    WDRCompressor comp(20);
    std::vector<double> coeffs = {0.1, 0.2, 0.3, 0.4};
    double T = calculate_test_T(coeffs); // T will be < 1.0 (e.g. 0.25)

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_UniformValues
TEST(WDRCompressorRoundTripTest, UniformValues) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {42.0, 42.0, 42.0, 42.0};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_SparseArray
TEST(WDRCompressorRoundTripTest, SparseArray) {
    WDRCompressor comp(26);
    std::vector<double> coeffs(1000, 0.0);
    coeffs[10] = 100.0;
    coeffs[500] = -50.0;
    coeffs[999] = 25.0;
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_LowPrecision
TEST(WDRCompressorRoundTripTest, LowPrecision) {
    // Use 8 passes. T=64. Stop threshold = 64 / 2^8 = 0.25.
    // Expect quantization errors around 0.125
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        // Allow larger error due to low pass count
        EXPECT_NEAR(decompressed[i], coeffs[i], 0.25); 
    }
}

// Test: RoundTrip_MultiplePasses
TEST(WDRCompressorRoundTripTest, MultiplePasses) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25};
    double T = calculate_test_T(coeffs);

    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());

    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RoundTrip_ProgressivePrecision
TEST(WDRCompressorRoundTripTest, ProgressivePrecision) {
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125};
    double T = calculate_test_T(coeffs);
    
    double prev_max_error = 1000.0;

    // Test increasing pass counts
    for (int num_passes = 2; num_passes <= 8; num_passes += 2) {
        WDRCompressor comp(num_passes);
        
        auto compressed = comp.compress(coeffs, T);
        auto decompressed = comp.decompress(compressed, T, coeffs.size());
        
        double current_max_error = 0.0;
        for (size_t i = 0; i < coeffs.size(); i++) {
            current_max_error = std::max(current_max_error, std::abs(decompressed[i] - coeffs[i]));
        }
        
        // Error should decrease (or stay same) as passes increase
        EXPECT_LE(current_max_error, prev_max_error);
        prev_max_error = current_max_error;
    }
}

// Test: RoundTrip_EmptyArray
TEST(WDRCompressorRoundTripTest, EmptyArray) {
    WDRCompressor comp(16);
    std::vector<double> coeffs;
    
    // Should act gracefully (return empty)
    auto compressed = comp.compress(coeffs, 1.0);
    EXPECT_TRUE(compressed.empty());
}

// Test: RoundTrip_DifferentNumPasses
TEST(WDRCompressorRoundTripTest, DifferentNumPasses) {
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    double T = calculate_test_T(coeffs);
    
    // 1. Compress with 26 passes
    WDRCompressor comp_enc(26);
    auto compressed = comp_enc.compress(coeffs, T);
    
    // 2. Decompress with 16 passes
    // Ideally, we should be able to decode the first 16 passes of a 26-pass stream
    // BUT: Arithmetic coding state must match exactly.
    // If the decoder stops reading early, that's fine.
    // If the decoder expects 16 passes but the stream has 26, it will just stop after 16.
    // This tests "Progressive Decoding" capability.
    WDRCompressor comp_dec(16); 
    auto decompressed = comp_dec.decompress(compressed, T, coeffs.size());
    
    // Result should match roughly (lossy due to fewer passes)
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 0.1); 
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}