#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <algorithm>
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

// Test fixture for WDR compressor pass tests
class WDRCompressorPassesTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test: SortingPassEncode_FindSignificantCoefficients
TEST_F(WDRCompressorPassesTest, SortingPassEncode_FindSignificantCoefficients) {
    WDRCompressor comp(26);
    std::vector<double> coeffs = {49.0, 10.0, -35.0, 8.0};
    double T = calculate_test_T(coeffs);
    
    // Compress
    auto compressed = comp.compress(coeffs, T);
    
    // Decompress
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    // Verify size and values
    ASSERT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: SortingPassEncode_CountEncoding
TEST_F(WDRCompressorPassesTest, SortingPassEncode_CountEncoding) {
    WDRCompressor comp(8);
    // Array designed to trigger specific count logic in sorting pass
    std::vector<double> coeffs = {100.0, 10.0, -50.0, 5.0, 80.0};
    double T = calculate_test_T(coeffs);
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1.0); // Lower precision with 8 passes
    }
}

// Test: RefinementPassEncode_UpperHalf
TEST_F(WDRCompressorPassesTest, RefinementPassEncode_UpperHalf) {
    // Test refinement logic where value is in the upper half of the interval [T, 2T)
    WDRCompressor comp(26);
    std::vector<double> coeffs = {49.0, 35.0, 60.0}; // Assuming T=32
    double T = calculate_test_T(coeffs); // T=32
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: RefinementPassEncode_LowerHalf
TEST_F(WDRCompressorPassesTest, RefinementPassEncode_LowerHalf) {
    // Test refinement logic where value is in the lower half of the interval
    WDRCompressor comp(26);
    std::vector<double> coeffs = {35.0, 40.0, 45.0}; // Assuming T=32
    double T = calculate_test_T(coeffs); // T=32
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
}

// Test: SortingPassDecode_CountDecoding
TEST_F(WDRCompressorPassesTest, SortingPassDecode_CountDecoding) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, -50.0, 10.0};
    double T = calculate_test_T(coeffs);
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
}

// Test: FullPass_SingleThreshold
TEST_F(WDRCompressorPassesTest, FullPass_SingleThreshold) {
    // Test compression with just 1 pass (extreme quantization)
    WDRCompressor comp(1); 
    std::vector<double> coeffs = {49.0, 10.0, -35.0, 8.0};
    double T = calculate_test_T(coeffs); // T=32
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    // With 1 pass: 
    // 49 (>=32) -> Becomes 32 + 16 = 48
    // 10 (<32)  -> Becomes 0
    // -35 (>=32) -> Becomes -48
    // 8 (<32)   -> Becomes 0
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    EXPECT_NEAR(decompressed[0], 48.0, 1e-6);
    EXPECT_NEAR(decompressed[1], 0.0, 1e-6);
    EXPECT_NEAR(decompressed[2], -48.0, 1e-6);
    EXPECT_NEAR(decompressed[3], 0.0, 1e-6);
}

// Test: FullPass_MultipleThresholds
TEST_F(WDRCompressorPassesTest, FullPass_MultipleThresholds) {
    WDRCompressor comp(4);
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.0, 6.0, 3.0};
    double T = calculate_test_T(coeffs);
     
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    // 4 passes should capture basic structure
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 8.0); // Coarse check
    }
}

// Test: ICSStateManagement
TEST_F(WDRCompressorPassesTest, ICSStateManagement) {
    // Test correct management of Insignificant Set over passes
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0};
    double T = calculate_test_T(coeffs);
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1.0);
    }
}

// Test: SCSStateManagement
TEST_F(WDRCompressorPassesTest, SCSStateManagement) {
    // Test correct management of Significant Set (Refinement)
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 90.0, 80.0, 70.0};
    double T = calculate_test_T(coeffs);
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1.0);
    }
}

// Test: EmptyResult
TEST_F(WDRCompressorPassesTest, EmptyResult) {
    // All coefficients small enough that they might not be significant in early passes
    WDRCompressor comp(2); // 2 passes with T=4 means stop at T=1.0
    std::vector<double> coeffs = {0.1, 0.2, 0.3};
    double T = 4.0; // Force high T
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    // Should all be zero because T never dropped low enough
    for (auto val : decompressed) {
        EXPECT_DOUBLE_EQ(val, 0.0);
    }
}

// Test: AllSignificant
TEST_F(WDRCompressorPassesTest, AllSignificant) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 80.0, 60.0};
    double T = calculate_test_T(coeffs);
    
    auto compressed = comp.compress(coeffs, T);
    auto decompressed = comp.decompress(compressed, T, coeffs.size());
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1.0);
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}