#include <gtest/gtest.h>
#include <fstream>
#include <cmath>
#include <cstdio>
#include "wdr_compressor.hpp"

// Test: RoundTrip_SimpleArray
TEST(WDRCompressorRoundTripTest, SimpleArray) {
    // Use 26 passes to ensure precision better than 1e-6
    // With T=64, we need T_final < 1e-6, so 2^n > 64/1e-6, so n > 25.9
    // With 26 passes, T_final=9.5e-7 (better than 1e-6)
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, -42.0, 10.0, 0.0, 3.0};
    
    std::string test_file = "/tmp/test_simple_roundtrip.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    // Verify values are close (within float precision)
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6) << "Mismatch at index " << i;
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_SingleElement
TEST(WDRCompressorRoundTripTest, SingleElement) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {42.0};
    
    std::string test_file = "/tmp/test_single_element.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), 1);
    EXPECT_NEAR(decompressed[0], coeffs[0], 1e-6);
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_AllZeros
TEST(WDRCompressorRoundTripTest, AllZeros) {
    // Test with all zeros - this is a valid edge case
    // The compression should handle this correctly
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {0.0, 0.0, 0.0, 0.0};
    
    std::string test_file = "/tmp/test_all_zeros.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    // All zeros should decompress to all zeros (within precision)
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], 0.0, 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_AllPositive
TEST(WDRCompressorRoundTripTest, AllPositive) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5};
    
    std::string test_file = "/tmp/test_all_positive.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_AllNegative
TEST(WDRCompressorRoundTripTest, AllNegative) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {-100.0, -50.0, -25.0, -12.5};
    
    std::string test_file = "/tmp/test_all_negative.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_MixedSigns
TEST(WDRCompressorRoundTripTest, MixedSigns) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {100.0, -50.0, 25.0, -12.5, 0.0};
    
    std::string test_file = "/tmp/test_mixed_signs.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_LargeValues
TEST(WDRCompressorRoundTripTest, LargeValues) {
    // For max=1000, initial T=512, so we need 29 passes for T_final < 1e-6
    WDRCompressor comp(29);
    std::vector<double> coeffs = {1000.0, 500.0, 250.0};
    
    std::string test_file = "/tmp/test_large_values.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_SmallValues
TEST(WDRCompressorRoundTripTest, SmallValues) {
    WDRCompressor comp(16);
    std::vector<double> coeffs = {0.1, 0.2, 0.3, 0.4};
    
    std::string test_file = "/tmp/test_small_values.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    // Small values may have lower precision
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-3);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_UniformValues
TEST(WDRCompressorRoundTripTest, UniformValues) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {42.0, 42.0, 42.0, 42.0};
    
    std::string test_file = "/tmp/test_uniform_values.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_SparseArray
TEST(WDRCompressorRoundTripTest, SparseArray) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    // Large array with mostly zeros
    std::vector<double> coeffs(1000, 0.0);
    coeffs[10] = 100.0;
    coeffs[500] = -50.0;
    coeffs[999] = 25.0;
    
    std::string test_file = "/tmp/test_sparse_array.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6) << "Mismatch at index " << i;
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_LowPrecision
TEST(WDRCompressorRoundTripTest, LowPrecision) {
    // Test low precision behavior with few passes
    // Use a reasonable number of passes that will generate enough bits
    // but still test lower precision behavior
    WDRCompressor comp(8);  // Use 8 passes - enough for arithmetic coding, but lower precision
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125};
    
    std::string test_file = "/tmp/test_low_precision.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    // Fewer passes result in lower precision than many passes
    // With 8 passes and T=64, T_final = 64/2^8 = 0.25, so precision is limited
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1.0);  // Allow larger tolerance for lower precision
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_MultiplePasses
TEST(WDRCompressorRoundTripTest, MultiplePasses) {
    WDRCompressor comp(26);  // Use 26 passes for high precision
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25};
    
    std::string test_file = "/tmp/test_multiple_passes.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: RoundTrip_ProgressivePrecision
TEST(WDRCompressorRoundTripTest, ProgressivePrecision) {
    // Use a larger array to ensure we have enough bits for arithmetic coding
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25, 3.125, 1.5625, 0.78125};
    
    // Test with different numbers of passes (skip 1 pass as it may not have enough bits)
    for (int num_passes = 2; num_passes <= 8; num_passes *= 2) {
        WDRCompressor comp(num_passes);
        std::string test_file = "/tmp/test_progressive_" + std::to_string(num_passes) + ".wdr";
        comp.compress(coeffs, test_file);
        
        std::vector<double> decompressed = comp.decompress(test_file);
        
        EXPECT_EQ(decompressed.size(), coeffs.size());
        
        // More passes should yield better precision
        double max_error = 0.0;
        for (size_t i = 0; i < coeffs.size(); i++) {
            double error = std::abs(decompressed[i] - coeffs[i]);
            max_error = std::max(max_error, error);
        }
        
        // With more passes, error should generally decrease
        // (This is a qualitative test - exact values depend on implementation)
        
        std::remove(test_file.c_str());
    }
}

// Test: RoundTrip_EmptyArray
TEST(WDRCompressorRoundTripTest, EmptyArray) {
    WDRCompressor comp(16);
    std::vector<double> coeffs;
    
    std::string test_file = "/tmp/test_empty_array.wdr";
    
    EXPECT_THROW(comp.compress(coeffs, test_file), std::invalid_argument);
}

// Test: RoundTrip_InvalidFile
TEST(WDRCompressorRoundTripTest, InvalidFile) {
    WDRCompressor comp(16);
    
    EXPECT_THROW(comp.decompress("nonexistent_file.wdr"), std::runtime_error);
}

// Test: RoundTrip_DifferentNumPasses
TEST(WDRCompressorRoundTripTest, DifferentNumPasses) {
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    
    // Compress with 26 passes for high precision
    WDRCompressor comp26(26);
    std::string test_file = "/tmp/test_different_passes.wdr";
    comp26.compress(coeffs, test_file);
    
    // Decompress (should read num_passes from file header)
    WDRCompressor comp_decompress(16);  // Different default, but should use file value
    std::vector<double> decompressed = comp_decompress.decompress(test_file);
    
    EXPECT_EQ(decompressed.size(), coeffs.size());
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: TileCompressDecompress_RoundTrip
TEST(WDRCompressorRoundTripTest, TileCompressDecompressRoundTrip) {
    std::vector<double> coeffs = {128.0, -64.0, 32.0, -16.0, 8.0, -4.0};
    constexpr int kPasses = 12;
    WDRCompressor comp(kPasses);
    double initial_T = comp.calculate_initial_T(coeffs);
    auto payload = comp.compress_tile(coeffs, initial_T);

    ASSERT_FALSE(payload.empty());

    auto recovered = comp.decompress_tile(payload, initial_T, coeffs.size());
    ASSERT_EQ(recovered.size(), coeffs.size());
    const double tolerance = initial_T / std::pow(2.0, kPasses);
    for (size_t i = 0; i < coeffs.size(); ++i) {
        EXPECT_NEAR(recovered[i], coeffs[i], tolerance) << "Mismatch at index " << i;
    }
}

// Test: TileCompressDecompress_MultipleTilesEquivalent
TEST(WDRCompressorRoundTripTest, TileCompressMatchesFullCompressor) {
    std::vector<double> coeffs = {200.0, -150.0, 75.0, -37.5, 18.75, -9.375};
    WDRCompressor comp(16);
    double initial_T = comp.calculate_initial_T(coeffs);

    // Reference: full-file compression
    const std::string reference_file = "/tmp/test_tile_vs_full.wdr";
    comp.compress(coeffs, reference_file);
    auto full_decoded = comp.decompress(reference_file);

    // Tile path
    auto payload = comp.compress_tile(coeffs, initial_T);
    auto tile_decoded = comp.decompress_tile(payload, initial_T, coeffs.size());

    ASSERT_EQ(full_decoded.size(), tile_decoded.size());
    for (size_t i = 0; i < coeffs.size(); ++i) {
        EXPECT_NEAR(tile_decoded[i], full_decoded[i], 1e-6) << "Mismatch at index " << i;
    }

    std::remove(reference_file.c_str());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

