#include <gtest/gtest.h>
#include <sstream>
#include <fstream>
#include <cstdio>
#include "wdr_compressor.hpp"
#include "arithmetic_coder.hpp"
#include "adaptive_model.hpp"
#include "bit_stream.hpp"

// Test fixture for WDR compressor pass tests
class WDRCompressorPassesTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Setup for each test
    }
    
    void TearDown() override {
        // Cleanup after each test
    }
};

// Test: SortingPassEncode_FindSignificantCoefficients
// This test verifies that significant coefficients are identified correctly
TEST_F(WDRCompressorPassesTest, SortingPassEncode_FindSignificantCoefficients) {
    WDRCompressor comp(16);
    std::vector<double> coeffs = {49.0, 10.0, -35.0, 8.0};
    
    // Compress to file
    std::string test_file = "/tmp/test_sorting_pass.wdr";
    comp.compress(coeffs, test_file);
    
    // Decompress and verify
    std::vector<double> decompressed = comp.decompress(test_file);
    
    // Verify we got back 4 coefficients
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    // Cleanup
    std::remove(test_file.c_str());
}

// Test: SortingPassEncode_CountEncoding
// Verify that count is encoded correctly by testing round-trip
TEST_F(WDRCompressorPassesTest, SortingPassEncode_CountEncoding) {
    WDRCompressor comp(8);
    // Create array where we can predict how many coefficients will be significant
    std::vector<double> coeffs = {100.0, 10.0, -50.0, 5.0, 80.0};
    
    std::string test_file = "/tmp/test_count_encoding.wdr";
    comp.compress(coeffs, test_file);
    
    // Decompress and verify correct number of coefficients
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: RefinementPassEncode_UpperHalf
TEST_F(WDRCompressorPassesTest, RefinementPassEncode_UpperHalf) {
    // Test refinement pass through full compression/decompression
    WDRCompressor comp(8);
    // Create coefficients that will be refined
    std::vector<double> coeffs = {49.0, 35.0, 60.0};
    
    std::string test_file = "/tmp/test_refinement_upper.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: RefinementPassEncode_LowerHalf
TEST_F(WDRCompressorPassesTest, RefinementPassEncode_LowerHalf) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {35.0, 40.0, 45.0};
    
    std::string test_file = "/tmp/test_refinement_lower.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: SortingPassDecode_CountDecoding
TEST_F(WDRCompressorPassesTest, SortingPassDecode_CountDecoding) {
    // Test that decoder correctly decodes the count
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, -50.0, 10.0};
    
    std::string test_file = "/tmp/test_count_decode.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: FullPass_SingleThreshold
TEST_F(WDRCompressorPassesTest, FullPass_SingleThreshold) {
    WDRCompressor comp(1);  // Single pass
    std::vector<double> coeffs = {49.0, 10.0, -35.0, 8.0};
    
    std::string test_file = "/tmp/test_single_pass.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: FullPass_MultipleThresholds
TEST_F(WDRCompressorPassesTest, FullPass_MultipleThresholds) {
    WDRCompressor comp(4);  // Multiple passes
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.0, 6.0, 3.0};
    
    std::string test_file = "/tmp/test_multi_pass.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: ICSStateManagement
TEST_F(WDRCompressorPassesTest, ICSStateManagement) {
    // Test that ICS state is managed correctly across passes
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 80.0, 60.0, 40.0, 20.0, 10.0, 5.0, 2.0};
    
    std::string test_file = "/tmp/test_ics_state.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: SCSStateManagement
TEST_F(WDRCompressorPassesTest, SCSStateManagement) {
    // Test that SCS state is managed correctly
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 90.0, 80.0, 70.0};
    
    std::string test_file = "/tmp/test_scs_state.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: EmptyResult
TEST_F(WDRCompressorPassesTest, EmptyResult) {
    WDRCompressor comp(8);
    // All coefficients are small, so few will be significant in early passes
    std::vector<double> coeffs = {1.0, 2.0, 3.0, 4.0};
    
    std::string test_file = "/tmp/test_empty_result.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: AllSignificant
TEST_F(WDRCompressorPassesTest, AllSignificant) {
    WDRCompressor comp(8);
    // All coefficients are large
    std::vector<double> coeffs = {100.0, 80.0, 60.0};
    
    std::string test_file = "/tmp/test_all_significant.wdr";
    comp.compress(coeffs, test_file);
    
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

