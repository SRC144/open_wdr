#include <gtest/gtest.h>
#include <fstream>
#include <cstring>
#include <cstdio>
#include "wdr_compressor.hpp"
#include "wdr_file_format.hpp"

// Test: WriteHeader_Basic
TEST(WDRFileFormatTest, WriteHeader_Basic) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    
    std::string test_file = "/tmp/test_header_basic.wdr";
    comp.compress(coeffs, test_file);
    
    // Read header manually
    std::ifstream file(test_file, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    
    // Read magic number
    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    EXPECT_EQ(magic, WDRFormat::MAGIC);
    
    // Read version
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(version));
    EXPECT_EQ(version, WDRFormat::VERSION);
    
    file.close();
    std::remove(test_file.c_str());
}

// Test: ReadHeader_Basic
TEST(WDRFileFormatTest, ReadHeader_Basic) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    
    std::string test_file = "/tmp/test_read_header.wdr";
    comp.compress(coeffs, test_file);
    
    // Decompress should read header correctly
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: ReadHeader_InvalidMagic
TEST(WDRFileFormatTest, ReadHeader_InvalidMagic) {
    // Create a file with invalid magic number
    std::string test_file = "/tmp/test_invalid_magic.wdr";
    std::ofstream file(test_file, std::ios::binary);
    
    uint32_t invalid_magic = 0xDEADBEEF;
    file.write(reinterpret_cast<const char*>(&invalid_magic), sizeof(invalid_magic));
    file.close();
    
    WDRCompressor comp(8);
    EXPECT_THROW(comp.decompress(test_file), std::runtime_error);
    
    std::remove(test_file.c_str());
}

// Test: ReadHeader_InvalidVersion
TEST(WDRFileFormatTest, ReadHeader_InvalidVersion) {
    // Create a file with invalid version
    std::string test_file = "/tmp/test_invalid_version.wdr";
    std::ofstream file(test_file, std::ios::binary);
    
    uint32_t magic = WDRFormat::MAGIC;
    uint32_t invalid_version = 999;
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    file.write(reinterpret_cast<const char*>(&invalid_version), sizeof(invalid_version));
    file.close();
    
    WDRCompressor comp(8);
    EXPECT_THROW(comp.decompress(test_file), std::runtime_error);
    
    std::remove(test_file.c_str());
}

// Test: WriteReadHeader_RoundTrip
TEST(WDRFileFormatTest, WriteReadHeader_RoundTrip) {
    // Use 26 passes to ensure T_final < 1e-6 even for coefficients up to 100
    WDRCompressor comp(26);
    std::vector<double> coeffs = {100.0, 50.0, 25.0, 12.5, 6.25};
    
    std::string test_file = "/tmp/test_header_roundtrip.wdr";
    comp.compress(coeffs, test_file);
    
    // Decompress and verify
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    // Verify values
    for (size_t i = 0; i < coeffs.size(); i++) {
        EXPECT_NEAR(decompressed[i], coeffs[i], 1e-6);
    }
    
    std::remove(test_file.c_str());
}

// Test: Header_InitialT
TEST(WDRFileFormatTest, Header_InitialT) {
    WDRCompressor comp(8);
    // Use coefficients that will give a specific initial T
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    
    std::string test_file = "/tmp/test_initial_t.wdr";
    comp.compress(coeffs, test_file);
    
    // Decompress and verify it works (initial T is read from header)
    std::vector<double> decompressed = comp.decompress(test_file);
    EXPECT_EQ(decompressed.size(), coeffs.size());
    
    std::remove(test_file.c_str());
}

// Test: Header_NumPasses
TEST(WDRFileFormatTest, Header_NumPasses) {
    // Test with different numbers of passes
    for (int num_passes = 1; num_passes <= 16; num_passes *= 2) {
        WDRCompressor comp(num_passes);
        std::vector<double> coeffs = {100.0, 50.0};
        
        std::string test_file = "/tmp/test_num_passes_" + std::to_string(num_passes) + ".wdr";
        comp.compress(coeffs, test_file);
        
        // Decompress (should use num_passes from header)
        WDRCompressor decomp_comp(99);  // Different default
        std::vector<double> decompressed = decomp_comp.decompress(test_file);
        
        EXPECT_EQ(decompressed.size(), coeffs.size());
        
        std::remove(test_file.c_str());
    }
}

// Test: Header_NumCoeffs
TEST(WDRFileFormatTest, Header_NumCoeffs) {
    WDRCompressor comp(8);
    // Test with different array sizes
    std::vector<std::vector<double>> test_arrays = {
        {100.0},
        {100.0, 50.0},
        {100.0, 50.0, 25.0, 12.5},
        std::vector<double>(100, 10.0)
    };
    
    for (size_t i = 0; i < test_arrays.size(); i++) {
        std::string test_file = "/tmp/test_num_coeffs_" + std::to_string(i) + ".wdr";
        comp.compress(test_arrays[i], test_file);
        
        std::vector<double> decompressed = comp.decompress(test_file);
        EXPECT_EQ(decompressed.size(), test_arrays[i].size());
        
        std::remove(test_file.c_str());
    }
}

// Test: Header_DataSize
TEST(WDRFileFormatTest, Header_DataSize) {
    WDRCompressor comp(8);
    std::vector<double> coeffs = {100.0, 50.0, 25.0};
    
    std::string test_file = "/tmp/test_data_size.wdr";
    comp.compress(coeffs, test_file);
    
    // Verify file exists and has reasonable size
    std::ifstream file(test_file, std::ios::binary | std::ios::ate);
    ASSERT_TRUE(file.is_open());
    std::streampos file_size = file.tellg();
    file.close();
    
    // File should be at least header size
    EXPECT_GE(file_size, WDRFormat::HEADER_SIZE);
    
    std::remove(test_file.c_str());
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

