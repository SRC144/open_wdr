#include <gtest/gtest.h>
#include <sstream>
#include "bit_stream.hpp"

// Test: BitOutputStream_WriteSingleBit
TEST(BitStreamTest, WriteSingleBit) {
    std::stringstream ss;
    BitOutputStream bit_stream(ss);
    
    // Write bits: 0, 1, 0, 1, 1, 0, 1, 0
    bit_stream.write_bit(false);  // 0
    bit_stream.write_bit(true);   // 1
    bit_stream.write_bit(false);  // 0
    bit_stream.write_bit(true);   // 1
    bit_stream.write_bit(true);   // 1
    bit_stream.write_bit(false);  // 0
    bit_stream.write_bit(true);   // 1
    bit_stream.write_bit(false);  // 0
    bit_stream.flush();
    
    // Read back as byte
    ss.seekg(0);
    uint8_t byte = ss.get();
    // Bits written MSB first: 0,1,0,1,1,0,1,0
    // Byte value: 0b01011010 = 0x5A
    EXPECT_EQ(byte, 0x5A);
}

// Test: BitOutputStream_WriteMultipleBits
TEST(BitStreamTest, WriteMultipleBits) {
    std::stringstream ss;
    BitOutputStream bit_stream(ss);
    
    // Write value 0b10110101 with 8 bits
    bit_stream.write_bits(0b10110101, 8);
    bit_stream.flush();
    
    // Read back
    ss.seekg(0);
    uint8_t byte = ss.get();
    EXPECT_EQ(byte, 0b10110101);
}

// Test: BitOutputStream_Flush
TEST(BitStreamTest, Flush) {
    std::stringstream ss;
    BitOutputStream bit_stream(ss);
    
    // Write 5 bits MSB first
    bit_stream.write_bit(true);   // 1 (MSB)
    bit_stream.write_bit(false);  // 0
    bit_stream.write_bit(true);   // 1
    bit_stream.write_bit(false);  // 0
    bit_stream.write_bit(true);   // 1
    bit_stream.flush();  // Should pad with zeros and write
    
    // Read back
    ss.seekg(0);
    uint8_t byte = ss.get();
    // Bits written MSB first: 1,0,1,0,1, then padded with 0,0,0
    // Byte: 0b10101000 = 0xA8
    EXPECT_EQ(byte, 0xA8);
}

// Test: BitInputStream_ReadSingleBit
TEST(BitStreamTest, ReadSingleBit) {
    std::stringstream ss;
    // Write a byte first
    ss.put(0x5A);  // 0b01011010
    ss.seekg(0);
    
    BitInputStream bit_stream(ss);
    
    // Read bits MSB first (matches write_bit which writes MSB first)
    // Byte 0x5A = 0b01011010, so MSB first: 0,1,0,1,1,0,1,0
    EXPECT_EQ(bit_stream.read_bit(), false);  // 0 (MSB)
    EXPECT_EQ(bit_stream.read_bit(), true);   // 1
    EXPECT_EQ(bit_stream.read_bit(), false);  // 0
    EXPECT_EQ(bit_stream.read_bit(), true);   // 1
    EXPECT_EQ(bit_stream.read_bit(), true);   // 1
    EXPECT_EQ(bit_stream.read_bit(), false);  // 0
    EXPECT_EQ(bit_stream.read_bit(), true);   // 1
    EXPECT_EQ(bit_stream.read_bit(), false);  // 0 (LSB)
}

// Test: BitInputStream_ReadMultipleBits
TEST(BitStreamTest, ReadMultipleBits) {
    std::stringstream ss;
    ss.put(0b10110101);
    ss.seekg(0);
    
    BitInputStream bit_stream(ss);
    
    uint32_t value = bit_stream.read_bits(8);
    EXPECT_EQ(value, 0b10110101);
}

// Test: BitInputStream_EOF
TEST(BitStreamTest, EndOfFile) {
    std::stringstream ss;
    ss.put(0x01);  // One byte
    ss.seekg(0);
    
    BitInputStream bit_stream(ss);
    
    // Read 8 bits (one byte)
    for (int i = 0; i < 8; i++) {
        EXPECT_FALSE(bit_stream.eof());
        bit_stream.read_bit();
    }
    
    // Should be at EOF now
    EXPECT_TRUE(bit_stream.eof());
    
    // Reading beyond should throw or return EOF
    EXPECT_THROW(bit_stream.read_bit(), std::runtime_error);
}

// Test: BitStream_RoundTrip
TEST(BitStreamTest, RoundTrip) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    
    // Write various bit patterns
    std::vector<bool> bits = {true, false, true, true, false, false, true, false,
                              true, true, true, false, false, false, true, true};
    
    for (bool bit : bits) {
        out_stream.write_bit(bit);
    }
    out_stream.flush();
    
    // Read back
    ss.seekg(0);
    BitInputStream in_stream(ss);
    
    for (size_t i = 0; i < bits.size(); i++) {
        EXPECT_EQ(in_stream.read_bit(), bits[i]) << "Bit mismatch at position " << i;
    }
}

// Test: BitStream_CrossByteBoundary
TEST(BitStreamTest, CrossByteBoundary) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    
    // Write 10 bits (crosses byte boundary)
    for (int i = 0; i < 10; i++) {
        out_stream.write_bit((i % 2) == 0);
    }
    out_stream.flush();
    
    // Read back
    ss.seekg(0);
    BitInputStream in_stream(ss);
    
    for (int i = 0; i < 10; i++) {
        EXPECT_EQ(in_stream.read_bit(), (i % 2) == 0);
    }
}

// Test: BitStream_EmptyStream
TEST(BitStreamTest, EmptyStream) {
    std::stringstream ss;
    BitInputStream in_stream(ss);
    
    // Should be at EOF immediately
    EXPECT_TRUE(in_stream.eof());
    
    // Reading should throw
    EXPECT_THROW(in_stream.read_bit(), std::runtime_error);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

