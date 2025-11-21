#include "bit_stream.hpp"
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

// ============================================================================
//  BitOutputStream Tests
// ============================================================================

TEST(BitStreamTest, WriteSingleBit) {
  std::vector<uint8_t> buffer;
  BitOutputStream bit_stream(buffer);

  // Write bits: 0, 1, 0, 1, 1, 0, 1, 0
  // MSB First logic: 01011010
  bit_stream.write_bit(false); // 0
  bit_stream.write_bit(true);  // 1
  bit_stream.write_bit(false); // 0
  bit_stream.write_bit(true);  // 1
  bit_stream.write_bit(true);  // 1
  bit_stream.write_bit(false); // 0
  bit_stream.write_bit(true);  // 1
  bit_stream.write_bit(false); // 0
  bit_stream.flush();

  ASSERT_EQ(buffer.size(), 1);
  // Byte value: 0b01011010 = 0x5A
  EXPECT_EQ(buffer[0], 0x5A);
}

TEST(BitStreamTest, WriteMultipleBits) {
  std::vector<uint8_t> buffer;
  BitOutputStream bit_stream(buffer);

  // Write value 0b10110101 (0xB5) with 8 bits
  bit_stream.write_bits(0b10110101, 8);
  bit_stream.flush();

  ASSERT_EQ(buffer.size(), 1);
  EXPECT_EQ(buffer[0], 0xB5);
}

TEST(BitStreamTest, FlushPadding) {
  std::vector<uint8_t> buffer;
  BitOutputStream bit_stream(buffer);

  // Write 5 bits: 1, 0, 1, 0, 1
  bit_stream.write_bit(true);  // 1 (MSB)
  bit_stream.write_bit(false); // 0
  bit_stream.write_bit(true);  // 1
  bit_stream.write_bit(false); // 0
  bit_stream.write_bit(true);  // 1

  // Flush should pad the remaining 3 bits with zeros
  // Result: 10101000 = 0xA8
  bit_stream.flush();

  ASSERT_EQ(buffer.size(), 1);
  EXPECT_EQ(buffer[0], 0xA8);
}

// ============================================================================
//  BitInputStream Tests
// ============================================================================

TEST(BitStreamTest, ReadSingleBit) {
  // Setup input: 0b01011010 (0x5A)
  std::vector<uint8_t> buffer = {0x5A};
  BitInputStream bit_stream(buffer);

  // Read bits MSB first
  EXPECT_EQ(bit_stream.read_bit(), false); // 0
  EXPECT_EQ(bit_stream.read_bit(), true);  // 1
  EXPECT_EQ(bit_stream.read_bit(), false); // 0
  EXPECT_EQ(bit_stream.read_bit(), true);  // 1
  EXPECT_EQ(bit_stream.read_bit(), true);  // 1
  EXPECT_EQ(bit_stream.read_bit(), false); // 0
  EXPECT_EQ(bit_stream.read_bit(), true);  // 1
  EXPECT_EQ(bit_stream.read_bit(), false); // 0
}

TEST(BitStreamTest, ReadMultipleBits) {
  std::vector<uint8_t> buffer = {0xB5}; // 0b10110101
  BitInputStream bit_stream(buffer);

  uint32_t value = bit_stream.read_bits(8);
  EXPECT_EQ(value, 0xB5);
}

TEST(BitStreamTest, EndOfFile) {
  std::vector<uint8_t> buffer = {0x01}; // One byte
  BitInputStream bit_stream(buffer);

  // Read 8 bits (consuming the byte)
  for (int i = 0; i < 8; i++) {
    EXPECT_FALSE(bit_stream.eof());
    bit_stream.read_bit();
  }

  // Should be at EOF now (byte exhausted)
  // Note: Depending on implementation details, EOF might trigger
  // ON the 9th read or AFTER the 8th.
  // Your implementation checks `byte_pos >= buffer.size()`.
  // After 8th bit, byte_pos increments to 1. buffer.size() is 1.
  // So eof() should return true immediately.
  EXPECT_TRUE(bit_stream.eof());

  // Reading beyond should throw
  EXPECT_THROW(bit_stream.read_bit(), std::runtime_error);
}

TEST(BitStreamTest, RoundTrip) {
  std::vector<uint8_t> buffer;
  BitOutputStream out_stream(buffer);

  // 16 bits pattern
  std::vector<bool> bits = {true, false, true, true,  false, false, true, false,
                            true, true,  true, false, false, false, true, true};

  for (bool bit : bits) {
    out_stream.write_bit(bit);
  }
  out_stream.flush();

  // Should produce exactly 2 bytes
  ASSERT_EQ(buffer.size(), 2);

  // Read back
  BitInputStream in_stream(buffer);

  for (size_t i = 0; i < bits.size(); i++) {
    bool val = in_stream.read_bit();
    EXPECT_EQ(val, bits[i]) << "Bit mismatch at index " << i;
  }
}

TEST(BitStreamTest, CrossByteBoundary) {
  std::vector<uint8_t> buffer;
  BitOutputStream out_stream(buffer);

  // Write 10 bits (crosses byte boundary)
  // 1, 0, 1, 0, 1, 0, 1, 0 | 1, 0
  for (int i = 0; i < 10; i++) {
    out_stream.write_bit((i % 2) ==
                         0); // Even index = true (1), Odd = false (0)
  }
  out_stream.flush();

  // Should take 2 bytes (8 + 2 padded)
  ASSERT_EQ(buffer.size(), 2);

  // Read back
  BitInputStream in_stream(buffer);

  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(in_stream.read_bit(), (i % 2) == 0);
  }
}

TEST(BitStreamTest, EmptyStream) {
  std::vector<uint8_t> buffer;
  BitInputStream in_stream(buffer);

  // Should be at EOF immediately
  EXPECT_TRUE(in_stream.eof());

  // Reading should throw
  EXPECT_THROW(in_stream.read_bit(), std::runtime_error);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}