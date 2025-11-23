#ifndef BIT_STREAM_HPP
#define BIT_STREAM_HPP

/**
 * @file bit_stream.hpp
 * @brief Memory-based Bit Stream utilities for WDR compression.
 * * This file defines the BitOutputStream and BitInputStream classes, which
 * facilitate writing and reading individual bits to and from underlying 
 * byte vectors (std::vector<uint8_t>).
 * * Key Features:
 * - Zero-Copy Architecture: operates on references to existing vectors.
 * - MSB First: Bits are packed from Most Significant Bit to Least Significant Bit.
 * - Exceptions: Throws std::runtime_error on buffer underflows/overflows.
 */

#include <vector>
#include <cstdint>
#include <stdexcept>

/**
 * @brief Writes bits to a dynamically growing memory buffer.
 * * The BitOutputStream allows writing single bits or multi-bit integers
 * into a std::vector<uint8_t>. It handles the buffering of partial bytes
 * internally.
 */
class BitOutputStream {
public:
    /**
     * @brief Construct a new Bit Output Stream object.
     * * @param target_buffer Reference to the output vector. The vector is NOT cleared
     * on construction; bits are appended to existing content.
     * The caller retains ownership of this vector.
     */
    explicit BitOutputStream(std::vector<uint8_t>& target_buffer);

    /**
     * @brief Destroy the Bit Output Stream object.
     * * Automatically calls flush() to ensure any remaining partial bits
     * are written to the buffer.
     */
    ~BitOutputStream();

    /**
     * @brief Write a single bit to the stream.
     * * The bit is added to an internal accumulator. When 8 bits are accumulated,
     * a byte is pushed to the target vector.
     * * @param bit The bit to write (true = 1, false = 0).
     */
    void write_bit(bool bit);

    /**
     * @brief Write multiple bits of an integer to the stream.
     * * Bits are written starting from the MSB (Most Significant Bit) down to the LSB.
     * * @param value The integer value containing the bits to write.
     * @param num_bits The number of bits to write (1 to 32).
     * @throw std::invalid_argument If num_bits is not between 1 and 32.
     */
    void write_bits(uint32_t value, int num_bits);

    /**
     * @brief Flushes any pending bits to the output vector.
     * * If there are bits in the temporary buffer that haven't formed a full byte
     * yet, they are padded with zeros (at the LSB end) and pushed to the vector.
     */
    void flush();

private:
    std::vector<uint8_t>& buffer_; ///< Reference to the user-owned output vector.
    uint8_t pending_byte_;         ///< Accumulator for bits currently being built.
    int bits_in_pending_;          ///< Count of bits currently in the accumulator (0-7).
};

/**
 * @brief Reads bits from a read-only memory buffer.
 * * The BitInputStream allows reading single bits or multi-bit integers
 * from a std::vector<uint8_t>. It is designed to throw exceptions on
 * underflow to stop the Arithmetic Coder safely.
 */
class BitInputStream {
public:
    /**
     * @brief Construct a new Bit Input Stream object.
     * * @param source_buffer Reference to the input vector containing compressed data.
     */
    explicit BitInputStream(const std::vector<uint8_t>& source_buffer);

    /**
     * @brief Read a single bit from the stream.
     * * @return true if the bit is 1.
     * @return false if the bit is 0.
     * @throw std::runtime_error If attempting to read past the end of the buffer.
     */
    bool read_bit();

    /**
     * @brief Read multiple bits to form an integer.
     * * Bits are read and reconstructed MSB first.
     * * @param num_bits The number of bits to read (1 to 32).
     * @return uint32_t The constructed integer value.
     * @throw std::invalid_argument If num_bits is not between 1 and 32.
     */
    uint32_t read_bits(int num_bits);

    /**
     * @brief Check if the end of the stream has been reached.
     * * @return true If no more bytes are available in the buffer.
     * @return false If there is still data to read.
     */
    bool eof() const;

private:
    const std::vector<uint8_t>& buffer_; ///< Reference to the input data.
    size_t byte_pos_;                    ///< Current index in the byte vector.
    int bits_consumed_in_byte_;          ///< Current bit index (0-7) within the current byte.
};

#endif // BIT_STREAM_HPP