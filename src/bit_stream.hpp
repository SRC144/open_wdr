#ifndef BIT_STREAM_HPP
#define BIT_STREAM_HPP

/**
 * @file bit_stream.hpp
 * @brief Bit-level I/O streams for arithmetic coding
 * 
 * This module provides bit-level input and output streams for reading and
 * writing bits to/from byte streams. These streams are used by the arithmetic
 * coder to write and read compressed data at the bit level.
 * 
 * Key features:
 * - Bit buffering: Accumulates bits in a buffer before writing bytes
 * - MSB-first ordering: Bits are written and read in MSB-first order to match
 *   arithmetic coding expectations
 * - Automatic flushing: Buffered bits are automatically flushed when needed
 */

#include <iostream>
#include <cstdint>

/**
 * @brief Bit-level output stream for writing bits to a file
 * 
 * This class wraps an std::ostream to provide bit-level writing capabilities.
 * Bits are accumulated in an internal buffer and written as bytes when the
 * buffer is full or when flush() is called.
 * 
 * Bits are written in MSB-first order to match the arithmetic coder's
 * expectations. The buffer automatically flushes when it contains 8 bits,
 * or when flush() is explicitly called.
 * 
 * @note The destructor automatically flushes remaining bits, ensuring no
 *       data is lost.
 */
class BitOutputStream {
public:
    /**
     * Constructor.
     * 
     * @param stream Output stream to write bits to
     */
    explicit BitOutputStream(std::ostream& stream);
    
    /**
     * Destructor. Automatically flushes remaining bits.
     */
    ~BitOutputStream();
    
    /**
     * Write a single bit to the stream.
     * 
     * @param bit Bit value to write (true = 1, false = 0)
     */
    void write_bit(bool bit);
    
    /**
     * Write multiple bits to the stream (MSB first).
     * 
     * @param value Value to write
     * @param num_bits Number of bits to write (1-32)
     */
    void write_bits(uint32_t value, int num_bits);
    
    /**
     * Flush the internal buffer to the stream.
     * Pads the last byte with zeros if necessary.
     */
    void flush();
    
    // Disable copy and assignment
    BitOutputStream(const BitOutputStream&) = delete;
    BitOutputStream& operator=(const BitOutputStream&) = delete;

private:
    std::ostream& stream_;
    uint8_t buffer_;
    int bits_in_buffer_;
};

/**
 * @brief Bit-level input stream for reading bits from a file
 * 
 * This class wraps an std::istream to provide bit-level reading capabilities.
 * Bytes are read from the stream and bits are extracted one at a time.
 * 
 * Bits are read in MSB-first order to match the arithmetic coder's
 * expectations. The buffer is automatically refilled when empty.
 * 
 * @note The stream throws std::runtime_error when attempting to read past EOF.
 */
class BitInputStream {
public:
    /**
     * Constructor.
     * 
     * @param stream Input stream to read bits from
     */
    explicit BitInputStream(std::istream& stream);
    
    /**
     * Read a single bit from the stream.
     * 
     * @return Bit value (true = 1, false = 0)
     * @throws std::runtime_error if EOF is reached
     */
    bool read_bit();
    
    /**
     * Read multiple bits from the stream (MSB first).
     * 
     * @param num_bits Number of bits to read (1-32)
     * @return Value read from stream
     * @throws std::runtime_error if EOF is reached
     */
    uint32_t read_bits(int num_bits);
    
    /**
     * Check if end of file has been reached.
     * 
     * @return True if EOF, false otherwise
     */
    bool eof() const;
    
    // Disable copy and assignment
    BitInputStream(const BitInputStream&) = delete;
    BitInputStream& operator=(const BitInputStream&) = delete;

private:
    std::istream& stream_;
    uint8_t buffer_;
    int bits_in_buffer_;
    bool eof_flag_;
    
    /**
     * Fill the buffer with the next byte from the stream.
     */
    void fill_buffer();
};

#endif // BIT_STREAM_HPP

