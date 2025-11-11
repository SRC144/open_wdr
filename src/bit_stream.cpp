#include "bit_stream.hpp"
#include <stdexcept>
#include <cassert>

BitOutputStream::BitOutputStream(std::ostream& stream)
    : stream_(stream), buffer_(0), bits_in_buffer_(0) {
}

BitOutputStream::~BitOutputStream() {
    flush();
}

void BitOutputStream::write_bit(bool bit) {
    // Add bit to buffer
    buffer_ |= (bit ? 1 : 0) << (7 - bits_in_buffer_);
    bits_in_buffer_++;
    
    // Write buffer if full
    if (bits_in_buffer_ == 8) {
        stream_.write(reinterpret_cast<const char*>(&buffer_), 1);
        if (!stream_.good()) {
            throw std::runtime_error("Failed to write to output stream");
        }
        buffer_ = 0;
        bits_in_buffer_ = 0;
    }
}

void BitOutputStream::write_bits(uint32_t value, int num_bits) {
    if (num_bits < 1 || num_bits > 32) {
        throw std::invalid_argument("num_bits must be between 1 and 32");
    }
    
    // Write bits MSB first (to match write_bit behavior and arithmetic coding)
    for (int i = num_bits - 1; i >= 0; i--) {
        bool bit = (value >> i) & 1;
        write_bit(bit);
    }
}

void BitOutputStream::flush() {
    // Write remaining bits with padding
    if (bits_in_buffer_ > 0) {
        stream_.write(reinterpret_cast<const char*>(&buffer_), 1);
        if (!stream_.good()) {
            throw std::runtime_error("Failed to flush output stream");
        }
        buffer_ = 0;
        bits_in_buffer_ = 0;
    }
    stream_.flush();
}

BitInputStream::BitInputStream(std::istream& stream)
    : stream_(stream), buffer_(0), bits_in_buffer_(0), eof_flag_(false) {
    fill_buffer();
}

bool BitInputStream::read_bit() {
    if (eof_flag_) {
        throw std::runtime_error("Attempted to read past end of file");
    }
    
    // Extract bit from buffer
    bool bit = (buffer_ >> (7 - bits_in_buffer_)) & 1;
    bits_in_buffer_++;
    
    // Refill buffer if empty
    if (bits_in_buffer_ == 8) {
        fill_buffer();
    }
    
    return bit;
}

uint32_t BitInputStream::read_bits(int num_bits) {
    if (num_bits < 1 || num_bits > 32) {
        throw std::invalid_argument("num_bits must be between 1 and 32");
    }
    
    uint32_t value = 0;
    // Read bits MSB first (to match read_bit behavior and arithmetic coding)
    for (int i = num_bits - 1; i >= 0; i--) {
        bool bit = read_bit();
        value |= (bit ? 1U : 0U) << i;
    }
    
    return value;
}

bool BitInputStream::eof() const {
    return eof_flag_ && bits_in_buffer_ == 0;
}

void BitInputStream::fill_buffer() {
    if (stream_.eof() || !stream_.good()) {
        eof_flag_ = true;
        buffer_ = 0;
        bits_in_buffer_ = 0;
        return;
    }
    
    char byte;
    stream_.read(&byte, 1);
    
    if (stream_.gcount() == 0) {
        eof_flag_ = true;
        buffer_ = 0;
        bits_in_buffer_ = 0;
    } else {
        buffer_ = static_cast<uint8_t>(byte);
        bits_in_buffer_ = 0;
    }
}

