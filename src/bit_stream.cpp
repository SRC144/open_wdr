/**
 * @file bit_stream.cpp
 * @brief Implementation of memory-based Bit Stream utilities.
 */

 #include "bit_stream.hpp"
 #include <cassert>
 
 // ============================================================================
 //  BitOutputStream Implementation
 // ============================================================================
 
 BitOutputStream::BitOutputStream(std::vector<uint8_t>& target_buffer)
     : buffer_(target_buffer), pending_byte_(0), bits_in_pending_(0) {
 }
 
 BitOutputStream::~BitOutputStream() {
     flush();
 }
 
 void BitOutputStream::write_bit(bool bit) {
     // Add bit to the temporary byte (MSB first logic)
     // If bit is true, we OR it into the correct position based on current count
     if (bit) {
         pending_byte_ |= (1 << (7 - bits_in_pending_));
     }
     
     bits_in_pending_++;
 
     // If byte is full (8 bits), push to vector and reset accumulator
     if (bits_in_pending_ == 8) {
         buffer_.push_back(pending_byte_);
         pending_byte_ = 0;
         bits_in_pending_ = 0;
     }
 }
 
 void BitOutputStream::write_bits(uint32_t value, int num_bits) {
     // Sanity check for valid bit width
     if (num_bits < 1 || num_bits > 32) {
         throw std::invalid_argument("num_bits must be between 1 and 32");
     }
 
     // Write bits MSB first to match write_bit behavior and Arithmetic Coding needs
     for (int i = num_bits - 1; i >= 0; i--) {
         bool bit = (value >> i) & 1;
         write_bit(bit);
     }
 }
 
 void BitOutputStream::flush() {
     // If there are leftover bits, push the partial byte
     // The pending_byte_ is already zero-padded at the LSB by default logic
     if (bits_in_pending_ > 0) {
         buffer_.push_back(pending_byte_);
         pending_byte_ = 0;
         bits_in_pending_ = 0;
     }
 }
 
 // ============================================================================
 //  BitInputStream Implementation
 // ============================================================================
 
 BitInputStream::BitInputStream(const std::vector<uint8_t>& source_buffer)
     : buffer_(source_buffer), byte_pos_(0), bits_consumed_in_byte_(0) {
 }
 
 bool BitInputStream::read_bit() {
     // Critical check: Prevent reading past valid memory
     if (byte_pos_ >= buffer_.size()) {
         throw std::runtime_error("BitInputStream: Buffer underflow (read past end)");
     }
 
     // Extract bit from current byte (MSB first)
     uint8_t current_byte = buffer_[byte_pos_];
     bool bit = (current_byte >> (7 - bits_consumed_in_byte_)) & 1;
 
     bits_consumed_in_byte_++;
 
     // Move to next byte if current is exhausted
     if (bits_consumed_in_byte_ == 8) {
         byte_pos_++;
         bits_consumed_in_byte_ = 0;
     }
 
     return bit;
 }
 
 uint32_t BitInputStream::read_bits(int num_bits) {
     if (num_bits < 1 || num_bits > 32) {
         throw std::invalid_argument("num_bits must be between 1 and 32");
     }
 
     uint32_t value = 0;
     // Read bits MSB first to reconstruct the integer
     for (int i = num_bits - 1; i >= 0; i--) {
         value |= (read_bit() ? 1U : 0U) << i;
     }
     return value;
 }
 
 bool BitInputStream::eof() const {
     return byte_pos_ >= buffer_.size();
 }