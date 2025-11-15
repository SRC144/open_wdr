#include "arithmetic_coder.hpp"
#include <stdexcept>
#include <cassert>

void ArithmeticCoder::start_encoding(BitOutputStream& stream, AdaptiveModel& model) {
    output_stream_ = &stream;
    encoding_model_ = &model;
    encoding_ = true;
    decoding_ = false;
    
    low_ = 0;
    high_ = TOP_VALUE;
    bits_to_follow_ = 0;
}

void ArithmeticCoder::encode_symbol(int symbol, AdaptiveModel& model) {
    if (!encoding_) {
        throw std::runtime_error("ArithmeticCoder: not in encoding mode");
    }
    
    const std::vector<int>& cum_freq = model.get_cumulative_freq();
    
    // Get the internal index for this symbol (1-indexed)
    int symbol_index = model.get_internal_index(symbol);
    
    if (symbol_index < 1 || symbol_index > model.get_num_symbols()) {
        throw std::invalid_argument("Symbol index out of range");
    }
    
    // Calculate the current range
    uint64_t range = static_cast<uint64_t>(high_ - low_) + 1;
    
    // Narrow the encoding interval to the range for this symbol
    // Note: cum_freq uses backward cumulative frequencies
    // Range for symbol is [cum_freq[symbol], cum_freq[symbol-1])
    high_ = low_ + (range * cum_freq[symbol_index - 1]) / cum_freq[0] - 1;
    low_ = low_ + (range * cum_freq[symbol_index]) / cum_freq[0];
    
    // Output bits and scale the interval
    for (;;) {
        if (high_ < HALF) {
            // Entire range is in lower half, output 0
            bit_plus_follow(false);
        } else if (low_ >= HALF) {
            // Entire range is in upper half, output 1
            bit_plus_follow(true);
            low_ -= HALF;
            high_ -= HALF;
        } else if (low_ >= FIRST_QTR && high_ < THIRD_QTR) {
            // Range is in middle half, handle underflow
            bits_to_follow_++;
            low_ -= FIRST_QTR;
            high_ -= FIRST_QTR;
        } else {
            // No bits can be output yet, exit loop
            break;
        }
        
        // Scale up the range
        low_ = 2 * low_;
        high_ = 2 * high_ + 1;
    }
}

void ArithmeticCoder::done_encoding() {
    if (!encoding_) {
        throw std::runtime_error("ArithmeticCoder: not in encoding mode");
    }
    
    // Output two bits to uniquely identify the final interval
    bits_to_follow_++;
    if (low_ < FIRST_QTR) {
        bit_plus_follow(false);
    } else {
        bit_plus_follow(true);
    }
    
    // Flush the output stream
    output_stream_->flush();
    
    encoding_ = false;
}

void ArithmeticCoder::start_decoding(BitInputStream& stream, AdaptiveModel& model) {
    input_stream_ = &stream;
    decoding_model_ = &model;
    decoding_ = true;
    encoding_ = false;
    
    // Read up to CODE_VALUE_BITS bits (pad with zeros if stream is shorter)
    value_ = 0;
    for (int i = 0; i < CODE_VALUE_BITS; i++) {
        bool bit = false;
        try {
            bit = stream.read_bit();
        } catch (const std::runtime_error&) {
            // End of stream: pad remaining bits with zeros
            // to keep encoder/decoder intervals aligned
            bit = false;
        }
        value_ = 2 * value_ + (bit ? 1 : 0);
    }
    
    low_ = 0;
    high_ = TOP_VALUE;
}

int ArithmeticCoder::decode_symbol(AdaptiveModel& model) {
    if (!decoding_) {
        throw std::runtime_error("ArithmeticCoder: not in decoding mode");
    }
    
    const std::vector<int>& cum_freq = model.get_cumulative_freq();
    int num_symbols = model.get_num_symbols();
    
    // Calculate the current range
    uint64_t range = static_cast<uint64_t>(high_ - low_) + 1;
    
    // Find the cumulative frequency corresponding to the current value
    uint64_t cum = (((static_cast<uint64_t>(value_ - low_) + 1) * cum_freq[0] - 1) / range);
    
    // Find the symbol index that corresponds to this cumulative frequency
    int symbol_index = 1;
    while (symbol_index <= num_symbols && cum_freq[symbol_index] > static_cast<int>(cum)) {
        symbol_index++;
    }
    
    // Ensure we found a valid symbol
    if (symbol_index > num_symbols) {
        symbol_index = num_symbols;
    }
    
    // Narrow the decoding interval to the range for this symbol
    high_ = low_ + (range * cum_freq[symbol_index - 1]) / cum_freq[0] - 1;
    low_ = low_ + (range * cum_freq[symbol_index]) / cum_freq[0];
    
    // Scale the interval and read new bits
    // Safety limit to prevent infinite loops during EOF handling
    // Under normal conditions, scaling should complete in < 32 iterations for 32-bit values
    const int MAX_SCALING_ITERATIONS = 64;
    int iteration_count = 0;
    
    for (;;) {
        // Safety check: prevent infinite loops
        // This can occur if the bitstream ends unexpectedly or if there's encoder/decoder mismatch
        if (++iteration_count > MAX_SCALING_ITERATIONS) {
            throw std::runtime_error("ArithmeticCoder: exceeded maximum scaling iterations. "
                                   "This may indicate corrupted data or encoder/decoder mismatch.");
        }
        
        if (high_ < HALF) {
            // Expand lower half (no change to value)
            // Do nothing, just scale
        } else if (low_ >= HALF) {
            // Expand upper half
            value_ -= HALF;
            low_ -= HALF;
            high_ -= HALF;
        } else if (low_ >= FIRST_QTR && high_ < THIRD_QTR) {
            // Expand middle half
            value_ -= FIRST_QTR;
            low_ -= FIRST_QTR;
            high_ -= FIRST_QTR;
        } else {
            // No scaling needed, exit loop
            break;
        }
        
        // Check for EOF before attempting to read
        if (input_stream_->eof()) {
            // Out of bits - we've reached the end of the stream
            // At this point, we should stop scaling and use the current interval
            // Pad value with zeros (conservative approach for end of stream)
            value_ = 2 * value_;
            low_ = 2 * low_;
            high_ = 2 * high_ + 1;
            // Break immediately when EOF is reached to avoid infinite loop
            // The interval at this point should be sufficient for decoding
            break;
        }
        
        // Scale up the range and read a new bit
        low_ = 2 * low_;
        high_ = 2 * high_ + 1;
        try {
            value_ = 2 * value_ + (input_stream_->read_bit() ? 1 : 0);
        } catch (const std::runtime_error& e) {
            // Stream ended unexpectedly while reading - this is an error condition
            // Pad value with zero and break to avoid infinite loop
            value_ = 2 * value_;
            // Break immediately when stream ends unexpectedly
            break;
        }
    }
    
    // Convert internal index (1-indexed) to symbol (0-indexed) using the model's mapping
    return model.get_symbol_from_internal_index(symbol_index);
}

void ArithmeticCoder::bit_plus_follow(bool bit) {
    if (!encoding_) {
        throw std::runtime_error("ArithmeticCoder: not in encoding mode");
    }
    
    // Output the bit
    output_stream_->write_bit(bit);
    
    // Output opposite bits for underflow handling
    while (bits_to_follow_ > 0) {
        output_stream_->write_bit(!bit);
        bits_to_follow_--;
    }
}

