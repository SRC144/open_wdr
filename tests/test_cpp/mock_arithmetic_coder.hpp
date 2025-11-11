#ifndef MOCK_ARITHMETIC_CODER_HPP
#define MOCK_ARITHMETIC_CODER_HPP

#include "arithmetic_coder.hpp"
#include "adaptive_model.hpp"
#include "bit_stream.hpp"
#include <vector>
#include <string>
#include <memory>

/**
 * Mock ArithmeticCoder for testing.
 * 
 * Records all encode/decode operations for verification.
 */
class MockArithmeticCoder {
public:
    struct RecordedCall {
        enum CallType {
            ENCODE,
            DECODE
        };
        CallType type;
        void* model_ptr;  // Pointer to identify which model was used
        int symbol;
    };

    MockArithmeticCoder() : encoding_(false), decoding_(false), decode_index_(0) {}

    void start_encoding(BitOutputStream& stream, AdaptiveModel& model) {
        (void)stream;  // Unused in mock
        encoding_ = true;
        decoding_ = false;
        encoding_model_ = &model;
        recorded_calls_.clear();
    }

    void encode_symbol(int symbol, AdaptiveModel& model) {
        if (!encoding_) {
            throw std::runtime_error("MockArithmeticCoder: not in encoding mode");
        }
        RecordedCall call;
        call.type = RecordedCall::ENCODE;
        call.model_ptr = &model;
        call.symbol = symbol;
        recorded_calls_.push_back(call);
    }

    void done_encoding() {
        encoding_ = false;
    }

    void start_decoding(BitInputStream& stream, AdaptiveModel& model) {
        (void)stream;  // Unused in mock
        decoding_ = true;
        encoding_ = false;
        decoding_model_ = &model;
        decode_index_ = 0;
    }

    int decode_symbol(AdaptiveModel& model) {
        if (!decoding_) {
            throw std::runtime_error("MockArithmeticCoder: not in decoding mode");
        }
        if (decode_index_ >= preloaded_sequence_.size()) {
            throw std::runtime_error("MockArithmeticCoder: no more symbols to decode");
        }
        
        RecordedCall call;
        call.type = RecordedCall::DECODE;
        call.model_ptr = &model;
        call.symbol = preloaded_sequence_[decode_index_];
        recorded_calls_.push_back(call);
        
        return preloaded_sequence_[decode_index_++];
    }

    bool is_encoding() const { return encoding_; }
    bool is_decoding() const { return decoding_; }

    // Access recorded calls
    const std::vector<RecordedCall>& get_recorded_calls() const {
        return recorded_calls_;
    }

    void clear_recorded_calls() {
        recorded_calls_.clear();
    }

    // Preload sequence for decoding tests
    void preload_sequence(const std::vector<int>& sequence) {
        preloaded_sequence_ = sequence;
        decode_index_ = 0;
    }

    // Get model pointer from recorded call
    static void* get_model_ptr(const RecordedCall& call) {
        return call.model_ptr;
    }

    // Check if two model pointers are the same
    static bool same_model(const RecordedCall& call1, const RecordedCall& call2) {
        return call1.model_ptr == call2.model_ptr;
    }

private:
    bool encoding_;
    bool decoding_;
    AdaptiveModel* encoding_model_;
    AdaptiveModel* decoding_model_;
    std::vector<RecordedCall> recorded_calls_;
    std::vector<int> preloaded_sequence_;
    size_t decode_index_;
};

#endif // MOCK_ARITHMETIC_CODER_HPP

