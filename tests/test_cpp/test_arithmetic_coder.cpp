#include <gtest/gtest.h>
#include <sstream>
#include "arithmetic_coder.hpp"
#include "adaptive_model.hpp"
#include "bit_stream.hpp"

// Test: ArithmeticCoder_InitialState
TEST(ArithmeticCoderTest, InitialState) {
    std::stringstream ss;
    BitOutputStream bit_stream(ss);
    AdaptiveModel model(2);
    ArithmeticCoder coder;
    
    coder.start_encoding(bit_stream, model);
    
    EXPECT_TRUE(coder.is_encoding());
    EXPECT_FALSE(coder.is_decoding());
}

// Test: ArithmeticCoder_Constants
TEST(ArithmeticCoderTest, Constants) {
    EXPECT_EQ(ArithmeticCoder::CODE_VALUE_BITS, 16);
    EXPECT_EQ(ArithmeticCoder::TOP_VALUE, 65535);
    EXPECT_EQ(ArithmeticCoder::FIRST_QTR, 16384);
    EXPECT_EQ(ArithmeticCoder::HALF, 32768);
    EXPECT_EQ(ArithmeticCoder::THIRD_QTR, 49152);
}

// Test: ArithmeticCoder_SimpleSequence_ManualCalculation
TEST(ArithmeticCoderTest, SimpleSequence_ManualCalculation) {
    std::stringstream ss;
    BitOutputStream bit_stream(ss);
    AdaptiveModel model(2);
    ArithmeticCoder coder;
    
    coder.start_encoding(bit_stream, model);
    
    // Encode sequence [0, 1, 0, 1]
    // With binary model, initial frequencies are [1, 1], so cum_freq = [2, 1, 0]
    // Symbol 0: range = [1, 2) out of [0, 2)
    // Symbol 1: range = [0, 1) out of [0, 2)
    
    // After encoding symbol 0:
    // low = 0 + (65536 * 1) / 2 = 32768
    // high = 0 + (65536 * 2) / 2 - 1 = 65535
    coder.encode_symbol(0, model);
    model.update_model(0);
    
    // After encoding symbol 1:
    // Current range: [32768, 65535], size = 32768
    // Symbol 1: range = [32768, 49151] (lower half of current range)
    coder.encode_symbol(1, model);
    model.update_model(1);
    
    // Continue encoding
    coder.encode_symbol(0, model);
    model.update_model(0);
    coder.encode_symbol(1, model);
    model.update_model(1);
    
    coder.done_encoding();
    
    // Verify encoding completed
    EXPECT_FALSE(coder.is_encoding());
}

// Test: ArithmeticCoder_RoundTrip_Simple
TEST(ArithmeticCoderTest, RoundTrip_Simple) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel encode_model(2);
    AdaptiveModel decode_model(2);
    ArithmeticCoder encoder;
    ArithmeticCoder decoder;
    
    // Encode sequence
    std::vector<int> original = {0, 1, 0, 1, 1, 0, 1, 0, 1, 1};
    encoder.start_encoding(out_stream, encode_model);
    
    for (int symbol : original) {
        encoder.encode_symbol(symbol, encode_model);
        encode_model.update_model(symbol);
    }
    encoder.done_encoding();
    out_stream.flush();
    
    // Decode sequence
    ss.seekg(0);
    BitInputStream in_stream(ss);
    decoder.start_decoding(in_stream, decode_model);
    
    std::vector<int> decoded;
    for (size_t i = 0; i < original.size(); i++) {
        int symbol = decoder.decode_symbol(decode_model);
        decode_model.update_model(symbol);
        decoded.push_back(symbol);
    }
    
    // Verify round-trip
    EXPECT_EQ(decoded, original);
}

// Test: ArithmeticCoder_RoundTrip_AdaptiveModel
TEST(ArithmeticCoderTest, RoundTrip_AdaptiveModel) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel encode_model(2);
    AdaptiveModel decode_model(2);
    ArithmeticCoder encoder;
    ArithmeticCoder decoder;
    
    // Encode sequence with adaptive model
    std::vector<int> original = {0, 0, 0, 1, 1, 0, 0, 1, 1, 1};
    encoder.start_encoding(out_stream, encode_model);
    
    for (int symbol : original) {
        encoder.encode_symbol(symbol, encode_model);
        encode_model.update_model(symbol);
    }
    encoder.done_encoding();
    out_stream.flush();
    
    // Decode sequence
    ss.seekg(0);
    BitInputStream in_stream(ss);
    decoder.start_decoding(in_stream, decode_model);
    
    std::vector<int> decoded;
    for (size_t i = 0; i < original.size(); i++) {
        int symbol = decoder.decode_symbol(decode_model);
        decode_model.update_model(symbol);
        decoded.push_back(symbol);
    }
    
    // Verify round-trip
    EXPECT_EQ(decoded, original);
    
    // Verify models have same frequencies (adaptation was synchronized)
    for (int i = 0; i < 2; i++) {
        EXPECT_EQ(encode_model.get_frequency(i), decode_model.get_frequency(i));
    }
}

// Test: ArithmeticCoder_RoundTrip_ComplexSequence
TEST(ArithmeticCoderTest, RoundTrip_ComplexSequence) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel encode_model(2);
    AdaptiveModel decode_model(2);
    ArithmeticCoder encoder;
    ArithmeticCoder decoder;
    
    // Encode long, complex sequence
    std::vector<int> original;
    for (int i = 0; i < 100; i++) {
        original.push_back(i % 2);
    }
    
    encoder.start_encoding(out_stream, encode_model);
    for (int symbol : original) {
        encoder.encode_symbol(symbol, encode_model);
        encode_model.update_model(symbol);
    }
    encoder.done_encoding();
    out_stream.flush();
    
    // Decode sequence
    ss.seekg(0);
    BitInputStream in_stream(ss);
    decoder.start_decoding(in_stream, decode_model);
    
    std::vector<int> decoded;
    for (size_t i = 0; i < original.size(); i++) {
        int symbol = decoder.decode_symbol(decode_model);
        decode_model.update_model(symbol);
        decoded.push_back(symbol);
    }
    
    // Verify round-trip
    EXPECT_EQ(decoded, original);
}

// Test: ArithmeticCoder_DoneEncoding
TEST(ArithmeticCoderTest, DoneEncoding) {
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel model(2);
    ArithmeticCoder coder;
    
    coder.start_encoding(out_stream, model);
    coder.encode_symbol(0, model);
    coder.done_encoding();
    
    EXPECT_FALSE(coder.is_encoding());
    
    // Verify bits were written (done_encoding flushes final bits)
    out_stream.flush();
    EXPECT_GT(ss.str().size(), 0);
}

// Test: ArithmeticCoder_StartDecoding
TEST(ArithmeticCoderTest, StartDecoding) {
    // First encode something
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel encode_model(2);
    ArithmeticCoder encoder;
    
    encoder.start_encoding(out_stream, encode_model);
    encoder.encode_symbol(0, encode_model);
    encoder.encode_symbol(1, encode_model);
    encoder.done_encoding();
    out_stream.flush();
    
    // Now decode
    ss.seekg(0);
    BitInputStream in_stream(ss);
    AdaptiveModel decode_model(2);
    ArithmeticCoder decoder;
    
    decoder.start_decoding(in_stream, decode_model);
    
    EXPECT_TRUE(decoder.is_decoding());
    EXPECT_FALSE(decoder.is_encoding());
}

// Test: ArithmeticCoder_ModelSeparation
TEST(ArithmeticCoderTest, ModelSeparation) {
    std::stringstream ss1, ss2;
    BitOutputStream out_stream1(ss1), out_stream2(ss2);
    AdaptiveModel model1(2);
    AdaptiveModel model2(2);
    ArithmeticCoder coder1, coder2;
    
    // Encode with model1
    coder1.start_encoding(out_stream1, model1);
    coder1.encode_symbol(0, model1);
    model1.update_model(0);
    coder1.done_encoding();
    
    // Encode with model2
    coder2.start_encoding(out_stream2, model2);
    coder2.encode_symbol(1, model2);
    model2.update_model(1);
    coder2.done_encoding();
    
    // Verify models have different frequencies
    EXPECT_NE(model1.get_frequency(0), model2.get_frequency(0));
    EXPECT_NE(model1.get_frequency(1), model2.get_frequency(1));
}

// Test: ArithmeticCoder_UnderflowCondition (simplified - full test requires access to internal state)
TEST(ArithmeticCoderTest, UnderflowCondition) {
    // This test verifies that underflow handling works correctly
    // Full verification would require access to internal state (low_, high_, bits_to_follow_)
    // For now, we test that encoding/decoding works correctly even with sequences that might trigger underflow
    
    std::stringstream ss;
    BitOutputStream out_stream(ss);
    AdaptiveModel encode_model(2);
    AdaptiveModel decode_model(2);
    ArithmeticCoder encoder;
    ArithmeticCoder decoder;
    
    // Create a sequence that might trigger underflow (many alternating symbols)
    std::vector<int> original;
    for (int i = 0; i < 50; i++) {
        original.push_back(i % 2);
    }
    
    encoder.start_encoding(out_stream, encode_model);
    for (int symbol : original) {
        encoder.encode_symbol(symbol, encode_model);
        encode_model.update_model(symbol);
    }
    encoder.done_encoding();
    out_stream.flush();
    
    // Decode and verify
    ss.seekg(0);
    BitInputStream in_stream(ss);
    decoder.start_decoding(in_stream, decode_model);
    
    std::vector<int> decoded;
    for (size_t i = 0; i < original.size(); i++) {
        int symbol = decoder.decode_symbol(decode_model);
        decode_model.update_model(symbol);
        decoded.push_back(symbol);
    }
    
    EXPECT_EQ(decoded, original);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

