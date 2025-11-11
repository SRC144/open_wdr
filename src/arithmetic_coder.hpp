#ifndef ARITHMETIC_CODER_HPP
#define ARITHMETIC_CODER_HPP

/**
 * @file arithmetic_coder.hpp
 * @brief Adaptive Arithmetic Coder implementation
 * 
 * This implementation is based on the adaptive arithmetic coding algorithm
 * from Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 * for data compression." Communications of the ACM, 30(6), 520-540.
 * 
 * The algorithm uses integer arithmetic to represent encoding intervals and
 * handles underflow to maintain precision. This implementation maintains
 * mathematical equivalence to the original algorithm described in the paper.
 * 
 * @see Witten, Neal, & Cleary (1987) for the original algorithm
 */

#include "bit_stream.hpp"
#include "adaptive_model.hpp"
#include <cstdint>

/**
 * @brief Adaptive Arithmetic Coder
 * 
 * This class implements the adaptive arithmetic coding algorithm from
 * Witten, Neal, Cleary (ACM 1987). It uses integer arithmetic to represent
 * the encoding interval and handles underflow to maintain precision.
 * 
 * The implementation maintains mathematical equivalence to the original C code
 * presented in the paper, using modern C++17 features for clarity and
 * type safety.
 * 
 * @see Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 *      for data compression." Communications of the ACM, 30(6), 520-540.
 */
class ArithmeticCoder {
public:
    // Constants for arithmetic coding
    static constexpr int CODE_VALUE_BITS = 16;
    static constexpr uint32_t TOP_VALUE = ((1UL << CODE_VALUE_BITS) - 1);
    static constexpr uint32_t FIRST_QTR = (TOP_VALUE / 4 + 1);
    static constexpr uint32_t HALF = (2 * FIRST_QTR);
    static constexpr uint32_t THIRD_QTR = (3 * FIRST_QTR);
    
    /**
     * Start encoding a stream of symbols.
     * 
     * @param stream Bit output stream to write encoded bits to
     * @param model Adaptive model for symbol probabilities
     */
    void start_encoding(BitOutputStream& stream, AdaptiveModel& model);
    
    /**
     * Encode a symbol.
     * 
     * @param symbol Symbol to encode (0 to num_symbols-1)
     * @param model Adaptive model for symbol probabilities
     */
    void encode_symbol(int symbol, AdaptiveModel& model);
    
    /**
     * Finish encoding the stream.
     * Flushes the final bits to uniquely identify the encoding interval.
     */
    void done_encoding();
    
    /**
     * Start decoding a stream of symbols.
     * 
     * @param stream Bit input stream to read encoded bits from
     * @param model Adaptive model for symbol probabilities
     */
    void start_decoding(BitInputStream& stream, AdaptiveModel& model);
    
    /**
     * Decode the next symbol.
     * 
     * @param model Adaptive model for symbol probabilities
     * @return Decoded symbol (0 to num_symbols-1)
     */
    int decode_symbol(AdaptiveModel& model);
    
    /**
     * Check if encoding is active.
     * 
     * @return True if encoding, false otherwise
     */
    bool is_encoding() const { return encoding_; }
    
    /**
     * Check if decoding is active.
     * 
     * @return True if decoding, false otherwise
     */
    bool is_decoding() const { return decoding_; }

private:
    // Encoding state
    uint32_t low_;              // Lower bound of encoding interval
    uint32_t high_;             // Upper bound of encoding interval
    uint32_t bits_to_follow_;   // Number of opposite bits to output (underflow handling)
    BitOutputStream* output_stream_;
    AdaptiveModel* encoding_model_;
    bool encoding_;
    
    // Decoding state
    uint32_t value_;            // Current code value being decoded
    BitInputStream* input_stream_;
    AdaptiveModel* decoding_model_;
    bool decoding_;
    
    /**
     * Output a bit plus any following opposite bits (underflow handling).
     * 
     * @param bit Bit to output (true = 1, false = 0)
     */
    void bit_plus_follow(bool bit);
};

#endif // ARITHMETIC_CODER_HPP

