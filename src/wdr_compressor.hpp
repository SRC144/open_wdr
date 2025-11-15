#ifndef WDR_COMPRESSOR_HPP
#define WDR_COMPRESSOR_HPP

/**
 * @file wdr_compressor.hpp
 * @brief Wavelet Difference Reduction (WDR) compressor implementation
 * 
 * This implementation is based on the Wavelet Difference Reduction algorithm
 * for embedded image compression. The WDR algorithm combines:
 * - Discrete Wavelet Transform (DWT) for spatial-frequency decomposition
 * - Differential coding and binary reduction for efficient index encoding
 * - Ordered bit-plane transmission for progressive compression
 * - Adaptive arithmetic coding for final lossless compression
 * 
 * The adaptive arithmetic coding stage uses the algorithm from
 * Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 * for data compression." Communications of the ACM, 30(6), 520-540.
 * 
 * @see WDR algorithm paper for the original WDR algorithm
 * @see Witten, Neal, & Cleary (1987) for the arithmetic coding algorithm
 */

#include <vector>
#include <string>
#include <utility>
#include <cstdint>
#include "arithmetic_coder.hpp"
#include "adaptive_model.hpp"
#include "bit_stream.hpp"
#include "wdr_file_format.hpp"
#include <fstream>
#include <cmath>

// Forward declaration for friend tests
namespace testing {
    class Test;
}

/**
 * @brief WDR Compressor
 * 
 * Implements the Wavelet Difference Reduction (WDR) compression algorithm.
 * 
 * The algorithm consists of three main stages:
 * 1. Sorting Pass: Find significant coefficients and encode positions using
 *    differential coding and binary reduction
 * 2. Refinement Pass: Refine existing significant coefficients by transmitting
 *    one bit per coefficient per pass
 * 3. Adaptive Arithmetic Coding: Final compression stage using the algorithm
 *    from Witten, Neal, Cleary (1987)
 * 
 * The compressor maintains three lists:
 * - ICS (Insignificant Coefficient Set): Coefficients not yet significant
 * - SCS (Significant Coefficient Set): Coefficients that are significant
 * - TPS (Temporary Pass Set): Coefficients found significant in current pass
 * 
 * @note This implementation uses separate adaptive models for index bits,
 *       sign bits, and refinement bits to maintain compression efficiency.
 * 
 * @see WDR algorithm paper for detailed algorithm description
 * @see Witten, Neal, & Cleary (1987) for arithmetic coding details
 */
class WDRCompressor {
    // Friend test classes
    friend class WDRCompressorTest;
    friend class WDRCompressorPassesTest;
    
public:
    /**
     * Constructor.
     * 
     * @param num_passes Number of bit-plane passes (default: 16)
     */
    explicit WDRCompressor(int num_passes = 16);
    
    /**
     * Compress coefficients to a .wdr file.
     * 
     * @param coeffs Input coefficients (1D array)
     * @param output_file Output file path
     */
    void compress(const std::vector<double>& coeffs, const std::string& output_file);
    
    /**
     * Decompress coefficients from a .wdr file.
     * 
     * @param input_file Input file path
     * @return Decompressed coefficients (1D array)
     */
    std::vector<double> decompress(const std::string& input_file);

    // Test-friendly public accessors for unit testing
    // These methods are public to allow comprehensive unit testing
    // In production, they could be made private with friend test classes
    
    /**
     * Calculate the initial threshold T (public for testing).
     */
    double calculate_initial_T(const std::vector<double>& coeffs);
    
    /**
     * Apply differential coding (public for testing).
     */
    std::vector<int> differential_encode(const std::vector<int>& indices);
    
    /**
     * Apply inverse differential coding (public for testing).
     */
    std::vector<int> differential_decode(const std::vector<int>& diff_indices);
    
    /**
     * Apply binary reduction (public for testing).
     */
    std::vector<bool> binary_reduce(int value);
    
    /**
     * Apply binary expansion (public for testing).
     */
    int binary_expand(const std::vector<bool>& bits);

private:
    int num_passes_;
    double initial_T_;
    
    // Coefficient sets for encoding/decoding
    std::vector<double> ICS_;  // Insignificant Coefficient Set
    std::vector<std::pair<double, double>> SCS_;  // Significant Coefficient Set: (value, center)
    std::vector<double> TPS_;  // Temporary Pass Set
    
    // For decoding: track which positions in the original array have been decoded
    std::vector<size_t> scs_to_array_pos_;  // Maps SCS index to array position (for decoding)
    std::vector<int> scs_signs_;  // Maps SCS index to sign (1 for positive, 0 for negative)
    std::vector<double> reconstructed_array_;  // Reconstructed coefficient array
    
    /**
     * Sorting pass (encoding).
     * 
     * Finds significant coefficients and encodes their positions.
     * Uses separate adaptive models for index bits and sign bits.
     * 
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param index_model Adaptive model for binary-reduced index bits
     * @param sign_model Adaptive model for sign bits
     */
    void sorting_pass_encode(double T, ArithmeticCoder& coder, AdaptiveModel& index_model, AdaptiveModel& sign_model);
    
    /**
     * Refinement pass (encoding).
     * 
     * Refines existing significant coefficients.
     * 
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param refinement_model Adaptive model for refinement bits
     */
    void refinement_pass_encode(double T, ArithmeticCoder& coder, AdaptiveModel& refinement_model);
    
    /**
     * Sorting pass (decoding).
     * 
     * Decodes positions and reconstructs significant coefficients.
     * Uses separate adaptive models for index bits and sign bits.
     * 
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param index_model Adaptive model for binary-reduced index bits
     * @param sign_model Adaptive model for sign bits
     * @param decoded_positions Output: positions in the original array that were decoded
     * @param decoded_signs Output: signs for decoded coefficients (1 for positive, 0 for negative)
     * @param ics_to_array_map Mapping from ICS index to array position
     */
    void sorting_pass_decode(double T, ArithmeticCoder& coder, AdaptiveModel& index_model, AdaptiveModel& sign_model, std::vector<size_t>& decoded_positions, std::vector<int>& decoded_signs, std::vector<size_t>& ics_to_array_map);
    
    /**
     * Refinement pass (decoding).
     * 
     * Decodes refinement bits and updates coefficients.
     * 
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param refinement_model Adaptive model for refinement bits
     */
    void refinement_pass_decode(double T, ArithmeticCoder& coder, AdaptiveModel& refinement_model);
    
    /**
     * Write file header.
     * 
     * @param stream Output file stream
     * @param initial_T Initial threshold
     * @param num_coeffs Number of coefficients
     * @param data_size Size of compressed data in bytes
     */
    void write_header(std::ofstream& stream, double initial_T, uint64_t num_coeffs, uint64_t data_size);
    
    /**
     * Read file header.
     * 
     * @param stream Input file stream
     * @param initial_T Output: initial threshold
     * @param num_passes Output: number of passes
     * @param num_coeffs Output: number of coefficients
     * @param data_size Output: size of compressed data
     */
    void read_header(std::ifstream& stream, double& initial_T, uint32_t& num_passes, uint64_t& num_coeffs, uint64_t& data_size);
};

#endif // WDR_COMPRESSOR_HPP

