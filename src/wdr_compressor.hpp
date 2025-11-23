#ifndef WDR_COMPRESSOR_HPP
#define WDR_COMPRESSOR_HPP

/**
 * @file wdr_compressor.hpp
 * @brief Wavelet Difference Reduction (WDR) compressor implementation (Memory-Based)
 *
 * This implementation is based on the Wavelet Difference Reduction algorithm
 * for embedded image compression. 
 * 
 * Tian, J., Wells, R.O. (2002). Embedded Image Coding Using Wavelet Difference Reduction.
 * In: Topiwala, P.N. (eds) Wavelet Image and Video Compression. 
 * The International Series in Engineering and Computer Science, vol 450. 
 * Springer, Boston, MA. https://doi.org/10.1007/0-306-47043-8_17
 * 
 * The WDR algorithm combines:
 * - Discrete Wavelet Transform (DWT) for spatial-frequency decomposition
 * - Differential coding and binary reduction for efficient index encoding
 * - Ordered bit-plane transmission for progressive compression
 * - Adaptive arithmetic coding for final lossless compression
 *
 * The adaptive arithmetic coding stage uses the algorithm from
 * Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 * for data compression." Communications of the ACM, 30(6), 520-540.
 *
 * Refactored for Tiled BigTIFF architecture:
 * - Operates strictly on in-memory buffers (std::vector)
 * - Removes internal file I/O and legacy headers
 * - Accepts Global Threshold (T) for tile consistency
 *
 * @see @cite Tian2002 for the original WDR algorithm
 * @see @cite Witten1987 for the arithmetic coding algorithm
 */

#include "adaptive_model.hpp"
#include "arithmetic_coder.hpp"
#include "bit_stream.hpp"
#include <cmath>
#include <cstdint>
#include <list>
#include <vector>
#include <utility>

/**
 * @brief WDR Compressor (Memory/Tile Mode)
 * * Implements the Wavelet Difference Reduction (WDR) compression algorithm.
 * The algorithm consists of three main stages:
 * 1. Sorting Pass: Find significant coefficients and generate symbols for
 * their positions and signs.
 * 2. Refinement Pass: Refine existing significant coefficients by generating
 * one bit-symbol per coefficient per pass.
 * 3. Adaptive Arithmetic Coding: Final compression stage that encodes the
 * single, interleaved symbol stream.
 * * The compressor maintains three lists:
 * - ICS (Insignificant Coefficient Set): Coefficients not yet significant
 * - SCS (Significant Coefficient Set): Coefficients that are significant
 * - TPS (Temporary Pass Set): Coefficients found significant in current pass
 * * Key Architectural Changes for Tiling:
 * - Compress: Accepts raw coefficients + Global Threshold (T). Returns raw bytes.
 * - Decompress: Accepts raw bytes + Global Threshold (T). Returns coefficients.
 * - State: The object instance holds configuration (passes), but the 
 * methods are stateless regarding the bitstream (zero-copy).
 */
class WDRCompressor {
public:
    /**
     * @brief Construct a new WDRCompressor object.
     * * @param num_passes The number of bit-plane passes to perform (default: 16).
     * Must match during compression and decompression.
     */
    explicit WDRCompressor(int num_passes = 16);

    /**
     * @brief Compress a block of coefficients into a byte vector.
     * * Performs the WDR encoding process on a single tile.
     * * @param coeffs The flattened wavelet coefficients (e.g., a 512x512 tile).
     * @param initial_T The Global Threshold calculated from the *entire* image 
     * (Pass 1). This ensures bit-plane alignment across tiles, preventing artifacts
     * where adjacent tiles have different quantization levels.
     * @return std::vector<uint8_t> The compressed binary stream.
     */
    std::vector<uint8_t> compress(const std::vector<double>& coeffs, 
                                  double initial_T);

    /**
     * @brief Decompress a byte vector back into coefficients.
     * * Reconstructs the wavelet coefficients from the compressed WDR stream.
     * * @param compressed_data The raw WDR byte stream.
     * @param initial_T The Global Threshold used during compression.
     * @param num_coeffs The expected number of coefficients (e.g., 512*512).
     * @return std::vector<double> The reconstructed coefficients.
     */
    std::vector<double> decompress(const std::vector<uint8_t>& compressed_data, 
                                   double initial_T, 
                                   uint64_t num_coeffs);

    // --- Testing & Debugging Helpers ---
    // These methods are public to allow comprehensive unit testing.
    // In production, they could be made private with friend test classes.

    /**
     * @brief Apply differential coding (public for testing).
     */
    std::vector<int> differential_encode(const std::vector<int>& indices);

    /**
     * @brief Apply inverse differential coding (public for testing).
     */
    std::vector<int> differential_decode(const std::vector<int>& diff_indices);

private:
    int num_passes_;
    double initial_T_;

    /**
     * @brief Defines the context of a symbol, mapping to a specific adaptive
     * model (exploit redundancy in position stream and refinement stream).
     *
     * This is an in-memory-only "tag" for the encoder. The decoder knows the
     * context based on its state-machine logic.
     */
    enum class SymbolContext : uint8_t {
        /**
         * @brief For the sorting pass.
         * The alphabet for this context includes:
         * 0 = index bit 0
         * 1 = index bit 1
         * 2 = positive sign (End-of-Message symbol)
         * 3 = negative sign (End-of-Message symbol)
         * 4 = zero positive (End-of-Message symbol)
         * 5 = zero negative (End-of-Message symbol)
         ** See compress() on the cpp for more details.
         */
        SORTING_PASS,

        /**
         * @brief For the refinement pass.
         * The alphabet for this context includes:
         * 0 = refinement bit 0
         * 1 = refinement bit 1
         */
        REFINEMENT_PASS
    };

    /**
     * @brief Represents a single symbol generated by the WDR passes.
     *
     * This struct is used to create an in-memory list (the "script") of all
     * symbols before the final arithmetic encoding step.
     */
    struct WDRSymbol {
        SymbolContext context;
        uint8_t symbol; // The symbol to be encoded

        WDRSymbol(SymbolContext c, uint8_t s) : context(c), symbol(s) {}
    };

    // --- Internal State (Reset per tile) ---
    
    const std::vector<double>* original_coeffs_ptr_ = nullptr;
    std::list<size_t> ICS_indices_list_;         ///< Insignificant Coefficient Set
    std::vector<std::pair<double, double>> SCS_; ///< Significant Coefficient Set: (value, center)
    std::vector<double> TPS_;                    ///< Temporary Pass Set

    // Decoding specific state
    std::vector<size_t> scs_to_array_pos_;    ///< Maps SCS index to array position
    std::vector<int> scs_signs_;              ///< Maps SCS index to sign (1 for pos, 0 for neg)
    std::vector<double> reconstructed_array_; ///< Reconstructed coefficient array

    // --- Core Algorithm Steps ---

    /**
     * Sorting pass (encoding).
     *
     * Finds significant coefficients and generates symbols for their positions
     * and signs. Appends symbols to the master symbol_stream.
     *
     * @param T Current threshold
     * @param symbol_stream The master list of symbols to append to
     */
    void sorting_pass_encode(double T, std::vector<WDRSymbol>& symbol_stream);

    /**
     * Refinement pass (encoding).
     *
     * Generates refinement bit symbols for existing significant coefficients.
     * Appends symbols to the master symbol_stream.
     *
     * @param T Current threshold
     * @param symbol_stream The master list of symbols to append to
     */
    void refinement_pass_encode(double T, std::vector<WDRSymbol>& symbol_stream);
    
    /**
     * Sorting pass (decoding).
     *
     * Decodes positions and signs from the arithmetic coder using a
     * *single* sorting model, reconstructing significant coefficients.
     *
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param sorting_model Adaptive model for sorting pass symbols
     * @param decoded_positions Output: positions in the original array
     * @param decoded_signs Output: signs for decoded coefficients
     * @param ics_to_array_map Mapping from ICS index to array position
     */
    void sorting_pass_decode(double T, ArithmeticCoder& coder,
                             AdaptiveModel& sorting_model,
                             std::vector<size_t>& decoded_positions,
                             std::vector<int>& decoded_signs,
                             std::list<size_t>& ics_to_array_map);
                             
    /**
     * Refinement pass (decoding).
     *
     * Decodes refinement bits and updates coefficients in the SCS.
     *
     * @param T Current threshold
     * @param coder Arithmetic coder
     * @param refinement_model Adaptive model for refinement bits
     */
    void refinement_pass_decode(double T, ArithmeticCoder& coder,
                                AdaptiveModel& refinement_model);

    // --- Arithmetic Coding Helpers ---

    /**
     * @brief Performs the final arithmetic encoding step (Encoder's state machine).
     *
     * Iterates over the symbol list for the current pass and encodes each symbol
     * using the GLOBAL adaptive models for the sorting and refinement passes.
     *
     * @param symbol_stream_for_pass The ordered list of symbols for the current pass.
     * @param coder The arithmetic coder.
     * @param sorting_model The GLOBAL adaptive model for the sorting pass.
     * @param refinement_model The GLOBAL adaptive model for the refinement pass.
     */
    void arithmetic_encode_stream(const std::vector<WDRSymbol>& symbol_stream_for_pass,
                                  ArithmeticCoder& coder, 
                                  AdaptiveModel& sorting_model,
                                  AdaptiveModel& refinement_model);

    /**
     * @brief Helper to write a binary-reduced value to the symbol stream.
     *
     * Encapsulates the logic of outputting index bits (0, 1) and
     * the sign/EOM symbol (2, 3, 4, 5).
     */
    void write_binary_reduced(std::vector<WDRSymbol>& symbol_stream, int value, bool sign);

    /**
     * @brief Helper to read one binary-expanded value during decode.
     *
     * Reads symbols (0, 1) from the coder until an EOM symbol (2, 3, 4, 5)
     * is found, reconstructing the index.
     */
    int read_binary_expanded(ArithmeticCoder& coder, AdaptiveModel& sorting_model, bool& sign_out);
};

#endif // WDR_COMPRESSOR_HPP