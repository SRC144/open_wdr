#ifndef ADAPTIVE_MODEL_HPP
#define ADAPTIVE_MODEL_HPP

/**
 * @file adaptive_model.hpp
 * @brief Adaptive probability model for arithmetic coding
 * 
 * This implementation is based on the adaptive source model from
 * Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 * for data compression." Communications of the ACM, 30(6), 520-540.
 * 
 * The model updates symbol frequencies as symbols are processed and maintains
 * frequency-sorted order for efficient decoding. It includes frequency rescaling
 * and symbol reordering to adapt to the data's local statistics.
 * 
 * @see Witten, Neal, & Cleary (1987) for the original adaptive model algorithm
 */

#include <vector>
#include <cstdint>

/**
 * @brief Adaptive probability model for arithmetic coding
 * 
 * This class implements an adaptive model that updates symbol frequencies
 * as symbols are processed. It maintains frequency-sorted order for efficient
 * decoding. The implementation is based on the adaptive model from
 * Witten, Neal, Cleary (ACM 1987).
 * 
 * Key features:
 * - Frequency rescaling when total frequency exceeds MAX_FREQUENCY
 * - Symbol reordering to keep frequent symbols at lower indices
 * - Backward cumulative frequency table for arithmetic coding
 * 
 * For WDR, we use a binary alphabet (2 symbols: 0 and 1).
 * 
 * @see Witten, I.H., Neal, R.M., & Cleary, J.G. (1987). "Arithmetic coding
 *      for data compression." Communications of the ACM, 30(6), 520-540.
 */
class AdaptiveModel {
public:
    /**
     * Constructor.
     * 
     * @param num_symbols Number of symbols in the alphabet (default: 2 for binary)
     */
    explicit AdaptiveModel(int num_symbols = 2);
    
    /**
     * Initialize the model with flat probabilities.
     * All symbols start with frequency 1.
     */
    void start_model();
    
    /**
     * Update the model after encoding/decoding a symbol.
     * 
     * This method:
     * 1. Rescales frequencies if needed (when total frequency exceeds MAX_FREQUENCY)
     * 2. Re-sorts symbols by frequency (moves frequent symbols to lower indices)
     * 3. Increments the symbol's frequency
     * 4. Updates cumulative frequencies
     * 
     * @param symbol Symbol index to update (0 to num_symbols-1)
     */
    void update_model(int symbol);
    
    /**
     * Get the cumulative frequency table.
     * 
     * The cumulative frequency table is backward:
     * - cum_freq[0] is the total frequency
     * - cum_freq[i] is the cumulative frequency for symbols with index >= i
     * 
     * @return Reference to the cumulative frequency vector
     */
    const std::vector<int>& get_cumulative_freq() const;
    
    /**
     * Get the number of symbols in the alphabet.
     * 
     * @return Number of symbols
     */
    int get_num_symbols() const;
    
    /**
     * Get the frequency of a symbol.
     * 
     * @param symbol Symbol index
     * @return Frequency of the symbol
     */
    int get_frequency(int symbol) const;
    
    /**
     * Get the internal index for a symbol (for arithmetic coding).
     * 
     * @param symbol Symbol index (0 to num_symbols-1)
     * @return Internal index (1 to num_symbols)
     */
    int get_internal_index(int symbol) const;
    
    /**
     * Get the symbol from an internal index (for arithmetic coding).
     * 
     * @param internal_index Internal index (1 to num_symbols)
     * @return Symbol index (0 to num_symbols-1)
     */
    int get_symbol_from_internal_index(int internal_index) const;
    
    /**
     * Reset the model to initial state.
     */
    void reset();

private:
    static constexpr int MAX_FREQUENCY = 16383;  // 2^14 - 1
    
    int num_symbols_;
    std::vector<int> freq_;           // Symbol frequencies
    std::vector<int> cum_freq_;       // Cumulative frequencies (backward)
    std::vector<int> symbol_to_index_; // Map from symbol to frequency-sorted index
    std::vector<int> index_to_symbol_; // Map from frequency-sorted index to symbol
    
    /**
     * Rescale frequencies by halving them when total frequency exceeds MAX_FREQUENCY.
     * This prevents overflow and weights recent symbols more heavily.
     */
    void rescale_frequencies();
    
    /**
     * Re-sort symbols by frequency, moving frequent symbols to lower indices.
     * 
     * @param symbol_index Current index of the symbol to update
     * @return New index of the symbol after re-sorting
     */
    int resort_symbol(int symbol_index);
};

#endif // ADAPTIVE_MODEL_HPP

