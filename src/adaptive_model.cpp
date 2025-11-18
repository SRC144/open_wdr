#include "adaptive_model.hpp"
#include <algorithm>
#include <stdexcept>
#include <cassert>

AdaptiveModel::AdaptiveModel(int num_symbols)
    : num_symbols_(num_symbols) {
    // In our case we pass only the 6 symbol or 2 symbol alphabets.
    if (num_symbols < 2 || num_symbols > 256) {
        throw std::invalid_argument("num_symbols must be between 2 and 256");
    }
    
    freq_.resize(num_symbols_ + 1);
    cum_freq_.resize(num_symbols_ + 1);
    symbol_to_index_.resize(num_symbols_);
    index_to_symbol_.resize(num_symbols_ + 1);  // 1-indexed, so need +1
    
    start_model();
}

void AdaptiveModel::start_model() {
    // Initialize symbol mapping (initially identity mapping)
    /*  We use num_symbols instead of the No_of_chars like the paper since our
        alphabet is limited to maximum 6 symbols, no need for 256.
    */
    for (int i = 0; i < num_symbols_; i++) { 
        symbol_to_index_[i] = i + 1;  // Symbols are 1-indexed internally
        index_to_symbol_[i + 1] = i;
    }

    for (int i = 0; i <= num_symbols_; i++) {
        freq_[i] = 1;                    // Initialize all symbol frequencies to 1
        cum_freq_[i] = num_symbols_ - i; // Initialize cumulative frequencies (backward)
    }
    freq_[0] = 0;  // Freq[0] must not be the same as freq[1]
}

void AdaptiveModel::update_model(int symbol) {
    if (symbol < 0 || symbol >= num_symbols_) {
        throw std::invalid_argument("Symbol index out of range");
    }
    
    // Check if we need to rescale
    if (cum_freq_[0] >= MAX_FREQUENCY) {
        rescale_frequencies();
    }
    
    // Get the current index of the symbol (1-indexed)
    int symbol_index = symbol_to_index_[symbol];
    
    // Find the new correct position for this symbol after incrementing
    // We need to move it up in the frequency-sorted order if it becomes more frequent
    int new_index = resort_symbol(symbol_index);
    
    // Increment the frequency
    freq_[new_index]++;
    
    // Update cumulative frequencies for all indices <= new_index
    for (int i = new_index; i > 0; i--) {
        cum_freq_[i - 1]++;
    }
}

const std::vector<int>& AdaptiveModel::get_cumulative_freq() const {
    return cum_freq_;
}

int AdaptiveModel::get_num_symbols() const {
    return num_symbols_;
}

int AdaptiveModel::get_frequency(int symbol) const {
    if (symbol < 0 || symbol >= num_symbols_) {
        throw std::invalid_argument("Symbol index out of range");
    }
    int index = symbol_to_index_[symbol];
    return freq_[index];
}

int AdaptiveModel::get_internal_index(int symbol) const {
    if (symbol < 0 || symbol >= num_symbols_) {
        throw std::invalid_argument("Symbol index out of range");
    }
    return symbol_to_index_[symbol];
}

int AdaptiveModel::get_symbol_from_internal_index(int internal_index) const {
    if (internal_index < 1 || internal_index > num_symbols_) {
        throw std::invalid_argument("Internal index out of range");
    }
    return index_to_symbol_[internal_index];
}

void AdaptiveModel::reset() {
    start_model();
}

void AdaptiveModel::rescale_frequencies() {
    // Halve all frequencies (keeping them non-zero)
    /*  As we encode data we count the symbols. The count can get too big and
        cause integer overflow. The paper sets a hard limit MAX_FREQUENCY and if the cumulative
        frequencies exceed that we halve all frequencies, preserving the proportions.*/
    int cum = 0;
    for (int i = num_symbols_; i >= 0; i--) {
        freq_[i] = (freq_[i] + 1) / 2;  // Round up to avoid zero
        cum_freq_[i] = cum;
        cum += freq_[i];
    }
    // Ensure freq[0] is 0 (it must not be the same as freq[1])
    freq_[0] = 0;
}

int AdaptiveModel::resort_symbol(int symbol_index) {
    // Find the new index by moving up while frequencies are equal
    // This maintains frequency-sorted order (most frequent symbols have lower indices)
    int i = symbol_index;
    
    // Move up while the frequency of the current symbol is equal to the one above
    // We want to move the symbol up in the sorted order
    while (i > 1 && freq_[i] == freq_[i - 1]) {
        i--;
    }
    
    // If the symbol needs to move, swap it with the symbol at the new position
    if (i < symbol_index) {
        // Get the symbols at these indices
        int symbol_i = index_to_symbol_[i];
        int symbol_old = index_to_symbol_[symbol_index];
        
        // Swap the symbols in the index_to_symbol mapping
        index_to_symbol_[i] = symbol_old;
        index_to_symbol_[symbol_index] = symbol_i;
        
        // Update the symbol_to_index mapping
        symbol_to_index_[symbol_i] = symbol_index;
        symbol_to_index_[symbol_old] = i;
    }
    
    return i;
}

