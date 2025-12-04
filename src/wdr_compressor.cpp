#include "wdr_compressor.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

// ============================================================================
//  Constructor
// ============================================================================

WDRCompressor::WDRCompressor(int num_passes) : num_passes_(num_passes) {
  if (num_passes < 1) {
    throw std::invalid_argument("num_passes must be >= 1");
  }
}

// ============================================================================
//  COMPRESS (Memory to Memory)
// ============================================================================

std::vector<uint8_t> WDRCompressor::compress(const std::vector<double> &coeffs,
                                             double initial_T) {
  if (coeffs.empty()) {
    return {}; // Return empty vector for empty input
  }

  // 1. Configuration
  // We set the global threshold passed from the "Pass 1" analysis.
  initial_T_ = initial_T;
  original_coeffs_ptr_ = &coeffs;

  // 2. Reset Internal State
  ICS_indices_list_.clear();
  SCS_.clear();
  TPS_.clear();

  // Initialize ICS with all coefficient indices
  for (size_t i = 0; i < coeffs.size(); i++) {
    ICS_indices_list_.push_back(i);
  }

  // 3. Prepare Output Buffer
  std::vector<uint8_t> output_buffer;
  // Heuristic: Reserve 25% of the raw size (doubles * 8 bytes).
  // This prevents frequent reallocations during the push_back operations
  // which is critical for the zero-copy performance.
  output_buffer.reserve(coeffs.size() * sizeof(double) / 4);

  // 4. Setup Pipeline
  // BitOutputStream now writes directly to our output_buffer vector.
  BitOutputStream bit_stream(output_buffer);

  /// Create GLOBAL models and coder. They persist for the entire compression
  /// process.

  // The sorting_model uses a 6-symbol alphabet to resolve the
  // ambiguity between a differential index of 0 and 1.
  //
  // The 0/1 Ambiguity:
  // Binary Reduction = "remove the MSB".
  // - Value 1 (binary '1') -> remove MSB '1' -> empty bit list '[]'
  // - Value 0 (binary '0') -> (no MSB)     -> empty bit list '[]'
  //
  // To solve this, we define 4 EOM (End-Of-Message) symbols:
  // 0 = bit '0'
  // 1 = bit '1'
  // 2 = POS_SIGN (EOM, implies value 1)
  // 3 = NEG_SIGN (EOM, implies value 1)
  // 4 = ZERO_POS (EOM, implies value 0)
  // 5 = ZERO_NEG (EOM, implies value 0)

  AdaptiveModel sorting_model(6);    // Symbols 0-5
  AdaptiveModel refinement_model(2); // Symbols 0-1
  sorting_model.start_model();
  refinement_model.start_model();

  ArithmeticCoder coder;
  coder.start_encoding(bit_stream, sorting_model);

  // 5. Main Compression Loop
  // Generates tagged state symbols (SORTING_PASS and REFINEMENT_PASS)
  double T = initial_T_;
  for (int pass = 0; pass < num_passes_; pass++) {
    // Create a TEMPORARY vector for this pass only.
    // This keeps symbol memory usage at O(1) relative to the whole file.
    std::vector<WDRSymbol> pass_symbol_stream;

    // Sorting Pass: Generates symbols into local stream
    sorting_pass_encode(T, pass_symbol_stream);

    // Refinement Pass: Generates symbols into local stream
    refinement_pass_encode(T, pass_symbol_stream);

    // "FLUSH" this pass's symbols to the GLOBAL persistent coder
    arithmetic_encode_stream(pass_symbol_stream, coder, sorting_model,
                             refinement_model);

    // Update Significant Coeff Set (SCS) based on findings (TPS)
    // Move coefficients from TPS to SCS with initial reconstruction value
    for (double val : TPS_) {
      double center = T + T / 2.0; // 1.5*T
      SCS_.push_back(std::make_pair(val, center));
    }
    TPS_.clear();

    // Decrease Threshold (next bit-plane)
    T = T / 2.0;
  }

  // 6. Finalize
  // Tell the GLOBAL persistent coder we are finished and flush bits
  coder.done_encoding();
  bit_stream.flush();

  // RVO (Return Value Optimization) ensures no copy happens here
  return output_buffer;
}

// ============================================================================
//  DECOMPRESS (Memory to Memory)
// ============================================================================

std::vector<double>
WDRCompressor::decompress(const std::vector<uint8_t> &compressed_data,
                          double initial_T, uint64_t num_coeffs) {
  // Handle edge cases
  if (num_coeffs == 0)
    return {};
  if (compressed_data.empty())
    return std::vector<double>(num_coeffs, 0.0);

  // 1. Configuration
  initial_T_ = initial_T;

  // Reset State
  ICS_indices_list_.clear();
  SCS_.clear();
  TPS_.clear();
  scs_to_array_pos_.clear();
  scs_signs_.clear();

  // Initialize Output Array
  reconstructed_array_.assign(num_coeffs, 0.0);

  std::list<size_t> ics_to_array_map;
  for (size_t i = 0; i < num_coeffs; i++) {
    ics_to_array_map.push_back(i);
  }

  // 2. Setup Pipeline
  BitInputStream bit_stream(compressed_data);

  // Initialize models matching the compression configuration
  // 0,1,POS(2),NEG(3),ZERO_POS(4),ZERO_NEG(5)
  AdaptiveModel sorting_model(6);
  AdaptiveModel refinement_model(2);
  sorting_model.start_model();
  refinement_model.start_model();

  ArithmeticCoder coder;
  coder.start_decoding(bit_stream, sorting_model);

  // 3. Main Decompression Loop
  double T = initial_T_;
  for (int pass = 0; pass < num_passes_; pass++) {
    std::vector<size_t> pass_decoded_positions;
    std::vector<int> pass_decoded_signs;

    // Sorting Pass Decode
    sorting_pass_decode(T, coder, sorting_model, pass_decoded_positions,
                        pass_decoded_signs, ics_to_array_map);

    // Refinement Pass Decode
    refinement_pass_decode(T, coder, refinement_model);

    // Move new significant coefficients from TPS logic to SCS
    for (size_t i = 0; i < pass_decoded_positions.size(); i++) {
      size_t array_pos = pass_decoded_positions[i];
      int sign = pass_decoded_signs[i];
      double center = TPS_[i];

      double signed_value = (sign == 1) ? center : -center;

      SCS_.push_back(std::make_pair(signed_value, center));
      scs_to_array_pos_.push_back(array_pos);
      scs_signs_.push_back(sign);
    }
    TPS_.clear();

    // Update the actual coefficient array with current precision
    for (size_t i = 0; i < SCS_.size(); i++) {
      reconstructed_array_[scs_to_array_pos_[i]] = SCS_[i].first;
    }

    T = T / 2.0;
  }

  return reconstructed_array_;
}

// ============================================================================
//  Internal Helpers
// ============================================================================

void WDRCompressor::arithmetic_encode_stream(
    const std::vector<WDRSymbol> &symbol_stream_for_pass,
    ArithmeticCoder &coder, AdaptiveModel &sorting_model,
    AdaptiveModel &refinement_model) {
  // Iterate over the symbols for the current pass and encode them using the
  // GLOBAL models
  for (const WDRSymbol &sym : symbol_stream_for_pass) {
    if (sym.context == SymbolContext::SORTING_PASS) {
      coder.encode_symbol(sym.symbol, sorting_model);
      sorting_model.update_model(sym.symbol); // API usage
    } else {
      // REFINEMENT_PASS
      coder.encode_symbol(sym.symbol, refinement_model);
      refinement_model.update_model(sym.symbol); // API usage
    }
  }
}

void WDRCompressor::sorting_pass_encode(double T,
                                        std::vector<WDRSymbol> &symbol_stream) {
  std::vector<int> indices;
  std::vector<int> signs;

  // Step 1: Find significant coefficients

  // We store the iterators to erase *after* we find the significant
  // coefficients. for this pass.
  std::vector<std::list<size_t>::iterator> iterators_to_erase;

  int current_list_index = 0;
  for (auto it = ICS_indices_list_.begin(); it != ICS_indices_list_.end();
       ++it) {
    // Get the coefficient value from the original coefficients
    double val = (*original_coeffs_ptr_)[*it];

    // If the coefficient is significant, we save its list index and sign.
    if (std::abs(val) >= T) {
      indices.push_back(current_list_index);
      signs.push_back(val >= 0 ? 1 : 0);
      TPS_.push_back(val);

      // Save the iterator to erase it
      iterators_to_erase.push_back(it);
    }
    current_list_index++;
  }

  // Remove the significant coefficients from the ICS_indices_list_.
  for (auto it : iterators_to_erase) {
    ICS_indices_list_.erase(it); // This is O(1) (double linked list)
  }

  // Step 2: Calculate count of significant coefficients
  int count = static_cast<int>(indices.size());
  int max_count = static_cast<int>(ICS_indices_list_.size() + count);

  int bits_needed = 1;
  if (max_count > 0) {
    bits_needed = static_cast<int>(std::ceil(std::log2(max_count + 1)));
    if (bits_needed == 0)
      bits_needed = 1;
  }

  // Generate symbols for the count (MSB first)
  // Use SORTING_PASS context, as this is the only model the decoder will have
  // in this state.
  for (int bit_pos = bits_needed - 1; bit_pos >= 0; bit_pos--) {
    int bit = (count >> bit_pos) & 1;
    symbol_stream.push_back(WDRSymbol(SymbolContext::SORTING_PASS, bit));
  }

  // Differential encoding + add signs + write to symbol stream binary-reduced
  if (count > 0) {
    std::vector<int> diff_indices = differential_encode(indices);
    for (size_t i = 0; i < diff_indices.size(); i++) {
      write_binary_reduced(symbol_stream, diff_indices[i], (signs[i] == 1));
    }
  }
}

void WDRCompressor::refinement_pass_encode(
    double T, std::vector<WDRSymbol> &symbol_stream) {

  // Refine the coefficients
  for (auto &pair : SCS_) {
    double val = pair.first;
    double center = pair.second;
    double abs_val = std::abs(val);

    // Calculate interval using unsigned magnitude
    double low = center - T;
    double high = center + T;

    // Calculate bit based on interval
    int bit;
    if (abs_val >= center) {
      bit = 1;
      center = (center + high) / 2.0;
    } else {
      bit = 0;
      center = (low + center) / 2.0;
    }

    pair.second = center; // Update unsigned center
    symbol_stream.push_back(WDRSymbol(SymbolContext::REFINEMENT_PASS, bit));
  }
}

void WDRCompressor::sorting_pass_decode(double T, ArithmeticCoder &coder,
                                        AdaptiveModel &sorting_model,
                                        std::vector<size_t> &decoded_positions,
                                        std::vector<int> &decoded_signs,
                                        std::list<size_t> &ics_to_array_map) {
  // Clear decoded positions and signs
  decoded_positions.clear();
  decoded_signs.clear();

  // Calculate max_count and bits_needed
  int max_count = static_cast<int>(ics_to_array_map.size());
  int bits_needed = 1;
  if (max_count > 0) {
    bits_needed = static_cast<int>(std::ceil(std::log2(max_count + 1)));
    if (bits_needed == 0)
      bits_needed = 1;
  }

  // Step 1: Decode count using sorting_model
  int count = 0;
  for (int bit_pos = bits_needed - 1; bit_pos >= 0; bit_pos--) {
    int bit = coder.decode_symbol(sorting_model);
    sorting_model.update_model(bit); // API usage
    count = (count << 1) | bit;
  }

  if (count < 0 || count > max_count) {
    throw std::runtime_error("Invalid count decoded: " + std::to_string(count));
  }
  if (count == 0)
    return; // No coefficients to decode

  // Step 2: Decode the list of significant indices (and their signs) in the
  // current pass.
  std::vector<int> diff_indices;
  std::vector<int> signs;
  for (int i = 0; i < count; i++) {
    bool sign_out;
    int diff_idx = read_binary_expanded(coder, sorting_model, sign_out);
    diff_indices.push_back(diff_idx);
    signs.push_back(sign_out ? 1 : 0);
  }

  // Step 4: Code positions & signs and update ICS. We update ics_to_array_map
  // which is defined at the beginning of the decompress function. We map the
  // indices in the current pass to our ICS .

  // Differential decode the indices to get the indices in the current pass.
  std::vector<int> ics_indices = differential_decode(diff_indices);
  std::vector<std::list<size_t>::iterator> iterators_to_erase;

  // Get positions to erase in the ICS (our global ICS).
  auto list_it = ics_to_array_map.begin();
  int current_list_index = 0;
  for (int target_index : ics_indices) {
    // Advance from our last position to the next target
    int steps_to_take = target_index - current_list_index;
    std::advance(list_it, steps_to_take);

    // Save the iterator to erase it
    iterators_to_erase.push_back(list_it);

    // Update our position
    current_list_index = target_index;
  }

  // Save the data and erase the iterators from the ICS (our global ICS).
  for (size_t i = 0; i < iterators_to_erase.size(); ++i) {
    auto it = iterators_to_erase[i];
    int sign = signs[i];

    // Save the data
    decoded_positions.push_back(*it); // *it is the *original array index*
    decoded_signs.push_back(sign);
    TPS_.push_back(T + T / 2.0);

    // Erase from the list (O(1))
    ics_to_array_map.erase(it);
  }
}

void WDRCompressor::refinement_pass_decode(double T, ArithmeticCoder &coder,
                                           AdaptiveModel &refinement_model) {
  // Decode the coefficients in the SCS
  for (size_t idx = 0; idx < SCS_.size(); idx++) {
    auto &pair = SCS_[idx];
    double center = pair.second; // Unsigned center

    // Calculate interval using unsigned magnitude
    double low = center - T;
    double high = center + T;

    // Decode the bit
    int bit = coder.decode_symbol(refinement_model);
    refinement_model.update_model(bit); // API usage

    // Update the center based on the bit
    if (bit == 1) {
      center = (center + high) / 2.0;
    } else {
      center = (low + center) / 2.0;
    }

    // Update the signed, refined value
    double sign = (scs_signs_[idx] == 1) ? 1.0 : -1.0;
    pair.second = center;       // Update unsigned center
    pair.first = sign * center; // Update signed, refined value
  }
}

std::vector<int>
WDRCompressor::differential_encode(const std::vector<int> &indices) {
  if (indices.empty()) {
    return {};
  }

  std::vector<int> diff_indices;
  diff_indices.push_back(indices[0]);

  for (size_t i = 1; i < indices.size(); i++) {
    diff_indices.push_back(indices[i] - indices[i - 1]);
  }

  return diff_indices;
}

std::vector<int>
WDRCompressor::differential_decode(const std::vector<int> &diff_indices) {
  if (diff_indices.empty()) {
    return {};
  }

  std::vector<int> indices;
  indices.push_back(diff_indices[0]);

  for (size_t i = 1; i < diff_indices.size(); i++) {
    indices.push_back(indices[i - 1] + diff_indices[i]);
  }

  return indices;
}

void WDRCompressor::write_binary_reduced(std::vector<WDRSymbol> &symbol_stream,
                                         int value, bool sign) {
  // Differential-coded indices are always >= 0
  if (value == 0) {
    // Handle the value '0' case explicitly.
    // This resolves the 0/1 ambiguity (see comment in compress() in this file
    // for details). We send a unique, signed symbol for zero.
    uint8_t zero_sym = sign ? 4 : 5; // 4 = ZERO_POS, 5 = ZERO_NEG
    symbol_stream.push_back(WDRSymbol(SymbolContext::SORTING_PASS, zero_sym));
    return;
  }

  int msb_pos = 0;
  int temp = value;
  while (temp > 1) {
    temp >>= 1;
    msb_pos++;
  }

  // Write bits from LSB up to (but not including) MSB
  for (int i = 0; i < msb_pos; i++) {
    int bit = (value >> i) & 1;
    symbol_stream.push_back(WDRSymbol(SymbolContext::SORTING_PASS, bit));
  }

  uint8_t sign_sym = sign ? 2 : 3; // 2 = POS, 3 = NEG
  symbol_stream.push_back(WDRSymbol(SymbolContext::SORTING_PASS, sign_sym));
}

int WDRCompressor::read_binary_expanded(ArithmeticCoder &coder,
                                        AdaptiveModel &sorting_model,
                                        bool &sign_out) {
  std::vector<bool> bits; // Stores LSB-first bits
  int value = 0;

  while (true) {
    uint8_t sym = coder.decode_symbol(sorting_model);
    sorting_model.update_model(sym); // API usage

    if (sym == 0) {
      bits.push_back(false);
    } else if (sym == 1) {
      bits.push_back(true);
    } else if (sym == 2) { // POS_SIGN
      sign_out = true;
      break;
    } else if (sym == 3) { // NEG_SIGN
      sign_out = false;
      break;
    } else if (sym == 4) { // ZERO_POS
      sign_out = true;
      return 0;
    } else if (sym == 5) { // ZERO_NEG
      sign_out = false;
      return 0;
    } else {
      throw std::runtime_error("Invalid symbol decoded in sorting pass");
    }
  }

  // Reconstruct value: MSB is 1 at position bits.size()
  value = 1 << bits.size();
  for (size_t i = 0; i < bits.size(); i++) {
    if (bits[i]) {
      value |= (1 << i);
    }
  }

  return value;
}