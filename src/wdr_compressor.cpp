#include "wdr_compressor.hpp"
#include "bit_stream.hpp"
#include "wdr_file_format.hpp"
#include <cmath>
#include <cstring>
#include <iostream>
#include <iterator>
#include <list>
#include <sstream>
#include <stdexcept>

WDRCompressor::WDRCompressor(int num_passes) : num_passes_(num_passes) {
  if (num_passes < 1) {
    throw std::invalid_argument("num_passes must be >= 1");
  }
}

double WDRCompressor::calculate_initial_T(const std::vector<double> &coeffs) {
  if (coeffs.empty()) {
    throw std::invalid_argument("Empty coefficient array");
  }

  // Find maximum absolute value
  double max_abs = 0.0;
  for (double coeff : coeffs) {
    double abs_coeff = std::abs(coeff);
    if (abs_coeff > max_abs) {
      max_abs = abs_coeff;
    }
  }

  if (max_abs == 0.0) {
    return 1.0; // Default threshold for all-zero coefficients
  }

  // Find the largest power of 2 such that max_abs < 2*T and max_abs >= T
  // T = 2^k where k is the largest integer such that 2^k <= max_abs < 2^(k+1)
  double T = std::pow(2.0, std::floor(std::log2(max_abs)));

  // Ensure max_abs >= T (if max_abs is exactly a power of 2, T might be too
  // large)
  if (max_abs < T) {
    T = T / 2.0;
  }

  // Ensure max_abs < 2*T
  if (max_abs >= 2.0 * T) {
    T = T * 2.0;
  }

  return T;
}

void WDRCompressor::compress(const std::vector<double> &coeffs,
                             const std::string &output_file) {
  if (coeffs.empty()) {
    throw std::invalid_argument("Empty coefficient array");
  }

  initial_T_ = calculate_initial_T(coeffs);
  const auto payload = compress_tile(coeffs, initial_T_);

  std::ofstream file(output_file, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + output_file);
  }

  const uint64_t num_coeffs = coeffs.size();
  const uint64_t data_size = payload.size();
  write_header(file, initial_T_, num_coeffs, data_size);

  if (!payload.empty()) {
    file.write(reinterpret_cast<const char *>(payload.data()),
               static_cast<std::streamsize>(payload.size()));
  }

  if (!file.good()) {
    throw std::runtime_error("Failed to write compressed payload to file");
  }
  file.close();
}

/**
 * @brief Flushes a pass's generated symbols to the persistent arithmetic coder.
 *
 * This function is the "executor" part of the "flushing" architecture.
 * It takes the small, temporary symbol vector for a *single pass*
 * and encodes its symbols one-by-one using the persistent, global
 * models. This keeps the encoder's memory usage scalable (O(1)
 * relative to the total number of symbols).
 *
 * @param symbol_stream_for_pass The script of symbols for this pass only.
 * @param coder The persistent, global ArithmeticCoder.
 * @param sorting_model The persistent, global model for the sorting pass.
 * @param refinement_model The persistent, global model for the refinement pass.
 */
void WDRCompressor::arithmetic_encode_stream(
    const std::vector<WDRSymbol> &symbol_stream_for_pass,
    ArithmeticCoder &coder, AdaptiveModel &sorting_model,
    AdaptiveModel &refinement_model) const {
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

std::vector<double> WDRCompressor::decompress(const std::string &input_file) {
  std::ifstream file(input_file, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open input file: " + input_file);
  }

  double initial_T;
  uint32_t num_passes;
  uint64_t num_coeffs;
  uint64_t data_size;
  read_header(file, initial_T, num_passes, num_coeffs, data_size);

  initial_T_ = initial_T;
  num_passes_ = num_passes;

  std::vector<uint8_t> payload(data_size);
  if (data_size > 0) {
    file.read(reinterpret_cast<char *>(payload.data()),
              static_cast<std::streamsize>(data_size));
    if (static_cast<uint64_t>(file.gcount()) != data_size) {
      throw std::runtime_error("Compressed data truncated");
    }
  }

  file.close();
  return decompress_tile(payload, initial_T_, num_coeffs);
}

std::vector<uint8_t>
WDRCompressor::compress_tile(const std::vector<double> &coeffs,
                             double initial_T) const {
  if (coeffs.empty()) {
    throw std::invalid_argument("Empty coefficient array");
  }
  if (initial_T <= 0.0) {
    throw std::invalid_argument("initial_T must be > 0");
  }

  EncoderState state;
  state.coeffs = &coeffs;
  for (size_t i = 0; i < coeffs.size(); ++i) {
    state.ics_indices.push_back(i);
  }

  AdaptiveModel sorting_model(6);    // 0-5
  AdaptiveModel refinement_model(2); // 0-1
  sorting_model.start_model();
  refinement_model.start_model();

  std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
  BitOutputStream bit_stream(buffer);
  ArithmeticCoder coder;
  coder.start_encoding(bit_stream, sorting_model);

  double T = initial_T;
  for (int pass = 0; pass < num_passes_; pass++) {
    std::vector<WDRSymbol> pass_symbol_stream;

    sorting_pass_encode(T, pass_symbol_stream, state);
    refinement_pass_encode(T, pass_symbol_stream, state);

    arithmetic_encode_stream(pass_symbol_stream, coder, sorting_model,
                             refinement_model);

    for (double val : state.tps) {
      double center = T + T / 2.0;
      state.scs.push_back(std::make_pair(val, center));
    }
    state.tps.clear();

    T = T / 2.0;
  }

  coder.done_encoding();
  bit_stream.flush();

  const std::string payload_str = buffer.str();
  return std::vector<uint8_t>(payload_str.begin(), payload_str.end());
}

std::vector<double>
WDRCompressor::decompress_tile(const std::vector<uint8_t> &payload,
                               double initial_T, uint64_t coeff_count) const {
  if (coeff_count == 0) {
    throw std::invalid_argument("coeff_count must be > 0");
  }
  if (initial_T <= 0.0) {
    throw std::invalid_argument("initial_T must be > 0");
  }

  std::stringstream buffer(std::ios::in | std::ios::out | std::ios::binary);
  if (!payload.empty()) {
    buffer.write(reinterpret_cast<const char *>(payload.data()),
                 static_cast<std::streamsize>(payload.size()));
    buffer.seekg(0);
  }

  BitInputStream bit_stream(buffer);
  ArithmeticCoder coder;

  AdaptiveModel sorting_model(6);
  AdaptiveModel refinement_model(2);
  sorting_model.start_model();
  refinement_model.start_model();

  coder.start_decoding(bit_stream, sorting_model);

  DecoderState state;
  state.reconstructed_array.assign(static_cast<size_t>(coeff_count), 0.0);
  for (size_t i = 0; i < coeff_count; ++i) {
    state.ics_indices.push_back(i);
  }

  double T = initial_T;

  for (int pass = 0; pass < num_passes_; pass++) {
    std::vector<size_t> pass_decoded_positions;
    std::vector<int> pass_decoded_signs;
    state.decoded_centers.clear();

    sorting_pass_decode(T, coder, sorting_model, pass_decoded_positions,
                        pass_decoded_signs, state.decoded_centers,
                        state.ics_indices);

    refinement_pass_decode(T, coder, refinement_model, state);

    for (size_t i = 0; i < pass_decoded_positions.size(); i++) {
      size_t array_pos = pass_decoded_positions[i];
      int sign = pass_decoded_signs[i];
      double center = state.decoded_centers[i];
      double signed_value = (sign == 1) ? center : -center;

      state.scs.push_back(std::make_pair(signed_value, center));
      state.scs_to_array_pos.push_back(array_pos);
      state.scs_signs.push_back(sign);
    }
    state.decoded_centers.clear();

    for (size_t i = 0; i < state.scs.size(); i++) {
      size_t array_pos = state.scs_to_array_pos[i];
      double reconstructed_value = state.scs[i].first;
      state.reconstructed_array[array_pos] = reconstructed_value;
    }

    T = T / 2.0;
  }

  return state.reconstructed_array;
}

void WDRCompressor::sorting_pass_encode(double T,
                                        std::vector<WDRSymbol> &symbol_stream,
                                        EncoderState &state) const {
  if (state.coeffs == nullptr) {
    throw std::invalid_argument(
        "EncoderState.coeffs must be set before sorting_pass_encode");
  }
  state.tps.clear();

  std::vector<int> indices;
  std::vector<int> signs;

  // Step 1: Find significant coefficients

  // We store the iterators to erase *after* we find the significant
  // coefficients. for this pass.
  std::vector<std::list<size_t>::iterator> iterators_to_erase;

  int current_list_index = 0;
  for (auto it = state.ics_indices.begin(); it != state.ics_indices.end();
       ++it) {
    // Get the coefficient value from the original coefficients
    double val = (*(state.coeffs))[*it];

    // If the coefficient is significant, we save its list index and sign.
    if (std::abs(val) >= T) {
      indices.push_back(current_list_index);
      signs.push_back(val >= 0 ? 1 : 0);
      state.tps.push_back(val);

      // Save the iterator to erase it
      iterators_to_erase.push_back(it);
    }
    current_list_index++;
  }

  // Remove the significant coefficients from the ICS_indices_list_.
  for (auto it : iterators_to_erase) {
    state.ics_indices.erase(it); // This is O(1) (double linked list)
  }

  // Step 2: Calculate count of significant coefficients
  int count = static_cast<int>(indices.size());
  int max_count = static_cast<int>(state.ics_indices.size() + count);

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
    double T, std::vector<WDRSymbol> &symbol_stream,
    EncoderState &state) const {

  // Refine the coefficients
  for (auto &pair : state.scs) {
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
                                        std::vector<double> &decoded_centers,
                                        std::list<size_t> &ics_to_array_map) const {
  // Clear decoded positions and signs
  decoded_positions.clear();
  decoded_signs.clear();
  decoded_centers.clear();

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
    decoded_centers.push_back(T + T / 2.0);

    // Erase from the list (O(1))
    ics_to_array_map.erase(it);
  }
}

void WDRCompressor::refinement_pass_decode(double T, ArithmeticCoder &coder,
                                           AdaptiveModel &refinement_model,
                                           DecoderState &state) const {
  // Decode the coefficients in the SCS
  for (size_t idx = 0; idx < state.scs.size(); idx++) {
    auto &pair = state.scs[idx];
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
    double sign = (state.scs_signs[idx] == 1) ? 1.0 : -1.0;
    pair.second = center;       // Update unsigned center
    pair.first = sign * center; // Update signed, refined value
  }
}

std::vector<int>
WDRCompressor::differential_encode(const std::vector<int> &indices) const {
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
WDRCompressor::differential_decode(const std::vector<int> &diff_indices) const {
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
                                         int value, bool sign) const {
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
                                        bool &sign_out) const {
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

void WDRCompressor::write_header(std::ofstream &stream, double initial_T,
                                 uint64_t num_coeffs, uint64_t data_size) {
  // Write magic number
  uint32_t magic = WDRFormat::MAGIC;
  stream.write(reinterpret_cast<const char *>(&magic), sizeof(magic));

  // Write version
  uint32_t version = WDRFormat::VERSION;
  stream.write(reinterpret_cast<const char *>(&version), sizeof(version));

  // Write initial_T (scaled to integer)
  uint64_t initial_T_int =
      static_cast<uint64_t>(initial_T * WDRFormat::T_SCALE);
  stream.write(reinterpret_cast<const char *>(&initial_T_int),
               sizeof(initial_T_int));

  // Write num_passes
  uint32_t num_passes = num_passes_;
  stream.write(reinterpret_cast<const char *>(&num_passes), sizeof(num_passes));

  // Write num_coeffs
  stream.write(reinterpret_cast<const char *>(&num_coeffs), sizeof(num_coeffs));

  // Write data_size
  stream.write(reinterpret_cast<const char *>(&data_size), sizeof(data_size));

  // Write reserved field
  uint32_t reserved = 0;
  stream.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));

  if (!stream.good()) {
    throw std::runtime_error("Failed to write file header");
  }
}

void WDRCompressor::read_header(std::ifstream &stream, double &initial_T,
                                uint32_t &num_passes, uint64_t &num_coeffs,
                                uint64_t &data_size) {
  // Read the magic number
  uint32_t magic;
  stream.read(reinterpret_cast<char *>(&magic), sizeof(magic));
  if (stream.gcount() != sizeof(magic) || magic != WDRFormat::MAGIC) {
    throw std::runtime_error("Invalid file format: bad magic number");
  }

  // Read the version
  uint32_t version;
  stream.read(reinterpret_cast<char *>(&version), sizeof(version));
  if (stream.gcount() != sizeof(version) || version != WDRFormat::VERSION) {
    throw std::runtime_error("Unsupported file format version");
  }

  // Read the initial_T
  uint64_t initial_T_int;
  stream.read(reinterpret_cast<char *>(&initial_T_int), sizeof(initial_T_int));
  initial_T = static_cast<double>(initial_T_int) / WDRFormat::T_SCALE;

  // Read the num_passes
  stream.read(reinterpret_cast<char *>(&num_passes), sizeof(num_passes));

  // Read the num_coeffs
  stream.read(reinterpret_cast<char *>(&num_coeffs), sizeof(num_coeffs));

  // Read the data_size
  stream.read(reinterpret_cast<char *>(&data_size), sizeof(data_size));

  // Read the reserved field
  uint32_t reserved;
  stream.read(reinterpret_cast<char *>(&reserved), sizeof(reserved));

  if (stream.gcount() != sizeof(reserved)) {
    throw std::runtime_error("Failed to read file header or file is truncated");
  }
}