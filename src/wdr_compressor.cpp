#include "wdr_compressor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
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

  // Calculate initial threshold
  initial_T_ = calculate_initial_T(coeffs);

  // Initialize coefficient sets
  ICS_.clear();
  SCS_.clear();
  TPS_.clear();

  // Initialize ICS with all coefficients
  ICS_ = coeffs;

  // Open output file
  std::ofstream file(output_file, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open output file: " + output_file);
  }

  // Write header (we'll update data_size later)
  uint64_t data_size_placeholder = 0;
  uint64_t num_coeffs = coeffs.size();
  write_header(file, initial_T_, num_coeffs, data_size_placeholder);

  // Get position of data start
  std::streampos data_start_pos = file.tellp();

  // Create the master symbol stream
  std::vector<WDRSymbol> symbol_stream;

  // Main compression loop - *generates* tagged state symbols (SORTING_PASS and
  // REFINEMENT_PASS)
  double T = initial_T_;
  for (int pass = 0; pass < num_passes_; pass++) {
    // Sorting pass - appends to symbol_stream
    sorting_pass_encode(T, symbol_stream);

    // Refinement pass - appends to symbol_stream
    refinement_pass_encode(T, symbol_stream);

    // Move coefficients from TPS to SCS with initial reconstruction value
    for (double val : TPS_) {
      double center = T + T / 2.0; // 1.5*T
      SCS_.push_back(std::make_pair(val, center));
    }

    TPS_.clear();

    // Update threshold
    T = T / 2.0;
  }

  // Final encoding step - *executes* the master symbol stream
  BitOutputStream bit_stream(file);
  arithmetic_encode_stream(symbol_stream, bit_stream);
  bit_stream.flush(); // Ensure final byte is written

  // Get position of data end
  std::streampos data_end_pos = file.tellp();
  uint64_t data_size = static_cast<uint64_t>(data_end_pos - data_start_pos);

  // Rewrite header with correct data size
  file.seekp(0);
  write_header(file, initial_T_, num_coeffs, data_size);

  file.close();
}

void WDRCompressor::arithmetic_encode_stream(
    const std::vector<WDRSymbol> &symbol_stream, BitOutputStream &out_stream) {

  // Create and initialize models for the state machine
  // The sorting_model uses a 6-symbol alphabet to resolve the
  // ambiguity between a differential index of 0 and 1.
  //
  // The 0/1 Ambiguity:
  // Binary Reduction = "remove the MSB".
  // - Value 1 (binary '1') -> remove MSB '1' -> empty bit list '[]'
  // - Value 0 (binary '0') -> (no MSB)     -> empty bit list '[]'
  //
  // To solve this, we define 4 EOM symbols:
  // 0 = bit '0'
  // 1 = bit '1'
  // 2 = POS_SIGN (EOM, implies value 1)
  // 3 = NEG_SIGN (EOM, implies value 1)
  // 4 = ZERO_POS (EOM, implies value 0)
  // 5 = ZERO_NEG (EOM, implies value 0)
  AdaptiveModel sorting_model(6);
  AdaptiveModel refinement_model(2); // 0,1

  // Initialize models
  sorting_model.start_model();
  refinement_model.start_model();

  ArithmeticCoder coder;
  // API usage: Start encoding with a default model
  coder.start_encoding(out_stream, sorting_model);

  // Execute the script
  for (const WDRSymbol &sym : symbol_stream) {
    if (sym.context == SymbolContext::SORTING_PASS) {
      coder.encode_symbol(sym.symbol, sorting_model);
      sorting_model.update_model(sym.symbol); // API usage
    } else {
      // REFINEMENT_PASS
      coder.encode_symbol(sym.symbol, refinement_model);
      refinement_model.update_model(sym.symbol); // API usage
    }
  }

  coder.done_encoding();
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

  ICS_.clear();
  SCS_.clear();
  TPS_.clear();

  BitInputStream bit_stream(file);
  ArithmeticCoder coder;

  // Create and initialize models for the state machine

  // 0,1,POS(2),NEG(3),ZERO_POS(4),ZERO_NEG(5) See arithmetic_encode_stream() for details
  AdaptiveModel sorting_model(6);    
  AdaptiveModel refinement_model(2); // 0,1

  // API usage: Initialize models
  sorting_model.start_model();
  refinement_model.start_model();

  // Coder requires a default model to start decoding, 
  // in practice the states define the model used for decoding.
  // we will fix this later, for now we bypass the api requirement by using 
  // the sorting_model.
  coder.start_decoding(bit_stream, sorting_model);

  scs_to_array_pos_.clear();
  scs_signs_.clear();
  reconstructed_array_.resize(num_coeffs, 0.0);

  std::vector<size_t> ics_to_array_map;
  ics_to_array_map.reserve(num_coeffs);
  for (size_t i = 0; i < num_coeffs; i++) {
    ics_to_array_map.push_back(i);
  }

  double T = initial_T_;

  // Run the state machine
  for (int pass = 0; pass < num_passes_; pass++) {
    // State 1: Sorting Pass
    std::vector<size_t> pass_decoded_positions;
    std::vector<int> pass_decoded_signs;

    // Call 6-argument version, per HPP
    sorting_pass_decode(T, coder, sorting_model, pass_decoded_positions,
                        pass_decoded_signs, ics_to_array_map);

    // State 2: Refinement Pass
    refinement_pass_decode(T, coder, refinement_model);

    // Move coefficients from TPS to SCS
    for (size_t i = 0; i < pass_decoded_positions.size(); i++) {
      size_t array_pos = pass_decoded_positions[i];
      int sign = pass_decoded_signs[i];
      double center = TPS_[i]; // Unsigned center

      double signed_value = (sign == 1) ? center : -center;
      SCS_.push_back(std::make_pair(signed_value, center));

      scs_to_array_pos_.push_back(array_pos);
      scs_signs_.push_back(sign);
    }
    TPS_.clear();

    // Update reconstructed array
    for (size_t i = 0; i < SCS_.size(); i++) {
      size_t array_pos = scs_to_array_pos_[i];
      double reconstructed_value = SCS_[i].first; // Signed, refined value
      reconstructed_array_[array_pos] = reconstructed_value;
    }

    T = T / 2.0;
  }

  file.close();
  return reconstructed_array_;
}

void WDRCompressor::sorting_pass_encode(double T,
                                        std::vector<WDRSymbol> &symbol_stream) {
  std::vector<int> indices;
  std::vector<int> signs;

  // Step 1: Find significant coefficients
  for (size_t i = 0; i < ICS_.size(); i++) {
    if (std::abs(ICS_[i]) >= T) {
      indices.push_back(static_cast<int>(i));
      signs.push_back(ICS_[i] >= 0 ? 1 : 0);
      TPS_.push_back(ICS_[i]);
    }
  }

  // Step 2: Calculate count of significant coefficients
  int count = static_cast<int>(indices.size());
  int max_count = static_cast<int>(ICS_.size());

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

  // Compact the ICS list
  std::vector<double> new_ICS;
  for (size_t i = 0; i < ICS_.size(); i++) {
    if (std::abs(ICS_[i]) < T) {
      new_ICS.push_back(ICS_[i]);
    }
  }
  ICS_ = new_ICS;
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
                                        std::vector<size_t> &ics_to_array_map) {
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

  // Step 2: Decode indices using sorting_model
  std::vector<int> diff_indices;
  std::vector<int> signs;
  for (int i = 0; i < count; i++) {
    bool sign_out;
    int diff_idx = read_binary_expanded(coder, sorting_model, sign_out);
    diff_indices.push_back(diff_idx);
    signs.push_back(sign_out ? 1 : 0);
  }

  std::vector<int> ics_indices = differential_decode(diff_indices);

  // Step 4: Map ICS indices to array positions
  TPS_.clear();
  std::vector<bool> ics_indices_decoded(ics_to_array_map.size(), false);

  for (size_t i = 0; i < ics_indices.size(); i++) {
    int ics_index = ics_indices[i];
    if (ics_index < 0 ||
        ics_index >= static_cast<int>(ics_to_array_map.size())) {
      throw std::runtime_error("Decoded invalid ICS index");
    }
    ics_indices_decoded[ics_index] = true;

    decoded_positions.push_back(ics_to_array_map[ics_index]);
    decoded_signs.push_back(signs[i]);
    TPS_.push_back(T + T / 2.0); // 1.5*T (unsigned)
  }

  // Step 5: Compact ics_to_array_map
  std::vector<size_t> new_ics_to_array_map;
  for (size_t i = 0; i < ics_to_array_map.size(); i++) {
    if (!ics_indices_decoded[i]) {
      new_ics_to_array_map.push_back(ics_to_array_map[i]);
    }
  }
  ics_to_array_map = new_ics_to_array_map;
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
    // This resolves the 0/1 ambiguity (see comment in arithmetic_encode_stream).
    // We send a unique, signed symbol for zero.
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