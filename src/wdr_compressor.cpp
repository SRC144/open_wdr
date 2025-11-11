#include "wdr_compressor.hpp"
#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <cstring>
#include <iostream>
#include <sstream>
#include <iomanip>

WDRCompressor::WDRCompressor(int num_passes)
    : num_passes_(num_passes) {
    if (num_passes < 1) {
        throw std::invalid_argument("num_passes must be >= 1");
    }
}

double WDRCompressor::calculate_initial_T(const std::vector<double>& coeffs) {
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
        return 1.0;  // Default threshold for all-zero coefficients
    }
    
    // Find the largest power of 2 such that max_abs < 2*T and max_abs >= T
    // T = 2^k where k is the largest integer such that 2^k <= max_abs < 2^(k+1)
    double T = std::pow(2.0, std::floor(std::log2(max_abs)));
    
    // Ensure max_abs >= T (if max_abs is exactly a power of 2, T might be too large)
    if (max_abs < T) {
        T = T / 2.0;
    }
    
    // Ensure max_abs < 2*T
    if (max_abs >= 2.0 * T) {
        T = T * 2.0;
    }
    
    return T;
}

void WDRCompressor::compress(const std::vector<double>& coeffs, const std::string& output_file) {
    if (coeffs.empty()) {
        throw std::invalid_argument("Empty coefficient array");
    }
    
    // Calculate initial threshold
    initial_T_ = calculate_initial_T(coeffs);
    
    // Debug: Log compression start
    WDR_DEBUG_LOG("ENCODER: Starting compression: num_coeffs=" << coeffs.size() 
        << " initial_T=" << std::fixed << std::setprecision(6) << initial_T_ 
        << " num_passes=" << num_passes_);
    
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
    // For now, write a placeholder
    uint64_t data_size_placeholder = 0;
    uint64_t num_coeffs = coeffs.size();
    write_header(file, initial_T_, num_coeffs, data_size_placeholder);
    
    // Get position of data start
    std::streampos data_start_pos = file.tellp();
    
    // Create bit stream and arithmetic coder
    BitOutputStream bit_stream(file);
    
    // Create separate adaptive models for different symbol types
    AdaptiveModel index_model(2);    // For binary-reduced index bits
    AdaptiveModel sign_model(2);     // For sign bits
    AdaptiveModel refinement_model(2); // For refinement bits
    
    ArithmeticCoder coder;
    coder.start_encoding(bit_stream, index_model);  // Use index_model for initialization (coder needs a model reference)
    
    // Main compression loop
    double T = initial_T_;
    for (int pass = 0; pass < num_passes_; pass++) {
        WDR_DEBUG_LOG("=== ENCODER PASS " << pass << " ===");
        
        // Sorting pass (uses index_model and sign_model)
        sorting_pass_encode(T, coder, index_model, sign_model);
        
        // Refinement pass (uses refinement_model)
        refinement_pass_encode(T, coder, refinement_model);
        
        // Move coefficients from TPS to SCS with initial reconstruction value
        // center = T + T/2 = 1.5*T (before halving T)
        for (double val : TPS_) {
            double center = T + T / 2.0;  // 1.5*T
            SCS_.push_back(std::make_pair(val, center));
        }
        
        // Debug: Log SCS state after adding new coefficients
        WDR_DEBUG_LOG("ENCODER: After pass " << pass << ": SCS_size=" << SCS_.size());
        
        TPS_.clear();
        
        // Update threshold
        T = T / 2.0;
    }
    
    // Finish encoding
    coder.done_encoding();
    
    // Get position of data end
    std::streampos data_end_pos = file.tellp();
    uint64_t data_size = static_cast<uint64_t>(data_end_pos - data_start_pos);
    
    // Rewrite header with correct data size
    file.seekp(0);
    write_header(file, initial_T_, num_coeffs, data_size);
    
    file.close();
}

std::vector<double> WDRCompressor::decompress(const std::string& input_file) {
    // Open input file
    std::ifstream file(input_file, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open input file: " + input_file);
    }
    
    // Read header
    double initial_T;
    uint32_t num_passes;
    uint64_t num_coeffs;
    uint64_t data_size;
    read_header(file, initial_T, num_passes, num_coeffs, data_size);
    
    initial_T_ = initial_T;
    num_passes_ = num_passes;
    
    // Initialize coefficient sets
    ICS_.clear();
    SCS_.clear();
    TPS_.clear();
    
    // Initialize ICS with zeros (we'll decode which ones become significant)
    ICS_.resize(num_coeffs, 0.0);
    
    // Create bit stream and arithmetic coder
    BitInputStream bit_stream(file);
    
    // Create separate adaptive models for different symbol types
    AdaptiveModel index_model(2);    // For binary-reduced index bits
    AdaptiveModel sign_model(2);     // For sign bits
    AdaptiveModel refinement_model(2); // For refinement bits
    
    ArithmeticCoder coder;
    coder.start_decoding(bit_stream, index_model);  // Use index_model for initialization
    
    // Initialize decoder-side structures
    scs_to_array_pos_.clear();
    scs_signs_.clear();
    reconstructed_array_.resize(num_coeffs, 0.0);
    
    // Initialize ICS as empty (we'll track positions via mapping)
    ICS_.clear();
    
    // Maintain mapping from ICS index to array position
    // Initially: all positions are in ICS
    std::vector<size_t> ics_to_array_map;
    ics_to_array_map.reserve(num_coeffs);
    for (size_t i = 0; i < num_coeffs; i++) {
        ics_to_array_map.push_back(i);
    }
    
    // Main decompression loop
    double T = initial_T_;
    
    // Debug: Log initial state
    WDR_DEBUG_LOG("DECODER: Starting decompression: num_coeffs=" << num_coeffs 
        << " initial_T=" << std::fixed << std::setprecision(6) << initial_T_ 
        << " num_passes=" << num_passes_);
    
    for (int pass = 0; pass < num_passes_; pass++) {
        WDR_DEBUG_LOG("=== DECODER PASS " << pass << " ===");
        
        // Sorting pass (decode positions and signs for NEW significant coefficients)
        // This must match the encoder's order: sorting pass comes first in the bitstream
        std::vector<size_t> pass_decoded_positions;
        std::vector<int> pass_decoded_signs;
        sorting_pass_decode(T, coder, index_model, sign_model, pass_decoded_positions, pass_decoded_signs, ics_to_array_map);
        
        // Refinement pass (decode refinement bits and update SCS centers)
        // This refines coefficients that were added in PREVIOUS passes (already in SCS)
        // Note: On first pass, SCS is empty, so this does nothing
        refinement_pass_decode(T, coder, refinement_model);
        
        // Move coefficients from TPS to SCS with initial reconstruction value
        // Also track their array positions and signs
        for (size_t i = 0; i < pass_decoded_positions.size() && i < TPS_.size() && i < pass_decoded_signs.size(); i++) {
            size_t array_pos = pass_decoded_positions[i];
            int sign = pass_decoded_signs[i];
            double center = TPS_[i];  // Center from sorting_pass_decode (unsigned, 1.5*T)
            
            // Store in SCS: use center as both value and center (value will be updated during refinement)
            // The actual reconstructed value will be sign * center when we update the array
            SCS_.push_back(std::make_pair(center, center));  // Both are unsigned centers representing |val|
            scs_to_array_pos_.push_back(array_pos);
            scs_signs_.push_back(sign);
        }
        
        // Debug: Log SCS state after adding new coefficients
        if (SCS_.size() <= 10) {
            std::ostringstream oss_scs;
            oss_scs << "DECODER: After adding new coeffs: SCS_size=" << SCS_.size() << " scs_to_array_pos=[";
            for (size_t i = 0; i < scs_to_array_pos_.size(); i++) {
                if (i > 0) oss_scs << ", ";
                oss_scs << scs_to_array_pos_[i];
            }
            oss_scs << "]";
            WDR_DEBUG_LOG(oss_scs.str());
        } else {
            WDR_DEBUG_LOG("DECODER: After adding new coeffs: SCS_size=" << SCS_.size());
        }
        
        TPS_.clear();
        
        // Note: ICS indices are already removed from ics_to_array_map inside sorting_pass_decode
        
        // Update reconstructed array from all coefficients in SCS (including newly added and refined ones)
        // Apply the sign to the center to get the final reconstructed value
        for (size_t i = 0; i < SCS_.size(); i++) {
            if (i < scs_to_array_pos_.size() && i < scs_signs_.size()) {
                size_t array_pos = scs_to_array_pos_[i];
                int sign = scs_signs_[i];
                double center = SCS_[i].second;  // Current center (unsigned, representing |val|)
                
                // Apply sign to get the final reconstructed value
                double reconstructed_value = (sign == 1) ? center : -center;
                reconstructed_array_[array_pos] = reconstructed_value;
            }
        }
        
        // Debug: Log reconstructed array state
        {
            std::ostringstream oss_arr;
            oss_arr << "DECODER: After pass " << pass << ": reconstructed_array=[";
            size_t print_count = std::min(reconstructed_array_.size(), size_t(10));
            for (size_t i = 0; i < print_count; i++) {
                if (i > 0) oss_arr << ", ";
                oss_arr << std::fixed << std::setprecision(6) << reconstructed_array_[i];
            }
            if (reconstructed_array_.size() > print_count) {
                oss_arr << ", ... (total " << reconstructed_array_.size() << ")";
            }
            oss_arr << "]";
            WDR_DEBUG_LOG(oss_arr.str());
        }
        
        // Update threshold for next pass
        T = T / 2.0;
    }
    
    file.close();
    
    // Debug: Log final reconstructed array before returning
    {
        std::ostringstream oss_final;
        oss_final << "DECODER: Final reconstructed_array=[";
        size_t print_count_final = std::min(reconstructed_array_.size(), size_t(10));
        for (size_t i = 0; i < print_count_final; i++) {
            if (i > 0) oss_final << ", ";
            oss_final << std::fixed << std::setprecision(6) << reconstructed_array_[i];
        }
        if (reconstructed_array_.size() > print_count_final) {
            oss_final << ", ... (total " << reconstructed_array_.size() << ")";
        }
        oss_final << "]";
        WDR_DEBUG_LOG(oss_final.str());
    }
    
    // Debug: Log array size
    WDR_DEBUG_LOG("DECODER: Returning array of size " << reconstructed_array_.size());
    
    return reconstructed_array_;
}

void WDRCompressor::sorting_pass_encode(double T, ArithmeticCoder& coder, AdaptiveModel& index_model, AdaptiveModel& sign_model) {
    // Debug: Log ICS state before sorting pass
    {
        std::ostringstream oss;
        oss << "ENCODER: T=" << std::fixed << std::setprecision(6) << T 
            << " ICS_size=" << ICS_.size();
        if (ICS_.size() <= 10) {
            oss << " ICS=[";
            for (size_t i = 0; i < ICS_.size(); i++) {
                if (i > 0) oss << ", ";
                oss << ICS_[i];
            }
            oss << "]";
        } else {
            oss << " ICS=[";
            for (size_t i = 0; i < 5; i++) {
                if (i > 0) oss << ", ";
                oss << ICS_[i];
            }
            oss << ", ... (total " << ICS_.size() << ")]";
        }
        WDR_DEBUG_LOG(oss.str());
    }
    
    std::vector<int> indices;
    std::vector<int> signs;
    
    // Find significant coefficients
    for (size_t i = 0; i < ICS_.size(); i++) {
        if (std::abs(ICS_[i]) >= T) {
            indices.push_back(static_cast<int>(i));
            signs.push_back(ICS_[i] >= 0 ? 1 : 0);
            TPS_.push_back(ICS_[i]);
        }
    }
    
    // Debug: Log found significant coefficients
    {
        std::ostringstream oss;
        oss << "ENCODER: Found " << indices.size() << " significant coefficients";
        if (!indices.empty()) {
            oss << " indices=[";
            for (size_t i = 0; i < indices.size(); i++) {
                if (i > 0) oss << ", ";
                oss << indices[i];
            }
            oss << "] values=[";
            for (size_t i = 0; i < indices.size(); i++) {
                if (i > 0) oss << ", ";
                oss << ICS_[indices[i]];
            }
            oss << "] signs=[";
            for (size_t i = 0; i < signs.size(); i++) {
                if (i > 0) oss << ", ";
                oss << signs[i];
            }
            oss << "]";
        }
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Encode the count of significant coefficients first
    // This allows the decoder to know how many coefficients to expect
    // We ALWAYS encode the count, even if it's 0, so the decoder knows what to expect
    
    // Encode count of significant coefficients
    int count = static_cast<int>(indices.size());
    int max_count = static_cast<int>(ICS_.size());
    
    // Calculate bits needed to encode count
    // Handle edge case: if max_count is 0, we still need to encode (count will be 0)
    int bits_needed = 1;  // At least 1 bit
    if (max_count > 0) {
        bits_needed = static_cast<int>(std::ceil(std::log2(max_count + 1)));
    }
    
    // Validate count
    if (count < 0 || count > max_count) {
        throw std::runtime_error("Invalid count: " + std::to_string(count) + " (max: " + std::to_string(max_count) + ")");
    }
    
    // Encode count bit by bit (MSB first) using index_model
    for (int bit_pos = bits_needed - 1; bit_pos >= 0; bit_pos--) {
        int bit = (count >> bit_pos) & 1;
        coder.encode_symbol(bit, index_model);
        index_model.update_model(bit);
    }
    
    if (count == 0) {
        return;  // No coefficients to encode (but count was encoded)
    }
    
    // Apply differential coding
    std::vector<int> diff_indices = differential_encode(indices);
    
    // Debug: Log differential encoding result
    {
        std::ostringstream oss;
        oss << "ENCODER: diff_indices=[";
        for (size_t i = 0; i < diff_indices.size(); i++) {
            if (i > 0) oss << ", ";
            oss << diff_indices[i];
        }
        oss << "]";
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Calculate maximum number of bits needed for reduced indices
    int reduced_bits_needed = 0;
    if (max_count > 0) {
        reduced_bits_needed = std::max(0, static_cast<int>(std::ceil(std::log2(max_count + 1))) - 1);
    }
    // Calculate bits needed to encode the length of reduced indices
    // We need at least 1 bit to encode length (even if it's always 0)
    int length_bits = 1;
    if (reduced_bits_needed > 0) {
        length_bits = std::max(1, static_cast<int>(std::ceil(std::log2(reduced_bits_needed + 1))));
    }
    
    // Encode each index using binary reduction and arithmetic coding
    // Interleave: length(1) + P'(1) + sign(1) + length(2) + P'(2) + sign(2) + ...
    for (size_t i = 0; i < diff_indices.size(); i++) {
        int diff_idx = diff_indices[i];
        
        // Encode index bits using index_model
        // Handle special case: diff_idx == 0 cannot use binary reduction
        // We encode a flag: 0 = value is 0, 1 = value > 0, then encode the value
        if (diff_idx == 0) {
            // Encode flag: 0 means the value is 0
            coder.encode_symbol(0, index_model);
            index_model.update_model(0);
        } else {
            // Encode flag: 1 means the value is > 0
            coder.encode_symbol(1, index_model);
            index_model.update_model(1);
            
            // Apply binary reduction
            std::vector<bool> reduced_bits = binary_reduce(diff_idx);
            int reduced_length = static_cast<int>(reduced_bits.size());
            
            // Encode the length of the reduced index
            for (int bit_pos = length_bits - 1; bit_pos >= 0; bit_pos--) {
                int bit = (reduced_length >> bit_pos) & 1;
                coder.encode_symbol(bit, index_model);
                index_model.update_model(bit);
            }
            
            // Encode reduced bits (LSB first) using index_model
            for (bool bit : reduced_bits) {
                int symbol = bit ? 1 : 0;
                coder.encode_symbol(symbol, index_model);
                index_model.update_model(symbol);
            }
        }
        // Note: diff_idx < 0 should not happen with proper differential coding
        
        // Encode sign using sign_model
        int sign = signs[i];
        coder.encode_symbol(sign, sign_model);
        sign_model.update_model(sign);
    }
    
    // Remove significant coefficients from ICS (compact the list)
    std::vector<double> new_ICS;
    for (size_t i = 0; i < ICS_.size(); i++) {
        if (std::abs(ICS_[i]) < T) {
            new_ICS.push_back(ICS_[i]);
        }
    }
    
    // Debug: Log ICS state after removal
    {
        std::ostringstream oss;
        oss << "ENCODER: After removal: ICS_size=" << new_ICS.size() 
            << " (removed " << (ICS_.size() - new_ICS.size()) << " coefficients)";
        if (new_ICS.size() <= 10) {
            oss << " ICS=[";
            for (size_t i = 0; i < new_ICS.size(); i++) {
                if (i > 0) oss << ", ";
                oss << new_ICS[i];
            }
            oss << "]";
        }
        WDR_DEBUG_LOG(oss.str());
    }
    
    ICS_ = new_ICS;
}

void WDRCompressor::refinement_pass_encode(double T, ArithmeticCoder& coder, AdaptiveModel& refinement_model) {
    for (auto& pair : SCS_) {
        double val = pair.first;
        double center = pair.second;
        
        // Work with absolute values: the algorithm refines |val|, not val
        double abs_val = std::abs(val);
        
        // Calculate interval (center is always positive, representing |val|)
        double low = center - T;
        double high = center + T;
        
        // Determine which half the absolute value is in
        int bit;
        if (abs_val >= center) {
            bit = 1;
            center = (center + high) / 2.0;
        } else {
            bit = 0;
            center = (low + center) / 2.0;
        }
        
        // Update center (always positive, representing |val|)
        pair.second = center;
        
        // Encode bit using refinement_model
        coder.encode_symbol(bit, refinement_model);
        refinement_model.update_model(bit);
    }
}

void WDRCompressor::sorting_pass_decode(double T, ArithmeticCoder& coder, AdaptiveModel& index_model, AdaptiveModel& sign_model, std::vector<size_t>& decoded_positions, std::vector<int>& decoded_signs, std::vector<size_t>& ics_to_array_map) {
    decoded_positions.clear();
    decoded_signs.clear();
    
    // Debug: Log ICS map state before decoding
    {
        std::ostringstream oss;
        oss << "DECODER: T=" << std::fixed << std::setprecision(6) << T 
            << " ICS_map_size=" << ics_to_array_map.size();
        if (ics_to_array_map.size() <= 10) {
            oss << " ICS_map=[";
            for (size_t i = 0; i < ics_to_array_map.size(); i++) {
                if (i > 0) oss << ", ";
                oss << ics_to_array_map[i];
            }
            oss << "]";
        } else {
            oss << " ICS_map=[";
            for (size_t i = 0; i < 5; i++) {
                if (i > 0) oss << ", ";
                oss << ics_to_array_map[i];
            }
            oss << ", ... (total " << ics_to_array_map.size() << ")]";
        }
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Step 1: Decode the count of significant coefficients
    int max_count = static_cast<int>(ics_to_array_map.size());
    
    // Calculate bits needed to encode count
    // Handle edge case: if max_count is 0, we still need to decode (count will be 0)
    int bits_needed = 1;  // At least 1 bit
    if (max_count > 0) {
        bits_needed = static_cast<int>(std::ceil(std::log2(max_count + 1)));
    }
    
    int count = 0;
    for (int bit_pos = bits_needed - 1; bit_pos >= 0; bit_pos--) {
        int bit = coder.decode_symbol(index_model);
        index_model.update_model(bit);
        count = (count << 1) | bit;
    }
    
    // Debug: Log decoded count
    WDR_DEBUG_LOG("DECODER: Decoded count=" << count << " (max_count=" << max_count << ")");
    
    // Validate count
    if (count < 0 || count > max_count) {
        throw std::runtime_error("Invalid count decoded: " + std::to_string(count));
    }
    
    if (count == 0) {
        WDR_DEBUG_LOG("DECODER: No coefficients to decode in this pass");
        return;  // No coefficients to decode in this pass
    }
    
    // Step 2: Calculate length_bits (number of bits needed to encode reduced index length)
    int reduced_bits_needed = 0;
    if (max_count > 0) {
        reduced_bits_needed = std::max(0, static_cast<int>(std::ceil(std::log2(max_count + 1))) - 1);
    }
    // Calculate bits needed to encode the length of reduced indices
    // We need at least 1 bit to encode length (even if it's always 0)
    int length_bits = 1;
    if (reduced_bits_needed > 0) {
        length_bits = std::max(1, static_cast<int>(std::ceil(std::log2(reduced_bits_needed + 1))));
    }
    
    // Step 3: Decode the interleaved stream of reduced indices and signs
    std::vector<int> diff_indices;
    std::vector<int> signs;
    
    for (int i = 0; i < count; i++) {
        // Decode flag: 0 = value is 0, 1 = value > 0
        int flag = coder.decode_symbol(index_model);
        index_model.update_model(flag);
        
        int diff_idx = 0;
        if (flag == 1) {
            // Value > 0: decode the length of the reduced index
            int reduced_length = 0;
            for (int bit_pos = length_bits - 1; bit_pos >= 0; bit_pos--) {
                int bit = coder.decode_symbol(index_model);
                index_model.update_model(bit);
                reduced_length = (reduced_length << 1) | bit;
            }
            
            // Decode binary-reduced differential index
            std::vector<bool> reduced_bits;
            for (int j = 0; j < reduced_length; j++) {
                int bit = coder.decode_symbol(index_model);
                index_model.update_model(bit);
                reduced_bits.push_back(bit == 1);
            }
            
            // Expand the reduced index
            diff_idx = binary_expand(reduced_bits);
        }
        // If flag == 0, diff_idx remains 0
        diff_indices.push_back(diff_idx);
        
        // Decode sign
        int sign = coder.decode_symbol(sign_model);
        sign_model.update_model(sign);
        signs.push_back(sign);
    }
    
    // Debug: Log decoded differential indices
    {
        std::ostringstream oss;
        oss << "DECODER: diff_indices=[";
        for (size_t i = 0; i < diff_indices.size(); i++) {
            if (i > 0) oss << ", ";
            oss << diff_indices[i];
        }
        oss << "] signs=[";
        for (size_t i = 0; i < signs.size(); i++) {
            if (i > 0) oss << ", ";
            oss << signs[i];
        }
        oss << "]";
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Step 4: Invert differential coding to get original ICS indices
    std::vector<int> ics_indices = differential_decode(diff_indices);
    
    // Debug: Log decoded ICS indices
    {
        std::ostringstream oss;
        oss << "DECODER: ics_indices=[";
        for (size_t i = 0; i < ics_indices.size(); i++) {
            if (i > 0) oss << ", ";
            oss << ics_indices[i];
        }
        oss << "]";
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Step 5: Map ICS indices to actual array positions and reconstruct coefficients
    TPS_.clear();
    std::vector<bool> ics_indices_decoded(ics_to_array_map.size(), false);
    for (size_t i = 0; i < ics_indices.size(); i++) {
        int ics_index = ics_indices[i];
        int sign = signs[i];
        
        // Map ICS index to array position
        if (ics_index >= 0 && ics_index < static_cast<int>(ics_to_array_map.size())) {
            size_t array_pos = ics_to_array_map[ics_index];
            decoded_positions.push_back(array_pos);
            decoded_signs.push_back(sign);
            
            // Mark this ICS index as decoded (so we can remove it from the map later)
            ics_indices_decoded[ics_index] = true;
            
            // Calculate initial reconstruction value (always positive, representing |val|)
            // The center is the center of the interval [T, 2T) for the absolute value
            double center = T + T / 2.0;  // 1.5*T (unsigned)
            
            // Store center in TPS (unsigned, representing |val|)
            TPS_.push_back(center);
            
            // Debug: Log per-coefficient mapping
            WDR_DEBUG_LOG("DECODER: coeff[" << i << "] ics_index=" << ics_index 
                << " -> array_pos=" << array_pos << " sign=" << sign 
                << " center=" << std::fixed << std::setprecision(6) << center);
        } else {
            // Debug: Log invalid ICS index
            WDR_DEBUG_LOG("DECODER: ERROR: Invalid ics_index=" << ics_index 
                << " (ics_to_array_map.size()=" << ics_to_array_map.size() << ")");
        }
    }
    
    // Debug: Log decoded array positions
    {
        std::ostringstream oss;
        oss << "DECODER: decoded_positions=[";
        for (size_t i = 0; i < decoded_positions.size(); i++) {
            if (i > 0) oss << ", ";
            oss << decoded_positions[i];
        }
        oss << "]";
        WDR_DEBUG_LOG(oss.str());
    }
    
    // Step 6: Remove decoded ICS indices from the map (by creating a new map without them)
    std::vector<size_t> new_ics_to_array_map;
    for (size_t i = 0; i < ics_to_array_map.size(); i++) {
        if (!ics_indices_decoded[i]) {
            new_ics_to_array_map.push_back(ics_to_array_map[i]);
        }
    }
    
    // Debug: Log ICS map state after removal
    {
        std::ostringstream oss;
        oss << "DECODER: After removal: ICS_map_size=" << new_ics_to_array_map.size() 
            << " (removed " << (ics_to_array_map.size() - new_ics_to_array_map.size()) << " indices)";
        if (new_ics_to_array_map.size() <= 10) {
            oss << " ICS_map=[";
            for (size_t i = 0; i < new_ics_to_array_map.size(); i++) {
                if (i > 0) oss << ", ";
                oss << new_ics_to_array_map[i];
            }
            oss << "]";
        }
        WDR_DEBUG_LOG(oss.str());
    }
    
    ics_to_array_map = new_ics_to_array_map;
}

void WDRCompressor::refinement_pass_decode(double T, ArithmeticCoder& coder, AdaptiveModel& refinement_model) {
    for (auto& pair : SCS_) {
        double center = pair.second;
        
        // Calculate interval
        double low = center - T;
        double high = center + T;
        
        // Decode bit using refinement_model
        int bit = coder.decode_symbol(refinement_model);
        refinement_model.update_model(bit);
        
        // Update center based on bit
        if (bit == 1) {
            center = (center + high) / 2.0;
        } else {
            center = (low + center) / 2.0;
        }
        
        // Update center and value
        pair.second = center;
        pair.first = center;  // For reconstruction
    }
}

std::vector<int> WDRCompressor::differential_encode(const std::vector<int>& indices) {
    if (indices.empty()) {
        return {};
    }
    
    std::vector<int> diff_indices;
    diff_indices.push_back(indices[0]);
    
    for (size_t i = 1; i < indices.size(); i++) {
        diff_indices.push_back(indices[i] - indices[i-1]);
    }
    
    return diff_indices;
}

std::vector<int> WDRCompressor::differential_decode(const std::vector<int>& diff_indices) {
    if (diff_indices.empty()) {
        return {};
    }
    
    std::vector<int> indices;
    indices.push_back(diff_indices[0]);
    
    for (size_t i = 1; i < diff_indices.size(); i++) {
        indices.push_back(indices[i-1] + diff_indices[i]);
    }
    
    return indices;
}

std::vector<bool> WDRCompressor::binary_reduce(int value) {
    if (value <= 0) {
        throw std::invalid_argument("Value must be > 0 for binary reduction");
    }
    
    // Find the position of the MSB
    int msb_pos = 0;
    int temp = value;
    while (temp > 1) {
        temp >>= 1;
        msb_pos++;
    }
    
    // Extract bits after MSB (LSB first)
    std::vector<bool> bits;
    for (int i = 0; i < msb_pos; i++) {
        bits.push_back((value >> i) & 1);
    }
    
    return bits;
}

int WDRCompressor::binary_expand(const std::vector<bool>& bits) {
    if (bits.empty()) {
        // Empty bits means the value was 1 (binary "1" with MSB removed)
        // Value 0 is handled separately with a flag
        return 1;
    }
    
    // Reconstruct value by adding MSB = 1
    int value = 1 << bits.size();  // MSB = 1, shift by number of bits
    for (size_t i = 0; i < bits.size(); i++) {
        if (bits[i]) {
            value |= (1 << i);
        }
    }
    
    return value;
}

void WDRCompressor::write_header(std::ofstream& stream, double initial_T, uint64_t num_coeffs, uint64_t data_size) {
    // Write magic number
    uint32_t magic = WDRFormat::MAGIC;
    stream.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    
    // Write version
    uint32_t version = WDRFormat::VERSION;
    stream.write(reinterpret_cast<const char*>(&version), sizeof(version));
    
    // Write initial_T (scaled to integer)
    uint64_t initial_T_int = static_cast<uint64_t>(initial_T * WDRFormat::T_SCALE);
    stream.write(reinterpret_cast<const char*>(&initial_T_int), sizeof(initial_T_int));
    
    // Write num_passes
    uint32_t num_passes = num_passes_;
    stream.write(reinterpret_cast<const char*>(&num_passes), sizeof(num_passes));
    
    // Write num_coeffs
    stream.write(reinterpret_cast<const char*>(&num_coeffs), sizeof(num_coeffs));
    
    // Write data_size
    stream.write(reinterpret_cast<const char*>(&data_size), sizeof(data_size));
    
    // Write reserved field
    uint32_t reserved = 0;
    stream.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    
    if (!stream.good()) {
        throw std::runtime_error("Failed to write file header");
    }
}

void WDRCompressor::read_header(std::ifstream& stream, double& initial_T, uint32_t& num_passes, uint64_t& num_coeffs, uint64_t& data_size) {
    // Read magic number
    uint32_t magic;
    stream.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != WDRFormat::MAGIC) {
        throw std::runtime_error("Invalid file format: bad magic number");
    }
    
    // Read version
    uint32_t version;
    stream.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (version != WDRFormat::VERSION) {
        throw std::runtime_error("Unsupported file format version");
    }
    
    // Read initial_T
    uint64_t initial_T_int;
    stream.read(reinterpret_cast<char*>(&initial_T_int), sizeof(initial_T_int));
    initial_T = static_cast<double>(initial_T_int) / WDRFormat::T_SCALE;
    
    // Read num_passes
    stream.read(reinterpret_cast<char*>(&num_passes), sizeof(num_passes));
    
    // Read num_coeffs
    stream.read(reinterpret_cast<char*>(&num_coeffs), sizeof(num_coeffs));
    
    // Read data_size
    stream.read(reinterpret_cast<char*>(&data_size), sizeof(data_size));
    
    // Read reserved field
    uint32_t reserved;
    stream.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
    
    if (!stream.good()) {
        throw std::runtime_error("Failed to read file header");
    }
}

