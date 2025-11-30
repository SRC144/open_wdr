#include "cwdr_processor.hpp"
#include <stdexcept>

// ============================================================================
// Constants (From Paper)
// ============================================================================

// RGB to YUV matrix (Based on formula 7/324 in paper)
// Y = 0.299R + 0.587G + 0.114B
// U = -0.14713R - 0.28886G + 0.436B
// V = 0.615R - 0.51499G - 0.10001B
// NOTE: The paper's U/V coefficients might be optimized for a specific range;
// we use the common standard implementation matching the structure.

// YUV to RGB matrix (Based on formula 8/353 in paper)
// R = 1.164Y + 0U + 1.596V
// G = 1.164Y - 0.392U - 0.813V
// B = 1.164Y + 2.017U + 0V

// ============================================================================
// Constructor
// ============================================================================

CWDRProcessor::CWDRProcessor(int num_passes) 
    : y_compressor_(num_passes), 
      u_compressor_(num_passes), 
      v_compressor_(num_passes), 
      num_passes_(num_passes) {}

// ============================================================================
// Color Transformations (Static Helpers)
// ============================================================================

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
CWDRProcessor::rgb_to_yuv(const std::vector<double>& rgb_data) {
    if (rgb_data.size() % 3 != 0) {
        throw std::invalid_argument("RGB data size must be a multiple of 3.");
    }

    size_t num_pixels = rgb_data.size() / 3;
    std::vector<double> y_data(num_pixels);
    std::vector<double> u_data(num_pixels);
    std::vector<double> v_data(num_pixels);

    for (size_t i = 0; i < num_pixels; ++i) {
        double R = rgb_data[i * 3 + 0];
        double G = rgb_data[i * 3 + 1];
        double B = rgb_data[i * 3 + 2];

        // Formula 7/324 (Standard conversion used in the CWDR context)
        y_data[i] = 0.299 * R + 0.587 * G + 0.114 * B;
        u_data[i] = -0.14713 * R - 0.28886 * G + 0.436 * B;
        v_data[i] = 0.615 * R - 0.51499 * G - 0.10001 * B;
    }
    return std::make_tuple(std::move(y_data), std::move(u_data), std::move(v_data));
}

std::vector<double> CWDRProcessor::yuv_to_rgb(const std::vector<double>& y_data,
                                           const std::vector<double>& u_data,
                                           const std::vector<double>& v_data) {
    size_t num_pixels = y_data.size();
    if (u_data.size() != num_pixels || v_data.size() != num_pixels) {
        throw std::invalid_argument("YUV channel sizes must be equal.");
    }
    
    std::vector<double> rgb_data(num_pixels * 3);

    for (size_t i = 0; i < num_pixels; ++i) {
        double Y = y_data[i];
        double U = u_data[i];
        double V = v_data[i];

        // Formula 8/353 (Inverse conversion)
        // Note: The paper's matrix implies Y is shifted/scaled from 0-255 range.
        // We use the common inverse conversion matching the paper's structure for the transformation matrix.
        rgb_data[i * 3 + 0] = 1.164 * Y + 1.596 * V;          // R
        rgb_data[i * 3 + 1] = 1.164 * Y - 0.392 * U - 0.813 * V; // G
        rgb_data[i * 3 + 2] = 1.164 * Y + 2.017 * U;          // B
    }
    return rgb_data;
}

// ============================================================================
// CWDR Compression/Decompression
// ============================================================================

std::vector<uint8_t> CWDRProcessor::compress(const std::vector<double>& rgb_data, double initial_T) {
    // 1. Transform RGB to YUV and split into 3 streams [cite: 333]
    auto [y_stream, u_stream, v_stream] = rgb_to_yuv(rgb_data);

    // 2. Apply WDR to each stream 
    // The initial_T (Global Threshold) should ideally be calculated only from the Y channel
    // but the CWDR paper structure applies the WDR to each stream. We use the
    // initial_T provided externally (likely Y-derived).
    
    std::vector<uint8_t> y_compressed = y_compressor_.compress(y_stream, initial_T);
    std::vector<uint8_t> u_compressed = u_compressor_.compress(u_stream, initial_T);
    std::vector<uint8_t> v_compressed = v_compressor_.compress(v_stream, initial_T);

    // 3. Concatenate the 3 compressed bitstreams [cite: 246]
    std::vector<uint8_t> final_stream;
    final_stream.reserve(y_compressed.size() + u_compressed.size() + v_compressed.size());

    // NOTE: In a real system, we'd embed the size of each stream, 
    // but here we just concatenate and assume the decoder knows the sizes.
    // For this example, we'll embed the size of Y and U streams at the start.
    
    // Simplification for the decoder: embed the size of the first two streams (Y and U)
    uint64_t y_size = y_compressed.size();
    uint64_t u_size = u_compressed.size();

    // Use simple memcpy or loop to embed 8 bytes for size of Y stream
    const uint8_t* y_size_bytes = reinterpret_cast<const uint8_t*>(&y_size);
    final_stream.insert(final_stream.end(), y_size_bytes, y_size_bytes + sizeof(uint64_t));

    // Embed 8 bytes for size of U stream
    const uint8_t* u_size_bytes = reinterpret_cast<const uint8_t*>(&u_size);
    final_stream.insert(final_stream.end(), u_size_bytes, u_size_bytes + sizeof(uint64_t));
    
    // Concatenate actual compressed data
    final_stream.insert(final_stream.end(), y_compressed.begin(), y_compressed.end());
    final_stream.insert(final_stream.end(), u_compressed.begin(), u_compressed.end());
    final_stream.insert(final_stream.end(), v_compressed.begin(), v_compressed.end());

    return final_stream;
}

std::vector<double> CWDRProcessor::decompress(const std::vector<uint8_t>& compressed_data, 
                                           double initial_T, 
                                           uint64_t num_coeffs) {
    if (compressed_data.size() < 2 * sizeof(uint64_t)) {
        throw std::runtime_error("Compressed stream too short to read size metadata.");
    }
    
    // 1. Read sizes (assuming the same 8-byte metadata structure from compress)
    const uint8_t* data_ptr = compressed_data.data();

    // Read size of Y stream
    uint64_t y_size = *reinterpret_cast<const uint64_t*>(data_ptr);
    data_ptr += sizeof(uint64_t);

    // Read size of U stream
    uint64_t u_size = *reinterpret_cast<const uint64_t*>(data_ptr);
    data_ptr += sizeof(uint64_t);

    size_t header_size = 2 * sizeof(uint64_t);
    size_t v_size = compressed_data.size() - header_size - y_size - u_size;

    // 2. Separate streams
    std::vector<uint8_t> y_stream(data_ptr, data_ptr + y_size);
    data_ptr += y_size;

    std::vector<uint8_t> u_stream(data_ptr, data_ptr + u_size);
    data_ptr += u_size;

    std::vector<uint8_t> v_stream(data_ptr, data_ptr + v_size);

    // 3. Decompress each stream
    std::vector<double> y_decompressed = y_compressor_.decompress(y_stream, initial_T, num_coeffs);
    std::vector<double> u_decompressed = u_compressor_.decompress(u_stream, initial_T, num_coeffs);
    std::vector<double> v_decompressed = v_compressor_.decompress(v_stream, initial_T, num_coeffs);

    // 4. Convert YUV back to RGB and combine [cite: 352, 353, 350]
    return yuv_to_rgb(y_decompressed, u_decompressed, v_decompressed);
}