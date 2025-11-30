#ifndef CWDR_PROCESSOR_HPP
#define CWDR_PROCESSOR_HPP

#include "wdr_compressor.hpp"
#include <vector>
#include <tuple>
#include <cmath>

/**
 * @brief CWDR Processor for Color Image Compression (RGB -> YUV -> 3x WDR -> RGB).
 * * Implements the color extension of WDR by transforming the image to YUV space
 * and compressing each channel independently using three WDRCompressor instances.
 * * @see @cite Zerva2023 for the CWDR method
 */
class CWDRProcessor {
public:
    explicit CWDRProcessor(int num_passes);

    /**
     * @brief Converts a flattened RGB array (R1, G1, B1, R2, G2, B2, ...) to 
     * three separate YUV channels.
     * @param rgb_data Flattened vector of pixel values (must be 3 * num_pixels long).
     * @return Tuple containing (Y_channel, U_channel, V_channel).
     */
    static std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> 
    rgb_to_yuv(const std::vector<double>& rgb_data);

    /**
     * @brief Converts three separate YUV channels back to a flattened RGB array.
     * @return std::vector<double> Flattened RGB data.
     */
    static std::vector<double> yuv_to_rgb(const std::vector<double>& y_data,
                                           const std::vector<double>& u_data,
                                           const std::vector<double>& v_data);
                                           
    /**
     * @brief Compresses all three channels and concatenates the bitstreams.
     * @param rgb_data Flattened RGB coefficients.
     * @param initial_T Global Threshold calculated from the Y channel's max coefficient.
     * @return std::vector<uint8_t> Concatenated compressed stream.
     */
    std::vector<uint8_t> compress(const std::vector<double>& rgb_data, double initial_T);

    /**
     * @brief Decompresses the concatenated bitstream and reconstructs the RGB coefficients.
     * @param compressed_data Concatenated bitstream.
     * @param initial_T Global Threshold.
     * @param num_coeffs Total number of coefficients per channel (num_pixels).
     * @return std::vector<double> Reconstructed flattened RGB coefficients.
     */
    std::vector<double> decompress(const std::vector<uint8_t>& compressed_data, 
                                   double initial_T, 
                                   uint64_t num_coeffs);

private:
    WDRCompressor y_compressor_; // Compressor for Luminance
    WDRCompressor u_compressor_; // Compressor for Chrominance U
    WDRCompressor v_compressor_; // Compressor for Chrominance V
    uint64_t num_passes_;
};

#endif // CWDR_PROCESSOR_HPP