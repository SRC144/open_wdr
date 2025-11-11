/**
 * @file bindings.cpp
 * @brief Python bindings for WDR compression module
 * 
 * This module provides Python bindings for the WDR compression functionality
 * using pybind11. It exposes the WDRCompressor class to Python, allowing
 * compression and decompression of NumPy arrays.
 * 
 * The bindings handle conversion between NumPy arrays and std::vector<double>,
 * and provide proper memory management through pybind11's automatic conversion.
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "wdr_compressor.hpp"
#include <stdexcept>
#include <vector>

namespace py = pybind11;

/**
 * @brief Compress coefficients to a .wdr file
 * 
 * This function compresses a 1D NumPy array of coefficients to a .wdr file
 * using the WDR compression algorithm.
 * 
 * @param coeffs NumPy array of coefficients (1D, float64)
 * @param output_filepath Output file path for the compressed .wdr file
 * @param num_passes Number of bit-plane passes (default: 26 for high precision)
 * 
 * @throws std::invalid_argument if the array is not 1D, is empty, or num_passes is invalid
 * @throws std::runtime_error if compression fails
 * 
 * @note The default of 26 passes provides high precision (error < 1e-6).
 *       For lower precision, fewer passes can be used.
 */
void compress(py::array_t<double> coeffs, const std::string& output_filepath, int num_passes = 26) {
    // Check array properties
    if (coeffs.ndim() != 1) {
        throw std::invalid_argument("Coefficients array must be 1D");
    }
    
    if (coeffs.size() == 0) {
        throw std::invalid_argument("Coefficients array must not be empty");
    }
    
    if (num_passes <= 0) {
        throw std::invalid_argument("Number of passes must be positive");
    }
    
    // Get pointer to data
    auto buf = coeffs.request();
    double* ptr = static_cast<double*>(buf.ptr);
    
    // Convert to std::vector
    std::vector<double> coeffs_vec(ptr, ptr + coeffs.size());
    
    // Create compressor and compress
    WDRCompressor compressor(num_passes);
    try {
        compressor.compress(coeffs_vec, output_filepath);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Compression failed: ") + e.what());
    }
}

/**
 * @brief Decompress coefficients from a .wdr file
 * 
 * This function decompresses a .wdr file and returns the decompressed
 * coefficients as a 1D NumPy array.
 * 
 * @param input_filepath Input file path for the compressed .wdr file
 * @return NumPy array of decompressed coefficients (1D, float64)
 * 
 * @throws std::runtime_error if decompression fails or the file is invalid
 * 
 * @note The returned array uses pybind11's automatic memory management,
 *       ensuring proper ownership and lifetime management.
 */
py::array_t<double> decompress(const std::string& input_filepath) {
    // Create compressor and decompress
    WDRCompressor compressor;
    std::vector<double> coeffs_vec;
    
    try {
        coeffs_vec = compressor.decompress(input_filepath);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Decompression failed: ") + e.what());
    }
    
    // Convert to NumPy array using pybind11's automatic conversion
    // This ensures proper memory management and ownership
    py::array_t<double> result = py::cast(coeffs_vec);
    
    return result;
}

/**
 * @brief Python module initialization
 * 
 * This function initializes the Python module and exposes the compression
 * and decompression functions to Python.
 * 
 * Module name: wdr_coder
 * 
 * Exposed functions:
 * - compress(coeffs, output_filepath, num_passes=26): Compress coefficients to a .wdr file
 * - decompress(input_filepath): Decompress coefficients from a .wdr file
 */
PYBIND11_MODULE(wdr_coder, m) {
    m.doc() = "WDR Image Compression Coder - Python bindings for WDR compression algorithm";
    
    m.def("compress", &compress, 
          "Compress coefficients to a .wdr file.\n\n"
          "Args:\n"
          "    coeffs: NumPy array of coefficients (1D, float64)\n"
          "    output_filepath: Output file path for the compressed .wdr file\n"
          "    num_passes: Number of bit-plane passes (default: 26 for high precision)\n\n"
          "Returns:\n"
          "    None\n\n"
          "Raises:\n"
          "    ValueError: If the array is not 1D, is empty, or num_passes is invalid\n"
          "    RuntimeError: If compression fails",
          py::arg("coeffs"),
          py::arg("output_filepath"),
          py::arg("num_passes") = 26);
    
    m.def("decompress", &decompress,
          "Decompress coefficients from a .wdr file.\n\n"
          "Args:\n"
          "    input_filepath: Input file path for the compressed .wdr file\n\n"
          "Returns:\n"
          "    NumPy array of decompressed coefficients (1D, float64)\n\n"
          "Raises:\n"
          "    RuntimeError: If decompression fails or the file is invalid",
          py::arg("input_filepath"));
}

