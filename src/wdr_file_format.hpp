#ifndef WDR_FILE_FORMAT_HPP
#define WDR_FILE_FORMAT_HPP

/**
 * @file wdr_file_format.hpp
 * @brief WDR file format definitions
 * 
 * This module defines the binary file format for .wdr compressed files.
 * The file format consists of a fixed-size header followed by variable-length
 * compressed data.
 * 
 * File structure:
 * - Header (40 bytes): Contains metadata about the compressed data
 * - Compressed data (variable length): Arithmetic-coded bitstream
 * 
 * The header stores the initial threshold, number of passes, number of
 * coefficients, and size of compressed data. This information is required
 * for decompression.
 */

#include <cstdint>

/**
 * @brief WDR file format constants and definitions
 * 
 * This namespace contains constants and definitions for the WDR file format,
 * including magic numbers, version numbers, and header structure information.
 */

namespace WDRFormat {
    // Magic number: "WDR\0"
    static constexpr uint32_t MAGIC = 0x57445200;
    
    // File format version
    static constexpr uint32_t VERSION = 1;
    
    // Header size in bytes
    static constexpr size_t HEADER_SIZE = 40;
    
    // Scale factor for initial_T (store as integer)
    static constexpr double T_SCALE = 65536.0;  // 2^16
}

#endif // WDR_FILE_FORMAT_HPP

