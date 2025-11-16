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
#include <cstddef>

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

    // ------------------------------------------------------------------
    // Tiled/streaming format (".wdrt") additions
    // ------------------------------------------------------------------

    // Magic number: "WDRT"
    static constexpr uint32_t TILED_MAGIC = 0x54445257;

    // File format version for tiled variant
    static constexpr uint32_t TILED_VERSION = 1;

    // Maximum bytes reserved for storing the wavelet name in the header.
    static constexpr size_t WAVELET_NAME_BYTES = 32;

    /**
     * Flags that annotate optional data present in the tiled header.
     */
    enum class TiledHeaderFlags : uint32_t {
        NONE = 0,
        HAS_QUANT_STEP = 1u << 0,   // quant_step field contains valid value
        RESERVED = 1u << 31
    };

    inline TiledHeaderFlags operator|(TiledHeaderFlags lhs, TiledHeaderFlags rhs) {
        return static_cast<TiledHeaderFlags>(static_cast<uint32_t>(lhs) |
                                             static_cast<uint32_t>(rhs));
    }

    inline TiledHeaderFlags &operator|=(TiledHeaderFlags &lhs, TiledHeaderFlags rhs) {
        lhs = lhs | rhs;
        return lhs;
    }

    /**
     * @brief Header describing a tiled WDR stream.
     *
     * Layout is fixed-width and little-endian. Strings are stored inline using
     * a fixed 32-byte buffer to avoid dynamic parsing.
     */
    struct TiledFileHeader {
        uint32_t magic = TILED_MAGIC;                 // "WDRT"
        uint32_t version = TILED_VERSION;             // format version
        uint32_t flags = static_cast<uint32_t>(TiledHeaderFlags::NONE);
        uint32_t num_passes = 0;                      // bit-plane passes
        uint32_t num_scales = 0;                      // DWT scales
        uint32_t tile_width = 0;                      // planned tile width
        uint32_t tile_height = 0;                     // planned tile height
        uint32_t image_width = 0;                     // full image width
        uint32_t image_height = 0;                    // full image height
        double global_initial_T = 0.0;                // shared T
        double quant_step = 0.0;                      // optional quant step
        uint64_t total_tiles = 0;                     // count of tile chunks
        char wavelet_name[WAVELET_NAME_BYTES] = {};   // zero-padded wavelet
    };

    static constexpr size_t TILED_HEADER_SIZE = sizeof(TiledFileHeader);

    /**
     * @brief Metadata prefixing every tile chunk in a tiled WDR stream.
     *
     * Each chunk is length-prefixed (chunk_size_bytes) followed by the raw
     * arithmetic-coded payload returned by the C++ worker.
     */
    struct TileChunkHeader {
        uint64_t tile_index = 0;      // sequential tile ID (scanline order)
        uint32_t origin_x = 0;        // pixel x-offset of the tile
        uint32_t origin_y = 0;        // pixel y-offset of the tile
        uint32_t coeff_count = 0;     // flattened coefficient count
        uint32_t tile_pixel_width = 0;
        uint32_t tile_pixel_height = 0;
        uint64_t chunk_size_bytes = 0; // number of bytes that follow
    };

    static constexpr size_t TILE_CHUNK_HEADER_SIZE = sizeof(TileChunkHeader);
}

#endif // WDR_FILE_FORMAT_HPP

