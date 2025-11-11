#ifndef WDR_DEBUG_HPP
#define WDR_DEBUG_HPP

/**
 * @file wdr_debug.hpp
 * @brief Debug logging macro for WDR compressor
 * 
 * This header provides compile-time debug logging support.
 * To enable debug logging, define WDR_DEBUG before including this header:
 *   #define WDR_DEBUG
 *   #include "wdr_debug.hpp"
 */

#ifdef WDR_DEBUG
#include <iostream>
#include <sstream>

/**
 * @brief Macro for debug logging
 * 
 * Usage: WDR_DEBUG_LOG("message " << variable);
 * 
 * When WDR_DEBUG is defined, this macro outputs to stderr.
 * When WDR_DEBUG is not defined, this macro compiles to nothing.
 */
#define WDR_DEBUG_LOG(msg) do { \
    std::ostringstream _oss; \
    _oss << "[WDR_DEBUG] " << msg; \
    std::cerr << _oss.str() << std::endl; \
} while(0)

#else
#define WDR_DEBUG_LOG(msg) ((void)0)
#endif

#endif // WDR_DEBUG_HPP

