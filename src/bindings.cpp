/**
 * @file bindings.cpp
 * @brief Python bindings for WDR compression module.
 */

#include "wdr_compressor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(coder, m) {
  m.doc() = "WDR Memory Coder";

  py::class_<WDRCompressor>(m, "WDRCompressor")
      .def(py::init<int>(), py::arg("num_passes") = 16)

      // --- COMPRESS ---
      //Numpy -> Vector works automatically
      .def("compress", &WDRCompressor::compress,
           "Compress a flattened coefficients array to a raw WDR byte vector using a global threshold.\n",
           py::arg("coeffs"), py::arg("global_T"))

      // --- DECOMPRESS ---
      // We use a lambda to accept 'py::bytes' and convert it to
      // 'std::vector<uint8_t>'
      .def(
          "decompress",
          [](WDRCompressor &self, py::bytes data, double global_T,
             uint64_t num_coeffs) {
            // 1. Convert Python bytes to C++ string (Zero-ish copy depending on
            // implementation)
            std::string s = data;

            // 2. Cast to vector<uint8_t>
            // This is safe because char and uint8_t are both 1 byte
            std::vector<uint8_t> vec(s.begin(), s.end());

            // 3. Call the actual C++ method
            return self.decompress(vec, global_T, num_coeffs);
          },
          "Decompress raw bytes back to coefficients.\n",
          py::arg("compressed_data"), py::arg("global_T"),
          py::arg("num_coeffs"));
}