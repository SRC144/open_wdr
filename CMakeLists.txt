cmake_minimum_required(VERSION 3.15)
project(wdr_coder)

# Set C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /EHsc)
else()
    add_compile_options(-Wall -Wextra -pedantic -O3)
endif()

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find NumPy using Python (since CMake's find_package doesn't always work)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; import os; print(os.path.dirname(numpy.__file__))"
    OUTPUT_VARIABLE NUMPY_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)

if(NUMPY_INCLUDE_DIR)
    message(STATUS "Found NumPy include directory: ${NUMPY_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "NumPy not found. Please install it with: pip install numpy")
endif()

# Include directories
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${NUMPY_INCLUDE_DIR})

# Fetch pybind11
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Source files
set(SOURCES
    src/bit_stream.cpp
    src/adaptive_model.cpp
    src/arithmetic_coder.cpp
    src/wdr_compressor.cpp
    src/bindings.cpp
)

set(HEADERS
    src/bit_stream.hpp
    src/adaptive_model.hpp
    src/arithmetic_coder.hpp
    src/wdr_compressor.hpp
    src/wdr_file_format.hpp
)

# Python module
pybind11_add_module(wdr_coder ${SOURCES} ${HEADERS})

# Include directories for the module
target_include_directories(wdr_coder PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link libraries
target_link_libraries(wdr_coder PRIVATE
    ${Python3_LIBRARIES}
)

# Set output directory
set_target_properties(wdr_coder PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Enable testing
enable_testing()

# Find Google Test
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# Create a library for WDR code (without pybind11) for testing
add_library(wdr_core STATIC
    src/bit_stream.cpp
    src/adaptive_model.cpp
    src/arithmetic_coder.cpp
    src/wdr_compressor.cpp
)

target_include_directories(wdr_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Test executables will be added in tests/test_cpp/CMakeLists.txt
add_subdirectory(tests/test_cpp)
