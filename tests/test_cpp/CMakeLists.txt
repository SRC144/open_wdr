# Test CMakeLists.txt

# Mock arithmetic coder for testing
add_library(mock_arithmetic_coder STATIC
    mock_arithmetic_coder.cpp
)

target_include_directories(mock_arithmetic_coder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/src
)

# Test: Adaptive Model
add_executable(test_adaptive_model
    test_adaptive_model.cpp
)

target_link_libraries(test_adaptive_model
    wdr_core
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME AdaptiveModelTests COMMAND test_adaptive_model)

# Test: Bit Stream
add_executable(test_bit_stream
    test_bit_stream.cpp
)

target_link_libraries(test_bit_stream
    wdr_core
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME BitStreamTests COMMAND test_bit_stream)

# Test: Arithmetic Coder
add_executable(test_arithmetic_coder
    test_arithmetic_coder.cpp
)

target_link_libraries(test_arithmetic_coder
    wdr_core
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME ArithmeticCoderTests COMMAND test_arithmetic_coder)

# Test: Differential Coding
add_executable(test_differential_coding
    test_differential_coding.cpp
)

target_link_libraries(test_differential_coding
    wdr_core
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME DifferentialCodingTests COMMAND test_differential_coding)

# Test: WDR Compressor Passes
add_executable(test_wdr_compressor_passes
    test_wdr_compressor_passes.cpp
    mock_arithmetic_coder.cpp
)

target_link_libraries(test_wdr_compressor_passes
    wdr_core
    mock_arithmetic_coder
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME WDRCompressorPassesTests COMMAND test_wdr_compressor_passes)

# Test: WDR Compressor Round-Trip
add_executable(test_wdr_compressor_roundtrip
    test_wdr_compressor_roundtrip.cpp
)

target_link_libraries(test_wdr_compressor_roundtrip
    wdr_core
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME WDRCompressorRoundTripTests COMMAND test_wdr_compressor_roundtrip)
