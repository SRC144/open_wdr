"""
Unit tests for wdr.utils.helpers
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from wdr.utils import helpers as hlp

pytestmark = [
    pytest.mark.filterwarnings("ignore:Level value of .*boundary effects.:UserWarning"),
]


def test_load_image():
    """Test load_image function."""
    # Create a test image
    test_img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        hlp.save_image(temp_path, test_img)
    
    try:
        # Load the image
        loaded_img = hlp.load_image(temp_path)
        
        # Check properties
        assert loaded_img.shape == test_img.shape
        assert loaded_img.dtype == np.float64
        assert np.allclose(loaded_img, test_img.astype(np.float64), atol=1.0)
    finally:
        # Clean up
        os.unlink(temp_path)


def test_save_image():
    """Test save_image function."""
    # Create a test image
    test_img = np.random.rand(50, 50) * 255
    test_img = test_img.astype(np.float64)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save the image
        hlp.save_image(temp_path, test_img)
        
        # Check that file exists
        assert os.path.exists(temp_path)
        
        # Load and verify
        loaded_img = hlp.load_image(temp_path)
        assert loaded_img.shape == test_img.shape
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_do_dwt():
    """Test do_dwt function."""
    # Create a test image
    test_img = np.random.rand(64, 64) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=3, wavelet='bior4.4')
    
    # Check that coefficients are returned
    assert coeffs is not None
    # pywt.wavedec2 returns a list, not a tuple
    assert isinstance(coeffs, (tuple, list))
    assert len(coeffs) == 4  # cA, (cH, cV, cD) for 3 levels


def test_do_dwt_idwt_round_trip():
    """Test that DWT and IDWT are inverse operations."""
    # Create a test image
    test_img = np.random.rand(64, 64) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=3, wavelet='bior4.4')
    
    # Perform IDWT
    reconstructed_img = hlp.do_idwt(coeffs, wavelet='bior4.4')
    
    # Check that shapes match (may have slight differences due to padding)
    assert reconstructed_img.shape == test_img.shape
    
    # Check that images are close (within tolerance)
    # Note: Perfect reconstruction may not be possible due to floating-point errors
    assert np.allclose(reconstructed_img, test_img, atol=1e-6)


def test_flatten_coeffs():
    """Test flatten_coeffs function."""
    # Create a test image
    test_img = np.random.rand(32, 32) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Check properties
    assert flat_coeffs.ndim == 1
    assert len(shape_metadata['subbands']) > 0
    assert 'subbands' in shape_metadata


def test_flatten_unflatten_round_trip():
    """Test that flatten_coeffs and unflatten_coeffs are inverse operations."""
    # Create a test image
    test_img = np.random.rand(32, 32) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Unflatten coefficients
    unflattened_coeffs = hlp.unflatten_coeffs(flat_coeffs, shape_metadata)
    
    # Check that coefficient structures match
    assert len(unflattened_coeffs) == len(coeffs)
    
    # Check approximation coefficients
    np.testing.assert_array_equal(unflattened_coeffs[0], coeffs[0])
    
    # Check detail coefficients - verify every single subband is identical
    for i in range(1, len(coeffs)):
        assert len(unflattened_coeffs[i]) == len(coeffs[i])
        for j in range(len(coeffs[i])):
            np.testing.assert_array_equal(unflattened_coeffs[i][j], coeffs[i][j])

# Test: Perfect Round-Trip Flattening with Known Structure
def test_flatten_unflatten_known_structure():
    """Test flatten/unflatten with a known, simple coefficient structure."""
    import pywt
    
    # Create a known coefficient structure: [LL, (HL, LH, HH)]
    # Simple 4x4 arrays for each subband
    ll = np.array([[1.0, 2.0], [3.0, 4.0]])
    hl = np.array([[5.0, 6.0], [7.0, 8.0]])
    lh = np.array([[9.0, 10.0], [11.0, 12.0]])
    hh = np.array([[13.0, 14.0], [15.0, 16.0]])
    
    coeffs = [ll, (hl, lh, hh)]
    
    # Flatten
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Unflatten
    unflattened_coeffs = hlp.unflatten_coeffs(flat_coeffs, shape_metadata)
    
    # Verify every subband is identical
    np.testing.assert_array_equal(unflattened_coeffs[0], ll)
    np.testing.assert_array_equal(unflattened_coeffs[1][0], hl)
    np.testing.assert_array_equal(unflattened_coeffs[1][1], lh)
    np.testing.assert_array_equal(unflattened_coeffs[1][2], hh)

# Test: Flatten with Different Wavelets
def test_flatten_unflatten_different_wavelets():
    """Test flatten/unflatten with different wavelets."""
    test_img = np.random.rand(64, 64) * 255
    wavelets = ['bior4.4', 'haar', 'db4', 'coif2']
    
    for wavelet in wavelets:
        coeffs = hlp.do_dwt(test_img, scales=2, wavelet=wavelet)
        flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
        unflattened_coeffs = hlp.unflatten_coeffs(flat_coeffs, shape_metadata)
        
        # Verify structure
        assert len(unflattened_coeffs) == len(coeffs)
        np.testing.assert_array_equal(unflattened_coeffs[0], coeffs[0])
        
        for i in range(1, len(coeffs)):
            for j in range(len(coeffs[i])):
                np.testing.assert_array_equal(unflattened_coeffs[i][j], coeffs[i][j])

# Test: Flatten with Different Scales
def test_flatten_unflatten_different_scales():
    """Test flatten/unflatten with different numbers of scales."""
    test_img = np.random.rand(64, 64) * 255
    
    for scales in [1, 2, 3, 4]:
        coeffs = hlp.do_dwt(test_img, scales=scales, wavelet='bior4.4')
        flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
        unflattened_coeffs = hlp.unflatten_coeffs(flat_coeffs, shape_metadata)
        
        # Verify structure
        assert len(unflattened_coeffs) == len(coeffs)
        np.testing.assert_array_equal(unflattened_coeffs[0], coeffs[0])
        
        for i in range(1, len(coeffs)):
            for j in range(len(coeffs[i])):
                np.testing.assert_array_equal(unflattened_coeffs[i][j], coeffs[i][j])


def test_flatten_unflatten_idwt_round_trip():
    """Test full round trip: DWT -> flatten -> unflatten -> IDWT."""
    # Create a test image
    test_img = np.random.rand(32, 32) * 255
    
    # Perform DWT
    coeffs = hlp.do_dwt(test_img, scales=2, wavelet='bior4.4')
    
    # Flatten coefficients
    flat_coeffs, shape_metadata = hlp.flatten_coeffs(coeffs)
    
    # Unflatten coefficients
    unflattened_coeffs = hlp.unflatten_coeffs(flat_coeffs, shape_metadata)
    
    # Perform IDWT
    reconstructed_img = hlp.do_idwt(unflattened_coeffs, wavelet='bior4.4')
    
    # Check that images are close
    assert np.allclose(reconstructed_img, test_img, atol=1e-6)


def test_empty_image():
    """Test error handling for empty image."""
    # Create an empty image
    empty_img = np.array([])
    
    # Should raise an error
    with pytest.raises(ValueError):
        hlp.do_dwt(empty_img, scales=1)


def test_invalid_scales():
    """Test error handling for invalid scales."""
    # Create a test image
    test_img = np.random.rand(32, 32) * 255
    
    # Should raise an error for invalid scales
    with pytest.raises(ValueError):
        hlp.do_dwt(test_img, scales=0)
    
    with pytest.raises(ValueError):
        hlp.do_dwt(test_img, scales=-1)


def test_quantize_coeffs():
    """Test quantize_coeffs function."""
    # Create test coefficients
    coeffs = np.array([1.23, 4.56, 7.89, -2.34, -5.67], dtype=np.float64)
    step_size = 0.5
    
    # Quantize
    quantized, step = hlp.quantize_coeffs(coeffs, step_size)
    
    # Check properties
    assert quantized.shape == coeffs.shape
    assert quantized.dtype == coeffs.dtype
    assert step == step_size
    
    # Check quantization: values should be multiples of step_size
    for val in quantized:
        # Round to nearest multiple of step_size
        expected = np.round(val / step_size) * step_size
        assert np.abs(val - expected) < 1e-10
    
    # Check that quantization error is bounded
    error = np.abs(coeffs - quantized)
    assert np.all(error <= step_size / 2.0 + 1e-10)


def test_quantize_coeffs_edge_cases():
    """Test quantize_coeffs with edge cases."""
    # Test with zeros
    coeffs = np.zeros(10, dtype=np.float64)
    quantized, step = hlp.quantize_coeffs(coeffs, 0.5)
    np.testing.assert_array_equal(quantized, coeffs)
    
    # Test with negative values
    coeffs = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    quantized, _ = hlp.quantize_coeffs(coeffs, 1.0)
    expected = np.array([-10.0, -5.0, 0.0, 5.0, 10.0], dtype=np.float64)
    np.testing.assert_array_equal(quantized, expected)
    
    # Test with invalid step size
    with pytest.raises(ValueError):
        hlp.quantize_coeffs(coeffs, 0.0)
    
    with pytest.raises(ValueError):
        hlp.quantize_coeffs(coeffs, -1.0)


def test_dequantize_coeffs():
    """Test dequantize_coeffs function."""
    # Create quantized coefficients
    quantized = np.array([1.0, 4.5, 8.0, -2.0, -5.5], dtype=np.float64)
    step_size = 0.5
    
    # Dequantize (should be identity for uniform quantization)
    dequantized = hlp.dequantize_coeffs(quantized, step_size)
    
    # Check that dequantization is identity
    np.testing.assert_array_equal(dequantized, quantized)
    
    # Test with invalid step size
    with pytest.raises(ValueError):
        hlp.dequantize_coeffs(quantized, 0.0)
    
    with pytest.raises(ValueError):
        hlp.dequantize_coeffs(quantized, -1.0)


def test_quantize_dequantize_round_trip():
    """Test that quantize and dequantize are inverse operations."""
    # Create test coefficients
    coeffs = np.array([1.23, 4.56, 7.89, -2.34, -5.67], dtype=np.float64)
    step_size = 0.5
    
    # Quantize
    quantized, step = hlp.quantize_coeffs(coeffs, step_size)
    
    # Dequantize
    dequantized = hlp.dequantize_coeffs(quantized, step)
    
    # For uniform quantization, dequantized should equal quantized
    np.testing.assert_array_equal(dequantized, quantized)
    
    # Quantized values should be close to original (within step_size/2)
    error = np.abs(coeffs - quantized)
    assert np.all(error <= step_size / 2.0 + 1e-10)


def test_calculate_quantization_step():
    """Test calculate_quantization_step function."""
    # Create test coefficients
    coeffs = np.array([100.0, -42.0, 10.0, 0.0, 3.0], dtype=np.float64)
    
    # Test threshold_based method with compression_focused=True
    step = hlp.calculate_quantization_step(coeffs, num_passes=26, method='threshold_based', compression_focused=True)
    assert step > 0
    assert step < 100.0  # Should be reasonable
    
    # Test threshold_based method with compression_focused=False
    step_fine = hlp.calculate_quantization_step(coeffs, num_passes=26, method='threshold_based', compression_focused=False)
    assert step_fine > 0
    assert step_fine < step  # Fine quantization should have smaller step
    
    # Test fixed_precision method
    step_fixed = hlp.calculate_quantization_step(coeffs, num_passes=26, method='fixed_precision')
    assert step_fixed > 0
    
    # Test with all-zero coefficients
    zero_coeffs = np.zeros(10, dtype=np.float64)
    step_zero = hlp.calculate_quantization_step(zero_coeffs, num_passes=26, method='threshold_based')
    assert step_zero == 1.0  # Default step for all-zero
    
    # Test with invalid method
    with pytest.raises(ValueError):
        hlp.calculate_quantization_step(coeffs, num_passes=26, method='invalid_method')
    
    # Test with adaptive method (not implemented)
    with pytest.raises(NotImplementedError):
        hlp.calculate_quantization_step(coeffs, num_passes=26, method='adaptive')


def test_calculate_quantization_step_compression_focused():
    """Test that compression_focused creates larger step sizes."""
    # Create test coefficients with large range
    coeffs = np.random.rand(1000) * 1000.0
    coeffs = coeffs.astype(np.float64)
    
    # Calculate steps
    step_compression = hlp.calculate_quantization_step(coeffs, num_passes=26, method='threshold_based', compression_focused=True)
    step_fine = hlp.calculate_quantization_step(coeffs, num_passes=26, method='threshold_based', compression_focused=False)
    
    # Compression-focused should have larger step
    assert step_compression > step_fine
    
    # Test quantization with both steps
    quantized_compression, _ = hlp.quantize_coeffs(coeffs, step_compression)
    quantized_fine, _ = hlp.quantize_coeffs(coeffs, step_fine)
    
    # Compression-focused should have fewer unique values
    unique_compression = len(np.unique(quantized_compression))
    unique_fine = len(np.unique(quantized_fine))
    assert unique_compression <= unique_fine


def test_quantization_creates_redundancy():
    """Test that quantization creates redundancy (repeated values)."""
    # Create test coefficients with many unique values
    coeffs = np.random.rand(1000) * 100.0
    coeffs = coeffs.astype(np.float64)
    
    # Count unique values before quantization
    unique_before = len(np.unique(coeffs))
    assert unique_before == 1000  # All values should be unique
    
    # Quantize with reasonable step size
    step_size = 1.0
    quantized, _ = hlp.quantize_coeffs(coeffs, step_size)
    
    # Count unique values after quantization
    unique_after = len(np.unique(quantized))
    
    # After quantization, we should have fewer unique values
    assert unique_after < unique_before
    
    # Reduction should be significant for reasonable step sizes
    reduction_ratio = (unique_before - unique_after) / unique_before
    assert reduction_ratio > 0.1  # At least 10% reduction


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

