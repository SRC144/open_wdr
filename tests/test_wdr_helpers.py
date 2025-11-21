"""
Unit tests for wdr.utils.helpers
"""

import numpy as np
import pytest
import tempfile
import os
import tifffile
from PIL import Image
from wdr.utils import helpers as hlp

# --- Fixtures ---

@pytest.fixture
def random_image():
    """Returns a random float64 grayscale image (512x512)."""
    return np.random.rand(512, 512) * 255

@pytest.fixture
def odd_image():
    """Returns an image with odd dimensions (non-power-of-2, non-tile-aligned)."""
    return np.random.rand(600, 300) * 255

# --- I/O Tests ---

def test_save_and_load_png(random_image):
    """Test saving and loading a standard PNG (PIL path)."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
    
    try:
        # Save
        hlp.save_image(temp_path, random_image)
        assert os.path.exists(temp_path)
        
        # Load
        loaded = hlp.load_image(temp_path)
        
        # Verify
        assert loaded.shape == random_image.shape
        # Saved as uint8, so we allow rounding errors up to 1.0
        assert np.allclose(loaded, np.clip(random_image, 0, 255).astype(np.uint8), atol=1.0)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_load_tiff_memmap(random_image):
    """Test loading a TIFF file (Should trigger tifffile.memmap path)."""
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as f:
        temp_path = f.name
        
    try:
        # Create a real TIFF file using tifffile directly
        data = random_image.astype(np.uint8)
        tifffile.imwrite(temp_path, data)
        
        # Test helper load
        loaded = hlp.load_image(temp_path)
        
        # Verify properties
        assert isinstance(loaded, (np.ndarray, np.memmap))
        assert loaded.shape == (512, 512)
        assert np.allclose(loaded, data)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

def test_load_image_strict_check():
    """Test that load_image raises ValueError for RGB images."""
    # Create an RGB image
    rgb_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_path = f.name
        Image.fromarray(rgb_img).save(temp_path)
        
    try:
        # Should raise ValueError because WDR core requires 2D input
        with pytest.raises(ValueError, match="WDR Library Error"):
            hlp.load_image(temp_path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

# --- Tiling Tests (New) ---

def test_yield_tiles_exact_fit():
    """Test tiling on an image that is a perfect multiple of tile size."""
    img = np.random.rand(1024, 1024) # 2x2 grid of 512 tiles
    tile_size = 512
    
    tiles = list(hlp.yield_tiles(img, tile_size))
    
    assert len(tiles) == 4
    for tile in tiles:
        assert tile.shape == (512, 512)

def test_yield_tiles_padding(odd_image):
    """Test tiling on irregular image (600x300) with tile size 512."""
    # Grid should be:
    # Height 600 / 512 -> ceil(1.17) -> 2 rows
    # Width 300 / 512  -> ceil(0.58) -> 1 col
    # Total 2 tiles
    tile_size = 512
    tiles = list(hlp.yield_tiles(odd_image, tile_size))
    
    assert len(tiles) == 2
    
    # Verify padding
    # Tile 0: Top-Left (0:512, 0:300) -> Padded to (512, 512)
    # Tile 1: Bottom-Left (512:600, 0:300) -> Padded to (512, 512)
    for tile in tiles:
        assert tile.shape == (512, 512)
        
    # Verify content of first tile (top-left corner should match)
    # The logic uses edge padding, so exact match is only valid for the valid region
    # Tile 0 valid region is [0:512, 0:300]
    assert np.allclose(tiles[0][:512, :300], odd_image[:512, :300])

# --- Wavelet & Coeffs Tests ---

def test_dwt_idwt_round_trip(random_image):
    """Test DWT -> IDWT loop."""
    coeffs = hlp.do_dwt(random_image, scales=2, wavelet='bior4.4')
    recon = hlp.do_idwt(coeffs, wavelet='bior4.4')
    
    # Biorthogonal wavelets provide near-perfect reconstruction
    assert np.allclose(random_image, recon, atol=1e-5)

def test_flatten_unflatten_structure(random_image):
    """Test flattening coefficients and verifying metadata keys."""
    coeffs = hlp.do_dwt(random_image, scales=2, wavelet='bior4.4')
    flat, meta = hlp.flatten_coeffs(coeffs)
    
    assert flat.ndim == 1
    assert 'subbands' in meta
    
    # Check new metadata keys
    subband_info = meta['subbands'][0]
    assert 'shape' in subband_info
    assert 'start_idx' in subband_info
    assert 'end_idx' in subband_info
    assert 'scan_order' in subband_info

def test_flatten_unflatten_round_trip(random_image):
    """Test DWT -> Flatten -> Unflatten -> IDWT pipeline."""
    coeffs = hlp.do_dwt(random_image, scales=2, wavelet='bior4.4')
    
    # Flatten
    flat, meta = hlp.flatten_coeffs(coeffs)
    
    # Unflatten
    coeffs_recon = hlp.unflatten_coeffs(flat, meta)
    
    # Verify length
    assert len(coeffs) == len(coeffs_recon)
    
    # Verify values
    for c_orig, c_recon in zip(coeffs, coeffs_recon):
        if isinstance(c_orig, tuple):
            for b_orig, b_recon in zip(c_orig, c_recon):
                np.testing.assert_array_equal(b_orig, b_recon)
        else:
            np.testing.assert_array_equal(c_orig, c_recon)

# --- Helper Logic Tests ---

def test_quantize_dequantize():
    """Test uniform quantization logic."""
    data = np.array([10.0, 20.0, 30.5, -10.2])
    step = 2.0
    
    q, s = hlp.quantize_coeffs(data, step)
    
    # 30.5 / 2 = 15.25 -> round 15.0 -> * 2 = 30.0
    expected = np.array([10.0, 20.0, 30.0, -10.0])
    
    np.testing.assert_allclose(q, expected)
    assert s == step
    
    # Dequantize (Identity for uniform)
    dq = hlp.dequantize_coeffs(q, s)
    np.testing.assert_array_equal(q, dq)

def test_scan_for_max_coefficient(random_image):
    """Test the global scanning helper."""
    # Create a spike
    random_image[50, 50] = 10000.0
    
    # Run scan
    max_val = hlp.scan_for_max_coefficient(random_image, tile_size=256, scales=2, wavelet='bior4.4')
    
    # The max coeff in wavelet domain usually correlates with max pixel value,
    # but won't be exactly 10000. It should be large though.
    assert isinstance(max_val, float)
    assert max_val > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])