"""
Unit tests for wdr.utils.metrics
"""

import numpy as np
import pytest
import tempfile
import os
import math
from wdr.utils import metrics as met

# --- StreamMetrics Tests ---

def test_stream_metrics_init():
    """Test initialization of StreamMetrics."""
    metrics = met.StreamMetrics(max_val=255.0)
    assert metrics.sse == 0.0
    assert metrics.total_pixels == 0
    assert metrics.max_val == 255.0

def test_stream_metrics_perfect_match():
    """Test comparison of identical tiles (should have infinite PSNR)."""
    metrics = met.StreamMetrics()
    
    tile = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    metrics.update(tile, tile)
    
    mse, psnr = metrics.get_results()
    assert mse == 0.0
    assert psnr == float('inf')

def test_stream_metrics_known_difference():
    """Test comparison with known errors."""
    metrics = met.StreamMetrics()
    
    # Create two 2x2 tiles
    # Tile A: [10, 10, 10, 10]
    # Tile B: [12, 10, 14, 10]
    # Diff:   [ 2,  0,  4,  0]
    # SqDiff: [ 4,  0, 16,  0]
    # SSE = 20
    # Pixels = 4
    # MSE = 5.0
    
    tile_a = np.full((2, 2), 10, dtype=np.uint8)
    tile_b = np.array([[12, 10], [14, 10]], dtype=np.uint8)
    
    metrics.update(tile_a, tile_b)
    
    mse, psnr = metrics.get_results()
    
    assert mse == 5.0
    
    expected_psnr = 20 * math.log10(255.0 / math.sqrt(5.0))
    assert math.isclose(psnr, expected_psnr, rel_tol=1e-9)

def test_stream_metrics_accumulation():
    """Test that metrics accumulate correctly over multiple updates."""
    metrics = met.StreamMetrics()
    
    # Update 1: Perfect match (10 pixels)
    tile1 = np.zeros((2, 5), dtype=np.uint8)
    metrics.update(tile1, tile1)
    
    # Update 2: All pixels off by 1 (10 pixels)
    # SSE += 10 * (1^2) = 10
    tile2_a = np.zeros((2, 5), dtype=np.uint8)
    tile2_b = np.ones((2, 5), dtype=np.uint8)
    metrics.update(tile2_a, tile2_b)
    
    # Total pixels: 20
    # Total SSE: 10
    # MSE: 0.5
    
    mse, _ = metrics.get_results()
    assert mse == 0.5

def test_stream_metrics_shape_mismatch():
    """Test that metrics handle slightly different tile shapes (e.g. padding)."""
    metrics = met.StreamMetrics()
    
    # Original: 100x100
    # Recon: 102x102 (padded)
    tile_orig = np.zeros((100, 100), dtype=np.uint8)
    tile_recon = np.zeros((102, 102), dtype=np.uint8)
    
    # Should run without error and use the intersection (100x100)
    metrics.update(tile_orig, tile_recon)
    
    assert metrics.total_pixels == 100 * 100
    mse, psnr = metrics.get_results()
    assert mse == 0.0

def test_stream_metrics_empty():
    """Test getting results with no updates."""
    metrics = met.StreamMetrics()
    mse, psnr = metrics.get_results()
    
    assert mse == 0.0
    assert psnr == float('inf')

def test_stream_metrics_float_overflow_protection():
    """Test that calculation doesn't overflow with uint8 inputs."""
    metrics = met.StreamMetrics()
    
    # 0 vs 255
    # diff = 255
    # diff^2 = 65025 (fits in int32/float, but overflows int8/uint8)
    tile_a = np.array([[0]], dtype=np.uint8)
    tile_b = np.array([[255]], dtype=np.uint8)
    
    metrics.update(tile_a, tile_b)
    
    assert metrics.sse == 65025.0

# --- Helper Functions Tests ---

def test_get_file_size():
    """Test file size helper."""
    with tempfile.NamedTemporaryFile(delete=False) as f:
        # Write 1024 bytes
        f.write(b'\x00' * 1024)
        temp_path = f.name
    
    try:
        size = met.get_file_size(temp_path)
        assert size == 1024
    finally:
        os.unlink(temp_path)

def test_get_file_size_missing():
    """Test file size helper with missing file."""
    size = met.get_file_size("non_existent_file_12345.tmp")
    assert size == 0

def test_format_size():
    """Test human readable size formatting."""
    assert met.format_size(0) == "0 B"
    assert met.format_size(100) == "100.0 B"
    assert met.format_size(1024) == "1.0 KB"
    assert met.format_size(1536) == "1.5 KB" # 1.5 * 1024
    assert met.format_size(1024**2) == "1.0 MB"
    assert met.format_size(1024**3) == "1.0 GB"
    
    # Test rounding
    # 1.25 MB = 1.25 * 1024 * 1024 = 1310720
    assert met.format_size(1310720) == "1.25 MB"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])