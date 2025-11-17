"""
Unit tests for wdr.utils.batched_metrics
"""

import numpy as np
import pytest
import tempfile
import os
from pathlib import Path
from wdr.utils.batched_metrics import BatchedMetrics, compute_batched_metrics
from wdr.utils.tile_reader import PillowTileReader
from wdr.utils import helpers as hlp


class MockTileReader:
    """Mock tile reader for testing."""
    
    def __init__(self, img_array: np.ndarray):
        self._img = img_array.astype(np.float64)
        self._height, self._width = self._img.shape
    
    def size(self):
        return self._width, self._height
    
    def read_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        y1 = y + height
        x1 = x + width
        return self._img[y:y1, x:x1].copy()
    
    def close(self):
        pass


def test_batched_metrics_identical_images():
    """Test that identical images produce perfect metrics (MSE=0, PSNR=inf)."""
    img = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8).astype(np.float64)
    
    metrics = BatchedMetrics()
    metrics.add_batch(img, img)
    
    mse, rmse, psnr = metrics.finalize()
    
    assert mse == 0.0
    assert rmse == 0.0
    assert psnr == float('inf')


def test_batched_metrics_small_difference():
    """Test metrics with a small known difference."""
    original = np.ones((10, 10), dtype=np.float64) * 100.0
    reconstructed = original + 1.0  # Add 1 to each pixel
    
    metrics = BatchedMetrics()
    metrics.add_batch(original, reconstructed)
    
    mse, rmse, psnr = metrics.finalize()
    
    # MSE should be 1.0 (mean of (1.0)^2)
    assert abs(mse - 1.0) < 1e-10
    assert abs(rmse - 1.0) < 1e-10
    # PSNR = 20 * log10(255 / 1.0) ≈ 48.13 dB
    assert abs(psnr - 48.1308) < 0.01


def test_batched_metrics_multiple_batches():
    """Test that multiple batches accumulate correctly."""
    metrics = BatchedMetrics()
    
    # Add two batches
    batch1_orig = np.ones((5, 5), dtype=np.float64) * 100.0
    batch1_recon = batch1_orig + 1.0
    metrics.add_batch(batch1_orig, batch1_recon)
    
    batch2_orig = np.ones((5, 5), dtype=np.float64) * 200.0
    batch2_recon = batch2_orig + 2.0
    metrics.add_batch(batch2_orig, batch2_recon)
    
    mse, rmse, psnr = metrics.finalize()
    
    # Total pixels: 25 + 25 = 50
    # Sum of squared errors: 25 * 1^2 + 25 * 2^2 = 25 + 100 = 125
    # MSE = 125 / 50 = 2.5
    expected_mse = 2.5
    assert abs(mse - expected_mse) < 1e-10
    assert abs(rmse - np.sqrt(expected_mse)) < 1e-10


def test_batched_metrics_shape_mismatch():
    """Test that shape mismatch raises ValueError."""
    metrics = BatchedMetrics()
    orig = np.ones((10, 10), dtype=np.float64)
    recon = np.ones((10, 11), dtype=np.float64)
    
    with pytest.raises(ValueError, match="same shape"):
        metrics.add_batch(orig, recon)


def test_batched_metrics_empty_accumulator():
    """Test that finalizing empty accumulator raises ValueError."""
    metrics = BatchedMetrics()
    
    with pytest.raises(ValueError, match="No pixels"):
        metrics.finalize()


def test_batched_metrics_clipping():
    """Test that values are clipped to [0, 255] range."""
    original = np.array([[100.0, 200.0], [50.0, 150.0]], dtype=np.float64)
    # Reconstructed has values outside [0, 255] range
    reconstructed = np.array([[300.0, -10.0], [50.0, 150.0]], dtype=np.float64)
    
    metrics = BatchedMetrics()
    metrics.add_batch(original, reconstructed)
    
    mse, rmse, psnr = metrics.finalize()
    
    # Clipped reconstructed: [[255.0, 0.0], [50.0, 150.0]]
    # Errors: [[155.0, 200.0], [0.0, 0.0]]
    # Squared errors: [[24025.0, 40000.0], [0.0, 0.0]]
    # MSE = (24025 + 40000) / 4 = 16006.25
    expected_mse = 16006.25
    assert abs(mse - expected_mse) < 1e-6


def test_compute_batched_metrics_with_readers():
    """Test compute_batched_metrics with actual tile readers."""
    # Create a test image
    img_array = np.random.randint(0, 256, size=(64, 64), dtype=np.uint8)
    
    # Save to temporary files
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f1:
        orig_path = f1.name
        hlp.save_image(orig_path, img_array)
    
    # Create a slightly modified version
    modified = img_array.astype(np.float64) + 1.0
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f2:
        recon_path = f2.name
        hlp.save_image(recon_path, modified)
    
    try:
        orig_reader = PillowTileReader(Path(orig_path), grayscale=True)
        recon_reader = PillowTileReader(Path(recon_path), grayscale=True)
        
        try:
            mse, rmse, psnr = compute_batched_metrics(
                orig_reader,
                recon_reader,
                tile_width=32,
                tile_height=32,
            )
            
            # Should have small MSE (around 1.0 since we added 1.0 to each pixel)
            assert mse > 0
            assert rmse > 0
            assert psnr < float('inf')
            assert psnr > 0  # Should be positive
        finally:
            orig_reader.close()
            recon_reader.close()
    finally:
        os.unlink(orig_path)
        os.unlink(recon_path)


def test_compute_batched_metrics_dimension_mismatch():
    """Test that dimension mismatch raises ValueError."""
    orig_img = np.ones((100, 100), dtype=np.float64)
    recon_img = np.ones((100, 101), dtype=np.float64)
    
    orig_reader = MockTileReader(orig_img)
    recon_reader = MockTileReader(recon_img)
    
    try:
        with pytest.raises(ValueError, match="same dimensions"):
            compute_batched_metrics(orig_reader, recon_reader, 32, 32)
    finally:
        orig_reader.close()
        recon_reader.close()

