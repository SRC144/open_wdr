"""
Batched metrics computation for gigapixel images.

This module provides streaming/batched computation of PSNR and MSE metrics
without loading entire images into memory. It processes images tile-by-tile,
accumulating statistics across batches.
"""

import numpy as np
from typing import Tuple

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


class TileReader(Protocol):
    """Protocol for tile readers that support block-based access."""
    
    def size(self) -> Tuple[int, int]:
        """Return (width, height) of the image."""
        ...
    
    def read_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        """Read a block of pixels starting at (x, y) with given dimensions."""
        ...


class BatchedMetrics:
    """
    Accumulates metrics (MSE, RMSE, PSNR) across batches of image tiles.
    
    This class allows computing image quality metrics for gigapixel images
    without loading the entire image into memory. It processes tiles in batches,
    accumulating sum of squared errors and pixel counts.
    
    Theory:
        MSE = (1/N) * Σ(original[i] - reconstructed[i])²
        We accumulate: sum_squared_errors = Σ(original[i] - reconstructed[i])²
        and total_pixels = N, then compute MSE = sum_squared_errors / total_pixels
    """
    
    def __init__(self):
        """Initialize an empty metrics accumulator."""
        self.sum_squared_errors = 0.0
        self.total_pixels = 0
        self.min_error = float('inf')
        self.max_error = float('-inf')
    
    def add_batch(self, original: np.ndarray, reconstructed: np.ndarray) -> None:
        """
        Add a batch of pixels to the metrics accumulator.
        
        Args:
            original: Original image tile (2D array, dtype float64, range [0, 255])
            reconstructed: Reconstructed image tile (2D array, dtype float64, range [0, 255])
        
        Raises:
            ValueError: If arrays have different shapes
        """
        if original.shape != reconstructed.shape:
            raise ValueError(
                f"Original and reconstructed tiles must have same shape. "
                f"Got {original.shape} vs {reconstructed.shape}"
            )
        
        # Clip both arrays to valid range [0, 255]
        original_clipped = np.clip(original, 0, 255)
        reconstructed_clipped = np.clip(reconstructed, 0, 255)
        
        # Compute squared errors for this batch
        errors = original_clipped - reconstructed_clipped
        squared_errors = errors ** 2
        
        # Accumulate statistics
        batch_sum_sq = float(np.sum(squared_errors))
        batch_pixels = int(squared_errors.size)
        
        self.sum_squared_errors += batch_sum_sq
        self.total_pixels += batch_pixels
        
        # Track min/max error for diagnostics
        batch_min = float(np.min(errors))
        batch_max = float(np.max(errors))
        if batch_min < self.min_error:
            self.min_error = batch_min
        if batch_max > self.max_error:
            self.max_error = batch_max
    
    def finalize(self) -> Tuple[float, float, float]:
        """
        Compute final metrics from accumulated statistics.
        
        Returns:
            Tuple of (mse, rmse, psnr):
            - mse: Mean Squared Error
            - rmse: Root Mean Squared Error
            - psnr: Peak Signal-to-Noise Ratio in dB (or inf if MSE=0)
        
        Raises:
            ValueError: If no pixels were added (total_pixels == 0)
        """
        if self.total_pixels == 0:
            raise ValueError("No pixels were added to metrics accumulator")
        
        # Compute MSE
        mse = self.sum_squared_errors / self.total_pixels
        
        # Compute RMSE
        rmse = np.sqrt(mse)
        
        # Compute PSNR
        if mse == 0.0:
            psnr = float('inf')
        else:
            max_pixel = 255.0
            psnr = 20.0 * np.log10(max_pixel / rmse)
        
        return mse, rmse, psnr


def compute_batched_metrics(
    original_reader: TileReader,
    reconstructed_reader: TileReader,
    tile_width: int,
    tile_height: int,
) -> Tuple[float, float, float]:
    """
    Compute PSNR/MSE metrics by processing images tile-by-tile.
    
    This function streams both original and reconstructed images in tiles,
    accumulating metrics without loading full images into memory. It's designed
    for gigapixel images where full-image reconstruction would exceed memory limits.
    
    Args:
        original_reader: TileReader for the original image
        reconstructed_reader: TileReader for the reconstructed image
        tile_width: Width of tiles to process (must match compression tile width)
        tile_height: Height of tiles to process (must match compression tile height)
    
    Returns:
        Tuple of (mse, rmse, psnr) as computed by BatchedMetrics.finalize()
    
    Raises:
        ValueError: If image dimensions don't match between readers
    """
    orig_width, orig_height = original_reader.size()
    recon_width, recon_height = reconstructed_reader.size()
    
    if orig_width != recon_width or orig_height != recon_height:
        raise ValueError(
            f"Original and reconstructed images must have same dimensions. "
            f"Got {orig_width}x{orig_height} vs {recon_width}x{recon_height}"
        )
    
    metrics = BatchedMetrics()
    
    # Process tiles matching the compression grid
    tiles_x = int(np.ceil(orig_width / tile_width))
    tiles_y = int(np.ceil(orig_height / tile_height))
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            origin_x = tx * tile_width
            origin_y = ty * tile_height
            
            # Calculate actual block dimensions (handle edge tiles)
            block_w = min(tile_width, orig_width - origin_x)
            block_h = min(tile_height, orig_height - origin_y)
            
            # Read corresponding tiles from both images
            orig_tile = original_reader.read_block(origin_x, origin_y, block_w, block_h)
            recon_tile = reconstructed_reader.read_block(origin_x, origin_y, block_w, block_h)
            
            # Ensure both are float64 and 2D
            if orig_tile.dtype != np.float64:
                orig_tile = orig_tile.astype(np.float64)
            if recon_tile.dtype != np.float64:
                recon_tile = recon_tile.astype(np.float64)
            
            # Handle potential 3D arrays (RGB -> grayscale conversion)
            if len(orig_tile.shape) == 3:
                # Convert RGB to grayscale using standard weights
                orig_tile = np.dot(orig_tile[..., :3], [0.2989, 0.5870, 0.1140])
            if len(recon_tile.shape) == 3:
                recon_tile = np.dot(recon_tile[..., :3], [0.2989, 0.5870, 0.1140])
            
            # Ensure 2D
            if len(orig_tile.shape) != 2:
                raise ValueError(f"Expected 2D tile, got shape {orig_tile.shape}")
            if len(recon_tile.shape) != 2:
                raise ValueError(f"Expected 2D tile, got shape {recon_tile.shape}")
            
            # Add batch to metrics
            metrics.add_batch(orig_tile, recon_tile)
    
    return metrics.finalize()

