import math
import os
import numpy as np

class StreamMetrics:
    """
    Accumulates error statistics (SSE) for streaming image comparison.
    Memory Usage: O(1) - Stores only scalar sums, not pixel arrays.
    """
    def __init__(self, max_val=255.0):
        self.sse = 0.0        # Sum of Squared Errors
        self.total_pixels = 0
        self.max_val = max_val

    def update(self, original_tile: np.ndarray, reconstructed_tile: np.ndarray):
        """
        Compare two tiles and accumulate the error.
        Tiles are discarded immediately after calculation.
        """
        # 1. Handle shape mismatches (e.g., padding differences at edges)
        if original_tile.shape != reconstructed_tile.shape:
            h = min(original_tile.shape[0], reconstructed_tile.shape[0])
            w = min(original_tile.shape[1], reconstructed_tile.shape[1])
            original_tile = original_tile[:h, :w]
            reconstructed_tile = reconstructed_tile[:h, :w]

        # 2. Calculate Squared Error for this specific tile
        # Convert to float64 to prevent overflow
        diff = original_tile.astype(np.float64) - reconstructed_tile.astype(np.float64)
        
        # 3. Accumulate
        self.sse += np.sum(diff ** 2)
        self.total_pixels += original_tile.size

    def get_results(self):
        """
        Returns the final metrics based on accumulated data.
        """
        if self.total_pixels == 0:
            return 0.0, float('inf')

        mse = self.sse / self.total_pixels
        
        if mse == 0:
            return 0.0, float('inf')
            
        psnr = 20 * math.log10(self.max_val / math.sqrt(mse))
        return mse, psnr

def get_file_size(filepath: str) -> int:
    try:
        return os.path.getsize(filepath)
    except OSError:
        return 0

def format_size(size_bytes: int) -> str:
    if size_bytes == 0: return "0 B"
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB']
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {suffixes[i]}"