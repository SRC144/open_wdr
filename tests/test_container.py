"""
Unit tests for WDR Container (.wdr file format).
Verifies binary structure, header parsing, and random access.
"""

import pytest
import tempfile
import os
import struct
import numpy as np
from wdr.container import WDRTileWriter, WDRTileReader, MAGIC, HEADER_SIZE

@pytest.fixture
def temp_wdr_file():
    """Creates a temporary file path and cleans it up after test."""
    fd, path = tempfile.mkstemp(suffix=".wdr")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)

def test_header_round_trip(temp_wdr_file):
    """Test that metadata written to header is read back correctly."""
    # Setup params
    width, height = 1024, 2048
    tile_size = 512
    global_T = 128.5
    scales = 3
    wavelet = "bior4.4"
    quant_step = 2.0
    num_passes = 16

    # 1. Write
    writer = WDRTileWriter(
        temp_wdr_file, width, height, tile_size, 
        global_T, scales, wavelet, quant_step, num_passes
    )
    writer.close() # Write header + index

    # 2. Read
    reader = WDRTileReader(temp_wdr_file)
    
    assert reader.width == width
    assert reader.height == height
    assert reader.tile_size == tile_size
    assert reader.global_T == global_T
    assert reader.scales == scales
    assert reader.wavelet == wavelet
    assert reader.quant_step == quant_step
    assert reader.num_passes == num_passes
    
    reader.close()

def test_grid_calculation(temp_wdr_file):
    """Test row/col calculation for uneven image sizes."""
    # 600x600 image with 512 tiles
    # Should be 2x2 grid (512 + remainder)
    writer = WDRTileWriter(temp_wdr_file, 600, 600, 512, 1.0, 2, "db1", 0.0, 8)
    
    assert writer.cols == 2
    assert writer.rows == 2
    assert writer.total_tiles == 4
    writer.close()
    
    reader = WDRTileReader(temp_wdr_file)
    assert reader.cols == 2
    assert reader.rows == 2
    reader.close()

def test_tile_data_integrity(temp_wdr_file):
    """Test that binary blobs are written and retrieved correctly."""
    # 2x2 Grid (1024x1024 image, 512 tiles)
    writer = WDRTileWriter(temp_wdr_file, 1024, 1024, 512, 1.0, 2, "db1", 0.0, 8)
    
    # Create dummy compressed data (unique for each tile)
    blob_00 = b'\xAA\xBB'       # Tile 0
    blob_01 = b'\xCC\xDD\xEE'   # Tile 1
    blob_10 = b'\x00\x01\x02'   # Tile 2
    blob_11 = b'\xFF'           # Tile 3
    
    # Write in Row-Major Order
    writer.add_tile(blob_00)
    writer.add_tile(blob_01)
    writer.add_tile(blob_10)
    writer.add_tile(blob_11)
    
    writer.close()
    
    # Read back
    reader = WDRTileReader(temp_wdr_file)
    
    # Random Access checks
    assert reader.get_tile_bytes(0, 0) == blob_00
    assert reader.get_tile_bytes(0, 1) == blob_01
    assert reader.get_tile_bytes(1, 0) == blob_10
    assert reader.get_tile_bytes(1, 1) == blob_11
    
    reader.close()

def test_invalid_magic_bytes(temp_wdr_file):
    """Test that reader rejects non-WDR files."""
    # Create a fake file with wrong magic
    with open(temp_wdr_file, 'wb') as f:
        f.write(b'FAKE') # Wrong magic
        f.write(b'\x00' * 100)
        
    with pytest.raises(ValueError, match="Invalid WDR File"):
        WDRTileReader(temp_wdr_file)

def test_out_of_bounds_read(temp_wdr_file):
    """Test reading a tile index that doesn't exist."""
    writer = WDRTileWriter(temp_wdr_file, 512, 512, 512, 1.0, 2, "db1", 0.0, 8)
    writer.add_tile(b'\x00') # 1x1 grid
    writer.close()
    
    reader = WDRTileReader(temp_wdr_file)
    
    # Grid is 1x1 (row 0, col 0 only)
    assert reader.get_tile_bytes(0, 0) == b'\x00'
    assert reader.get_tile_bytes(0, 1) is None # Out of bounds col
    assert reader.get_tile_bytes(1, 0) is None # Out of bounds row
    
    reader.close()

def test_missing_tiles_warning(temp_wdr_file, capsys):
    """Test that writer warns if we close before writing all tiles."""
    # Expects 4 tiles (2x2)
    writer = WDRTileWriter(temp_wdr_file, 1024, 1024, 512, 1.0, 2, "db1", 0.0, 8)
    
    # Only write 1
    writer.add_tile(b'\x00')
    
    writer.close()
    
    # Check stdout for warning
    captured = capsys.readouterr()
    assert "Warning: Expected 4 tiles, but wrote 1" in captured.out

if __name__ == "__main__":
    pytest.main([__file__, "-v"])