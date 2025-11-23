"""
WDR Input/Output Module.

High-level interface for compressing images to WDR archives and 
streaming them back. Handles tiling, headers, and file management.
"""

import numpy as np
from typing import Generator, Optional, Callable
from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp
from wdr.container import WDRTileWriter, WDRTileReader

def compress(
    image_source: np.ndarray, 
    output_path: str, 
    global_T: float,
    tile_size: int = 512,
    scales: int = 2,
    wavelet: str = 'bior4.4',
    num_passes: int = 16,
    quant_step: Optional[float] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> None:
    """
    Compresses a 2D image array into a WDR archive file.
    
    Streams the image tile-by-tile to keep memory usage low (O(1)).
    
    Args:
        image_source: 2D numpy array (Single channel).
        output_path: Destination .wdr file path.
        global_T: Threshold calculated from Pass 1 analysis.
        tile_size: Size of compression blocks (default 512).
        scales: DWT decomposition levels.
        wavelet: Wavelet type (default 'bior4.4').
        num_passes: Bitplane passes (Quality/Size tradeoff).
        quant_step: Optional pre-quantization step size.
        progress_callback: Function accepting float (0.0-1.0).
    """
    if image_source.ndim != 2:
        raise ValueError("WDR IO Error: Input must be 2D single-channel.")

    h, w = image_source.shape
    
    # Initialize the Archive Writer
    writer = WDRTileWriter(
        output_path, w, h, tile_size, global_T, 
        scales, wavelet, quant_step, num_passes
    )
    
    # Initialize the Core Codec
    compressor = wdr_coder.WDRCompressor(num_passes)
    
    processed = 0
    total_tiles = writer.total_tiles
    
    # Stream Tiles -> Encode -> Write
    for tile in hlp.yield_tiles(image_source, tile_size):
        # 1. DWT
        coeffs = hlp.do_dwt(tile, scales, wavelet)
        flat, _ = hlp.flatten_coeffs(coeffs)
        
        # 2. Quantize
        if quant_step is not None and quant_step > 0:
            flat, _ = hlp.quantize_coeffs(flat, quant_step)
            
        # 3. WDR Encoding (C++)
        compressed_vec = compressor.compress(flat, global_T)
        
        # 4. Write Blob
        writer.add_tile(bytes(compressed_vec))
        
        processed += 1
        if progress_callback and processed % 10 == 0:
            progress_callback(processed / total_tiles)

    writer.close()
    if progress_callback: 
        progress_callback(1.0)

def decompress(
    wdr_path: str,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Generator[np.ndarray, None, None]:
    """
    Streams decompressed tiles from a WDR archive.
    
    Returns a generator that yields 2D numpy arrays (tiles).
    Memory efficient: Only holds one tile in RAM at a time.
    
    Args:
        wdr_path: Path to source .wdr file.
        progress_callback: Function accepting float (0.0-1.0).
        
    Yields:
        np.ndarray: Reconstructed tile (float64).
    """
    reader = WDRTileReader(wdr_path)
    decompressor = wdr_coder.WDRCompressor(reader.num_passes)
    
    # Pre-calc constant metadata for DWT reconstruction
    dummy = np.zeros((reader.tile_size, reader.tile_size), dtype=np.float64)
    coeffs = hlp.do_dwt(dummy, reader.scales, reader.wavelet)
    flat, shape_meta = hlp.flatten_coeffs(coeffs)
    num_coeffs = len(flat)
    
    processed = 0
    total = reader.rows * reader.cols
    
    # Iterate Row-Major
    for r in range(reader.rows):
        for c in range(reader.cols):
            # 1. Read Blob
            blob = reader.get_tile_bytes(r, c)
            if not blob:
                yield np.zeros((reader.tile_size, reader.tile_size), dtype=np.uint8)
                continue

            # 2. WDR Decoding (C++)
            flat_recon = decompressor.decompress(blob, reader.global_T, num_coeffs)
            flat_recon = np.array(flat_recon, dtype=np.float64)
            
            # 3. Dequantize
            if reader.quant_step > 0:
                flat_recon = hlp.dequantize_coeffs(flat_recon, reader.quant_step)
            
            # 4. Inverse DWT
            coeffs_recon = hlp.unflatten_coeffs(flat_recon, shape_meta)
            tile_recon = hlp.do_idwt(coeffs_recon, wavelet=reader.wavelet)
            tile_recon = tile_recon[:reader.tile_size, :reader.tile_size] # Crop padding

            yield tile_recon
            
            processed += 1
            if progress_callback and processed % 10 == 0:
                progress_callback(processed / total)

    reader.close()
    if progress_callback: 
        progress_callback(1.0)