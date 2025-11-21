"""
WDR Input/Output Module.

High-level interface for compressing images to WDR archives and 
streaming them back. Handles tiling, headers, and file management.
"""

import numpy as np
from typing import Generator, Optional, Callable
from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp
from wdr.utils import metrics as met
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
) -> dict:
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
        
    Returns:
        dict: Compression statistics (ratio, raw_size, compressed_size).
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
    if progress_callback: progress_callback(1.0)

    # Calculate Stats
    raw_size = image_source.nbytes
    compressed_size = met.get_file_size(output_path)
    cr = raw_size / compressed_size if compressed_size > 0 else 0
    
    return {
        "raw_size": raw_size,
        "compressed_size": compressed_size,
        "compression_ratio": cr
    }

def decompress(
    wdr_path: str,
    reference_image: Optional[np.ndarray] = None,
    progress_callback: Optional[Callable[[float], None]] = None
) -> Generator[np.ndarray, None, None]:
    """
    Streams decompressed tiles from a WDR archive.
    
    Returns a generator that yields 2D numpy arrays (tiles).
    Memory efficient: Only holds one tile in RAM at a time.
    
    Args:
        wdr_path: Path to source .wdr file.
        reference_image: Optional original image for on-the-fly metrics.
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
    
    # Optional Metrics
    metrics = met.StreamMetrics(max_val=255.0) if reference_image is not None else None
    
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
            
            # 5. Update Metrics (Optional)
            if metrics is not None:
                y, x = r * reader.tile_size, c * reader.tile_size
                y_end = min(y + reader.tile_size, reader.height)
                x_end = min(x + reader.tile_size, reader.width)
                
                tile_orig = reference_image[y:y_end, x:x_end]
                tile_valid = tile_recon[:tile_orig.shape[0], :tile_orig.shape[1]]
                metrics.update(tile_orig, tile_valid)

            yield tile_recon
            
            processed += 1
            if progress_callback and processed % 10 == 0:
                progress_callback(processed / total)

    reader.close()
    if progress_callback: progress_callback(1.0)
    
    if metrics:
        mse, psnr = metrics.get_results()
        print(f"\n[WDR Metrics] MSE: {mse:.4f} | PSNR: {psnr:.2f} dB")