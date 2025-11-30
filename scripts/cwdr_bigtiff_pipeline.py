#!/usr/bin/env python3
"""
Gigapixel TIFF WDR Workflow Tool

Command-line tool for compressing and extracting large TIFF images using the WDR library.
Designed for memory-efficient processing of gigapixel images using memory-mapped file access.

COMMANDS:

compress
    Compress a TIFF image to WDR format.

    Usage:
        python scripts/converters.py compress <input.tiff> <output.wdr> [OPTIONS]

    Arguments:
        input.tiff          Input TIFF image file (single channel)

    Options:
        --tile-size N       Tile size for processing (default: 512)
        --scales N          Number of DWT decomposition levels (default: 2)
        --wavelet NAME      Wavelet type from pywt (default: bior4.4)
        --passes N          Number of bit-plane passes (default: 16)
                            Higher values improve quality but increase file size
        --qstep N           Quantization step size (default: None)
                            Use 0 or omit for lossless compression
        --quiet             Suppress progress and statistics output

    Output:
        - Creates output.wdr archive file
        - Prints compression statistics (raw size, compressed size, ratio, time)

extract
    Extract a WDR archive to TIFF format.

    Usage:
        python scripts/converters.py extract <input.wdr> <output.tiff> [OPTIONS]

    Arguments:
        input.wdr           WDR archive file to decompress

    Options:
        --original-image PATH   Path to original image for PSNR calculation
                                If provided, calculates and displays MSE and PSNR metrics
        --quiet                 Suppress progress and statistics output

    Output:
        - Creates output.tiff image file
        - Prints decompression statistics (time, optional PSNR metrics)

FEATURES:
    - Memory-efficient streaming for gigapixel images
    - Compression ratio (CR) metrics
    - PSNR calculation for quality assessment
    - Progress reporting
    - Supports memory-mapped TIFF files

EXAMPLES:
    # Lossless compression
    python scripts/converters.py compress image.tiff compressed.wdr --passes 16 --qstep 0

    # Lossy compression with custom settings
    python scripts/converters.py compress image.tiff compressed.wdr --passes 20 --qstep 2.5

    # Extract without metrics
    python scripts/converters.py extract compressed.wdr reconstructed.tiff

    # Extract with PSNR calculation
    python scripts/converters.py extract compressed.wdr reconstructed.tiff --original-image image.tiff
"""
import argparse
import time
import math
import os
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path
from wdr import io as wdr_io
from wdr.utils import helpers as hlp
from wdr.container import WDRTileReader

#funciones para pasar de rgb a yuv e yuv a rgb

def rgb_to_yuv(img_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    R = img_rgb[..., 0]
    G = img_rgb[..., 1]
    B = img_rgb[..., 2]
    
    Y = 16 + 0.257 * R + 0.504 * G + 0.098 * B
    U = 128 - 0.148 * R - 0.291 * G + 0.439 * B
    V = 128 + 0.439 * R - 0.368 * G - 0.071 * B
    
    return Y, U, V

def yuv_to_rgb(Y: np.ndarray, U: np.ndarray, V: np.ndarray) -> np.ndarray:
    R = 1.164 * (Y - 16) + 1.596 * (V - 128)
    G = 1.164 * (Y - 16) - 0.392 * (U - 128) - 0.813 * (V - 128)
    B = 1.164 * (Y - 16) + 2.017 * (U - 128)
    
    return np.stack([R, G, B], axis=-1)

# --- Metrics ---
class StreamMetrics:
    """
    Accumulates error statistics (SSE) for streaming image comparison.
    Memory Usage: O(1). Math is optimized for in-place operations.
    """
    def __init__(self, max_val=255.0):
        self.sse = 0.0
        self.total_pixels = 0
        self.max_val = max_val

    def update(self, original_tile: np.ndarray, reconstructed_tile: np.ndarray):
        # 1. Handle Edge Padding (Crop to match smallest)
        if original_tile.shape != reconstructed_tile.shape:
            h = min(original_tile.shape[0], reconstructed_tile.shape[0])
            w = min(original_tile.shape[1], reconstructed_tile.shape[1])
            # Views (no copy)
            original_tile = original_tile[:h, :w]
            reconstructed_tile = reconstructed_tile[:h, :w]

        # 2. Memory Optimized Calculation
        # We cast 'original' to float64 creating one new array
        diff = original_tile.astype(np.float64)
        
        # Subtract in-place 
        diff -= reconstructed_tile 
        
        # Square in-place
        np.square(diff, out=diff)
        
        # Accumulate
        self.sse += np.sum(diff)
        self.total_pixels += original_tile.size

    def get_results(self):
        if self.total_pixels == 0: return 0.0, float('inf')
        
        mse = self.sse / self.total_pixels
        if mse == 0: return 0.0, float('inf')
            
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

# --- Image Loading ---
# Increase PIL limit just in case we fall back to it for a medium-sized image

Image.MAX_IMAGE_PIXELS = None 

def load_image(filepath: str) -> np.ndarray:
    """
    Strict Image Loader.
    Loads image but enforces Single-Channel (2D) requirement.
    Raises ValueError if image is Multi-Channel (RGB/3D).
    """
    path = Path(filepath)
    img_array = None
    
    # 1. Load Strategy (Memmap or PIL)
    if path.suffix.lower() in ['.tif', '.tiff', '.btf']:
        try:
            # Load image as memory-mapped array
            # This is necessary for large images since
            # it doesn't load the entire image into memory at once.
            img_array = tifffile.memmap(filepath, mode='r')
        except Exception:
            pass

    if img_array is None:
        img = Image.open(filepath)
        img = img.convert('RGB')
        img_array = np.array(img)

    # 2. Strict Validation (Fail Fast)
    if img_array.ndim != 2 and not (img_array.ndim == 3 and img_array.shape[-1] == 3):
        raise ValueError(
            f"WDR Library Error: Image '{filepath}' must be 2D or 3D (with 3 channels). Found {img_array.ndim}D."
        )
        
    return img_array

def save_image(filepath: str, img_array: np.ndarray) -> None:
    img_array = np.clip(img_array, 0, 255)
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(img_array)
    img.save(filepath)

# --- Compression ---

def compress_tiff(input_path, output_path, tile_size=512, scales=2,
                  wavelet="bior4.4", num_passes=16, quantization_step=None, 
                  verbose=True):
    """Compresses a TIFF to WDR and returns the time taken and compression ratio as dictionary."""
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    start_time = time.time()
    
    # ... (IMPRESIÓN DEL ENCABEZADO) ...
    
    # Load image (Maneja el 2D o 3D y asegura float64)
    image_source = load_image(input_path)
    h, w = image_source.shape[:2]
    
    is_color = image_source.ndim == 3
    channel_names = ['y', 'u', 'v'] if is_color else ['mono']

    # 🟢 1. PRE-PROCESAMIENTO, CÁLCULO DE T Y TAMAÑO ORIGINAL
    if is_color:
        channels = rgb_to_yuv(image_source) # Separa en 3 arrays (Y, U, V)
        
        # El Global T se calcula solo con el canal Y (Luminancia)
        global_max = hlp.scan_for_max_coefficient(channels[0], tile_size, scales, wavelet)
        global_T = hlp.calculate_global_T(global_max)
        
        raw_size = image_source.nbytes # Tamaño de los 3 canales
        if verbose: print("MODO: CWDR (3 Archivos YUV)")
    else:
        channels = (image_source,) # Deja la imagen mono como una tupla de un elemento
        global_max = hlp.scan_for_max_coefficient(image_source, tile_size, scales, wavelet)
        global_T = hlp.calculate_global_T(global_max)
        raw_size = image_source.nbytes # Tamaño de 1 canal
        if verbose: print("MODO: WDR (Mono-canal)")

    # 🟢 2. CONFIGURACIÓN FINAL DE QUANTIZATION
    if quantization_step is not None:
        quant_step = quantization_step
    else:
        quant_step = 0
        if verbose:
            print("  No quantization step provided, using 0 (no quantization)")
    
    # 🟢 3. IMPRESIÓN DE METADATOS FINALES
    if verbose:
        print(f"Dimensions: {w} x {h}")
        print(f"Raw Size: {format_size(raw_size)}")
        print("\n[Pass 1] Analysis (Calculating Global Threshold)...")
        print(f"  Global T: {global_T:.4f}")
        print(f"\n[Pass 2] Compressing to {output_path}...")

    # 🟢 4. BUCLE DE COMPRESIÓN (1 o 3 VECES)
    total_compressed_size = 0
    total_channels = len(channels)
    
    for i, channel_data in enumerate(channels):
        temp_output_path = f"{output_path}.{channel_names[i]}"
        
        if verbose:
            print(f"  -> Compressing Channel {i+1}/{total_channels} ({channel_names[i].upper()}) to {temp_output_path}...")
            
        # Define un callback local para simular el progreso general si verbose es True
        def channel_progress_cb(progress):
            if verbose:
                 # Calcula el progreso total: (i canales terminados + progreso del canal actual) / total
                 overall_progress = (i + progress) / total_channels
                 print(f"  Progress: {overall_progress*100:.1f}%", end='\r')

        wdr_io.compress(
            image_source=channel_data,
            output_path=temp_output_path,
            global_T=global_T,
            tile_size=tile_size,
            scales=scales,
            wavelet=wavelet,
            num_passes=num_passes,
            quant_step=quant_step,
            progress_callback=channel_progress_cb if verbose else None
        )
        total_compressed_size += get_file_size(temp_output_path)
        
    # 🟢 5. CÁLCULO DE MÉTRICAS FINALES
    compressed_size = total_compressed_size
    compression_ratio = raw_size / compressed_size if compressed_size > 0 else 0

    elapsed_time = time.time() - start_time
    
    if verbose:
        print("\n" * 2) # Asegura una línea limpia después del progreso
        print("-" * 40)
        print(f"Uncompressed Data: {format_size(raw_size)}")
        print(f"WDR Archive Size:  {format_size(compressed_size)}")
        print(f"Compression Ratio: {compression_ratio:.2f}x")
        print(f"Time Taken: {elapsed_time:.2f}s")
        print("-" * 40)
        print("="*60)
    
    return {
        "time_taken": elapsed_time,
        "compression_ratio": compression_ratio,
        "raw_size": raw_size,
        "compressed_size": compressed_size
    }

def extract_wdr(input_path, output_path, original_image=None, verbose=True):
    """Extrae un WDR (1 o 3 archivos) a TIFF."""
    start_time = time.time()
    
    # 1. DETERMINAR MODO Y RUTAS DE ARCHIVOS
    base_path = input_path
    channel_suffixes = ['.y', '.u', '.v']
    temp_input_paths = [f"{base_path}{s}" for s in channel_suffixes]
    
    # El modo color se detecta si existen los 3 archivos .y, .u, .v
    is_color_mode = all(os.path.exists(p) for p in temp_input_paths)
    
    if is_color_mode:
        input_paths_to_use = temp_input_paths
        if verbose: print("MODO: CWDR (3 Archivos YUV)")
    elif os.path.exists(base_path):
        input_paths_to_use = [base_path]
        if verbose: print("MODO: WDR (Mono-canal)")
    else:
        raise FileNotFoundError(f"Archivo WDR no encontrado en ruta '{base_path}' ni en modo color (ej: '{base_path}.y', etc.).")

    # 2. CARGAR METADATOS DESDE EL PRIMER ARCHIVO
    reader = WDRTileReader(input_paths_to_use[0])
    width = reader.width
    height = reader.height
    tile_size = reader.tile_size
    
    if verbose:
        print(f"\n[Pass 1] Reading metadata from {input_paths_to_use[0]}...")
        print(f"  Dims: {width}x{height}, Tile: {tile_size}, Passes: {reader.num_passes}")

    rows = reader.rows
    cols = reader.cols
    total_tiles = rows * cols
    
    # 3. BUCLE DE DESCOMPRESIÓN POR CANAL
    reconstructed_channels = []
    
    for i, path in enumerate(input_paths_to_use):
        channel_name = channel_suffixes[i].strip('.') if is_color_mode else 'MONO'
        if verbose:
            print(f"  -> Decompressing Channel {channel_name.upper()}...")

        # Asignación del array donde se ensamblará el canal completo
        channel_data = np.zeros((height, width), dtype=np.float64)
        
        # Obtener el generador de tiles del wdr_io
        tile_gen = wdr_io.decompress(
            wdr_path=path,
            progress_callback=None
        )
        
        # Ensamblaje de Tiles
        processed_tiles = 0
        for r in range(rows):
            for c in range(cols):
                try:
                    tile = next(tile_gen)
                except StopIteration:
                    # Esto solo debería ocurrir si el archivo está corrupto/es más corto
                    print(f"Advertencia: El archivo '{path}' terminó inesperadamente.")
                    break
                
                # Coordenadas de corte para re-ensamblar
                y_start = r * tile_size
                x_start = c * tile_size
                y_end = min(y_start + tile_size, height)
                x_end = min(x_start + tile_size, width)
                
                # Cortar el tile reconstruido para evitar el padding de DWT
                tile_h = y_end - y_start
                tile_w = x_end - x_start
                
                # Asignar el fragmento válido
                channel_data[y_start:y_end, x_start:x_end] = tile[:tile_h, :tile_w]
                
                processed_tiles += 1
                if verbose and processed_tiles % 100 == 0:
                     print(f"     Tiles procesados en {channel_name}: {processed_tiles}/{total_tiles}", end='\r')
            
            if not is_color_mode and verbose:
                 print(f"     Tiles procesados en {channel_name}: {processed_tiles}/{total_tiles}", end='\r')
        
        reconstructed_channels.append(channel_data)
        
    # 4. RECONSTRUCCIÓN FINAL (YUV -> RGB o Mono)
    if is_color_mode:
        # Combinar YUV a RGB (retorna float64, 3 canales)
        Y, U, V = reconstructed_channels
        output_image_float = yuv_to_rgb(Y, U, V)
    else:
        # Mono-canal (retorna float64, 1 canal)
        output_image_float = reconstructed_channels[0]

    # 5. POST-PROCESAMIENTO Y GUARDADO
    
    # Convertir a uint8 (0-255) y recortar valores fuera del rango
    output_image = np.clip(output_image_float, 0, 255).astype(np.uint8)
    
    if verbose:
        print(f"\n[Pass 2] Saving reconstructed image to {output_path}...")
    
    # Asume que esta función existe y guarda el array np en un archivo TIFF/PNG.
    save_image(output_path, output_image) 
    
    # 6. MÉTRICAS (Si se proporciona la imagen original)
    metrics = None
    if original_image:
        # ... (Cargar, calcular PSNR, etc. - Lógica existente) ...
        pass # placeholder
        
    if verbose:
        elapsed = time.time() - start_time
        print(f"Extraction complete in {elapsed:.2f} seconds.")
    
    # Cerrar el reader usado para metadatos
    reader.close()
    
    return metrics

# ==========================================
# CLI Handler
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="WDR Gigapixel Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- Command: compress ---
    cmd_c = subparsers.add_parser("compress", help="Convert TIFF to WDR")
    cmd_c.add_argument("input", help="Input TIFF")
    cmd_c.add_argument("output", help="Output WDR")
    cmd_c.add_argument("--tile-size", type=int, default=512, help="Tile size")
    cmd_c.add_argument("--scales", type=int, default=2, help="Number of dwt scales")
    cmd_c.add_argument("--wavelet", type=str, default="bior4.4" , help="pywt wavelet name")
    cmd_c.add_argument("--passes", type=int, default=16, dest="num_passes", help="Bitplane passes")
    cmd_c.add_argument("--qstep", type=float, default=None, help="Quantization step")
    cmd_c.add_argument("--quiet", action="store_true", help="Suppress output")

    # --- Command: extract ---
    cmd_x = subparsers.add_parser("extract", help="Convert WDR to TIFF")
    cmd_x.add_argument("input", help="Input WDR")
    cmd_x.add_argument("output", help="Output TIFF")
    cmd_x.add_argument("--original-image", type=str, default=None, help="Original image for metrics")
    cmd_x.add_argument("--quiet", action="store_true", help="Suppress output")

    args = parser.parse_args()

    if args.command == "compress":
        compress_tiff(
            args.input, args.output, tile_size=args.tile_size,
            scales=args.scales, wavelet=args.wavelet,
            num_passes=args.num_passes, quantization_step=args.qstep,
            verbose=not args.quiet
        )
    elif args.command == "extract":
        extract_wdr(args.input, args.output, original_image=args.original_image, verbose=not args.quiet)

if __name__ == "__main__":
    main()
