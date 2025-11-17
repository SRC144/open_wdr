#!/usr/bin/env python3
"""
Main example script for WDR image compression pipeline.

This script demonstrates the full compression and decompression pipeline:
1. Load an image
2. Perform Discrete Wavelet Transform (DWT)
3. Flatten coefficients using WDR scanning order
4. Compress using WDR algorithm
5. Decompress using WDR algorithm
6. Unflatten coefficients
7. Perform Inverse Discrete Wavelet Transform (IDWT)
8. Save reconstructed image

The script supports command-line arguments for customization:
- Input image file path
- Output .wdr file path
- Number of wavelet decomposition scales (default: 2, recommended: 2-3)
- Wavelet name (default: bior4.4)
- Optional reconstructed image output path

Example usage:
    python main.py input.png output.wdr
    python main.py input.png output.wdr --scales 2 --reconstructed recon.png

Note: Using scales=6+ introduces boundary artifacts as warned by PyWavelets,
      so scales=2-3 is recommended for practical use.
"""

import argparse
import math
import os
import shutil
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

from wdr import coder as wdr_coder
from wdr.utils import helpers as hlp
from wdr.utils.tile_reader import create_tile_reader
from wdr.utils.batched_metrics import compute_batched_metrics


def calculate_psnr(original: np.ndarray, compressed: np.ndarray) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images.

    PSNR is a metric used to measure the quality of reconstructed images.
    Higher PSNR values indicate better quality. For lossless compression,
    PSNR should be infinite (or very high).

    Formula: PSNR = 20 * log10(MAX_PIXEL / sqrt(MSE))
    where MSE is the Mean Squared Error between original and compressed images.

    Args:
        original: Original image array (2D NumPy array, dtype float64)
        compressed: Compressed/reconstructed image array (2D NumPy array, dtype float64)

    Returns:
        PSNR value in dB. Returns float('inf') if images are identical (MSE = 0).

    Note:
        The function assumes images are in the range [0, 255] with max_pixel = 255.0.
    """
    # Calculate MSE
    mse = np.mean((original - compressed) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float("inf")

    # Calculate PSNR
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    return psnr


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "compressed"
PIL_DEFAULT_PIXEL_LIMIT = 178_956_970


@dataclass
class TileCacheEntry:
    index: int
    origin_x: int
    origin_y: int
    width: int
    height: int
    coeff_path: Path
    coeff_count: int


@dataclass
class CacheSummary:
    entries: List[TileCacheEntry]
    image_width: int
    image_height: int
    global_max_abs: float


WDR_MAGIC = 0x57445200
WDRT_MAGIC = 0x54445257
TILED_HEADER_STRUCT = struct.Struct("<IIIIIIIIIddQ32s")
TILE_CHUNK_STRUCT = struct.Struct("<QIIIIIQ")


def _resolve_output_path(path_arg: str, default_dir: Path) -> Path:
    """
    Resolve an output path, defaulting to ``default_dir`` when only a filename is provided.
    """
    target = Path(path_arg)
    if target.parent == Path("."):
        target = default_dir / target.name
    target.parent.mkdir(parents=True, exist_ok=True)
    return target.resolve()


def _calculate_initial_threshold_from_max(max_abs: float) -> float:
    if max_abs <= 0.0:
        return 1.0
    T = 2.0 ** math.floor(math.log2(max_abs))
    if max_abs < T:
        T /= 2.0
    if max_abs >= 2.0 * T:
        T *= 2.0
    return T


def _guard_large_image(
    path: Path,
    allow_large: bool,
    custom_limit: Optional[int],
) -> tuple[int, int, str]:
    """
    Inspect the image dimensions and manage Pillow's decompression-bomb guard.
    Returns (width, height, mode) for downstream reader selection.
    """
    original_limit = Image.MAX_IMAGE_PIXELS

    def _open_once() -> tuple[int, int, str]:
        with Image.open(path) as probe:
            return probe.size[0], probe.size[1], probe.mode

    if not allow_large:
        try:
            return _open_once()
        except Image.DecompressionBombError as exc:
            limit = original_limit or PIL_DEFAULT_PIXEL_LIMIT
            raise ValueError(
                f"Image '{path.name}' exceeds Pillow's safety limit "
                f"({limit:,} pixels). Re-run with --allow-large-image "
                f"and optionally --max-image-pixels to acknowledge processing."
            ) from exc

    provisional_limit = custom_limit if custom_limit is not None else None
    Image.MAX_IMAGE_PIXELS = provisional_limit
    try:
        width, height, mode = _open_once()
    except Image.DecompressionBombError as exc:
        Image.MAX_IMAGE_PIXELS = original_limit
        raise ValueError(
            f"Image '{path.name}' still exceeds the provided --max-image-pixels "
            f"limit ({custom_limit:,} pixels). Increase the limit or down-sample."
        ) from exc

    pixel_count = width * height
    requested_limit = custom_limit if custom_limit is not None else PIL_DEFAULT_PIXEL_LIMIT
    if custom_limit is not None and pixel_count > custom_limit:
        Image.MAX_IMAGE_PIXELS = original_limit
        raise ValueError(
            f"Image '{path.name}' has {pixel_count:,} pixels which exceeds "
            f"the specified --max-image-pixels value ({custom_limit:,})."
        )

    Image.MAX_IMAGE_PIXELS = max(pixel_count * 2, requested_limit)
    print(
        f"  Allowing large image ({width}x{height}, mode={mode}); "
        f"Image.MAX_IMAGE_PIXELS set to {Image.MAX_IMAGE_PIXELS:,}"
    )
    return width, height, mode


def _prepare_cache_dir(cache_dir_arg: str, keep_cache: bool) -> Tuple[Path, bool]:
    if cache_dir_arg:
        path = Path(cache_dir_arg).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path, False
    tmp_path = Path(tempfile.mkdtemp(prefix="wdr_tiles_"))
    return tmp_path, not keep_cache


def _cache_tiles(args, cache_dir: Path, reader) -> CacheSummary:
    entries: List[TileCacheEntry] = []
    global_max_abs = 0.0

    image_width, image_height = reader.size()
    tiles_x = math.ceil(image_width / args.tile_width)
    tiles_y = math.ceil(image_height / args.tile_height)

    print(f"  Image size: {image_height}x{image_width}")
    print(f"  Planned tiles: {tiles_x * tiles_y} ({tiles_x} x {tiles_y})")

    tile_index = 0
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            origin_x = tx * args.tile_width
            origin_y = ty * args.tile_height
            tile_width = min(args.tile_width, image_width - origin_x)
            tile_height = min(args.tile_height, image_height - origin_y)

            tile_array = reader.read_block(origin_x, origin_y, tile_width, tile_height)
            wavelet_coeffs = hlp.do_dwt(tile_array, scales=args.scales, wavelet=args.wavelet)
            flat_coeffs, _ = hlp.flatten_coeffs(wavelet_coeffs)

            if flat_coeffs.size > 0:
                tile_max = float(np.max(np.abs(flat_coeffs)))
                if tile_max > global_max_abs:
                    global_max_abs = tile_max

            coeff_path = cache_dir / f"tile_{tile_index:06d}.npy"
            np.save(coeff_path, flat_coeffs, allow_pickle=False)

            entries.append(
                TileCacheEntry(
                    index=tile_index,
                    origin_x=origin_x,
                    origin_y=origin_y,
                    width=tile_width,
                    height=tile_height,
                    coeff_path=coeff_path,
                    coeff_count=int(flat_coeffs.size),
                )
            )
            tile_index += 1

    return CacheSummary(entries=entries, image_width=image_width, image_height=image_height, global_max_abs=global_max_abs)


def _resolve_quantization_step(args, global_max_abs: float) -> Tuple[float, bool]:
    if args.quantization_step is None:
        probe = np.array([global_max_abs], dtype=np.float64)
        step = hlp.calculate_quantization_step(
            probe,
            num_passes=args.num_passes,
            method=args.quantization_method,
            compression_focused=True,
        )
        return float(step), True

    if args.quantization_step == 0:
        return 0.0, False

    if args.quantization_step < 0:
        raise ValueError("--quantization-step must be >= 0")

    return float(args.quantization_step), True


def _pack_tiled_header(
    num_passes: int,
    num_scales: int,
    tile_width: int,
    tile_height: int,
    image_width: int,
    image_height: int,
    global_initial_T: float,
    quantization_step: float,
    quantization_enabled: bool,
    total_tiles: int,
    wavelet: str,
) -> bytes:
    flags = 1 if quantization_enabled and quantization_step > 0 else 0
    quant_step_value = quantization_step if quantization_enabled and quantization_step > 0 else 0.0

    wavelet_bytes = wavelet.encode("ascii", errors="ignore")[:32]
    wavelet_bytes = wavelet_bytes.ljust(32, b"\0")

    return TILED_HEADER_STRUCT.pack(
        WDRT_MAGIC,
        1,
        flags,
        num_passes,
        num_scales,
        tile_width,
        tile_height,
        image_width,
        image_height,
        global_initial_T,
        quant_step_value,
        total_tiles,
        wavelet_bytes,
    )


def _pack_tile_chunk_header(entry: TileCacheEntry, chunk_size: int) -> bytes:
    return TILE_CHUNK_STRUCT.pack(
        entry.index,
        entry.origin_x,
        entry.origin_y,
        entry.coeff_count,
        entry.width,
        entry.height,
        chunk_size,
    )


def _encode_tiles(
    entries: List[TileCacheEntry],
    output_path: Path,
    header_bytes: bytes,
    global_initial_T: float,
    quantization_step: float,
    quantization_enabled: bool,
    num_passes: int,
) -> Dict[str, int]:
    total_coeff_bytes = 0

    with open(output_path, "wb") as f_out:
        f_out.write(header_bytes)

        for entry in entries:
            flat_coeffs = np.load(entry.coeff_path, allow_pickle=False)
            if quantization_enabled and quantization_step > 0:
                quantized_coeffs, _ = hlp.quantize_coeffs(flat_coeffs, quantization_step)
            else:
                quantized_coeffs = flat_coeffs

            quantized_coeffs = np.ascontiguousarray(quantized_coeffs)
            total_coeff_bytes += quantized_coeffs.nbytes

            payload = wdr_coder.compress_tile(quantized_coeffs, global_initial_T, num_passes)
            chunk_header = _pack_tile_chunk_header(entry, len(payload))
            f_out.write(chunk_header)
            if payload:
                f_out.write(payload)

    return {"coeff_bytes": total_coeff_bytes}


def _read_tiled_header(stream) -> Dict[str, float]:
    header_bytes = stream.read(TILED_HEADER_STRUCT.size)
    if len(header_bytes) != TILED_HEADER_STRUCT.size:
        raise ValueError("File is too small to contain a tiled WDR header")

    (
        magic,
        version,
        flags,
        num_passes,
        num_scales,
        tile_width,
        tile_height,
        image_width,
        image_height,
        global_initial_T,
        quant_step,
        total_tiles,
        wavelet_bytes,
    ) = TILED_HEADER_STRUCT.unpack(header_bytes)

    if magic != WDRT_MAGIC:
        raise ValueError("Not a tiled WDR file (bad magic number)")
    if version != 1:
        raise ValueError(f"Unsupported tiled WDR version: {version}")

    wavelet = wavelet_bytes.split(b"\0", 1)[0].decode("ascii", errors="ignore")
    quantization_enabled = bool(flags & 0x1)

    return {
        "num_passes": num_passes,
        "num_scales": num_scales,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "image_width": image_width,
        "image_height": image_height,
        "global_initial_T": global_initial_T,
        "quantization_step": quant_step,
        "quantization_enabled": quantization_enabled,
        "total_tiles": total_tiles,
        "wavelet": wavelet,
    }


def _get_shape_metadata(
    tile_height: int,
    tile_width: int,
    scales: int,
    wavelet: str,
    cache: Dict[Tuple[int, int, int, str], Dict[str, object]],
):
    key = (tile_height, tile_width, scales, wavelet)
    if key not in cache:
        zero_tile = np.zeros((tile_height, tile_width), dtype=np.float64)
        coeffs = hlp.do_dwt(zero_tile, scales=scales, wavelet=wavelet)
        _, shape_metadata = hlp.flatten_coeffs(coeffs)
        cache[key] = shape_metadata
    return cache[key]


def _decompress_tiled_file(input_path: Path) -> Tuple[np.ndarray, Dict[str, float]]:
    with open(input_path, "rb") as f_in:
        header = _read_tiled_header(f_in)

        reconstructed = np.zeros(
            (header["image_height"], header["image_width"]), dtype=np.float64
        )
        shape_cache: Dict[Tuple[int, int, int, str], Dict[str, object]] = {}
        quant_step = header["quantization_step"]

        for _ in range(header["total_tiles"]):
            chunk_header_bytes = f_in.read(TILE_CHUNK_STRUCT.size)
            if len(chunk_header_bytes) != TILE_CHUNK_STRUCT.size:
                raise ValueError("Unexpected EOF while reading tile header")

            (
                tile_index,
                origin_x,
                origin_y,
                coeff_count,
                tile_width,
                tile_height,
                chunk_size,
            ) = TILE_CHUNK_STRUCT.unpack(chunk_header_bytes)

            payload = f_in.read(chunk_size)
            if len(payload) != chunk_size:
                raise ValueError(f"Tile {tile_index} payload truncated")

            coeffs = wdr_coder.decompress_tile(
                payload,
                header["global_initial_T"],
                coeff_count,
                header["num_passes"],
            )
            coeffs = np.asarray(coeffs, dtype=np.float64)

            if header["quantization_enabled"] and quant_step > 0:
                coeffs = hlp.dequantize_coeffs(coeffs, quant_step)

            shape_metadata = _get_shape_metadata(
                tile_height,
                tile_width,
                header["num_scales"],
                header["wavelet"],
                shape_cache,
            )
            coeff_structure = hlp.unflatten_coeffs(coeffs, shape_metadata)
            tile_image = hlp.do_idwt(coeff_structure, wavelet=header["wavelet"])

            reconstructed[
                origin_y : origin_y + tile_height, origin_x : origin_x + tile_width
            ] = tile_image[:tile_height, :tile_width]

    return reconstructed, header


def _reconstruct_full_image(reader, tile_width: int, tile_height: int) -> np.ndarray:
    """
    Stitch the entire source image using the provided tile reader (used for metrics when
    Pillow cannot reopen the original).
    """
    width, height = reader.size()
    canvas = np.zeros((height, width), dtype=np.float64)

    tiles_x = math.ceil(width / tile_width)
    tiles_y = math.ceil(height / tile_height)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            origin_x = tx * tile_width
            origin_y = ty * tile_height
            block_w = min(tile_width, width - origin_x)
            block_h = min(tile_height, height - origin_y)
            block = reader.read_block(origin_x, origin_y, block_w, block_h)
            canvas[origin_y : origin_y + block_h, origin_x : origin_x + block_w] = block
    return canvas


def run_tiled_compression(args, output_path: Path, reader):
    cache_dir, cleanup_cache = _prepare_cache_dir(args.tile_cache_dir, args.keep_tile_cache)
    print(f"Using tile cache directory: {cache_dir}")

    try:
        print("Pass 1: Scanning image tiles and caching flattened coefficients...")
        cache_summary = _cache_tiles(args, cache_dir, reader)
        if not cache_summary.entries:
            raise ValueError("No tiles were generated from the input image")

        global_initial_T = _calculate_initial_threshold_from_max(cache_summary.global_max_abs)
        quant_step_value, quant_enabled = _resolve_quantization_step(args, cache_summary.global_max_abs)

        if quant_enabled and quant_step_value > 0:
            print(f"  Quantization step (global): {quant_step_value:.6f}")
        else:
            print("  Quantization disabled (precision preserved, larger files expected)")

        print(f"  Global T derived from max |coeff|: {global_initial_T:.6f}")

        header_bytes = _pack_tiled_header(
            num_passes=args.num_passes,
            num_scales=args.scales,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            image_width=cache_summary.image_width,
            image_height=cache_summary.image_height,
            global_initial_T=global_initial_T,
            quantization_step=quant_step_value,
            quantization_enabled=quant_enabled,
            total_tiles=len(cache_summary.entries),
            wavelet=args.wavelet,
        )

        print("Pass 2: Compressing cached tiles...")
        encode_stats = _encode_tiles(
            cache_summary.entries,
            output_path,
            header_bytes,
            global_initial_T,
            quant_step_value,
            quant_enabled,
            args.num_passes,
        )

        return {
            "global_initial_T": global_initial_T,
            "quantization_step": quant_step_value if quant_enabled else 0.0,
            "quantization_enabled": quant_enabled and quant_step_value > 0,
            "image_width": cache_summary.image_width,
            "image_height": cache_summary.image_height,
            "tile_count": len(cache_summary.entries),
            "coeff_bytes": encode_stats["coeff_bytes"],
        }
    finally:
        if cleanup_cache:
            shutil.rmtree(cache_dir, ignore_errors=True)
        reader.close()


def main():
    parser = argparse.ArgumentParser(
        description="WDR Image Compression Pipeline (Tiled)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s input.png output.wdrt
  %(prog)s input.png output.wdrt --scales 2
  %(prog)s input.png output.wdrt --scales 2 --reconstructed reconstructed.png
  %(prog)s input.png output.wdrt --scales 3 --wavelet bior4.4 --tile-width 512 --tile-height 512
        """,
    )

    parser.add_argument("input_image", help="Input image file path")
    parser.add_argument("output_wdr", help="Output .wdrt file path")
    parser.add_argument(
        "--scales",
        type=int,
        default=2,
        help="Number of wavelet decomposition scales (default: 2, recommended: 2-3). Using scales >=6 introduces boundary artifacts per PyWavelets.",
    )
    parser.add_argument(
        "--reconstructed",
        type=str,
        default=None,
        help="Path to save reconstructed image (optional)",
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default="bior4.4",
        help="Wavelet name (default: bior4.4)",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=26,
        help="Number of bit-plane passes (default: 26 for high precision)",
    )
    parser.add_argument(
        "--quantization-step",
        type=float,
        default=None,
        help="Quantization step size (default: auto). Set to 0 to disable quantization entirely.",
    )
    parser.add_argument(
        "--quantization-method",
        type=str,
        default="threshold_based",
        choices=["threshold_based", "fixed_precision"],
        help="Method used when auto-calculating quantization step (default: threshold_based).",
    )
    parser.add_argument(
        "--tile-width",
        type=int,
        default=512,
        help="Tile width in pixels (default: 512)",
    )
    parser.add_argument(
        "--tile-height",
        type=int,
        default=512,
        help="Tile height in pixels (default: 512)",
    )
    parser.add_argument(
        "--tile-cache-dir",
        type=str,
        default=None,
        help="Optional directory for intermediate flattened tiles (useful for debugging). Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-tile-cache",
        action="store_true",
        help="Keep the temporary tile cache directory instead of deleting it after compression.",
    )
    parser.add_argument(
        "--allow-large-image",
        action="store_true",
        help="Opt-in to processing images larger than Pillow's default decompression guard after dimension validation.",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=None,
        help="Custom pixel limit to compare against before lifting the Pillow guard (use with --allow-large-image).",
    )
    parser.add_argument(
        "--tiff-reader",
        choices=["auto", "pillow", "tifffile"],
        default="auto",
        help="Backend to read tiles. 'auto' switches to tifffile for gigapixel TIFFs.",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Skip PSNR/MSE evaluation (useful when the source image cannot be reopened safely).",
    )

    args = parser.parse_args()

    input_path = Path(args.input_image).expanduser().resolve()
    args.input_image = str(input_path)
    width, height, _ = _guard_large_image(input_path, args.allow_large_image, args.max_image_pixels)
    pixel_count = width * height

    if args.tile_width <= 0 or args.tile_height <= 0:
        raise ValueError("Tile dimensions must be positive integers")

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = _resolve_output_path(args.output_wdr, DEFAULT_OUTPUT_DIR)
    args.output_wdr = str(output_path)

    if args.reconstructed:
        recon_path = _resolve_output_path(args.reconstructed, DEFAULT_OUTPUT_DIR)
        args.reconstructed = str(recon_path)

    print("=" * 60)
    print("WDR Image Compression Pipeline (Tiled)")
    print("=" * 60)
    print(f"Input image: {args.input_image}")
    print(f"Output file: {args.output_wdr}")
    print(f"Wavelet scales: {args.scales}")
    print(f"Wavelet: {args.wavelet}")
    print(f"Tile size: {args.tile_width} x {args.tile_height}")
    print()

    try:
        reader = create_tile_reader(
            input_path,
            args.tiff_reader,
            grayscale=True,
            pixel_count=pixel_count,
        )

        stats = run_tiled_compression(args, output_path, reader)

        compressed_size = os.path.getsize(args.output_wdr)
        raw_pixel_data_size = stats["image_width"] * stats["image_height"] * 8
        algo_cr = stats["coeff_bytes"] / compressed_size if compressed_size > 0 else 0
        benchmark_cr = (
            raw_pixel_data_size / compressed_size if compressed_size > 0 else 0
        )

        print("-" * 30)
        print(f"Tiles processed: {stats['tile_count']}")
        print(f"Compressed file size: {compressed_size:,} bytes")
        print(
            f"Algorithm CR (Quantized coeffs -> WDRT): {algo_cr:.3f}x"
        )
        print(f"True System CR (Pixels -> WDRT): {benchmark_cr:.3f}x")
        if stats["quantization_enabled"]:
            print(
                f"Quantization step used: {stats['quantization_step']:.6e}"
            )
        else:
            print("Quantization disabled")
        print("-" * 30)

        if benchmark_cr < 1:
            print(
                f"  ⚠️  WARNING: Compressed file is {((1 - benchmark_cr) * 100):.1f}% larger than the float64 pixel buffer"
            )
        print()

        if args.reconstructed:
            print("Decompressing tiled file for reconstruction...")
            reconstructed_img, header = _decompress_tiled_file(output_path)
            hlp.save_image(args.reconstructed, reconstructed_img)
            print(f"  Reconstructed image saved: {args.reconstructed}")

            if args.skip_metrics:
                print("Metrics skipped (--skip-metrics set).")
            else:
                print("Evaluating reconstruction quality (batched)...")
                # Use batched metrics to avoid loading entire images into memory
                original_reader = create_tile_reader(
                    input_path,
                    args.tiff_reader,
                    grayscale=True,
                    pixel_count=pixel_count,
                )
                # Create reader for reconstructed image (saved to disk)
                reconstructed_path = Path(args.reconstructed)
                reconstructed_reader = create_tile_reader(
                    reconstructed_path,
                    "pillow",  # Reconstructed images are saved as PNG/JPEG, use Pillow
                    grayscale=True,
                    pixel_count=pixel_count,
                )
                try:
                    mse, rmse, psnr = compute_batched_metrics(
                        original_reader,
                        reconstructed_reader,
                        args.tile_width,
                        args.tile_height,
                    )
                finally:
                    original_reader.close()
                    reconstructed_reader.close()

                print(f"  MSE: {mse:.6f}")
                print(f"  RMSE: {rmse:.6f}")
                print(f"  PSNR: {psnr:.2f} dB")
                if header["quantization_enabled"] and header["quantization_step"] > 0:
                    print(f"  Quantization step: {header['quantization_step']:.6e}")
                print()

        print("=" * 60)
        print("Pipeline completed successfully!")
        print("=" * 60)

    except FileNotFoundError as e:
        print(f"Error: File not found: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: Invalid value: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: Runtime error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Unexpected error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
