"""
Tile reader abstractions for feeding the tiled WDR pipeline.

The default reader uses Pillow, which is sufficient for moderate-size images.
For gigapixel TIFFs, Pillow/libtiff tends to decode entire strips when cropping,
so we provide a Tifffile-backed reader that can pull grayscale windows without
loading the whole file into memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple
import warnings

import numpy as np
from PIL import Image

DEFAULT_AUTO_THRESHOLD = 250_000_000  # ~250 MP
_GRAY_COEFFS = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)


class BaseTileReader:
    """Abstract base reader."""

    def size(self) -> Tuple[int, int]:
        raise NotImplementedError

    def read_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


def _to_float64(tile: np.ndarray, grayscale: bool) -> np.ndarray:
    arr = np.asarray(tile)
    if grayscale:
        if arr.ndim == 3:
            channels = arr.shape[-1]
            if channels >= 3:
                arr = np.tensordot(arr[..., :3], _GRAY_COEFFS, axes=([-1], [0]))
            elif channels == 1:
                arr = arr[..., 0]
    return arr.astype(np.float64, copy=False)


class PillowTileReader(BaseTileReader):
    def __init__(self, path: Path, grayscale: bool):
        self._image = Image.open(path)
        self._grayscale = grayscale
        self._width, self._height = self._image.size

    def size(self) -> Tuple[int, int]:
        return self._width, self._height

    def read_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        box = (x, y, x + width, y + height)
        region = self._image.crop(box)
        if self._grayscale and region.mode != "L":
            region = region.convert("L")
        return _to_float64(np.array(region), self._grayscale)

    def close(self) -> None:
        if self._image:
            self._image.close()
            self._image = None


class TifffileTileReader(BaseTileReader):
    def __init__(self, path: Path, grayscale: bool):
        try:
            import tifffile  # type: ignore
        except ImportError as exc:  # pragma: no cover - exercised in runtime env
            raise ImportError(
                "tifffile is required for --tiff-reader=tifffile"
            ) from exc
        try:
            import zarr  # noqa: F401  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "zarr is required for tifffile-backed tile reading"
            ) from exc

        self._tiff = tifffile.TiffFile(path)
        series = self._tiff.series[0]
        self._zarr = series.asarray()
        shape = self._zarr.shape
        if len(shape) == 2:
            self._height, self._width = int(shape[0]), int(shape[1])
        elif len(shape) == 3:
            self._height, self._width = int(shape[0]), int(shape[1])
        else:
            raise ValueError("Unsupported TIFF shape for tiled reading.")
        self._grayscale = grayscale

    def size(self) -> Tuple[int, int]:
        return self._width, self._height

    def read_block(self, x: int, y: int, width: int, height: int) -> np.ndarray:
        y1 = y + height
        x1 = x + width
        block = self._zarr[y:y1, x:x1]
        return _to_float64(block, self._grayscale)

    def close(self) -> None:
        if self._tiff:
            self._tiff.close()
            self._tiff = None


def decide_reader_backend(preference: str, pixel_count: int, auto_threshold: int = DEFAULT_AUTO_THRESHOLD) -> str:
    pref = preference.lower()
    if pref not in {"auto", "pillow", "tifffile"}:
        raise ValueError(f"Unknown reader preference '{preference}'.")
    if pref == "tifffile":
        return "tifffile"
    if pref == "pillow":
        return "pillow"
    # auto
    return "tifffile" if pixel_count >= auto_threshold else "pillow"


def create_tile_reader(
    path: Path,
    preference: str,
    grayscale: bool,
    pixel_count: int,
    auto_threshold: int = DEFAULT_AUTO_THRESHOLD,
) -> BaseTileReader:
    backend = decide_reader_backend(preference, pixel_count, auto_threshold)
    if backend == "tifffile":
        try:
            return TifffileTileReader(path, grayscale)
        except ImportError as exc:
            if preference.lower() == "tifffile":
                raise
            warnings.warn(
                "tifffile not available, falling back to Pillow reader. "
                "Install 'tifffile' and 'zarr' for better gigapixel support.",
                RuntimeWarning,
            )
    return PillowTileReader(path, grayscale)

