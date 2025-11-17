import io
from pathlib import Path

import numpy as np
import pytest

from main import (
    TILE_CHUNK_STRUCT,
    TileCacheEntry,
    _calculate_initial_threshold_from_max,
    _pack_tile_chunk_header,
    _pack_tiled_header,
    _read_tiled_header,
)
from wdr.utils.tile_reader import decide_reader_backend, DEFAULT_AUTO_THRESHOLD


def test_calculate_initial_threshold_from_max_matches_cpp_logic():
    cases = [
        (0.0, 1.0),
        (0.75, 0.5),
        (1.0, 1.0),
        (3.14, 2.0),
        (4.0, 4.0),
        (5.0, 4.0),
        (16.0, 16.0),
    ]
    for max_abs, expected in cases:
        assert _calculate_initial_threshold_from_max(max_abs) == expected


def test_tiled_header_round_trip_preserves_metadata():
    header_bytes = _pack_tiled_header(
        num_passes=26,
        num_scales=3,
        tile_width=512,
        tile_height=512,
        image_width=2048,
        image_height=1024,
        global_initial_T=8.0,
        quantization_step=0.125,
        quantization_enabled=True,
        total_tiles=16,
        wavelet="bior4.4",
    )

    stream = io.BytesIO(header_bytes)
    parsed = _read_tiled_header(stream)

    assert parsed["num_passes"] == 26
    assert parsed["num_scales"] == 3
    assert parsed["tile_width"] == 512
    assert parsed["tile_height"] == 512
    assert parsed["image_width"] == 2048
    assert parsed["image_height"] == 1024
    assert parsed["global_initial_T"] == 8.0
    assert parsed["quantization_enabled"] is True
    assert np.isclose(parsed["quantization_step"], 0.125)
    assert parsed["total_tiles"] == 16
    assert parsed["wavelet"] == "bior4.4"


def test_tile_chunk_header_layout_matches_struct():
    entry = TileCacheEntry(
        index=5,
        origin_x=128,
        origin_y=64,
        width=512,
        height=480,
        coeff_path=Path("dummy.npy"),  # Not needed for header packing
        coeff_count=4096,
    )

    payload_size = 12345
    packed = _pack_tile_chunk_header(entry, payload_size)
    unpacked = TILE_CHUNK_STRUCT.unpack(packed)

    assert unpacked[0] == entry.index
    assert unpacked[1] == entry.origin_x
    assert unpacked[2] == entry.origin_y
    assert unpacked[3] == entry.coeff_count
    assert unpacked[4] == entry.width
    assert unpacked[5] == entry.height
    assert unpacked[6] == payload_size


def test_decide_reader_backend_auto_threshold():
    assert (
        decide_reader_backend("auto", DEFAULT_AUTO_THRESHOLD + 1)
        == "tifffile"
    )
    assert (
        decide_reader_backend("auto", DEFAULT_AUTO_THRESHOLD - 1)
        == "pillow"
    )


def test_decide_reader_backend_explicit():
    assert decide_reader_backend("pillow", 10) == "pillow"
    assert decide_reader_backend("tifffile", 10) == "tifffile"
    with pytest.raises(ValueError):
        decide_reader_backend("unknown", 10)

