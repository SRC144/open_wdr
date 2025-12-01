import numpy as np
import pywt
import math
from typing import Tuple, Dict, Any, Generator

# --- Tiling Logic ---

def yield_tiles(image: np.ndarray, tile_size: int) -> Generator[np.ndarray, None, None]:
    """
    Generator that yields tiles (numpy arrays).
    Handles both RAM arrays (PIL) and Disk-backed arrays (memmap).
    """
    h, w = image.shape[:2] # Handle potential 3D shape (H, W, C) if memmap read RGB
    
    # Handle RGB memmaps: If input is (H, W, 3), we convert on the fly
    is_rgb = len(image.shape) == 3
    
    n_rows = (h + tile_size - 1) // tile_size
    n_cols = (w + tile_size - 1) // tile_size

    for row in range(n_rows):
        for col in range(n_cols):
            y = row * tile_size
            x = col * tile_size
            
            # 1. SLICE (This triggers the disk read for memmaps)
            # We convert to float64 immediately to detach from the memmap buffer
            chunk = image[y : min(y + tile_size, h), x : min(x + tile_size, w)]
            
            # # If RGB, convert to grayscale using standard luminosity weights
            # if is_rgb:
            #     # Y = 0.299R + 0.587G + 0.114B
            #     chunk = np.dot(chunk[...,:3], [0.299, 0.587, 0.114])

            # Ensure it's float64 for DWT
            tile = np.array(chunk, dtype=np.float64)
            
            # 2. PAD (If Edge Tile)
            th, tw = tile.shape
            if th < tile_size or tw < tile_size:
                pad_h = tile_size - th
                pad_w = tile_size - tw
                tile = np.pad(tile, ((0, pad_h), (0, pad_w)), mode='edge')
                
            yield tile

# --- Shared Logic ---

def scan_for_max_coefficient(image: np.ndarray, tile_size: int, scales: int, wavelet: str) -> float:
    global_max = 0.0
    # yield_tiles now handles the safe slicing from disk
    for tile in yield_tiles(image, tile_size):
        coeffs = do_dwt(tile, scales=scales, wavelet=wavelet)
        flat, _ = flatten_coeffs(coeffs)
        local_max = np.max(np.abs(flat))
        if local_max > global_max:
            global_max = local_max
    return global_max

def calculate_global_T(max_abs: float) -> float:
    if max_abs == 0.0: 
        return 1.0
    return 2.0 ** math.floor(math.log2(max_abs))

def do_dwt(img_array: np.ndarray, scales: int = 2, wavelet: str = 'bior4.4') -> Tuple:
    if scales <= 0: raise ValueError("scales > 0")
    return pywt.wavedec2(img_array, wavelet, mode='symmetric', level=scales)

def do_idwt(coeffs: Tuple, wavelet: str = 'bior4.4') -> np.ndarray:
    return pywt.waverec2(coeffs, wavelet, mode='symmetric')

def flatten_coeffs(coeffs: Tuple) -> Tuple[np.ndarray, Dict[str, Any]]:
    flat_list = []
    shape_metadata = {'subbands': [], 'wavelet': None, 'mode': 'sym'}
    if len(coeffs) == 0: raise ValueError("Empty coefficients")
    
    # LL
    cA = coeffs[0]
    level = len(coeffs) - 1
    flat_list.extend(cA.flatten('C'))
    shape_metadata['subbands'].append({
        'name': f'LL_{level}', 'shape': cA.shape, 
        'start_idx': 0, 'end_idx': len(flat_list), 'scan_order': 'row'
    })
    
    # Details
    for i in range(1, len(coeffs)):
        level = len(coeffs) - i
        cH, cV, cD = coeffs[i]
        
        # HL (Column)
        hl_flat = cH.T.flatten('C')
        s = len(flat_list)
        flat_list.extend(hl_flat)
        shape_metadata['subbands'].append({
            'name': f'HL_{level}', 'shape': cH.shape, 
            'start_idx': s, 'end_idx': len(flat_list), 'scan_order': 'column'
        })
        
        # LH (Row)
        lh_flat = cV.flatten('C')
        s = len(flat_list)
        flat_list.extend(lh_flat)
        shape_metadata['subbands'].append({
            'name': f'LH_{level}', 'shape': cV.shape, 
            'start_idx': s, 'end_idx': len(flat_list), 'scan_order': 'row'
        })
        
        # HH (Row)
        hh_flat = cD.flatten('C')
        s = len(flat_list)
        flat_list.extend(hh_flat)
        shape_metadata['subbands'].append({
            'name': f'HH_{level}', 'shape': cD.shape, 
            'start_idx': s, 'end_idx': len(flat_list), 'scan_order': 'row'
        })
        
    return np.array(flat_list, dtype=np.float64), shape_metadata

def unflatten_coeffs(flat_array: np.ndarray, shape_metadata: Dict[str, Any]) -> Tuple:
    coeffs_list = []
    subbands = shape_metadata['subbands']
    for meta in subbands:
        data = flat_array[meta['start_idx']:meta['end_idx']]
        if meta['scan_order'] == 'column':
            shape = meta['shape']
            data = data.reshape(shape[1], shape[0]).T
        else:
            data = data.reshape(meta['shape'])
        coeffs_list.append(data)
        
    coeffs = [coeffs_list[0]]
    i = 1
    while i < len(coeffs_list):
        if i + 2 <= len(coeffs_list):
            coeffs.append((coeffs_list[i], coeffs_list[i+1], coeffs_list[i+2]))
            i += 3
        else:
            break
    return tuple(coeffs)

def quantize_coeffs(coeffs: np.ndarray, step_size: float) -> Tuple[np.ndarray, float]:
    if step_size <= 0: raise ValueError("step > 0")
    return np.round(coeffs / step_size) * step_size, step_size

def dequantize_coeffs(coeffs: np.ndarray, step_size: float) -> np.ndarray:
    return coeffs.copy()