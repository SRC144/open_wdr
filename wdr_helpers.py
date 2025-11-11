"""
Python helper functions for WDR image compression pipeline.

This module provides functions for:
- Image I/O (load_image, save_image): Load and save grayscale images
- Discrete Wavelet Transform (do_dwt, do_idwt): Perform forward and inverse DWT
- Coefficient flattening/unflattening (flatten_coeffs, unflatten_coeffs): 
  Convert wavelet coefficient structures to/from 1D arrays using WDR scanning order

The flattening/unflattening functions implement the WDR-specific "coarse-to-fine"
scanning order, which processes coefficients in order of importance for compression.

Scanning order: LL_N → HL_N → LH_N → HH_N → HL_{N-1} → ... → HH_1
- HL bands: scanned column-by-column (vertically)
- LL, LH, HH bands: scanned row-by-row (horizontally)

This scanning order is critical for the WDR algorithm's performance, as it ensures
that important coefficients (in lower frequency bands) are processed before less
important ones (in higher frequency bands).
"""

import numpy as np
import pywt
from PIL import Image
from pathlib import Path
from typing import Tuple, Dict, Any


def load_image(filepath: str) -> np.ndarray:
    """
    Load an image from a file and convert to grayscale NumPy array.
    
    This function loads an image file, converts it to grayscale if necessary,
    and returns it as a 2D NumPy array suitable for wavelet transformation.
    
    Args:
        filepath: Path to the image file (supports formats supported by PIL/Pillow)
        
    Returns:
        NumPy array of shape (height, width) with dtype float64.
        Values are in the range [0, 255].
        
    Raises:
        FileNotFoundError: If the image file does not exist
        ValueError: If the image cannot be loaded or is not a valid 2D image
        
    Example:
        >>> img = load_image("test.png")
        >>> print(img.shape)
        (256, 256)
        >>> print(img.dtype)
        float64
    """
    img = Image.open(filepath)
    
    # Convert to grayscale if multi-channel
    if img.mode != 'L':
        img = img.convert('L')
    
    # Convert to NumPy array
    img_array = np.array(img, dtype=np.float64)
    
    # Validate 2D array (single channel)
    if len(img_array.shape) != 2:
        raise ValueError(f"Expected 2D image, got shape {img_array.shape}")
    
    return img_array


def save_image(filepath: str, img_array: np.ndarray) -> None:
    """
    Save a NumPy array as an image file.
    
    This function saves a 2D NumPy array as a grayscale image file.
    Values are clipped to the valid range [0, 255] before saving.
    
    Args:
        filepath: Path where the image will be saved (format determined by extension)
        img_array: NumPy array of shape (height, width) with dtype float64 or uint8
        
    Raises:
        ValueError: If the image array is not 2D
        IOError: If the image cannot be saved
        
    Example:
        >>> img = np.array([[100, 200], [50, 150]], dtype=np.float64)
        >>> save_image("output.png", img)
    """
    # Clip values to valid range [0, 255]
    img_array = np.clip(img_array, 0, 255)
    
    # Convert to uint8 if needed
    if img_array.dtype != np.uint8:
        img_array = img_array.astype(np.uint8)
    
    # Create directory if it doesn't exist
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Create PIL Image and save
    img = Image.fromarray(img_array, mode='L')
    img.save(filepath)


def do_dwt(img_array: np.ndarray, scales: int = 2, wavelet: str = 'bior4.4') -> Tuple:
    """
    Perform 2D Discrete Wavelet Transform on an image.
    
    This function performs a multi-level 2D Discrete Wavelet Transform (DWT) on
    an image, decomposing it into frequency bands at different scales. The DWT
    is the first stage of the WDR compression pipeline.
    
    The function uses PyWavelets (pywt) to perform the transform. The default
    wavelet 'bior4.4' (biorthogonal 4.4) is commonly used in image compression
    due to its good compression performance and near-orthogonality.
    
    Args:
        img_array: Input image array of shape (height, width) with dtype float64
        scales: Number of decomposition scales (default: 2, recommended: 2-3).
                Note: scales=6+ introduces boundary artifacts as warned by PyWavelets,
                so scales=2-3 is recommended for practical use.
        wavelet: Wavelet name (default: 'bior4.4'). Must be a valid PyWavelets wavelet.
        
    Returns:
        Coefficient structure compatible with pywt.waverec2().
        Structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
        Where cA = LL (approximation), cH = HL (horizontal detail),
        cV = LH (vertical detail), cD = HH (diagonal detail)
        
    Raises:
        ValueError: If scales <= 0 or if the wavelet name is invalid
        
    Example:
        >>> img = load_image("test.png")
        >>> coeffs = do_dwt(img, scales=2, wavelet='bior4.4')
        >>> print(len(coeffs))
        3
    """
    if scales <= 0:
        raise ValueError(f"scales must be > 0, got {scales}")
    
    # Perform 2D DWT
    coeffs = pywt.wavedec2(img_array, wavelet, mode='sym', level=scales)
    
    return coeffs


def do_idwt(coeffs: Tuple, wavelet: str = 'bior4.4') -> np.ndarray:
    """
    Perform 2D Inverse Discrete Wavelet Transform.
    
    This function performs the inverse DWT (IDWT) to reconstruct an image from
    wavelet coefficients. It is the inverse operation of do_dwt() and is used
    in the decompression pipeline to reconstruct the image after WDR decompression.
    
    Args:
        coeffs: Coefficient structure from pywt.wavedec2() or unflatten_coeffs().
                Must be compatible with pywt.waverec2().
        wavelet: Wavelet name (default: 'bior4.4'). Must match the wavelet used
                 in the forward DWT.
        
    Returns:
        Reconstructed image array of shape (height, width) with dtype float64.
        Values are in the range [0, 255] (may exceed due to DWT precision).
        
    Raises:
        ValueError: If the coefficient structure is invalid or the wavelet name is invalid
        
    Example:
        >>> coeffs = do_dwt(img, scales=2)
        >>> reconstructed = do_idwt(coeffs, wavelet='bior4.4')
        >>> print(reconstructed.shape)
        (256, 256)
    """
    # Perform 2D IDWT
    img_array = pywt.waverec2(coeffs, wavelet, mode='sym')
    
    return img_array


def flatten_coeffs(coeffs: Tuple) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Flatten wavelet coefficients into a 1D array using WDR scanning order.
    
    This function converts a wavelet coefficient structure into a 1D array
    following the WDR-specific "coarse-to-fine" scanning order. This scanning
    order is critical for the WDR algorithm, as it processes coefficients in
    order of importance (lower frequency bands first).
    
    Scanning order: LL_N → HL_N → LH_N → HH_N → HL_{N-1} → ... → HH_1
    - HL bands: scanned column-by-column (vertically, transpose then row-major)
    - LL, LH, HH bands: scanned row-by-row (horizontally, standard row-major)
    
    The function also generates shape metadata that is required for unflattening
    the coefficients back to the original structure.
    
    Args:
        coeffs: Coefficient structure from pywt.wavedec2().
                Structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
        
    Returns:
        Tuple of (flat_array, shape_metadata):
        - flat_array: 1D NumPy array of all coefficients (dtype float64)
        - shape_metadata: Dictionary containing:
            - 'subbands': List of subband information (name, shape, scan_order, indices)
            - 'wavelet': Wavelet name (None, to be set by caller)
            - 'mode': DWT mode ('sym')
        
    Raises:
        ValueError: If the coefficient structure is empty or invalid
        
    Example:
        >>> coeffs = do_dwt(img, scales=2)
        >>> flat_coeffs, shape_data = flatten_coeffs(coeffs)
        >>> print(flat_coeffs.shape)
        (65536,)
        >>> print(len(shape_data['subbands']))
        7  # LL_2, HL_2, LH_2, HH_2, HL_1, LH_1, HH_1
        
    See Also:
        unflatten_coeffs: Inverse operation to reconstruct coefficient structure
    """
    flat_list = []
    shape_metadata = {
        'subbands': [],
        'wavelet': None,  # Will be set by caller if needed
        'mode': 'sym'
    }
    
    # Extract coefficients
    # coeffs structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
    # Where cA = LL, cH = HL, cV = LH, cD = HH
    
    if len(coeffs) == 0:
        raise ValueError("Empty coefficient structure")
    
    # Get the approximation coefficients (LL_N)
    cA = coeffs[0]
    level = len(coeffs) - 1  # Level N (0-indexed: level N-1 in array)
    
    # Scan LL_N (row-by-row)
    ll_flat = cA.flatten('C')  # Row-major (C-style)
    start_idx = len(flat_list)
    flat_list.extend(ll_flat)
    shape_metadata['subbands'].append({
        'name': f'LL_{level}',
        'shape': cA.shape,
        'start_idx': start_idx,
        'end_idx': len(flat_list),
        'scan_order': 'row'
    })
    
    # Scan detail coefficients for each level (from level N down to level 1)
    for i in range(1, len(coeffs)):
        level = len(coeffs) - i  # Current level (N, N-1, ..., 1)
        cH, cV, cD = coeffs[i]  # HL, LH, HH
        
        # Scan HL (column-by-column: transpose then row-major)
        hl_flat = cH.T.flatten('C')  # Transpose, then row-major
        start_idx = len(flat_list)
        flat_list.extend(hl_flat)
        shape_metadata['subbands'].append({
            'name': f'HL_{level}',
            'shape': cH.shape,
            'start_idx': start_idx,
            'end_idx': len(flat_list),
            'scan_order': 'column'
        })
        
        # Scan LH (row-by-row)
        lh_flat = cV.flatten('C')  # Row-major
        start_idx = len(flat_list)
        flat_list.extend(lh_flat)
        shape_metadata['subbands'].append({
            'name': f'LH_{level}',
            'shape': cV.shape,
            'start_idx': start_idx,
            'end_idx': len(flat_list),
            'scan_order': 'row'
        })
        
        # Scan HH (row-by-row)
        hh_flat = cD.flatten('C')  # Row-major
        start_idx = len(flat_list)
        flat_list.extend(hh_flat)
        shape_metadata['subbands'].append({
            'name': f'HH_{level}',
            'shape': cD.shape,
            'start_idx': start_idx,
            'end_idx': len(flat_list),
            'scan_order': 'row'
        })
    
    # Convert to NumPy array
    flat_array = np.array(flat_list, dtype=np.float64)
    
    return flat_array, shape_metadata


def unflatten_coeffs(flat_array: np.ndarray, shape_metadata: Dict[str, Any]) -> Tuple:
    """
    Unflatten a 1D array back into wavelet coefficient structure.
    
    This function is the inverse of flatten_coeffs(). It reconstructs the
    original wavelet coefficient structure from a 1D array using the shape
    metadata generated during flattening.
    
    The function correctly handles the WDR scanning order, reversing the
    column-by-column scanning for HL bands and row-by-row scanning for
    LL, LH, and HH bands.
    
    Args:
        flat_array: 1D NumPy array of coefficients (dtype float64).
                    Must match the shape of the array produced by flatten_coeffs().
        shape_metadata: Dictionary with subband information from flatten_coeffs().
                        Must contain 'subbands' key with subband information.
        
    Returns:
        Coefficient structure compatible with pywt.waverec2().
        Structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
        
    Raises:
        ValueError: If the flat_array size doesn't match the metadata,
                    or if the coefficient structure is incomplete
        
    Example:
        >>> flat_coeffs, shape_data = flatten_coeffs(coeffs)
        >>> # ... compress and decompress flat_coeffs ...
        >>> reconstructed_coeffs = unflatten_coeffs(decompressed_flat_coeffs, shape_data)
        >>> reconstructed_img = do_idwt(reconstructed_coeffs)
        
    See Also:
        flatten_coeffs: Forward operation to flatten coefficient structure
    """
    coeffs_list = []
    subbands = shape_metadata['subbands']
    
    # Reconstruct coefficients
    for subband_info in subbands:
        name = subband_info['name']
        shape = subband_info['shape']
        start_idx = subband_info['start_idx']
        end_idx = subband_info['end_idx']
        scan_order = subband_info['scan_order']
        
        # Extract coefficients for this subband
        subband_flat = flat_array[start_idx:end_idx]
        
        # Reshape based on scan order
        if scan_order == 'column':
            # Column-by-column: reshape to transposed shape, then transpose back
            subband_2d = subband_flat.reshape(shape[1], shape[0])  # Transposed shape
            subband_2d = subband_2d.T  # Transpose back
        else:  # row
            # Row-by-row: standard reshape
            subband_2d = subband_flat.reshape(shape)
        
        # Add to coefficient list
        coeffs_list.append(subband_2d)
    
    # Reconstruct pywt coefficient structure
    # Structure: [cA_n, (cH_n, cV_n, cD_n), (cH_{n-1}, cV_{n-1}, cD_{n-1}), ..., (cH_1, cV_1, cD_1)]
    coeffs = [coeffs_list[0]]  # LL_N
    
    # Group detail coefficients by level
    i = 1
    while i < len(coeffs_list):
        # Each level has 3 detail bands: HL, LH, HH
        if i + 2 < len(coeffs_list):
            cH = coeffs_list[i]      # HL
            cV = coeffs_list[i + 1]  # LH
            cD = coeffs_list[i + 2]  # HH
            coeffs.append((cH, cV, cD))
            i += 3
        elif i + 2 == len(coeffs_list):
            # Last level - we have exactly 3 bands
            cH = coeffs_list[i]      # HL
            cV = coeffs_list[i + 1]  # LH
            cD = coeffs_list[i + 2]  # HH
            coeffs.append((cH, cV, cD))
            break
        else:
            raise ValueError(f"Invalid coefficient structure: incomplete level at index {i}")
    
    return tuple(coeffs)


def quantize_coeffs(coeffs: np.ndarray, step_size: float) -> Tuple[np.ndarray, float]:
    """
    Quantize coefficients to create redundancy for compression (optional).
    
    This function applies uniform quantization to coefficients, rounding them
    to the nearest multiple of step_size. This creates repeated values that
    arithmetic coding can exploit for better compression.
    
    Quantization is **optional** but recommended for WDR compression efficiency.
    Without quantization, float64 coefficients have high precision with mostly
    unique values, preventing arithmetic coding from finding redundancy. This
    can result in compressed files larger than the original.
    
    With quantization, compression ratios of 1.5-2.5x are typical with acceptable
    quality loss (PSNR > 40 dB).
    
    Args:
        coeffs: Input coefficients (1D array, float64)
        step_size: Quantization step size. Larger step_size creates more
                   redundancy but introduces more quantization error.
                   Must be > 0. Set quantization_step=0 in main.py to disable.
        
    Returns:
        Tuple of (quantized_coeffs, step_size):
        - quantized_coeffs: Quantized coefficients (float64, but with reduced precision)
        - step_size: Quantization step size (for reconstruction/dequantization)
        
    Example:
        >>> coeffs = np.array([1.23, 4.56, 7.89], dtype=np.float64)
        >>> quantized, step = quantize_coeffs(coeffs, step_size=0.5)
        >>> print(quantized)
        [1.0, 4.5, 8.0]
        >>> print(step)
        0.5
        
    Note:
        This function is part of the compression pipeline optimization.
        Quantization strategies and step size calculation are under active development.
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    
    # Uniform quantization: round to nearest multiple of step_size
    quantized = np.round(coeffs / step_size) * step_size
    
    return quantized, step_size


def dequantize_coeffs(quantized_coeffs: np.ndarray, step_size: float) -> np.ndarray:
    """
    Dequantize coefficients (identity for uniform quantization).
    
    For uniform quantization, dequantization is an identity operation since
    the quantized values are already in the correct range. This function
    exists for API consistency and potential future non-uniform quantization.
    
    Args:
        quantized_coeffs: Quantized coefficients (1D array, float64)
        step_size: Quantization step size (used for verification, not needed for uniform quantization)
        
    Returns:
        Dequantized coefficients (same as quantized for uniform quantization)
        
    Example:
        >>> quantized = np.array([1.0, 4.5, 8.0], dtype=np.float64)
        >>> dequantized = dequantize_coeffs(quantized, step_size=0.5)
        >>> np.array_equal(quantized, dequantized)
        True
    """
    if step_size <= 0:
        raise ValueError(f"step_size must be > 0, got {step_size}")
    
    # For uniform quantization, dequantization is identity
    return quantized_coeffs.copy()


def calculate_quantization_step(coeffs: np.ndarray, num_passes: int = 26, method: str = 'threshold_based', compression_focused: bool = True) -> float:
    """
    Calculate quantization step size based on coefficient statistics.
    
    This function provides different strategies for calculating quantization step size:
    - 'threshold_based': Uses WDR's initial threshold approach (recommended)
    - 'fixed_precision': Uses a fixed precision value
    - 'adaptive': Uses different step sizes for different coefficient ranges
    
    For compression efficiency, use a larger step size that creates more redundancy.
    The step size can be aligned with the initial threshold or scaled for better compression.
    
    Args:
        coeffs: Input coefficients (1D array, float64)
        num_passes: Number of bit-plane passes (default: 26)
        method: Method for calculating step size:
                - 'threshold_based': step based on initial threshold (recommended)
                - 'fixed_precision': step = 0.01 (fixed)
                - 'adaptive': step based on coefficient magnitude (not implemented)
        compression_focused: If True, use a larger step size for better compression
                            (default: True). If False, use fine quantization aligned
                            with refinement precision.
        
    Returns:
        Quantization step size (float)
        
    Example:
        >>> coeffs = np.array([100.0, -42.0, 10.0, 0.0, 3.0], dtype=np.float64)
        >>> step = calculate_quantization_step(coeffs, num_passes=26, method='threshold_based')
        >>> print(f"Step size: {step:.6f}")
        Step size: 0.007812
    """
    if method == 'threshold_based':
        # Calculate initial threshold T (same as WDR compressor)
        max_abs = np.max(np.abs(coeffs))
        if max_abs == 0.0:
            return 1.0  # Default step for all-zero coefficients
        
        # Find the largest power of 2 such that max_abs < 2*T and max_abs >= T
        initial_T = 2.0 ** np.floor(np.log2(max_abs))
        
        # Ensure max_abs >= T
        if max_abs < initial_T:
            initial_T = initial_T / 2.0
        
        # Ensure max_abs < 2*T
        if max_abs >= 2.0 * initial_T:
            initial_T = initial_T * 2.0
        
        if compression_focused:
            # For compression efficiency, use a larger step size
            # Use step = T / (2^N) where N is smaller (e.g., 6-10) to create more redundancy
            # This creates quantized values that align with WDR's threshold levels
            # Typical: T / 64 or T / 128 for good compression with reasonable quality
            quantization_levels = 64  # Creates 64 quantization levels per threshold interval
            step_size = initial_T / quantization_levels
            
            # Alternative: Use step = T / (2^log2(num_coeffs/desired_unique_ratio))
            # But for simplicity, use fixed quantization levels
        else:
            # Fine quantization aligned with refinement precision
            # Calculate step size: T / (2^num_passes)
            # This aligns quantization precision with WDR's refinement precision
            step_size = initial_T / (2.0 ** num_passes)
        
        return step_size
    
    elif method == 'fixed_precision':
        # Fixed precision quantization
        # For compression, use a step that creates redundancy
        # Typical values: 0.1, 0.5, 1.0 depending on coefficient range
        max_abs = np.max(np.abs(coeffs))
        if max_abs > 100:
            return 1.0  # Larger step for large coefficients
        elif max_abs > 10:
            return 0.1  # Medium step for medium coefficients
        else:
            return 0.01  # Smaller step for small coefficients
    
    elif method == 'adaptive':
        # Adaptive quantization (not implemented)
        raise NotImplementedError("Adaptive quantization method not yet implemented")
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'threshold_based', 'fixed_precision', or 'adaptive'")
