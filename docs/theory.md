# WDR Algorithm Theory

[English](theory.md) | [Español](theory.es.md) | [Back to README](../README.md)

## Table of Contents

1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [WDR Algorithm Details](#wdr-algorithm-details)
4. [Adaptive Arithmetic Coding](#adaptive-arithmetic-coding)
5. [File Format](#file-format)
6. [References & Attribution](#references--attribution)

## Introduction

The Wavelet Difference Reduction (WDR) method is an embedded image compression technique that combines the power of wavelet transforms with efficient index coding and progressive transmission. Unlike tree-based methods like EZW (Embedded Zerotree Wavelet) or SPIHT (Set Partitioning in Hierarchical Trees), WDR uses a direct approach to find and encode the positions of significant wavelet coefficients.

### Key Advantages

- **Embedded Bitstream**: The compressed data is fully embedded, meaning the decoder can stop at any point to meet a target bit rate or distortion level
- **Progressive Transmission**: Images can be transmitted and displayed progressively, with quality improving as more bits are received
- **Lossless Compression**: With sufficient passes, the algorithm can achieve lossless compression
- **No Training Required**: The adaptive arithmetic coding stage requires no prior training or model parameters

### Algorithm Overview

The WDR algorithm consists of three main stages:

1. **Discrete Wavelet Transform (DWT)**: Transforms the image into frequency domain using wavelets
2. **Index Coding**: Encodes the positions of significant coefficients using differential coding and binary reduction
3. **Adaptive Arithmetic Coding**: Compresses the resulting symbol stream using adaptive arithmetic coding

## Mathematical Foundations

### Discrete Wavelet Transform (DWT)

The Discrete Wavelet Transform decomposes an image into multiple frequency bands at different scales. For a 2D image, the DWT produces:

- **LL (Low-Low)**: Approximation coefficients (low frequency in both directions)
- **HL (High-Low)**: Horizontal detail coefficients (high frequency horizontally, low frequency vertically)
- **LH (Low-High)**: Vertical detail coefficients (low frequency horizontally, high frequency vertically)
- **HH (High-High)**: Diagonal detail coefficients (high frequency in both directions)

This decomposition can be repeated on the LL band to create multiple scales (levels) of decomposition.

### Wavelet Basis Functions

Wavelets are mathematical functions that localize energy in both time (or space) and frequency domains. This dual localization makes them ideal for representing transient signals like edges and textures in images.

### Multi-Resolution Analysis

The DWT provides a multi-resolution representation of the image, where:
- Coarse scales capture large-scale features and smooth regions
- Fine scales capture details, edges, and textures

This multi-resolution structure is exploited by WDR through its "coarse-to-fine" scanning order.

## WDR Algorithm Details

### Scanning Order

The WDR algorithm processes wavelet coefficients in a specific "coarse-to-fine" scanning order. This order ensures that important coefficients (those in lower frequency bands) are processed and transmitted before less important ones.

#### Scanning Sequence

For an N-level DWT decomposition, the scanning order is:

```
LL_N → HL_N → LH_N → HH_N → HL_{N-1} → LH_{N-1} → HH_{N-1} → ... → HL_1 → LH_1 → HH_1
```

#### Scanning Heuristics

- **HL subbands**: Scanned **column-by-column** (vertically)
- **LL, LH, HH subbands**: Scanned **row-by-row** (horizontally)

This scanning order is critical for the algorithm's performance, as it ensures that coefficients are processed in order of importance.

#### Diagram: WDR Scanning Order

```
3-Level DWT Decomposition:

┌─────────┬─────────┬─────────┬─────────┐
│   LL_3  │   HL_3  │         │         │
│         │    ↓    │         │         │
│         │    ↓    │   LH_3  │   HH_3  │
├─────────┼─────────┤    →    │    →    │
│   HL_2  │         │         │         │
│    ↓    │         │         │         │
│    ↓    │         │         │         │
├─────────┼─────────┼─────────┼─────────┤
│         │   LH_2  │   HH_2  │         │
│         │    →    │    →    │         │
│         │         │         │         │
├─────────┴─────────┴─────────┴─────────┤
│   HL_1  │   LH_1  │   HH_1  │         │
│    ↓    │    →    │    →    │         │
└─────────┴─────────┴─────────┴─────────┘

Scanning Order: 
  1. LL_3 (row-by-row: →)
  2. HL_3 (column-by-column: ↓)
  3. LH_3 (row-by-row: →)
  4. HH_3 (row-by-row: →)
  5. HL_2 (column-by-column: ↓)
  6. LH_2 (row-by-row: →)
  7. HH_2 (row-by-row: →)
  8. HL_1 (column-by-column: ↓)
  9. LH_1 (row-by-row: →)
 10. HH_1 (row-by-row: →)

Legend: 
  ↓ = column-by-column (vertical scan)
  → = row-by-row (horizontal scan)
```

### Sorting Pass

The sorting pass identifies coefficients that become "significant" (their absolute value exceeds the current threshold T) and encodes their positions.

#### Process

1. **Find Significant Coefficients**: Iterate through the ICS (Insignificant Coefficient Set) and identify coefficients where |x| ≥ T
2. **Store Indices**: Store the indices of significant coefficients in list P
3. **Store Signs**: Store the signs of significant coefficients
4. **Differential Coding**: Encode the indices using differential coding
5. **Binary Reduction**: Apply binary reduction to the differential indices
6. **Encode**: Encode the binary-reduced indices and signs using arithmetic coding
7. **Update ICS**: Remove significant coefficients from ICS and re-enumerate

#### Diagram: Sorting Pass Flowchart

```mermaid
flowchart TD
    A[Start: ICS with coefficients] --> B[Iterate through ICS]
    B --> C{Is |x| >= T?}
    C -->|Yes| D[Add index to P]
    C -->|No| E[Keep in ICS]
    D --> F[Store sign]
    F --> G[Move to TPS]
    E --> H{More coefficients?}
    G --> H
    H -->|Yes| B
    H -->|No| I[Apply Differential Coding to P]
    I --> J[Apply Binary Reduction]
    J --> K[Encode with Arithmetic Coding]
    K --> L[Remove from ICS]
    L --> M[Move TPS to SCS]
    M --> N[End]
```

#### Sorting Pass Example

**Input:**
- ICS: `[10, -5, 35, 8, -42, 3]`
- T = 32

**Process:**
1. Find significant: indices 2 (35) and 4 (-42)
2. P = `[2, 4]`, signs = `[1, 0]` (positive, negative)
3. Differential coding: P' = `[2, 2]` (4 - 2 = 2)
4. Binary reduction: 
   - 2 = `10` → `0` (remove MSB)
   - 2 = `10` → `0` (remove MSB)
5. Encode: `0`, `1`, `0`, `0` (indices and signs interleaved)
6. Update ICS: Remove indices 2 and 4, re-enumerate → `[10, -5, 8, 3]`

#### Differential Coding

Differential coding encodes the differences between adjacent values in a monotonically increasing sequence.

**Example:**
- Original indices: `P = {1, 2, 5, 36, 42}`
- Differential encoding: `P' = {1, 1, 3, 31, 6}`

The first value remains unchanged, and each subsequent value is the difference from the previous one.

**Encoding Process:**
```
P[0] = 1  → P'[0] = 1          (first value unchanged)
P[1] = 2  → P'[1] = 2 - 1 = 1  (difference from previous)
P[2] = 5  → P'[2] = 5 - 2 = 3  (difference from previous)
P[3] = 36 → P'[3] = 36 - 5 = 31 (difference from previous)
P[4] = 42 → P'[4] = 42 - 36 = 6 (difference from previous)
```

**Decoding Process:** Reverse by taking the partial sum:
```
P'[0] = 1  → P[0] = 1
P'[1] = 1  → P[1] = 1 + 1 = 2
P'[2] = 3  → P[2] = 2 + 3 = 5
P'[3] = 31 → P[3] = 5 + 31 = 36
P'[4] = 6  → P[4] = 36 + 6 = 42
```

This encoding is efficient when indices are clustered together, as the differences are typically small.

#### Binary Reduction

Binary reduction represents a positive binary integer by removing the Most Significant Bit (MSB). This reduces the number of bits needed to represent the number, as the MSB is always '1' for positive numbers.

**Example:**
- Number: `19` in binary is `10011`
- Binary reduction: Remove MSB → `0011`

**Encoding Process:**
```
Value: 19
Binary: 10011
        ^
        MSB (always 1 for positive numbers)

Remove MSB: 0011
```

**Decoding Process:** Reverse by prepending a '1' as the MSB:
```
Reduced: 0011
Prepend '1': 10011
Convert to decimal: 19
```

**Visual Example:**
```
Original:  19 = 10011 (5 bits)
            ^
            MSB
Reduced:   0011 (4 bits)
            ^
            Prepend '1' to decode

This saves 1 bit per number (20% reduction for 5-bit numbers).
```

In WDR, the sign of the coefficient is used as a delimiter between reduced indices in the bitstream, allowing the decoder to know where one index ends and the next begins.

### Refinement Pass

The refinement pass adds one bit of precision to coefficients that were already found significant in previous passes.

#### Process

For each coefficient in the SCS (Significant Coefficient Set):

1. **Calculate Interval**: 
   - `low = center - T`
   - `high = center + T`

2. **Determine Bit**:
   - If the true value is in the upper half `[center, high)`: output bit '1'
   - If the true value is in the lower half `[low, center)`: output bit '0'

3. **Update Center**:
   - If bit is '1': `center = (center + high) / 2`
   - If bit is '0': `center = (low + center) / 2`

4. **Encode Bit**: Encode the bit using arithmetic coding with the refinement model

This process narrows the interval containing the true coefficient value, adding one bit of precision per pass.

#### Diagram: Refinement Pass

```
Pass 0: Coefficient found significant at T = 32
  True value: x = 49
  Interval: [32, 64)
  Initial center: 48 (1.5*T)
  
Pass 1: T = 16
  Current center: 48
  Interval: [32, 64) = [48-16, 48+16)
  ┌─────────────────────────────────────┐
  │ [32)        [48)        [64)        │
  │   │──────────┼──────────│           │
  │   │  lower   │  upper   │           │
  │   │  half    │  half    │           │
  │   └──────────┴──────────┘           │
  │           x=49 is here → output '1' │
  └─────────────────────────────────────┘
  New center: (48 + 64) / 2 = 56
  
Pass 2: T = 8
  Current center: 56
  Interval: [48, 64) = [56-8, 56+8)
  ┌─────────────────────────────────────┐
  │ [48)        [56)        [64)        │
  │   │──────────┼──────────│           │
  │   │  lower   │  upper   │           │
  │   └──────────┴──────────┘           │
  │      x=49 is here → output '0'      │
  └─────────────────────────────────────┘
  New center: (48 + 56) / 2 = 52
  
And so on... The interval narrows with each pass.
```

### List Management

The WDR algorithm maintains three key data structures that track coefficients throughout the compression process:

#### ICS (Insignificant Coefficient Set)
- Contains coefficients that are not yet significant (|x| < T)
- Initially contains all coefficients in scanning order
- Shrinks as coefficients become significant and are moved to SCS
- Coefficients are re-enumerated after each pass to maintain sequential indexing

#### SCS (Significant Coefficient Set)
- Contains coefficients that are significant (|x| ≥ T)
- Stores tuples of `(value, center)` where:
  - `value`: The original coefficient value (or current approximation)
  - `center`: The current reconstruction center (used for refinement)
- Grows as coefficients become significant
- Coefficients in SCS are refined in each pass to add precision
- The center value is updated during refinement to narrow the interval

#### TPS (Temporary Pass Set)
- Contains coefficients that became significant in the current pass
- Used to transfer coefficients from ICS to SCS
- Cleared after each pass
- Stores the initial reconstruction value (center = T + T/2) for new significant coefficients

#### Diagram: List State Transitions

```
Initial State:
  ICS = [all coefficients]
  SCS = []
  TPS = []

After Sorting Pass (T = 32):
  ICS = [insignificant coefficients]
  SCS = []
  TPS = [new significant coefficients]

After Moving TPS to SCS:
  ICS = [insignificant coefficients]
  SCS = [significant coefficients with initial center]
  TPS = []

After Refinement Pass:
  ICS = [insignificant coefficients]
  SCS = [significant coefficients with refined center]
  TPS = []

After Next Sorting Pass (T = 16):
  ICS = [remaining insignificant coefficients]
  SCS = [previous significant coefficients]
  TPS = [new significant coefficients at T=16]

And so on...
```

### Main Loop

The WDR algorithm executes the following loop until the desired precision is reached:

1. **Sorting Pass**: Find and encode new significant coefficients
2. **Refinement Pass**: Refine existing significant coefficients
3. **Update Threshold**: Halve the threshold T (T = T / 2)
4. **Move to SCS**: Move coefficients from TPS to SCS
5. **Repeat**: Continue with the next pass

The threshold T is halved each pass, allowing the algorithm to progressively identify and refine coefficients at finer scales.

## Adaptive Arithmetic Coding

### Overview

The final stage of WDR compression uses adaptive arithmetic coding to compress the symbol stream generated by the sorting and refinement passes. This implementation uses the algorithm from Witten, Neal, and Cleary (1987).

### Algorithm Description

Arithmetic coding represents an entire message as a single fraction within the interval [0, 1). Each symbol narrows this interval based on its probability. The "adaptive" aspect means the probability model updates its estimates as it processes each symbol.

### Key Features

- **Integer Arithmetic**: Uses fixed-precision integer math to represent intervals, avoiding floating-point operations
- **Incremental Operation**: Outputs bits as soon as the most significant bits of the interval match
- **Underflow Handling**: Includes a mechanism to prevent loss of precision when the interval becomes very small

### Implementation Note

The C++ implementation in this project is based on the mathematical algorithm from Witten, Neal, & Cleary (1987). Full credit and references are provided in the code comments and documentation. The implementation maintains mathematical equivalence to the original algorithm while using modern C++17 features for clarity and type safety.

### Underflow Handling

When the encoding interval becomes very small but straddles the midpoint, the algorithm cannot output a bit immediately. Instead, it:
1. Tracks the number of "opposite bits" to output later
2. Scales the interval to expand it
3. Outputs the opposite bits when a bit can finally be output

This mechanism ensures that precision is maintained even when the interval is very small.

## File Format

### .wdr File Structure

A `.wdr` file consists of:

1. **Header**: Contains metadata about the compressed data
   - Initial threshold T
   - Number of passes
   - Number of coefficients
   - Size of compressed data

2. **Compressed Data**: The arithmetic-coded bitstream
   - Contains encoded sorting pass data (indices and signs)
   - Contains encoded refinement pass data (refinement bits)

### Header Format

The header is a binary structure containing:
- `double initial_T`: Initial threshold value
- `uint32_t num_passes`: Number of bit-plane passes
- `uint64_t num_coeffs`: Number of coefficients in the original array
- `uint64_t data_size`: Size of compressed data in bytes

### Bitstream Organization

The bitstream is organized as a sequence of passes:

```
Pass 0:
  - Sorting pass: count, indices, signs
  - Refinement pass: (empty, no coefficients in SCS yet)

Pass 1:
  - Sorting pass: count, indices, signs
  - Refinement pass: refinement bits for Pass 0 coefficients

Pass 2:
  - Sorting pass: count, indices, signs
  - Refinement pass: refinement bits for all previous coefficients

... and so on
```

## References & Attribution

### WDR Algorithm

This implementation is based on the Wavelet Difference Reduction algorithm for embedded image compression. The algorithm combines discrete wavelet transforms with efficient index coding and progressive transmission.

**Full Citation:**
[WDR paper citation - to be filled with actual paper details]

### Adaptive Arithmetic Coding

The adaptive arithmetic coding implementation is based on the algorithm from:

**Witten, I.H., Neal, R.M., & Cleary, J.G. (1987).** "Arithmetic coding for data compression." *Communications of the ACM*, 30(6), 520-540.

This paper presents the adaptive arithmetic coding algorithm used in the final compression stage of WDR. The C++ implementation maintains mathematical equivalence to the original algorithm while using modern C++17 features.

### Implementation Statement

This implementation is based on the theoretical sources cited above. Full credit is given to the original authors and researchers who developed these algorithms. The code comments and documentation include proper attribution to ensure academic integrity and give credit where it is due.

### Additional Resources

- PyWavelets: Python library for discrete wavelet transforms
- NumPy: Numerical computing library for Python
- Pillow: Python Imaging Library for image I/O

---

[English](theory.md) | [Español](theory.es.md) | [Back to README](../README.md)

