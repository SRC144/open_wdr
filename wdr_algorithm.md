# WDR Algorithm Context (`wdr_algorithm.md`)

## 1. Overview

The Wavelet Difference Reduction (WDR) method is an embedded image compression technique that consists of three main stages:
1.  Discrete Wavelet Transform
2.  [cite_start]Differential Coding [cite: 2511]
3.  [cite_start]Binary Reduction [cite: 2511]

[cite_start]Unlike tree-based methods like EZW [cite: 2512][cite_start], WDR uses a direct approach to find the positions of significant wavelet coefficients[cite: 2513]. [cite_start]The resulting bitstream is fully embedded, meaning it can be stopped at any point to meet a target bit rate or distortion, making it ideal for progressive image transmission[cite: 2514, 2515].

The WDR algorithm features:
* [cite_start]A **Discrete Wavelet Transform** to remove spatial and spectral redundancies[cite: 2525].
* [cite_start]**Index coding** (differential coding and binary reduction) to efficiently represent the positions of significant coefficients[cite: 2526].
* [cite_start]**Ordered bit-plane transmission** (successive approximation) to facilitate progressive transmission[cite: 2528].
* [cite_start]**Adaptive arithmetic coding** for final lossless compression, which requires no training[cite: 2529].

---

## 2. Theory Prerequisites

### Transient Signals
A wavelet transform is well-suited for image compression due to its good localization in both the spatial and frequency domains. [cite_start]This allows it to efficiently handle "transient signals"[cite: 2522]. In an image, transients are abrupt, short-lived changes like sharp edges and fine textures. Because wavelets can pinpoint *where* a specific frequency occurs, they can represent these information-rich transients very sparsely (with only a few large coefficients), leading to high compression ratios.

### Differential Coding
[cite_start]Differential Coding takes the difference between adjacent values in a monotonically increasing set of integers[cite: 2558].
* [cite_start]**Example:** An integer set `S = {1, 2, 5, 36, 42}` [cite: 2560]
* [cite_start]Is encoded as: `S' = {1, 1, 3, 31, 6}` [cite: 2561-2562]

[cite_start]This is reversible by taking the partial sum of `S'`[cite: 2563].

### Binary Reduction
[cite_start]Binary Reduction is a way to represent positive binary integers by removing the Most Significant Bit (MSB)[cite: 2566].
* **Example:** The number `19` is `10011` in binary.
* [cite_start]Its binary reduction is `0011`[cite: 2567].

[cite_start]This process is reversed by adding a '1' back as the MSB[cite: 2570]. [cite_start]In practice, a special "end of message" symbol is needed to separate the reduced numbers in the bitstream[cite: 2569]. [cite_start]In WDR, the *sign* of the coefficient is used as this symbol[cite: 2584].

### Bit Planes and Successive Approximation
This is the concept of transmitting an image's data one "bit plane" at a time, from the most significant bit (MSB) down to the least significant bit (LSB).
* The first pass (MSB) provides a rough, coarse approximation of the image.
* Each subsequent pass adds another bit of precision, refining the image.
* This is an **embedded** process, as the decoder can stop at any point and still have a valid (though less precise) image.

### Interval-Based Reconstruction
This is how WDR implements successive approximation.

Suppose a coefficient **`x`** (the true value) is found in the sorting pass when the threshold is `T`. We know `x` is in the interval `[T, 2T)`.
1.  **Initial Reconstruction:** Its first reconstruction value, **`xb`**, is set to the center of this interval: `xb = T + T/2`. This `xb` value is what's tracked in the `SCS` list.
2.  **Refinement:** In the next round, the new threshold is `T_new = T_old / 2`.
    * We first reconstruct the *current* interval for `xb` by calculating:
        * `low = xb - T_new`
        * `high = xb + T_new`
    * We now refine `xb` by checking which half of this `[low, high)` interval the *true value* `x` falls into. The midpoint for this check is `xb` itself.
    * **If $x \ge xb$**: The value is in the **upper half `[xb, high)`**. We output a bit **'1'**.
    * **If $x < xb$**: The value is in the **lower half `[low, xb)`**. We output a bit **'0'**.
3.  **Update:** We update `xb` in the `SCS` list to be the center of the *new*, smaller interval it belongs to.
4.  **Repeat:** This process repeats, shrinking the interval and approximating `xb` closer to the true value `x` with every pass.

---

## 3. Flow of the WDR

The overall flow of the WDR algorithm is a repeating loop:
1.  **Get Coefficients:** Start with the wavelet coefficients from the DWT.
2.  [cite_start]**Sorting Pass:** Find all "insignificant" coefficients (in the `ICS` list) that are *newly significant* (i.e., $|x| \ge T$) [cite: 2580-2581]. [cite_start]Encode their positions using **Index Coding** (Differential Coding + Binary Reduction)[cite: 2582].
3.  [cite_start]**Refinement Pass:** For all coefficients *already found* in previous rounds (in the `SCS` list), add one more bit of precision (one "bit plane") by refining their intervals [cite: 2587-2591].
4.  [cite_start]**Update & Loop:** Halve the threshold $T$ and repeat from Step 2, adding to the same bitstream[cite: 2591].
5.  [cite_start]**Final Compression:** The entire generated symbol stream is compressed using **Adaptive Arithmetic Coding**[cite: 2592].

---

## 4. Description of the Algorithm

1.  **DWT:** Get the wavelet coefficient matrices from the image.
2.  [cite_start]**Sort:** Sort all coefficients into a 1D list (`ICS`) based on the "coarse-to-fine" scanning order[cite: 2573].
    * [cite_start]**Order:** Scan $LL_N \rightarrow HL_N \rightarrow LH_N \rightarrow HH_N \rightarrow HL_{N-1} \rightarrow \dots \rightarrow HH_1$[cite: 2574].
    * **Scanning Heuristic:**
        * [cite_start]$HL$ subbands: **Column-by-column**[cite: 2574].
        * [cite_start]$LL$, $LH$, $HH$ subbands: **Row-by-row**[cite: 2574].
3.  **Initialize Threshold:** Define an initial global threshold `T` such that for all coefficients $|x_i| < 2T$ and for at least one coefficient $|x_j| [cite_start]\ge T$[cite: 2578]. Store this as `Initial_T`.
4.  **Initialize Structures:**
    * `SCS`: Set of Significant Coefficients `[empty]`. (Encoder stores `(val, center)` tuples).
    * `TPS`: Temporary Pass Set `[empty]`. (Stores original values found this round) [cite_start][cite: 2576].
    * [cite_start]`ICS`: Set of Insignificant Coefficients `[all sorted coefficients]` [cite: 2576-2577].
    * `out_bit`: Output bitstream `[null]`.
5.  **Loop** until the desired bit precision is reached:
    * **5.1 Sorting/Significance Pass:**
        * **5.1.1:** Iterate through `ICS`. For every coefficient $x_i$ where $|x_i| [cite_start]\ge T$, move its **original value** $x_i$ from `ICS` to `TPS`[cite: 2580].
        * [cite_start]**5.1.2:** As you find each $x_i$, store its current **index** in `ICS` in a list `P`, and store its **sign** in a `signs` list[cite: 2581, 2584].
        * [cite_start]**5.1.3:** After the scan, apply **Differential Coding** [cite: 2558] [cite_start]and then **Binary Reduction** [cite: 2565] to the index list `P` to get `P'`.
        * [cite_start]**5.1.4:** Append the interleaved index and sign data to the `out_bit` stream (e.g., `P'(1) + sign(1) + P'(2) + sign(2)...`) [cite: 2584-2585].
        * [cite_start]**5.1.5:** Update the indices in `ICS` to re-enumerate them (compact the list)[cite: 2586].
    * **5.2 Refinement Pass:**
        * **5.2.1:** For each tuple `(val_k, center_k)` in the `SCS` list:
            * `5.2.1.1` Obtain the current interval `[low, high)` by calculating `low = center_k - T` and `high = center_k + T`.
            * `5.2.1.2` Check if the original value `val_k` is in the lower half `[low, center_k)` or the upper half `[center_k, high)`. [cite_start]Append '0' or '1' to `out_bit` accordingly [cite: 2589-2591].
            * `5.2.1.3` Update the tuple's center: `center_k = (low + center_k) / 2` (if lower) or `center_k = (center_k + high) / 2` (if upper).
    * **5.3 End of Round:**
        * **5.2.2:** For each coefficient $x_k$ in `TPS`:
            * `5.2.2.1` Calculate its initial reconstruction value: `center_k = T + T/2`.
            * [cite_start]Append the new tuple `(x_k, center_k)` to the `SCS` list[cite: 2591].
        * [cite_start]**5.2.3:** Reset `TPS` to be empty[cite: 2591].
        * [cite_start]**5.2.4:** Update `T = T / 2`[cite: 2591].
6.  [cite_start]**Encode:** The final `out_bit` stream is encoded using adaptive arithmetic coding[cite: 2592].
7.  [cite_start]**Return:** Return `Initial_T` and the final compressed `out_bit` stream[cite: 2579].