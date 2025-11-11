# Adaptive Arithmetic Coding (`aac_algorithm.md`)

## 1\. Overview

[cite\_start]This document provides the context and full C implementation for the adaptive arithmetic coding algorithm as presented by Witten, Neal, and Cleary[cite: 223]. [cite\_start]This algorithm is the exact one referenced by the WDR paper for its final compression stage[cite: 213].

The core idea is to represent an entire message as a single fraction within the interval `[0, 1)`. [cite\_start]Each symbol in the message narrows this interval based on its probability, which is supplied by a **model** [cite: 283-285]. [cite\_start]The "adaptive" part means this model updates its probability estimates (symbol frequencies) as it processes each symbol, allowing it to adapt to the data's local statistics [cite: 251, 1748-1750].

This implementation is notable for:

  * [cite\_start]**Integer Arithmetic:** It uses fixed-precision integer math to represent the interval `[low, high]`, avoiding floating-point numbers[cite: 442, 1577].
  * [cite\_start]**Incremental Operation:** It outputs bits as soon as the most significant bits of `low` and `high` match, rather than waiting until the end of the message [cite: 439-441, 1584-1585].
  * [cite\_start]**Underflow Handling:** It includes a clever mechanism (using `bits_to_follow`) to prevent a loss of precision when `low` and `high` become very close but straddle the halfway point [cite: 1641-1642, 1654].

## 2\. Core C Implementation (from Witten, Neal, Cleary)

[cite\_start]Below is the complete, working C code from Figures 3 and 4 of the paper[cite: 451, 1108, 1442]. This implementation is designed to be compiled as-is.

### `arithmetic_coding.h`

This header defines the core constants for the integer arithmetic.

```c
/* DECLARATIONS USED FOR ARITHMETIC ENCODING AND DECODING */
/* SIZE OF ARITHMETIC CODE VALUES. */
#define Code_value_bits 16 /* Number of bits in a code value */
typedef long code_value; /* Type of an arithmetic code value */

#define Top_value (((long)1<<Code_value_bits)-1) /* Largest code value */

/* HALF AND QUARTER POINTS IN THE CODE VALUE RANGE. */
#define First_qtr (Top_value/4+1) /* Point after first quarter */
#define Half (2*First_qtr) /* Point after first half */
#define Third_qtr (3*First_qtr) /* Point after third quarter */
```

[cite\_start][cite: 455-484]

### `model.h`

This header defines the *interface* between the coder and the probability model.

```c
/* INTERFACE TO THE MODEL. */

/* THE SET OF SYMBOLS THAT MAY BE ENCODED. */
#define No_of_chars 256 /* Number of character symbols */
#define EOF_symbol (No_of_chars+1) /* Index of EOF symbol */
#define No_of_symbols (No_of_chars+1) /* Total number of symbols */

/* TRANSLATION TABLES BETWEEN CHARACTERS AND SYMBOL INDEXES. */
int char_to_index[No_of_chars]; /* To index from character */
unsigned char index_to_char[No_of_symbols+1]; /* To character from index */

/* CUMULATIVE FREQUENCY TABLE. */
#define Max_frequency 16383 /* Maximum allowed frequency count 2^14 - 1 */
int cum_freq[No_of_symbols+1]; /* Cumulative symbol frequencies */
```

[cite\_start][cite: 485-524]

### `encode.c`

This file contains the main loop for encoding, which reads symbols, calls the encoder, and updates the model.

```c
/* MAIN PROGRAM FOR ENCODING. */
#include <stdio.h>
#include "model.h"

main()
{
    start_model(); /* Set up other modules. */
    start_outputing_bits();
    start_encoding();
    for (;;) { /* Loop through characters. */
        int ch; int symbol;
        ch = getc(stdin); /* Read the next character. */
        if (ch==EOF) break; /* Exit loop on end-of-file.*/
        symbol = char_to_index[ch]; /* Translate to an index. */
        encode_symbol(symbol,cum_freq); /* Encode that symbol. */
        update_model(symbol); /* Update the model. */
    }
    encode_symbol(EOF_symbol,cum_freq); /* Encode the EOF symbol. */
    done_encoding(); /* Send the last few bits. */
    done_outputing_bits();
    exit(0);
}
```

[cite\_start][cite: 528-570]

### `arithmetic_encode.c`

This file contains the core logic for encoding a symbol and handling the integer arithmetic, bit shifting, and underflow.

```c
/* ARITHMETIC ENCODING ALGORITHM. */
#include "arithmetic_coding.h"

static void bit_plus_follow(); /* Routine that follows */

/* CURRENT STATE OF THE ENCODING. */
static code_value low, high; /* Ends of the current code region */
static long bits_to_follow; /* Number of opposite bits to output after the next bit. */

/* START ENCODING A STREAM OF SYMBOLS. */
start_encoding()
{
    low = 0; /* Full code range. */
    high = Top_value;
    bits_to_follow = 0; /* No bits to follow next. */
}

/* ENCODE A SYMBOL. */
encode_symbol(symbol,cum_freq)
    int symbol; /* Symbol to encode */
    int cum_freq[]; /* Cumulative symbol frequencies */
{
    long range; /* Size of the current code region */
    range = (long)(high-low)+1;
    high = low + (range*cum_freq[symbol-1])/cum_freq[0]-1; /* Narrow the code region */
    low = low + (range*cum_freq[symbol])/cum_freq[0]; /* to that allotted to this */
                                                    /* symbol. */
    for (;;) { /* Loop to output bits. */
        if (high<Half) {
            bit_plus_follow(0); /* Output 0 if in low half. */
        }
        else if (low>=Half) { /* Output 1 if in high half.*/
            bit_plus_follow(1);
            low -= Half;
            high -= Half; /* Subtract offset to top. */
        }
        else if (low>=First_qtr && high<Third_qtr) { /* Output an opposite bit */
                                                    /* later if in middle half. */
            bits_to_follow += 1;
            low -= First_qtr; /* Subtract offset to middle*/
            high -= First_qtr;
        }
        else break; /* Otherwise exit loop. */
        low = 2*low;
        high = 2*high+1; /* Scale up code range. */
    }
}

/* FINISH ENCODING THE STREAM. */
done_encoding()
{
    bits_to_follow += 1; /* Output two bits that */
    if (low<First_qtr) bit_plus_follow(0); /* select the quarter that */
    else bit_plus_follow(1); /* the current code range */
} /* contains. */

/* OUTPUT BITS PLUS FOLLOWING OPPOSITE BITS. */
static void bit_plus_follow(bit)
    int bit;
{
    output_bit(bit); /* Output the bit. */
    while (bits_to_follow>0) {
        output_bit(!bit); /* Output bits to follow */
        bits_to_follow -= 1; /* opposite bits. Set */
    } /* bits_to_follow to zero. */
}
```

[cite\_start][cite: 581-752]

### `decode.c`

This file contains the main loop for decoding, which reads bits, calls the decoder, and updates the model.

```c
/* MAIN PROGRAM FOR DECODING. */
#include <stdio.h>
#include "model.h"

main()
{
    start_model(); /* Set up other modules. */
    start_inputing_bits();
    start_decoding();
    for (;;) { /* Loop through characters. */
        int ch; int symbol;
        symbol = decode_symbol(cum_freq); /* Decode next symbol. */
        if (symbol==EOF_symbol) break; /* Exit loop if EOF symbol. */
        ch = index_to_char[symbol]; /* Translate to a character. */
        putc(ch,stdout); /* Write that character. */
        update_model(symbol); /* Update the model. */
    }
    exit(0);
}
```

[cite\_start][cite: 753-798]

### `arithmetic_decode.c`

This file contains the core logic for decoding a symbol. It mirrors the encoder's operations to stay in sync.

```c
/* ARITHMETIC DECODING ALGORITHM. */
#include "arithmetic_coding.h"

/* CURRENT STATE OF THE DECODING. */
static code_value value; /* Currently-seen code value */
static code_value low, high; /* Ends of current code region */

/* START DECODING A STREAM OF SYMBOLS. */
start_decoding()
{
    int i;
    value = 0; /* Input bits to fill the */
    for (i=1; i<=Code_value_bits; i++) { /* code value. */
        value = 2*value+input_bit();
    }
    low = 0; /* Full code range. */
    high = Top_value;
}

/* DECODE THE NEXT SYMBOL. */
int decode_symbol(cum_freq)
    int cum_freq[]; /* Cumulative symbol frequencies */
{
    long range; /* Size of current code region */
    int cum; /* Cumulative frequency calculated */
    int symbol; /* Symbol decoded */
    range = (long)(high-low)+1;
    cum = (((long)(value-low)+1)*cum_freq[0]-1)/range; /* Find cum freq for value. */
    for (symbol=1; cum_freq[symbol]>cum; symbol++); /* Then find symbol. */
    high = low + (range*cum_freq[symbol-1])/cum_freq[0]-1; /* Narrow the code region */
    low = low + (range*cum_freq[symbol])/cum_freq[0]; /* to that allotted to this */
                                                    /* symbol. */
    for (;;) { /* Loop to get rid of bits. */
        if (high<Half) {
            /* nothing */ /* Expand low half. */
        }
        else if (low>=Half) { /* Expand high half. */
            value -= Half;
            low -= Half; /* Subtract offset to top. */
            high -= Half;
        }
        else if (low>=First_qtr && high<Third_qtr) { /* Expand middle half. */
            value -= First_qtr;
            low -= First_qtr; /* Subtract offset to middl.*/
            high -= First_qtr;
        }
        else break; /* Otherwise exit loop. */
        low = 2*low;
        high = 2*high+1; /* Scale up code range. */
        value = 2*value+input_bit(); /* Move in next input bit. */
    }
    return symbol;
}
```

[cite\_start][cite: 807-941]

### `adaptive_model.c`

This is the **adaptive** model implementation. It initializes all frequencies to 1 and updates them after each symbol is processed.

```c
/* THE ADAPTIVE SOURCE MODEL */
#include "model.h"

int freq[No_of_symbols+1]; /* Symbol frequencies */

/* INITIALIZE THE MODEL. */
start_model()
{
    int i;
    for (i=0; i<No_of_chars; i++) { /* Set up tables that */
        char_to_index[i] = i+1; /* translate between symbol*/
        index_to_char[i+1] = i; /* indexes and characters. */
    }
    for (i=0; i<=No_of_symbols; i++) { /* Set up initial frequency */
        freq[i] = 1; /* counts to be one for all */
        cum_freq[i] = No_of_symbols-i; /* symbols. */
    }
    freq[0] = 0; /* Freq[0] must not be the */
} /* same as freq[1]. */

/* UPDATE THE MODEL TO ACCOUNT FOR A NEW SYMBOL. */
update_model(symbol)
    int symbol; /* Index of new symbol */
{
    int i; /* New index for symbol */
    if (cum_freq[0]==Max_frequency) { /* See if frequency counts */
        int cum; /* are at their maximum. */
        cum = 0; /* If so, halve all the */
        for (i = No_of_symbols; i>=0; i--) { /* counts (keeping them */
            freq[i] = (freq[i]+1)/2; /* non-zero). */
            cum_freq[i] = cum;
            cum += freq[i];
        }
    }
    for (i=symbol; freq[i]==freq[i-1]; i--); /* Find symbol's new index. */
    if (i<symbol) {
        int ch_i, ch_symbol;
        ch_i = index_to_char[i]; /* Update the translation */
        ch_symbol = index_to_char[symbol]; /* tables if the symbol has */
        index_to_char[i] = ch_symbol; /* moved. */
        index_to_char[symbol] = ch_i;
        char_to_index[ch_i] = symbol;
        char_to_index[ch_symbol] = i;
    }
    freq[i] += 1; /* Increment the frequency */
    while (i>0) { /* count for the symbol and */
        i -= 1; /* update the cumulative */
        cum_freq[i] += 1; /* frequencies. */
    }
}
```

[cite\_start][cite: 1442-1558]

### `bit_input.c` & `bit_output.c`

These files (Figures 3, 9, 10 in the paper) handle the packing and unpacking of bits into bytes (`char`) for file I/O. [cite\_start]They are not shown here for brevity but are included in the paper's full C implementation[cite: 954, 1036].

-----

## 3\. Implementation Guide and Key Concepts

This is a step-by-step guide to *how* the C code works.

### 1\. Data Structures & Constants

  * [cite\_start]**`Code_value_bits` (16):** The Coder uses 16-bit integers for `low` and `high`[cite: 462, 1578]. [cite\_start]`Top_value` is $2^{16}-1$[cite: 469].
  * [cite\_start]**`Max_frequency` (16383):** The model's *total* frequency count (`cum_freq[0]`) must not exceed this[cite: 517, 1729]. This is $2^{14}-1$. [cite\_start]This $c=16, f=14$ relationship is crucial to prevent overflow, as `f <= c - 2` [cite: 1708, 1673-1674].
  * [cite\_start]**`cum_freq[]`:** This is a **backward cumulative array** [cite: 435-436]. [cite\_start]`cum_freq[0]` is the total frequency[cite: 437, 1573]. The range for a `symbol` is `[cum_freq[symbol], cum_freq[symbol-1])`.
  * [cite\_start]**`char_to_index[]` & `index_to_char[]`:** These tables translate 8-bit characters to internal symbol indexes (1 to 257)[cite: 508, 511, 1570]. [cite\_start]The adaptive model *changes* these tables to keep frequent symbols at low indexes, which speeds up the decoder's search [cite: 1569, 1761-1762].

### 2\. Encoder Logic

1.  [cite\_start]**Start:** `low` is set to `0`, `high` is set to `Top_value`[cite: 612, 617].
2.  **`encode_symbol(symbol)`:**
      * [cite\_start]It calculates the current `range = (high - low) + 1`[cite: 645].
      * It narrows the interval using integer division. [cite\_start]The new `high` and `low` are calculated based on the symbol's proportional share of the range [cite: 640-641, 646-647].
3.  [cite\_start]**Bit-Shifting Loop:** After narrowing, the `for(;;)` loop [cite: 657] runs. This is the incremental output.
      * **Case 1: `high < Half`:** The entire range `[low, high]` is in the lower half. The MSB must be **0**. [cite\_start]The coder outputs '0', then scales the entire range by 2 (e.g., `[0, 0.4]` becomes `[0, 0.8]`) by setting `low = 2*low` and `high = 2*high+1` [cite: 661-664, 700-704].
      * **Case 2: `low >= Half`:** The entire range is in the upper half. The MSB must be **1**. [cite\_start]The coder outputs '1', subtracts the `Half` offset from `low` and `high`, then scales by 2 [cite: 667-668, 671-676, 700-704].
      * **Case 3 (Underflow): `low >= First_qtr && high < Third_qtr`:** The range is stuck in the middle "underflow" region (e.g., `[0.4, 0.6]`). No bit can be output. [cite\_start]The coder scales this *middle* range to the full range (by subtracting `First_qtr` and scaling by 2) and increments `bits_to_follow` [cite: 681-683, 687, 689-692, 700-704].
      * [cite\_start]**`bit_plus_follow(bit)`:** This routine outputs the given `bit`, then outputs `bits_to_follow` *opposite* bits [cite: 733-752]. This "flushes" the delayed underflow bits.
4.  [cite\_start]**End:** `done_encoding()` flushes two final bits to uniquely identify the final range [cite: 715, 718-726].

### 3\. Decoder Logic

The decoder's job is to *perfectly mirror* the encoder's arithmetic.

1.  **Start:** `low=0`, `high=Top_value`. [cite\_start]The `value` buffer is filled with the first 16 bits from the bitstream [cite: 826-838, 843-846].
2.  **`decode_symbol()`:**
      * [cite\_start]It calculates the current `range` *exactly* as the encoder did[cite: 871].
      * **Find Symbol:** It "inverts" the encoder's formula. [cite\_start]It calculates what the cumulative frequency `cum` *must have been* to produce the current `value`[cite: 864, 873].
      * [cite\_start]It then searches `cum_freq[]` for the `symbol` that matches this `cum`[cite: 866].
      * [cite\_start]**Narrow Interval:** It updates `low` and `high` using the *exact same formulas* as the encoder [cite: 874-875, 880].
3.  **Bit-Shifting Loop:** The decoder *must* run the exact same shifting loop to stay in sync.
      * [cite\_start]**Case 1, 2, 3:** It performs the *same* `if/else if` checks for `Half` and `First_qtr` [cite: 884-916].
      * [cite\_start]**Update `value`:** As it scales `low` and `high`, it *also* scales `value` and shifts in a new `input_bit()` to keep the `value` buffer full [cite: 933-935].

### 4\. Adaptive Model Logic

  * [cite\_start]**`start_model()`:** Initializes all symbol frequencies `freq[i]` to 1[cite: 1471]. This "flat" model reflects no prior knowledge. [cite\_start]It then computes the full `cum_freq` array [cite: 1473-1474].
  * **`update_model(symbol)`:** This is called *after* a symbol is processed.
    1.  [cite\_start]**Rescaling:** First, it checks if `cum_freq[0]` (the total count) has hit `Max_frequency`[cite: 1491]. [cite\_start]If so, it **halves all individual `freq[]` counts** (`(freq[i]+1)/2` to prevent any count from becoming zero) and re-computes the *entire* `cum_freq[]` array from scratch [cite: 1493-1501]. This is a "forgetting" mechanism that weights recent symbols more heavily.
    2.  [cite\_start]**Re-sorting:** It finds the new correct frequency-sorted rank for the `symbol`[cite: 1517]. [cite\_start]If the symbol's rank changed (i.e., it's now more frequent than a symbol it was previously less frequent than), it **swaps** them in the `index_to_char` and `char_to_index` tables [cite: 1518-1528].
    3.  [cite\_start]**Increment:** It increments the `freq[i]` for the (newly ranked) symbol[cite: 1529].
    4.  [cite\_start]**Update `cum_freq`:** It updates all `cum_freq[j]` for `j < i` by adding 1, reflecting the increment [cite: 1531-1535].