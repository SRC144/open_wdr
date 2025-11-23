import struct
import numpy as np

# Magic signature: "WDR" + "T" (Tiled)
# This identifies the file type efficiently
MAGIC = b'WDRT'
VERSION = 1
HEADER_SIZE = 128  # Reserved space for metadata

class WDRTileWriter:
    """
    Writes a custom WDR Tiled Archive (.wdr / .wca).
    
    Architecture:
    1. HEADER (128 Bytes): Global metadata (size, threshold, passes, etc.)
    2. INDEX TABLE (Variable): Lookup table for tile offsets.
    3. BLOB DATA (Variable): Sequential compressed data chunks.
    """
    def __init__(self, filepath, width, height, tile_size, global_T, scales, wavelet, quant_step, num_passes):
        self.f = open(filepath, 'wb')
        
        # Metadata
        self.width = width
        self.height = height
        self.tile_size = tile_size
        self.global_T = global_T
        self.scales = scales
        self.wavelet = wavelet
        self.quant_step = quant_step if quant_step else 0.0
        self.num_passes = num_passes

        # Calculate Grid Dimensions
        # Ceiling division handles padding logic implicitly for the grid count
        self.cols = (width + tile_size - 1) // tile_size
        self.rows = (height + tile_size - 1) // tile_size
        self.total_tiles = self.cols * self.rows
        
        # In-memory index of tile locations (offset, size)
        self.tile_index = [] 
        
        # --- 1. WRITE HEADER ---
        self._write_header()
        
        # --- 2. RESERVE INDEX TABLE SPACE ---
        # We write zeros now and seek back to fill it in at close().
        # Size = Total Tiles * 12 bytes (8 bytes offset + 4 bytes size)
        self.table_start_pos = self.f.tell()
        table_size_bytes = self.total_tiles * 12
        self.f.write(b'\x00' * table_size_bytes)

    def _write_header(self):
        """
        Writes the 128-byte fixed header.
        Struct Format (< = Little Endian):
          4s: Magic (4)
          I:  Version (4)
          I:  Width (4)
          I:  Height (4)
          I:  TileSize (4)
          d:  Global_T (8)
          d:  Quant_Step (8)
          I:  Scales (4)
          I:  Num_Passes (4)
          ------------------
          Total: 44 bytes used for numbers
        """
        self.f.seek(0)
        
        data = struct.pack(
            '<4sIIIIddII',
            MAGIC,
            VERSION,
            self.width,
            self.height,
            self.tile_size,
            self.global_T,
            self.quant_step,
            self.scales,
            self.num_passes
        )
        self.f.write(data)
        
        # Write Wavelet String (Fixed 32 bytes, null-padded)
        # Napari/Readers need this to know how to run IDWT
        wav_bytes = self.wavelet.encode('ascii')[:32]
        self.f.write(wav_bytes.ljust(32, b'\x00'))
        
        # Pad the rest of the 128-byte header with zeros for future compatibility
        current_pos = self.f.tell()
        padding = HEADER_SIZE - current_pos
        if padding > 0:
            self.f.write(b'\x00' * padding)

    def add_tile(self, compressed_bytes):
        """
        Appends a compressed tile blob to the file.
        NOTE: Must be called in Row-Major order (Tile 0,0 -> Tile 0,1 -> ...)
        """
        offset = self.f.tell()
        size = len(compressed_bytes)
        
        self.f.write(compressed_bytes)
        
        # Store location in memory
        self.tile_index.append((offset, size))

    def close(self):
        """
        Finalizes the file by writing the Index Table at the reserved spot.
        """
        if len(self.tile_index) != self.total_tiles:
            print(f"Warning: Expected {self.total_tiles} tiles, but wrote {len(self.tile_index)}")
        
        # Jump back to the space we reserved after the header
        self.f.seek(self.table_start_pos)
        
        # Write the lookup table
        # Q = uint64 (Offset), I = uint32 (Size)
        for offset, size in self.tile_index:
            self.f.write(struct.pack('<QI', offset, size))
            
        self.f.close()


class WDRTileReader:
    """
    Reads a custom WDR Tiled Archive.
    Optimized for random access (seeking).
    """
    def __init__(self, filepath):
        self.f = open(filepath, 'rb')
        
        # Read Header
        self.f.seek(0)
        # Read the numeric part (44 bytes)
        header_data = self.f.read(44)
        (magic, ver, w, h, ts, gt, qs, sc, np_val) = struct.unpack('<4sIIIIddII', header_data)
        
        if magic != MAGIC:
            raise ValueError("Invalid WDR File: Magic mismatch")
            
        self.width = w
        self.height = h
        self.tile_size = ts
        self.global_T = gt
        self.quant_step = qs
        self.scales = sc
        self.num_passes = np_val
        
        # Read Wavelet String
        wav_data = self.f.read(32)
        self.wavelet = wav_data.decode('ascii').strip('\x00')
        
        # Calculate Grid
        self.cols = (w + ts - 1) // ts
        self.rows = (h + ts - 1) // ts
        self.total_tiles = self.cols * self.rows
        
        # Read Index Table
        # Jump to where the table starts (byte 128)
        self.f.seek(HEADER_SIZE)
        
        # Read the entire table into RAM (10k tiles = ~120KB, very cheap)
        table_bytes = self.f.read(self.total_tiles * 12)
        
        self.offsets = []
        self.sizes = []
        
        # Parse the binary table
        for i in range(self.total_tiles):
            chunk = table_bytes[i*12 : (i+1)*12]
            off, sz = struct.unpack('<QI', chunk)
            self.offsets.append(off)
            self.sizes.append(sz)
            
    def get_tile_bytes(self, row, col):
        """
        O(1) Random Access to any tile's compressed data.
        """
        if row >= self.rows or col >= self.cols:
            return None
            
        idx = row * self.cols + col
        offset = self.offsets[idx]
        size = self.sizes[idx]
        
        self.f.seek(offset)
        return self.f.read(size)

    def close(self):
        self.f.close()