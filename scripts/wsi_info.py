#!/usr/bin/env python3
"""
WSI Info Tool

A simple utility to inspect Whole Slide Image (WSI) metadata using OpenSlide.
Useful for determining dimensions and tile counts before processing.

Usage:
    python wsi_info.py <image_path>
"""

import sys
import os

try:
    import openslide
except ImportError:
    print("Error: openslide-python is not installed.")
    print("Install it with: pip install openslide-python")
    sys.exit(1)

def print_slide_info(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    try:
        slide = openslide.OpenSlide(filepath)
        
        print(f"File: {os.path.basename(filepath)}")
        print(f"Format: {slide.detect_format(filepath)}")
        print("-" * 30)
        
        # Dimensions (Level 0 is full resolution)
        w, h = slide.dimensions
        print(f"Dimensions (Level 0): {w} x {h}")
        print(f"Level Count: {slide.level_count}")
        
        print("-" * 30)
        print("Downsample Levels:")
        for i in range(slide.level_count):
            lw, lh = slide.level_dimensions[i]
            ds = slide.level_downsamples[i]
            print(f"  Level {i}: {lw} x {lh} (Downsample: {ds:.2f}x)")
            
        print("-" * 30)
        print("Properties (First 10):")
        # Print a few properties to verify metadata reading
        count = 0
        for k, v in slide.properties.items():
            print(f"  {k}: {v}")
            count += 1
            if count >= 10:
                print("  ... (more)")
                break
                
        slide.close()

    except Exception as e:
        print(f"Error reading slide: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python wsi_info.py <image_path>")
        sys.exit(1)
        
    print_slide_info(sys.argv[1])