#!/usr/bin/env python3
"""
CLI script to run Sparse SfM using SuperPoint feature detection,
LightGlue contextual matching, and pycolmap incremental reconstruction.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reconstruction.sparse_reconstruction import run_sparse_reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Run Sparse SfM Reconstruction via SuperPoint + LightGlue + pycolmap."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images to reconstruct.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to store sparse model & matches.")
    parser.add_argument("--max_keypoints", type=int, default=4096, help="Max SuperPoint keypoints (default: 4096).")
    parser.add_argument("--nms_radius", type=int, default=3, help="SuperPoint NMS radius (default: 3).")
    parser.add_argument("--resize_max", type=int, default=1600, help="Resize max dimension (default: 1600).")

    args = parser.parse_args()

    run_sparse_reconstruction(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        max_keypoints=args.max_keypoints,
        nms_radius=args.nms_radius,
        resize_max=args.resize_max
    )


if __name__ == "__main__":
    main()
