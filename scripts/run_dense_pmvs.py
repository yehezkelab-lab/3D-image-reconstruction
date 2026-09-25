#!/usr/bin/env python3
"""
CLI script to run Dense 3D Point Cloud Reconstruction using COLMAP Undistorter and CMVS-PMVS2.
Generates option-all.ply dense point cloud.
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.reconstruction.dense_reconstruction import run_dense_reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Run Dense 3D Point Cloud Reconstruction using COLMAP Undistorter + PMVS2."
    )
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images.")
    parser.add_argument("--sparse_dir", type=str, required=True, help="Directory containing sparse model.")
    parser.add_argument("--workspace_dir", type=str, required=True, help="Directory for PMVS workspace.")
    parser.add_argument("--pmvs_bin", type=str, default=None, help="Path to pmvs2 executable binary.")

    args = parser.parse_args()

    run_dense_reconstruction(
        image_dir=args.image_dir,
        sparse_model_dir=args.sparse_dir,
        workspace_dir=args.workspace_dir,
        pmvs_bin_path=args.pmvs_bin
    )


if __name__ == "__main__":
    main()
