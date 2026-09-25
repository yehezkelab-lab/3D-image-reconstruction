#!/usr/bin/env python3
"""
Intelligent Image Enhancement & 3D Reconstruction Pipeline Runner
End-to-end command-line interface for:
  1. Synthetic degradation simulation
  2. Dual-stream deep neural enhancement (DarkIR + InstructIR)
  3. DSP contrast optimization (CLAHE in LAB space)
  4. Sparse SfM reconstruction (SuperPoint + LightGlue + pycolmap)
  5. Dense 3D point cloud reconstruction (PMVS2)
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.degradation.degrader import degrade_dataset
from src.enhancement.clahe_runner import enhance_directory_clahe
from src.enhancement.darkir_runner import DarkIRRunner
from src.enhancement.instructir_runner import InstructIRRunner
from src.enhancement.dual_pipeline import DualEnhancementPipeline
from src.reconstruction.sparse_reconstruction import run_sparse_reconstruction
from src.reconstruction.dense_reconstruction import run_dense_reconstruction


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end 3D Image Reconstruction with AI-based Enhancement."
    )
    parser.add_argument(
        "--mode",
        choices=["degrade", "enhance", "clahe", "sparse", "dense", "all"],
        default="all",
        help="Pipeline stage to execute (default: all)"
    )
    parser.add_argument("--raw_images", type=str, default="data/sample_images", help="Path to input raw/clean images.")
    parser.add_argument("--degraded_dir", type=str, default="output/degraded", help="Directory for degraded images.")
    parser.add_argument("--enhanced_dir", type=str, default="output/enhanced_dual", help="Directory for dual-enhanced images.")
    parser.add_argument("--sparse_dir", type=str, default="output/sparse", help="Directory for sparse SfM output.")
    parser.add_argument("--dense_dir", type=str, default="output/dense", help="Directory for dense reconstruction workspace.")

    # Model Weights
    parser.add_argument("--darkir_weights", type=str, default=None, help="Path to DarkIR_allLOL.pt")
    parser.add_argument("--darkir_config", type=str, default=None, help="Path to DarkIR config YAML")
    parser.add_argument("--instructir_im_weights", type=str, default=None, help="Path to im_instructir-7d.pt")
    parser.add_argument("--instructir_lm_weights", type=str, default=None, help="Path to lm_instructir-7d.pt")
    parser.add_argument("--instructir_config", type=str, default=None, help="Path to InstructIR config YAML")
    parser.add_argument("--prompt", type=str, default="Please remove the blur and restore the image to be sharp.")
    parser.add_argument("--pmvs_bin", type=str, default=None, help="Path to pmvs2 executable binary")

    args = parser.parse_args()

    print("=" * 70)
    print(" 3D IMAGE RECONSTRUCTION UNDER ADVERSE LOW-LIGHT CONDITIONS")
    print("=" * 70)

    # 1. Degradation
    if args.mode in ["degrade", "all"]:
        print("\n--- STAGE: SYNTHETIC DEGRADATION ---")
        degrade_dataset(args.raw_images, args.degraded_dir)

    # 2. Dual Neural Enhancement + CLAHE
    input_for_enhancement = args.degraded_dir if args.mode == "all" else args.raw_images
    if args.mode in ["enhance", "all"]:
        print("\n--- STAGE: DUAL-STREAM NEURAL ENHANCEMENT & CLAHE ---")
        darkir = None
        if args.darkir_weights and Path(args.darkir_weights).exists():
            darkir = DarkIRRunner(args.darkir_weights, args.darkir_config)

        instructir = None
        if args.instructir_im_weights and args.instructir_lm_weights:
            if Path(args.instructir_im_weights).exists() and Path(args.instructir_lm_weights).exists():
                instructir = InstructIRRunner(
                    args.instructir_im_weights,
                    args.instructir_lm_weights,
                    args.instructir_config
                )

        if not darkir and not instructir:
            print("[*] Note: Pretrained weights not specified or not found.")
            print("    Applying CLAHE contrast enhancement as direct DSP baseline.")
            enhance_directory_clahe(input_for_enhancement, args.enhanced_dir)
        else:
            pipeline = DualEnhancementPipeline(
                darkir_runner=darkir,
                instructir_runner=instructir,
                apply_clahe=True
            )
            pipeline.process_directory(input_for_enhancement, args.enhanced_dir, prompt=args.prompt)

    # 3. CLAHE Standalone
    if args.mode == "clahe":
        print("\n--- STAGE: CLAHE HISTOGRAM EQUALIZATION ---")
        enhance_directory_clahe(args.raw_images, args.enhanced_dir)

    # 4. Sparse Reconstruction
    sparse_input = args.enhanced_dir if args.mode == "all" else args.raw_images
    sparse_sfm_dir = Path(args.sparse_dir) / "sfm"
    if args.mode in ["sparse", "all"]:
        print("\n--- STAGE: SPARSE RECONSTRUCTION (SUPERPOINT + LIGHTGLUE) ---")
        sparse_sfm_dir = run_sparse_reconstruction(
            image_dir=sparse_input,
            output_dir=args.sparse_dir
        )

    # 5. Dense Reconstruction
    if args.mode in ["dense", "all"]:
        print("\n--- STAGE: DENSE 3D RECONSTRUCTION (COLMAP UNDISTORT + PMVS2) ---")
        run_dense_reconstruction(
            image_dir=sparse_input,
            sparse_model_dir=sparse_sfm_dir,
            workspace_dir=args.dense_dir,
            pmvs_bin_path=args.pmvs_bin
        )

    print("\n" + "=" * 70)
    print(" Pipeline execution finished.")
    print("=" * 70)


if __name__ == "__main__":
    main()
