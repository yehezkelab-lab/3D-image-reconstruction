#!/usr/bin/env python3
"""
CLI script to run Dual-Stream Deep Learning Enhancement + CLAHE.
Generates an augmented 2x dataset (dark-recovered and deblurred variants).
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.enhancement.clahe_runner import enhance_directory_clahe
from src.enhancement.darkir_runner import DarkIRRunner
from src.enhancement.instructir_runner import InstructIRRunner
from src.enhancement.dual_pipeline import DualEnhancementPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run Dual-Stream Enhancement (DarkIR + InstructIR + CLAHE)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Path to input degraded images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to save dual enhanced images.")
    parser.add_argument("--darkir_weights", type=str, default=None, help="Path to DarkIR_allLOL.pt.")
    parser.add_argument("--darkir_config", type=str, default=None, help="Path to DarkIR config yaml.")
    parser.add_argument("--instructir_im_weights", type=str, default=None, help="Path to im_instructir-7d.pt.")
    parser.add_argument("--instructir_lm_weights", type=str, default=None, help="Path to lm_instructir-7d.pt.")
    parser.add_argument("--instructir_config", type=str, default=None, help="Path to InstructIR config yaml.")
    parser.add_argument("--prompt", type=str, default="Please remove the blur and restore the image to be sharp.")
    parser.add_argument("--no_clahe", action="store_true", help="Disable CLAHE post-processing.")
    parser.add_argument("--clahe_clip", type=float, default=2.0, help="CLAHE clip limit (default: 2.0).")

    args = parser.parse_args()

    darkir = None
    if args.darkir_weights and Path(args.darkir_weights).exists():
        print(f"[*] Initializing DarkIR model from {args.darkir_weights}")
        darkir = DarkIRRunner(
            weights_path=args.darkir_weights,
            config_path=args.darkir_config
        )

    instructir = None
    if args.instructir_im_weights and args.instructir_lm_weights:
        if Path(args.instructir_im_weights).exists() and Path(args.instructir_lm_weights).exists():
            print(f"[*] Initializing InstructIR model with prompt: '{args.prompt}'")
            instructir = InstructIRRunner(
                image_model_weights=args.instructir_im_weights,
                language_model_weights=args.instructir_lm_weights,
                config_path=args.instructir_config
            )

    pipeline = DualEnhancementPipeline(
        darkir_runner=darkir,
        instructir_runner=instructir,
        apply_clahe=not args.no_clahe,
        clahe_clip_limit=args.clahe_clip
    )

    pipeline.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        prompt=args.prompt
    )


if __name__ == "__main__":
    main()
