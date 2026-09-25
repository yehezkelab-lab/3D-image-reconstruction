#!/usr/bin/env python3
"""
CLI script to simulate environmental degradation on a dataset of images.
Applies:
  1. Directional Motion Blur
  2. Severe Underexposure (alpha attenuation)
  3. Thermal Sensor Gaussian Noise
"""

import sys
import argparse
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.degradation.degrader import degrade_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Simulate physical camera degradation (motion blur, underexposure, sensor noise)."
    )
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing clean input images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save degraded images.")
    parser.add_argument("--blur_kernel", type=int, default=15, help="Motion blur kernel size (default: 15).")
    parser.add_argument("--blur_angle", type=float, default=45.0, help="Motion blur angle in degrees (default: 45.0).")
    parser.add_argument("--alpha", type=float, default=0.45, help="Underexposure factor (default: 0.45).")
    parser.add_argument("--noise_sigma", type=float, default=20.0, help="Sensor noise standard deviation (default: 20.0).")

    args = parser.parse_args()

    print(f"[*] Starting degradation simulation on: {args.input_dir}")
    print(f"    - Motion blur: size={args.blur_kernel}, angle={args.blur_angle}°")
    print(f"    - Underexposure alpha: {args.alpha}")
    print(f"    - Gaussian noise sigma: {args.noise_sigma}")

    count = degrade_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        blur_kernel_size=args.blur_kernel,
        blur_angle=args.blur_angle,
        alpha=args.alpha,
        noise_sigma=args.noise_sigma
    )

    print(f"[+] Successfully degraded {count} images and saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
