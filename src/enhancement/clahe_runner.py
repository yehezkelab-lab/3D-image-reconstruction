"""
Contrast Limited Adaptive Histogram Equalization (CLAHE)
Applies local contrast enhancement on the lightness channel (L) of LAB color space.
Bridges deep-learning outputs and geometric computer vision requirements by maximizing local gradients.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


def apply_clahe_lab(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Applies CLAHE on the Lightness channel (L) in the CIELAB color space.

    Args:
        image (np.ndarray): Input BGR image (uint8).
        clip_limit (float): Threshold for contrast limiting (default 2.0).
        tile_grid_size (Tuple[int, int]): Size of grid for histogram equalization (default (8, 8)).

    Returns:
        np.ndarray: Contrast-enhanced BGR image.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def enhance_directory_clahe(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> int:
    """
    Applies CLAHE to all images in a directory.

    Returns:
        int: Number of processed images.
    """
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in in_path.iterdir() if p.suffix.lower() in valid_exts]

    count = 0
    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        enhanced = apply_clahe_lab(img, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        out_file = out_path / img_p.name
        cv2.imwrite(str(out_file), enhanced)
        count += 1

    return count
