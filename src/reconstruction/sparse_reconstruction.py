"""
Sparse 3D Reconstruction Pipeline using SuperPoint + LightGlue + pycolmap (HLOC)
Replaces fragile handcrafted features (SIFT/ORB) with deep learning interest points
and contextual transformer matching, enabling camera registration on degraded/restored imagery.
"""

import os
import sys
import re
import shutil
from pathlib import Path
from typing import Union, Optional, Dict, Any


def sanitize_image_filenames(image_dir: Path) -> None:
    """
    Renames images to contain only alphanumeric characters, underscores, dashes, and periods.
    Prevents COLMAP / PMVS parsing issues with special characters.
    """
    for item in image_dir.iterdir():
        if item.is_file():
            clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', item.name)
            if clean != item.name:
                item.rename(image_dir / clean)


def run_sparse_reconstruction(
    image_dir: Union[str, Path],
    output_dir: Union[str, Path],
    max_keypoints: int = 4096,
    nms_radius: int = 3,
    resize_max: int = 1600
) -> Path:
    """
    Executes deep feature extraction (SuperPoint) and contextual matching (LightGlue),
    followed by pycolmap incremental SfM reconstruction.

    Args:
        image_dir (Union[str, Path]): Directory containing input images (e.g. 56 dual-enhanced images).
        output_dir (Union[str, Path]): Output directory for features, matches, and SfM model.
        max_keypoints (int): Maximum keypoints per image (default 4096).
        nms_radius (int): Non-maximum suppression radius (default 3).
        resize_max (int): Max image dimension for feature extraction (default 1600).

    Returns:
        Path: Path to the generated sparse model directory (containing cameras, images, points3D).
    """
    img_path = Path(image_dir)
    out_path = Path(output_dir)
    sfm_dir = out_path / "sfm"

    out_path.mkdir(parents=True, exist_ok=True)
    sfm_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize file names
    sanitize_image_filenames(img_path)

    try:
        from hloc import extract_features, match_features, reconstruction, pairs_from_exhaustive
    except ImportError:
        raise ImportError(
            "Hierarchical-Localization (hloc) is required. "
            "Install with: pip install git+https://github.com/cvg/Hierarchical-Localization.git"
        )

    feature_conf = {
        "model": {
            "name": "superpoint",
            "max_keypoints": max_keypoints,
            "nms_radius": nms_radius
        },
        "output": f"feats-superpoint-n{max_keypoints}-rmax{resize_max}",
        "preprocessing": {
            "grayscale": True,
            "resize_force": True,
            "resize_max": resize_max
        }
    }

    features_path = out_path / "features.h5"
    pairs_path = out_path / "pairs-exhaustive.txt"
    matches_path = out_path / "matches.h5"

    print("[*] Extracting deep geometric keypoints with SuperPoint...")
    extract_features.main(feature_conf, img_path, feature_path=features_path)

    print("[*] Generating exhaustive image pairs...")
    pairs_from_exhaustive.main(pairs_path, features=features_path)

    print("[*] Performing contextual transformer matching with LightGlue...")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    match_features.main(matcher_conf, pairs_path, features=features_path, matches=matches_path)

    print("[*] Running incremental SfM sparse triangulation via pycolmap...")
    reconstruction.main(
        sfm_dir,
        img_path,
        pairs_path,
        features=features_path,
        matches=matches_path
    )

    print(f"[+] Sparse reconstruction successfully completed in: {sfm_dir}")
    return sfm_dir
