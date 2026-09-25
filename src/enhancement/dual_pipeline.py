"""
Dual-Stream Enhancement Pipeline Coordinator
Implements the core paper architecture:
1. Split degraded dataset into two parallel deep learning paths:
   - Path A: DarkIR (Neural Illumination & Denoising)
   - Path B: InstructIR (Guided Semantic Deblurring)
2. Apply Contrast-Limited Adaptive Histogram Equalization (CLAHE) on both outputs.
3. Form an augmented 2x dataset ensuring data redundancy for feature matching.
"""

import os
from pathlib import Path
from typing import Union, Optional, List
import cv2
from tqdm import tqdm

from .clahe_runner import apply_clahe_lab
from .darkir_runner import DarkIRRunner
from .instructir_runner import InstructIRRunner


class DualEnhancementPipeline:
    """
    Executes the dual deep learning enhancement and CLAHE pipeline.
    """

    def __init__(
        self,
        darkir_runner: Optional[DarkIRRunner] = None,
        instructir_runner: Optional[InstructIRRunner] = None,
        apply_clahe: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_grid_size: tuple = (8, 8)
    ):
        self.darkir_runner = darkir_runner
        self.instructir_runner = instructir_runner
        self.apply_clahe = apply_clahe
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_grid_size = clahe_grid_size

    def process_image(
        self,
        image_bgr: cv2.Mat,
        prompt: str = "Please remove the blur and restore the image to be sharp."
    ) -> dict:
        """
        Enhance a single image through both branches.
        Returns a dictionary with 'dark' and 'instruct' images.
        """
        results = {}

        # Branch 1: DarkIR
        if self.darkir_runner:
            dark_enhanced = self.darkir_runner.enhance_image(image_bgr)
        else:
            dark_enhanced = image_bgr.copy()

        # Branch 2: InstructIR
        if self.instructir_runner:
            instruct_enhanced = self.instructir_runner.enhance_image(image_bgr, prompt=prompt)
        else:
            instruct_enhanced = image_bgr.copy()

        # Contrast Optimization via CLAHE
        if self.apply_clahe:
            dark_final = apply_clahe_lab(
                dark_enhanced,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_grid_size
            )
            instruct_final = apply_clahe_lab(
                instruct_enhanced,
                clip_limit=self.clahe_clip_limit,
                tile_grid_size=self.clahe_grid_size
            )
        else:
            dark_final = dark_enhanced
            instruct_final = instruct_enhanced

        results["dark"] = dark_final
        results["instruct"] = instruct_final
        return results

    def process_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        prompt: str = "Please remove the blur and restore the image to be sharp."
    ) -> int:
        """
        Process an entire directory of degraded images.
        Generates 2 output images per input image (<name>_dark.<ext> and <name>_instruct.<ext>).
        """
        in_p = Path(input_dir)
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        images = [f for f in in_p.iterdir() if f.suffix.lower() in valid_exts]

        print(f"[*] Processing {len(images)} images via Dual-Stream Pipeline...")
        total_created = 0

        for img_path in tqdm(images, desc="Dual Stream Enhancement"):
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            stem = img_path.stem
            ext = img_path.suffix

            res = self.process_image(img_bgr, prompt=prompt)

            dark_out = out_p / f"{stem}_dark{ext}"
            instruct_out = out_p / f"{stem}_instruct{ext}"

            cv2.imwrite(str(dark_out), res["dark"])
            cv2.imwrite(str(instruct_out), res["instruct"])
            total_created += 2

        print(f"[+] Complete. Created {total_created} augmented images in {out_p}")
        return total_created
