"""
InstructIR Guided Semantic Deblur Runner
Restores structural sharpness and linear edge geometry using text-instruction guidance.
Default prompt: "Please remove the blur and restore the image to be sharp."
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

from .tiling import tile_process_tensor


class _Dict2Namespace:
    def __init__(self, d: dict):
        for k, v in d.items():
            if isinstance(v, dict):
                self.__dict__[k] = _Dict2Namespace(v)
            else:
                self.__dict__[k] = v


class InstructIRRunner:
    """
    Wrapper for InstructIR (Language-Guided Deblurring) inference.
    """

    def __init__(
        self,
        image_model_weights: Union[str, Path],
        language_model_weights: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        instructir_repo_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.image_weights_path = Path(image_model_weights)
        self.lm_weights_path = Path(language_model_weights)
        self.config_path = Path(config_path) if config_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if instructir_repo_path:
            p_str = str(Path(instructir_repo_path).resolve())
            if p_str not in sys.path:
                sys.path.append(p_str)

        self.model = None
        self.language_model = None
        self.lm_head = None
        self._load_models()

    def _load_models(self):
        """Loads InstructIR image and language model weights."""
        if not self.image_weights_path.exists():
            raise FileNotFoundError(f"Image weights not found at: {self.image_weights_path}")
        if not self.lm_weights_path.exists():
            raise FileNotFoundError(f"Language weights not found at: {self.lm_weights_path}")

        try:
            from models import instructir
            from text.models import LanguageModel, LMHead
        except ImportError:
            raise ImportError(
                "Could not import InstructIR modules. "
                "Ensure InstructIR repository is cloned and in PYTHONPATH."
            )

        if self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path, "r") as f:
                cfg_dict = yaml.safe_load(f)
            cfg = _Dict2Namespace(cfg_dict)
        else:
            # Fallback default configuration
            cfg = _Dict2Namespace({
                "model": {
                    "in_ch": 3,
                    "width": 32,
                    "enc_blks": [2, 2, 4, 8],
                    "middle_blk_num": 12,
                    "dec_blks": [2, 2, 2, 2],
                    "textdim": 768
                },
                "llm": {
                    "model": "google/flan-t5-base",
                    "model_dim": 768,
                    "embd_dim": 768,
                    "nclasses": 7
                }
            })

        self.model = instructir.create_model(
            input_channels=cfg.model.in_ch,
            width=cfg.model.width,
            enc_blks=cfg.model.enc_blks,
            middle_blk_num=cfg.model.middle_blk_num,
            dec_blks=cfg.model.dec_blks,
            txtdim=cfg.model.textdim
        )
        self.model.load_state_dict(
            torch.load(str(self.image_weights_path), map_location="cpu"),
            strict=True
        )
        self.model = self.model.to(self.device).eval()

        self.language_model = LanguageModel(model=cfg.llm.model).to(self.device).eval()
        self.lm_head = LMHead(
            embedding_dim=cfg.llm.model_dim,
            hidden_dim=cfg.llm.embd_dim,
            num_classes=cfg.llm.nclasses
        )
        self.lm_head.load_state_dict(
            torch.load(str(self.lm_weights_path), map_location="cpu"),
            strict=True
        )
        self.lm_head = self.lm_head.to(self.device).eval()

    def enhance_image(
        self,
        image_bgr: np.ndarray,
        prompt: str = "Please remove the blur and restore the image to be sharp.",
        tile_size: int = 400,
        overlap: int = 32
    ) -> np.ndarray:
        """
        Enhance a blurred image using text instruction guidance with tiled inference.
        """
        # Compute text embedding
        with torch.no_grad():
            text_embd, _ = self.lm_head(self.language_model([prompt]))
            text_embd = text_embd.to(self.device)

        def _forward(tile: torch.Tensor) -> torch.Tensor:
            return self.model(tile, text_embd)

        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        to_tensor = transforms.ToTensor()
        inp = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0).to(self.device)

        out_tensor = tile_process_tensor(inp, _forward, tile_size=tile_size, overlap=overlap)
        out_tensor = out_tensor.squeeze(0).cpu()

        to_pil = transforms.ToPILImage()
        enhanced_pil = to_pil(out_tensor)
        enhanced_rgb = np.array(enhanced_pil)
        return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
