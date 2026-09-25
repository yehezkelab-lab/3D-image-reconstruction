"""
DarkIR Neural Illumination & Low-Light Enhancement Runner
Recovers hidden geometry in deep shadows and suppresses camera sensor noise.
"""

import os
import sys
import inspect
from pathlib import Path
from typing import Optional, Union, Dict, Any
import numpy as np
import torch
import cv2

from .tiling import tile_process_image


class DarkIRRunner:
    """
    Wrapper for DarkIR (Neural Illumination) inference.
    """

    def __init__(
        self,
        weights_path: Union[str, Path],
        config_path: Optional[Union[str, Path]] = None,
        darkir_repo_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        self.weights_path = Path(weights_path)
        self.config_path = Path(config_path) if config_path else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if darkir_repo_path:
            darkir_path_str = str(Path(darkir_repo_path).resolve())
            if darkir_path_str not in sys.path:
                sys.path.append(darkir_path_str)

        self._load_model()

    def _load_model(self):
        """Loads DarkIR architecture and weights."""
        if not self.weights_path.exists():
            raise FileNotFoundError(f"DarkIR weights not found at: {self.weights_path}")

        try:
            from archs.DarkIR import DarkIR
        except ImportError:
            raise ImportError(
                "Could not import 'DarkIR' from archs.DarkIR. "
                "Ensure DarkIR repository is cloned and in PYTHONPATH."
            )

        kwargs: Dict[str, Any] = {}
        if self.config_path and self.config_path.exists():
            import yaml
            with open(self.config_path, "r") as f:
                opt = yaml.safe_load(f)
            network_opt = opt.get("network_G", opt.get("network_g", opt.get("network", {})))
            valid_params = inspect.signature(DarkIR.__init__).parameters
            for param_name in valid_params:
                if param_name in ["self", "args", "kwargs"]:
                    continue
                val = network_opt.get(param_name, None)
                if param_name == "in_channels" and "inp_channels" in network_opt:
                    val = network_opt["inp_channels"]
                if param_name == "inp_channels" and "in_channels" in network_opt:
                    val = network_opt["in_channels"]
                if val is not None:
                    kwargs[param_name] = val

        self.model = DarkIR(**kwargs)
        checkpoint = torch.load(str(self.weights_path), map_location="cpu")
        state_dict = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self.model.load_state_dict(state_dict, strict=True)
        self.model = self.model.to(self.device).eval()

    def enhance_image(
        self,
        image_bgr: np.ndarray,
        tile_size: int = 400,
        overlap: int = 32
    ) -> np.ndarray:
        """
        Enhance an underexposed/noisy image using DarkIR with tiling.
        """
        def _forward(tile: torch.Tensor) -> torch.Tensor:
            return self.model(tile, side_loss=False)

        return tile_process_image(
            image_bgr=image_bgr,
            model_fn=_forward,
            tile_size=tile_size,
            overlap=overlap,
            device=self.device
        )
