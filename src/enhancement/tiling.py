"""
Tiled Image Inference Utility
Solves GPU Out-Of-Memory (OOM) constraints for high-resolution images
by splitting images into overlapping tiles and performing seamless linear blending.
"""

import torch
import numpy as np
from typing import Callable, Optional, Tuple, Any


def tile_process_tensor(
    tensor: torch.Tensor,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = 400,
    overlap: int = 32
) -> torch.Tensor:
    """
    Process a 4D tensor (B, C, H, W) through a neural network using overlapping tiles.

    Args:
        tensor (torch.Tensor): Input tensor in range [0, 1] on CUDA or CPU.
        model_fn (Callable): Function that processes a patch (B, C, h_tile, w_tile) -> (B, C, h_tile, w_tile).
        tile_size (int): Spatial dimensions of each tile.
        overlap (int): Overlap between adjacent tiles in pixels.

    Returns:
        torch.Tensor: Seamlessly reconstructed output tensor of identical spatial shape.
    """
    b, c, h, w = tensor.shape
    stride = tile_size - overlap

    # Accumulators for blending
    output = torch.zeros_like(tensor)
    counts = torch.zeros_like(tensor)

    with torch.no_grad():
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + tile_size, h)
                x2 = min(x + tile_size, w)
                y1 = max(y2 - tile_size, 0)
                x1 = max(x2 - tile_size, 0)

                tile = tensor[:, :, y1:y2, x1:x2]
                out_tile = model_fn(tile)

                if isinstance(out_tile, (list, tuple)):
                    out_tile = out_tile[0]

                output[:, :, y1:y2, x1:x2] += out_tile
                counts[:, :, y1:y2, x1:x2] += 1.0

    return (output / counts).clamp(0.0, 1.0)


def tile_process_image(
    image_bgr: np.ndarray,
    model_fn: Callable[[torch.Tensor], torch.Tensor],
    tile_size: int = 400,
    overlap: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Convenience wrapper taking a BGR numpy image (uint8), converting to RGB tensor,
    running tiled inference, and converting back to BGR numpy image (uint8).
    """
    import cv2
    from PIL import Image
    from torchvision import transforms

    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    to_tensor = transforms.ToTensor()
    inp = to_tensor(Image.fromarray(img_rgb)).unsqueeze(0)

    if device == "cuda" and torch.cuda.is_available():
        inp = inp.cuda()

    out_tensor = tile_process_tensor(inp, model_fn, tile_size=tile_size, overlap=overlap)
    out_tensor = out_tensor.squeeze(0).cpu()

    to_pil = transforms.ToPILImage()
    enhanced_pil = to_pil(out_tensor)
    enhanced_rgb = np.array(enhanced_pil)
    return cv2.cvtColor(enhanced_rgb, cv2.COLOR_RGB2BGR)
