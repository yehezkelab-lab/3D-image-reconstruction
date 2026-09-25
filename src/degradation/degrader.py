"""
Degradation Simulator for 3D Reconstruction Benchmarking
Simulates physical degradation under extreme operational conditions:
1. Linear motion blur (camera or object displacement during exposure)
2. Underexposure (severe photon attenuation / low-light)
3. Thermal sensor noise (additive Gaussian noise at high ISO)
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, Optional


def create_motion_blur_kernel(kernel_size: int = 15, angle: float = 45.0) -> np.ndarray:
    """
    Generate a directional motion blur convolution kernel.

    Args:
        kernel_size (int): Size of the kernel (odd integer).
        angle (float): Motion angle in degrees.

    Returns:
        np.ndarray: Normalized 2D convolution kernel.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    center = (kernel_size - 1) / 2.0

    # Calculate line endpoints based on angle
    rad = np.deg2rad(angle)
    dx = np.cos(rad) * center
    dy = np.sin(rad) * center

    pt1 = (int(round(center - dx)), int(round(center - dy)))
    pt2 = (int(round(center + dx)), int(round(center + dy)))

    cv2.line(kernel, pt1, pt2, 1.0, thickness=1)
    kernel_sum = kernel.sum()
    if kernel_sum > 0:
        kernel /= kernel_sum
    else:
        kernel[int(center), int(center)] = 1.0
    return kernel


def apply_motion_blur(image: np.ndarray, kernel_size: int = 15, angle: float = 45.0) -> np.ndarray:
    """
    Apply 2D directional motion blur convolution: g(x, y) = h(x, y) * f(x, y).

    Args:
        image (np.ndarray): Input BGR image (uint8).
        kernel_size (int): Motion blur kernel dimension.
        angle (float): Angle of motion vector in degrees.

    Returns:
        np.ndarray: Blurred image.
    """
    kernel = create_motion_blur_kernel(kernel_size=kernel_size, angle=angle)
    return cv2.filter2D(image, -1, kernel)


def apply_underexposure(image: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Simulate underexposure / low light by linear signal attenuation:
    I_dark(x, y) = I_original(x, y) * alpha (where alpha < 1.0).

    Args:
        image (np.ndarray): Input BGR image (uint8).
        alpha (float): Attenuation factor (0.0 < alpha <= 1.0).

    Returns:
        np.ndarray: Darkened image.
    """
    img_float = image.astype(np.float32) * alpha
    return np.clip(img_float, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """
    Simulate thermal sensor noise with additive zero-mean Gaussian distribution:
    L_noisy(x, y) = I(x, y) + n(x, y), where n ~ N(0, sigma^2).

    Args:
        image (np.ndarray): Input BGR image (uint8).
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy image.
    """
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def degrade_image(
    image: np.ndarray,
    blur_kernel_size: int = 15,
    blur_angle: float = 45.0,
    alpha: float = 0.45,
    noise_sigma: float = 20.0
) -> np.ndarray:
    """
    Apply full synthetic degradation sequence:
    1. Motion Blur (attenuates high frequencies & distorts gradients)
    2. Underexposure (compresses dynamic range)
    3. Additive Gaussian noise (severely drops SNR and produces spurious features)

    Args:
        image (np.ndarray): Clean input BGR image.
        blur_kernel_size (int): Size of blur kernel.
        blur_angle (float): Angle of blur vector.
        alpha (float): Attenuation factor for exposure.
        noise_sigma (float): Standard deviation of sensor noise.

    Returns:
        np.ndarray: Degraded image.
    """
    blurred = apply_motion_blur(image, kernel_size=blur_kernel_size, angle=blur_angle)
    darkened = apply_underexposure(blurred, alpha=alpha)
    degraded = apply_gaussian_noise(darkened, sigma=noise_sigma)
    return degraded


def degrade_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    blur_kernel_size: int = 15,
    blur_angle: float = 45.0,
    alpha: float = 0.45,
    noise_sigma: float = 20.0
) -> int:
    """
    Degrades all images in the input directory and writes them to the output directory.

    Returns:
        int: Number of processed images.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = [p for p in input_path.iterdir() if p.suffix.lower() in valid_exts]

    count = 0
    for img_p in images:
        img = cv2.imread(str(img_p))
        if img is None:
            continue
        deg = degrade_image(
            img,
            blur_kernel_size=blur_kernel_size,
            blur_angle=blur_angle,
            alpha=alpha,
            noise_sigma=noise_sigma
        )
        out_file = output_path / img_p.name
        cv2.imwrite(str(out_file), deg)
        count += 1

    return count
