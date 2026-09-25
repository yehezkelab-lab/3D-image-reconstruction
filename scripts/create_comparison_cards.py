"""
Generate polished Before & After comparison visual cards for README.md.
"""

from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

img_dir = Path("docs/images")

def add_header(img: Image.Image, title: str, bg_color=(30, 30, 30), text_color=(255, 255, 255)) -> Image.Image:
    header_height = 40
    new_img = Image.new("RGB", (img.width, img.height + header_height), bg_color)
    new_img.paste(img, (0, header_height))
    draw = ImageDraw.Draw(new_img)
    # Simple default font
    draw.text((15, 10), title, fill=text_color)
    return new_img

def make_side_by_side(
    img1_path: Path,
    title1: str,
    img2_path: Path,
    title2: str,
    out_path: Path,
    target_height: int = 600
):
    im1 = Image.open(img1_path).convert("RGB")
    im2 = Image.open(img2_path).convert("RGB")

    # Resize to common height
    w1 = int(im1.width * (target_height / im1.height))
    im1 = im1.resize((w1, target_height), Image.Resampling.LANCZOS)
    
    w2 = int(im2.width * (target_height / im2.height))
    im2 = im2.resize((w2, target_height), Image.Resampling.LANCZOS)

    im1_titled = add_header(im1, title1, bg_color=(180, 40, 40))    # Reddish for Degraded
    im2_titled = add_header(im2, title2, bg_color=(30, 130, 60))    # Greenish for Restored/Clean

    gap = 20
    total_w = im1_titled.width + im2_titled.width + gap
    total_h = im1_titled.height

    combined = Image.new("RGB", (total_w, total_h), (240, 240, 240))
    combined.paste(im1_titled, (0, 0))
    combined.paste(im2_titled, (im1_titled.width + gap, 0))

    combined.save(str(out_path), quality=95)
    print(f"Generated side-by-side card: {out_path.name}")

# 1. Input Comparison (Full Scene)
make_side_by_side(
    img_dir / "01_degraded_input_room.jpg",
    "BEFORE: Degraded Input (Low-Light + Blur + Noise)",
    img_dir / "01_original_input_room.jpg",
    "REFERENCE: Original Clean Input Scene",
    img_dir / "comparison_01_scene_before_after.jpg",
    target_height=600
)

# 2. Input Comparison (Picture Frames Close-Up)
make_side_by_side(
    img_dir / "02_degraded_frames_closeup.jpg",
    "BEFORE: Severe Gradient & Edge Degradation",
    img_dir / "02_original_frames_closeup.jpg",
    "REFERENCE: Clean Structural Edges",
    img_dir / "comparison_02_frames_closeup_before_after.jpg",
    target_height=400
)

# 3. Dense 3D Point Cloud Comparison (Before vs After)
make_side_by_side(
    img_dir / "06_dense_pointcloud_degraded.jpg",
    "BEFORE: Degraded Baseline Model (118,647 pts - Fragmented)",
    img_dir / "07_dense_pointcloud_integrated_ours.jpg",
    "AFTER: Our Integrated Pipeline (694,216 pts - Dense & Sharp)",
    img_dir / "comparison_03_dense_3d_before_after.jpg",
    target_height=650
)
