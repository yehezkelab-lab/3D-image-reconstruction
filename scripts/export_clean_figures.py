import pymupdf
from pathlib import Path

pdf_path = r"C:\Users\1chiz\OneDrive\Desktop\חיזגיל\לימודים\תואר ראשון\2026\פרויקט\פרויקט שלי\הגשות\121.pdf"
doc = pymupdf.open(pdf_path)

out_dir = Path(r"C:\Users\1chiz\3D-image-reconstruction\docs\images")
out_dir.mkdir(parents=True, exist_ok=True)

# Clean extracted images mapping from 121.pdf
extracted_mapping = {
    # Page 17 - Input comparisons
    (17, 335): "01_degraded_input_room.jpg",
    (17, 336): "01_original_input_room.jpg",
    (17, 333): "02_degraded_frames_closeup.jpg",
    (17, 334): "02_original_frames_closeup.jpg",
    # Page 30 & 31 - Overall comparisons
    (30, 387): "03_sparse_model_comparison_raw.png",
    (31, 391): "04_dense_model_comparison_raw.jpg",
    # Page 33 & 34 - Dense 3D Point Cloud Closeups
    (33, 397): "05_dense_pointcloud_original.jpg",
    (33, 399): "06_dense_pointcloud_degraded.jpg",
    (34, 403): "07_dense_pointcloud_integrated_ours.jpg",
}

for (page_num, xref), filename in extracted_mapping.items():
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    target_path = out_dir / filename
    with open(target_path, "wb") as f:
        f.write(image_bytes)
    print(f"Saved raw extracted from 121.pdf: {filename} ({base_image['width']}x{base_image['height']})")

# Render high-res 300 DPI crops for comparison figures
# Page 30 - Sparse Model Comparison
page_30 = doc[29]
rect_p30 = pymupdf.Rect(100, 265, 530, 595)
pix_30 = page_30.get_pixmap(dpi=300, clip=rect_p30)
pix_30.save(str(out_dir / "03_sparse_model_comparison_300dpi.png"))
print("Saved 300 DPI crop: 03_sparse_model_comparison_300dpi.png")

# Page 31 - Dense Model Comparison
page_31 = doc[30]
rect_p31 = pymupdf.Rect(100, 65, 530, 395)
pix_31 = page_31.get_pixmap(dpi=300, clip=rect_p31)
pix_31.save(str(out_dir / "04_dense_model_comparison_300dpi.png"))
print("Saved 300 DPI crop: 04_dense_model_comparison_300dpi.png")
