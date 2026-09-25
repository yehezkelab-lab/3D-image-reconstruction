import pymupdf
from pathlib import Path

pdf_path = r"C:\Users\1chiz\OneDrive\Desktop\חיזגיל\לימודים\תואר ראשון\2026\פרויקט\פרויקט שלי\הגשות\121 - לא חתום.pdf"
doc = pymupdf.open(pdf_path)

out_dir = Path(r"C:\Users\1chiz\3D-image-reconstruction\docs\images")
out_dir.mkdir(parents=True, exist_ok=True)

# 1. Clean extracted images mapping
extracted_mapping = {
    # Page 17 - Input comparisons
    (17, 230): "01_degraded_input_room.jpg",
    (17, 231): "01_original_input_room.jpg",
    (17, 228): "02_degraded_frames_closeup.jpg",
    (17, 229): "02_original_frames_closeup.jpg",
    # Page 30 & 31 - Overall comparisons
    (30, 263): "03_sparse_model_comparison_raw.png",
    (31, 266): "04_dense_model_comparison_raw.jpg",
    # Page 33 & 34 - Dense 3D Point Cloud Closeups
    (33, 270): "05_dense_pointcloud_original.jpg",
    (33, 272): "06_dense_pointcloud_degraded.jpg",
    (34, 275): "07_dense_pointcloud_integrated_ours.jpg",
}

for (page_num, xref), filename in extracted_mapping.items():
    base_image = doc.extract_image(xref)
    image_bytes = base_image["image"]
    target_path = out_dir / filename
    with open(target_path, "wb") as f:
        f.write(image_bytes)
    print(f"Saved raw extracted: {filename} ({base_image['width']}x{base_image['height']})")

# 2. Also render high-res 300 DPI crops for figures
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
