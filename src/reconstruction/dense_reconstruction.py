"""
Dense 3D Reconstruction Pipeline using COLMAP Undistorter + CMVS-PMVS2
Converts sparse reconstruction into a dense 3D point cloud using Patch-based Multi-View Stereo.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Union, Optional


def run_dense_reconstruction(
    image_dir: Union[str, Path],
    sparse_model_dir: Union[str, Path],
    workspace_dir: Union[str, Path],
    pmvs_bin_path: Optional[str] = None
) -> Optional[Path]:
    """
    Runs lens undistortion via COLMAP and executes PMVS2 dense reconstruction.

    Args:
        image_dir (Union[str, Path]): Path to directory with images.
        sparse_model_dir (Union[str, Path]): Path to sparse reconstruction model.
        workspace_dir (Union[str, Path]): Workspace directory for PMVS files.
        pmvs_bin_path (Optional[str]): Path to pmvs2 executable binary.

    Returns:
        Optional[Path]: Path to the generated dense point cloud (.ply), if successful.
    """
    img_path = Path(image_dir).resolve()
    sparse_path = Path(sparse_model_dir).resolve()
    work_path = Path(workspace_dir).resolve()
    pmvs_target = work_path / "pmvs"

    work_path.mkdir(parents=True, exist_ok=True)

    # 1. Convert binary COLMAP model to TXT if necessary
    bin_images = sparse_path / "images.bin"
    if bin_images.exists():
        print("[*] Converting sparse binary model to text format...")
        subprocess.run(
            ["colmap", "model_converter", "--input_path", str(sparse_path), "--output_path", str(sparse_path), "--output_type", "TXT"],
            check=False
        )

    # 2. Sanitize and synchronize images.txt filenames with physical images
    imgs_txt = sparse_path / "images.txt"
    if imgs_txt.exists():
        print("[*] Synchronizing camera names in images.txt...")
        phys_files = {
            re.sub(r'[^a-zA-Z0-9]', '', f.name).lower(): f
            for f in img_path.glob("**/*") if f.is_file()
        }

        with open(imgs_txt, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()

        new_lines = []
        is_img = True
        for line in lines:
            if line.startswith("#"):
                new_lines.append(line)
                continue
            if is_img:
                parts = line.strip().split()
                if len(parts) >= 10:
                    orig = " ".join(parts[9:])
                    safe = re.sub(r'[^a-zA-Z0-9.]', '_', orig).lower()
                    sim = re.sub(r'[^a-zA-Z0-9]', '', orig).lower()
                    if sim in phys_files:
                        dest = img_path / safe
                        if not dest.exists():
                            shutil.copyfile(phys_files[sim], dest)
                    line = " ".join(parts[:9] + [safe]) + "\n"
                new_lines.append(line)
                is_img = not is_img
            else:
                new_lines.append(line)
                is_img = not is_img

        with open(imgs_txt, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    # 3. COLMAP image_undistorter
    print("[*] Running COLMAP image undistorter (output PMVS format)...")
    subprocess.run([
        "colmap", "image_undistorter",
        "--image_path", str(img_path),
        "--input_path", str(sparse_path),
        "--output_path", str(work_path),
        "--output_type", "PMVS"
    ], check=True)

    # 4. Patch option-all with valid camera indices
    opt_file = pmvs_target / "option-all"
    txt_dir = pmvs_target / "txt"
    if txt_dir.exists():
        valid = [
            int(f.stem) for f in txt_dir.iterdir()
            if f.suffix == ".txt" and f.stat().st_size > 0
        ]
        if opt_file.exists():
            with open(opt_file, "r", encoding="utf-8") as f:
                lines = f.read().splitlines()
            with open(opt_file, "w", encoding="utf-8") as f:
                updated_lines = [
                    f"timages {len(valid)} " + " ".join(map(str, valid))
                    if l.strip().startswith("timages") else l
                    for l in lines
                ]
                f.write("\n".join(updated_lines) + "\n")

    # 5. Run PMVS2 binary
    pmvs_executable = pmvs_bin_path or shutil.which("pmvs2") or "/content/CMVS-PMVS/program/build/main/pmvs2"
    if not os.path.exists(pmvs_executable) and not shutil.which(pmvs_executable):
        print(f"[!] PMVS2 executable not found at '{pmvs_executable}'. Please install CMVS-PMVS or check path.")
        return None

    print(f"[*] Running PMVS2 dense reconstruction: {pmvs_executable} {pmvs_target}/ option-all")
    ret = subprocess.run([pmvs_executable, f"{pmvs_target}/", "option-all"])
    if ret.returncode == 0:
        ply_file = pmvs_target / "models" / "option-all.ply"
        if ply_file.exists():
            print(f"[+] Dense 3D point cloud successfully generated: {ply_file}")
            return ply_file

    print("[!] Dense reconstruction finished with errors or PLY was not found.")
    return None
