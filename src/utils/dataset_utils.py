"""
Utility functions for dataset handling, sanitization, and archiving.
"""

import os
import re
import shutil
import zipfile
from pathlib import Path
from typing import Union, List


def sanitize_filename(filename: str) -> str:
    """Sanitize filename to standard characters only."""
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)


def extract_archive(archive_path: Union[str, Path], target_dir: Union[str, Path]) -> Path:
    """Extracts a zip archive into target directory, flattening nested root folder if present."""
    arc_p = Path(archive_path)
    tgt_p = Path(target_dir)
    tgt_p.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(arc_p, "r") as z:
        z.extractall(tgt_p)

    # Flatten if files are in a single subdirectory
    entries = list(tgt_p.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        nested = entries[0]
        for item in nested.iterdir():
            shutil.move(str(item), str(tgt_p))
        nested.rmdir()

    return tgt_p


def create_zip_archive(source_dir: Union[str, Path], output_zip_path: Union[str, Path]) -> Path:
    """Creates a zip archive from a directory."""
    src_p = Path(source_dir)
    out_p = Path(output_zip_path)
    if out_p.suffix == ".zip":
        out_base = str(out_p.with_suffix(""))
    else:
        out_base = str(out_p)

    archive = shutil.make_archive(out_base, "zip", src_p)
    return Path(archive)
