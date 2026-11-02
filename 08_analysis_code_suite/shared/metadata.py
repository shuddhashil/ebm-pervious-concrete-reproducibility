# shared/metadata.py
"""
Metadata and hash capture for the analysis suite.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any


def compute_file_hash(path: Path, algorithm: str = "sha256") -> str:
    """Compute a cryptographic hash of the file at *path*."""
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def capture_metadata(path: Path) -> Dict[str, Any]:
    """Capture basic metadata for *path*."""
    stat = path.stat()
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "hash_sha256": compute_file_hash(path),
    }


def save_metadata(metadata: Dict[str, Any], output_path: Path) -> None:
    """Write metadata dictionary as JSON to *output_path*."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)