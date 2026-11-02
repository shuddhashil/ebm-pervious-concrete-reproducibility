# shared/paths.py
"""
Portable path discovery utilities for the analysis suite.
"""

import os
from pathlib import Path
from typing import List, Optional

def _get_project_root() -> Path:
    """Return the project root directory (where this file resides)."""
    return Path(__file__).parent.parent

def get_data_dir() -> Path:
    """Return the data directory.
    Prefers the environment variable ANALYSIS_DATA_DIR, otherwise uses ./data
    relative to the project root.
    """
    env = os.getenv("ANALYSIS_DATA_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _get_project_root() / "data"

def get_output_dir() -> Path:
    """Return the output directory.
    Prefers the environment variable ANALYSIS_OUTPUT_DIR, otherwise uses ./output
    relative to the project root.
    """
    env = os.getenv("ANALYSIS_OUTPUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return _get_project_root() / "output"

def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)

def resolve_path(path_str: str) -> Path:
    """Resolve a path string, expanding user home and environment variables."""
    return Path(path_str).expanduser().resolve()

def find_input_files(pattern: str, directory: Optional[Path] = None) -> List[Path]:
    """Find files matching a glob pattern in a directory (defaults to data dir)."""
    if directory is None:
        directory = get_data_dir()
    return sorted(directory.glob(pattern))