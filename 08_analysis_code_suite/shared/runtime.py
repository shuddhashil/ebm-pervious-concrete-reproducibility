#!/usr/bin/env python3
"""
runtime.py - Shared utilities for the analysis suite.

Provides:
* Portable package-root discovery.
* Default input file locations (authoritative).
* Common argparse handling (input, output, dry-run, force).
* Safe output-directory creation (no overwrite unless forced).
* Simple metadata capture for CSV/JSON files.
* Dry-run result printer.
* Dependency checks with actionable pip install hints.
"""

import argparse
import importlib
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any


def get_package_root() -> Path:
    """
    Return the root directory of the installed package (the directory that
    contains the ``08_analysis_code_suite`` folder). Works both when the code
    is run from a source checkout or from an installed wheel.
    """
    # __file__ points to this file inside the package.
    return Path(__file__).resolve().parents[2]


def get_default_input_paths() -> List[Path]:
    """
    Authoritative default input CSV files required by the suite.
    """
    root = get_package_root()
    return [
        root / "01_core_data" / "phase3_mahal_kde_dataset.csv",
        root / "01_core_data" / "study_id_map.csv",
        root / "04_oof_and_diagnostics" / "nested_group_oof_predictions.csv",
        root / "06_behavior_and_interactions" / "phase8" / "phase8_applicability_domain.csv",
    ]


def parse_common_args(
    description: str,
    require_inputs: bool = True,
    require_output: bool = True,
) -> argparse.Namespace:
    """
    Build an ArgumentParser with the common flags used across the suite.
    ``require_inputs``/``require_output`` are ignored when ``--dry-run`` is set.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--input",
        nargs="+",
        help="One or more input CSV files (default: authoritative set).",
    )
    parser.add_argument(
        "--output",
        help="Output directory or file path (default: ./outputs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without writing any files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files/directories.",
    )
    args = parser.parse_args()

    # Resolve defaults when not supplied
    if not args.input:
        args.input = [str(p) for p in get_default_input_paths()]
    if not args.output:
        args.output = str(Path.cwd() / "outputs")

    # Enforce required arguments unless we are in dry-run mode
    if not args.dry_run:
        if require_inputs and not args.input:
            parser.error("--input is required unless --dry-run is specified.")
        if require_output and not args.output:
            parser.error("--output is required unless --dry-run is specified.")

    return args


def ensure_output_path(output_path: Path, force: bool = False) -> None:
    """
    Create the output directory if it does not exist.
    If it exists and ``force`` is False, raise an error to avoid overwriting.
    """
    if output_path.exists():
        if not force:
            raise FileExistsError(
                "Output path " + str(output_path) + " already exists. Use --force to overwrite."
            )
    else:
        output_path.mkdir(parents=True, exist_ok=True)


def capture_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Return a small metadata dictionary for a CSV or JSON file.
    Includes size, SHA-256 hash, and a quick content preview.
    """
    meta: Dict[str, Any] = {"path": str(file_path), "size_bytes": file_path.stat().st_size}
    # Compute SHA-256
    hasher = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    meta["sha256"] = hasher.hexdigest()

    # Content preview
    try:
        if file_path.suffix.lower() == ".json":
            with file_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            meta["preview"] = data if isinstance(data, (dict, list)) else str(data)
        elif file_path.suffix.lower() in {".csv", ".tsv"}:
            import pandas as pd

            df = pd.read_csv(file_path, nrows=5)
            meta["preview"] = df.head().to_dict(orient="records")
        else:
            meta["preview"] = None
    except Exception as exc:
        meta["preview_error"] = str(exc)

    return meta


def dry_run_print(message: str) -> None:
    """Standardised prefix for dry-run output."""
    print("[DRY RUN] " + message)


def check_dependencies(packages: List[Tuple[str, str]]) -> None:
    """
    Verify that each ``(module_name, pip_name)`` pair can be imported.
    If a module is missing, print a helpful pip install command and exit.
    """
    missing = []
    for module_name, pip_name in packages:
        try:
            importlib.import_module(module_name)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print(
            "Missing required packages:",
            ", ".join(missing),
            file=sys.stderr,
        )
        print(
            "You can install them with:",
            "pip install " + " ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


# Exported symbols for ``from shared.runtime import *``
__all__ = [
    "get_package_root",
    "get_default_input_paths",
    "parse_common_args",
    "ensure_output_path",
    "capture_metadata",
    "dry_run_print",
    "check_dependencies",
]