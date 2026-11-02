# shared/validation.py
"""
Strict input/row-alignment/overwrite validation for the analysis suite.
"""

import os
from pathlib import Path
from typing import List

import pandas as pd


class ValidationError(Exception):
    """Base class for validation errors."""
    pass


def validate_file_exists(path: Path) -> None:
    """Raise ValidationError if the file does not exist."""
    if not path.is_file():
        raise ValidationError(f"Input file does not exist: {path}")


def validate_file_readable(path: Path) -> None:
    """Raise ValidationError if the file is not readable."""
    if not os.access(path, os.R_OK):
        raise ValidationError(f"Input file not readable: {path}")


def validate_file_writable(path: Path) -> None:
    """Raise ValidationError if the file cannot be written (or its parent directory)."""
    parent = path.parent
    if not parent.exists():
        # Directory does not exist; we need to be able to create it
        if not os.access(parent, os.W_OK):
            raise ValidationError(f"Cannot write to parent directory: {parent}")
    else:
        if not os.access(parent, os.W_OK):
            raise ValidationError(f"Parent directory not writable: {parent}")


def validate_row_alignment(dfs: List[pd.DataFrame]) -> None:
    """Raise ValidationError if any DataFrame has different columns."""
    if not dfs:
        return
    reference_columns = list(dfs[0].columns)
    for i, df in enumerate(dfs[1:], start=1):
        if list(df.columns) != reference_columns:
            raise ValidationError(
                f"Row alignment mismatch: file {i} columns {list(df.columns)} "
                f"do not match reference columns {reference_columns}"
            )


def validate_overwrite(path: Path, force: bool = False) -> None:
    """Raise ValidationError if the output file exists and force is False."""
    if path.is_file() and not force:
        raise ValidationError(f"Output file already exists (use --force to overwrite): {path}")


def validate_inputs(
    input_paths: List[Path],
    output_path: Path,
    force: bool = False,
) -> None:
    """Perform all validation steps for a batch of inputs and an output location."""
    for path in input_paths:
        validate_file_exists(path)
        validate_file_readable(path)
    validate_file_writable(output_path)
    validate_overwrite(output_path, force)

    # Load dataframes for row alignment (only if files are CSV)
    dfs = []
    for path in input_paths:
        # Assume CSV for now; could be extended
        try:
            df = pd.read_csv(path)
            dfs.append(df)
        except Exception as e:
            raise ValidationError(f"Failed to read {path} for row alignment: {e}")
    validate_row_alignment(dfs)