# shared/statistics.py
"""
Reusable statistics helpers for the analysis suite.
"""

import pandas as pd
from typing import Dict, Any


def describe_column(series: pd.Series) -> Dict[str, float]:
    """Return basic descriptive statistics for a pandas Series."""
    # Ensure numeric
    numeric = pd.to_numeric(series, errors="coerce")
    # Drop NaNs for calculations
    clean = numeric.dropna()
    if clean.empty:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(clean.count()),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),  # population std
        "min": float(clean.min()),
        "max": float(clean.max()),
    }


def compute_summary(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for each numeric column in a DataFrame."""
    summary = {}
    for col in df.columns:
        series = df[col]
        # Only numeric columns
        if pd.api.types.is_numeric_dtype(series):
            summary[col] = describe_column(series)
        else:
            # For non‑numeric columns, provide placeholder stats
            summary[col] = {
                "count": int(series.count()),
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
            }
    return summary