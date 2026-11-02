"""
Utility functions for flyash bridge analysis.
Path handling, logging, seeds, version recording.
"""
import sys
import os
import json
import logging
import platform
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

# Project root and analysis directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_DIR = Path(__file__).resolve().parent.parent  # flyash analysis/
SRC_DIR = ANALYSIS_DIR / "src"
OUTPUTS_DIR = ANALYSIS_DIR / "outputs"
FIGURES_DIR = ANALYSIS_DIR / "figures"
LOGS_DIR = ANALYSIS_DIR / "logs"
DOCS_DIR = ANALYSIS_DIR / "documentation"

# Ensure output directories exist
for d in [OUTPUTS_DIR, FIGURES_DIR, LOGS_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def setup_logging(script_name: str, level=logging.INFO) -> logging.Logger:
    """Configure file + console logging."""
    log_file = LOGS_DIR / f"{script_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger(script_name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", "%H:%M:%S")
    
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    
    logger.info(f"Logging to {log_file}")
    return logger


def record_environment(logger: logging.Logger) -> dict:
    """Record package versions and environment info."""
    import sklearn, interpret, numpy, pandas, scipy, matplotlib, joblib
    try:
        import optuna
        optuna_version = optuna.__version__
    except ImportError:
        # Optuna is not needed to reproduce the already-selected EBM settings.
        # Record its absence rather than preventing a read-only analysis run.
        optuna_version = None
    
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    env = {
        "timestamp": datetime.now().isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "sklearn": sklearn.__version__,
            "interpret": interpret.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
            "joblib": joblib.__version__,
            "optuna": optuna_version,
        },
        "numpy_random_state": np.random.get_state()[1][0] if hasattr(np.random, 'get_state') else None,
    }
    
    env_file = OUTPUTS_DIR / "environment.json"
    with open(env_file, "w") as f:
        json.dump(convert(env), f, indent=2)
    
    logger.info(f"Environment recorded: {env_file}")
    for pkg, ver in env["packages"].items():
        logger.info(f"  {pkg}: {ver}")
    
    return env


def set_global_seeds(seed: int = 42):
    """Set all random seeds for reproducibility."""
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def get_paths():
    """Return dictionary of all important project paths."""
    return {
        "project_root": PROJECT_ROOT,
        "analysis_dir": ANALYSIS_DIR,
        "dataset": PROJECT_ROOT / "01_core_data" / "phase5_augmented_dataset.csv",
        "analysis_records": PROJECT_ROOT / "01_core_data" / "analysis_records.csv",
        "primary_ebm": PROJECT_ROOT / "02_models_and_support" / "primary_row_split_ebm.pkl",
        "ad_isofor": PROJECT_ROOT / "02_models_and_support" / "phase3_new_isoforest_kde_mahalanobis.pkl",
        "ad_support": PROJECT_ROOT / "02_models_and_support" / "phase3_kde_mahalanobis.pkl",
        "sigma_model": PROJECT_ROOT / "02_models_and_support" / "sigma_model.pkl",
        "feature_importance_grouped": PROJECT_ROOT / "04_oof_and_diagnostics" / "feature_importance_grouped.csv",
        "interaction_fast_scores": PROJECT_ROOT / "04_oof_and_diagnostics" / "interaction_fast_scores.csv",
        "cv_performance": PROJECT_ROOT / "04_oof_and_diagnostics" / "cv_performance.csv",
        "shape_bands": PROJECT_ROOT / "06_behavior_and_interactions" / "phase8" / "shape_function_bands.csv",
        "interaction_slices": PROJECT_ROOT / "06_behavior_and_interactions" / "phase8" / "interaction_slices.csv",
        "tipping_points": PROJECT_ROOT / "06_behavior_and_interactions" / "phase8" / "tipping_points.csv",
        "fa_evidence_grid": PROJECT_ROOT / "robustness_verification" / "final_reconciliation" / "outputs" / "fa_interaction_evidence_masked_grid.csv",
        "training_code": PROJECT_ROOT / "03_training_code" / "EBM.py",
        "ebm_engine": PROJECT_ROOT / "08_analysis_code_suite" / "shared" / "ebm_engine.py",
        "outputs": OUTPUTS_DIR,
        "figures": FIGURES_DIR,
        "logs": LOGS_DIR,
        "docs": DOCS_DIR,
    }


def save_json(obj, path: Path):
    """Save JSON with Path handling."""
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    with open(path, "w") as f:
        json.dump(convert(obj), f, indent=2)


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)
