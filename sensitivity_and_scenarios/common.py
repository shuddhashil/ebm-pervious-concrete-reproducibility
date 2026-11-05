"""Shared set-up for the sensitivity and scenario analyses.

All four analyses use the primary Sobol sweep of the design subspace (2^17 points, seed 20260829),
the final model with k = 1 and the baseline support thresholds (KDE cut-off 0.10, Mahalanobis
quantile 0.95, local-consistency tolerance 6 MPa). The baseline results are checked against the
threshold-grid results of robustness_verification.py before anything is written.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "08_analysis_code_suite"))
sys.path.insert(0, str(ROOT / "robustness_verification" / "src"))
from shared import ebm_engine as eng  # noqa: E402
import robustness_verification as rv  # noqa: E402

OUT = ROOT / "results" / "sensitivity"
OUT.mkdir(parents=True, exist_ok=True)

# Threshold-grid results: a fresh run of robustness_verification.py writes them to
# robustness_verification/outputs; the stored copy in results/robustness is used otherwise.
_fresh = ROOT / "robustness_verification" / "outputs" / "support_sensitivity_results.csv"
GRID_FILE = _fresh if _fresh.exists() else ROOT / "results" / "robustness" / "support_sensitivity_results.csv"

# Candidates of Table 4 as printed (target MPa, FA/B %, B/A, W/B)
TABLE4 = [(10, 50.0, .116, .305), (14, 50.0, .123, .305), (17, 50.0, .123, .325),
          (20, 34.7, .160, .325), (24, 15.8, .313, .399), (28, 19.1, .349, .397)]
TABLE_TARGETS = [10, 14, 17, 20, 24, 28]


def baseline_grid():
    g = pd.read_csv(GRID_FILE)
    g = g[(g.kde_cutoff == .10) & (g.mahal_quantile == .95) & (g.local_tolerance_MPa == 6.)]
    return g.set_index("target_strength_MPa")


def load_sweep():
    """Context, the 37 CS targets, the Sobol points and their canonical evaluation."""
    ctx = eng.load_context(False)
    targets = sorted(pd.read_csv(GRID_FILE).target_strength_MPa.unique())
    X = rv.sobol(ctx.bounds, 17, 20260829)
    z = rv.CanonicalEvaluator(ctx).evaluate(X)
    return ctx, targets, X, z


def check_baseline(supported_fraction, supported_min_co2, targets):
    """Baseline results must equal the central setting of the threshold grid."""
    g = baseline_grid().loc[targets]
    if supported_fraction is not None:
        assert np.allclose(supported_fraction, g.supported_fraction), "baseline supported fraction differs"
    assert np.allclose(supported_min_co2, g.minimum_supported_CO2, equal_nan=True), "baseline CO2 differs"
