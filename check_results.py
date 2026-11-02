"""Compare whatever results exist with the values reported in the paper. Prints OK, DIFF or NOT RUN per item."""
import json
from pathlib import Path
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent
V = R / "results/validation"
RV = R / "robustness_verification/outputs"
RC = R / "robustness_verification/robustness_2_reconciliation/outputs"
FA = R / "flyash analysis/resampled_perturbation/outputs"
fails = 0


def show(name, expected, fn):
    global fails
    try:
        got = fn()
    except FileNotFoundError:
        print(f"NOT RUN  {name}")
        return
    dec = len(str(expected).split(".")[1]) if "." in str(expected) else 0
    ok = round(float(got), dec) == float(expected)
    fails += not ok
    print(f"{'OK  ' if ok else 'DIFF'}     {name}: paper {expected}, code {float(got):.{dec}f}")


def cv(m, c="mean"):
    return lambda: pd.read_csv(V / "cv_summary.csv", index_col=0).loc[m, c]


def loso(k):
    return lambda: json.load(open(V / "loso.json"))[k]


def fractions(col, target):
    def f():
        q = pd.read_csv(RV / "qmc_supported_fraction_by_scramble.csv")
        return q[(q.power == 17) & (q.target_strength_MPa == target)][col].mean() * 100
    return f


def premium(t):
    return lambda: pd.read_csv(RV / "support_premium_by_target.csv").set_index("target_strength_MPa").support_premium_pct[t]


def optimum(col, t, row):
    return lambda: pd.read_csv(RC / "authoritative_optima_by_target.csv").query("target_strength_MPa==@t").iloc[row][col]


for n, e, f in [
    ("Table 3 train R2", 0.966, cv("train_r2")), ("Table 3 test R2", 0.904, cv("test_r2")), ("Table 3 test R2 SD", 0.040, cv("test_r2", "std")),
    ("Table 3 test RMSE", 2.865, cv("test_rmse")), ("Table 3 test MAE", 1.918, cv("test_mae")), ("Table 3 test MAPE", 14.78, cv("test_mape")),
    ("Lowest split R2", 0.72, lambda: pd.read_csv(V / "cv_50_splits.csv").test_r2.min()),
    ("R2 without leverage rows", 0.876, lambda: pd.read_csv(V / "leverage.csv").no_leverage_r2[0]),
    ("Permutation 99th percentile", -0.056, lambda: np.percentile(pd.read_csv(V / "permutation_100.csv").perm_r2, 99)),
    ("Final model refit difference", 0.0, lambda: pd.read_csv(V / "final_model_check.csv").max_abs_diff[0]),
    ("LOSO rejected", 9, loso("rejected")), ("LOSO R2", -1.19, loso("r2")), ("LOSO RMSE", 12.93, loso("rmse")),
    ("LOSO median rho", 0.59, loso("median_rho")), ("LOSO positive studies", 19, loso("positive_rho")),
    ("Between-study share %", 60.1, loso("between_study_share")), ("Offset share %", 74.6, loso("offset_share")),
    ("R2 offsets removed", 0.44, loso("r2_offset_removed")), ("RMSE offsets removed", 6.52, loso("rmse_offset_removed")),
    ("Matched gap between studies", 5.93, loso("median_gap_between_tol0.5")), ("Matched gap within study", 1.61, loso("median_gap_within_tol0.5")),
    ("Porosity rho", -0.65, loso("porosity_rho")), ("Porosity p", 0.029, loso("porosity_p")),
    ("Base feasible % at 10 MPa", 27.95, fractions("base_feasible_fraction", 10)), ("Base feasible % at 28 MPa", 0.73, fractions("base_feasible_fraction", 28)),
    ("Supported % at 10 MPa", 13.29, fractions("evidence_supported_fraction", 10)), ("Supported % at 28 MPa", 0.61, fractions("evidence_supported_fraction", 28)),
    ("Premium % at 23.5 MPa", 20.4, premium(23.5)), ("Premium % at 24.5 MPa", 22.9, premium(24.5)), ("Premium % at 28 MPa", 4.8, premium(28)),
    ("Optimum B/A at 23.5 MPa", 0.316, optimum("BA", 23.5, 1)), ("Optimum W/B at 23.5 MPa", 0.391, optimum("WB", 23.5, 1)), ("Optimum CO2 at 23.5 MPa", 0.284, optimum("CO2_index", 23.5, 1)),
]:
    show(n, e, f)
print(f"\n{fails} difference(s).")
