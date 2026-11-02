# shared/ebm_engine.py
"""
Central engine for all seven analyses.

Ports the EXACT logic of the published optimisation pipeline
(05_existing_optimization/optimisatoin_final.py):

  * same embodied-CO2 factors and central transport distances
  * same EBM input builder (fixed median Cu / median WAS / curing = 28 d)
  * same applicability-domain (Isolation Forest) constraint
  * same Mahalanobis envelope, distance penalty, local-consistency check
  * same spatial sigma(B/A, W/B) uncertainty model (loaded from sigma_model.pkl)
   * same differential-evolution population set-up (popsize=20, tol=1e-6)

All paths are ABSOLUTE and hard-coded so the scripts run in Spyder without
any configuration.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import chi2

# --------------------------------------------------------------------------
# Absolute locations (hard-coded on purpose - no configuration needed)
# --------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[2]

PATH_EBM_BUNDLE = BASE / "02_models_and_support" / "primary_row_split_ebm.pkl"
PATH_AD_ISOFOR = BASE / "02_models_and_support" / "phase3_new_isoforest_kde_mahalanobis.pkl"
PATH_AD_SUPPORT = BASE / "02_models_and_support" / "phase3_kde_mahalanobis.pkl"
PATH_SIGMA_MODEL = BASE / "02_models_and_support" / "sigma_model.pkl"
PATH_DATASET = BASE / "01_core_data" / "phase5_augmented_dataset.csv"
PATH_STUDY_MAP = BASE / "01_core_data" / "study_id_map.csv"
PATH_OOF_GROUPED = BASE / "04_oof_and_diagnostics" / "nested_group_oof_predictions.csv"
PATH_EBM_GROUPED = BASE / "02_models_and_support" / "final_ebm_grouped.pkl"
PATH_MODEL_METADATA = BASE / "02_models_and_support" / "final_model_metadata.json"

OUTPUT_ROOT = BASE / "09_analysis_outputs"

TARGET_COL = "Strength_con"

# Embodied CO2 (identical to optimisatoin_final.py)
CO2_CEMENT_PROD = 1.043
CO2_FA_PROD = 0.00397
CO2_AGG_PROD = 0.00503
CO2_WATER_PROD = 0.00212
TRANSPORT_CO2_FACTOR = 0.000062
DIST_CEMENT_MID = 225
DIST_FA_MID = 160
DIST_AGG_MID = 85
CO2_CEMENT_CENTRAL = CO2_CEMENT_PROD + DIST_CEMENT_MID * TRANSPORT_CO2_FACTOR
CO2_FA_CENTRAL = CO2_FA_PROD + DIST_FA_MID * TRANSPORT_CO2_FACTOR
CO2_AGG_CENTRAL = CO2_AGG_PROD + DIST_AGG_MID * TRANSPORT_CO2_FACTOR
CO2_WATER_CENTRAL = CO2_WATER_PROD

# Safeguard hyperparameters (identical to optimisatoin_final.py)
DIST_PENALTY_WEIGHT = 0.05
DIST_PENALTY_THRESHOLD = 2.0
LOCAL_K = 5
LOCAL_TOL = 6.0
MAHAL_QUANTILE = 0.95
CS_STD_K = 1.0
CS_BIAS_OVERRIDE = None
SIGMA_COMBINATION = "max"
CURING_FIXED = 28

# Reliability screening thresholds (from phase 3)
KDE_SUPPORT_MIN = 0.10          # w(x) threshold used for low_support_flag
KDE_LOW_FLAG_THRESHOLD = 0.1

# The 8 physical inputs used by the KDE / Mahalanobis support field
SUPPORT_INPUTS = [
    "Cu", "WAS_mm", "B/A", "W/B", "Sand/A", "curing period",
    "FlyAsh/B (%)", "SP (%)",
]


class Context(dict):
    """Simple attribute-access dict for the shared engine context."""

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc


def _resolve(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "Required input file(s) not found:\n  " + "\n  ".join(missing)
        )


def load_context(verbose: bool = True) -> Context:
    """Load every artifact needed by the analyses into one context object."""
    _resolve([
        PATH_EBM_BUNDLE, PATH_AD_ISOFOR, PATH_AD_SUPPORT,
        PATH_SIGMA_MODEL, PATH_DATASET,
    ])

    ctx = Context()

    # ---- EBM bundle (primary row-split model) ----
    bundle = joblib.load(PATH_EBM_BUNDLE)
    ctx.model = bundle["model"]
    ctx.features = list(bundle["features"])
    ctx.use_log = bool(bundle["use_log_target"])
    ctx.feature_ranges = bundle["feature_ranges"]

    # ---- Dataset ----
    df = pd.read_csv(PATH_DATASET)
    ctx.df = df
    ctx.row_index = np.arange(len(df), dtype=int)
    ctx.duplicate_summary = {
        "exact_rows": int(df.duplicated().sum()),
        "predictor_rows": int(df.drop(columns=[TARGET_COL]).duplicated().sum()),
        "policy": "retained; duplicates are flagged, never silently dropped",
    }
    ctx.n_train = len(df)
    ctx.strength_min = float(df[TARGET_COL].min())
    ctx.strength_max = float(df[TARGET_COL].max())
    ctx.log_s_min = float(np.log(ctx.strength_min))
    ctx.log_s_max = float(np.log(ctx.strength_max))

    ctx.cu_fixed = float(df["Cu"].median())
    ctx.was_fixed = float(df["WAS_mm"].median())
    ctx.wb_median = float(df["W/B"].median())
    ctx.preprocessing_contract = {
        "features": ctx.features,
        "raw_transforms": {"Cu": "log", "curing period": "log"},
        "fixed_design_values": {"Cu": ctx.cu_fixed, "WAS_mm": ctx.was_fixed,
                                "Sand/A": 0.0, "curing period": CURING_FIXED,
                                "SP (%)": 0.0},
    }

    # Optimisation box = intersection of model range and data range
    fa_lo_r, fa_hi_r = ctx.feature_ranges["FlyAsh/B (%)"]
    ba_lo_r, ba_hi_r = ctx.feature_ranges["B/A"]
    wb_lo_r, wb_hi_r = ctx.feature_ranges["W/B"]
    ctx.bounds = [
        (max(fa_lo_r, float(df["FlyAsh/B (%)"].min())),
         min(fa_hi_r, float(df["FlyAsh/B (%)"].max()))),
        (max(ba_lo_r, float(df["B/A"].min())),
         min(ba_hi_r, float(df["B/A"].max()))),
        (max(wb_lo_r, float(df["W/B"].min())),
         min(wb_hi_r, float(df["W/B"].max()))),
    ]

    # ---- Isolation-Forest applicability domain (as used in phase 9) ----
    ad = joblib.load(PATH_AD_ISOFOR)
    ctx.ad_scaler = ad["scaler"]
    ctx.ad_if_model = ad["if_model"]
    ctx.ad_if_threshold = ad["if_threshold"]
    ad_names = list(ad.get("feature_names", SUPPORT_INPUTS))
    if ad_names != SUPPORT_INPUTS:
        raise RuntimeError(
            "AD scaler feature order mismatch:\n"
            f"  expected {SUPPORT_INPUTS}\n  found   {ad_names}"
        )

    # ---- KDE / Mahalanobis support field (phase-3 dual criterion) ----
    sup = joblib.load(PATH_AD_SUPPORT)
    ctx.sup_scaler = sup["scaler"]
    ctx.sup_kde = sup["kde"]
    ctx.sup_mean_vec = sup["mean_vec"]
    ctx.sup_inv_cov = sup["inv_cov"]
    ctx.sup_d_max = float(sup["D_MAX"])

    sup_train = ctx.df[SUPPORT_INPUTS].values.astype(float)
    sup_std = ctx.sup_scaler.transform(sup_train)
    ctx.sup_train_std = sup_std
    train_log_dens = ctx.sup_kde.score_samples(sup_std)
    ctx.sup_density_max = float(np.exp(train_log_dens).max())

    # ---- sigma(B/A, W/B) model (fitted in phase 9, reused verbatim) ----
    sig = joblib.load(PATH_SIGMA_MODEL)
    ctx.sigma_poly = sig["poly"]
    ctx.sigma_ridge = sig["ridge"]
    ctx.sigma_mae_scale = float(sig["mae_to_sigma_scale"])
    ctx.sigma_floor = float(sig.get("floor", 0.5))
    ctx.sigma_r2_cv = float(sig.get("r2_cv", np.nan))

    # ---- Training-set residual basis (fixed Cu/WAS/curing, as phase 9) ----
    ctx.cs_bias, ctx.cs_rmse, ctx.mean_bag_std = _fit_training_residuals(ctx)

    # ---- Mahalanobis envelope on the (FA/B, B/A, W/B) design trio ----
    trio = df[["FlyAsh/B (%)", "B/A", "W/B"]].values.astype(float)
    ctx.trio_mean = trio.mean(axis=0)
    trio_cov = np.cov(trio.T) + 1e-8 * np.eye(3)
    ctx.trio_cov_inv = np.linalg.inv(trio_cov)
    ctx.trio_mahal_thresh = float(chi2.ppf(MAHAL_QUANTILE, df=3))

    ctx.train_ad_scaled = ctx.ad_scaler.transform(
        df[SUPPORT_INPUTS].values.astype(float))
    ctx.train_cs_obs = df[TARGET_COL].values.astype(float)
    validate_ebm_matrix(ctx, build_ebm_training_input(ctx, df), "training data")

    if verbose:
        print(f"[engine] EBM features ({len(ctx.features)}): {ctx.features}")
        print(f"[engine] use_log_target = {ctx.use_log}")
        print(f"[engine] training strength range: "
              f"[{ctx.strength_min:.2f}, {ctx.strength_max:.2f}] MPa")
        print(f"[engine] global bias = {ctx.cs_bias:+.4f} MPa | "
              f"RMSE = {ctx.cs_rmse:.3f} MPa")
        print(f"[engine] optimisation box: FA/B {ctx.bounds[0]}, "
              f"B/A {ctx.bounds[1]}, W/B {ctx.bounds[2]}")
        print(f"[engine] duplicates retained: exact={ctx.duplicate_summary['exact_rows']}, "
              f"predictor={ctx.duplicate_summary['predictor_rows']}")
    return ctx


# --------------------------------------------------------------------------
# EBM input construction / prediction
# --------------------------------------------------------------------------
def build_ebm_input(ctx, fa_b_pct, ba, wb):
    """Build the 11-feature EBM input row(s) with fixed Cu / WAS / curing."""
    fa_arr = np.atleast_1d(np.asarray(fa_b_pct, dtype=float))
    ba_arr = np.atleast_1d(np.asarray(ba, dtype=float))
    wb_arr = np.atleast_1d(np.asarray(wb, dtype=float))
    n = max(len(fa_arr), len(ba_arr), len(wb_arr))
    fa_arr = np.resize(fa_arr, n)
    ba_arr = np.resize(ba_arr, n)
    wb_arr = np.resize(wb_arr, n)

    cols = {f: np.zeros(n) for f in ctx.features}
    if "Cu" in cols:
        cols["Cu"] = np.log(ctx.cu_fixed) * np.ones(n)
    if "WAS_mm" in cols:
        cols["WAS_mm"] = ctx.was_fixed * np.ones(n)
    if "B/A" in cols:
        cols["B/A"] = ba_arr
    if "W/B" in cols:
        cols["W/B"] = wb_arr
    if "Sand/A" in cols:
        cols["Sand/A"] = np.zeros(n)
    if "curing period" in cols:
        cols["curing period"] = np.log(float(CURING_FIXED)) * np.ones(n)
    if "FlyAsh/B (%)" in cols:
        cols["FlyAsh/B (%)"] = fa_arr
    if "SP (%)" in cols:
        cols["SP (%)"] = np.zeros(n)
    if "FA_used" in cols:
        cols["FA_used"] = (fa_arr > 0).astype(float)
    if "Sand_used" in cols:
        cols["Sand_used"] = np.zeros(n)
    if "SP_used" in cols:
        cols["SP_used"] = np.zeros(n)

    X = np.column_stack([cols[f] for f in ctx.features])
    return validate_ebm_matrix(ctx, X, "conditional design")


def build_ebm_training_input(ctx, df=None):
    """Transform observed raw rows under the contract stored with the model.

    The primary pickle was trained with log(Cu) and log(curing period), unlike
    later source code. All refitted comparison models must use this matrix.
    """
    df = ctx.df if df is None else df
    missing = [f for f in ("Cu", "WAS_mm", "B/A", "W/B", "Sand/A",
                           "curing period", "FlyAsh/B (%)", "SP (%)")
               if f not in df]
    if missing:
        raise ValueError("raw data lack EBM inputs: " + ", ".join(missing))
    n = len(df)
    cols = {f: np.zeros(n) for f in ctx.features}
    values = {
        "Cu": np.log(df["Cu"].to_numpy(float)),
        "WAS_mm": df["WAS_mm"].to_numpy(float),
        "B/A": df["B/A"].to_numpy(float), "W/B": df["W/B"].to_numpy(float),
        "Sand/A": df["Sand/A"].to_numpy(float),
        "curing period": np.log(df["curing period"].to_numpy(float)),
        "FlyAsh/B (%)": df["FlyAsh/B (%)"].to_numpy(float),
        "SP (%)": df["SP (%)"].to_numpy(float),
    }
    cols.update({f: values[f] for f in values if f in cols})
    if "FA_used" in cols:
        cols["FA_used"] = (values["FlyAsh/B (%)"] > 0).astype(float)
    if "Sand_used" in cols:
        cols["Sand_used"] = (values["Sand/A"] > 0).astype(float)
    if "SP_used" in cols:
        cols["SP_used"] = (values["SP (%)"] > 0).astype(float)
    return validate_ebm_matrix(ctx, np.column_stack([cols[f] for f in ctx.features]),
                               "training data")


def validate_ebm_matrix(ctx, X, source):
    """Reject malformed or incompatible saved-model feature matrices."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(ctx.features):
        raise ValueError(f"{source}: expected {len(ctx.features)} EBM columns "
                         f"in saved order {ctx.features}; got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{source}: EBM matrix contains non-finite values")
    for i, name in enumerate(ctx.features):
        if name not in ctx.feature_ranges:
            continue
        lo, hi = ctx.feature_ranges[name]
        if X[:, i].min() < lo - 1e-9 or X[:, i].max() > hi + 1e-9:
            raise ValueError(f"{source}: transformed {name!r} is outside saved "
                             f"model range [{lo}, {hi}]")
    return X


def load_study_map(ctx):
    """Load and prove row-index alignment before attaching study labels."""
    study_map = pd.read_csv(PATH_STUDY_MAP)
    required = {"row_index", "study"}
    if not required.issubset(study_map.columns):
        raise ValueError("study_id_map.csv must contain row_index and study")
    if study_map["row_index"].duplicated().any():
        raise ValueError("study_id_map.csv row_index is not unique")
    expected = set(ctx.row_index)
    found = set(study_map["row_index"].astype(int))
    if found != expected:
        raise ValueError("study_id_map.csv row_index does not exactly match dataset rows")
    return study_map.set_index("row_index").loc[ctx.row_index].reset_index()


def predict_cs_raw(ctx, X):
    """Model prediction in native (log) space for an input matrix."""
    return np.asarray(ctx.model.predict(X), dtype=float)


def predict_cs(ctx, fa_b_pct, ba, wb):
    """Strength prediction in MPa (handles vectorised input)."""
    X = build_ebm_input(ctx, fa_b_pct, ba, wb)
    raw = predict_cs_raw(ctx, X)
    cs = np.exp(raw) if ctx.use_log else raw
    return cs if cs.size > 1 else float(cs[0])


def predict_bag_matrix(ctx, X):
    """Per-bag predictions for an input matrix -> array (n_bags, n)."""
    model = ctx.model
    if not hasattr(model, "bagged_scores_"):
        return None
    orig_scores = [t.copy() for t in model.term_scores_]
    orig_int = (float(model.intercept_) if np.isscalar(model.intercept_)
                else model.intercept_.copy())
    has_bi = hasattr(model, "bagged_intercept_")
    n_bags = len(model.bagged_scores_[0])
    out = np.empty((n_bags, X.shape[0]), dtype=float)
    for b in range(n_bags):
        for t_idx in range(len(model.term_scores_)):
            bags = model.bagged_scores_[t_idx]
            if b < len(bags):
                model.term_scores_[t_idx] = bags[b]
        if has_bi and b < len(model.bagged_intercept_):
            model.intercept_ = model.bagged_intercept_[b]
        raw = model.predict(X)
        out[b] = np.exp(raw) if ctx.use_log else raw
    for t_idx in range(len(model.term_scores_)):
        model.term_scores_[t_idx] = orig_scores[t_idx]
    model.intercept_ = orig_int
    return out


def predict_with_bags(ctx, X):
    """Bag spread (std across outer bags) for each row of X."""
    bag_pred = predict_bag_matrix(ctx, X)
    if bag_pred is None:
        return np.full(X.shape[0], np.nan)
    return bag_pred.std(axis=0)


# --------------------------------------------------------------------------
# CO2 / uncertainty / support metrics
# --------------------------------------------------------------------------
def co2_central(fa_frac, ba, wb):
    """Central embodied-CO2 index (identical formula to phase 9)."""
    fa_frac = np.asarray(fa_frac, dtype=float)
    ba = np.asarray(ba, dtype=float)
    wb = np.asarray(wb, dtype=float)
    binder = ((1 - fa_frac) * CO2_CEMENT_CENTRAL + fa_frac * CO2_FA_CENTRAL) * ba
    return binder + CO2_AGG_CENTRAL + CO2_WATER_CENTRAL * wb * ba


def predict_local_sigma(ctx, ba, wb):
    """Spatially varying local sigma (B/A, W/B) - identical to phase 9.

    Replicates the phase-9 fallback: if the sigma model failed its 5-fold
    CV check (R2_cv < 0, as in the published run: R2_cv = -0.5134), the
    spatial model is rejected and a constant sigma is used:
    max(MEAN_BAG_STD * MAE_TO_SIGMA_SCALE, floor)  (= 1.602 MPa here).
    """
    ba = np.atleast_1d(np.asarray(ba, dtype=float))
    wb = np.atleast_1d(np.asarray(wb, dtype=float))
    n = max(len(ba), len(wb))
    if ctx.sigma_r2_cv < 0.0:
        out = np.full(n, max(ctx.mean_bag_std * ctx.sigma_mae_scale,
                             ctx.sigma_floor))
    else:
        X_q = ctx.sigma_poly.transform(np.column_stack([ba, wb]))
        raw = ctx.sigma_ridge.predict(X_q)
        out = np.maximum(raw * ctx.sigma_mae_scale, ctx.sigma_floor)
    return out if out.size > 1 else float(out[0])


def combine_uncertainty(ctx, bag_std, local_sigma):
    bag_std = np.asarray(bag_std, dtype=float)
    local_sigma = np.asarray(local_sigma, dtype=float)
    out = np.where(np.isnan(bag_std), local_sigma,
                   np.maximum(bag_std, local_sigma))
    if SIGMA_COMBINATION == "quadrature":
        out = np.where(np.isnan(bag_std), local_sigma,
                       np.sqrt(bag_std ** 2 + local_sigma ** 2))
    return out if out.size > 1 else float(out[0])


def ad_input_matrix(ctx, fa_b_pct, ba, wb):
    """8-column applicability-domain input (fixed Cu / WAS / curing)."""
    fa_arr = np.atleast_1d(np.asarray(fa_b_pct, dtype=float))
    ba_arr = np.atleast_1d(np.asarray(ba, dtype=float))
    wb_arr = np.atleast_1d(np.asarray(wb, dtype=float))
    n = max(len(fa_arr), len(ba_arr), len(wb_arr))
    rows = np.column_stack([
        np.resize(fa_arr * 0 + ctx.cu_fixed, n),
        np.resize(fa_arr * 0 + ctx.was_fixed, n),
        np.resize(ba_arr, n),
        np.resize(wb_arr, n),
        np.zeros(n),
        np.full(n, float(CURING_FIXED)),
        np.resize(fa_arr, n),
        np.zeros(n),
    ])
    return rows


def support_metrics(ctx, fa_b_pct, ba, wb):
    """KDE w(x), Mahalanobis distance, NN distance, IF in-domain flag."""
    X_ad = ad_input_matrix(ctx, fa_b_pct, ba, wb)
    sup_std = ctx.sup_scaler.transform(X_ad)

    log_dens = ctx.sup_kde.score_samples(sup_std)
    w_x = np.exp(log_dens) / ctx.sup_density_max

    diff = sup_std - ctx.sup_mean_vec
    mahal_sq = np.einsum("ij,jk,ik->i", diff, ctx.sup_inv_cov, diff)
    mahal_dist = np.sqrt(np.maximum(mahal_sq, 0.0))

    dmat = cdist(sup_std, ctx.sup_train_std)
    nn_dist = dmat.min(axis=1)

    if_scaled = ctx.ad_scaler.transform(X_ad)
    in_ad = (ctx.ad_if_model.score_samples(if_scaled)
             >= -ctx.ad_if_threshold).astype(int)

    return {
        "kde_wx": w_x,
        "mahal_dist": mahal_dist,
        "nn_dist": nn_dist,
        "in_ad": in_ad,
        "in_mahal": (mahal_dist <= ctx.sup_d_max).astype(int),
        "low_support": (w_x < KDE_LOW_FLAG_THRESHOLD).astype(int),
    }


def support_metrics_observed(ctx, df=None):
    """Support metrics for observed raw rows without conditional-value substitution."""
    df = ctx.df if df is None else df
    if not set(SUPPORT_INPUTS).issubset(df.columns):
        raise ValueError("observed data lack required support inputs")
    X_ad = df[SUPPORT_INPUTS].to_numpy(float)
    sup_std = ctx.sup_scaler.transform(X_ad)
    log_dens = ctx.sup_kde.score_samples(sup_std)
    w_x = np.exp(log_dens) / ctx.sup_density_max
    diff = sup_std - ctx.sup_mean_vec
    mahal_sq = np.einsum("ij,jk,ik->i", diff, ctx.sup_inv_cov, diff)
    mahal_dist = np.sqrt(np.maximum(mahal_sq, 0.0))
    distances = cdist(sup_std, ctx.sup_train_std)
    # For the engine's own observed training rows, exclude the trivial self-match.
    if df is ctx.df and distances.shape[0] == distances.shape[1]:
        np.fill_diagonal(distances, np.inf)
    nn_dist = distances.min(axis=1)
    if_scaled = ctx.ad_scaler.transform(X_ad)
    return {"kde_wx": w_x, "mahal_dist": mahal_dist, "nn_dist": nn_dist,
            "in_ad": (ctx.ad_if_model.score_samples(if_scaled)
                      >= -ctx.ad_if_threshold).astype(int),
            "in_mahal": (mahal_dist <= ctx.sup_d_max).astype(int),
            "low_support": (w_x < KDE_LOW_FLAG_THRESHOLD).astype(int)}


def log_extrap_flag(ctx, fa_b_pct, ba, wb):
    """Flag raw log-space predictions outside the training log-strength range."""
    X = build_ebm_input(ctx, fa_b_pct, ba, wb)
    raw = predict_cs_raw(ctx, X)
    return ((raw > ctx.log_s_max) | (raw < ctx.log_s_min)).astype(int)


# --------------------------------------------------------------------------
# Training-residual basis (identical convention to phase 9)
# --------------------------------------------------------------------------
def _fit_training_residuals(ctx):
    df = ctx.df
    fa = df["FlyAsh/B (%)"].values.astype(float)
    ba = df["B/A"].values.astype(float)
    wb = df["W/B"].values.astype(float)
    obs = df[TARGET_COL].values.astype(float)

    X = build_ebm_input(ctx, fa, ba, wb)
    pred = predict_cs_raw(ctx, X)
    pred = np.exp(pred) if ctx.use_log else pred
    residuals = pred - obs
    cs_bias = float(np.mean(residuals)) if CS_BIAS_OVERRIDE is None \
        else float(CS_BIAS_OVERRIDE)
    cs_rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bag_stds = predict_with_bags(ctx, X)
    mean_bag_std = float(np.nanmean(bag_stds))
    return cs_bias, cs_rmse, mean_bag_std


# --------------------------------------------------------------------------
# Phase-9 constraint / objective functions (DE-compatible, 1-D input x)
# --------------------------------------------------------------------------
def mahal_constraint(ctx, x):
    diff = np.asarray(x, dtype=float) - ctx.trio_mean
    return ctx.trio_mahal_thresh - float(diff @ ctx.trio_cov_inv @ diff)


def nearest_train_distance(ctx, x):
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, x[0], x[1], x[2]))
    return float(cdist(scaled, ctx.train_ad_scaled).min())


def local_consistency_constraint(ctx, x):
    cs_pred = float(predict_cs(ctx, x[0], x[1], x[2]))
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, x[0], x[1], x[2]))
    dists = np.sqrt(np.sum((ctx.train_ad_scaled - scaled) ** 2, axis=1))
    near_idx = np.argsort(dists)[:LOCAL_K]
    nb_mean_cs = float(ctx.train_cs_obs[near_idx].mean())
    return LOCAL_TOL - abs(cs_pred - nb_mean_cs)


def ad_constraint(ctx, x):
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, x[0], x[1], x[2]))
    return float(ctx.ad_if_model.score_samples(scaled)[0]) + ctx.ad_if_threshold


def effective_cs(ctx, fa_b_pct, ba, wb):
    """Bias-corrected, uncertainty-discounted strength (phase-9 definition)."""
    cs_mean = np.atleast_1d(np.asarray(
        predict_cs(ctx, fa_b_pct, ba, wb), dtype=float))
    X = build_ebm_input(ctx, fa_b_pct, ba, wb)
    bag_std = predict_with_bags(ctx, X)
    local_sig = np.asarray(predict_local_sigma(ctx, ba, wb), dtype=float)
    unc = np.atleast_1d(combine_uncertainty(ctx, bag_std, local_sig))
    eff = cs_mean - ctx.cs_bias - CS_STD_K * unc
    return cs_mean, eff, bag_std, local_sig, unc


def cs_constraint(ctx, x, cs_target):
    _, eff, _, _, _ = effective_cs(ctx, x[0], x[1], x[2])
    return float(eff[0]) - cs_target


def objective_penalised(ctx, x):
    co2_val = float(co2_central(x[0] / 100.0, x[1], x[2]))
    min_dist = nearest_train_distance(ctx, x)
    excess = max(0.0, min_dist - DIST_PENALTY_THRESHOLD)
    return co2_val + DIST_PENALTY_WEIGHT * excess ** 2


# --------------------------------------------------------------------------
# Vectorised (population-batched) objective / constraints for DE
# --------------------------------------------------------------------------
def _vec_input(x):
    """Vectorised-DE input: scipy passes (N,) or (N, S) -> (S, N)."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(1, -1)
    if x.shape[0] == 3:          # (N=3, S) -> (S, 3)
        return x.T
    return x


def _constraint_input(x):
    return _vec_input(x)


def _maybe_scalar(out, x, constraint=False):
    x = np.asarray(x)
    if x.ndim == 1:
        return float(np.atleast_1d(out)[0])
    out = np.asarray(out)
    if constraint:
        return np.atleast_2d(out)     # (M, S) for vectorised constraints
    return out.ravel()                # (S,) for the vectorised objective


def _objective_vec(ctx, x):
    X = _vec_input(x)
    # With hard constraints, SciPy may evaluate an empty feasible subset.
    # Its vectorised objective contract requires an equally empty result.
    if X.shape[0] == 0:
        return np.empty(0, dtype=float)
    fa, ba, wb = X[:, 0], X[:, 1], X[:, 2]
    co2 = co2_central(fa / 100.0, ba, wb)
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, fa, ba, wb))
    d_min = cdist(scaled, ctx.train_ad_scaled).min(axis=1)
    excess = np.maximum(0.0, d_min - DIST_PENALTY_THRESHOLD)
    out = co2 + DIST_PENALTY_WEIGHT * excess ** 2
    return _maybe_scalar(out, x)


def _cs_constraint_vec(ctx, x, cs_target):
    X = _constraint_input(x)
    _, eff, _, _, _ = effective_cs(ctx, X[:, 0], X[:, 1], X[:, 2])
    return _maybe_scalar(np.atleast_1d(eff) - cs_target, x, constraint=True)


def _ad_constraint_vec(ctx, x):
    X = _constraint_input(x)
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, X[:, 0], X[:, 1],
                                                     X[:, 2]))
    out = ctx.ad_if_model.score_samples(scaled) + ctx.ad_if_threshold
    return _maybe_scalar(out, x, constraint=True)


def _local_consistency_vec(ctx, x):
    X = _constraint_input(x)
    fa, ba, wb = X[:, 0], X[:, 1], X[:, 2]
    cs_pred = np.asarray(predict_cs(ctx, fa, ba, wb), dtype=float)
    scaled = ctx.ad_scaler.transform(ad_input_matrix(ctx, fa, ba, wb))
    dists = cdist(scaled, ctx.train_ad_scaled)
    near_idx = np.argsort(dists, axis=1)[:, :LOCAL_K]
    nb_mean = ctx.train_cs_obs[near_idx].mean(axis=1)
    out = LOCAL_TOL - np.abs(cs_pred - nb_mean)
    return _maybe_scalar(out, x, constraint=True)


def _mahal_constraint_vec(ctx, x):
    X = _constraint_input(x)
    diff = X - ctx.trio_mean
    out = ctx.trio_mahal_thresh - np.einsum(
        "ij,jk,ik->i", diff, ctx.trio_cov_inv, diff)
    return _maybe_scalar(out, x, constraint=True)


def optimize_target(ctx, cs_target, seed=42, maxiter=600, popsize=20,
                    tol=1e-6):
    """Single constrained-DE carbon minimisation at one strength target.

    Uses scipy's vectorised mode: the objective and constraints accept the
    whole population as an (n, 3) array, so the 16-bag EBM variance is
    evaluated in batches.  Mathematically identical to the phase-9
    formulation; only the evaluation batching differs.  SciPy's constrained
    trust-constr polish step is disabled because it can replace a feasible DE
    point with an infeasible one without rechecking constraints.
    """
    from scipy.optimize import differential_evolution, NonlinearConstraint

    constraints = [
        NonlinearConstraint(
            lambda x: _cs_constraint_vec(ctx, x, cs_target), 0.0, np.inf),
        NonlinearConstraint(lambda x: _ad_constraint_vec(ctx, x),
                            0.0, np.inf),
        NonlinearConstraint(
            lambda x: _local_consistency_vec(ctx, x), 0.0, np.inf),
        NonlinearConstraint(lambda x: _mahal_constraint_vec(ctx, x),
                            0.0, np.inf),
    ]
    result = differential_evolution(
        lambda x: _objective_vec(ctx, x),
        ctx.bounds,
        constraints=constraints,
        seed=int(seed),
        maxiter=maxiter,
        tol=tol,
        workers=1,
        polish=False,
        popsize=popsize,
        vectorized=True,
        updating="deferred",
    )
    fa_opt, ba_opt, wb_opt = [float(v) for v in result.x]
    cs_mean, eff, bag_std, local_sig, unc = effective_cs(
        ctx, fa_opt, ba_opt, wb_opt)
    sup = support_metrics(ctx, fa_opt, ba_opt, wb_opt)
    record = {
        "cs_target": cs_target,
        "seed": int(seed),
        "success": bool(result.success),
        "feasible": bool(getattr(result, "maxcv", np.inf) <= 1e-8),
        "max_constraint_violation": float(getattr(result, "maxcv", np.nan)),
        "message": str(result.message)[:120],
        "nit": int(getattr(result, "nit", -1)),
        "FA_B_pct": fa_opt,
        "BA": ba_opt,
        "WB": wb_opt,
        "CS_pred_MPa": float(np.atleast_1d(cs_mean)[0]),
        "CS_effective_MPa": float(np.atleast_1d(eff)[0]),
        "CS_bag_std": float(np.atleast_1d(bag_std)[0]),
        "local_sigma": float(np.atleast_1d(local_sig)[0]),
        "combined_uncertainty": float(np.atleast_1d(unc)[0]),
        "CO2_index": float(co2_central(fa_opt / 100.0, ba_opt, wb_opt)),
        "kde_wx": float(sup["kde_wx"][0]),
        "mahal_dist": float(sup["mahal_dist"][0]),
        "nn_dist": float(sup["nn_dist"][0]),
        "in_ad": int(sup["in_ad"][0]),
        "in_mahal": int(sup["in_mahal"][0]),
        "low_support": int(sup["low_support"][0]),
        "log_extrap": int(log_extrap_flag(ctx, fa_opt, ba_opt, wb_opt)[0]),
        "mahal_trio": float(
            (np.array([fa_opt, ba_opt, wb_opt]) - ctx.trio_mean)
            @ ctx.trio_cov_inv
            @ (np.array([fa_opt, ba_opt, wb_opt]) - ctx.trio_mean)),
        "objective": float(result.fun) if np.isfinite(result.fun) else np.nan,
    }
    return record


# --------------------------------------------------------------------------
# LHS candidate sampling + achievable-fraction accounting
# --------------------------------------------------------------------------
def run_lhs(ctx, n=500, seed=0):
    """Latin-hypercube candidates inside the optimisation box (phase-9 seed)."""
    from scipy.stats import qmc
    lo = np.array([b[0] for b in ctx.bounds])
    hi = np.array([b[1] for b in ctx.bounds])
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    raw = qmc.scale(sampler.random(n), lo, hi)
    return raw[:, 0], raw[:, 1], raw[:, 2]


def evaluate_candidates(ctx, fa, ba, wb):
    """Vectorised evaluation of candidate mixtures with all diagnostics."""
    fa = np.asarray(fa, dtype=float)
    ba = np.asarray(ba, dtype=float)
    wb = np.asarray(wb, dtype=float)
    X = build_ebm_input(ctx, fa, ba, wb)
    raw = predict_cs_raw(ctx, X)
    cs = np.exp(raw) if ctx.use_log else raw
    _, eff, bag_std, local_sig, unc = effective_cs(ctx, fa, ba, wb)
    sup = support_metrics(ctx, fa, ba, wb)
    out = pd.DataFrame({
        "FA_B_pct": fa,
        "BA": ba,
        "WB": wb,
        "CS_pred_MPa": cs,
        "CS_effective_MPa": eff,
        "CS_bag_std": bag_std,
        "local_sigma": local_sig,
        "combined_uncertainty": unc,
        "CO2_index": co2_central(fa / 100.0, ba, wb),
        "kde_wx": sup["kde_wx"],
        "mahal_dist": sup["mahal_dist"],
        "nn_dist": sup["nn_dist"],
        "in_ad": sup["in_ad"],
        "in_mahal": sup["in_mahal"],
        "low_support": sup["low_support"],
        "log_extrap": ((raw > ctx.log_s_max) | (raw < ctx.log_s_min)).astype(int),
    })
    return out


def reliable_mask(eval_df):
    """Phase-9 reliability screening: in AD + inside Mahalanobis envelope
    + adequate KDE support + no log-space extrapolation."""
    return ((eval_df["in_ad"] == 1) & (eval_df["in_mahal"] == 1)
            & (eval_df["low_support"] == 0) & (eval_df["log_extrap"] == 0))


def achievable_fractions(eval_df, cs_target):
    """Nominal vs reliability-bounded achievability at one target."""
    n = len(eval_df)
    meets_strength = eval_df["CS_effective_MPa"].values >= cs_target
    nominal = meets_strength & (eval_df["log_extrap"].values == 0)
    reliable = nominal & reliable_mask(eval_df).values
    return {
        "n_candidates": n,
        "frac_nominal": float(nominal.mean()),
        "frac_reliable": float(reliable.mean()),
        "frac_in_ad": float((eval_df["in_ad"] == 1).mean()),
        "frac_in_mahal": float((eval_df["in_mahal"] == 1).mean()),
        "frac_kde_supported": float((eval_df["low_support"] == 0).mean()),
        "frac_no_log_extrap": float((eval_df["log_extrap"] == 0).mean()),
    }


if __name__ == "__main__":
    c = load_context(verbose=True)
    print("[engine] smoke test: predicting CS at (FA/B=30, B/A=0.2, W/B=0.35)")
    print("  CS =", predict_cs(c, 30.0, 0.2, 0.35), "MPa")
    print("  CO2 =", co2_central(0.30, 0.2, 0.35))
    print("[engine] OK")
