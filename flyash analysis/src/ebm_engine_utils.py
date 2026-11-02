"""
Safe wrappers around ebm_engine for use in analysis scripts.
Handles the fact that ebm_engine uses fixed curing=28 for prediction,
but we need to predict at observed curing periods.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any
import sys
from pathlib import Path

# Add project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "08_analysis_code_suite" / "shared"))

from ebm_engine import load_context, Context


def load_engine_context(verbose: bool = False) -> Context:
    """Load the full ebm_engine context."""
    return load_context(verbose=verbose)


def predict_cs_safe(ctx: Context, fa_b_pct: float, ba: float, wb: float, curing_days: float) -> float:
    """
    Predict strength at specific curing period.
    ebm_engine.build_ebm_input uses fixed CURING_FIXED=28, so we need custom input building.
    """
    # Build input manually with correct curing period
    fa_arr = np.atleast_1d(np.asarray(fa_b_pct, dtype=float))
    ba_arr = np.atleast_1d(np.asarray(ba, dtype=float))
    wb_arr = np.atleast_1d(np.asarray(wb, dtype=float))
    curing_arr = np.atleast_1d(np.asarray(curing_days, dtype=float))
    n = max(len(fa_arr), len(ba_arr), len(wb_arr), len(curing_arr))
    fa_arr = np.resize(fa_arr, n)
    ba_arr = np.resize(ba_arr, n)
    wb_arr = np.resize(wb_arr, n)
    curing_arr = np.resize(curing_arr, n)
    
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
        cols["curing period"] = np.log(curing_arr)
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
    X = validate_ebm_matrix(ctx, X, "custom prediction")
    
    raw = ctx.model.predict(X)
    cs = np.exp(raw) if ctx.use_log else raw
    return float(cs[0]) if cs.size == 1 else cs


def validate_ebm_matrix(ctx: Context, X: np.ndarray, source: str) -> np.ndarray:
    """Validate EBM input matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(ctx.features):
        raise ValueError(f"{source}: expected {len(ctx.features)} EBM columns; got {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError(f"{source}: EBM matrix contains non-finite values")
    for i, name in enumerate(ctx.features):
        if name not in ctx.feature_ranges:
            continue
        lo, hi = ctx.feature_ranges[name]
        if X[:, i].min() < lo - 1e-9 or X[:, i].max() > hi + 1e-9:
            raise ValueError(f"{source}: transformed {name!r} outside saved model range [{lo}, {hi}]")
    return X


def co2_central_safe(fa_frac: float, ba: float, wb: float) -> float:
    """CO2 calculation matching ebm_engine.co2_central exactly."""
    CO2_CEMENT_CENTRAL = 1.043 + 225 * 0.000062
    CO2_FA_CENTRAL = 0.00397 + 160 * 0.000062
    CO2_AGG_CENTRAL = 0.00503 + 85 * 0.000062
    CO2_WATER_CENTRAL = 0.00212
    
    fa_frac = np.asarray(fa_frac, dtype=float)
    ba = np.asarray(ba, dtype=float)
    wb = np.asarray(wb, dtype=float)
    binder = ((1 - fa_frac) * CO2_CEMENT_CENTRAL + fa_frac * CO2_FA_CENTRAL) * ba
    return float(binder + CO2_AGG_CENTRAL + CO2_WATER_CENTRAL * wb * ba)


def support_metrics_safe(ctx: Context, fa_b_pct: float, ba: float, wb: float, curing_days: float) -> Dict[str, Any]:
    """Support metrics at specific curing period."""
    from scipy.spatial.distance import cdist
    
    fa_arr = np.atleast_1d(np.asarray(fa_b_pct, dtype=float))
    ba_arr = np.atleast_1d(np.asarray(ba, dtype=float))
    wb_arr = np.atleast_1d(np.asarray(wb, dtype=float))
    curing_arr = np.atleast_1d(np.asarray(curing_days, dtype=float))
    n = max(len(fa_arr), len(ba_arr), len(wb_arr), len(curing_arr))
    fa_arr = np.resize(fa_arr, n)
    ba_arr = np.resize(ba_arr, n)
    wb_arr = np.resize(wb_arr, n)
    curing_arr = np.resize(curing_arr, n)
    
    # 8-column AD input (matches ebm_engine.ad_input_matrix but with variable curing)
    rows = np.column_stack([
        np.resize(fa_arr * 0 + ctx.cu_fixed, n),
        np.resize(fa_arr * 0 + ctx.was_fixed, n),
        np.resize(ba_arr, n),
        np.resize(wb_arr, n),
        np.zeros(n),
        curing_arr,  # <-- variable curing, not fixed 28
        np.resize(fa_arr, n),
        np.zeros(n),
    ])
    
    sup_std = ctx.sup_scaler.transform(rows)
    log_dens = ctx.sup_kde.score_samples(sup_std)
    w_x = np.exp(log_dens) / ctx.sup_density_max
    
    diff = sup_std - ctx.sup_mean_vec
    mahal_sq = np.einsum("ij,jk,ik->i", diff, ctx.sup_inv_cov, diff)
    mahal_dist = np.sqrt(np.maximum(mahal_sq, 0.0))
    
    dmat = cdist(sup_std, ctx.sup_train_std)
    nn_dist = dmat.min(axis=1)
    
    if_scaled = ctx.ad_scaler.transform(rows)
    in_ad = (ctx.ad_if_model.score_samples(if_scaled) >= -ctx.ad_if_threshold).astype(int)
    
    # Log extrapolation check
    # Build EBM input for log-space prediction - MUST match ctx.features order exactly
    # ctx.features = ['Cu', 'WAS_mm', 'B/A', 'W/B', 'Sand/A', 'FlyAsh/B (%)', 'SP (%)', 'curing period', 'FA_used', 'Sand_used', 'SP_used']
    X_ebm = np.column_stack([
        np.log(ctx.cu_fixed) * np.ones(n),      # Cu (log)
        ctx.was_fixed * np.ones(n),             # WAS_mm
        ba_arr,                                 # B/A
        wb_arr,                                 # W/B
        np.zeros(n),                            # Sand/A
        fa_arr,                                 # FlyAsh/B (%)
        np.zeros(n),                            # SP (%)
        np.log(curing_arr),                     # curing period (log)
        (fa_arr > 0).astype(float),             # FA_used
        np.zeros(n),                            # Sand_used
        np.zeros(n),                            # SP_used
    ])
    X_ebm = validate_ebm_matrix(ctx, X_ebm, "log_extrap_check")
    raw = ctx.model.predict(X_ebm)
    log_extrap = ((raw > ctx.log_s_max) | (raw < ctx.log_s_min)).astype(int)
    
    return {
        "kde_wx": float(w_x[0]) if n == 1 else w_x,
        "mahal_dist": float(mahal_dist[0]) if n == 1 else mahal_dist,
        "nn_dist": float(nn_dist[0]) if n == 1 else nn_dist,
        "in_ad": int(in_ad[0]) if n == 1 else in_ad,
        "in_mahal": int((mahal_dist <= ctx.sup_d_max)[0]) if n == 1 else (mahal_dist <= ctx.sup_d_max).astype(int),
        "low_support": int((w_x < 0.1)[0]) if n == 1 else (w_x < 0.1).astype(int),
        "log_extrap": int(log_extrap[0]) if n == 1 else log_extrap,
    }


def reliable_mask_safe(sup: Dict[str, Any]) -> bool:
    """Check if a single point passes all support criteria."""
    return (sup.get("in_ad", 0) == 1 and 
            sup.get("in_mahal", 0) == 1 and 
            sup.get("low_support", 1) == 0 and 
            sup.get("log_extrap", 1) == 0)


def get_training_data(ctx: Context) -> pd.DataFrame:
    """Get the training dataset from context."""
    return ctx.df.copy()


def get_ebm_main_effect(ctx: Context, feature_name: str) -> pd.DataFrame:
    """Extract main effect shape function from EBM."""
    expl = ctx.model.explain_global()
    for i, term in enumerate(ctx.model.term_features_):
        if len(term) == 1 and ctx.features[term[0]] == feature_name:
            d = expl.data(i)
            names = np.array(d['names'])
            scores = np.array(d['scores'])
            
            # Align names to scores
            if len(names) == len(scores) + 1:
                x = (names[:-1] + names[1:]) / 2.0
            elif len(names) == len(scores):
                x = names
            else:
                raise ValueError(f"Shape mismatch for {feature_name}")
            
            # Convert from log scale if needed
            if feature_name in ["Cu", "curing period"]:
                x = np.exp(x)
            
            return pd.DataFrame({"x": x, "y": scores, "feature": feature_name})
    return pd.DataFrame()


def get_ebm_interaction(ctx: Context, feat1: str, feat2: str) -> pd.DataFrame:
    """Extract interaction surface from EBM."""
    from scipy.interpolate import RegularGridInterpolator
    
    expl = ctx.model.explain_global()
    f1_idx = ctx.features.index(feat1)
    f2_idx = ctx.features.index(feat2)
    
    for i, term in enumerate(ctx.model.term_features_):
        if len(term) == 2 and set(term) == {f1_idx, f2_idx}:
            d = expl.data(i)
            ln = np.array(d['left_names'], dtype=float)
            rn = np.array(d['right_names'], dtype=float)
            sc = np.array(d['scores'])
            
            if sc.ndim == 1:
                n_l = len(ln)
                n_r = len(sc) // n_l
                sc = sc.reshape(n_l, n_r)
            
            # Align
            if len(ln) == sc.shape[0] + 1:
                ln = (ln[:-1] + ln[1:]) / 2.0
            if len(rn) == sc.shape[1] + 1:
                rn = (rn[:-1] + rn[1:]) / 2.0
            if sc.shape[0] != len(ln) and sc.shape[1] == len(ln):
                sc = sc.T
            
            # Convert from log scale if needed
            if feat1 in ["Cu", "curing period"]:
                ln = np.exp(ln)
            if feat2 in ["Cu", "curing period"]:
                rn = np.exp(rn)
            
            # Return as long-form DataFrame
            rows = []
            for xi, xv in enumerate(ln):
                for yi, yv in enumerate(rn):
                    rows.append({"x1": xv, "x2": yv, "interaction": sc[xi, yi]})
            return pd.DataFrame(rows)
    return pd.DataFrame()