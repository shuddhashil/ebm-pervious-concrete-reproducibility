"""
Part C — EBM Resampling Stability Analysis (5×10 = 50 fits)
"""
import numpy as np
import pandas as pd
import joblib
import warnings
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import RepeatedStratifiedKFold
from interpret.glassbox import ExplainableBoostingRegressor
from tqdm import tqdm

warnings.filterwarnings("ignore")

from utils import get_paths, setup_logging, set_global_seeds
from ebm_engine_utils import Context, load_engine_context, get_training_data


def run_stability_analysis(logger, ctx, n_splits: int = 5, n_repeats: int = 10, seed: int = 42):
    """Run 50-fit resampling stability analysis."""
    paths = get_paths()
    set_global_seeds(seed)
    
    # Load training data and prepare
    df = get_training_data(ctx)
    logger.info(f"Training data: {len(df)} rows")
    
    # Get EBM configuration from training code reference
    # We need to replicate the EXACT configuration from EBM.py
    ebm_config = get_ebm_config_from_primary_bundle(paths["primary_ebm"], ctx)
    logger.info(f"EBM config: {ebm_config}")
    
    # Prepare features and target exactly as training
    X, y, feature_names = prepare_training_matrix(df, ctx)
    logger.info(f"Feature matrix: {X.shape}, features: {feature_names}")
    
    # Stratification key (same as training code)
    strat_key = make_strat_key(df, y)
    
    # Repeated stratified K-fold
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed + 999)
    
    # Common evaluation grids
    grids = create_evaluation_grids(df, ctx)
    logger.info(f"Evaluation grids: {[k + ':' + str(len(v)) for k, v in grids.items()]}")
    
    # Storage
    manifest_rows = []
    main_effect_curves = {feat: [] for feat in ["FlyAsh/B (%)", "B/A", "W/B", "curing period"]}
    interaction_curves = {
        "FlyAsh/B (%)__B/A": [],
        "B/A__curing period": [],
        "W/B__curing period": [],
    }
    
    cv_perfs = []
    
    # The original evaluation fits used sample weights.  Re-use those
    # row-aligned weights if they are present in the saved primary bundle.
    bundle = joblib.load(paths["primary_ebm"])
    sample_weights = np.asarray(bundle.get("sample_weights"), dtype=float) if bundle.get("sample_weights") is not None else None
    if sample_weights is not None and len(sample_weights) != len(X):
        raise ValueError("Saved sample weights are not aligned with the training dataset.")

    for run_idx, (train_idx, test_idx) in enumerate(tqdm(cv.split(X, strat_key), total=n_splits*n_repeats, desc="Resampling")):
        repeat_num = run_idx // n_splits
        fold_num = run_idx % n_splits
        split_seed = seed + run_idx
        
        set_global_seeds(split_seed)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train EBM
        fold_weights = sample_weights[train_idx] if sample_weights is not None else None
        model = train_ebm_fold(ebm_config, X_train, y_train, feature_names, split_seed, fold_weights)
        
        # Evaluate on test
        test_pred = model.predict(X_test)
        test_pred_orig = np.exp(test_pred) if ebm_config["use_log_target"] else test_pred
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        test_r2 = r2_score(y_test, test_pred_orig)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred_orig))
        test_mae = mean_absolute_error(y_test, test_pred_orig)
        
        # Manifest entry
        manifest_rows.append({
            "split_id": run_idx,
            "repeat": repeat_num + 1,
            "fold": fold_num + 1,
            "seed": split_seed,
            "train_n": len(train_idx),
            "test_n": len(test_idx),
            "test_r2": test_r2,
            "test_rmse": test_rmse,
            "test_mae": test_mae,
        })
        
        cv_perfs.append({"run": run_idx, "r2": test_r2, "rmse": test_rmse, "mae": test_mae})
        
        # Evaluate main effects on common grids
        for feat in main_effect_curves:
            grid_x = grids[feat]
            curve = evaluate_main_effect(model, feature_names, feat, grid_x, ebm_config["use_log_target"])
            main_effect_curves[feat].append(curve)
        
        # Evaluate interactions on common grids
        for inter_name in interaction_curves:
            feat1, feat2 = inter_name.split("__")
            grid1 = grids[feat1]
            grid2 = grids[feat2]
            surface = evaluate_interaction(model, feature_names, feat1, feat2, grid1, grid2, ebm_config["use_log_target"])
            interaction_curves[inter_name].append(surface)
        
        if (run_idx + 1) % 10 == 0:
            logger.info(f"  Completed {run_idx + 1}/50 fits")
    
    # Aggregate results
    manifest_df = pd.DataFrame(manifest_rows)
    cv_perf_df = pd.DataFrame(cv_perfs)
    
    # Save manifest and CV performance
    manifest_csv = paths["outputs"] / "stability_resample_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)
    logger.info(f"Manifest: {manifest_csv}")
    
    cv_perf_csv = paths["outputs"] / "stability_cv_performance.csv"
    cv_perf_df.to_csv(cv_perf_csv, index=False)
    logger.info(f"CV Performance: {cv_perf_csv}")
    
    # Process main effects
    main_grid_df, turning_df = process_main_effects(main_effect_curves, grids, logger)
    main_grid_csv = paths["outputs"] / "stability_main_effect_grid.csv"
    main_grid_df.to_csv(main_grid_csv, index=False)
    logger.info(f"Main effect grid: {main_grid_csv}")
    
    turning_csv = paths["outputs"] / "stability_main_effect_turning_regions.csv"
    turning_df.to_csv(turning_csv, index=False)
    logger.info(f"Turning regions: {turning_csv}")
    
    # Process interactions
    inter_grid_df, inter_summary_df = process_interactions(interaction_curves, grids, df, ctx, logger)
    inter_grid_csv = paths["outputs"] / "stability_interaction_grid.csv"
    inter_grid_df.to_csv(inter_grid_csv, index=False)
    logger.info(f"Interaction grid: {inter_grid_csv}")
    
    inter_sum_csv = paths["outputs"] / "stability_interaction_summary.csv"
    inter_summary_df.to_csv(inter_sum_csv, index=False)
    logger.info(f"Interaction summary: {inter_sum_csv}")
    
    return manifest_df, main_grid_df, inter_grid_df


def get_ebm_config_from_primary_bundle(primary_ebm_path: Path, ctx: Context) -> dict:
    """Use the saved model's tuned parameters without hard-coded paths."""
    bundle = joblib.load(primary_ebm_path)
    final_params = bundle.get("final_params", {})
    
    config = {
        "learning_rate": final_params.get("learning_rate", 0.05),
        "max_bins": final_params.get("max_bins", 64),
        "min_samples_leaf": final_params.get("min_samples_leaf", 8),
        "max_leaves": final_params.get("max_leaves", 3),
        "smoothing_rounds": final_params.get("smoothing_rounds", 80),
        "max_interaction_bins": final_params.get("max_interaction_bins", 32),
        "interactions": final_params.get("interactions", []),  # Will be set below
        # Evaluation models in EBM.py used four outer bags; the 16 bags in
        # the final model are a different, full-data uncertainty procedure.
        "outer_bags": 4,
        "inner_bags": 0,
        "n_jobs": 2,
        "validation_size": 0.15,
        "early_stopping_rounds": 40,
        "use_log_target": True,
        "random_state": 42,
    }
    
    config["interactions"] = [tuple(pair) for pair in final_params["interactions"]]
    config["feature_names"] = list(ctx.features)
    
    return config


def prepare_training_matrix(df: pd.DataFrame, ctx: Context) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare training matrix exactly as in training code."""
    # Features in the same order as training
    feature_names = ctx.features
    
    # Build matrix using ebm_engine's training input builder
    from ebm_engine import build_ebm_training_input
    X = build_ebm_training_input(ctx, df)
    y = df["Strength_con"].values.astype(float)
    
    return X, y, feature_names


def make_strat_key(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Create stratification key matching training code."""
    y_s = pd.Series(y)
    y_bins = pd.qcut(y_s, q=3, labels=False, duplicates='drop').astype(str)
    curing = df['curing period']
    c_bins = pd.cut(curing, bins=[0, 7, 28, 1000], labels=False).astype(str)
    if 'FA_used' in df.columns:
        fa_bins = df['FA_used'].astype(int).astype(str)
    else:
        fa_bins = (df['FlyAsh/B (%)'] > 0).astype(int).astype(str)
    return (y_bins + "_" + c_bins + "_" + fa_bins).values


def create_evaluation_grids(df: pd.DataFrame, ctx: Context) -> Dict[str, np.ndarray]:
    """Create common evaluation grids for all resampled models."""
    grids = {}
    
    # FlyAsh/B (%) - observed range, 50 points
    fa_min, fa_max = df["FlyAsh/B (%)"].min(), df["FlyAsh/B (%)"].max()
    grids["FlyAsh/B (%)"] = np.linspace(fa_min, fa_max, 50)
    
    # B/A - observed range
    ba_min, ba_max = df["B/A"].min(), df["B/A"].max()
    grids["B/A"] = np.linspace(ba_min, ba_max, 50)
    
    # W/B - observed range
    wb_min, wb_max = df["W/B"].min(), df["W/B"].max()
    grids["W/B"] = np.linspace(wb_min, wb_max, 50)
    
    # Curing spans orders of magnitude, so use a log-spaced grid in its full
    # observed range. Exact 7 and 28 days remain covered by the substitution
    # analysis, while this curve assesses the whole main effect.
    cure_min = max(float(df["curing period"].min()), np.finfo(float).eps)
    cure_max = float(df["curing period"].max())
    grids["curing period"] = np.geomspace(cure_min, cure_max, 50)
    
    return grids


def train_ebm_fold(config: dict, X_train: np.ndarray, y_train: np.ndarray,
                   feature_names: List[str], seed: int, sample_weight=None) -> ExplainableBoostingRegressor:
    """Train a single EBM fold."""
    # Transform target
    y_train_internal = np.log(y_train) if config["use_log_target"] else y_train
    
    model = ExplainableBoostingRegressor(
        learning_rate=config["learning_rate"],
        max_bins=config["max_bins"],
        min_samples_leaf=config["min_samples_leaf"],
        max_leaves=config["max_leaves"],
        smoothing_rounds=config["smoothing_rounds"],
        max_interaction_bins=config["max_interaction_bins"],
        interactions=config["interactions"],
        outer_bags=config["outer_bags"],
        inner_bags=config["inner_bags"],
        n_jobs=config["n_jobs"],
        validation_size=config["validation_size"],
        early_stopping_rounds=config["early_stopping_rounds"],
        random_state=seed,
    )
    
    model.fit(X_train, y_train_internal, sample_weight=sample_weight)
    return model


def evaluate_main_effect(model: ExplainableBoostingRegressor, feature_names: List[str],
                         feat: str, grid_x: np.ndarray, use_log: bool) -> np.ndarray:
    """Evaluate one native EBM main-effect curve on a common raw-value grid."""
    # Find term index
    feat_idx = feature_names.index(feat)
    term_idx = next((i for i, tf in enumerate(model.term_features_) if tf == (feat_idx,)), None)
    if term_idx is None:
        return np.full(len(grid_x), np.nan)
    
    # ``bins_`` includes sentinel bins and varies across interpret versions.
    # explain_global provides the displayed bin boundaries and matching native
    # scores, which is the stable public representation to interpolate.
    data = model.explain_global().data(term_idx)
    edges = np.asarray(data["names"], dtype=float)
    scores = np.asarray(data["scores"], dtype=float)
    if len(edges) != len(scores) + 1:
        return np.full(len(grid_x), np.nan)
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    
    # Convert from log scale if needed
    if feat in ["Cu", "curing period"]:
        bin_centers = np.exp(bin_centers)
    
    # Interpolate onto grid
    curve = np.interp(grid_x, bin_centers, scores, left=np.nan, right=np.nan)
    return curve


def evaluate_interaction(model: ExplainableBoostingRegressor, feature_names: List[str],
                         feat1: str, feat2: str, grid1: np.ndarray, grid2: np.ndarray, 
                         use_log: bool) -> np.ndarray:
    """Evaluate interaction surface on grid."""
    f1_idx = feature_names.index(feat1)
    f2_idx = feature_names.index(feat2)
    
    # Find term index
    term_idx = next((i for i, tf in enumerate(model.term_features_) 
                     if set(tf) == {f1_idx, f2_idx}), None)
    if term_idx is None:
        return np.full((len(grid1), len(grid2)), np.nan)
    
    data = model.explain_global().data(term_idx)
    scores = np.asarray(data["scores"], dtype=float)
    left_edges = np.asarray(data["left_names"], dtype=float)
    right_edges = np.asarray(data["right_names"], dtype=float)
    # Global explanations are stored in the model's native feature order, not
    # necessarily in the caller's requested order.  Re-orient the surface so
    # the first grid always corresponds to ``feat1``.
    if tuple(model.term_features_[term_idx]) != (f1_idx, f2_idx):
        scores = scores.T
        left_edges, right_edges = right_edges, left_edges
    if len(left_edges) != scores.shape[0] + 1 or len(right_edges) != scores.shape[1] + 1:
        return np.full((len(grid1), len(grid2)), np.nan)
    left_centers = (left_edges[:-1] + left_edges[1:]) / 2.0
    right_centers = (right_edges[:-1] + right_edges[1:]) / 2.0
    
    # Convert from log scale
    if feat1 in ["Cu", "curing period"]:
        left_centers = np.exp(left_centers)
    if feat2 in ["Cu", "curing period"]:
        right_centers = np.exp(right_centers)
    
    # Ensure correct orientation
    if scores.shape[0] != len(left_centers) and scores.shape[1] == len(left_centers):
        scores = scores.T
        left_centers, right_centers = right_centers, left_centers
    
    # Interpolate on grid
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((left_centers, right_centers), scores, 
                                     bounds_error=False, fill_value=np.nan)
    
    X1, X2 = np.meshgrid(grid1, grid2, indexing='ij')
    points = np.column_stack([X1.ravel(), X2.ravel()])
    surface = interp(points).reshape(X1.shape)
    
    return surface


def process_main_effects(curves_dict: Dict, grids: Dict, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate main effect curves across resamples."""
    rows = []
    turning_rows = []
    
    for feat, curves in curves_dict.items():
        if len(curves) == 0:
            continue
        curves_arr = np.array(curves)  # (n_fits, n_grid)
        grid_x = grids[feat]
        
        # Skip if all NaN
        if np.all(np.isnan(curves_arr)):
            continue
        
        # Statistics across fits
        median_curve = np.nanmedian(curves_arr, axis=0)
        p025 = np.nanpercentile(curves_arr, 2.5, axis=0)
        p975 = np.nanpercentile(curves_arr, 97.5, axis=0)
        std_curve = np.nanstd(curves_arr, axis=0)
        
        # Sign consistency of local slopes
        slopes = np.gradient(curves_arr, grid_x, axis=1)
        median_slope = np.nanmedian(slopes, axis=0)
        sign_consistency = np.mean(np.sign(slopes) == np.sign(median_slope), axis=0)
        
        for i, xv in enumerate(grid_x):
            rows.append({
                "feature": feat,
                "x_value": xv,
                "median": median_curve[i],
                "p025": p025[i],
                "p975": p975[i],
                "std": std_curve[i],
                "sign_consistency": sign_consistency[i],
                "n_fits": np.sum(~np.isnan(curves_arr[:, i])),
            })
        
        # Identify turning regions (where median slope changes sign)
        valid = ~np.isnan(median_slope)
        if valid.sum() > 2:
            slope_sign = np.sign(median_slope[valid])
            sign_changes = np.where(np.diff(slope_sign) != 0)[0]
            for sc in sign_changes:
                x_left = grid_x[valid][sc]
                x_right = grid_x[valid][sc + 1]
                turning_rows.append({
                    "feature": feat,
                    "turning_region": f"({x_left:.2f}, {x_right:.2f})",
                    "left_x": x_left,
                    "right_x": x_right,
                    "slope_before": slope_sign[sc],
                    "slope_after": slope_sign[sc + 1],
                })
    
    return pd.DataFrame(rows), pd.DataFrame(turning_rows)


def process_interactions(curves_dict: Dict, grids: Dict, df: pd.DataFrame, ctx: Context, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate interaction surfaces across resamples."""
    rows = []
    summary_rows = []
    
    for inter_name, surfaces in curves_dict.items():
        if len(surfaces) == 0:
            continue
        surfaces_arr = np.array(surfaces)  # (n_fits, n_grid1, n_grid2)
        feat1, feat2 = inter_name.split("__")
        grid1 = grids[feat1]
        grid2 = grids[feat2]
        
        if np.all(np.isnan(surfaces_arr)):
            continue
        
        # Statistics
        median_surf = np.nanmedian(surfaces_arr, axis=0)
        p025 = np.nanpercentile(surfaces_arr, 2.5, axis=0)
        p975 = np.nanpercentile(surfaces_arr, 97.5, axis=0)
        
        # Sign consistency
        median_sign = np.sign(median_surf)
        sign_consist = np.mean(np.sign(surfaces_arr) == median_sign, axis=0)
        
        # Support mask (observed data density)
        support_mask = compute_support_mask(df, ctx, feat1, feat2, grid1, grid2)
        
        for i, x1 in enumerate(grid1):
            for j, x2 in enumerate(grid2):
                rows.append({
                    "interaction": inter_name,
                    "x1": x1,
                    "x2": x2,
                    "median": median_surf[i, j],
                    "p025": p025[i, j],
                    "p975": p975[i, j],
                    "sign_consistency": sign_consist[i, j],
                    "support_mask": support_mask[i, j],
                    "stability_flag": "stable" if sign_consist[i, j] > 0.8 and support_mask[i, j] > 0.5 else 
                                     ("moderate" if sign_consist[i, j] > 0.6 else "unstable"),
                })
        
        # Summary
        valid_cells = ~np.isnan(median_surf)
        if valid_cells.sum() > 0:
            summary_rows.append({
                "interaction": inter_name,
                "n_grid_cells": int(valid_cells.sum()),
                "median_abs_interaction": float(np.nanmedian(np.abs(median_surf))),
                "frac_stable": float(np.mean(sign_consist[valid_cells] > 0.8)),
                "frac_moderate": float(np.mean((sign_consist[valid_cells] > 0.6) & (sign_consist[valid_cells] <= 0.8))),
                "frac_unstable": float(np.mean(sign_consist[valid_cells] <= 0.6)),
                "mean_support": float(np.nanmean(support_mask)),
            })
    
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def compute_support_mask(df: pd.DataFrame, ctx: Context, feat1: str, feat2: str, 
                         grid1: np.ndarray, grid2: np.ndarray) -> np.ndarray:
    """Compute observed data density mask for interaction grid."""
    # Simple 2D histogram of observed data
    from scipy.stats import gaussian_kde
    
    obs1 = df[feat1].values
    obs2 = df[feat2].values
    
    # KDE on observed data
    try:
        kde = gaussian_kde(np.vstack([obs1, obs2]))
        X1, X2 = np.meshgrid(grid1, grid2, indexing='ij')
        points = np.vstack([X1.ravel(), X2.ravel()])
        density = kde(points).reshape(X1.shape)
        # Normalize to 0-1
        density = density / (density.max() + 1e-10)
        return density
    except:
        # Fallback: 2D histogram
        H, _, _ = np.histogram2d(obs1, obs2, bins=[grid1, grid2])
        H = H.T  # Transpose to match grid orientation
        return H / (H.max() + 1e-10)


if __name__ == "__main__":
    logger = setup_logging("stability")
    ctx = load_engine_context(verbose=False)
    run_stability_analysis(logger, ctx)
    logger.info("Part C complete.")
