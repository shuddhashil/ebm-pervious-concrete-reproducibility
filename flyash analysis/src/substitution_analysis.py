"""
Part B — +10 pp Fly Ash Substitution Analysis
Observation-anchored counterfactual analysis with evidence support.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings("ignore")

from utils import get_paths, setup_logging
from ebm_engine_utils import load_engine_context, predict_cs_safe, co2_central_safe, support_metrics_safe, reliable_mask_safe


def run_substitution_analysis(logger, ctx) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete +10 pp fly ash substitution analysis."""
    paths = get_paths()
    
    # Load observed data
    df = pd.read_csv(paths["dataset"])
    logger.info(f"Loaded {len(df)} observed rows")
    
    # Observed FA range
    fa_min = df["FlyAsh/B (%)"].min()
    fa_max = df["FlyAsh/B (%)"].max()
    logger.info(f"Observed FA/B range: [{fa_min:.1f}, {fa_max:.1f}]%")
    
    # ---- Build counterfactuals ----
    rows = []
    support_funnel = {"total": 0, "eligible": 0, "both_supported": 0}
    
    for idx, row in df.iterrows():
        support_funnel["total"] += 1
        
        fa_obs = row["FlyAsh/B (%)"]
        fa_cf = fa_obs + 10.0
        
        # Domain check
        if fa_cf > fa_max:
            eligible = False
            domain_flag = "exceeds_max"
        elif fa_cf < fa_min:
            eligible = False
            domain_flag = "below_min"
        else:
            eligible = True
            domain_flag = "ok"
        
        # Preserve all other variables
        ba = row["B/A"]
        wb = row["W/B"]
        curing = row["curing period"]
        cu = row["Cu"]
        was = row["WAS_mm"]
        sand_a = row["Sand/A"]
        sp_pct = row["SP (%)"]
        
        # Predictions
        pred_obs = np.nan
        pred_cf = np.nan
        delta_strength = np.nan
        co2_obs = np.nan
        co2_cf = np.nan
        delta_co2 = np.nan
        
        sup_obs = {}
        sup_cf = {}
        supported_obs = False
        supported_cf = False
        both_supported = False
        
        if eligible:
            support_funnel["eligible"] += 1
            
            # Predict strength (uses observed curing period per row)
            pred_obs = predict_cs_safe(ctx, fa_obs, ba, wb, curing)
            pred_cf = predict_cs_safe(ctx, fa_cf, ba, wb, curing)
            
            if not (np.isnan(pred_obs) or np.isnan(pred_cf)):
                delta_strength = pred_cf - pred_obs
            
            # CO2
            co2_obs = co2_central_safe(fa_obs / 100.0, ba, wb)
            co2_cf = co2_central_safe(fa_cf / 100.0, ba, wb)
            delta_co2 = co2_cf - co2_obs
            
            # Support metrics for both endpoints
            sup_obs = support_metrics_safe(ctx, fa_obs, ba, wb, curing)
            sup_cf = support_metrics_safe(ctx, fa_cf, ba, wb, curing)
            
            supported_obs = reliable_mask_safe(sup_obs)
            supported_cf = reliable_mask_safe(sup_cf)
            both_supported = supported_obs and supported_cf
            
            if both_supported:
                support_funnel["both_supported"] += 1
        
        rows.append({
            "source_row_id": int(idx),
            # Observed
            "fa_obs": fa_obs,
            "ba_obs": ba,
            "wb_obs": wb,
            "curing_obs": curing,
            "cu_obs": cu,
            "was_obs": was,
            "sand_a_obs": sand_a,
            "sp_pct_obs": sp_pct,
            "strength_obs": row["Strength_con"],
            # Counterfactual
            "fa_cf": fa_cf if eligible else np.nan,
            "ba_cf": ba,
            "wb_cf": wb,
            "curing_cf": curing,
            "domain_flag": domain_flag,
            "eligible": eligible,
            # Predictions
            "pred_obs_mpa": pred_obs,
            "pred_cf_mpa": pred_cf,
            "delta_strength_mpa": delta_strength,
            # CO2
            "co2_obs": co2_obs,
            "co2_cf": co2_cf,
            "delta_co2": delta_co2,
            # Support - observed
            "sup_obs_kde_wx": sup_obs.get("kde_wx", np.nan),
            "sup_obs_mahal_dist": sup_obs.get("mahal_dist", np.nan),
            "sup_obs_nn_dist": sup_obs.get("nn_dist", np.nan),
            "sup_obs_in_ad": sup_obs.get("in_ad", np.nan),
            "sup_obs_in_mahal": sup_obs.get("in_mahal", np.nan),
            "sup_obs_low_support": sup_obs.get("low_support", np.nan),
            "sup_obs_log_extrap": sup_obs.get("log_extrap", np.nan),
            "supported_obs": supported_obs,
            # Support - counterfactual
            "sup_cf_kde_wx": sup_cf.get("kde_wx", np.nan),
            "sup_cf_mahal_dist": sup_cf.get("mahal_dist", np.nan),
            "sup_cf_nn_dist": sup_cf.get("nn_dist", np.nan),
            "sup_cf_in_ad": sup_cf.get("in_ad", np.nan),
            "sup_cf_in_mahal": sup_cf.get("in_mahal", np.nan),
            "sup_cf_low_support": sup_cf.get("low_support", np.nan),
            "sup_cf_log_extrap": sup_cf.get("log_extrap", np.nan),
            "supported_cf": supported_cf,
            # Combined
            "both_endpoints_supported": both_supported,
        })
    
    all_df = pd.DataFrame(rows)
    
    # ---- Save row-level all candidates ----
    all_csv = paths["outputs"] / "flyash_plus10_row_level_all_candidates.csv"
    all_df.to_csv(all_csv, index=False)
    logger.info(f"All candidates: {all_csv} ({len(all_df)} rows)")
    
    # ---- Supported subset ----
    supported_df = all_df[all_df["both_endpoints_supported"] == True].copy()
    supp_csv = paths["outputs"] / "flyash_plus10_supported_candidates.csv"
    supported_df.to_csv(supp_csv, index=False)
    logger.info(f"Supported candidates: {supp_csv} ({len(supported_df)} rows)")
    
    # ---- Support funnel ----
    funnel_df = pd.DataFrame([support_funnel])
    funnel_csv = paths["outputs"] / "flyash_plus10_support_funnel.csv"
    funnel_df.to_csv(funnel_csv, index=False)
    logger.info(f"Support funnel: {funnel_csv}")
    
    # ---- Feature ranges & eligibility ----
    ranges_df = pd.DataFrame([{
        "variable": "FlyAsh/B (%)",
        "observed_min": fa_min,
        "observed_max": fa_max,
        "counterfactual_min": fa_min + 10,
        "counterfactual_max": fa_max + 10,
        "eligible_count": support_funnel["eligible"],
        "total_count": support_funnel["total"],
        "eligibility_rate": support_funnel["eligible"] / support_funnel["total"],
    }])
    ranges_csv = paths["outputs"] / "flyash_plus10_feature_ranges_and_eligibility.csv"
    ranges_df.to_csv(ranges_csv, index=False)
    logger.info(f"Feature ranges: {ranges_csv}")
    
    # ---- Subgroup summaries ----
    summary_overall, summary_by_subgroup = compute_subgroup_summaries(supported_df, all_df, logger)
    
    overall_csv = paths["outputs"] / "flyash_plus10_summary_overall.csv"
    summary_overall.to_csv(overall_csv, index=False)
    logger.info(f"Overall summary: {overall_csv}")
    
    subgroup_csv = paths["outputs"] / "flyash_plus10_summary_by_age_ba_wb.csv"
    summary_by_subgroup.to_csv(subgroup_csv, index=False)
    logger.info(f"Subgroup summary: {subgroup_csv}")
    
    return all_df, supported_df, summary_overall, summary_by_subgroup, funnel_df, ranges_df


def compute_subgroup_summaries(supported_df: pd.DataFrame, all_df: pd.DataFrame, logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute summaries for all required subgroups."""
    
    # Overall summary
    overall = summarize_group(supported_df, all_df, "overall", logger)
    
    # Subgroups
    subgroups = []
    
    # Exact curing ages
    for age in [7, 28]:
        supp_sub = supported_df[supported_df["curing_obs"] == age]
        all_sub = all_df[all_df["curing_obs"] == age]
        subgroups.append(summarize_group(supp_sub, all_sub, f"age_{age}d", logger))
    
    # B/A tertiles (from eligible observed)
    eligible = all_df[all_df["eligible"] == True]
    if len(eligible) > 0:
        ba_tertiles = eligible["ba_obs"].quantile([1/3, 2/3]).values
        for i, (lo, hi) in enumerate([(-np.inf, ba_tertiles[0]), (ba_tertiles[0], ba_tertiles[1]), (ba_tertiles[1], np.inf)]):
            mask = (eligible["ba_obs"] > lo) & (eligible["ba_obs"] <= hi)
            supp_sub = supported_df[supported_df["source_row_id"].isin(eligible[mask]["source_row_id"])]
            all_sub = all_df[all_df["source_row_id"].isin(eligible[mask]["source_row_id"])]
            subgroups.append(summarize_group(supp_sub, all_sub, f"ba_tertile_{i+1}", logger, 
                                           extra={"ba_range": f"({lo:.3f}, {hi:.3f}]"}))
    
    # W/B tertiles
    if len(eligible) > 0:
        wb_tertiles = eligible["wb_obs"].quantile([1/3, 2/3]).values
        for i, (lo, hi) in enumerate([(-np.inf, wb_tertiles[0]), (wb_tertiles[0], wb_tertiles[1]), (wb_tertiles[1], np.inf)]):
            mask = (eligible["wb_obs"] > lo) & (eligible["wb_obs"] <= hi)
            supp_sub = supported_df[supported_df["source_row_id"].isin(eligible[mask]["source_row_id"])]
            all_sub = all_df[all_df["source_row_id"].isin(eligible[mask]["source_row_id"])]
            subgroups.append(summarize_group(supp_sub, all_sub, f"wb_tertile_{i+1}", logger,
                                           extra={"wb_range": f"({lo:.3f}, {hi:.3f}]"}))

    # Age-by-tertile summaries are required for the figure that compares
    # exact 7- and 28-day behaviour.  The earlier implementation calculated
    # age-only and tertile-only results, which cannot answer that question.
    for age in [7, 28]:
        age_eligible = eligible[eligible["curing_obs"] == age]
        for prefix, col, label in [("ba", "ba_obs", "ba_range"), ("wb", "wb_obs", "wb_range")]:
            if age_eligible.empty:
                continue
            cutpoints = eligible[col].quantile([1 / 3, 2 / 3]).values
            bounds = [(-np.inf, cutpoints[0]), (cutpoints[0], cutpoints[1]), (cutpoints[1], np.inf)]
            for i, (lo, hi) in enumerate(bounds, start=1):
                source_ids = age_eligible.loc[
                    (age_eligible[col] > lo) & (age_eligible[col] <= hi), "source_row_id"
                ]
                supp_sub = supported_df[supported_df["source_row_id"].isin(source_ids)]
                all_sub = all_df[all_df["source_row_id"].isin(source_ids)]
                subgroups.append(summarize_group(
                    supp_sub, all_sub, f"age_{age}d_{prefix}_tertile_{i}", logger,
                    extra={label: f"({lo:.3f}, {hi:.3f}]"},
                ))
    
    # ``summarize_group`` deliberately returns a one-row DataFrame so the
    # caller can also use it directly for the overall result.  Concatenating
    # those rows is required here; passing a list of DataFrames to the
    # DataFrame constructor creates an inhomogeneous object array.
    subgroup_df = pd.concat(subgroups, ignore_index=True, sort=False)
    return overall, subgroup_df


def summarize_group(supp_df: pd.DataFrame, all_df: pd.DataFrame, group_name: str, logger, extra: dict = None) -> pd.DataFrame:
    """Summarize a single subgroup."""
    n_orig = len(all_df)
    n_eligible = int(all_df["eligible"].sum()) if "eligible" in all_df.columns else 0
    n_supported = len(supp_df)
    
    if n_supported == 0:
        logger.warning(f"  {group_name}: NO supported cases (n_orig={n_orig}, n_eligible={n_eligible})")
        return pd.DataFrame([{
            "group": group_name,
            "n_original": n_orig,
            "n_eligible": n_eligible,
            "n_supported": 0,
            "warning": "No evidence-supported cases",
            **(extra or {})
        }])
    
    # Strength metrics
    ds = supp_df["delta_strength_mpa"].dropna()
    dc = supp_df["delta_co2"].dropna()
    
    # Carbon reduction among negative delta_co2
    co2_red = dc[dc < 0]
    median_co2_reduction = float(co2_red.median()) if len(co2_red) > 0 else np.nan
    
    summary = {
        "group": group_name,
        "n_original": n_orig,
        "n_eligible": n_eligible,
        "n_supported": n_supported,
        "delta_strength_median": float(ds.median()),
        "delta_strength_mean": float(ds.mean()),
        "delta_strength_q25": float(ds.quantile(0.25)),
        "delta_strength_q75": float(ds.quantile(0.75)),
        "prop_strength_nonneg": float((ds >= 0).mean()),
        "prop_strength_ge_minus1": float((ds >= -1).mean()),
        "delta_co2_median": float(dc.median()),
        "delta_co2_mean": float(dc.mean()),
        "median_co2_reduction": median_co2_reduction,
        "warning": "Small n" if n_supported < 10 else "",
    }
    # Add extra fields as strings to avoid DataFrame inhomogeneous issues
    if extra:
        for k, v in extra.items():
            summary[k] = str(v)
    
    logger.info(f"  {group_name}: n_orig={n_orig}, n_elig={n_eligible}, n_supp={n_supported}, "
                f"dStr={summary['delta_strength_median']:.2f}, dCO2={summary['delta_co2_median']:.4f}")
    
    return pd.DataFrame([summary])


if __name__ == "__main__":
    logger = setup_logging("substitution")
    
    # Load engine context
    import sys
    paths = get_paths()
    sys.path.insert(0, str(paths["project_root"] / "08_analysis_code_suite" / "shared"))
    from ebm_engine import load_context
    ctx = load_context(verbose=False)
    
    run_substitution_analysis(logger, ctx)
    logger.info("Part B complete.")
