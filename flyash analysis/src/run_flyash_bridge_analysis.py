#!/usr/bin/env python3
"""
Main entry point for Fly-Ash Bridge Analysis.

Runs all four parts:
  A. Data Audit & Model Contract
  B. +10 pp Fly-Ash Substitution Analysis
  C. Resampling Stability Analysis (5×10 = 50 fits)
  D. Interaction Fraction Decomposition
  + Figures & Report

Usage:
  python src/run_flyash_bridge_analysis.py --run-all
  python src/run_flyash_bridge_analysis.py --substitution-only
  python src/run_flyash_bridge_analysis.py --stability-only
  python src/run_flyash_bridge_analysis.py --render-only
  python src/run_flyash_bridge_analysis.py --n-repeats 10 --n-splits 5 --seed 42
"""
import argparse
import sys
import time
from pathlib import Path
import pandas as pd

# Add analysis src to path
ANALYSIS_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ANALYSIS_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from utils import setup_logging, record_environment, set_global_seeds, get_paths
import logging


def run_part_a(logger):
    """Part A: Data Audit & Model Contract"""
    logger.info("=" * 60)
    logger.info("PART A: Data Audit & Model Contract")
    logger.info("=" * 60)
    from data_audit import run_data_audit, write_audit_outputs
    from ebm_engine_utils import load_engine_context
    
    ctx = load_engine_context(verbose=False)
    audit, df, ctx = run_data_audit(logger)
    write_audit_outputs(audit, df, ctx, logger)
    logger.info("Part A complete.")
    return ctx


def run_part_b(logger, ctx):
    """Part B: +10 pp Fly-Ash Substitution"""
    logger.info("=" * 60)
    logger.info("PART B: +10 pp Fly-Ash Substitution Analysis")
    logger.info("=" * 60)
    from substitution_analysis import run_substitution_analysis
    
    run_substitution_analysis(logger, ctx)
    logger.info("Part B complete.")


def run_part_c(logger, ctx, n_splits=5, n_repeats=10, seed=42):
    """Part C: Resampling Stability Analysis"""
    logger.info("=" * 60)
    logger.info(f"PART C: Resampling Stability ({n_repeats}×{n_splits} = {n_repeats*n_splits} fits)")
    logger.info("=" * 60)
    from stability_analysis import run_stability_analysis
    
    run_stability_analysis(logger, ctx, n_splits=n_splits, n_repeats=n_repeats, seed=seed)
    logger.info("Part C complete.")


def run_part_d(logger):
    """Part D: Interaction Fraction Decomposition"""
    logger.info("=" * 60)
    logger.info("PART D: Interaction Fraction Decomposition")
    logger.info("=" * 60)
    from importance_decomposition import run_importance_decomposition
    
    run_importance_decomposition(logger)
    logger.info("Part D complete.")


def run_figures(logger):
    """Render all figures"""
    logger.info("=" * 60)
    logger.info("RENDERING FIGURES")
    logger.info("=" * 60)
    from plotting import render_all_figures
    
    render_all_figures(logger)
    logger.info("Figures complete.")


def run_report(logger):
    """Generate final report"""
    logger.info("=" * 60)
    logger.info("GENERATING REPORT")
    logger.info("=" * 60)
    from reporting import generate_report
    
    generate_report(logger)
    logger.info("Report complete.")


def main():
    parser = argparse.ArgumentParser(description="Fly-Ash Bridge Analysis")
    parser.add_argument("--run-all", action="store_true", help="Run all parts (A+B+C+D+Figures+Report)")
    parser.add_argument("--substitution-only", action="store_true", help="Run only Part B")
    parser.add_argument("--stability-only", action="store_true", help="Run only Part C")
    parser.add_argument("--render-only", action="store_true", help="Render only figures (requires prior outputs)")
    parser.add_argument("--n-repeats", type=int, default=10, help="Number of CV repeats (default: 10)")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of CV splits (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--skip-stability", action="store_true", help="Skip Part C (useful for quick runs)")
    
    args = parser.parse_args()
    
    # Default to --run-all if no specific mode selected
    if not any([args.run_all, args.substitution_only, args.stability_only, args.render_only]):
        args.run_all = True
    
    # Setup
    logger = setup_logging("main")
    logger.info(f"Fly-Ash Bridge Analysis started")
    logger.info(f"Args: {vars(args)}")
    
    # Record environment
    env = record_environment(logger)
    set_global_seeds(args.seed)
    
    start_time = time.time()
    ctx = None
    
    try:
        if args.run_all or args.substitution_only or args.stability_only:
            # Need context for B and C
            from ebm_engine_utils import load_engine_context
            ctx = load_engine_context(verbose=True)
        
        if args.run_all:
            # Part A
            ctx = run_part_a(logger)
            
            # Part B
            run_part_b(logger, ctx)
            
            # Part C (unless skipped)
            if not args.skip_stability:
                run_part_c(logger, ctx, n_splits=args.n_splits, n_repeats=args.n_repeats, seed=args.seed)
            else:
                logger.info("Skipping Part C (--skip-stability)")
            
            # Part D
            run_part_d(logger)
            
            # Figures
            run_figures(logger)
            
            # Report
            run_report(logger)
            
        elif args.substitution_only:
            run_part_b(logger, ctx)
            
        elif args.stability_only:
            run_part_c(logger, ctx, n_splits=args.n_splits, n_repeats=args.n_repeats, seed=args.seed)
            
        elif args.render_only:
            run_figures(logger)
            run_report(logger)
        
        elapsed = time.time() - start_time
        logger.info(f"Total elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        
        # Print completion summary
        print_completion_summary(logger)
        
    except Exception as e:
        logger.exception(f"Analysis failed: {e}")
        sys.exit(1)


def print_completion_summary(logger):
    """Print compact completion summary."""
    paths = get_paths()
    
    # Count outputs
    outputs = list(paths["outputs"].glob("*.csv"))
    figures = list(paths["figures"].glob("*.png"))
    
    # Load key metrics
    try:
        funnel = pd.read_csv(paths["outputs"] / "flyash_plus10_support_funnel.csv")
        n_total = int(funnel['total'].values[0])
        n_eligible = int(funnel['eligible'].values[0])
        n_supported = int(funnel['both_supported'].values[0])
    except:
        n_total = n_eligible = n_supported = 0
    
    try:
        overall = pd.read_csv(paths["outputs"] / "flyash_plus10_summary_overall.csv")
        if len(overall) > 0:
            dstr = overall['delta_strength_median'].values[0]
            dco2 = overall['delta_co2_median'].values[0]
        else:
            dstr = dco2 = 0
    except:
        dstr = dco2 = 0
    
    summary = f"""
{'='*60}
COMPLETION SUMMARY
{'='*60}
Scripts created:
  - src/utils.py
  - src/ebm_engine_utils.py
  - src/data_audit.py
  - src/substitution_analysis.py
  - src/stability_analysis.py
  - src/importance_decomposition.py
  - src/plotting.py
  - src/reporting.py
  - src/run_flyash_bridge_analysis.py

Outputs created ({len(outputs)} CSVs):
  - outputs/data_audit.csv
  - outputs/flyash_plus10_row_level_all_candidates.csv
  - outputs/flyash_plus10_supported_candidates.csv
  - outputs/flyash_plus10_summary_overall.csv
  - outputs/flyash_plus10_summary_by_age_ba_wb.csv
  - outputs/flyash_plus10_support_funnel.csv
  - outputs/flyash_plus10_feature_ranges_and_eligibility.csv
  - outputs/stability_resample_manifest.csv
  - outputs/stability_cv_performance.csv
  - outputs/stability_main_effect_grid.csv
  - outputs/stability_main_effect_turning_regions.csv
  - outputs/stability_interaction_grid.csv
  - outputs/stability_interaction_summary.csv
  - outputs/feature_importance_with_interaction_fraction.csv
  - outputs/environment.json

Figures created ({len(figures)} PNGs @ 300 dpi):
  - fig_01_flyash_plus10_strength_carbon_overview.png
  - fig_02_flyash_plus10_by_age_ba_wb.png
  - fig_03_flyash_plus10_strength_vs_carbon.png
  - fig_04_flyash_main_effect_resampling_stability.png
  - fig_05_key_main_effects_resampling_stability.png
  - fig_06_flyash_ba_interaction_resampling_stability.png
  - fig_07_key_interactions_resampling_stability.png
  - fig_08_feature_importance_decomposition.png

Documentation:
  - documentation/model_and_data_contract.md
  - documentation/ANALYSIS_SPEC.md
  - FLYASH_ANALYSIS_REPORT.md

Key Metrics:
  - Observed candidates: {n_total}
  - Eligible after +10 pp domain rule: {n_eligible}
  - Evidence-supported at both endpoints: {n_supported}
  - Median ΔStrength (supported): {dstr:.2f} MPa
  - Median ΔCO₂ (supported): {dco2:.4f}

Assumptions/Limitations requiring human review:
  1. Primary EBM uses fixed Cu/WAS/curing=28 for prediction; Part B uses observed curing per row
  2. Support criteria applied at both endpoints (original + counterfactual)
  3. Resampling stability uses 5×10 CV with same stratification as training
  4. CO2 factors from ebm_engine.py (central estimates, fixed transport)
  5. Counterfactuals hold all non-FA variables constant (no mix redesign)
"""
    # Some Windows consoles still use cp1252 and cannot print symbols such as
    # Δ or CO₂.  The files have already been written; make the optional
    # console summary non-fatal as well.
    encoding = sys.stdout.encoding or "utf-8"
    print(summary.encode(encoding, errors="replace").decode(encoding))
    logger.info(summary)


if __name__ == "__main__":
    import pandas as pd
    main()
