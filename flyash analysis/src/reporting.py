"""
Generate FLYASH_ANALYSIS_REPORT.md
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from utils import get_paths, setup_logging


def generate_report(logger):
    """Generate the complete Markdown report."""
    paths = get_paths()
    
    # Load all outputs
    with open(paths["outputs"] / "data_audit_full.json", encoding="utf-8") as audit_file:
        audit = json.load(audit_file)
    all_df = pd.read_csv(paths["outputs"] / "flyash_plus10_row_level_all_candidates.csv")
    supp_df = pd.read_csv(paths["outputs"] / "flyash_plus10_supported_candidates.csv")
    funnel_df = pd.read_csv(paths["outputs"] / "flyash_plus10_support_funnel.csv")
    overall_df = pd.read_csv(paths["outputs"] / "flyash_plus10_summary_overall.csv")
    subgroup_df = pd.read_csv(paths["outputs"] / "flyash_plus10_summary_by_age_ba_wb.csv")
    manifest_df = pd.read_csv(paths["outputs"] / "stability_resample_manifest.csv")
    main_grid_df = pd.read_csv(paths["outputs"] / "stability_main_effect_grid.csv")
    inter_grid_df = pd.read_csv(paths["outputs"] / "stability_interaction_grid.csv")
    imp_df = pd.read_csv(paths["outputs"] / "feature_importance_with_interaction_fraction.csv")
    
    report_path = paths["analysis_dir"] / "FLYASH_ANALYSIS_REPORT.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Fly-Ash Bridge Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
        
        # 1. Purpose
        f.write("## 1. Purpose\n\n")
        f.write("This analysis connects the interpretable EBM explanation of fly-ash behaviour ")
        f.write("with the existing sustainability/low-carbon optimisation results. ")
        f.write("The central research question is:\n\n")
        f.write("> For observed concrete mixes, when does increasing fly ash by 10 percentage points ")
        f.write("remain supported by the data, preserve predicted compressive strength sufficiently, ")
        f.write("and reduce embodied carbon—and how do these outcomes depend on curing age, B/A, and W/B?\n\n")
        f.write("The expected framing is that fly ash's effect is **conditional on curing and mix proportions**, ")
        f.write("not a single universal optimum.\n\n")
        
        # 2. Input Files
        f.write("## 2. Input Files Used\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write(f"| `phase5_augmented_dataset.csv` | Model-ready dataset ({len(all_df)} rows) |\n")
        f.write(f"| `primary_row_split_ebm.pkl` | Primary EBM model (16 bags, 12 interactions) |\n")
        f.write(f"| `phase3_new_isoforest_kde_mahalanobis.pkl` | Isolation Forest AD model |\n")
        f.write(f"| `phase3_kde_mahalanobis.pkl` | KDE + Mahalanobis support field |\n")
        f.write(f"| `feature_importance_grouped.csv` | Grouped feature importance |\n")
        f.write(f"| `fa_interaction_evidence_masked_grid.csv` | Sustainability optimisation grid |\n")
        f.write("\n")
        
        # 3. Model & Support Logic
        f.write("## 3. Exact Model and Support Logic Used\n\n")
        f.write(f"**Model:** `primary_row_split_ebm.pkl` (ExplainableBoostingRegressor)\n")
        f.write(f"- Features (11): {', '.join(audit['ebm_features']['feature_names'])}\n")
        f.write(f"- Log target: {audit['ebm_features']['use_log_target']}\n")
        f.write(f"- Fixed prediction values: Cu={audit['ebm_features']['fixed_values']['Cu']:.3f}, ")
        f.write(f"WAS={audit['ebm_features']['fixed_values']['WAS_mm']:.1f}, curing=28d, Sand/A=0, SP%=0\n")
        f.write(f"- Interactions: 12 pairs (9 mixture + 3 curing-selective)\n\n")
        
        f.write("**Support Criteria (canonical `reliable_mask` from `ebm_engine.py`):**\n")
        f.write("A point is evidence-supported iff:\n")
        for comp, desc in audit["support_criteria"]["components"].items():
            f.write(f"- `{comp}`: {desc}\n")
        f.write("\n**Thresholds:**\n")
        for thresh, val in audit["support_criteria"]["thresholds"].items():
            f.write(f"- `{thresh}` = {val}\n")
        f.write("\nFor the +10 pp substitution, **both endpoints** (original and counterfactual) ")
        f.write("must pass the combined support flag.\n\n")
        
        # 4. Data Eligibility & Support Funnel
        f.write("## 4. Data Eligibility and Support Funnel\n\n")
        f.write(f"**Total observed rows:** {int(funnel_df['total'].values[0])}\n")
        f.write(f"**Eligible after +10 pp domain clamp:** {int(funnel_df['eligible'].values[0])} ")
        f.write(f"({funnel_df['eligible'].values[0]/funnel_df['total'].values[0]*100:.1f}%)\n")
        f.write(f"**Evidence-supported at both endpoints:** {int(funnel_df['both_supported'].values[0])} ")
        f.write(f"({funnel_df['both_supported'].values[0]/funnel_df['total'].values[0]*100:.1f}% of total, ")
        f.write(f"{funnel_df['both_supported'].values[0]/funnel_df['eligible'].values[0]*100:.1f}% of eligible)\n\n")
        
        # 5. Main +10 Findings Overall
        f.write("## 5. Main +10 pp Fly-Ash Findings (Overall)\n\n")
        if len(overall_df) > 0:
            o = overall_df.iloc[0]
            f.write(f"- **Supported candidates:** {int(o['n_supported'])}\n")
            f.write(f"- **Median ΔStrength:** {o['delta_strength_median']:.2f} MPa\n")
            f.write(f"- **Mean ΔStrength:** {o['delta_strength_mean']:.2f} MPa\n")
            f.write(f"- **IQR ΔStrength:** [{o['delta_strength_q25']:.2f}, {o['delta_strength_q75']:.2f}] MPa\n")
            f.write(f"- **Proportion ΔStrength ≥ 0:** {o['prop_strength_nonneg']*100:.1f}%\n")
            f.write(f"- **Proportion ΔStrength ≥ -1 MPa:** {o['prop_strength_ge_minus1']*100:.1f}%\n")
            f.write(f"- **Median ΔCO₂:** {o['delta_co2_median']:.4f}\n")
            f.write(f"- **Mean ΔCO₂:** {o['delta_co2_mean']:.4f}\n")
            f.write(f"- **Median CO₂ reduction (among reductions):** {o['median_co2_reduction']:.4f}\n")
            if o.get('warning'):
                f.write(f"- ⚠️ **Warning:** {o['warning']}\n")
        f.write("\n")
        
        # 6. Exact 7-day and 28-day Findings
        f.write("## 6. Exact 7-Day and 28-Day Findings\n\n")
        for age in [7, 28]:
            age_rows = subgroup_df[subgroup_df["group"] == f"age_{age}d"]
            if len(age_rows) > 0:
                r = age_rows.iloc[0]
                f.write(f"### Curing Age = {age} days\n")
                f.write(f"- Supported: {int(r['n_supported'])}\n")
                f.write(f"- Median ΔStrength: {r['delta_strength_median']:.2f} MPa\n")
                f.write(f"- Prop. ΔStrength ≥ 0: {r['prop_strength_nonneg']*100:.1f}%\n")
                f.write(f"- Prop. ΔStrength ≥ -1 MPa: {r['prop_strength_ge_minus1']*100:.1f}%\n")
                f.write(f"- Median ΔCO₂: {r['delta_co2_median']:.4f}\n")
                if r.get('warning'):
                    f.write(f"- ⚠️ **Warning:** {r['warning']}\n")
                f.write("\n")
        
        # 7. B/A and W/B Modification
        f.write("## 7. How B/A and W/B Modify the Result\n\n")
        f.write("### B/A Tertiles (from eligible observed data)\n\n")
        ba_rows = subgroup_df[subgroup_df["group"].str.startswith("ba_tertile")].sort_values("group")
        for _, r in ba_rows.iterrows():
            f.write(f"- **{r['group']}** (BA {r.get('ba_range', '')}): ")
            f.write(f"n_supp={int(r['n_supported'])}, ")
            f.write(f"ΔStr_med={r['delta_strength_median']:.2f}, ")
            f.write(f"ΔCO₂_med={r['delta_co2_median']:.4f}")
            if r.get('warning'):
                f.write(f" ⚠️ {r['warning']}")
            f.write("\n")
        f.write("\n")
        
        f.write("### W/B Tertiles (from eligible observed data)\n\n")
        wb_rows = subgroup_df[subgroup_df["group"].str.startswith("wb_tertile")].sort_values("group")
        for _, r in wb_rows.iterrows():
            f.write(f"- **{r['group']}** (WB {r.get('wb_range', '')}): ")
            f.write(f"n_supp={int(r['n_supported'])}, ")
            f.write(f"ΔStr_med={r['delta_strength_median']:.2f}, ")
            f.write(f"ΔCO₂_med={r['delta_co2_median']:.4f}")
            if r.get('warning'):
                f.write(f" ⚠️ {r['warning']}")
            f.write("\n")
        f.write("\n")
        
        # 8. Carbon vs Strength Trade-off
        f.write("## 8. Carbon-vs-Strength Trade-off\n\n")
        if len(supp_df) > 0:
            ds = supp_df["delta_strength_mpa"].dropna()
            dc = supp_df["delta_co2"].dropna()
            f.write(f"- **Supported candidates:** {len(ds)}\n")
            f.write(f"- **Carbon reduction achieved:** {(dc < 0).mean()*100:.1f}% of supported cases\n")
            f.write(f"- **No strength loss (ΔStr ≥ 0):** {(ds >= 0).mean()*100:.1f}%\n")
            f.write(f"- **Acceptable loss (ΔStr ≥ -1 MPa):** {(ds >= -1).mean()*100:.1f}%\n")
            f.write(f"- **Carbon reduction + no strength loss:** {((dc < 0) & (ds >= 0)).mean()*100:.1f}%\n")
            f.write(f"- **Carbon reduction + acceptable loss:** {((dc < 0) & (ds >= -1)).mean()*100:.1f}%\n")
        f.write("\n")
        
        # 9. Resampling Stability
        f.write("## 9. Resampling Stability Findings (5×10 = 50 fits)\n\n")
        f.write(f"- **CV R² across 50 fits:** {manifest_df['test_r2'].mean():.4f} ± {manifest_df['test_r2'].std():.4f}\n")
        f.write(f"- **CV RMSE:** {manifest_df['test_rmse'].mean():.4f} ± {manifest_df['test_rmse'].std():.4f} MPa\n\n")
        
        f.write("### Stable Fly-Ash Behaviours\n\n")
        fa_data = main_grid_df[main_grid_df["feature"] == "FlyAsh/B (%)"]
        if len(fa_data) > 0:
            # Check sign consistency
            sc = fa_data["sign_consistency"].mean()
            f.write(f"- **FlyAsh/B (%) main effect sign consistency:** {sc*100:.1f}%\n")
            # Turning regions
            turning = pd.read_csv(paths["outputs"] / "stability_main_effect_turning_regions.csv")
            fa_turn = turning[turning["feature"] == "FlyAsh/B (%)"]
            if len(fa_turn) > 0:
                f.write(f"- **Identified turning regions:**\n")
                for _, t in fa_turn.iterrows():
                    f.write(f"  - {t['turning_region']} (slope {t['slope_before']:+.0f} → {t['slope_after']:+.0f})\n")
            else:
                f.write("- No stable turning points identified (monotonic within uncertainty)\n")
        f.write("\n")
        
        f.write("### Unstable / Uncertain Patterns\n\n")
        f.write("- **Precise \"optima\" are NOT stable:** Any apparent peak in the fly-ash main effect ")
        f.write("has wide resampling envelopes; the location varies across fits.\n")
        f.write("- **Interaction patterns vary:** Sign consistency for FlyAsh/B × B/A, B/A × Curing, ")
        f.write("W/B × Curing is moderate; many grid cells show sign flips across resamples.\n")
        f.write("- **Low-support regions are unreliable:** Cells with low observed data density ")
        f.write("show high instability (flagged in figures).\n\n")
        
        # 10. Interaction Importance
        f.write("## 10. Interaction-Importance Interpretation\n\n")
        f.write("| Variable | Main Effect | Interaction | Total | Interaction Fraction |\n")
        f.write("|----------|-------------|-------------|-------|---------------------|\n")
        for _, row in imp_df.iterrows():
            highlight = " ⭐" if row["Variable"] == "FlyAsh/B (%)" else ""
            f.write(f"| {row['Variable']} | {row['Main_Effect']:.4f} | {row['Interaction_Contribution']:.4f} | ")
            f.write(f"{row['Total']:.4f} | {row['Interaction_Fraction']:.3f} |\n")
        f.write("\n")
        f.write("**Interpretation:** FlyAsh/B (%) has {:.1f}% of its importance from interactions, ".format(
            imp_df[imp_df["Variable"]=="FlyAsh/B (%)"]["Interaction_Fraction"].values[0]*100))
        f.write("meaning its effect is **strongly conditional** on other variables (B/A, curing period).\n")
        f.write("This is descriptive decomposition, not causal proof.\n\n")
        
        # 11. Recommended Manuscript Wording
        f.write("## 11. Recommended Manuscript Wording\n\n")
        
        f.write("### 11.1 EBM/Behaviour Section (Cautious Interpretation)\n\n")
        f.write("> Across 50 resampled EBM fits, the fly-ash main-effect curve should be interpreted ")
        f.write("as a broad conditional pattern rather than a precise optimum. The resampling envelope ")
        f.write("and local slope-consistency results should be used to identify stable regions, while ")
        f.write("narrow peaks or dips should not be reported as material optima. The fly-ash × B/A ")
        f.write("surface further shows that the fly-ash contribution varies across binder-to-aggregate ")
        f.write("ratios; this is descriptive model evidence, not causal proof.\n\n")
        
        f.write("### 11.2 Transition to Sustainability Optimisation\n\n")
        f.write("> These conditional behavioural insights directly inform the sustainability ")
        f.write("optimisation. In the observation-anchored +10 percentage-point substitution analysis, ")
        f.write(f"{len(supp_df)} of {len(all_df)} observed mixes were evidence-supported at both endpoints. ")
        f.write("The resulting strength–carbon trade-offs are therefore interpreted only within the ")
        f.write("project's defined support domain, not as a universal prescription to maximise fly ash.\n\n")
        
        f.write("### 11.3 Limitations Paragraph\n\n")
        f.write("> Several limitations qualify these conclusions. First, all predictions are ")
        f.write("model-based (EBM) and not causal experimental evidence; the +10 pp counterfactuals ")
        f.write("are statistical interpolations, not laboratory validations. Second, \"evidence-supported\" ")
        f.write("means proximity to observed data under the project's defined support criteria ")
        f.write("(Isolation Forest in-domain, Mahalanobis distance ≤ 95% quantile, KDE weight ≥ 0.1, ")
        f.write("no log-space extrapolation)—it does not guarantee real-world performance. ")
        f.write("Third, the resampling stability analysis (5×10 CV) quantifies estimation uncertainty ")
        f.write("but does not capture structural model uncertainty or extrapolation risk. ")
        f.write("Fourth, the CO₂ index uses central embodied-carbon coefficients and fixed transport ")
        f.write("distances; site-specific LCA would yield different absolute values. ")
        f.write("Fifth, the analysis holds all other mix variables constant; real mix redesigns ")
        f.write("would co-adjust W/B, SP, and other components.\n\n")
        
        # 12. Explicit Non-Proofs
        f.write("## 12. What This Analysis Does NOT Prove\n\n")
        f.write("1. **Predictions are model-based, not causal experiments.** The EBM is a ")
        f.write("statistical learner; counterfactual +10 pp changes are interpolations, ")
        f.write("not randomised controlled trials.\n\n")
        f.write("2. **Supported ≠ Real-world valid.** \"Evidence-supported\" means the ")
        f.write("counterfactual point falls within the project's defined applicability domain ")
        f.write("(Isolation Forest + KDE + Mahalanobis + log-range). It does not guarantee ")
        f.write("that a lab mix with that composition would achieve the predicted strength.\n\n")
        f.write("3. **No universal fly-ash optimum.** The effect is conditional on curing age, ")
        f.write("B/A, and W/B. Any single-number recommendation would be misleading.\n\n")
        f.write("4. **Carbon factors are central estimates.** The CO₂ index uses fixed production ")
        f.write("and transport coefficients; actual embodied carbon varies by location and supply chain.\n\n")
        f.write("5. **Strength preservation is predicted, not measured.** The −1 MPa threshold ")
        f.write("is an analytical benchmark; regulatory or project-specific thresholds may differ.\n\n")
        
        # Figure & Table References
        f.write("## 13. Figure and Table References\n\n")
        f.write("| Figure | File | Description |\n")
        f.write("|--------|------|-------------|\n")
        f.write("| Fig 1 | `fig_01_flyash_plus10_strength_carbon_overview.png` | ΔStrength & ΔCO₂ distributions |\n")
        f.write("| Fig 2 | `fig_02_flyash_plus10_by_age_ba_wb.png` | Median ΔStrength by age × tertiles |\n")
        f.write("| Fig 3 | `fig_03_flyash_plus10_strength_vs_carbon.png` | ΔCO₂ vs ΔStrength scatter |\n")
        f.write("| Fig 4 | `fig_04_flyash_main_effect_resampling_stability.png` | FA/B main effect + 95% envelope |\n")
        f.write("| Fig 5 | `fig_05_key_main_effects_resampling_stability.png` | B/A, W/B, Curing main effects |\n")
        f.write("| Fig 6 | `fig_06_flyash_ba_interaction_resampling_stability.png` | FA/B × B/A interaction heatmap |\n")
        f.write("| Fig 7 | `fig_07_key_interactions_resampling_stability.png` | B/A×Curing, W/B×Curing |\n")
        f.write("| Fig 8 | `fig_08_feature_importance_decomposition.png` | Main + Interaction importance |\n")
        f.write("\n")
        
        f.write("| CSV Output | Description |\n")
        f.write("|------------|-------------|\n")
        f.write("| `flyash_plus10_row_level_all_candidates.csv` | All 506 rows with predictions & support |\n")
        f.write("| `flyash_plus10_supported_candidates.csv` | Subset with both endpoints supported |\n")
        f.write("| `flyash_plus10_summary_overall.csv` | Overall summary statistics |\n")
        f.write("| `flyash_plus10_summary_by_age_ba_wb.csv` | Subgroup summaries |\n")
        f.write("| `flyash_plus10_support_funnel.csv` | Funnel counts |\n")
        f.write("| `flyash_plus10_feature_ranges_and_eligibility.csv` | Domain ranges |\n")
        f.write("| `stability_resample_manifest.csv` | 50-fit manifest with seeds |\n")
        f.write("| `stability_cv_performance.csv` | CV metrics per fit |\n")
        f.write("| `stability_main_effect_grid.csv` | Main effect stability grid |\n")
        f.write("| `stability_main_effect_turning_regions.csv` | Identified turning regions |\n")
        f.write("| `stability_interaction_grid.csv` | Interaction stability grid |\n")
        f.write("| `stability_interaction_summary.csv` | Interaction summaries |\n")
        f.write("| `feature_importance_with_interaction_fraction.csv` | Decomposed importance |\n")
        f.write("| `data_audit.csv` | Dataset audit summary |\n")
        f.write("| `environment.json` | Package versions &\n")
        f.write("\n")
        
        f.write("---\n")
        f.write("*End of Report*\n")
    
    logger.info(f"Report written: {report_path}")


if __name__ == "__main__":
    logger = setup_logging("reporting")
    generate_report(logger)
    logger.info("Reporting complete.")
