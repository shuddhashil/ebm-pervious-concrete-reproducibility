"""
Part A — Data Audit and Model Contract
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from utils import get_paths, setup_logging


def run_data_audit(logger) -> Dict[str, Any]:
    """Complete data audit and model contract documentation."""
    paths = get_paths()
    
    # Load dataset
    df = pd.read_csv(paths["dataset"])
    logger.info(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Load EBM context to get model contract
    import sys
    sys.path.insert(0, str(paths["project_root"] / "08_analysis_code_suite" / "shared"))
    from ebm_engine import load_context
    
    ctx = load_context(verbose=False)
    
    # ---- 1. Basic dataset audit ----
    audit = {
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "dtypes": {c: str(df[c].dtype) for c in df.columns},
        "missingness": df.isnull().sum().to_dict(),
        "missingness_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
    }
    
    # Numeric ranges
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    ranges = {}
    for c in numeric_cols:
        ranges[c] = {
            "min": float(df[c].min()),
            "max": float(df[c].max()),
            "mean": float(df[c].mean()),
            "median": float(df[c].median()),
            "std": float(df[c].std()),
        }
    audit["numeric_ranges"] = ranges
    
    # Target specifics
    audit["target"] = {
        "column": "Strength_con",
        "unit": "MPa",
        "min": float(df["Strength_con"].min()),
        "max": float(df["Strength_con"].max()),
        "mean": float(df["Strength_con"].mean()),
        "median": float(df["Strength_con"].median()),
    }
    
    # ---- 2. EBM feature contract ----
    audit["ebm_features"] = {
        "feature_names": ctx.features,
        "feature_count": len(ctx.features),
        "feature_order": ctx.features,
        "use_log_target": ctx.use_log,
        "feature_ranges": {k: list(v) for k, v in ctx.feature_ranges.items()},
        "fixed_values": {
            "Cu": float(ctx.cu_fixed),
            "WAS_mm": float(ctx.was_fixed),
            "curing_period_fixed": 28,
            "Sand_A_fixed": 0.0,
            "SP_pct_fixed": 0.0,
        },
        "transforms": {
            "Cu": "log",
            "curing period": "log",
            "FlyAsh/B (%)": "logit(fraction)",
            "Sand/A": "logit(fraction)",
        },
    }
    
    # ---- 3. Key variable definitions ----
    audit["key_variables"] = {
        "FlyAsh/B (%)": {
            "meaning": "Fly ash mass / Total binder mass × 100",
            "unit": "percent (0–50)",
            "note": "Stored as raw percentage in dataset; EBM uses logit(fraction/100)",
        },
        "B/A": {
            "meaning": "Binder / Aggregate mass ratio",
            "unit": "ratio",
        },
        "W/B": {
            "meaning": "Water / Binder mass ratio",
            "unit": "ratio",
        },
        "curing period": {
            "meaning": "Curing age at strength test",
            "unit": "days",
            "note": "Exact 7 and 28 are principal strata; EBM uses log(curing)",
        },
        "Strength_con": {
            "meaning": "Compressive strength (harmonized to 150mm cube equivalent)",
            "unit": "MPa",
        },
    }
    
    # ---- 4. Indicator column update rules ----
    audit["indicator_updates"] = {
        "FA_used": "Set to (FlyAsh/B (%) > 0).astype(float) — MUST update when FA changes",
        "Sand_used": "Set to (Sand/A > 0).astype(float) — fixed at 0 for prediction",
        "SP_used": "Set to (SP (%) > 0).astype(float) — fixed at 0 for prediction",
    }
    
    # ---- 5. Support criteria (from ebm_engine.py) ----
    audit["support_criteria"] = {
        "canonical_combined_flag": "reliable_mask() in ebm_engine.py",
        "definition": "(in_ad == 1) AND (in_mahal == 1) AND (low_support == 0) AND (log_extrap == 0)",
        "components": {
            "in_ad": "Isolation Forest in-domain (score ≥ -threshold)",
            "in_mahal": "Mahalanobis distance ≤ D_MAX (95% quantile, df=3)",
            "low_support": "KDE weight w(x) < 0.1",
            "log_extrap": "Log-space prediction outside training log-strength range",
        },
        "thresholds": {
            "KDE_LOW_FLAG_THRESHOLD": 0.1,
            "MAHAL_QUANTILE": 0.95,
            "CURING_FIXED": 28,
        },
        "support_inputs_8": [
            "Cu", "WAS_mm", "B/A", "W/B", "Sand/A", "curing period",
            "FlyAsh/B (%)", "SP (%)",
        ],
    }
    
    # ---- 6. CO2 factors ----
    audit["co2_factors"] = {
        "CO2_CEMENT_PROD": 1.043,
        "CO2_FA_PROD": 0.00397,
        "CO2_AGG_PROD": 0.00503,
        "CO2_WATER_PROD": 0.00212,
        "TRANSPORT_CO2_FACTOR": 0.000062,
        "DIST_CEMENT_MID": 225,
        "DIST_FA_MID": 160,
        "DIST_AGG_MID": 85,
        "CO2_CEMENT_CENTRAL": 1.043 + 225 * 0.000062,
        "CO2_FA_CENTRAL": 0.00397 + 160 * 0.000062,
        "CO2_AGG_CENTRAL": 0.00503 + 85 * 0.000062,
        "CO2_WATER_CENTRAL": 0.00212,
        "formula": "binder = ((1-fa_frac)*CO2_CEMENT_CENTRAL + fa_frac*CO2_FA_CENTRAL) * BA; total = binder + CO2_AGG_CENTRAL + CO2_WATER_CENTRAL * WB * BA",
    }
    
    # ---- 7. Model provenance ----
    audit["model_provenance"] = {
        "model_file": str(paths["primary_ebm"]),
        "model_type": "ExplainableBoostingRegressor (interpret)",
        "training_data": "phase5_augmented_dataset.csv (506 rows)",
        "training_config": {
            "outer_bags": 16,
            "inner_bags": 0,
            "interactions": 12,
            "use_log_target": True,
            "seed": 42,
        },
        "support_bundles": {
            "isolation_forest": str(paths["ad_isofor"]),
            "kde_mahalanobis": str(paths["ad_support"]),
        },
        "note": "This is the PRIMARY row-split EBM. Do NOT use final_ebm_grouped.pkl.",
    }
    
    return audit, df, ctx


def write_audit_outputs(audit: dict, df: pd.DataFrame, ctx, logger):
    """Write all Part A outputs."""
    paths = get_paths()
    
    # data_audit.csv — flattened summary
    audit_rows = []
    audit_rows.append({"section": "dataset", "metric": "row_count", "value": audit["row_count"]})
    audit_rows.append({"section": "dataset", "metric": "column_count", "value": audit["column_count"]})
    for col, miss in audit["missingness"].items():
        audit_rows.append({"section": "missingness", "metric": col, "value": int(miss)})
    for col, rng in audit["numeric_ranges"].items():
        for stat, val in rng.items():
            audit_rows.append({"section": f"range_{col}", "metric": stat, "value": val})
    for col, val in audit["target"].items():
        audit_rows.append({"section": "target", "metric": col, "value": val})
    
    audit_df = pd.DataFrame(audit_rows)
    audit_csv = paths["outputs"] / "data_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    logger.info(f"Written: {audit_csv}")
    
    # model_and_data_contract.md
    md_path = paths["docs"] / "model_and_data_contract.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Fly-Ash Bridge Analysis — Model & Data Contract\n\n")
        f.write(f"**Generated:** {pd.Timestamp.now().isoformat()}\n\n")
        
        f.write("## 1. Dataset Summary\n")
        f.write(f"- **Rows:** {audit['row_count']}\n")
        f.write(f"- **Columns:** {audit['column_count']}\n")
        f.write(f"- **Target:** {audit['target']['column']} ({audit['target']['unit']})\n")
        f.write(f"- **Target range:** [{audit['target']['min']:.1f}, {audit['target']['max']:.1f}] MPa\n")
        f.write(f"- **Missing values:** {sum(audit['missingness'].values())} total\n\n")
        
        f.write("## 2. EBM Model Contract\n")
        f.write(f"- **Model file:** `{audit['model_provenance']['model_file']}`\n")
        f.write(f"- **Model type:** {audit['model_provenance']['model_type']}\n")
        f.write(f"- **Features ({audit['ebm_features']['feature_count']}):** {', '.join(audit['ebm_features']['feature_names'])}\n")
        f.write(f"- **Feature order (critical):** {audit['ebm_features']['feature_order']}\n")
        f.write(f"- **Log target:** {audit['ebm_features']['use_log_target']}\n")
        f.write(f"- **Fixed prediction values:** Cu={audit['ebm_features']['fixed_values']['Cu']:.3f}, ")
        f.write(f"WAS={audit['ebm_features']['fixed_values']['WAS_mm']:.1f}, ")
        f.write(f"curing=28d, Sand/A=0, SP%=0\n")
        f.write(f"- **Transforms:** {audit['ebm_features']['transforms']}\n\n")
        
        f.write("## 3. Key Variable Definitions\n")
        for var, info in audit["key_variables"].items():
            f.write(f"### {var}\n")
            for k, v in info.items():
                f.write(f"- **{k}:** {v}\n")
            f.write("\n")
        
        f.write("## 4. Indicator Column Updates\n")
        for ind, rule in audit["indicator_updates"].items():
            f.write(f"- **{ind}:** {rule}\n")
        f.write("\n")
        
        f.write("## 5. Support Criteria (Canonical)\n")
        sc = audit["support_criteria"]
        f.write(f"- **Combined flag:** `{sc['definition'].replace('≥', '>=')}`\n")
        f.write("- **Components:**\n")
        for comp, desc in sc["components"].items():
            f.write(f"  - `{comp}`: {desc}\n")
        f.write("- **Thresholds:**\n")
        for thresh, val in sc["thresholds"].items():
            f.write(f"  - `{thresh}`: {val}\n")
        f.write(f"- **Support input features (8):** {sc['support_inputs_8']}\n\n")
        
        f.write("## 6. CO2 Calculation\n")
        f.write("Identical to `ebm_engine.co2_central()`:\n")
        for k, v in audit["co2_factors"].items():
            if k != "formula":
                f.write(f"- `{k}` = {v}\n")
        f.write(f"- **Formula:** {audit['co2_factors']['formula']}\n\n")
        
        f.write("## 7. Model Provenance\n")
        for k, v in audit["model_provenance"].items():
            if isinstance(v, dict):
                f.write(f"- **{k}:**\n")
                for k2, v2 in v.items():
                    f.write(f"  - {k2}: {v2}\n")
            else:
                f.write(f"- **{k}:** {v}\n")
    
    logger.info(f"Written: {md_path}")
    
    # Also save full audit as JSON for programmatic use
    audit_json = paths["outputs"] / "data_audit_full.json"
    with open(audit_json, "w") as f:
        json.dump(audit, f, indent=2, default=str)
    logger.info(f"Written: {audit_json}")


if __name__ == "__main__":
    logger = setup_logging("data_audit")
    audit, df, ctx = run_data_audit(logger)
    write_audit_outputs(audit, df, ctx, logger)
    logger.info("Part A complete.")