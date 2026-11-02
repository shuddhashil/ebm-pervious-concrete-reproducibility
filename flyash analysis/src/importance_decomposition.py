"""
Part D — Interaction Fraction Decomposition
"""
import pandas as pd
import numpy as np
from pathlib import Path

from utils import get_paths, setup_logging


def run_importance_decomposition(logger):
    """Calculate interaction fraction for each feature."""
    paths = get_paths()
    
    # Load grouped feature importance
    imp_df = pd.read_csv(paths["feature_importance_grouped"])
    logger.info(f"Loaded importance: {len(imp_df)} features")
    
    # Rename columns for clarity
    imp_df.columns = ["Rank", "Variable", "Main_Effect", "Interaction_Contribution", "Total", "Total_Pct"]
    
    # Calculate interaction fraction
    imp_df["Interaction_Fraction"] = imp_df["Interaction_Contribution"] / imp_df["Total"]
    imp_df["Interaction_Fraction"] = imp_df["Interaction_Fraction"].round(4)
    
    # Highlight fly ash
    fa_row = imp_df[imp_df["Variable"] == "FlyAsh/B (%)"]
    if len(fa_row) > 0:
        logger.info(f"FlyAsh/B (%): Main={fa_row['Main_Effect'].values[0]:.4f}, "
                    f"Inter={fa_row['Interaction_Contribution'].values[0]:.4f}, "
                    f"Total={fa_row['Total'].values[0]:.4f}, "
                    f"Frac={fa_row['Interaction_Fraction'].values[0]:.4f}")
    
    # Save
    out_csv = paths["outputs"] / "feature_importance_with_interaction_fraction.csv"
    imp_df.to_csv(out_csv, index=False)
    logger.info(f"Saved: {out_csv}")
    
    return imp_df


if __name__ == "__main__":
    logger = setup_logging("importance")
    run_importance_decomposition(logger)
    logger.info("Part D complete.")