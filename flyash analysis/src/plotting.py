"""
Plotting module — All 8 publication-quality figures
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from utils import get_paths, setup_logging

# Colorblind-friendly palette
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
    'grey': '#949494',
    'light_blue': '#A6CEE3',
    'light_orange': '#FDBF6F',
    'light_green': '#B2DF8A',
    'light_red': '#FB9A99',
}

DIVERGING_CMAP = LinearSegmentedColormap.from_list('diverging', ['#0173B2', '#FFFFFF', '#D55E00'])

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#EEEEEE',
    'grid.linewidth': 0.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.transparent': False,
})


def save_fig(fig, name: str):
    """Save figure as 300 dpi PNG."""
    paths = get_paths()
    fig.savefig(paths["figures"] / f"{name}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def add_legend_below(ax, handles, labels, ncol=3):
    """Add horizontal legend below plot."""
    ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=ncol, frameon=False, fontsize=9)


# ============================================================
# FIGURE 1: FlyAsh +10 Strength & Carbon Overview
# ============================================================
def plot_fig01_overview(supported_df: pd.DataFrame):
    """Distribution of delta_strength and delta_CO2 for supported candidates."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    ds = supported_df["delta_strength_mpa"].dropna()
    dc = supported_df["delta_co2"].dropna()
    
    # Strength change
    ax = axes[0]
    ax.hist(ds, bins=20, color=COLORS['blue'], alpha=0.7, edgecolor='white', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.axvline(-1, color=COLORS['red'], linestyle=':', linewidth=1)
    ax.set_xlabel('Δ Predicted Strength (MPa)')
    ax.set_ylabel('Density')
    ax.text(0.02, 0.98, f'n = {len(ds)}', transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Carbon change
    ax = axes[1]
    ax.hist(dc, bins=20, color=COLORS['green'], alpha=0.7, edgecolor='white', density=True)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Δ CO₂ Index')
    ax.set_ylabel('Density')
    ax.text(0.02, 0.98, f'n = {len(dc)}', transform=ax.transAxes, va='top', fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    fig.tight_layout()
    save_fig(fig, "fig_01_flyash_plus10_strength_carbon_overview")


# ============================================================
# FIGURE 2: By Age, B/A, W/B
# ============================================================
def plot_fig02_by_age_ba_wb(summary_df: pd.DataFrame, supported_df: pd.DataFrame):
    """Median delta strength across B/A and W/B tertiles for age 7 & 28."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()
    
    panels = [
        ("age_7d", "B/A Tertile", "age_7d_ba", "7 days · B/A"),
        ("age_7d", "W/B Tertile", "age_7d_wb", "7 days · W/B"),
        ("age_28d", "B/A Tertile", "age_28d_ba", "28 days · B/A"),
        ("age_28d", "W/B Tertile", "age_28d_wb", "28 days · W/B"),
    ]
    
    for idx, (age_key, x_label, filter_key, panel_label) in enumerate(panels):
        ax = axes[idx]
        # This is a small panel identifier, not a figure title.  Without it,
        # the four age/variable comparisons cannot be interpreted in a paper.
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, va="top", ha="left", fontsize=10)
        
        # Filter summary for this age
        age_sum = summary_df[summary_df["group"].str.startswith(age_key)].copy()
        if len(age_sum) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        # Separate BA and WB tertiles
        ba_sum = age_sum[age_sum["group"].str.contains("ba_tertile")]
        wb_sum = age_sum[age_sum["group"].str.contains("wb_tertile")]
        
        if "ba" in filter_key:
            plot_df = ba_sum.sort_values("group")
            x_vals = range(len(plot_df))
            x_labels = [f'T{i+1}' for i in range(len(plot_df))]
        else:
            plot_df = wb_sum.sort_values("group")
            x_vals = range(len(plot_df))
            x_labels = [f'T{i+1}' for i in range(len(plot_df))]
        
        if len(plot_df) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        medians = plot_df["delta_strength_median"].values
        q25 = plot_df["delta_strength_q25"].values
        q75 = plot_df["delta_strength_q75"].values
        n_supp = plot_df["n_supported"].values
        
        # Color by sign
        colors = [COLORS['green'] if m >= 0 else COLORS['red'] for m in medians]
        
        bars = ax.bar(x_vals, medians, color=colors, alpha=0.7, edgecolor='white', width=0.6)
        ax.errorbar(x_vals, medians, yerr=[medians - q25, q75 - medians], 
                    fmt='none', color='black', capsize=3, linewidth=1)
        
        # Add n labels
        for xv, med, n in zip(x_vals, medians, n_supp):
            ax.text(xv, med + (0.1 if med >= 0 else -0.3), f'n={int(n)}', 
                    ha='center', va='bottom' if med >= 0 else 'top', fontsize=8)
        
        ax.axhline(0, color='black', linewidth=0.8)
        ax.axhline(-1, color=COLORS['red'], linestyle=':', linewidth=0.8)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel('Median Δ Strength (MPa)')
        # Age and variable are conveyed through the panel order and axis
        # labels; keep the plot free of in-figure titles for manuscript use.
        
        if idx >= 2:
            ax.set_xlabel('Tertile')
    
    fig.tight_layout()
    save_fig(fig, "fig_02_flyash_plus10_by_age_ba_wb")


# ============================================================
# FIGURE 3: Strength vs Carbon Scatter
# ============================================================
def plot_fig03_strength_vs_carbon(supported_df: pd.DataFrame):
    """Scatter: ΔCO₂ vs ΔStrength, colored by curing age."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    for age, color, label in [(7, COLORS['red'], '7 days'), (28, COLORS['blue'], '28 days')]:
        sub = supported_df[supported_df["curing_obs"] == age]
        if len(sub) == 0:
            continue
        ax.scatter(sub["delta_co2"], sub["delta_strength_mpa"], 
                   c=color, alpha=0.6, s=30, label=label, edgecolors='white', linewidth=0.3)
    
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    
    # Highlight favorable quadrant (carbon reduction + no strength loss)
    ax.axhspan(0, ax.get_ylim()[1], xmin=0, xmax=0.5, alpha=0.1, color='green')
    
    ax.set_xlabel('Δ CO₂ Index (negative = reduction)')
    ax.set_ylabel('Δ Predicted Strength (MPa)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    
    fig.tight_layout()
    save_fig(fig, "fig_03_flyash_plus10_strength_vs_carbon")


# ============================================================
# FIGURE 4: FlyAsh Main Effect Resampling Stability
# ============================================================
def plot_fig04_flyash_main_effect(main_grid_df: pd.DataFrame, ctx):
    """FlyAsh/B main effect: median + 95% envelope."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    fa_data = main_grid_df[main_grid_df["feature"] == "FlyAsh/B (%)"].sort_values("x_value")
    if len(fa_data) == 0:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        save_fig(fig, "fig_04_flyash_main_effect_resampling_stability")
        return
    
    x = fa_data["x_value"].values
    median = fa_data["median"].values
    p025 = fa_data["p025"].values
    p975 = fa_data["p975"].values
    
    # Envelope
    ax.fill_between(x, p025, p975, color=COLORS['blue'], alpha=0.2, label='95% resampling envelope')
    # Median
    ax.plot(x, median, color=COLORS['blue'], linewidth=2.5, label='Median (50 fits)')
    
    # Primary model reference (thin dashed)
    try:
        from ebm_engine_utils import get_ebm_main_effect
        primary_curve = get_ebm_main_effect(ctx, "FlyAsh/B (%)")
        if len(primary_curve) > 0:
            ax.plot(primary_curve["x"], primary_curve["y"], 
                    color=COLORS['grey'], linewidth=1, linestyle='--', label='Primary model')
    except:
        pass
    
    # Data rug
    df = pd.read_csv(get_paths()["dataset"])
    ax.scatter(df["FlyAsh/B (%)"], np.full(len(df), median.min() - 0.02*(median.max()-median.min())),
               color=COLORS['grey'], alpha=0.3, s=5, marker='|', label='Observed FA/B')
    
    ax.set_xlabel('FlyAsh/B (%)')
    ax.set_ylabel('EBM Contribution (log-strength)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, frameon=False, fontsize=9)
    
    fig.tight_layout()
    save_fig(fig, "fig_04_flyash_main_effect_resampling_stability")


# ============================================================
# FIGURE 5: Key Main Effects Stability
# ============================================================
def plot_fig05_key_main_effects(main_grid_df: pd.DataFrame):
    """B/A, W/B, curing period main effects with envelopes."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    
    features = ["B/A", "W/B", "curing period"]
    colors = [COLORS['orange'], COLORS['green'], COLORS['purple']]
    
    for ax, feat, color in zip(axes, features, colors):
        data = main_grid_df[main_grid_df["feature"] == feat].sort_values("x_value")
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        x = data["x_value"].values
        median = data["median"].values
        p025 = data["p025"].values
        p975 = data["p975"].values
        
        ax.fill_between(x, p025, p975, color=color, alpha=0.2)
        ax.plot(x, median, color=color, linewidth=2)
        
        ax.set_xlabel(feat)
        if ax == axes[0]:
            ax.set_ylabel('EBM Contribution (log-strength)')
    
    fig.tight_layout()
    save_fig(fig, "fig_05_key_main_effects_resampling_stability")


# ============================================================
# FIGURE 6: FlyAsh × B/A Interaction Stability
# ============================================================
def plot_fig06_flyash_ba_interaction(inter_grid_df: pd.DataFrame):
    """Heatmap of FlyAsh/B × B/A interaction with stability hatching."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    data = inter_grid_df[inter_grid_df["interaction"] == "FlyAsh/B (%)__B/A"]
    if len(data) == 0:
        for ax in axes:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        save_fig(fig, "fig_06_flyash_ba_interaction_resampling_stability")
        return
    
    # Pivot for heatmap
    pivot_median = data.pivot_table(index="x1", columns="x2", values="median")
    # Some edge cells have an undefined median but a defined consistency;
    # explicitly align every auxiliary panel to the plotted median grid.
    pivot_sign = data.pivot_table(index="x1", columns="x2", values="sign_consistency").reindex(
        index=pivot_median.index, columns=pivot_median.columns
    )
    pivot_support = data.pivot_table(index="x1", columns="x2", values="support_mask").reindex(
        index=pivot_median.index, columns=pivot_median.columns
    )
    
    x1_vals = pivot_median.index.values
    x2_vals = pivot_median.columns.values
    
    # Panel 1: Median interaction
    ax = axes[0]
    vmax = np.nanmax(np.abs(pivot_median.values))
    im = ax.contourf(x2_vals, x1_vals, pivot_median.values, 
                     levels=15, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extend='both')
    ax.contour(x2_vals, x1_vals, pivot_median.values, levels=15, 
               colors='k', linewidths=0.3, alpha=0.4)
    ax.set_xlabel('B/A')
    ax.set_ylabel('FlyAsh/B (%)')
    plt.colorbar(im, ax=ax, label='Interaction Score')
    
    # Panel 2: Sign consistency with hatching for low support
    ax = axes[1]
    im2 = ax.contourf(x2_vals, x1_vals, pivot_sign.values,
                      levels=np.linspace(0, 1, 11), cmap='RdYlGn', vmin=0, vmax=1, extend='both')
    
    # Hatch low support regions
    for i in range(len(x1_vals)):
        for j in range(len(x2_vals)):
            if not np.isnan(pivot_support.iloc[i, j]) and pivot_support.iloc[i, j] < 0.3:
                rect = mpatches.Rectangle((x2_vals[j] - (x2_vals[1]-x2_vals[0])/2,
                                           x1_vals[i] - (x1_vals[1]-x1_vals[0])/2),
                                          x2_vals[1]-x2_vals[0], x1_vals[1]-x1_vals[0],
                                          fill=False, hatch='///', edgecolor='gray', linewidth=0)
                ax.add_patch(rect)
    
    ax.set_xlabel('B/A')
    ax.set_ylabel('FlyAsh/B (%)')
    plt.colorbar(im2, ax=ax, label='Sign Consistency')
    
    fig.tight_layout()
    save_fig(fig, "fig_06_flyash_ba_interaction_resampling_stability")


# ============================================================
# FIGURE 7: Key Interactions (B/A×Curing, W/B×Curing)
# ============================================================
def plot_fig07_key_interactions(inter_grid_df: pd.DataFrame):
    """Two panels: B/A × curing, W/B × curing."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    interactions = [
        ("B/A__curing period", "B/A", "Curing Period (days)"),
        ("W/B__curing period", "W/B", "Curing Period (days)"),
    ]
    
    for ax, (inter_name, x1_label, x2_label) in zip(axes, interactions):
        data = inter_grid_df[inter_grid_df["interaction"] == inter_name]
        if len(data) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            continue
        
        pivot_median = data.pivot_table(index="x1", columns="x2", values="median")
        pivot_sign = data.pivot_table(index="x1", columns="x2", values="sign_consistency").reindex(
            index=pivot_median.index, columns=pivot_median.columns
        )
        pivot_support = data.pivot_table(index="x1", columns="x2", values="support_mask").reindex(
            index=pivot_median.index, columns=pivot_median.columns
        )
        
        x1_vals = pivot_median.index.values
        x2_vals = pivot_median.columns.values
        
        vmax = np.nanmax(np.abs(pivot_median.values))
        im = ax.contourf(x2_vals, x1_vals, pivot_median.values,
                         levels=15, cmap='RdBu_r', vmin=-vmax, vmax=vmax, extend='both')
        ax.contour(x2_vals, x1_vals, pivot_median.values, levels=15,
                   colors='k', linewidths=0.3, alpha=0.4)
        
        # Hatch unstable/low support
        for i in range(len(x1_vals)):
            for j in range(len(x2_vals)):
                sc = pivot_sign.iloc[i, j] if not np.isnan(pivot_sign.iloc[i, j]) else 1
                sm = pivot_support.iloc[i, j] if not np.isnan(pivot_support.iloc[i, j]) else 1
                if sc < 0.6 or sm < 0.3:
                    rect = mpatches.Rectangle((x2_vals[j] - (x2_vals[1]-x2_vals[0])/2,
                                               x1_vals[i] - (x1_vals[1]-x1_vals[0])/2),
                                              x2_vals[1]-x2_vals[0], x1_vals[1]-x1_vals[0],
                                              fill=False, hatch='///', edgecolor='gray', linewidth=0)
                    ax.add_patch(rect)
        
        ax.set_xlabel(x2_label)
        ax.set_ylabel(x1_label)
        plt.colorbar(im, ax=ax, label='Interaction Score')
    
    fig.tight_layout()
    save_fig(fig, "fig_07_key_interactions_resampling_stability")


# ============================================================
# FIGURE 8: Feature Importance Decomposition
# ============================================================
def plot_fig08_importance_decomposition(imp_df: pd.DataFrame):
    """Horizontal stacked bars: main + interaction per feature."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Sort by total importance
    imp_df = imp_df.sort_values("Total", ascending=True)
    
    y_pos = np.arange(len(imp_df))
    main_vals = imp_df["Main_Effect"].values
    inter_vals = imp_df["Interaction_Contribution"].values
    
    # Highlight fly ash
    colors_main = [COLORS['red'] if v == 'FlyAsh/B (%)' else COLORS['blue'] for v in imp_df["Variable"]]
    colors_inter = [COLORS['orange'] if v == 'FlyAsh/B (%)' else COLORS['light_blue'] for v in imp_df["Variable"]]
    
    bars1 = ax.barh(y_pos, main_vals, color=colors_main, alpha=0.8, edgecolor='white', height=0.6, label='Main Effect')
    bars2 = ax.barh(y_pos, inter_vals, left=main_vals, color=colors_inter, alpha=0.8, edgecolor='white', height=0.6, label='Interaction')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([v.replace(" (%)", "").replace(" period", "") for v in imp_df["Variable"]])
    ax.set_xlabel('Importance (EBM)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
    
    # Add total values
    for i, (main, inter, total) in enumerate(zip(main_vals, inter_vals, imp_df["Total"])):
        ax.text(total + 0.01, i, f'{total:.3f}', va='center', ha='left', fontsize=8)
    
    fig.tight_layout()
    save_fig(fig, "fig_08_feature_importance_decomposition")


# ============================================================
# MAIN PLOTTING ORCHESTRATOR
# ============================================================
def render_all_figures(logger):
    """Load all data and render all 8 figures."""
    paths = get_paths()
    
    # Load data
    supported_df = pd.read_csv(paths["outputs"] / "flyash_plus10_supported_candidates.csv")
    summary_df = pd.read_csv(paths["outputs"] / "flyash_plus10_summary_by_age_ba_wb.csv")
    main_grid_df = pd.read_csv(paths["outputs"] / "stability_main_effect_grid.csv")
    inter_grid_df = pd.read_csv(paths["outputs"] / "stability_interaction_grid.csv")
    imp_df = pd.read_csv(paths["outputs"] / "feature_importance_with_interaction_fraction.csv")
    
    # Load context for primary model reference
    import sys
    sys.path.insert(0, str(paths["project_root"] / "08_analysis_code_suite" / "shared"))
    from ebm_engine import load_context
    ctx = load_context(verbose=False)
    
    logger.info("Rendering Figure 1...")
    plot_fig01_overview(supported_df)
    
    logger.info("Rendering Figure 2...")
    plot_fig02_by_age_ba_wb(summary_df, supported_df)
    
    logger.info("Rendering Figure 3...")
    plot_fig03_strength_vs_carbon(supported_df)
    
    logger.info("Rendering Figure 4...")
    plot_fig04_flyash_main_effect(main_grid_df, ctx)
    
    logger.info("Rendering Figure 5...")
    plot_fig05_key_main_effects(main_grid_df)
    
    logger.info("Rendering Figure 6...")
    plot_fig06_flyash_ba_interaction(inter_grid_df)
    
    logger.info("Rendering Figure 7...")
    plot_fig07_key_interactions(inter_grid_df)
    
    logger.info("Rendering Figure 8...")
    plot_fig08_importance_decomposition(imp_df)
    
    logger.info("All figures rendered.")


if __name__ == "__main__":
    logger = setup_logging("plotting")
    render_all_figures(logger)
    logger.info("Plotting complete.")
