"""Final, non-destructive reconciliation of legacy DE and FA/B interaction results.

This script deliberately reuses the exact CanonicalEvaluator from
``robustness_verification.py``.  It does not retrain models, change the Python
environment, or overwrite any historical output.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import logging
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


def find_project_root(script_path: Path) -> Path:
    """Locate the EBM project from files that uniquely identify its layout."""
    for parent in script_path.resolve().parents:
        engine = parent / "08_analysis_code_suite" / "shared" / "ebm_engine.py"
        verifier = parent / "robustness_verification" / "src" / "robustness_verification.py"
        if engine.is_file() and verifier.is_file():
            return parent
    raise FileNotFoundError(
        "Could not locate the EBM project. Expected both "
        "08_analysis_code_suite/shared/ebm_engine.py and "
        "robustness_verification/src/robustness_verification.py above this script."
    )


PROJECT_ROOT = find_project_root(Path(__file__))
RV_MODULE = PROJECT_ROOT / "robustness_verification" / "src" / "robustness_verification.py"
RV_SPEC = importlib.util.spec_from_file_location("ebm_robustness_verification", RV_MODULE)
if RV_SPEC is None or RV_SPEC.loader is None:
    raise ImportError(f"Could not load the canonical evaluator from {RV_MODULE}")
rv = importlib.util.module_from_spec(RV_SPEC)
sys.modules[RV_SPEC.name] = rv
RV_SPEC.loader.exec_module(rv)

RV_ROOT = PROJECT_ROOT / "robustness_verification"
SOURCE_OUTPUTS = RV_ROOT / "outputs"
FINAL_ROOT = RV_ROOT / "robustness_2_reconciliation"
OUT_DIR = FINAL_ROOT / "outputs"
FIG_DIR = FINAL_ROOT / "figures"
LOG_DIR = FINAL_ROOT / "logs"

DE_INPUT = (
    PROJECT_ROOT
    / "09_analysis_outputs"
    / "analysis_02_optimizer_stability"
    / "stability_runs.csv"
)
SOBOL_INPUT = SOURCE_OUTPUTS / "verified_optima_by_target.csv"
FA_INPUT = SOURCE_OUTPUTS / "fa_b_ba_interaction_grid.csv"

COORDS = ["FA_B_pct", "BA", "WB"]
SOLUTION_TYPES = {
    "absolute_base": "base_feasible",
    "evidence_supported": "evidence_supported",
}
SIGN_CATEGORIES = [
    "consistently_beneficial",
    "mixed_majority_pos",
    "sign_uncertain",
    "mixed_majority_neg",
    "consistently_detrimental",
]
SIGN_COLORS = {
    "consistently_beneficial": "#006d2c",
    "mixed_majority_pos": "#66bd63",
    "sign_uncertain": "#fdae61",
    "mixed_majority_neg": "#d73027",
    "consistently_detrimental": "#7f0000",
}


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hash of a single file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_folder(folder: Path) -> str:
    """Compute SHA-256 hash of all files in a folder (for manifest)."""
    h = hashlib.sha256()
    for item in sorted(folder.rglob("*")):
        if item.is_file():
            rel = item.relative_to(folder.parent)
            h.update(rel.as_posix().encode("utf-8"))
            h.update(item.read_bytes())
    return h.hexdigest()


def setup_logging() -> logging.Logger:
    for folder in (OUT_DIR, FIG_DIR, LOG_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger("final_reconciliation")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(LOG_DIR / "final_reconciliation.log", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(fh)
    log.addHandler(sh)
    return log


def sign_category(pct_positive: float) -> str:
    if not np.isfinite(pct_positive):
        return "sign_uncertain"
    if pct_positive >= 0.90:
        return "consistently_beneficial"
    if pct_positive > 0.60:
        return "mixed_majority_pos"
    if pct_positive >= 0.40:
        return "sign_uncertain"
    if pct_positive > 0.10:
        return "mixed_majority_neg"
    return "consistently_detrimental"


def classify_gap(feasible: bool, gap: float, reference: float, kind: str) -> str:
    """Classify DE candidate gap against Sobol/local reference.

    Uses absolute tolerance of 1e-8 for "competitive" classification.
    When not feasible, returns canonically_infeasible_{base,evidence} labels.
    """
    if not feasible:
        if kind == "absolute_base":
            return "canonically_infeasible_base"
        else:
            return "canonically_infeasible_evidence"
    if not np.isfinite(gap):
        return "reference_unavailable"
    rel = gap / abs(reference) if abs(reference) > 1e-12 else np.inf
    if abs(gap) <= 1e-8:
        return "globally_competitive_within_tolerance"
    if gap < 0:
        return "better_than_sobol"
    if rel <= 0.01:
        return "near_optimal_within_1_pct"
    return "inferior_to_sobol"


def load_evaluator(log: logging.Logger):
    log.info("Loading the existing trained model and canonical evaluator ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx = rv.eng.load_context(False)
    return ctx, rv.CanonicalEvaluator(ctx, rv.SupportConfig())


def canonical_de_reconciliation(ev, log: logging.Logger) -> dict:
    log.info("[1/6] Loading and rechecking all 740 legacy DE candidates under canonical rules ...")
    de = pd.read_csv(DE_INPUT)
    sobol = pd.read_csv(SOBOL_INPUT)
    if len(de) != 740:
        raise ValueError(f"Expected 740 legacy DE rows, found {len(de)}")
    if de[COORDS + ["cs_target", "seed"]].isna().any().any():
        raise ValueError("Legacy DE input has missing target, seed, or coordinate values")
    targets = sorted(float(x) for x in de["cs_target"].unique())
    if len(targets) != 37:
        raise ValueError(f"Expected 37 DE targets, found {len(targets)}")

    groups = []
    for i, target in enumerate(targets, 1):
        g = de[np.isclose(de["cs_target"].astype(float), target)].copy().reset_index(drop=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            canonical = ev.evaluate(g[COORDS].to_numpy(float), target_strength=target)
        legacy = g.add_prefix("legacy_")
        merged = pd.concat([legacy, canonical], axis=1)
        merged.insert(0, "target_strength_MPa", target)
        merged.insert(1, "candidate_row_id", [f"DE-{i:02d}-{j:02d}" for j in range(len(g))])
        groups.append(merged)
        if i == 1 or i % 5 == 0 or i == len(targets):
            log.info("  Canonical DE progress: %d/%d targets", i, len(targets))
    candidates = pd.concat(groups, ignore_index=True)

    coordinate_delta = np.max(
        np.abs(
            candidates[COORDS].to_numpy(float)
            - candidates[[f"legacy_{c}" for c in COORDS]].to_numpy(float)
        )
    )
    if coordinate_delta > 1e-12:
        raise AssertionError(f"Coordinate mismatch after merge: {coordinate_delta:.3g}")

    # Re-evaluate each saved Sobol/local optimum with the same current evaluator.
    reference_rows = []
    for _, old in sobol.iterrows():
        target = float(old["target_strength_MPa"])
        kind = str(old["solution_type"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            now = ev.evaluate(old[COORDS].to_numpy(float).reshape(1, 3), target_strength=target).iloc[0]
        row = now.to_dict()
        row.update(
            target_strength_MPa=target,
            solution_type=kind,
            source_method="sobol_local",
            legacy_seed=np.nan,
            original_saved_CO2_index=float(old["CO2_index"]),
        )
        reference_rows.append(row)
    references = pd.DataFrame(reference_rows)

    # Per-row comparisons for both feasibility definitions.
    for kind, feasible_col in SOLUTION_TYPES.items():
        gaps, relative, classes = [], [], []
        for _, row in candidates.iterrows():
            target = float(row["target_strength_MPa"])
            ref = references[
                np.isclose(references["target_strength_MPa"], target)
                & (references["solution_type"] == kind)
            ]
            if len(ref) != 1:
                gaps.append(np.nan)
                relative.append(np.nan)
                classes.append("reference_unavailable")
                continue
            ref_co2 = float(ref.iloc[0]["CO2_index"])
            gap = float(row["CO2_index"]) - ref_co2
            rel = 100.0 * gap / abs(ref_co2) if abs(ref_co2) > 1e-12 else np.nan
            feasible = bool(row[feasible_col])
            gaps.append(gap)
            relative.append(rel)
            classes.append(classify_gap(feasible, gap, ref_co2, kind))
        prefix = "base" if kind == "absolute_base" else "evidence"
        candidates[f"{prefix}_gap_vs_sobol"] = gaps
        candidates[f"{prefix}_relative_gap_pct"] = relative
        candidates[f"{prefix}_comparison_class"] = classes

    candidate_path = OUT_DIR / "de_candidates_canonical.csv"
    candidates.to_csv(candidate_path, index=False)

    # Select the minimum-CO2 feasible candidate from Sobol/local + canonical DE.
    authoritative = []
    summaries = []
    for target in targets:
        target_de = candidates[np.isclose(candidates["target_strength_MPa"], target)]
        summary = {"target_strength_MPa": target, "de_candidates_total": int(len(target_de))}
        for kind, feasible_col in SOLUTION_TYPES.items():
            prefix = "base" if kind == "absolute_base" else "evidence"
            ref = references[
                np.isclose(references["target_strength_MPa"], target)
                & (references["solution_type"] == kind)
            ]
            if len(ref) != 1:
                raise AssertionError(f"Missing unique {kind} reference for target {target}")
            ref_row = ref.iloc[0]
            ref_feasible = bool(ref_row[feasible_col])
            de_feasible = target_de[target_de[feasible_col].astype(bool)]
            choices = []
            if ref_feasible:
                choices.append((float(ref_row["CO2_index"]), "sobol_local", np.nan, ref_row))
            for _, dr in de_feasible.iterrows():
                choices.append((float(dr["CO2_index"]), "legacy_DE_canonically_rechecked", dr["legacy_seed"], dr))
            if not choices:
                raise AssertionError(f"No feasible {kind} candidate for target {target}")
            best_co2, source, seed, best = min(choices, key=lambda x: x[0])
            ref_co2 = float(ref_row["CO2_index"])
            gap = best_co2 - ref_co2
            rel = 100.0 * gap / abs(ref_co2) if abs(ref_co2) > 1e-12 else np.nan
            selected = {c: best[c] for c in best.index if not str(c).startswith("legacy_")}
            selected.update(
                target_strength_MPa=target,
                solution_type=kind,
                source_method=source,
                legacy_seed=seed,
                objective_difference_from_sobol=gap,
                relative_difference_pct=rel,
                selection_reason=(
                    "Canonically feasible DE candidate has lower CO2 than the saved Sobol/local candidate."
                    if source.startswith("legacy_DE")
                    else "Saved Sobol/local candidate remains the lowest-CO2 canonically feasible candidate."
                ),
            )
            authoritative.append(selected)

            summary.update(
                {
                    f"{prefix}_de_feasible_count": int(target_de[feasible_col].astype(bool).sum()),
                    f"{prefix}_de_infeasible_count": int((~target_de[feasible_col].astype(bool)).sum()),
                    f"{prefix}_de_better_count": int((target_de[f"{prefix}_comparison_class"] == "better_than_sobol").sum()),
                    f"{prefix}_de_competitive_count": int((target_de[f"{prefix}_comparison_class"] == "globally_competitive_within_tolerance").sum()),
                    f"{prefix}_de_near_optimal_count": int((target_de[f"{prefix}_comparison_class"] == "near_optimal_within_1_pct").sum()),
                    f"{prefix}_best_de_gap_vs_sobol": float(de_feasible["CO2_index"].min() - ref_co2) if len(de_feasible) else np.nan,
                    f"{prefix}_selected_source": source,
                    f"{prefix}_selected_CO2_index": best_co2,
                }
            )
        summaries.append(summary)

    authoritative_df = pd.DataFrame(authoritative)
    summary_df = pd.DataFrame(summaries)
    authoritative_df.to_csv(OUT_DIR / "authoritative_optima_by_target.csv", index=False)
    summary_df.to_csv(OUT_DIR / "de_reconciliation_summary_by_target.csv", index=False)

    plot_de_gaps(candidates, references)
    write_de_report(candidates, authoritative_df, summary_df)
    return {
        "candidates": candidates,
        "references": references,
        "authoritative": authoritative_df,
        "summary": summary_df,
        "coordinate_delta": float(coordinate_delta),
    }


def plot_de_gaps(candidates: pd.DataFrame, references: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    specifications = [
        ("absolute_base", "base_feasible", "base_gap_vs_sobol", "Absolute-base feasibility"),
        ("evidence_supported", "evidence_supported", "evidence_gap_vs_sobol", "Evidence-supported feasibility"),
    ]
    for ax, (_, feasible_col, gap_col, title) in zip(axes, specifications):
        feasible = candidates[candidates[feasible_col].astype(bool)]
        ax.scatter(
            feasible["target_strength_MPa"],
            feasible[gap_col],
            s=14,
            alpha=0.28,
            color="#377eb8",
            label="Canonically feasible DE candidates",
        )
        best = feasible.groupby("target_strength_MPa", as_index=False)[gap_col].min()
        ax.plot(best["target_strength_MPa"], best[gap_col], color="#d95f02", marker="o", ms=3, lw=1.4, label="Best DE gap")
        ax.axhline(0, color="black", lw=1, ls="--", label="Saved Sobol/local objective")
        ax.set_title(title)
        ax.set_ylabel("CO2 objective gap (DE - Sobol/local)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="best")
    axes[-1].set_xlabel("Target compressive strength (MPa)")
    fig.suptitle("Legacy DE candidates after canonical re-evaluation", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "de_canonical_gap_by_target.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / "de_canonical_gap_by_target.pdf", bbox_inches="tight")
    plt.close(fig)


def write_de_report(candidates: pd.DataFrame, authoritative: pd.DataFrame, summary: pd.DataFrame) -> None:
    base_feasible = int(candidates["base_feasible"].astype(bool).sum())
    evidence_feasible = int(candidates["evidence_supported"].astype(bool).sum())
    selected_counts = authoritative.groupby(["solution_type", "source_method"]).size().to_dict()
    text = f"""# Differential-evolution canonical reconciliation

Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}

## What was tested

All {len(candidates):,} legacy differential-evolution (DE) coordinates were re-evaluated by the same `CanonicalEvaluator` used for the saved Sobol/local optima. The legacy `feasible` field was retained only as historical information; it was not used to decide the new result.

## Main result

- Canonically absolute-base feasible DE candidates: **{base_feasible:,} / {len(candidates):,}**.
- Canonically evidence-supported DE candidates: **{evidence_feasible:,} / {len(candidates):,}**.
- Authoritative rows produced: **{len(authoritative):,}** (37 targets × 2 solution definitions).
- Absolute-base selections from canonical DE: **{selected_counts.get(('absolute_base', 'legacy_DE_canonically_rechecked'), 0)}**; retained Sobol/local: **{selected_counts.get(('absolute_base', 'sobol_local'), 0)}**.
- Evidence-supported selections from canonical DE: **{selected_counts.get(('evidence_supported', 'legacy_DE_canonically_rechecked'), 0)}**; retained Sobol/local: **{selected_counts.get(('evidence_supported', 'sobol_local'), 0)}**.

## Interpretation

A negative canonical gap means a rechecked DE candidate has a lower CO2 objective than the saved Sobol/local candidate for the same target and feasibility definition. A DE candidate is allowed to replace the saved candidate only if the current canonical evaluator marks it feasible under that same definition.

The detailed per-candidate table preserves every legacy column with a `legacy_` prefix and adds the full canonical predictions, uncertainty, support diagnostics, slacks, feasibility flags, and separate comparison classes for the two solution definitions.

## Files

- `outputs/de_candidates_canonical.csv`
- `outputs/de_reconciliation_summary_by_target.csv`
- `outputs/authoritative_optima_by_target.csv`
- `figures/de_canonical_gap_by_target.png` and `.pdf`
"""
    (FINAL_ROOT / "DE_RECONCILIATION_REPORT.md").write_text(text, encoding="utf-8")


def fa_masked_analysis(log: logging.Logger) -> dict:
    log.info("[2/6] Applying the two-endpoint evidence mask to the FA/B interaction grid ...")
    grid = pd.read_csv(FA_INPUT)
    required = {
        "contrast_valid",
        "evidence_supported",
        "evidence_supported_shifted",
        "delta_CS_FA10_MPa",
        "delta_CO2_FA10",
        "bag_pct_positive_delta",
        "bag_std_delta_MPa",
        "W_B_percentile_value",
    }
    missing = sorted(required - set(grid.columns))
    if missing:
        raise KeyError(f"FA interaction grid is missing required columns: {missing}")
    if len(grid) != 2325:
        raise ValueError(f"Expected 2,325 interaction rows, found {len(grid)}")
    grid["contrast_evidence_supported"] = (
        grid["contrast_valid"].astype(bool)
        & grid["evidence_supported"].astype(bool)
        & grid["evidence_supported_shifted"].astype(bool)
    )
    grid["evidence_sign_category"] = [sign_category(float(x)) for x in grid["bag_pct_positive_delta"]]
    grid.loc[~grid["contrast_evidence_supported"], "evidence_sign_category"] = "unsupported"
    grid.to_csv(OUT_DIR / "fa_interaction_evidence_masked_grid.csv", index=False)

    rows = []
    for wb, group in grid.groupby("W_B_percentile_value", sort=True):
        rows.append(interaction_summary_row(group, f"wb_0p{str(int(wb*100)).zfill(2)}"))
    rows.append(interaction_summary_row(grid, "overall"))
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT_DIR / "fa_interaction_supported_summary.csv", index=False)
    plot_fa_mask(grid, summary)
    write_fa_report(grid, summary)
    return {"grid": grid, "summary": summary}


def interaction_summary_row(group: pd.DataFrame, label: str) -> dict:
    supported = group[group["contrast_evidence_supported"].astype(bool)]
    row = {
        "W_B_group": label,
        "grid_count": int(len(group)),
        "supported_contrast_count": int(len(supported)),
        "supported_fraction_pct": 100.0 * len(supported) / len(group) if len(group) else np.nan,
        "positive_delta_CS_count": int((supported["delta_CS_FA10_MPa"] > 0).sum()),
        "positive_delta_CS_pct": 100.0 * (supported["delta_CS_FA10_MPa"] > 0).mean() if len(supported) else np.nan,
        "median_delta_CS_FA10_MPa": float(supported["delta_CS_FA10_MPa"].median()) if len(supported) else np.nan,
        "mean_delta_CS_FA10_MPa": float(supported["delta_CS_FA10_MPa"].mean()) if len(supported) else np.nan,
        "median_delta_CO2_FA10": float(supported["delta_CO2_FA10"].median()) if len(supported) else np.nan,
        "median_bag_pct_positive_delta": float(supported["bag_pct_positive_delta"].median()) if len(supported) else np.nan,
        "median_bag_std_delta_MPa": float(supported["bag_std_delta_MPa"].median()) if len(supported) else np.nan,
    }
    for category in SIGN_CATEGORIES:
        n = int((supported["evidence_sign_category"] == category).sum())
        row[f"{category}_count"] = n
        row[f"{category}_pct"] = 100.0 * n / len(supported) if len(supported) else np.nan
    return row


def plot_fa_mask(grid: pd.DataFrame, summary: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    wbs = sorted(grid["W_B_percentile_value"].unique())
    fig, axes = plt.subplots(1, len(wbs), figsize=(17, 5.7), sharex=True, sharey=True)
    if len(wbs) == 1:
        axes = [axes]
    for ax, wb in zip(axes, wbs):
        g = grid[np.isclose(grid["W_B_percentile_value"], wb)]
        # Unsupported contrasts in light grey
        unsupported = g[~g["contrast_evidence_supported"].astype(bool)]
        ax.scatter(
            unsupported["FA_B_pct"], unsupported["BA"],
            marker="s", s=38, color="#d9d9d9", linewidths=0, label="Unsupported contrast"
        )
        # Supported contrasts coloured by sign_stability
        for category in SIGN_CATEGORIES:
            c = g[(g["contrast_evidence_supported"].astype(bool)) & (g["evidence_sign_category"] == category)]
            if len(c):
                ax.scatter(
                    c["FA_B_pct"], c["BA"],
                    marker="s", s=39, color=SIGN_COLORS[category], linewidths=0,
                    label=category.replace("_", " ")
                )
        supported = g[g["contrast_evidence_supported"].astype(bool)]
        positive_pct = 100.0 * (supported["delta_CS_FA10_MPa"] > 0).mean() if len(supported) else np.nan
        pct_text = f"{positive_pct:.1f}% positive" if np.isfinite(positive_pct) else "no supported contrasts"
        supported_count = int(supported.shape[0])
        total_count = int(g.shape[0])
        ax.set_title(f"W/B = {wb:.3g}\nSupported n={supported_count}/{total_count}; {pct_text}")
        ax.set_xlabel("FA/B (%)")
        ax.grid(alpha=0.12)
    axes[0].set_ylabel("B/A")
    handles = [Line2D([0], [0], marker="s", linestyle="", color="#d9d9d9", label="Unsupported", markersize=8)]
    handles += [
        Line2D([0], [0], marker="s", linestyle="", color=SIGN_COLORS[c], label=c.replace("_", " "), markersize=8)
        for c in SIGN_CATEGORIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(SIGN_CATEGORIES) + 1, fontsize=8, frameon=False)
    fig.suptitle("Effect of +10 percentage points FA/B: only contrasts supported at both endpoints", fontsize=13)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    # Use safe decimal encoding: wb_0p32, wb_0p34, wb_0p40
    wb_str = "_".join(f"wb_0p{int(round(wb * 100)):02d}" for wb in wbs)
    png_path = FIG_DIR / f"fa_interaction_evidence_masked_{wb_str}.png"
    pdf_path = FIG_DIR / f"fa_interaction_evidence_masked_{wb_str}.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)


def write_fa_report(grid: pd.DataFrame, summary: pd.DataFrame) -> None:
    overall = summary[summary["W_B_group"] == "overall"].iloc[0]
    supported = int(overall["supported_contrast_count"])
    text = f"""# Evidence-masked FA/B × B/A interaction analysis

Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}

## What was tested

The original {len(grid):,}-row interaction grid was preserved. A +10 percentage-point FA/B contrast is now counted only when the starting point and shifted endpoint both pass the existing evidence-support rules and the contrast is valid.

## Main result

- Evidence-supported contrasts: **{supported:,} / {len(grid):,} ({float(overall['supported_fraction_pct']):.1f}%)**.
- Supported contrasts with positive predicted strength change: **{int(overall['positive_delta_CS_count']):,} / {supported:,} ({float(overall['positive_delta_CS_pct']):.1f}%)**.
- Median supported strength change: **{float(overall['median_delta_CS_FA10_MPa']):.3f} MPa**.
- Mean supported strength change: **{float(overall['mean_delta_CS_FA10_MPa']):.3f} MPa**.
- Median supported CO2-index change: **{float(overall['median_delta_CO2_FA10']):.6f}**.

## Interpretation

Grey cells in the figure are not claimed as detrimental or beneficial: at least one endpoint lacks sufficient evidence support. Coloured cells show only two-endpoint-supported contrasts. The sign category is based on the fraction of bagged model components predicting a positive strength change, not merely the central prediction.

All percentages in the summary table use the number of supported contrasts as their denominator.

## Files

- `outputs/fa_interaction_evidence_masked_grid.csv`
- `outputs/fa_interaction_supported_summary.csv`
- `figures/fa_interaction_evidence_masked_wb_0p32_wb_0p34_wb_0p40.png` and `.pdf`
"""
    (FINAL_ROOT / "FA_INTERACTION_MASKED_REPORT.md").write_text(text, encoding="utf-8")


def validate_results(de_result: dict | None, fa_result: dict | None, before_hashes: dict, log: logging.Logger) -> dict:
    log.info("[3/6] Running independent structural and numerical checks ...")
    checks = []

    def check(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})
        if not passed:
            raise AssertionError(f"{name}: {detail}")

    if de_result is not None:
        d = de_result["candidates"]
        a = de_result["authoritative"]
        check("DE row count", len(d) == 740, f"rows={len(d)}")
        check("DE coordinate preservation", de_result["coordinate_delta"] <= 1e-12, f"max_abs_delta={de_result['coordinate_delta']:.3g}")
        check("Authoritative row count", len(a) == 74, f"rows={len(a)}")
        check(
            "Authoritative target/type uniqueness",
            a[["target_strength_MPa", "solution_type"]].drop_duplicates().shape[0] == 74,
            "expected 37 targets x 2 definitions",
        )
        feasible_ok = True
        minima_ok = True
        for _, row in a.iterrows():
            feasible_col = SOLUTION_TYPES[row["solution_type"]]
            feasible_ok &= bool(row[feasible_col])
            target = float(row["target_strength_MPa"])
            source_best = float(row["CO2_index"])
            ref = de_result["references"]
            ref = ref[np.isclose(ref["target_strength_MPa"], target) & (ref["solution_type"] == row["solution_type"])]
            feasible_values = []
            if len(ref) == 1 and bool(ref.iloc[0][feasible_col]):
                feasible_values.append(float(ref.iloc[0]["CO2_index"]))
            de_t = d[np.isclose(d["target_strength_MPa"], target) & d[feasible_col].astype(bool)]
            feasible_values.extend(de_t["CO2_index"].astype(float).tolist())
            minima_ok &= bool(feasible_values) and math.isclose(source_best, min(feasible_values), rel_tol=0, abs_tol=1e-12)
        check("Authoritative selections feasible", feasible_ok, "all 74 selected rows pass their named canonical definition")
        check("Authoritative selections are minima", minima_ok, "each row is the minimum CO2 among compared feasible sources")

    if fa_result is not None:
        g = fa_result["grid"]
        s = fa_result["summary"]
        check("FA grid row count", len(g) == 2325, f"rows={len(g)}")
        counts = g.groupby("W_B_percentile_value").size()
        check("FA W/B group sizes", len(counts) == 3 and (counts == 775).all(), f"group_counts={counts.to_dict()}")
        expected_mask = g["contrast_valid"].astype(bool) & g["evidence_supported"].astype(bool) & g["evidence_supported_shifted"].astype(bool)
        check("FA mask formula", bool((expected_mask == g["contrast_evidence_supported"].astype(bool)).all()), f"supported={int(expected_mask.sum())}")
        pct_ok = True
        for _, row in s.iterrows():
            if int(row["supported_contrast_count"]) > 0:
                pct_sum = sum(float(row[f"{c}_pct"]) for c in SIGN_CATEGORIES)
                pct_ok &= math.isclose(pct_sum, 100.0, rel_tol=0, abs_tol=1e-8)
        check("FA supported category percentages", pct_ok, "five sign-category percentages sum to 100% for every group")

    after_hashes = {path: sha256_file(Path(path)) for path in before_hashes}
    unchanged = (
        set(before_hashes) == set(after_hashes)
        and all(before_hashes[path] == after_hashes[path] for path in before_hashes)
    )
    check("Original inputs unchanged", unchanged, "SHA-256 hashes match before and after execution")
    manifest = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "inputs_before_sha256": {str(k): v for k, v in before_hashes.items()},
        "inputs_after_sha256": after_hashes,
        "checks": checks,
        "all_checks_passed": all(x["passed"] for x in checks),
    }
    (OUT_DIR / "validation_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def write_combined_report(de_result: dict | None, fa_result: dict | None, manifest: dict) -> None:
    sections = [
        "# Final robustness reconciliation",
        "",
        f"Generated: {dt.datetime.now(dt.timezone.utc).isoformat()}",
        "",
        "This folder contains two focused corrections. No trained model, package version, historical CSV, or original figure was changed.",
        "",
    ]
    if de_result is not None:
        a = de_result["authoritative"]
        replacements = int(a["source_method"].astype(str).str.startswith("legacy_DE").sum())
        sections += [
            "## 1. Canonical optimizer reconciliation",
            "",
            f"All 740 legacy DE candidates were rechecked under the current canonical feasibility definitions. The resulting 74-row authoritative table chose a canonically rechecked DE candidate in {replacements} target/definition cases and retained the saved Sobol/local candidate in {74 - replacements} cases.",
            "",
            "See `DE_RECONCILIATION_REPORT.md` for the full explanation.",
            "",
        ]
    if fa_result is not None:
        overall = fa_result["summary"].query("W_B_group == 'overall'").iloc[0]
        sections += [
            "## 2. Evidence-masked FA/B interaction atlas",
            "",
            f"The analysis retained all 2,325 grid rows but makes directional claims only for {int(overall['supported_contrast_count'])} contrasts where both endpoints are evidence-supported. Among those, {float(overall['positive_delta_CS_pct']):.1f}% have a positive central strength change.",
            "",
            "See `FA_INTERACTION_MASKED_REPORT.md` for the full explanation.",
            "",
        ]
    sections += [
        "## Validation",
        "",
        f"All {len(manifest['checks'])} automated checks passed. The SHA-256 manifest confirms that the source CSVs and evaluator script were unchanged.",
        "",
        "Machine-readable validation evidence is in `outputs/validation_manifest.json`.",
        "",
    ]
    (FINAL_ROOT / "FINAL_RECONCILIATION_REPORT.md").write_text("\n".join(sections), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--run-all", action="store_true", help="Run both reconciliations (default).")
    mode.add_argument("--de-only", action="store_true", help="Run only canonical DE reconciliation.")
    mode.add_argument("--fa-only", action="store_true", help="Run only the evidence-masked FA/B analysis.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log = setup_logging()
    run_de = not args.fa_only
    run_fa = not args.de_only
    # Only include inputs that will actually be used (accumulate, don't overwrite)
    # Always protect the canonical evaluator source as well as the selected CSV inputs.
    selected_inputs = [RV_MODULE]
    if run_de:
        selected_inputs.extend([DE_INPUT, SOBOL_INPUT])
    if run_fa:
        selected_inputs.append(FA_INPUT)
    missing = [str(p) for p in selected_inputs if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input(s): " + ", ".join(missing))
    before_hashes = {str(p): sha256_file(p) for p in selected_inputs}
    log.info("Final reconciliation started. Existing inputs will be read only.")
    ev = None
    if run_de:
        _, ev = load_evaluator(log)
    de_result = canonical_de_reconciliation(ev, log) if run_de else None
    fa_result = fa_masked_analysis(log) if run_fa else None
    manifest = validate_results(de_result, fa_result, before_hashes, log)
    log.info("[4/6] Writing the combined report ...")
    write_combined_report(de_result, fa_result, manifest)
    log.info("Complete. Results: %s", FINAL_ROOT)


if __name__ == "__main__":
    main()
