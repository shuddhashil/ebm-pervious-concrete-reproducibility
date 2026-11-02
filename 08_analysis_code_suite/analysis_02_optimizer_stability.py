#!/usr/bin/env python3
"""Analysis 2: seed-to-seed stability of the constrained EBM optimiser."""

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SUITEDIR = Path(__file__).resolve().parent
if str(SUITEDIR) not in sys.path:
    sys.path.insert(0, str(SUITEDIR))

from shared import ebm_engine as eng  # noqa: E402

DEFAULT_OUTPUT = eng.OUTPUT_ROOT / "analysis_02_optimizer_stability"


def parse_args():
    p = argparse.ArgumentParser(description="Analysis 2 - optimiser stability")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quick-smoke", action="store_true")
    p.add_argument("--seeds", type=int, default=20, help="Number of DE seeds.")
    p.add_argument("--base-seed", type=int, default=1000)
    p.add_argument("--min-pressure", type=float, default=10.0)
    p.add_argument("--max-pressure", type=float, default=28.0)
    p.add_argument("--step", type=float, default=0.5)
    p.add_argument("--maxiter", type=int, default=600)
    p.add_argument("--popsize", type=int, default=20)
    p.add_argument("--boundary-tol", type=float, default=1e-3,
                   help="Relative optimisation-box tolerance for a boundary hit.")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT))
    return p.parse_args()


def targets(args):
    if args.quick_smoke:
        return [14.0]
    if args.seeds < 1 or args.step <= 0 or args.min_pressure > args.max_pressure:
        raise ValueError("seeds must be positive and target range/step must be valid")
    return list(np.round(np.arange(args.min_pressure, args.max_pressure + args.step / 2,
                                  args.step), 6))


def regime(row):
    if not row.get("feasible", False):
        return "optimizer_failure"
    if row.get("in_ad") != 1:
        return "outside_ad"
    if row.get("in_mahal") != 1:
        return "outside_mahal"
    if row.get("low_support") == 1:
        return "low_kde_support"
    if row.get("log_extrap") == 1:
        return "log_extrapolation"
    return "reliable"


def add_boundary_flags(record, bounds, tol):
    flags = []
    for column, (lo, hi) in zip(("FA_B_pct", "BA", "WB"), bounds):
        value = record.get(column, np.nan)
        hit = bool(np.isfinite(value) and min(abs(value - lo), abs(value - hi))
                   <= tol * max(1.0, hi - lo))
        record["boundary_" + column] = int(hit)
        flags.append(hit)
    record["boundary_any"] = int(any(flags))


def percentile(series, q):
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(np.percentile(values, q)) if len(values) else np.nan


def summarise(runs):
    numeric = ["CO2_index", "FA_B_pct", "BA", "WB", "CS_effective_MPa", "objective"]
    rows = []
    for target, group in runs.groupby("cs_target", sort=True):
        ok = group[group["feasible"] == True]  # noqa: E712
        row = {"cs_target": target, "n_runs": len(group), "n_feasible": len(ok),
               "failure_rate": float(1 - len(ok) / len(group)),
               "boundary_hit_rate": float(group["boundary_any"].mean())}
        for col in ("boundary_FA_B_pct", "boundary_BA", "boundary_WB"):
            row[col + "_rate"] = float(group[col].mean())
        for col in numeric:
            values = pd.to_numeric(ok[col], errors="coerce").dropna()
            row[col + "_median"] = float(values.median()) if len(values) else np.nan
            row[col + "_iqr"] = float(values.quantile(.75) - values.quantile(.25)) if len(values) else np.nan
            row[col + "_p05"] = percentile(values, 5)
            row[col + "_p95"] = percentile(values, 95)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    target_grid = targets(args)
    seeds = [args.base_seed + i for i in range(1 if args.quick_smoke else args.seeds)]
    maxiter = min(args.maxiter, 20) if args.quick_smoke else args.maxiter
    out_dir = Path(args.output_dir)
    if args.dry_run:
        print(json.dumps({"dry_run": True, "targets": target_grid, "seeds": seeds,
                          "maxiter": maxiter, "popsize": args.popsize,
                          "output_dir": str(out_dir)}, indent=2))
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    ctx = eng.load_context(verbose=True)
    records, started = [], time.time()
    for target in target_grid:
        for seed in seeds:
            try:
                record = eng.optimize_target(ctx, target, seed=seed, maxiter=maxiter,
                                             popsize=args.popsize)
            except Exception as exc:
                record = {"cs_target": target, "seed": seed, "success": False,
                          "feasible": False, "message": "exception: " + str(exc)[:100]}
            add_boundary_flags(record, ctx.bounds, args.boundary_tol)
            record["regime"] = regime(record)
            records.append(record)
            print("target %.1f, seed %d: %s" % (target, seed, record["regime"]), flush=True)

    runs = pd.DataFrame(records)
    summary = summarise(runs)
    frequencies = (runs.groupby(["cs_target", "regime"], dropna=False).size()
                 .rename("count").reset_index())
    frequencies["frequency"] = frequencies["count"] / frequencies.groupby("cs_target")["count"].transform("sum")
    runs.to_csv(out_dir / "stability_runs.csv", index=False)
    summary.to_csv(out_dir / "stability_target_summary.csv", index=False)
    frequencies.to_csv(out_dir / "stability_regime_frequencies.csv", index=False)
    metadata = {"script": "analysis_02_optimizer_stability", "timestamp_utc": datetime.datetime.utcnow().isoformat() + "Z",
                "targets": target_grid, "seeds": seeds, "maxiter": maxiter, "popsize": args.popsize,
                "boundary_tol": args.boundary_tol, "runtime_seconds": round(time.time() - started, 1),
                "engine": "shared/ebm_engine.py"}
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(10, 7))
        for ax, col, label in zip(axes.ravel(), ["CO2_index", "FA_B_pct", "BA", "WB"],
                                  ["CO2 index", "FA/B (%)", "B/A", "W/B"]):
            for seed, group in runs.groupby("seed"):
                ax.plot(group["cs_target"], group[col], "o-", ms=2, alpha=.65, label=str(seed))
            ax.set(xlabel="strength target (MPa)", ylabel=label)
            ax.grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(out_dir / "fig_stability_trajectories.png", dpi=300)
        plt.close(fig)
    except Exception as exc:
        print("(figure skipped: %s)" % exc)
    print("Done. Outputs in: %s" % out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
