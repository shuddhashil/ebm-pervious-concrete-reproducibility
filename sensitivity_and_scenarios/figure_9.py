"""Figure 9: FA/B of the lowest-CO2 evidence-supported candidate (a) and CO2 support-premium onset (b) against
the FA production-stage emission factor, from 0 to 1.0 kg CO2-eq/kg (the cement factor is 1.043), with transport
at the midpoint distances. Also reports the first FA factor at which a 10-17 MPa selection changes its FA/B, and,
where the selection changes at the economic-allocation value 0.362, how close the new candidate was to the
baseline selection in CO2 index under the baseline factors.
Outputs: results/sensitivity/figure_9_data.csv, results/sensitivity/figure_9.png and .pdf"""
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import OUT, eng, load_sweep  # noqa: E402

ctx, targets, X, z = load_sweep()
fa, ba, wb = X.T
f = fa / 100
eff = z.CS_pred_MPa.to_numpy() - ctx.cs_bias - z.model_uncertainty_MPa.to_numpy()
B, E = z.base_feasible.to_numpy(), z.evidence_supported.to_numpy()
tf = eng.TRANSPORT_CO2_FACTOR
EC = eng.CO2_CEMENT_PROD + eng.DIST_CEMENT_MID * tf
EAGG = eng.CO2_AGG_PROD + eng.DIST_AGG_MID * tf
masks = {t: (B & (eff >= t), E & (eff >= t)) for t in targets}


def co2(e_fa_prod):
    return ((1 - f) * EC + f * (e_fa_prod + eng.DIST_FA_MID * tf)) * ba + EAGG + eng.CO2_WATER_PROD * wb * ba


def select(c):
    out = {}
    for t, (b, e) in masks.items():
        i = np.flatnonzero(e)[np.argmin(c[e])]
        out[t] = (i, c[b].min(), c[i])
    return out


base = select(co2(eng.CO2_FA_PROD))
assert np.isclose(base[10.0][2], 0.07339, atol=5e-5), "baseline CO2 at 10 MPa differs from the paper"
grid = np.round(np.arange(0, 1.0 + 1e-9, 0.002), 3)
FAB = {t: [] for t in targets}
onset = []
for g in grid:
    s = select(co2(g))
    for t in targets:
        FAB[t].append(fa[s[t][0]])
    onset.append(min(t for t in targets if 100 * (s[t][2] - s[t][1]) / s[t][1] > 0.1))
onset = np.array(onset)
pd.DataFrame({"e_FA_prod": grid, "premium_onset_MPa": onset,
              **{f"FA_B_{t:g}MPa": FAB[t] for t in targets}}).to_csv(OUT / "figure_9_data.csv", index=False)

print("premium onset over the factor range (MPa):", np.unique(onset))
low = [t for t in targets if t <= 17]
lim = min(grid[np.argmax(np.abs(np.array(FAB[t]) - FAB[t][0]) > 1)]
          if (np.abs(np.array(FAB[t]) - FAB[t][0]) > 1).any() else np.inf for t in low)
print("first FA factor at which a 10-17 MPa selection changes FA/B by more than 1 point:", lim)
c0, sw = co2(eng.CO2_FA_PROD), select(co2(0.362))
for t in targets:
    if sw[t][0] != base[t][0]:
        print(f"{t:5.1f} MPa at 0.362: FA/B {fa[base[t][0]]:.1f} -> {fa[sw[t][0]]:.1f}; CO2 difference under "
              f"baseline factors {100 * (c0[sw[t][0]] - c0[base[t][0]]) / c0[base[t][0]]:.2f}%")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#1b2a38",
                     "axes.labelcolor": "#1b2a38", "xtick.color": "#1b2a38", "ytick.color": "#1b2a38"})
fig, (a, b2) = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.45, 1]})
cols = {10: "#0b3c5d", 14: "#1f5a8a", 17: "#4a90c2", 20: "#d97a1f", 24: "#a34a12", 28: "#6b6b6b"}
mk = {10: "o", 14: "s", 17: "^", 20: "D", 24: "v", 28: "P"}
for t in (10, 14, 17, 20, 24, 28):
    a.plot(grid, np.array(FAB[float(t)]), color=cols[t], lw=1.8, marker=mk[t], markevery=40, ms=4.5,
           label=f"{t} MPa", zorder=3)
for ax in (a, b2):
    ax.axvline(eng.CO2_FA_PROD, color="#1b2a38", ls=":", lw=1.1)
    ax.axvline(0.362, color="#c0392b", ls="--", lw=1.1)
    ax.grid(ls=":", color="#c8c8c8")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("FA production-stage emission factor (kg CO$_2$-eq/kg)")
a.set_ylabel("FA/B of selected candidate (%)")
a.set_ylim(-2, 55)
a.legend(title="CS target", fontsize=7.5, title_fontsize=8, ncol=3, loc="center", bbox_to_anchor=(0.5, 0.45),
         frameon=False)
b2.plot(grid, onset, color="#d97a1f", lw=2.2)
b2.set_ylabel("CO$_2$ support-premium onset (MPa)")
b2.set_ylim(20, 27)
b2.text(0.5, 23.75, "23.5 MPa at every FA factor", fontsize=8, color="#d97a1f")
for ax, y in ((a, 47.0), (b2, 26.6)):
    ax.text(0.03, y, "Baseline", fontsize=7.5, color="#1b2a38", va="top")
    ax.text(0.375, y, "Economic\nallocation", fontsize=7.5, color="#c0392b", va="top")
for ax, lab in ((a, "(a)"), (b2, "(b)")):
    ax.text(-0.13 if ax is a else -0.2, 1.02, lab, transform=ax.transAxes, fontweight="bold", fontsize=10)
fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"figure_9.{ext}", dpi=500)
print("saved figure_9.png and figure_9.pdf")
