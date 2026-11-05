"""Table S5: sensitivity of the evidence-supported fraction, the selected candidates and the CO2 support-premium
onset to the safety-margin multiplier k (Equations 6-8), with k = 0.5, 1, 1.5 and 2 and all other settings at
the baseline. Output: results/sensitivity/margin_multiplier.csv"""
import numpy as np
import pandas as pd

from common import OUT, TABLE_TARGETS, check_baseline, load_sweep

ctx, targets, X, z = load_sweep()
fa, ba, wb = X.T
cs, unc = z.CS_pred_MPa.to_numpy(), z.model_uncertainty_MPa.to_numpy()
base_flag, ev_flag, co2 = z.base_feasible.to_numpy(), z.evidence_supported.to_numpy(), z.CO2_index.to_numpy()
rows = []
for k in (0.5, 1.0, 1.5, 2.0):
    eff = cs - ctx.cs_bias - k * unc
    for t in targets:
        s = eff >= t
        b, e = base_flag & s, ev_flag & s
        r = dict(k=k, target_MPa=t, base_fraction=b.mean(), supported_fraction=e.mean(),
                 base_min_CO2=co2[b].min() if b.any() else np.nan)
        if e.any():
            i = np.flatnonzero(e)[np.argmin(co2[e])]
            r.update(supported_min_CO2=co2[i], FA_B_pct=fa[i], BA=ba[i], WB=wb[i])
        rows.append(r)
out = pd.DataFrame(rows)
out["premium_pct"] = 100 * (out.supported_min_CO2 - out.base_min_CO2) / out.base_min_CO2
b1 = out[out.k == 1.0]
check_baseline(b1.supported_fraction.to_numpy(), b1.supported_min_CO2.to_numpy(), targets)
out.to_csv(OUT / "margin_multiplier.csv", index=False)

print("baseline k = 1 matches the threshold grid")
print("premium onset (MPa):"); print(out[out.premium_pct > 0.1].groupby("k").target_MPa.min().to_string())
t = out[out.target_MPa.isin(TABLE_TARGETS)]
print("supported fraction (%):"); print(t.pivot_table(index="k", columns="target_MPa", values="supported_fraction").mul(100).round(2).to_string())
print("CO2 index of selected candidate:"); print(t.pivot_table(index="k", columns="target_MPa", values="supported_min_CO2").round(3).to_string())
