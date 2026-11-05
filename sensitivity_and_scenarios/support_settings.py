"""Table S4: sensitivity of the evidence-supported fraction and the selected candidates to the KDE bandwidth
(0.5-1.5 times the Silverman value) and the Isolation Forest contamination (0.03-0.15), one setting at a
time with the other thresholds at the baseline. Model predictions do not depend on these settings, so the
sweep is evaluated once and only the KDE and Isolation Forest columns are recomputed for each setting.
The contamination sets the cut-off as the corresponding quantile of the Isolation Forest scores of the
training observations; the saved trees are kept.
Output: results/sensitivity/support_settings.csv"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KernelDensity

from common import OUT, TABLE_TARGETS, check_baseline, eng, load_sweep

ctx, targets, X, z = load_sweep()
fa, ba, wb = X.T
adraw = eng.ad_input_matrix(ctx, fa, ba, wb)
sup_c = ctx.sup_scaler.transform(adraw)
if_c = ctx.ad_scaler.transform(adraw)

kd0 = KernelDensity(kernel="gaussian", bandwidth="silverman").fit(ctx.sup_train_std)
assert np.allclose(kd0.score_samples(sup_c), ctx.sup_kde.score_samples(sup_c)), "KDE refit differs from saved model"
H = kd0.bandwidth_
if_train = ctx.ad_if_model.score_samples(ctx.train_ad_scaled)
if_cand = ctx.ad_if_model.score_samples(if_c)
assert np.isclose(np.percentile(if_train, 6.7), -ctx.ad_if_threshold), "IF cut-off is not the 6.7% training quantile"


def kde_col(h):
    kd = KernelDensity(kernel="gaussian", bandwidth=h).fit(ctx.sup_train_std)
    return np.exp(kd.score_samples(sup_c)) / np.exp(kd.score_samples(ctx.sup_train_std)).max()


fixed_base = (z.FA_B_lower_slack >= -1e-10) & (z.FA_B_upper_slack >= -1e-10) & (z.BA_lower_slack >= -1e-10) \
    & (z.BA_upper_slack >= -1e-10) & (z.WB_lower_slack >= -1e-10) & (z.WB_upper_slack >= -1e-10) \
    & (z.trio_mahal_slack >= 0) & z.local_consistency_pass
fixed_ev = z.mahal_pass & z.log_extrapolation_pass

settings = [("kde_bandwidth_x_silverman", f, H * f, None) for f in (0.5, 0.75, 1.0, 1.25, 1.5)] + \
           [("if_contamination", c, None, c) for c in (0.03, 0.05, 0.067, 0.10, 0.15)]
rows = []
for knob, level, h, c in settings:
    kde = kde_col(h) if h is not None else z.kde_support.to_numpy()
    ifp = (if_cand >= np.percentile(if_train, 100 * c)) if c is not None else z.if_pass.to_numpy()
    base = fixed_base & ifp
    ev = base & fixed_ev & (kde >= 0.10)
    for t in targets:
        s = z.CS_effective_MPa >= t
        b, e = base & s, ev & s
        r = dict(setting=knob, level=level, target_MPa=t, base_fraction=b.mean(), supported_fraction=e.mean(),
                 base_min_CO2=z[b].CO2_index.min() if b.any() else np.nan)
        if e.any():
            k = z[e].CO2_index.idxmin()
            r.update(supported_min_CO2=z.CO2_index[k], FA_B_pct=fa[k], BA=ba[k], WB=wb[k])
        rows.append(r)
res = pd.DataFrame(rows)
res["premium_pct"] = 100 * (res.supported_min_CO2 - res.base_min_CO2) / res.base_min_CO2
for knob, lvl in (("kde_bandwidth_x_silverman", 1.0), ("if_contamination", 0.067)):
    b = res[(res.setting == knob) & (res.level == lvl)]
    check_baseline(b.supported_fraction.to_numpy(), b.supported_min_CO2.to_numpy(), targets)
res.to_csv(OUT / "support_settings.csv", index=False)

print(f"Silverman bandwidth = {H:.2f}; baselines match the threshold grid")
print("premium onset (MPa):"); print(res[res.premium_pct > 0.1].groupby(["setting", "level"]).target_MPa.min().to_string())
t = res[res.target_MPa.isin(TABLE_TARGETS)]
print("supported fraction (%):"); print(t.pivot_table(index=["setting", "level"], columns="target_MPa", values="supported_fraction").mul(100).round(2).to_string())
print("CO2 index of selected candidate:"); print(t.pivot_table(index=["setting", "level"], columns="target_MPa", values="supported_min_CO2").round(3).to_string())
