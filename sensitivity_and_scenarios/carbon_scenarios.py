"""Carbon-accounting scenarios (Section 3.4, Section 4.4.2): FA allocation and transport assumptions.
Strength and support flags do not depend on the emission factors, so only the CO2 ranking is recomputed.

FA production factors (kg CO2-eq/kg):
  baseline   0.00397  by-product carrying only post-collection processing emissions (Pfeiffer et al., 2024)
  economic   0.362    economic allocation of power-plant emissions (Chen & Wang, 2022)
Transport cases (cement, FA, aggregate):
  midpoint   225, 160, 85 km at 62 g CO2/t-km (baseline)
  lower      50, 20, 20 km;  upper 400, 300, 150 km;  FA upper with cement lower 50, 300, 85 km
  material   material-specific transport increments of Pfeiffer et al. (2024): 0.02932, 0.01887, 0.00728 kg/kg
Output: results/sensitivity/carbon_scenarios.csv"""
import numpy as np
import pandas as pd

from common import OUT, TABLE_TARGETS, check_baseline, eng, load_sweep

ctx, targets, X, z = load_sweep()
fa, ba, wb = X.T
f = fa / 100.0
eff = z.CS_pred_MPa.to_numpy() - ctx.cs_bias - z.model_uncertainty_MPa.to_numpy()
base_flag, ev_flag = z.base_feasible.to_numpy(), z.evidence_supported.to_numpy()
ROAD = 0.000062  # kg CO2 per kg-km


def factors(e_fa_prod, transport):
    kind, (dc, dfa, dagg) = transport
    if kind == "increment":
        return eng.CO2_CEMENT_PROD + dc, e_fa_prod + dfa, eng.CO2_AGG_PROD + dagg
    return eng.CO2_CEMENT_PROD + dc * ROAD, e_fa_prod + dfa * ROAD, eng.CO2_AGG_PROD + dagg * ROAD


def co2(ec, efa, eagg):
    return ((1 - f) * ec + f * efa) * ba + eagg + eng.CO2_WATER_PROD * wb * ba


ALLOC = {"baseline": eng.CO2_FA_PROD, "economic": 0.362}
MID = (eng.DIST_CEMENT_MID, eng.DIST_FA_MID, eng.DIST_AGG_MID)
TRANS = {"midpoint": ("km", MID), "lower": ("km", (50, 20, 20)), "upper": ("km", (400, 300, 150)),
         "FA_upper_cement_lower": ("km", (50, 300, MID[2])),
         "material": ("increment", (0.02932, 0.01887, 0.00728))}

out = []
for an, e_fa in ALLOC.items():
    for tn, tr in TRANS.items():
        ec, efa, eagg = factors(e_fa, tr)
        c = co2(ec, efa, eagg)
        for t in targets:
            s = eff >= t
            b, e = base_flag & s, ev_flag & s
            r = dict(allocation=an, transport=tn, target_MPa=t, e_cement=ec, e_FA=efa,
                     CO2_saving_per_10pp_FA_x_BA=0.1 * (ec - efa),
                     base_min_CO2=c[b].min() if b.any() else np.nan)
            if e.any():
                i = np.flatnonzero(e)[np.argmin(c[e])]
                r.update(supported_min_CO2=c[i], FA_B_pct=fa[i], BA=ba[i], WB=wb[i], candidate=i)
            out.append(r)
out = pd.DataFrame(out)
out["premium_pct"] = 100 * (out.supported_min_CO2 - out.base_min_CO2) / out.base_min_CO2
b0 = out[(out.allocation == "baseline") & (out.transport == "midpoint")]
check_baseline(None, b0.supported_min_CO2.to_numpy(), targets)
out.to_csv(OUT / "carbon_scenarios.csv", index=False)

key = out.assign(scenario=out.allocation + " | " + out.transport)
ref = b0.set_index("target_MPa").candidate
changed = {sc: int((d.set_index("target_MPa").candidate != ref).sum()) for sc, d in key.groupby("scenario")}
onset = key[key.premium_pct > 0.1].groupby("scenario").target_MPa.min()
print("baseline matches the threshold grid")
print(pd.DataFrame({"premium_onset_MPa": onset, "targets_with_changed_candidate_of_37": pd.Series(changed)}).to_string())
t = key[key.target_MPa.isin(TABLE_TARGETS)]
print("FA/B of selected candidate (%):")
print(t.pivot_table(index="scenario", columns="target_MPa", values="FA_B_pct").round(1).to_string())
print("CO2 index of selected candidate:")
print(t.pivot_table(index="scenario", columns="target_MPa", values="supported_min_CO2").round(3).to_string())
print("CO2 saving per 10 percentage points of FA (x B/A):")
print(key.groupby("scenario").CO2_saving_per_10pp_FA_x_BA.first().round(3).to_string())
