"""Validation of the EBM on the 505-row corpus, using the fixed hyperparameters stored with the final model.

Writes to results/validation/:
  cv_50_splits.csv, cv_summary.csv   repeated 5-fold x 10 stratified CV (Table 3)
  figure_3.png, figure_S4.png        predicted vs observed, R2 distribution, residuals (first CV repeat)
  leverage.csv                       refit without the 34 leverage-flagged observations
  permutation_100.csv                100 permuted-target runs (one 5-fold CV each)
  final_model_check.csv              refit of the final model against the stored model
  loso_predictions.csv, loso.json    leave-one-study-out validation and study-offset analysis
"""
import json, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.model_selection import LeaveOneGroupOut, RepeatedStratifiedKFold, StratifiedKFold
from interpret.glassbox import ExplainableBoostingRegressor as EBR
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "validation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
bundle = joblib.load(ROOT / "02_models_and_support" / "primary_row_split_ebm.pkl")
PARAMS, FEAT = bundle["final_params"], bundle["features"]
P = pd.read_csv(ROOT / "01_core_data" / "phase5_augmented_dataset.csv")
A = pd.read_csv(ROOT / "01_core_data" / "analysis_records.csv", encoding="utf-8")
S = pd.read_csv(ROOT / "01_core_data" / "study_id_map.csv", encoding="utf-8")
POR = pd.read_csv(ROOT / "01_core_data" / "porosity_by_row.csv", encoding="utf-8")
assert len(P) == len(A) == len(S) == len(POR) == 505 and S.study.nunique() == 22
Xdf = P[FEAT].astype(float).copy()
Xdf["Cu"] = np.log(Xdf["Cu"])                       # Cu and curing age enter the model on a log scale
Xdf["curing period"] = np.log(Xdf["curing period"])
X, y, g = Xdf.values, P.Strength_con.values.astype(float), S.study.values


def fit(Xa, ya, bags=4, jobs=2):
    return EBR(**PARAMS, outer_bags=bags, inner_bags=0, n_jobs=jobs, random_state=SEED,
               validation_size=0.15, early_stopping_rounds=40).fit(Xa, np.log(ya))


def r2(t, p):
    return float(1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def met(t, p, s):
    return {f"{s}_r2": r2(t, p), f"{s}_rmse": float(np.sqrt(((t - p) ** 2).mean())),
            f"{s}_mae": float(np.abs(t - p).mean()), f"{s}_mape": float((np.abs(t - p) / t).mean() * 100)}


def strat_key(frame, ya):
    """Joint stratification: strength tercile, curing group and fly ash use."""
    bins = pd.qcut(pd.Series(ya), 3, labels=False, duplicates="drop").astype(str).values
    cur = pd.cut(frame["curing period"], [0, 7, 28, 1000], labels=False).astype(str).values
    fa = (frame["FlyAsh/B (%)"] > 0).astype(int).astype(str).values
    return np.array([f"{a}_{b}_{c}" for a, b, c in zip(bins, cur, fa)])


def cross_validation():
    splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=SEED + 999).split(X, strat_key(P, y)))

    def one(i, tr, te):
        m = fit(X[tr], y[tr])
        pte = np.exp(m.predict(X[te]))
        row = dict(run=i + 1, repeat=i // 5 + 1, fold=i % 5 + 1, **met(y[tr], np.exp(m.predict(X[tr])), "train"), **met(y[te], pte, "test"))
        return row, (te, pte)

    res = Parallel(n_jobs=2)(delayed(one)(i, tr, te) for i, (tr, te) in enumerate(splits))
    r = pd.DataFrame([a for a, _ in res])
    r.to_csv(OUT / "cv_50_splits.csv", index=False)
    s = r.drop(columns=["run", "repeat", "fold"]).agg(["mean", "std"]).T
    s.to_csv(OUT / "cv_summary.csv")
    te = np.concatenate([b[0] for _, b in res[:5]])
    pr = np.concatenate([b[1] for _, b in res[:5]])
    fr = pd.DataFrame(dict(row=te, observed=y[te], predicted=pr)).sort_values("row")
    fr.to_csv(OUT / "first_repeat_predictions.csv", index=False)
    print(s.round(3).to_string())
    print("test R2 min/p10/p90/max", r.test_r2.min(), r.test_r2.quantile(.1), r.test_r2.quantile(.9), r.test_r2.max())
    print("first repeat mean residual", (fr.observed - fr.predicted).mean())
    figures(r, s, fr)
    return s


def figures(r, s, fr):
    m, sd, rm, rs = s.loc["test_r2", "mean"], s.loc["test_r2", "std"], s.loc["test_rmse", "mean"], s.loc["test_rmse", "std"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    a = ax[0]
    a.scatter(fr.observed, fr.predicted, c="blue", alpha=.6, s=45, edgecolors="navy", linewidths=.3)
    lim = [fr.observed.min(), fr.observed.max()]
    a.plot(lim, lim, "r--", lw=2, label="1:1")
    a.set_xlabel("Observed (MPa)", fontsize=14)
    a.set_ylabel("Predicted (MPa)", fontsize=14)
    a.legend(loc="lower right", frameon=False, fontsize=12)
    a.text(.10, .96, f"$R^2$ = {m:.3f} ± {sd:.3f}\nRMSE = {rm:.2f} ± {rs:.2f} MPa", transform=a.transAxes, va="top", fontsize=13,
           bbox=dict(boxstyle="round", fc="white", ec="black"))
    b = ax[1]
    b.hist(r.test_r2, bins=15, color="#4a90c2", edgecolor="black")
    b.axvline(m, color="red", lw=3, label=f"Mean R² = {m:.3f}")
    b.axvline(m - sd, color="blue", ls="--", lw=2, label=f"±1 SD ({sd:.3f})")
    b.axvline(m + sd, color="blue", ls="--", lw=2)
    b.set_xlabel("Test R² per run", fontsize=14)
    b.set_ylabel("Frequency", fontsize=14)
    b.legend(loc="upper left", bbox_to_anchor=(.08, 1), frameon=False, fontsize=12)
    a.text(.015, .96, "(a)", transform=a.transAxes, va="top", fontsize=18, fontweight="bold")
    b.text(.015, .985, "(b)", transform=b.transAxes, va="top", fontsize=18, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "figure_3.png", dpi=300)
    plt.close(fig)
    e = fr.observed - fr.predicted
    fig, a = plt.subplots(figsize=(7, 7))
    a.hist(e, bins=45, color="lightcoral", edgecolor="black", alpha=.85)
    a.axvline(0, color="red", ls="--", label="Zero residual")
    a.axvline(e.mean(), color="blue", label=f"Mean = {e.mean():.3f}")
    a.set_xlabel("Residual (Observed - Predicted, MPa)", fontsize=14)
    a.set_ylabel("Frequency", fontsize=14)
    a.legend(frameon=False, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "figure_S4.png", dpi=300)
    plt.close(fig)


def leverage(full):
    keep = P.leverage_flag.values == 0
    Xc, yc = X[keep], y[keep]
    key = strat_key(P[keep].reset_index(drop=True), yc)
    sc = [r2(yc[te], np.exp(fit(Xc[tr], yc[tr]).predict(Xc[te])))
          for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xc, key)]
    out = dict(full_r2=full, no_leverage_r2=float(np.mean(sc)), n_removed=int((~keep).sum()))
    out["delta"] = out["full_r2"] - out["no_leverage_r2"]
    pd.DataFrame([out]).to_csv(OUT / "leverage.csv", index=False)
    print(out)


def permutation():
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    def it(i):
        yp = y[np.random.RandomState(SEED + i).permutation(len(y))]
        key = pd.qcut(pd.Series(yp), 3, labels=False, duplicates="drop").astype(str).values
        return float(np.mean([r2(yp[te], np.exp(fit(X[tr], yp[tr], jobs=1).predict(X[te]))) for tr, te in cv.split(X, key)]))

    r = Parallel(n_jobs=3)(delayed(it)(i) for i in range(100))
    pd.DataFrame(dict(perm_r2=r)).to_csv(OUT / "permutation_100.csv", index=False)
    print("permutation mean", np.mean(r), "p99", np.percentile(r, 99), "max", max(r))


def final_model():
    pred = fit(X, y, bags=16).predict(X)
    d = float(np.abs(pred - bundle["model"].predict(X)).max())
    pd.DataFrame([dict(outer_bags=16, max_abs_diff=d)]).to_csv(OUT / "final_model_check.csv", index=False)
    print("final model max diff", d)


def loso():
    def one(tr, te):
        assert not set(g[tr]) & set(g[te])
        raw = fit(X[tr], y[tr]).predict(X[te])
        return pd.DataFrame(dict(row=te, study=g[te], y_true=y[te], y_pred=np.exp(raw),
                                 in_range=(raw >= np.log(y[tr].min())) & (raw <= np.log(y[tr].max()))))

    o = pd.concat(Parallel(n_jobs=2)(delayed(one)(tr, te) for tr, te in LeaveOneGroupOut().split(X, y, g))).sort_values("row")
    o.to_csv(OUT / "loso_predictions.csv", index=False, encoding="utf-8")
    k = o[o.in_range].copy()
    t, p = k.y_true, k.y_pred
    e = t - p
    off = e.groupby(k.study).transform("mean")
    gm = t.groupby(k.study).transform("mean")
    rho = k.groupby("study")[["y_true", "y_pred"]].apply(
        lambda d: spearmanr(d.y_true, d.y_pred)[0] if d.y_pred.nunique() > 1 and len(d) >= 5 else np.nan)
    c = t - (p + off)
    res = dict(rejected=int((~o.in_range).sum()), r2=r2(t, p), rmse=float(np.sqrt((e ** 2).mean())),
               median_rho=float(rho.median()), positive_rho=int((rho > 0).sum()),
               between_study_share=float(((gm - t.mean()) ** 2).sum() / ((t - t.mean()) ** 2).sum() * 100),
               offset_share=float((off ** 2).sum() / (e ** 2).sum() * 100),
               r2_offset_removed=float(1 - (c ** 2).sum() / ((t - t.mean()) ** 2).sum()),
               rmse_offset_removed=float(np.sqrt((c ** 2).mean())))
    kr = A.loc[k.row.values].reset_index(drop=True)
    kr["err"] = e.values
    kr["study"] = k.study.values
    Q = ["Cu", "WAS_mm", "B/A", "W/B", "Sand/A", "curing_age_days", "FlyAsh/B (%)", "SP (%)"]
    s = kr.groupby("study")[Q + ["err"]].mean()
    res["max_abs_rho_offset_vs_inputs"] = float(max(abs(spearmanr(s[q], s.err)[0]) for q in Q))
    Z = ((kr[Q] - kr[Q].mean()) / kr[Q].std()).values
    D = np.linalg.norm(Z[:, None] - Z[None], axis=2)
    st, yy = kr.study.values, k.y_true.values
    gap = np.abs(yy[:, None] - yy[None])
    diff = st[:, None] != st[None]
    same = ~diff
    np.fill_diagonal(same, False)
    for tol in (.5,):
        res[f"median_gap_between_tol{tol}"] = float(np.median(gap[diff & (D < tol)]))
        res[f"median_gap_within_tol{tol}"] = float(np.median(gap[same & (D < tol)]))
    has = POR.por.notna().values
    kp = k[has[k.row.values]]
    por = POR[has].groupby("study").por.mean()
    bias = (kp.y_true - kp.y_pred).groupby(kp.study).mean()
    j = pd.concat([por, bias.rename("bias")], axis=1, join="inner")
    sr = spearmanr(j.por, j.bias)
    res.update(porosity_rows=int(has.sum()), porosity_studies=len(j), porosity_rho=float(sr[0]), porosity_p=float(sr[1]))
    json.dump(res, open(OUT / "loso.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    s = cross_validation()
    leverage(float(s.loc["test_r2", "mean"]))
    final_model()
    loso()
    permutation()
