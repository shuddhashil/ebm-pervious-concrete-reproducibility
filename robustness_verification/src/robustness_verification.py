"""Local, reproducible robustness/optimisation verification suite.

Run from ``robustness_verification/src`` after copying into the EBM project.
All candidate origins call CanonicalEvaluator; this is intentionally separate
from legacy scripts so no historic output is altered.
"""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]
SUITE = ROOT / "08_analysis_code_suite"
if str(SUITE) not in sys.path:
    sys.path.insert(0, str(SUITE))

from shared import ebm_engine as eng

import argparse
import datetime as dt
import json
import platform
import warnings
from dataclasses import asdict, dataclass
import importlib.metadata as im
import numpy as np
import pandas as pd
from scipy.optimize import minimize as sp_minimize
from scipy.spatial.distance import cdist
from scipy.stats import chi2, qmc

@dataclass(frozen=True)
class SupportConfig:
    kde_cutoff: float = .10
    mahal_quantile: float = .95
    local_tolerance_mpa: float = 6.
    include_nn_hard_limit: bool = False
    nn_hard_limit: float = 2.
    active_tolerance: float = .02


def _threshold(ctx, config):
    return ctx.sup_d_max * \
        np.sqrt(chi2.ppf(config.mahal_quantile, 8) / chi2.ppf(.95, 8))


class CanonicalEvaluator:
    def __init__(self, ctx, config=SupportConfig()
                 ): self.ctx, self.config = ctx, config

    def evaluate(self, X, target_strength=None):
        X = np.asarray(X, float)
        if X.ndim != 2 or X.shape[1] != 3: raise ValueError('Candidates must have shape (n,3): FA/B %, B/A, W/B')
        if not np.isfinite(X).all(): raise ValueError('Candidates contain NaN/inf')
        fa, ba, wb = X.T; c = self.ctx; n = len(X)
        raw = np.asarray(
    eng.predict_cs_raw(
        c, eng.build_ebm_input(
            c, fa, ba, wb)), float)
        cs = np.exp(raw) if c.use_log else raw
        _, effective, bag, local_sigma, unc = eng.effective_cs(c, fa, ba, wb)
        effective = np.asarray(effective); bag = np.asarray(
            bag); local_sigma = np.asarray(local_sigma); unc = np.asarray(unc)
        adraw = eng.ad_input_matrix(c, fa, ba, wb); supstd = c.sup_scaler.transform(
            adraw); ifstd = c.ad_scaler.transform(adraw)
        logdens = c.sup_kde.score_samples(
            supstd); kde = np.exp(logdens) / c.sup_density_max
        diff = supstd - c.sup_mean_vec; mahal = np.sqrt(np.maximum(
            np.einsum('ij,jk,ik->i', diff, c.sup_inv_cov, diff), 0))
        dist = cdist(
    ifstd,
    c.train_ad_scaled); nn = dist.min(1); ix = np.argpartition(
        dist,
        eng.LOCAL_K - 1,
        axis=1)[
            :,
             :eng.LOCAL_K]
        local_ref = c.train_cs_obs[ix].mean(
            1); local_gap = np.abs(cs - local_ref)
        if_score = c.ad_if_model.score_samples(
            ifstd); if_slack = if_score + c.ad_if_threshold
        trio = X - c.trio_mean; trio_metric = np.einsum(
    'ij,jk,ik->i',
    trio,
    c.trio_cov_inv,
     trio); trio_slack = c.trio_mahal_thresh - trio_metric
        mt = _threshold(c, self.config); mslack = mt - mahal
        log_pass = (
    raw >= c.log_s_min) & (
        raw <= c.log_s_max); local_slack = self.config.local_tolerance_mpa - local_gap
        bounds = np.asarray(c.bounds); lower = X - \
                            bounds[:, 0]; upper = bounds[:, 1] - X
        in_bounds = (lower >= -1e-10).all(1) & (upper >= -1e-10).all(1)
        strength = np.full(
    n, np.nan) if target_strength is None else effective - float(target_strength)
        base = in_bounds & (
    if_slack >= 0) & (
        trio_slack >= 0) & (
            local_slack >= 0)
        evidence = base & (
    mslack >= 0) & (
        kde >= self.config.kde_cutoff) & log_pass
        if self.config.include_nn_hard_limit: evidence &= nn <= self.config.nn_hard_limit
        reason = []
        for i in range(n):
            r = []
            if not in_bounds[i]: r.append('bounds')
            if target_strength is not None and strength[i] < 0: r.append('strength')
            if if_slack[i] < 0: r.append('isolation_forest')
            if trio_slack[i] < 0: r.append('legacy_trio_mahal')
            if local_slack[i] < 0: r.append('local_consistency')
            if mslack[i] < 0: r.append('support_mahal')
            if kde[i] < self.config.kde_cutoff: r.append('kde')
            if not log_pass[i]: r.append('log_extrapolation')
            if self.config.include_nn_hard_limit and nn[i] > self.config.nn_hard_limit: r.append('nearest_neighbor')
            reason.append(';'.join(r))
        out = pd.DataFrame({'FA_B_pct': fa, 'BA': ba, 'WB': wb, 'CS_pred_MPa': cs, 'CS_effective_MPa': effective,
          'model_uncertainty_MPa': unc, 'bag_std_MPa': bag, 'local_sigma_MPa': local_sigma, 'CO2_index': eng.co2_central(fa / 100, ba, wb),
          'strength_slack_MPa': strength, 'if_score': if_score, 'if_slack': if_slack, 'if_pass': if_slack >= 0,
          'mahal_distance': mahal, 'mahal_threshold': mt, 'mahal_slack': mslack, 'mahal_pass': mslack >= 0,
          'kde_support': kde, 'kde_cutoff': self.config.kde_cutoff, 'kde_margin': kde - self.config.kde_cutoff, 'kde_pass': kde >= self.config.kde_cutoff,
          'nn_distance': nn, 'nn_threshold': self.config.nn_hard_limit if self.config.include_nn_hard_limit else np.nan,
          'local_reference_CS_MPa': local_ref, 'local_consistency_gap_MPa': local_gap, 'local_consistency_slack_MPa': local_slack, 'local_consistency_pass': local_slack >= 0,
          'log_extrapolation_pass': log_pass, 'trio_mahal_metric': trio_metric, 'trio_mahal_slack': trio_slack,
          'FA_B_lower_slack': lower[:, 0], 'FA_B_upper_slack': upper[:, 0], 'BA_lower_slack': lower[:, 1], 'BA_upper_slack': upper[:, 1],
          'WB_lower_slack': lower[:, 2], 'WB_upper_slack': upper[:, 2], 'base_feasible': base, 'evidence_supported': evidence, 'infeasibility_reasons': reason})
        if target_strength is not None:
            out['base_feasible'] &= out.strength_slack_MPa >= 0; out['evidence_supported'] &= out.strength_slack_MPa >= 0
        return out


def sobol(bounds, m, seed):
    u = qmc.Sobol(3, scramble=True, seed=int(seed)).random_base2(int(m)); b = np.asarray(bounds); return qmc.scale(u, b[:, 0], b[:, 1])


def optimum(df, column):
    d = df[df[column]]
    return d.loc[d.CO2_index.idxmin()].copy() if len(d) else None


def add_optimum(rows, target, kind, record):
    if record is None: rows.append({'target_strength_MPa': target, 'solution_type': kind, 'feasible': False}); return
    r = record.to_dict(); r.update(
    target_strength_MPa=target,
    solution_type=kind,
     feasible=True); rows.append(r)


def write_fig(fig, path):
    fig.savefig(
    path.with_suffix('.png'),
    dpi=300,
    bbox_inches='tight'); fig.savefig(
        path.with_suffix('.pdf'),
         bbox_inches='tight')


def _local_refine(ev, X0, bounds_arr):
    refined = []
    for x0 in X0:
        try:
            res = sp_minimize(lambda x: float(ev.evaluate(x.reshape(1, -1)).CO2_index.iloc[0]),
                x0=np.asarray(x0, float), method='L-BFGS-B',
                bounds=[(bounds_arr[i, 0], bounds_arr[i, 1])
                         for i in range(3)],
                options={'maxiter': 200, 'ftol': 1e-12})
            if res.success: refined.append(res.x)
            else: refined.append(x0)
        except Exception: refined.append(x0)
    return np.array(refined)


def _pareto_mask(costs):
    n = len(costs); mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]: continue
        for j in range(n):
            if i == j or not mask[j]: continue
            if costs[j, 0] <= costs[i, 0] and costs[j, 1] <= costs[i, 1] and (
                costs[j, 0] < costs[i, 0] or costs[j, 1] < costs[i, 1]):
                mask[i] = False; break
    return mask


def _bag_sign_stability(ctx, fa, ba, wb, fa_shift):
    X = np.array([[fa, ba, wb], [fa + fa_shift, ba, wb]])
    X_ebm = eng.build_ebm_input(ctx, X[:, 0], X[:, 1], X[:, 2])
    bag_mat = eng.predict_bag_matrix(ctx, X_ebm)
    if bag_mat is None or bag_mat.shape[0] < 2:
        return {
    'bag_pct_positive_delta': np.nan,
    'sign_stability': 'bag_matrix_unavailable',
     'bag_std_delta_MPa': np.nan}
    cs_a = np.exp(bag_mat[:, 0]) if ctx.use_log else bag_mat[:, 0]
    cs_b = np.exp(bag_mat[:, 1]) if ctx.use_log else bag_mat[:, 1]
    delta = cs_b - cs_a
    pct_pos = float((delta > 0).mean())
    if pct_pos >= 0.9: stab = 'consistently_beneficial'
    elif pct_pos <= 0.1: stab = 'consistently_detrimental'
    elif 0.4 <= pct_pos <= 0.6: stab = 'sign_uncertain'
    else: stab = 'mixed_majority_' + ('pos' if pct_pos > 0.5 else 'neg')
    return {
    'bag_pct_positive_delta': pct_pos,
    'sign_stability': stab,
    'bag_std_delta_MPa': float(
        delta.std())}


def _has(p):
 try: im.version(p); return True
 except im.PackageNotFoundError: return False


def run(args):
    out = ROOT / 'robustness_verification'; (out / 'outputs').mkdir(exist_ok=True); (out / 'figures').mkdir(
        exist_ok=True); (out / 'configs').mkdir(exist_ok=True); (out / 'logs').mkdir(exist_ok=True)
    config = SupportConfig(); (
    out / 'configs' / 'run_config.json').write_text(
        json.dumps(
            {
                'support': asdict(config),
                'sobol_primary_power': args.sobol_power,
                'qmc_powers': args.qmc_powers,
                'scrambles': args.scrambles,
                'seeds': list(
                    range(
                        args.scrambles)),
                        'refine_top_n': args.refine_top_n,
                        'timestamp_utc': dt.datetime.now(dt.timezone.utc).isoformat().replace('+00:00', 'Z'),
                        'python': sys.version,
                        'platform': platform.platform(),
                        'packages': {
                            p: im.version(p) for p in [
                                'numpy',
                                'pandas',
                                'scipy',
                                'scikit-learn',
                                'interpret'] if _has(p)}},
                                 indent=2))
    ctx = eng.load_context(False); ev = CanonicalEvaluator(
    ctx,
    config); targets = sorted(
        pd.read_csv(
            ROOT /
             '09_analysis_outputs/analysis_02_optimizer_stability/stability_runs.csv').cs_target.unique())
    bounds_arr = np.asarray(ctx.bounds)

    # ── 1. Global Sobol search + local refinement ───────────────────────────
    cloud = ev.evaluate(
    sobol(
        ctx.bounds,
        args.sobol_power,
         args.global_seed)); optrows = []
    for t in targets:
      d = cloud.copy(); d['strength_slack_MPa'] = d.CS_effective_MPa - \
                     t; d['base_feasible'] &= d.strength_slack_MPa >= 0; d['evidence_supported'] &= d.strength_slack_MPa >= 0
      add_optimum(
    optrows, t, 'absolute_base', optimum(
        d, 'base_feasible')); add_optimum(
            optrows, t, 'evidence_supported', optimum(
                d, 'evidence_supported'))
    refined_rows = []
    for t in targets:
      for kind, col in [('absolute_base', 'base_feasible'),
                         ('evidence_supported', 'evidence_supported')]:
        d = cloud.copy(); d['strength_slack_MPa'] = d.CS_effective_MPa - \
                       t; d[col] &= d.strength_slack_MPa >= 0
        feas = d[d[col]]
        if len(feas) == 0: continue
        seeds = feas.nsmallest(min(args.refine_top_n, len(feas)), 'CO2_index')[
                               ['FA_B_pct', 'BA', 'WB']].values
        ref_X = _local_refine(ev, seeds, bounds_arr)
        if len(ref_X) == 0: continue
        with warnings.catch_warnings():
          warnings.simplefilter('ignore'); ref_eval = ev.evaluate(ref_X)
        ref_eval['strength_slack_MPa'] = ref_eval.CS_effective_MPa - \
            t; ref_eval[col] &= ref_eval.strength_slack_MPa >= 0
        best = optimum(ref_eval, col)
        if best is not None:
          rr = best.to_dict(); rr.update(
    target_strength_MPa=t,
    solution_type=kind + '_refined',
     feasible=True); refined_rows.append(rr)
    if refined_rows:
      ref_df = pd.DataFrame(refined_rows)
      for _, rr in ref_df.iterrows():
        base_kind = rr.solution_type.replace('_refined', '')
        existing_idx = [i for i, r in enumerate(
            optrows) if r['target_strength_MPa'] == rr['target_strength_MPa'] and r['solution_type'] == base_kind]
        if existing_idx:
          idx = existing_idx[0]
          if optrows[idx].get(
    'feasible',
    False) and rr.get(
        'CO2_index',
        np.inf) < optrows[idx].get(
            'CO2_index',
             np.inf):
            optrows[idx] = rr
        else: optrows.append(rr)
    opt = pd.DataFrame(optrows); opt.to_csv(
    out / 'outputs' / 'verified_optima_by_target.csv', index=False)

    # ── 2. DE gap comparison ────────────────────────────────────────────────
    de = pd.read_csv(
    ROOT /
     '09_analysis_outputs/analysis_02_optimizer_stability/stability_runs.csv'); gaps = []
    for t in targets:
      g = de[(de.cs_target == t) & (de.feasible == True)]; ver = opt[(
          opt.target_strength_MPa == t) & (opt.solution_type == 'absolute_base')]
      if len(ver) == 0 or not bool(ver.iloc[0].feasible): continue
      v = float(ver.iloc[0].CO2_index)
      for _, x in g.iterrows():
        gap = x.CO2_index - v
        rel = gap / abs(v) if abs(v) > 1e-12 else 0.
        if abs(gap) <= 1e-8: cls = 'globally_competitive'
        elif gap < 0: cls = 'de_superior'
        elif rel <= .01: cls = 'near_optimal'
        else: cls = 'clearly_inferior'
        gaps.append({'target_strength_MPa': t,
    'seed': x.seed,
    'DE_objective': x.CO2_index,
    'verified_best_objective': v,
    'optimality_gap': gap,
    'relative_gap_pct': 100 * rel,
     'classification': cls})
    pd.DataFrame(gaps).to_csv(
    out /
    'outputs' /
    'de_optimality_gaps.csv',
     index=False)

    # ── 3. QMC convergence (powers 15/16/17) ───────────────────────────────
    qrows = []
    for power in args.qmc_powers:
      for seed in range(args.scrambles):
        z = ev.evaluate(sobol(ctx.bounds, power, 10000 + seed))
        for t in targets:
          strength = z.CS_effective_MPa >= t; qrows.append(
    {
        'target_strength_MPa': t,
        'n': 2**power,
        'power': power,
        'scramble_seed': seed,
        'base_feasible_fraction': float(
            (z.base_feasible & strength).mean()),
            'evidence_supported_fraction': float(
                (z.evidence_supported & strength).mean())})
    qdf = pd.DataFrame(qrows); qdf.to_csv(
    out /
    'outputs' /
    'qmc_supported_fraction_by_scramble.csv',
     index=False)
    summary = qdf.groupby(
    [
        'target_strength_MPa', 'n', 'power']).agg(
            mean=(
                'evidence_supported_fraction', 'mean'), std=(
                    'evidence_supported_fraction', 'std'), standard_error=(
                        'evidence_supported_fraction', 'sem'), p025=(
                            'evidence_supported_fraction', lambda x: x.quantile(.025)), p975=(
                                'evidence_supported_fraction', lambda x: x.quantile(.975)), base_mean=(
                                    'base_feasible_fraction', 'mean')).reset_index(); summary.to_csv(
                                        out / 'outputs' / 'qmc_convergence_summary.csv', index=False)

    # ── 4. Threshold sensitivity matrix ─────────────────────────────────────
    sens = []
    for k in (.05, .10, .20):
     for mq in (.90, .95, .99):
      for lt in (4., 6., 8.):
       e = CanonicalEvaluator(
    ctx,
    SupportConfig(
        k,
        mq,
        lt)); z = e.evaluate(
            sobol(
                ctx.bounds,
                args.sobol_power,
                 args.global_seed))
       for t in targets:
        ss = z.CS_effective_MPa >= t; mask = z.evidence_supported & ss; best = optimum(
    z.assign(
        evidence_supported=mask),
        'evidence_supported'); r = {
            'target_strength_MPa': t,
            'kde_cutoff': k,
            'mahal_quantile': mq,
            'local_tolerance_MPa': lt,
            'supported_fraction': float(
                mask.mean())};
        if best is not None: r.update({'minimum_supported_CO2': best.CO2_index, 'FA_B_pct': best.FA_B_pct, 'BA': best.BA, 'WB': best.WB, 'active_constraints': best.infeasibility_reasons})
        sens.append(r)
    pd.DataFrame(sens).to_csv(
    out /
    'outputs' /
    'support_sensitivity_results.csv',
     index=False)

    # ── 5. Support premium + Pareto front ────────────────────────────────────
    piv = opt.pivot(
    index='target_strength_MPa',
    columns='solution_type',
    values='CO2_index').reset_index(); piv['support_premium_pct'] = 100 * (
        piv.get('evidence_supported') - piv.get('absolute_base')) / piv.get('absolute_base'); piv.to_csv(
            out / 'outputs' / 'support_premium_by_target.csv',
             index=False)
    pareto_targets = [
    t for t in [
        10.,
        12.,
        14.,
        16.,
        17.,
        18.,
        20.,
        22.,
        24.,
        26.,
        28.,
         30.] if t in targets]
    pareto_all = []
    for t in pareto_targets:
      feasible = cloud[(cloud.CS_effective_MPa >= t)
                        & (cloud.evidence_supported)]
      if len(feasible) < 3: continue
      costs = feasible[['CO2_index', 'kde_support']].values
      mask = _pareto_mask(costs)
      pf = feasible[mask].copy(
      ); pf['pareto_front'] = True; pf['target_strength_MPa'] = t
      rest = feasible[~mask].copy(
      ); rest['pareto_front'] = False; rest['target_strength_MPa'] = t
      pareto_all.append(pd.concat([pf, rest]))
    if pareto_all: pd.concat(pareto_all).to_csv(out / 'outputs' / 'pareto_front.csv', index=False)

    # ── 6. Active-constraint slacks ─────────────────────────────────────────
    active = opt[opt.feasible == True].copy(); cols = ['strength_slack_MPa',
    'if_slack',
    'trio_mahal_slack',
    'mahal_slack',
    'kde_margin',
    'local_consistency_slack_MPa',
    'FA_B_lower_slack',
    'FA_B_upper_slack',
    'BA_lower_slack',
    'BA_upper_slack',
    'WB_lower_slack',
     'WB_upper_slack'];
    ar = []
    for _, r in active.iterrows():
      for col in cols:
       v = r.get(
    col,
    np.nan); scale = max(
        1.,
        abs(
            r.get(
                'mahal_threshold',
                1.)) if 'mahal' in col else config.local_tolerance_mpa if 'local' in col else 1.); ar.append(
                    {
                        'target_strength_MPa': r.target_strength_MPa,
                        'solution_type': r.solution_type,
                        'constraint': col,
                        'raw_slack': v,
                        'normalised_slack': v / scale,
                        'active_or_near_active': abs(
                            v / scale) <= config.active_tolerance})
    pd.DataFrame(ar).to_csv(
    out /
    'outputs' /
    'active_constraint_slacks.csv',
     index=False)

    # ── 7. FA/B × B/A interaction grid (with KDE, per-bag sign-stability) ───
    qs = np.quantile(ctx.df['W/B'], [.25, .5, .75]); grid = []
    for wb in qs:
      for fa in np.linspace(*ctx.bounds[0], args.grid_n):
       for ba in np.linspace(*ctx.bounds[1], args.grid_n):
        if fa + 10 > ctx.bounds[0][1]: continue
        a = ev.evaluate([[fa, ba, wb]]).iloc[0]; b = ev.evaluate(
            [[fa + 10, ba, wb]]).iloc[0]
        stab = _bag_sign_stability(ctx, fa, ba, wb, 10.)
        r = a.to_dict(); r.update({'W_B_percentile_value': wb, 'delta_CS_FA10_MPa': b.CS_pred_MPa - a.CS_pred_MPa, 'delta_CO2_FA10': b.CO2_index - a.CO2_index, 'contrast_valid': True,
          'kde_support_shifted': b.kde_support, 'evidence_supported_shifted': b.evidence_supported, **stab})
        grid.append(r)
    pd.DataFrame(grid).to_csv(
    out /
    'outputs' /
    'fa_b_ba_interaction_grid.csv',
     index=False)

    # ── 8. Carbon uncertainty (placeholder) ──────────────────────────────────
    (out / 'outputs' / 'carbon_uncertainty_summary.csv').write_text(
        'status,note\nnot_parameterised,"No defensible emission-factor distributions located; machinery deliberately not used for scientific inference."\n')

    # ── 9. Figures ───────────────────────────────────────────────────────────
    _plots(
    out,
    opt,
    pd.DataFrame(gaps),
    summary,
    piv,
    pd.DataFrame(ar),
    cloud,
    targets,
     qdf)
    _report(out, targets, ctx, config)


def _plots(out, opt, gaps, q, premium, active, cloud, targets, qdf):
  import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

  # 1. Min-CO₂ vs target strength
  f, a = plt.subplots();
  for name, g in opt[opt.feasible == True].groupby('solution_type'):
    a.plot(g.target_strength_MPa, g.CO2_index, 'o-', label=name)
  a.set(xlabel='Target compressive strength (MPa)',
        ylabel='Embodied CO₂ index'); a.legend(fontsize=7);
  a.set_title('Minimum CO₂ vs target strength')
  write_fig(f, out / 'figures' / 'minimum_co2_vs_strength'); plt.close(f)

  # 2. DE optimality gap (classified)
  f, a = plt.subplots();
  if len(gaps):
    for cls, marker in [('de_superior', 'v'),
                        ('globally_competitive', 's'),
                        ('near_optimal', '^'),
                        ('clearly_inferior', 'x')]:
      sub = gaps[gaps.classification == cls]
      if len(sub):
        a.scatter(sub.target_strength_MPa, sub.optimality_gap,
                  marker=marker, label=cls, alpha=.7)
  a.axhline(0, color='k', linestyle='--', linewidth=.8)
  a.set(xlabel='Target compressive strength (MPa)',
        ylabel='DE objective gap (CO₂ index)'); a.legend(fontsize=7);
  a.set_title('DE vs Sobol verified best')
  write_fig(f, out / 'figures' / 'de_optimality_gap'); plt.close(f)

  # 3. QMC convergence (all resolutions)
  f, a = plt.subplots();
  for n_val in sorted(q.n.unique()):
    g = q[q.n == n_val];
    a.plot(g.target_strength_MPa, g['mean'], 'o-', label=f'n={n_val}', markersize=4)
    a.fill_between(g.target_strength_MPa, g.p025, g.p975, alpha=.15)
  a.set(xlabel='Target compressive strength (MPa)',
        ylabel='Evidence-supported fraction'); a.legend(fontsize=7);
  a.set_title('QMC convergence across resolutions')
  write_fig(f, out / 'figures' / 'supported_fraction_qmc'); plt.close(f)

  # 4. Support premium
  f, a = plt.subplots();
  a.plot(premium.target_strength_MPa, premium.support_premium_pct, 'o-')
  a.set(xlabel='Target compressive strength (MPa)',
        ylabel='Support premium (%)'); a.set_title(
    'Support premium: evidence-constrained vs unconstrained')
  write_fig(f, out / 'figures' / 'support_premium'); plt.close(f)

  # 5. Active-constraint heatmap
  if len(active):
    pivot = active.pivot_table(
      index='constraint', columns='target_strength_MPa',
      values='normalised_slack', aggfunc='first')
    # Keep the rendered image within Matplotlib's maximum pixel dimensions.
    # The former per-target/per-row scaling created impractically large files.
    fig_width = min(18, max(8, 4 + 0.55 * len(pivot.columns)))
    fig_height = min(12, max(5, 2 + 0.35 * len(pivot.index)))
    f, a = plt.subplots(figsize=(fig_width, fig_height))
    im_h = a.imshow(pivot.values, aspect='auto', cmap='RdYlGn_r', vmin=-.15, vmax=.15)
    a.set_xticks(range(len(pivot.columns)));
    a.set_xticklabels([f'{c:.1f}' for c in pivot.columns], rotation=45, fontsize=7)
    a.set_yticks(range(len(pivot.index))); a.set_yticklabels(pivot.index, fontsize=7)
    a.set_xlabel('Target CS (MPa)'); a.set_ylabel('Constraint');
    a.set_title('Normalised slack heatmap (green=comfortable, red=active)')
    plt.colorbar(im_h, ax=a, shrink=.8);
    write_fig(f, out / 'figures' / 'active_constraint_heatmap'); plt.close(f)

  # 6. Pareto front scatter (CO₂ vs KDE support)
  pareto_path = out / 'outputs' / 'pareto_front.csv'
  if pareto_path.exists():
    pf = pd.read_csv(pareto_path); f, a = plt.subplots()
    for t, g in pf.groupby('target_strength_MPa'):
      on = g[g.pareto_front]; off = g[~g.pareto_front]
      a.scatter(off.CO2_index, off.kde_support, s=10, alpha=.3,
                label=f'{t} MPa (all)')
      a.scatter(on.CO2_index, on.kde_support, s=30, edgecolors='k',
                facecolors='none', label=f'{t} MPa (Pareto)')
    a.set(xlabel='Embodied CO₂ index', ylabel='KDE support'); a.legend(fontsize=6);
    a.set_title('Pareto front: CO₂ vs KDE support')
    write_fig(f, out / 'figures' / 'pareto_co2_vs_kde'); plt.close(f)

  # 7. QMC convergence check (high-resolution only: 32k vs 65k vs 131k)
  f, a = plt.subplots(); hi = qdf[qdf.power >= 15]
  if len(hi):
    for t in sorted(hi.target_strength_MPa.unique()):
      sub = hi[hi.target_strength_MPa == t].sort_values('n')
      means = sub.groupby('n')['evidence_supported_fraction'].agg(['mean', 'std']).reset_index()
      a.errorbar(means.n, means['mean'], yerr=means['std'], fmt='o-', label=f'{t} MPa', markersize=4)
  a.set_xscale('log', base=2)
  a.set(xlabel='Sobol sample size (2^power)',
        ylabel='Evidence-supported fraction'); a.legend(fontsize=7);
  a.set_title('Convergence check: 32k → 65k → 131k')
  write_fig(f, out / 'figures' / 'qmc_convergence_check'); plt.close(f)

  # 8. FA interaction sign-stability atlas
  grid_path = out / 'outputs' / 'fa_b_ba_interaction_grid.csv'
  if grid_path.exists():
    gdf = pd.read_csv(grid_path)
    stab_map = {'consistently_beneficial': 0,
                'mixed_majority_pos': 1,
                'sign_uncertain': 2,
                'mixed_majority_neg': 3,
                'consistently_detrimental': 4,
                'bag_matrix_unavailable': np.nan}
    gdf['stab_code'] = gdf.sign_stability.map(stab_map)
    for wb_val, g in gdf.groupby('W_B_percentile_value'):
      f, a = plt.subplots()
      sc = a.scatter(g.FA_B_pct, g.BA, c=g.stab_code, cmap='RdYlGn', vmin=0, vmax=4, s=8)
      plt.colorbar(sc, ax=a, ticks=[0, 1, 2, 3, 4],
                   label='sign_stability').ax.set_yticklabels(
        ['beneficial', 'mixed+', 'uncertain', 'mixed−', 'detrimental'])
      a.set(xlabel='FA/B (%)', ylabel='B/A');
      a.set_title(f'FA/B×B/A sign-stability (W/B={wb_val:.4f})')
      write_fig(f, out / 'figures' / f'fa_interaction_wb_{wb_val:.4f}'); plt.close(f)

def _report(out,targets,ctx,config):
 (out/'REPORT.md').write_text(f'''# Robustness and optimisation verification report

## Repository audit

Reused the saved primary EBM, phase-3 Isolation Forest/KDE/Mahalanobis objects, sigma model, carbon equation and decision bounds from `08_analysis_code_suite/shared/ebm_engine.py`. The decision variables are FA/B (%), B/A and W/B; targets read from legacy `stability_runs.csv`: {targets}.

## Methodological inconsistency discovered

Legacy DE base feasibility applies effective strength, IF, local consistency and a three-variable Mahalanobis constraint. Legacy LHS reliability additionally applies KDE support and log-space screening. This suite makes both definitions explicit; each candidate is evaluated only by `CanonicalEvaluator`.

## Results and limits

All numerical outputs are in `outputs/`; figures are in `figures/`. The report deliberately does not turn numerical results into causal material claims. `carbon_uncertainty_summary.csv` records that factor distributions were not defensibly available, so no scientific carbon-uncertainty result is presented. Mahalanobis sensitivity maps the inherited 95% stored threshold on the chi-square quantile scale (8 support dimensions).

## Key outputs

- `verified_optima_by_target.csv`: Sobol + L-BFGS-B refined global optima per target
- `de_optimality_gaps.csv`: DE seed quality classification against verified global best
- `qmc_convergence_summary.csv`: QMC at 2^15/16/17 with 16 scrambles
- `support_sensitivity_results.csv`: 27-combo KDE/Mahal/local tolerance sensitivity
- `support_premium_by_target.csv`: Cost of evidence constraint per target
- `pareto_front.csv`: CO₂ vs KDE-support Pareto sets at key targets
- `active_constraint_slacks.csv`: Normalised slack per constraint per optimum
- `fa_b_ba_interaction_grid.csv`: FA/B×B/A grid with per-bag sign-stability
- `carbon_uncertainty_summary.csv`: Placeholder (distributions not available)

## Interpretation safeguards

DE seed frequency is diagnostic only. The Sobol best objective is the global cross-check (refined with L-BFGS-B). A boundary-active solution identifies a constrained optimum, not a physical material optimum. Per-bag sign-stability distinguishes robust from uncertain FA/B effects.
''',encoding='utf-8')


def render_existing_outputs():
    """Rebuild figures and the report from a completed run's saved tables."""
    out = ROOT / 'robustness_verification'
    outputs = out / 'outputs'
    required = [
        'verified_optima_by_target.csv', 'de_optimality_gaps.csv',
        'qmc_convergence_summary.csv', 'qmc_supported_fraction_by_scramble.csv',
        'support_premium_by_target.csv', 'active_constraint_slacks.csv']
    missing = [name for name in required if not (outputs / name).exists()]
    if missing:
        raise FileNotFoundError('Cannot rebuild figures; missing: ' + ', '.join(missing))
    opt = pd.read_csv(outputs / 'verified_optima_by_target.csv')
    gaps = pd.read_csv(outputs / 'de_optimality_gaps.csv')
    summary = pd.read_csv(outputs / 'qmc_convergence_summary.csv')
    qdf = pd.read_csv(outputs / 'qmc_supported_fraction_by_scramble.csv')
    premium = pd.read_csv(outputs / 'support_premium_by_target.csv')
    active = pd.read_csv(outputs / 'active_constraint_slacks.csv')
    targets = sorted(opt.target_strength_MPa.dropna().unique())
    _plots(out, opt, gaps, summary, premium, active, None, targets, qdf)
    _report(out, targets, None, SupportConfig())

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sobol-power',type=int,default=17);p.add_argument('--qmc-powers',type=int,nargs='+',default=[15,16,17]);p.add_argument('--scrambles',type=int,default=16);p.add_argument('--global-seed',type=int,default=20260829);p.add_argument('--grid-n',type=int,default=31);p.add_argument('--refine-top-n',type=int,default=50,help='Top Sobol candidates used as L-BFGS-B seeds');p.add_argument('--render-only',action='store_true',help='Rebuild figures and report from saved results without rerunning calculations');a=p.parse_args();render_existing_outputs() if a.render_only else run(a)
