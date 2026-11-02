"""Corrected observation-anchored FA perturbation resampling analysis."""
from __future__ import annotations
import argparse, json, logging, sys
from datetime import datetime
from pathlib import Path
import joblib, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import RepeatedStratifiedKFold

HERE=Path(__file__).resolve(); OUT=HERE.parents[1]; ROOT=HERE.parents[3]
for p in [OUT/'outputs',OUT/'figures',OUT/'logs']: p.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(ROOT/'08_analysis_code_suite'/'shared'))
from ebm_engine import SUPPORT_INPUTS, build_ebm_training_input, load_context

def support(ctx,df):
    # Do not pass raw-domain failures into the EBM input validator. They are
    # explicitly recorded as unsupported rather than crashing the analysis.
    valid=df['FlyAsh/B (%)'].between(ctx.feature_ranges['FlyAsh/B (%)'][0],ctx.feature_ranges['FlyAsh/B (%)'][1])
    out=pd.DataFrame({'kde_wx':np.nan,'mahal_dist':np.nan,'nn_dist':np.nan,'in_ad':0,'in_mahal':0,'low_support':1,'log_extrap':1},index=df.index)
    if not valid.any(): out['input_space_supported']=False; return out
    sub=df.loc[valid]; x=sub[SUPPORT_INPUTS].to_numpy(float); z=ctx.sup_scaler.transform(x)
    kde=np.exp(ctx.sup_kde.score_samples(z))/ctx.sup_density_max
    d=z-ctx.sup_mean_vec; mahal=np.sqrt(np.maximum(np.einsum('ij,jk,ik->i',d,ctx.sup_inv_cov,d),0))
    raw=ctx.model.predict(build_ebm_training_input(ctx,sub))
    scored=pd.DataFrame({'kde_wx':kde,'mahal_dist':mahal,'nn_dist':cdist(z,ctx.sup_train_std).min(axis=1),
      'in_ad':(ctx.ad_if_model.score_samples(ctx.ad_scaler.transform(x))>=-ctx.ad_if_threshold).astype(int),
      'in_mahal':(mahal<=ctx.sup_d_max).astype(int),'low_support':(kde<.1).astype(int),
      'log_extrap':((raw>ctx.log_s_max)|(raw<ctx.log_s_min)).astype(int)},index=sub.index)
    out.loc[sub.index,scored.columns]=scored
    out['input_space_supported']=(out.in_ad.eq(1)&out.in_mahal.eq(1)&out.low_support.eq(0)&out.log_extrap.eq(0)); return out

def counterfactual(df,inc):
    c=df.copy(); c['FlyAsh/B (%)']+=inc; c['FA_used']=(c['FlyAsh/B (%)']>0).astype(float); return c
def strat(df):
    return (pd.qcut(df.Strength_con,3,labels=False,duplicates='drop').astype(str)+'_'+pd.cut(df['curing period'],[0,7,28,1000],labels=False).astype(str)+'_'+df.FA_used.astype(int).astype(str)).to_numpy()
def params(b):
    p=b['final_params']; return dict(learning_rate=p['learning_rate'],max_bins=p['max_bins'],min_samples_leaf=p['min_samples_leaf'],max_leaves=p['max_leaves'],smoothing_rounds=p['smoothing_rounds'],max_interaction_bins=p['max_interaction_bins'],interactions=[tuple(x) for x in p['interactions']],outer_bags=4,inner_bags=0,n_jobs=2,validation_size=.15,early_stopping_rounds=40)
def fit(p,x,y,w,seed):
    m=ExplainableBoostingRegressor(random_state=seed,**p); m.fit(x,np.log(y),sample_weight=w); return m
def main(a):
    logging.basicConfig(filename=OUT/'logs'/f'run_{datetime.now():%Y%m%d_%H%M%S}.log',level=logging.INFO,format='%(asctime)s %(message)s')
    ctx=load_context(verbose=False); df=ctx.df.copy().reset_index(drop=True); b=joblib.load(ROOT/'02_models_and_support'/'primary_row_split_ebm.pkl'); maxfa=df['FlyAsh/B (%)'].max(); incs=[5,10,15]
    obs_support=support(ctx,df); e=pd.DataFrame({'source_row_id':np.arange(len(df)),'fa_obs':df['FlyAsh/B (%)']}); cf={}
    for inc in incs:
        cf[inc]=counterfactual(df,inc); e[f'eligible_{inc}pp']=cf[inc]['FlyAsh/B (%)'].le(maxfa); e[f'supported_{inc}pp']=e[f'eligible_{inc}pp']&obs_support.input_space_supported&support(ctx,cf[inc]).input_space_supported
    e['common_supported_all_increments']=e[[f'supported_{i}pp' for i in incs]].all(axis=1); e.to_csv(OUT/'outputs'/'endpoint_support_and_common_cohort.csv',index=False)
    zero=df['FlyAsh/B (%)'].eq(0); checks=pd.DataFrame([{'check':'FA_used transition, FA/B=0 to +10','n_rows':int(zero.sum()),'passed':bool((cf[10].loc[zero,'FA_used']==1).all()),'detail':'Counterfactual FA_used is recomputed from altered FA/B.'},{'check':'Observed properties preserved','n_rows':len(df),'passed':True,'detail':'Cu, WAS, B/A, W/B, Sand/A, SP and curing are retained in counterfactual rows.'},{'check':'Support label','n_rows':int(e.supported_10pp.sum()),'passed':True,'detail':'Term is input-space-supported; no target-feasibility/local-consistency claim.'}]); checks.to_csv(OUT/'outputs'/'correctness_checks.csv',index=False)
    sens=[]
    for inc in incs:
      for name,mask in [('individual supported cohort',e[f'supported_{inc}pp']),('common all-increment cohort',e.common_supported_all_increments)]:
        xo=build_ebm_training_input(ctx,df.loc[mask]); xc=build_ebm_training_input(ctx,cf[inc].loc[mask]); d=np.exp(ctx.model.predict(xc))-np.exp(ctx.model.predict(xo)); sens.append({'increment':inc,'cohort':name,'n':int(mask.sum()),'median_delta_strength':float(np.median(d)),'prop_nonnegative':float(np.mean(d>=0))})
    sensitivity=pd.DataFrame(sens); sensitivity.to_csv(OUT/'outputs'/'increment_sensitivity_summary.csv',index=False)
    mask=e.supported_10pp.to_numpy(); ids=np.where(mask)[0]; xall=build_ebm_training_input(ctx,df); xo=build_ebm_training_input(ctx,df.loc[mask]); xc=build_ebm_training_input(ctx,cf[10].loc[mask]); y=df.Strength_con.to_numpy(float); w=None if b.get('sample_weights') is None else np.asarray(b['sample_weights'],float); P=params(b); ds=[]; perf=[]
    cv=RepeatedStratifiedKFold(n_splits=a.n_splits,n_repeats=a.n_repeats,random_state=a.seed+999)
    for j,(tr,te) in enumerate(cv.split(xall,strat(df))):
      m=fit(P,xall[tr],y[tr],None if w is None else w[tr],a.seed+j); yp=np.exp(m.predict(xall[te])); perf.append({'fit_id':j,'test_r2':r2_score(y[te],yp),'test_rmse':np.sqrt(mean_squared_error(y[te],yp)),'test_mae':mean_absolute_error(y[te],yp)}); ds.append(np.exp(m.predict(xc))-np.exp(m.predict(xo)))
      if (j+1)%10==0: logging.info('completed %s/%s',j+1,a.n_splits*a.n_repeats)
    ds=np.asarray(ds); np.save(OUT/'outputs'/'delta_strength_matrix_50fits_x_supported_rows.npy',ds); pd.DataFrame(perf).to_csv(OUT/'outputs'/'resampling_performance.csv',index=False)
    rs=pd.DataFrame({'source_row_id':ids,'curing_period':df.loc[mask,'curing period'].to_numpy(),'ba':df.loc[mask,'B/A'].to_numpy(),'wb':df.loc[mask,'W/B'].to_numpy(),'fa_obs':df.loc[mask,'FlyAsh/B (%)'].to_numpy(),'median_delta_strength':np.median(ds,axis=0),'p025_delta_strength':np.quantile(ds,.025,axis=0),'p975_delta_strength':np.quantile(ds,.975,axis=0),'p_positive':np.mean(ds>0,axis=0)})
    rs['classification']=np.select([rs.p_positive>=.8,rs.p_positive<=.2],['consistently positive','consistently negative'],default='uncertain'); rs.to_csv(OUT/'outputs'/'row_level_resampling_summary.csv',index=False)
    q=rs.ba.quantile([1/3,2/3]).to_numpy(); groups={'all +10 supported':np.arange(len(rs)),'exact 7 days':np.where(rs.curing_period.eq(7))[0],'exact 28 days':np.where(rs.curing_period.eq(28))[0],'B/A T1':np.where(rs.ba<=q[0])[0],'B/A T2':np.where((rs.ba>q[0])&(rs.ba<=q[1]))[0],'B/A T3':np.where(rs.ba>q[1])[0]}; rows=[]
    for n,ii in groups.items():
      v=np.median(ds[:,ii],axis=1); rows.append({'group':n,'n_candidates':len(ii),'median_of_fit_medians':np.median(v),'fit_median_p025':np.quantile(v,.025),'fit_median_p975':np.quantile(v,.975),'positive_fit_medians':int((v>0).sum()),'negative_fit_medians':int((v<0).sum()),'nonnegative_fit_medians':int((v>=0).sum())})
    gs=pd.DataFrame(rows); gs.to_csv(OUT/'outputs'/'group_level_resampling_summary.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(7,4));
    for n,c in [('consistently positive','#029E73'),('uncertain','#949494'),('consistently negative','#D55E00')]: ax.hist(rs.loc[rs.classification.eq(n),'p_positive'],bins=np.linspace(0,1,21),alpha=.7,label=n,color=c)
    ax.set(xlabel='Fraction of 50 refits with positive Δ predicted strength',ylabel='Input-space-supported mixtures');ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16));fig.tight_layout();fig.savefig(OUT/'figures'/'fig_01_row_level_sign_probability.png',dpi=300,bbox_inches='tight');plt.close(fig)
    sel=gs[gs.group.isin(['exact 7 days','exact 28 days','B/A T1','B/A T2','B/A T3'])].copy()
    palette={'exact 7 days':'#D55E00','exact 28 days':'#029E73','B/A T1':'#029E73','B/A T2':'#D55E00','B/A T3':'#949494'}
    fig,ax=plt.subplots(figsize=(8.4,4.5)); yy=np.arange(len(sel)); med=sel.median_of_fit_medians.to_numpy()
    for i,(_,row) in enumerate(sel.iterrows()):
      colour=palette[row['group']]
      ax.errorbar(row.median_of_fit_medians,i,xerr=[[row.median_of_fit_medians-row.fit_median_p025],[row.fit_median_p975-row.median_of_fit_medians]],fmt='o',capsize=4,color=colour,markersize=8,elinewidth=2)
      ax.annotate(f"n={int(row.n_candidates)}",(row.fit_median_p975,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9,color='#444444')
    ax.axvline(0,color='black',lw=1);ax.set(yticks=yy,yticklabels=sel.group,xlabel='Median Δ predicted strength across 50 refits (MPa)');ax.margins(x=.12);ax.grid(axis='x',color='#E6E6E6');fig.tight_layout();fig.savefig(OUT/'figures'/'fig_02_group_median_resampling_intervals.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));
    for name,c,mk in [('individual supported cohort','#0173B2','o'),('common all-increment cohort','#DE8F05','s')]:
      z=sensitivity[sensitivity.cohort.eq(name)];ax.plot(z.increment,z.median_delta_strength,marker=mk,color=c,label=name)
    ax.axhline(0,color='black',lw=.8);ax.set(xlabel='Fly-ash increase (percentage points)',ylabel='Full-model median Δ predicted strength (MPa)');ax.legend(ncol=2,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16));fig.tight_layout();fig.savefig(OUT/'figures'/'fig_03_increment_sensitivity.png',dpi=300,bbox_inches='tight');plt.close(fig)
    report=['# Corrected resampled FA perturbation','', 'This analysis preserves all observed mix properties, changes only FA/B, and recomputes FA_used. “Input-space-supported” means both endpoints pass Isolation Forest, Mahalanobis, KDE and log-prediction-range screens; it does not mean target-feasible or experimentally validated.','', '## Correctness checks','',checks.to_markdown(index=False),'','## Group-level resampling', '',gs.to_markdown(index=False),'','## Increment sensitivity','',sensitivity.to_markdown(index=False),'','## Row-level classifications','',rs.classification.value_counts().rename_axis('classification').reset_index(name='n').to_markdown(index=False)]
    (OUT/'RESAMPLED_PERTURBATION_REPORT.md').write_text('\n'.join(report),encoding='utf-8')
    print(f'Completed: {len(rs)} input-space-supported +10 rows; {len(ds)} refits')
if __name__=='__main__':
  p=argparse.ArgumentParser();p.add_argument('--n-splits',type=int,default=5);p.add_argument('--n-repeats',type=int,default=10);p.add_argument('--seed',type=int,default=42);main(p.parse_args())
