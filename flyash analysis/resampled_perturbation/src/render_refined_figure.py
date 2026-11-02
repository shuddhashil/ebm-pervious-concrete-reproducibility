"""Re-render the manuscript candidate without retraining models."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
gs=pd.read_csv(ROOT/'outputs'/'group_level_resampling_summary.csv')
sel=gs[gs.group.isin(['exact 7 days','exact 28 days','B/A T1','B/A T2','B/A T3'])].copy()
palette={'exact 7 days':'#D55E00','exact 28 days':'#029E73','B/A T1':'#029E73','B/A T2':'#D55E00','B/A T3':'#949494'}
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(8.4,4.5))
for i,(_,row) in enumerate(sel.iterrows()):
    colour=palette[row['group']]
    ax.errorbar(row.median_of_fit_medians,i,xerr=[[row.median_of_fit_medians-row.fit_median_p025],[row.fit_median_p975-row.median_of_fit_medians]],fmt='o',capsize=4,color=colour,markersize=8,elinewidth=2)
    ax.annotate(f"n={int(row.n_candidates)}",(row.fit_median_p975,i),xytext=(7,0),textcoords='offset points',va='center',fontsize=9,color='#444444')
ax.axvline(0,color='black',lw=1)
ax.set(yticks=np.arange(len(sel)),yticklabels=sel.group,xlabel='Median Δ predicted strength across 50 refits (MPa)')
ax.margins(x=.12);ax.grid(axis='x',color='#E6E6E6')
fig.tight_layout();fig.savefig(ROOT/'figures'/'fig_02_group_median_resampling_intervals.png',dpi=300,bbox_inches='tight')
