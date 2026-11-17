"""Redraw Figure 2 from results/stage25/figure2_predictions.csv (aggregate predictions) with the legend below the panels.
Usage: python make_figure2.py <stage25_dir> <project_dir>"""
import sys
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
SRC, PRJ = Path(sys.argv[1]), Path(sys.argv[2])
p = pd.read_csv(SRC / 'figure2_predictions.csv')
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
groups = list(dict.fromkeys(p.group))
COL = dict(zip(groups, ['#1f77b4', '#ff7f0e', '#2ca02c']))
BIS60 = (93 / 60 - 1) ** (1 / 1.89)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6))
for ax, (yv, lab) in zip(axes, [('MEDIAN_CE', 'A  Within-procedure median'), ('MAX_CE', 'B  Within-procedure maximum')]):
    for g in groups:
        s = p[(p.outcome == yv) & (p.group == g)]
        ax.plot(s.AGE, s.fit, color=COL[g], lw=1.6, label=g)
        lo_, hi_ = (s.lo_twoway, s.hi_twoway) if 'lo_twoway' in s else (s.lo, s.hi)
        ax.fill_between(s.AGE, lo_, hi_, color=COL[g], alpha=.18, lw=0)
    a = np.linspace(18, 90, 200); c50 = 3.08 * np.exp(-0.00635 * (a - 35))
    ax.plot(a, c50, 'k--', lw=1.2, label='$C_{e50}$ (model BIS ≈ 47)')
    ax.plot(a, BIS60 * c50, ':', color='0.45', lw=1.2, label='Model BIS 60')
    ax.set_xlabel('Age (yr)'); ax.set_ylabel('Effect-site concentration (µg/ml)'); ax.set_xlim(18, 90)
    ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontweight='bold')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ('pdf', 'png'):
    fig.savefig(PRJ / 'figures' / f'figure2_adjusted_ce_by_age.{ext}', dpi=300, bbox_inches='tight')
print('figure 2 written')
