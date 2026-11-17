"""Builds every manuscript table (LaTeX) and the STROBE flow figure directly from the pipeline's aggregate outputs.
Usage: python make_tables_and_flow.py <results/pipeline> <overleaf_project_dir>
Re-run whenever the pipeline is re-run, so the manuscript tables can never drift from the analysis."""
import sys, json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SRC, DST = Path(sys.argv[1]), Path(sys.argv[2])
TAB, FIG = DST / 'tables', DST / 'figures'
TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
J = json.loads((SRC / 'manuscript_numbers.json').read_text())
GL = {'Colonoscopy only': 'Colonoscopy only', 'Endoscopy only': 'Upper endoscopy only', 'Both': 'Combined'}

def c(x):  # thousands separator
    return f'{int(round(x)):,}'
def p1(x):
    return f'{x:.1f}'
def tex_escape(s):
    return str(s).replace('%', r'\%').replace('&', r'\&')

# ---------------- Table 1 ----------------
t1 = pd.read_csv(SRC / 'table1.csv').set_index('Group')
rows = [('Procedures, n', 'Procedures, n'), ('Patients, n', 'Patients, n'),
        ('Age, yr, median (IQR)', 'Age, yr, median (IQR)'), ('Female sex, n (\\%)', 'Female, n (%)'),
        ('BMI, kg/m$^2$, median (IQR)', 'BMI, kg/m2, median (IQR)'),
        ('ASA physical status, n (\\%)', None),
        ('\\quad I', 'ASA I, n (%)'), ('\\quad II', 'ASA II, n (%)'), ('\\quad III', 'ASA III, n (%)'), ('\\quad IV--V', 'ASA IV-V, n (%)'),
        ('Charted propofol infusion, n (\\%)', 'Any charted infusion, n (%)'),
        ('Reconstructed interval, min, median (IQR)', 'Reconstruction interval, min, median (IQR)')]
cols = ['Overall', 'Colonoscopy only', 'Upper endoscopy only', 'Combined']
def fmt_cell(v):
    v = str(v)
    v = v.replace('.0 (', ' (').replace('.0-', '--').replace('.0)', ')') if v.count('.0') >= 2 and 'yr' not in v else v
    return v.replace('-', '--')
lines = [r'\begin{tabular}{lrrrr}', r'\toprule', ' & ' + ' & '.join(cols) + r' \\', r'\midrule']
for lab, key in rows:
    if key is None:
        lines.append(lab + ' & & & & \\\\'); continue
    vals = []
    for g in cols:
        v = str(t1.loc[g, key])
        if 'median (IQR)' in key:
            a, b = v.split(' (')
            lo, hi = b.rstrip(')').split('-')
            f = (lambda s: f'{float(s):.0f}') if key.startswith(('Age', 'Reconstruction')) else (lambda s: f'{float(s):.1f}')
            v = f'{f(a)} ({f(lo)}--{f(hi)})'
        vals.append(v.replace('%', r'\%'))
    lines.append(lab + ' & ' + ' & '.join(vals) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'table1.tex').write_text('\n'.join(lines) + '\n')

# ---------------- Table 2: primary and sensitivity analyses ----------------
sens = pd.read_csv(SRC / 'sensitivity_summary.csv').set_index('analysis')
st = pd.read_csv(SRC / 'tableS1_stop_timing_full_cohort.csv').set_index('scenario')
ov = pd.read_csv(SRC / 'exceedance_overall.csv')
prim = ov.loc[(ov.metric == 'median') & (ov.reference == 'ce50')].iloc[0]
primx = ov.loc[(ov.metric == 'max') & (ov.reference == 'ce50')].iloc[0]
T2 = [('Primary analysis (exact Eleveld solution, primary end rule)', 'Primary', None),
      ('Alternative reconstruction specifications$^a$', None, None),
      ('\\quad Exact Eleveld solution, alternative end rule', 'Previous late-bolus endpoint rule', None),
      ('\\quad PyTCI 1.1 software$^b$, alternative end rule', 'PyTCI 1.1 engine and previous endpoint (v1 values)', None),
      ('Eligibility', None, None),
      ('\\quad Bolus-only cases excluded by the 15-min rule included$^c$', 'Including bolus-only 15-min tail-rule cases', None),
      ('\\quad Age recorded as 90 yr excluded', 'Excluding age recorded as 90 yr', None),
      ('\\quad No anthropometric or procedure-scope exclusions', 'Previous eligibility (v1 case set, exact engine, v2 endpoint)', None),
      ("\\quad First procedure of each patient only", "Each patient's first procedure", None),
      ('\\quad Reconstructions ending at a documented stop only', 'END:terminal_stop', None),
      ('Care setting and resident involvement$^d$', None, None),
      ('\\quad Hospital unit serving more inpatients excluded', 'SET:Excluding the hospital procedure unit serving mainly inpatients', None),
      ('\\quad Both hospital endoscopy units excluded', 'SET:Excluding both hospital-based procedure units', None),
      ('\\quad Procedures with an anesthesia resident excluded', 'SET:Excluding procedures with an anesthesia resident', None),
      ('Reconstruction interval shortened or extended$^e$', None, None),
      ('\\quad Shortened by 2 min', None, 'stop_2min_late'),
      ('\\quad Shortened by 5 min', None, 'stop_5min_late'),
      ('\\quad Extended by 2 min', None, 'stop_2min_early'),
      ('\\quad Extended by 5 min', None, 'stop_5min_early')]
lines = [r'\begin{tabular}{lrrr}', r'\toprule',
         r'Analysis & Procedures, n & Median $C_e>C_{e50}$, \% & Maximum $C_e>C_{e50}$, \% \\', r'\midrule']
_setp = SRC / 'sensitivity_setting.csv'
sset = pd.read_csv(_setp).set_index('analysis') if _setp.exists() else None     # pipeline section 17
if sset is None:
    T2 = [t for i, t in enumerate(T2) if not (str(t[1]).startswith('SET:') or t[0].startswith('Care setting'))]
for lab, key, sk in T2:
    if isinstance(key, str) and key.startswith('END:'):      # by how the reconstruction ended (pipeline section 15)
        r = J['diagnostics']['by_end_reason'][key[4:]]
        lines.append(f'{lab} & {c(r["n"])} & {p1(r["median_gt_ce50"])} & -- \\\\'); continue
    if isinstance(key, str) and key.startswith('SET:'):
        r = sset.loc[key[4:]]
        lines.append(f'{lab} & {c(r.n)} & {p1(r.median_gt_ce50_pct)} & {p1(r.max_gt_ce50_pct)} \\\\'); continue
    if key == 'Primary':
        lines.append(f'{lab} & {c(prim.N)} & {p1(prim.pct)} ({p1(getattr(prim, 'twoway_lo', prim.cluster_lo))}--{p1(getattr(prim, 'twoway_hi', prim.cluster_hi))}) & '
                     f'{p1(primx.pct)} ({p1(getattr(primx, 'twoway_lo', primx.cluster_lo))}--{p1(getattr(primx, 'twoway_hi', primx.cluster_hi))}) \\\\'); continue
    if key is None and sk is None:
        lines.append(f'\\multicolumn{{4}}{{l}}{{\\textit{{{lab}}}}} \\\\'); continue
    if sk is not None:
        r = st.loc[sk]
        lines.append(f'{lab} & {c(r.n)} & {p1(r.scenario_pct)} ({"+" if r.change_pp >= 0 else "$-$"}{abs(r.change_pp):.1f}) & -- \\\\'); continue
    r = sens.loc[key]
    lines.append(f'{lab} & {c(r.n)} & {p1(r.median_gt_ce50_pct)} & {p1(r.max_gt_ce50_pct)} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'table2.tex').write_text('\n'.join(lines) + '\n')

# ---------------- Supplementary tables ----------------
# S1: exceedance against both references for the within-procedure median and maximum
lab_m = {'median': 'Within-procedure median', 'max': 'Within-procedure maximum', 'p25': 'Within-procedure 25th percentile'}
lab_r = {'ce50': '$C_{e50}$ (model BIS $\\approx$ 47)', 'bis60': 'Model BIS 60 concentration'}
lines = [r'\begin{tabular}{llrrr}', r'\toprule', r'Summary & Reference & n above & \% & 95\% CI$^a$ \\', r'\midrule']
for m in ['median', 'max']:
    for rr in ['ce50', 'bis60']:
        r = ov.loc[(ov.metric == m) & (ov.reference == rr)].iloc[0]
        lines.append(f'{lab_m[m]} & {lab_r[rr]} & {c(r.n)} & {p1(r.pct)} & {p1(getattr(r, 'twoway_lo', r.cluster_lo))}--{p1(getattr(r, 'twoway_hi', r.cluster_hi))} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'tableS1_exceedance.tex').write_text('\n'.join(lines) + '\n')

# S2: by age and procedure group
ag = pd.read_csv(SRC / 'exceedance_by_age_group.csv')
lines = [r'\begin{tabular}{lrrrrrrrr}', r'\toprule',
         r' & \multicolumn{2}{c}{All procedures} & \multicolumn{2}{c}{Colonoscopy only} & \multicolumn{2}{c}{Upper endoscopy only} & \multicolumn{2}{c}{Combined} \\',
         r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}',
         r'Age, yr & n & Median $>C_{e50}$, \% & n & Median $>C_{e50}$, \% & n & Median $>C_{e50}$, \% & n & Median $>C_{e50}$, \% \\', r'\midrule']
for a in ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-90']:
    aa = ag.loc[ag.AGE_BIN == a]
    cells = [c(aa.n.sum()), p1((aa.n * aa.median_gt_ce50_pct).sum() / aa.n.sum())]
    for g in ['Colonoscopy only', 'Endoscopy only', 'Both']:
        r = ag.loc[(ag.AGE_BIN == a) & (ag.PROC_GROUP_3 == g)].iloc[0]
        cells += [c(r.n), p1(r.median_gt_ce50_pct)]
    lines.append(a.replace('-', '--') + ' & ' + ' & '.join(cells) + r' \\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'tableS2_by_age_group.tex').write_text('\n'.join(lines) + '\n')

# S3: modal rate anchors
s2 = pd.read_csv(SRC / 'tableS2_modal_rate_anchors.csv')
lines = [r'\begin{tabular}{lrrrrr}', r'\toprule',
         r'Age, yr & n & 100, \% & 150, \% & 200, \% & Any of the three, \% \\', r'\midrule']
for r in s2.itertuples():
    lab = 'All ages' if r.age == 'Total' else r.age.replace('-', '--')
    if r.age == 'Total':
        lines.append(r'\midrule')
    lines.append(f'{lab} & {c(r.n)} & {p1(r.pct_100)} & {p1(r.pct_150)} & {p1(r.pct_200)} & {p1(r.pct_any)} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'tableS3_modal_rate.tex').write_text('\n'.join(lines) + '\n')

# S4: adjunct coefficients
s3 = pd.read_csv(SRC / 'tableS3_adjunct_coefficients_clusterCI.csv')
lines = [r'\begin{tabular}{llr}', r'\toprule', r'Outcome & Procedure group & Adjusted difference, $\mu$g/ml (95\% CI) \\', r'\midrule']
for r in s3.itertuples():
    o = 'Median $C_e$' if r.metric == 'MEDIAN_CE' else 'Maximum $C_e$'
    sg = lambda v: ('$-$' if v < 0 else '') + f'{abs(v):.2f}'
    lo_, hi_ = getattr(r, 'lo_twoway', r.lo), getattr(r, 'hi_twoway', r.hi)
    lines.append(f'{o} & {r.group} & {sg(r.diff)} ({sg(lo_)} to {sg(hi_)}) \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'tableS4_adjunct.tex').write_text('\n'.join(lines) + '\n')

# S5: iTIVA cross-check
g1 = pd.read_csv(SRC / 'G1_exact_vs_itiva.csv')
lines = [r'\begin{tabular}{lrrrr}', r'\toprule', r'Case & Minute & This study, $\mu$g/ml & iTIVA, $\mu$g/ml & Difference, \% \\', r'\midrule']
for r in g1.itertuples():
    lines.append(f'{r.case} & {r.minute} & {r.exact:.3f} & {r.itiva:.3f} & $-${abs(r.rel_pct):.2f} \\\\')
lines += [r'\bottomrule', r'\end{tabular}']
(TAB / 'tableS5_itiva.tex').write_text('\n'.join(lines) + '\n')

# ---------------- Figure 1: compact STROBE flow ----------------
fl = pd.read_csv(SRC / 'cohort_flow.csv').drop_duplicates('step', keep='last').set_index('step')
n = lambda s: int(fl.loc[s, 'n'])
main_boxes = [('Anesthesia cases with a propofol record\n(colonoscopy or upper endoscopy; no intubation or SGA)', n('Cases with a propofol record (extracted)')),
              ('Cases with reconstructed concentrations', n('Reconstructed (complete inputs, positive interval)')),
              ('Cases meeting demographic and record criteria', n('Complete concentration summaries')),
              ('Analytic cohort', n('Gastrointestinal endoscopy only (procedure-scope rule)'))]
side = [[f'Implausible or incomplete dosing history: {n("Cases with a propofol record (extracted)") - n("After dose-plausibility screening (Stage 02)"):,}',
         f'Missing reconstruction inputs or no positive interval: {n("After dose-plausibility screening (Stage 02)") - n("Reconstructed (complete inputs, positive interval)"):,}'],
        [f'Bolus-only, >15 min without propofol before end of record: {n("Reconstructed (complete inputs, positive interval)") - n("After bolus-only 15-min tail rule"):,}',
         f'Age outside 18–90 yr: {n("After bolus-only 15-min tail rule") - n("Age 18-90 yr"):,}',
         f'Missing ASA physical status: {n("Recorded sex") - n("Recorded ASA physical status"):,}',
         f'Missing BMI: {n("Recorded ASA physical status") - n("Recorded BMI"):,}'],
        [f'Implausible weight, height, or BMI: {n("Complete concentration summaries") - n("Plausible weight 35-250 kg, height 130-215 cm, BMI 14-70"):,}',
         f'Nongastrointestinal or surgical procedure component: {n("Plausible weight 35-250 kg, height 130-215 cm, BMI 14-70") - n("Gastrointestinal endoscopy only (procedure-scope rule)"):,}']]
assert n('Recorded sex') == n('Age 18-90 yr') and n('Complete concentration summaries') == n('Recorded BMI')
plt.rcParams.update({'font.size': 8.5, 'font.family': 'DejaVu Sans'})
fig, ax = plt.subplots(figsize=(7.0, 6.2)); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ys = [0.92, 0.64, 0.36, 0.08]
for (txt, k), y in zip(main_boxes, ys):
    ax.text(0.27, y, f'{txt}\nn = {k:,}', ha='center', va='center', fontweight='bold' if txt == 'Analytic cohort' else 'normal',
            bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='0.25', lw=0.9))
for i in range(3):
    ax.annotate('', xy=(0.27, ys[i + 1] + 0.07), xytext=(0.27, ys[i] - 0.07), arrowprops=dict(arrowstyle='-|>', color='0.25', lw=0.9))
    ym = (ys[i] + ys[i + 1]) / 2
    ax.plot([0.27, 0.52], [ym, ym], color='0.25', lw=0.9)
    ax.text(0.53, ym, 'Excluded\n' + '\n'.join(side[i]), ha='left', va='center', fontsize=7.8,
            bbox=dict(boxstyle='round,pad=0.45', fc='0.97', ec='0.45', lw=0.8))
fig.tight_layout()
for ext in ('pdf', 'png'):
    fig.savefig(FIG / f'figure1_strobe_flow.{ext}', dpi=300, bbox_inches='tight')
print('tables and flow figure written')

# ---------------- S6: threshold sensitivity with model-predicted BIS (if produced by the pipeline) ----------------
_p6 = SRC / 'tableS6_threshold_bis.csv'
if _p6.exists():
    t6 = pd.read_csv(_p6)
    lines = [r'\begin{tabular}{lrrr}', r'\toprule',
             r'Reference concentration & Model-predicted BIS & Median $C_e$ above, \% (95\% CI) & Maximum $C_e$ above, \% \\', r'\midrule']
    for r in t6.itertuples():
        lab = f'${r.multiple_of_ce50:.1f}\\times C_{{e50}}$' + (' (primary)' if abs(r.multiple_of_ce50 - 1) < 1e-9 else '')
        lines.append(f'{lab} & {r.model_bis:.1f} & {r.median_gt_pct:.1f} ({r.median_twoway_lo:.1f}--{r.median_twoway_hi:.1f}) & {r.max_gt_pct:.1f} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    (TAB / 'tableS6_threshold.tex').write_text('\n'.join(lines) + '\n')
