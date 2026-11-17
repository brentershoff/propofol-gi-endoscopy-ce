# %% [markdown]
# # Propofol effect-site concentrations during GI endoscopy: complete analysis pipeline
#
# **One notebook, raw extract to every manuscript number.** It contains the complete analysis: event preparation, cohort construction, exact reconstruction, all
# analyses, and validation.
#
# **Inputs** (restricted; stay in your Drive, never in the code repository):
# * `inputs/propall.pkl`: SQL extract of propofol administrations (anesthesia cases whose title contains
#   "colonoscopy" or "endoscopy", with no documented airway event (the event that records tracheal intubation or
#   laryngeal mask placement), and ERCP, EUS and enteroscopy excluded at extraction).
# * `inputs/endo_selected_meds_raw.pkl`: medication extract for the adjunct analysis (path can be set with
#   `PROP_GI_MEDS_RAW_PATH`).
# * `results/private_procedure_review/procedure_title_overrides_PRIVATE.csv`: the author's procedure-title adjudications.
# * Optional, only for the PyTCI comparison: `checkpoints/summary_custom_end_v1_full.pkl` (the saved PyTCI 1.1 run).
#
# **Outputs:** aggregate results in `results/pipeline/` (tables, figures, `manuscript_numbers.json`, validation files);
# restricted case-level checkpoints in `checkpoints/pipeline_*` (never share). Nothing from earlier stages is modified.
#
# **Reproduction checks:** when the earlier checkpoints and results exist in the project folder, the notebook compares
# every intermediate step with them (event file, 15-min-rule cases, case metadata, adjunct flags, the 209,406 cohort) and,
# at the end, compares every manuscript number with `results/stage25/`. It stops if anything differs.
#
# **Confidence intervals:** reported both clustered by patient and clustered two-way by patient and responsible provider
# (first CRNA, or the attending when no CRNA is recorded). The two-way intervals are the primary intervals in the manuscript.
#
# **Care setting and residents:** the primary cohort includes every location. Section 17 repeats the primary outcome
# without the hospital-based procedure units and without procedures that had an anesthesia resident (unit names are
# settings: `PROP_GI_INPATIENT_UNITS`, `PROP_GI_HOSPITAL_UNITS`; outputs contain categories and counts only).
#
# **PyTCI:** by default the slow PyTCI 1.1 reconstruction is **not** rerun. Instead, 300 randomly chosen procedures are
# recomputed with PyTCI through this notebook's event parser and must match the saved run to 1e-8 µg/ml (gate G4); the saved
# run then supplies the single PyTCI row of Table 2. Set `RUN_PYTCI_FULL = True` to recompute every procedure (hours).
#
# Runtime on Colab: roughly 30–60 min, most of it the full-cohort reconstruction (resumable in batches).

# %%
# --- Colab setup -----------------------------------------------------------------------------------------
try:
    from google.colab import drive  # noqa: F401
    drive.mount('/content/drive')
    IN_COLAB = True
except Exception:
    IN_COLAB = False
import subprocess, sys, importlib.util
if importlib.util.find_spec('PyTCI') is None:
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'PyTCI==1.1'])
    except Exception as e:
        print('PyTCI not installed; the legacy-reproduction gate (G4) will be skipped:', e)

# %%
import os, json, time, math, hashlib, pickle, re, warnings, platform, importlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.stats import norm
from scipy.interpolate import BSpline
import matplotlib
import matplotlib.pyplot as plt

ROOT = Path(os.environ.get('PROP_GI_ROOT',
    '/content/drive/MyDrive/Research/Propofol_GI_Reproduction/Endoscopy_Propofol_Colab_Reproduction/Propofol_GI_Reproduction'))
IN = ROOT / 'inputs'
CHECK = ROOT / 'checkpoints'
OUT = ROOT / 'results' / 'pipeline'
FIG = OUT / 'figures'
PRIVATE = ROOT / 'results' / 'private_procedure_review'
BATCH_DIR = CHECK / 'pipeline_exact_batches'
for d in (OUT, FIG, PRIVATE, BATCH_DIR):
    d.mkdir(parents=True, exist_ok=True)

STRICT = os.environ.get('PROP_GI_STRICT', '1') == '1'   # asserts the analyzed counts (222,936 / 222,745 / 214,625 / 209,406)
VERSION = 'pipeline_v1_2026-11-12 (analysis code = stage25_exact_eleveld_v1_2026-11-10)'
BATCH_SIZE = 20000
RUN_PYTCI_FULL = os.environ.get('PROP_GI_RUN_PYTCI_FULL', '0') == '1'   # set True to rerun PyTCI 1.1 for every procedure (hours)
SEED = 20261110
BOLUS_TAIL_S = 300
GHOST_GAP_MIN = 15
PLAUSIBLE = dict(weight=(35, 250), height=(130, 215), bmi=(14, 70))
OMEGA2_CE50 = 0.242          # Eleveld 2018 interindividual variance of log Ce50
BIS60_FACTOR = (93 / 60 - 1) ** (1 / 1.89)   # = 0.728829...
MIN_CELL = 11                # suppress aggregate cells below this size
GROUP_LABEL = {'Colonoscopy only': 'Colonoscopy only', 'Endoscopy only': 'Upper endoscopy only', 'Both': 'Combined'}
GROUPS = ['Colonoscopy only', 'Endoscopy only', 'Both']
AGE_BINS = [18, 30, 40, 50, 60, 70, 80, 91]
AGE_LABELS = ['18-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-90']
NUMBERS = {}   # every manuscript number lands here -> manuscript_numbers.json
print('Root:', ROOT, '| strict:', STRICT, '| version:', VERSION)

# %% [markdown]
# ## A. Raw extract to cleaned event stream
# Event interpretation follows the original preparation code. The per-case screening flags are computed vectorized; procedures in
# which two rows share a timestamp are computed with the original per-case function so that tie handling is identical.

# %%
tA = time.time()
raw = pd.read_pickle(IN / 'propall.pkl')
N_EXTRACTED = int(raw.OR_CASE_ID.nunique()); N_EXTRACTED_ROWS = int(len(raw))
PATIENT_KEY = raw[['OR_CASE_ID', 'PatientKey']].dropna().drop_duplicates('OR_CASE_ID')
# ---- responsible provider per procedure: first CRNA if recorded, otherwise first attending (anonymous codes) ----
_pc = [c for c in ['FIRST_CRNA_NAME', 'FIRST_ATT_NAME'] if c in raw.columns]
PROVIDERS = None
if _pc:
    _r = raw[['OR_CASE_ID', 'TAKEN_TIME'] + _pc].copy()
    _r['TAKEN_TIME'] = pd.to_datetime(_r.TAKEN_TIME, errors='coerce')
    for c in _pc:
        _s = _r[c].astype('string').str.strip()
        _r[c] = _s.mask(_s.isna() | _s.str.upper().isin(['', 'NAN', 'NONE', 'NULL']))
    _fst = _r.sort_values(['OR_CASE_ID', 'TAKEN_TIME'], kind='stable').groupby('OR_CASE_ID')[_pc].first()
    _crna = _fst['FIRST_CRNA_NAME'] if 'FIRST_CRNA_NAME' in _fst else pd.Series(pd.NA, index=_fst.index, dtype='string')
    _att = _fst['FIRST_ATT_NAME'] if 'FIRST_ATT_NAME' in _fst else pd.Series(pd.NA, index=_fst.index, dtype='string')
    _ptype = np.where(_crna.notna(), 'CRNA', np.where(_att.notna(), 'Attending only', 'None recorded'))
    _code = pd.factorize(pd.Series(_ptype, index=_fst.index) + '|' + _crna.fillna(_att).astype(str))[0]
    PROVIDERS = pd.DataFrame({'OR_CASE_ID': _fst.index, 'PTYPE': _ptype, 'PROVIDER': _code})
    PROVIDERS.loc[PROVIDERS.PTYPE == 'None recorded', 'PROVIDER'] = -1
    del _r, _fst
    print('Provider fields found:', _pc)
else:
    print('No provider fields in the extract: provider clustering will be skipped')
# ---- care location and anesthesia-resident minutes per procedure (used only by the section 17 sensitivity analyses) ----
_sc = [c for c in ['UCLA_LOC', 'ANES_RESIDENT_MINUTES'] if c in raw.columns]
SETTING = raw.groupby('OR_CASE_ID', sort=False)[_sc].first().reset_index() if _sc else None
print('Setting fields found:', _sc if _sc else 'none (section 17 will be skipped)')

print(f'Extract: {N_EXTRACTED_ROWS:,} rows, {N_EXTRACTED:,} cases')
if STRICT:
    assert N_EXTRACTED == 222_936 and N_EXTRACTED_ROWS == 1_145_055, 'Extract differs from the analyzed extract'

# ---- event interpretation (original preparation code) ----
propsig = raw.copy()
del raw
propsig['TAKEN_TIME'] = pd.to_datetime(propsig['TAKEN_TIME'], errors='coerce')
propsig['ANES_START'] = pd.to_datetime(propsig['ANES_START'], errors='coerce')
propsig['ANES_STOP'] = pd.to_datetime(propsig['ANES_STOP'], errors='coerce')
propsig['SIG'] = pd.to_numeric(propsig['SIG'], errors='coerce')
propsig['UNITS'] = propsig['UNITS'].fillna('').astype(str).str.upper().str.strip()
for col in ['AGE', 'ASA_SCORE', 'WEIGHT_KG', 'HEIGHT_IN', 'BMI']:
    if col in propsig.columns:
        propsig[col] = pd.to_numeric(propsig[col], errors='coerce')
propsig['SEX'] = propsig['SEX'].astype(str).str.upper().str[0]
if 'HEIGHT_CM' not in propsig.columns and 'HEIGHT_IN' in propsig.columns:
    propsig['HEIGHT_CM'] = propsig['HEIGHT_IN'] * 2.54
propsig = propsig.sort_values(['OR_CASE_ID', 'TAKEN_TIME']).reset_index(drop=True)
propsig['bolus_mg'] = np.where((propsig['UNITS'] == 'MG') & (propsig['SIG'] > 0), propsig['SIG'], 0.0)
propsig['infusion_rate_mcg_kg_min'] = np.where((propsig['UNITS'] == 'MCG/KG/MIN') & (propsig['SIG'] > 0), propsig['SIG'], 0.0)
propsig['is_stop_event'] = np.where((propsig['UNITS'] == 'MCG/KG/MIN') & (propsig['SIG'] == 0), True, False)
propsig['invalid_bolus_row'] = ((propsig['UNITS'] == 'MG') & (propsig['SIG'] > 500)).astype(int)
propsig['invalid_infusion_row'] = ((propsig['UNITS'] == 'MCG/KG/MIN') & (propsig['SIG'] > 500)).astype(int)
propsig['nonzero_prop_row'] = (((propsig['UNITS'] == 'MG') | (propsig['UNITS'] == 'MCG/KG/MIN')) & (propsig['SIG'] > 0)).astype(int)

# ---- original per-case helper (used for procedures with same-timestamp rows) ----
def compute_total_infusion_seconds(case_df):
    df = case_df.sort_values('TAKEN_TIME').copy()
    if df.empty:
        return 0.0
    anes_stop = df['ANES_STOP'].dropna().iloc[0] if df['ANES_STOP'].notna().any() else pd.NaT
    total_sec = 0.0
    times = df['TAKEN_TIME'].tolist()
    rates = df['infusion_rate_mcg_kg_min'].fillna(0).tolist()
    for i in range(len(df)):
        current_t = times[i]; current_rate = rates[i]
        if pd.isna(current_t):
            continue
        next_t = times[i + 1] if i < len(df) - 1 else anes_stop
        if pd.isna(next_t):
            continue
        dt = (next_t - current_t).total_seconds()
        if dt < 0:
            continue
        if current_rate > 0:
            total_sec += dt
    return float(total_sec)

# ---- vectorized per-case flags (same rules as the original per-case function) ----
p_ = propsig
cid_ = p_.OR_CASE_ID.to_numpy()
g_ = p_.groupby('OR_CASE_ID', sort=True)
cf = pd.DataFrame({'total_bolus_mg': g_.bolus_mg.sum(), 'max_bolus_mg': g_.bolus_mg.max(),
                   'max_infusion_rate': g_.infusion_rate_mcg_kg_min.max()})
cf['any_bolus'] = (cf.max_bolus_mg > 0).astype(int)
cf['any_infusion'] = (cf.max_infusion_rate > 0).astype(int)
# total infusion seconds: each row's rate runs to the next row (last row: first nonmissing ANES_STOP)
same_next = np.r_[cid_[1:] == cid_[:-1], False]
anes_first = p_.OR_CASE_ID.map(g_.ANES_STOP.first())           # first nonmissing ANES_STOP (GroupBy.first skips NaT)
tt_ = p_.TAKEN_TIME
nxt = tt_.shift(-1).where(same_next, anes_first)
dt_ = (nxt - tt_).dt.total_seconds()
contrib = np.where(tt_.notna() & nxt.notna() & (dt_ >= 0) & (p_.infusion_rate_mcg_kg_min > 0), dt_, 0.0)
cf['total_infusion_sec'] = pd.Series(contrib).groupby(cid_).sum()
dup_time = p_.duplicated(['OR_CASE_ID', 'TAKEN_TIME'], keep=False)
tie_cases = set(p_.loc[dup_time, 'OR_CASE_ID'])
if tie_cases:
    exact = p_.loc[p_.OR_CASE_ID.isin(tie_cases)].groupby('OR_CASE_ID').apply(compute_total_infusion_seconds)
    cf.loc[exact.index, 'total_infusion_sec'] = exact.to_numpy()
print(f'Procedures with same-timestamp rows (original function used): {len(tie_cases):,}')
# terminal stop: first stop entry with no later nonzero propofol entry
nz = (p_.bolus_mg > 0) | (p_.infusion_rate_mcg_kg_min > 0)
last_nz = p_.loc[nz & tt_.notna()].groupby('OR_CASE_ID').TAKEN_TIME.max()
stops = p_.loc[p_.is_stop_event & tt_.notna(), ['OR_CASE_ID', 'TAKEN_TIME']].copy()
stops['LAST_NZ'] = stops.OR_CASE_ID.map(last_nz)
stops = stops.loc[stops.LAST_NZ.isna() | (stops.TAKEN_TIME >= stops.LAST_NZ)]
cf['terminal_stop_time'] = stops.groupby('OR_CASE_ID').TAKEN_TIME.min().reindex(cf.index)
cf['terminal_stop_time'] = pd.to_datetime(cf['terminal_stop_time'])
cf['invalid_reason_tiny_bolus'] = ((cf.any_infusion == 0) & (cf.total_bolus_mg <= 10)).astype(int)
cf['invalid_reason_infusion_only_lt5min'] = ((cf.any_bolus == 0) & (cf.any_infusion == 1) & (cf.total_infusion_sec < 300)).astype(int)
cf['invalid_case'] = (((cf.any_bolus == 0) & (cf.any_infusion == 0)) | (cf.max_bolus_mg > 500) | (cf.max_infusion_rate > 500)
                      | (cf.invalid_reason_tiny_bolus == 1) | (cf.invalid_reason_infusion_only_lt5min == 1)).astype(int)
case_flags = cf.reset_index().rename(columns={'index': 'OR_CASE_ID'})
NUMBERS['screen_excluded'] = dict(
    no_nonzero=int(((cf.any_bolus == 0) & (cf.any_infusion == 0)).sum()), bolus_gt500=int((cf.max_bolus_mg > 500).sum()),
    rate_gt500=int((cf.max_infusion_rate > 500).sum()), bolus_only_le10mg=int(cf.invalid_reason_tiny_bolus.sum()),
    infusion_only_lt5min=int(cf.invalid_reason_infusion_only_lt5min.sum()), total=int(cf.invalid_case.sum()))

# ---- remove screened-out cases ----
invalid_case_ids = case_flags.loc[case_flags['invalid_case'] == 1, 'OR_CASE_ID'].tolist()
propsig_clean = propsig[~propsig['OR_CASE_ID'].isin(invalid_case_ids)].copy()
propsig_clean = propsig_clean.merge(case_flags[['OR_CASE_ID', 'terminal_stop_time']], on='OR_CASE_ID', how='left')
propsig_clean = propsig_clean.sort_values(['OR_CASE_ID', 'TAKEN_TIME']).reset_index(drop=True)
print(f'After dose-plausibility screening: {propsig_clean.OR_CASE_ID.nunique():,} cases '
      f'({NUMBERS["screen_excluded"]["total"]:,} excluded) | {time.time() - tA:.0f}s')
if STRICT:
    assert propsig_clean.OR_CASE_ID.nunique() == 222_745

# ---- reproduction check A: identical to the saved event file from the earlier analysis ----
def col_equal(a, b):
    """Value equality that ignores dtype differences between pandas versions (NaN/NaT equal to NaN/NaT)."""
    a, b = a.reset_index(drop=True), b.reset_index(drop=True)
    if len(a) != len(b):
        return False
    if a.equals(b):
        return True
    na, nb = a.isna().to_numpy(), b.isna().to_numpy()
    if not (na == nb).all():
        return False
    a, b = a[~na], b[~nb]
    for conv in (lambda s: pd.to_datetime(s).astype('datetime64[ns]').astype('int64'), lambda s: pd.to_numeric(s)):
        try:
            return bool((conv(a).to_numpy() == conv(b).to_numpy()).all())
        except Exception:
            pass
    return bool((a.astype(str).to_numpy() == b.astype(str).to_numpy()).all())
REPRO = {}
ref_p = CHECK / 'propsig_clean_v1_full.pkl'
if ref_p.exists():
    ref = pd.read_pickle(ref_p)
    cols_chk = ['OR_CASE_ID', 'TAKEN_TIME', 'UNITS', 'SIG', 'bolus_mg', 'infusion_rate_mcg_kg_min', 'ANES_START', 'ANES_STOP',
                'terminal_stop_time', 'AGE', 'SEX', 'WEIGHT_KG', 'HEIGHT_CM', 'ASA_SCORE', 'BMI', 'IM_PROC_NAME']
    cols_chk = [c for c in cols_chk if c in ref.columns]
    same_shape = ref.shape[0] == propsig_clean.shape[0]
    bad = [c for c in cols_chk if not (same_shape and col_equal(ref[c], propsig_clean[c]))]
    REPRO['A_event_file'] = 'identical' if same_shape and not bad else f'DIFFERS in {bad}'
    print('Check A (event file vs earlier analysis):', REPRO['A_event_file'])
    assert same_shape and not bad, 'Event file differs from the earlier analysis: STOP'
    del ref
else:
    print('Check A skipped (no saved event file from the earlier analysis)')

# %% [markdown]
# ## B. Case metadata and the bolus-only 15-min rule 

# %%
# Case metadata: first row in time order (as in the earlier analysis)
meta_cols = ['OR_CASE_ID', 'AGE', 'SEX', 'ASA_SCORE', 'BMI', 'IM_PROC_NAME', 'ANES_START', 'ANES_STOP']
_o = propsig_clean.sort_values(['OR_CASE_ID', 'TAKEN_TIME'], kind='stable')
META = _o.drop_duplicates('OR_CASE_ID')[[c for c in meta_cols if c in _o.columns]].reset_index(drop=True)
del _o
# Bolus-only 15-min rule: bolus-only, last propofol dose >15 min before the later of the last record and ANES_STOP
p_ = propsig
act_ = p_.SIG > 0
has_inf = p_.loc[(p_.UNITS == 'MCG/KG/MIN') & act_].OR_CASE_ID.unique()
last_dose = p_.loc[act_].groupby('OR_CASE_ID').TAKEN_TIME.max()
last_rec = p_.groupby('OR_CASE_ID').TAKEN_TIME.max()
first_stop = p_.drop_duplicates('OR_CASE_ID').set_index('OR_CASE_ID').ANES_STOP   # ANES_STOP of the first row
gh = pd.DataFrame({'last_dose': last_dose}).join(last_rec.rename('last_rec')).join(first_stop.rename('anes'))
gh = gh.loc[~gh.index.isin(has_inf)]
end_ = gh[['last_rec', 'anes']].max(axis=1)
gap_min = (end_ - gh.last_dose).dt.total_seconds() / 60
ghost_ids = set(gh.index[(gap_min > GHOST_GAP_MIN).fillna(False).to_numpy()])
print(f'Bolus-only cases with >15 min after the last dose: {len(ghost_ids):,}')
del propsig, p_

ref_g = CHECK / 'ghost_case_ids.pkl'
if ref_g.exists():
    with open(ref_g, 'rb') as fh:
        ref_ids = set(pickle.load(fh))
    REPRO['B_ghost_ids'] = 'identical' if ref_ids == ghost_ids else f'DIFFERS ({len(ref_ids ^ ghost_ids)} cases)'
    print('Check B1 (15-min rule vs earlier analysis):', REPRO['B_ghost_ids'])
    assert ref_ids == ghost_ids, '15-min rule cases differ from the earlier analysis: STOP'

# %% [markdown]
# ## C. Selected adjunct flags from the medication extract (classification as in the earlier analysis)

# %%
MEDS_PATH = Path(os.environ.get('PROP_GI_MEDS_RAW_PATH', str(IN / 'endo_selected_meds_raw.pkl')))
DRUG_FLAGS = None
if MEDS_PATH.exists():
    df_meds = pd.read_pickle(MEDS_PATH)
    meds = df_meds.loc[df_meds['DISPLAY_NAME'].notna()].copy()
    meds['DISPLAY_NAME_UPPER'] = meds['DISPLAY_NAME'].astype(str).str.upper().str.strip()
    meds['UNITS_UPPER'] = meds['UNITS'].astype(str).str.upper().str.strip()
    drug_patterns = [('propofol', r'\bPROPOFOL\b'), ('fentanyl', r'\bFENTANYL\b'), ('midazolam', r'\bMIDAZOLAM\b|\bVERSED\b'),
                     ('ketamine', r'\bKETAMINE\b'), ('remifentanil', r'\bREMIFENTANIL\b'), ('etomidate', r'\bETOMIDATE\b')]
    meds['MED_GROUP'] = 'other'
    for group, pattern in drug_patterns:
        mask = (meds['MED_GROUP'] == 'other') & meds['DISPLAY_NAME_UPPER'].str.contains(pattern, regex=True, na=False)
        meds.loc[mask, 'MED_GROUP'] = group
    meds.loc[meds['DISPLAY_NAME_UPPER'].str.contains('FENTANYL', na=False), 'MED_GROUP'] = 'fentanyl'
    epi = meds['DISPLAY_NAME_UPPER'].str.contains('BUPIVACAINE', na=False) & meds['DISPLAY_NAME_UPPER'].str.contains('FENTANYL', na=False)
    meds = meds.loc[~epi].copy()
    selected_groups = ['propofol', 'fentanyl', 'midazolam', 'ketamine', 'remifentanil', 'etomidate']
    all_cases = df_meds[['OR_CASE_ID']].drop_duplicates().copy()
    drug_flags = pd.crosstab(meds['OR_CASE_ID'], meds['MED_GROUP']).gt(0).astype(int).reset_index()
    drug_flags = all_cases.merge(drug_flags, on='OR_CASE_ID', how='left')
    for drug in selected_groups:
        if drug not in drug_flags.columns:
            drug_flags[drug] = 0
    drug_flags[selected_groups] = drug_flags[selected_groups].fillna(0).astype(int)
    drug_flags = drug_flags.rename(columns={d: d.upper() + '_ANY' for d in selected_groups})
    adj_cols = ['FENTANYL_ANY', 'MIDAZOLAM_ANY', 'KETAMINE_ANY', 'REMIFENTANIL_ANY', 'ETOMIDATE_ANY']
    drug_flags['ANY_SELECTED_ADJUNCT'] = drug_flags[adj_cols].max(axis=1)
    drug_flags['PROPOFOL_ONLY_SELECTED'] = ((drug_flags['PROPOFOL_ANY'] == 1) & (drug_flags['ANY_SELECTED_ADJUNCT'] == 0)).astype(int)
    DRUG_FLAGS = drug_flags
    del df_meds, meds
    print(f'Adjunct flags built for {len(DRUG_FLAGS):,} cases')
    ref_f = CHECK / 'case_level_selected_drug_flags.pkl'
    if ref_f.exists():
        rf = pd.read_pickle(ref_f)
        cmp_ = DRUG_FLAGS[['OR_CASE_ID', 'ANY_SELECTED_ADJUNCT', 'PROPOFOL_ONLY_SELECTED']].merge(
            rf[['OR_CASE_ID', 'ANY_SELECTED_ADJUNCT', 'PROPOFOL_ONLY_SELECTED']], on='OR_CASE_ID', how='outer', suffixes=('', '_ref'))
        ok_ = (len(rf) == len(DRUG_FLAGS) and (cmp_.ANY_SELECTED_ADJUNCT == cmp_.ANY_SELECTED_ADJUNCT_ref).all()
               and (cmp_.PROPOFOL_ONLY_SELECTED == cmp_.PROPOFOL_ONLY_SELECTED_ref).all())
        REPRO['C_adjunct_flags'] = 'identical' if ok_ else 'DIFFERS'
        print('Check C (adjunct flags vs earlier analysis):', REPRO['C_adjunct_flags'])
        assert ok_, 'Adjunct flags differ from the earlier analysis: STOP'
else:
    ref_f = CHECK / 'case_level_selected_drug_flags.pkl'
    if ref_f.exists():
        DRUG_FLAGS = pd.read_pickle(ref_f)
        print('Medication extract not found; using the saved adjunct flags from the earlier analysis')
    else:
        print('No medication extract or saved flags: the adjunct analysis will be skipped')

# %% [markdown]
# ## 1. Exact Eleveld engine
# Amount-based 3-compartment model + effect site; zero-order infusion carried as a 5th state with zero derivative.
# Between events the solution is a sum of exponentials, evaluated exactly for every second (eigendecomposition;
# falls back to repeated 1-s matrix exponentials if ill-conditioned).

# %%
def _janmahasatian(h_cm, w, sex):
    bmi = w / (h_cm / 100.0) ** 2
    return 9270.0 * w / (6680.0 + 216.0 * bmi) if sex == 'm' else 9270.0 * w / (8780.0 + 244.0 * bmi)

def _alsallami(age, h_cm, w, sex):
    f = (0.88 + 0.12 / (1 + (age / 13.4) ** -12.7)) if sex == 'm' else (1.11 - 0.11 / (1 + (age / 7.1) ** -1.1))
    return f * _janmahasatian(h_cm, w, sex)

def eleveld_params(age, w, h_cm, sex):
    """Published Eleveld 2018 PK (arterial, no opioids) and ke0. Units: L, L/min, 1/min."""
    sig = lambda x, e50, g: x ** g / (x ** g + e50 ** g)
    pma, pmaref = age * 52.0 + 40.0, 35 * 52.0 + 40.0
    v1 = 6.28 * sig(w, 33.6, 1) / sig(70.0, 33.6, 1)
    v2 = 25.5 * (w / 70.0) * math.exp(-0.0156 * (age - 35))
    v3 = 273.0 * _alsallami(age, h_cm, w, sex) / _alsallami(35, 170, 70, 'm')
    cl = (1.79 if sex == 'm' else 2.10) * (w / 70.0) ** 0.75 * sig(pma, 42.3, 9.06) / sig(pmaref, 42.3, 9.06)
    q3m, q3mref = sig(pma, 68.3, 1), sig(pmaref, 68.3, 1)
    q2 = 1.75 * (v2 / 25.5) ** 0.75 * (1 + 1.30 * (1 - q3m))
    q3 = 1.11 * (v3 / 273.0) ** 0.75 * q3m / q3mref
    ke0 = 0.146 * (w / 70.0) ** -0.25
    return v1, v2, v3, cl, q2, q3, ke0

class ExactEleveld:
    def __init__(self, age, w, h_cm, sex):
        v1, v2, v3, cl, q2, q3, ke0 = eleveld_params(age, w, h_cm, sex)
        k10, k12, k13, k21, k31 = cl / v1, q2 / v1, q3 / v1, q2 / v2, q3 / v3
        M = np.zeros((5, 5))
        M[:4, :4] = np.array([[-(k10 + k12 + k13), k21, k31, 0.0],
                              [k12, -k21, 0.0, 0.0],
                              [k13, 0.0, -k31, 0.0],
                              [ke0 / v1, 0.0, 0.0, -ke0]]) / 60.0
        M[0, 4] = 1.0
        if not np.all(np.isfinite(M)):
            raise ValueError('non-finite rate constants')
        self.M = M
        lam, V = np.linalg.eig(M)
        self.use_eig = (np.max(np.abs(lam.imag)) < 1e-12 and np.max(np.abs(V.imag)) < 1e-12
                        and np.linalg.cond(V.real) < 1e8)
        if self.use_eig:
            self.lam, self.V = lam.real, V.real
            self.Vinv = np.linalg.inv(self.V)
        else:
            self.E1 = expm(M)

    def propagate(self, x, n):
        """Advance state x by n seconds. Returns (Ce at seconds 1..n, new state)."""
        if n <= 0:
            return np.empty(0), x
        if self.use_eig:
            c = self.Vinv @ x
            k = np.arange(1, n + 1, dtype=float)
            ex = np.exp(np.outer(k, self.lam))
            ce = ex @ (self.V[3] * c)
            return ce, self.V @ (ex[-1] * c)
        ce = np.empty(n)
        for i in range(n):
            x = self.E1 @ x
            ce[i] = x[3]
        return ce, x

    def propagate_loop(self, x, n):
        E1 = expm(self.M)
        ce = np.empty(n)
        for i in range(n):
            x = E1 @ x
            ce[i] = x[3]
        return ce, x

def simulate_exact(model, rel_sec, kind, val, T, weight, post_tail=0, loop=False):
    """rel_sec sorted ints in [0, T]; kind 1 = bolus mg, 2 = rate ug/kg/min. Same semantics as the earlier PyTCI-based reconstruction
    (bolus added at its second; a rate entry sets a zero-order infusion that runs from that second on)."""
    x = np.zeros(5)
    ce = np.empty(T + 1)
    prop = model.propagate_loop if loop else model.propagate
    def apply(lo, hi):
        for j in range(lo, hi):
            if kind[j] == 1:
                x[0] += val[j]
            elif kind[j] == 2:
                x[4] = val[j] * weight / 60000.0
    n_ev = len(rel_sec)
    j = 0
    while j < n_ev and rel_sec[j] == 0:
        j += 1
    apply(0, j)
    ce[0] = x[3]
    cur = 0
    while j < n_ev:
        s = rel_sec[j]
        k = j
        while k < n_ev and rel_sec[k] == s:
            k += 1
        seg, xn = prop(x, s - cur)
        ce[cur + 1:s + 1] = seg
        x[:] = xn
        apply(j, k)
        cur, j = s, k
    if T > cur:
        seg, xn = prop(x, T - cur)
        ce[cur + 1:] = seg
        x[:] = xn
    post = np.empty(0)
    if post_tail > 0:
        x[4] = 0.0
        post, _ = prop(x, post_tail)
    return ce, post

# %% [markdown]
# ### Gate G1: exact engine vs iTIVA (two synthetic histories, values from your validated workbook)

# %%
def _synthetic(events):
    ev = sorted(events, key=lambda e: e[0])
    return (np.array([e[0] for e in ev]), np.array([1 if e[1] == 'MG' else 2 for e in ev]),
            np.array([float(e[2]) for e in ev]))
SYN = {
    'A': dict(age=45, w=75, h=175, sex='m', T=1080,
              ev=[(0, 'MG', 50), (0, 'R', 150), (480, 'R', 100), (720, 'MG', 25), (1080, 'R', 0)],
              itiva={1: .83, 3: 1.575, 5: 1.921, 8: 2.249, 10: 2.319, 12: 2.308, 15: 2.812, 18: 2.716}),
    'B': dict(age=75, w=60, h=160, sex='f', T=1260,
              ev=[(0, 'MG', 30), (60, 'R', 175), (300, 'R', 0), (420, 'MG', 20), (540, 'R', 100), (900, 'R', 0), (960, 'MG', 10)],
              itiva={1: .526, 3: 1.124, 5: 1.603, 7: 1.658, 9: 1.913, 12: 1.876, 15: 1.923, 16: 1.882, 18: 1.879, 21: 1.475}),
}
g1 = []
for name, c in SYN.items():
    m = ExactEleveld(c['age'], c['w'], c['h'], c['sex'])
    r, k, v = _synthetic(c['ev'])
    ce, _ = simulate_exact(m, r, k, v, c['T'], c['w'])
    ce_loop, _ = simulate_exact(m, r, k, v, c['T'], c['w'], loop=True)
    for mnt, iv in c['itiva'].items():
        g1.append(dict(case=name, minute=mnt, exact=ce[60 * mnt], itiva=iv, rel_pct=100 * (ce[60 * mnt] - iv) / iv))
    assert np.max(np.abs(ce - ce_loop)) < 1e-10, 'eigen vs loop mismatch on synthetic case'
g1 = pd.DataFrame(g1)
print(g1.round(4).to_string(index=False))
assert g1.rel_pct.abs().max() < 1.0, 'Exact engine differs from iTIVA by >1%: STOP'
NUMBERS['itiva_max_abs_rel_pct_exact'] = round(float(g1.rel_pct.abs().max()), 2)
g1.to_csv(OUT / 'G1_exact_vs_itiva.csv', index=False)
print('G1 PASS: exact engine within', NUMBERS['itiva_max_abs_rel_pct_exact'], '% of iTIVA at all sampled times')


# %% [markdown]
# ## 2. Event arrays; optional saved PyTCI run (only for gates G2-G4 and the PyTCI row of Table 2)

# %%
t0 = time.time()
ev = propsig_clean
del propsig_clean
_sp, _v1p = CHECK / 'summary_custom_end_v1_full.pkl', CHECK / 'primary_modeling_cohort_209406.pkl'
stored = pd.read_pickle(_sp) if _sp.exists() else None          # saved PyTCI 1.1 run (earlier analysis)
v1 = pd.read_pickle(_v1p) if _v1p.exists() else None             # saved 209,406-procedure cohort (earlier analysis)
print(f'events {len(ev):,} rows / {ev.OR_CASE_ID.nunique():,} cases; saved PyTCI run: {stored is not None}; '
      f'saved v1 cohort: {v1 is not None}')
if STRICT:
    assert ev.OR_CASE_ID.nunique() == 222_745
if stored is not None:
    # Check B2: case metadata identical to the saved run
    mm_ = META.merge(stored[[c for c in META.columns if c in stored.columns]], on='OR_CASE_ID', suffixes=('', '_ref'))
    bad_ = [c for c in META.columns if c != 'OR_CASE_ID' and c + '_ref' in mm_ and not col_equal(mm_[c], mm_[c + '_ref'])]
    REPRO['B_case_metadata'] = 'identical' if not bad_ else f'DIFFERS in {bad_}'
    print('Check B2 (case metadata vs saved run):', REPRO['B_case_metadata'], f'({len(mm_):,} cases)')
    assert not bad_, 'Case metadata differ from the saved run: STOP'

# contiguous case blocks in event-file order
codes, uniq = pd.factorize(ev.OR_CASE_ID, sort=False)
starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
assert len(starts) == len(uniq), 'Case rows are not contiguous in propsig_clean: STOP'
ends = np.r_[starts[1:], len(ev)]

def ns(series):
    return pd.to_datetime(series).astype('datetime64[ns]').to_numpy().astype('int64')
NAT = np.iinfo('int64').min
TT = ns(ev.TAKEN_TIME)
UNITS = ev.UNITS.astype(str).str.upper().str.strip().to_numpy()
KIND = np.where(UNITS == 'MG', 1, np.where(UNITS == 'MCG/KG/MIN', 2, 0)).astype(np.int8)
SIG = pd.to_numeric(ev.SIG, errors='coerce').to_numpy(dtype=float)
BOL = pd.to_numeric(ev.bolus_mg, errors='coerce').fillna(0).to_numpy()
INF = pd.to_numeric(ev.infusion_rate_mcg_kg_min, errors='coerce').fillna(0).to_numpy()
ANES = ns(ev.ANES_STOP)
TERM = ns(ev.terminal_stop_time)
AGE = pd.to_numeric(ev.AGE, errors='coerce').to_numpy(dtype=float)
WT = pd.to_numeric(ev.WEIGHT_KG, errors='coerce').to_numpy(dtype=float)
HT = pd.to_numeric(ev.HEIGHT_CM, errors='coerce').to_numpy(dtype=float)
SEXC = ev.SEX.astype(str).str.upper().str[:1].to_numpy()
NUMBERS['negative_or_zero_bolus_rows'] = int(((KIND == 1) & ~(SIG > 0) & ~np.isnan(SIG)).sum())
NUMBERS['rate_rows_missing_value'] = int(((KIND == 2) & np.isnan(SIG)).sum())

# %% [markdown]
# ## 3. Endpoint rules (legacy reproduced exactly; v2 = infusion-state rule)

# %%
def endpoints(lo, hi, order):
    """order: row indices of the case in processing order. Returns (legacy_end, legacy_reason, v2_end, v2_reason)."""
    anes = ANES[lo]
    term = TERM[lo]
    act = [i for i in order if BOL[i] > 0 or INF[i] > 0]
    if not act:
        r = (anes, 'no_activity')
        return r + r
    last_act_time = max(TT[i] for i in act)
    if term != NAT:
        e = min(term, anes) if anes != NAT else term
        return e, 'terminal_stop', e, 'terminal_stop'
    # legacy: is the LAST activity row an infusion row?
    last_row = [i for i in act if TT[i] == last_act_time][-1]
    if INF[last_row] > 0:
        leg = (anes, 'anes_stop_no_terminal_stop')
    else:
        b = last_act_time + BOLUS_TAIL_S * 10**9
        leg = ((min(b, anes) if anes != NAT else b), 'last_bolus_plus_5min')
    # v2: infusion STATE after all events (nonmissing rate entries, processing order)
    rate = 0.0
    for i in order:
        if KIND[i] == 2 and not np.isnan(SIG[i]):
            rate = SIG[i]
    if rate > 0:
        v2 = (anes, 'anes_stop_infusion_running')
    else:
        b = last_act_time + BOLUS_TAIL_S * 10**9
        v2 = ((min(b, anes) if anes != NAT else b), 'last_bolus_plus_5min')
    return leg + v2

# %% [markdown]
# ## 4. Full-cohort reconstruction (resumable batches of 20,000)

# %%

def _ts(v):
    return pd.NaT if v == NAT else pd.Timestamp(int(v))

def process_case(ci, want_traj=False):
    lo, hi = starts[ci], ends[ci]
    idx = np.arange(lo, hi)
    t = TT[idx]
    valid = t != NAT
    order = idx[valid][np.argsort(t[valid], kind='stable')]
    info = order[0] if len(order) else lo
    cid = uniq[ci]
    rec = dict(OR_CASE_ID=cid)
    age, w, h, sx = AGE[info], WT[info], HT[info], SEXC[info]
    if np.isnan(age) or np.isnan(w) or np.isnan(h) or ANES[lo] == NAT:
        rec['SKIP'] = 'missing_covariate'; return rec
    if sx not in ('M', 'F'):
        rec['SKIP'] = 'sex'; return rec
    if len(order) == 0:
        rec['SKIP'] = 'no_time'; return rec
    first = TT[order[0]]
    leg_end, leg_reason, v2_end, v2_reason = endpoints(lo, hi, order)
    rec.update(LEGACY_END_TIME=_ts(leg_end), LEGACY_END_REASON=leg_reason)
    if v2_end == NAT or v2_end <= first:
        rec['SKIP'] = 'nonpositive_window'; return rec
    if not (w > 0 and h > 0):
        rec['SKIP'] = 'model_error'; return rec
    try:
        model = ExactEleveld(float(age), float(w), float(h), sx.lower())
    except Exception:
        rec['SKIP'] = 'model_error'; return rec
    T = int((v2_end - first) / 1e9)
    keep = order[TT[order] <= v2_end]
    rel = np.round((TT[keep] - first) / 1e9).astype(np.int64)
    ok = (rel >= 0) & (rel <= T)
    keep, rel = keep[ok], rel[ok]
    k = KIND[keep].copy()
    v = SIG[keep].copy()
    drop = (k == 0) | np.isnan(v) | ((k == 1) & (v <= 0)) | ((k == 2) & (v < 0))
    ce, _ = simulate_exact(model, rel[~drop], k[~drop], v[~drop], T, float(w))
    ce50 = 3.08 * math.exp(-0.00635 * (age - 35))
    def summ(c, pfx):
        return {f'{pfx}MAX_CE': float(c.max()), f'{pfx}MEAN_CE': float(c.mean()), f'{pfx}MEDIAN_CE': float(np.median(c)),
                f'{pfx}P10_CE': float(np.percentile(c, 10)), f'{pfx}P25_CE': float(np.percentile(c, 25)),
                f'{pfx}P75_CE': float(np.percentile(c, 75)), f'{pfx}P90_CE': float(np.percentile(c, 90)),
                f'{pfx}FRAC_ABOVE_CE50': float((c > ce50).mean()),
                f'{pfx}FRAC_ABOVE_BIS60': float((c > BIS60_FACTOR * ce50).mean())}
    rec.update(summ(ce, ''))
    rec.update(FIRST_PROP_TIME=_ts(first), SIM_END_TIME=_ts(v2_end), SIM_END_REASON=v2_reason, SIM_DURATION_SEC=T,
               N_PROPOFOL_ROWS=int(len(keep)), WEIGHT_KG=float(w), HEIGHT_CM=float(h), ENGINE_EIG=bool(model.use_eig))
    # legacy-endpoint window (prefix of the same trajectory) for the endpoint sensitivity analysis
    if leg_end != v2_end and leg_end != NAT and leg_end > first:
        Tl = int((leg_end - first) / 1e9)
        rec.update(summ(ce[:Tl + 1], 'LEGACYEND_'))
        rec['LEGACYEND_DURATION_SEC'] = Tl
    if want_traj:
        return rec, (model, rel[~drop], k[~drop], v[~drop], T, float(w))
    return rec

_h = hashlib.sha256()
for _a in (TT, KIND, SIG, BOL, INF, ANES, TERM, AGE, WT, HT, SEXC.astype('U1')):
    _h.update(np.ascontiguousarray(_a).tobytes())
_h.update('|'.join(map(str, uniq)).encode())
def _codesig(obj):
    # deterministic fingerprint from compiled code (inspect.getsource fails for code defined in a notebook)
    if isinstance(obj, type):
        return b''.join(k.encode() + _codesig(v) for k, v in sorted(vars(obj).items()) if callable(v))
    def walk(c):
        out = [c.co_code, repr(c.co_names).encode(), repr(c.co_varnames).encode()]
        for k in c.co_consts:
            out.append(walk(k) if hasattr(k, 'co_code') else repr(sorted(map(repr, k)) if isinstance(k, frozenset) else k).encode())
        return b''.join(out)
    return walk(obj.__code__)
for _fn in (eleveld_params, _alsallami, _janmahasatian, ExactEleveld, simulate_exact, endpoints, process_case):
    _h.update(_codesig(_fn))
src_fingerprint = _h.hexdigest()           # any change to the data or the reconstruction code starts a fresh run
manifest = dict(version=VERSION, source=src_fingerprint, batch_size=BATCH_SIZE, bolus_tail_s=BOLUS_TAIL_S)
mpath = BATCH_DIR / 'manifest.json'
if mpath.exists() and json.loads(mpath.read_text()) != manifest:
    _old = sorted(BATCH_DIR.glob('batch_*.pkl'))
    for _p in _old:
        _p.unlink()
    print(f'Data or reconstruction code changed since the saved batches were made: removed {len(_old)} old batches')
mpath.write_text(json.dumps(manifest, indent=2))

n_cases = len(uniq)
n_batches = math.ceil(n_cases / BATCH_SIZE)
for b in range(n_batches):
    p = BATCH_DIR / f'batch_{b:04d}.pkl'
    if p.exists():
        continue
    tb = time.time()
    recs = [process_case(ci) for ci in range(b * BATCH_SIZE, min((b + 1) * BATCH_SIZE, n_cases))]
    tmp = p.with_suffix('.tmp')
    pd.DataFrame(recs).to_pickle(tmp); os.replace(tmp, p)
    print(f'batch {b + 1}/{n_batches} done in {time.time() - tb:.0f}s', flush=True)
allrec = pd.concat([pd.read_pickle(BATCH_DIR / f'batch_{b:04d}.pkl') for b in range(n_batches)], ignore_index=True)
assert len(allrec) == n_cases and allrec.OR_CASE_ID.is_unique
sims = allrec.loc[allrec.SKIP.isna()].copy() if 'SKIP' in allrec else allrec.copy()
for c in ('FIRST_PROP_TIME', 'SIM_END_TIME', 'LEGACY_END_TIME'):
    sims[c] = pd.to_datetime(sims[c])
print('Skip reasons:', allrec.SKIP.value_counts(dropna=False).to_dict() if 'SKIP' in allrec else {})
print('Simulated:', len(sims), '| eigen path:', int(sims.ENGINE_EIG.sum()), '| v2 endpoint reasons:', sims.SIM_END_REASON.value_counts().to_dict())

# %% [markdown]
# ### Gates G2-G4: reproduce the earlier PyTCI-based run where nothing should have changed

# %%
if STRICT:
    assert len(sims) == 214_625, 'Number of reconstructed cases differs from the analyzed data'
mis = pd.DataFrame(columns=['OR_CASE_ID'])
if stored is None:
    print('Saved PyTCI run not found: gates G2-G4 skipped')
else:
    # G2: same reconstructed case set as the earlier run
    same_set = set(sims.OR_CASE_ID) == set(stored.OR_CASE_ID)
    print('G2 reconstructed case set identical to the earlier run:', same_set, len(sims), len(stored))
    assert same_set, 'Reconstructed case set differs from the earlier run: STOP and inspect skip reasons'

    # G3: legacy endpoint rule reproduces stored SIM_END_TIME / SIM_END_REASON for every case
    chk = sims[['OR_CASE_ID', 'LEGACY_END_TIME', 'LEGACY_END_REASON']].merge(
        stored[['OR_CASE_ID', 'SIM_END_TIME', 'SIM_END_REASON']], on='OR_CASE_ID', validate='one_to_one')
    mis = chk.loc[(pd.to_datetime(chk.SIM_END_TIME) != chk.LEGACY_END_TIME) | (chk.SIM_END_REASON != chk.LEGACY_END_REASON)]
    print('G3 legacy endpoint mismatches before tie re-check:', len(mis))
    if len(mis):
        # The earlier run sorted with a non-stable sort, so the 'last activity row' is order-dependent ONLY when several
        # activity rows share the last activity timestamp (or rows lack a time). Every mismatch must be such a case.
        pos = dict(zip(uniq, range(len(uniq))))
        unexplained = 0
        for cid in mis.OR_CASE_ID:
            lo, hi = starts[pos[cid]], ends[pos[cid]]
            act = [i for i in range(lo, hi) if BOL[i] > 0 or INF[i] > 0]
            tt = [TT[i] for i in act if TT[i] != NAT]
            tie = (len(tt) > 0 and sum(x == max(tt) for x in tt) > 1) or any(TT[i] == NAT for i in range(lo, hi))
            unexplained += int(not tie)
        print('G3 mismatches not explained by same-timestamp ordering (must be 0):', unexplained)
        assert unexplained == 0, 'Legacy endpoint not reproduced: STOP'
        NUMBERS['endpoint_tie_order_cases'] = int(len(mis))
    print('G3 PASS')

    # G4: this notebook's event parser + legacy semantics + PyTCI reproduce the saved PyTCI Ce (300 cases)
    if importlib.util.find_spec('PyTCI') is not None:
        from PyTCI.models import propofol as _pp
        rng = np.random.default_rng(SEED)
        stored_i = stored.set_index('OR_CASE_ID')
        pos = dict(zip(uniq, range(len(uniq))))
        cand = chk.loc[~chk.OR_CASE_ID.isin(set(mis.OR_CASE_ID)), 'OR_CASE_ID'].to_numpy()
        samp = rng.choice(cand, size=min(300, len(cand)), replace=False)
        errs = []
        for cid in samp:
            ci = pos[cid]; lo, hi = starts[ci], ends[ci]
            idx = np.arange(lo, hi); t = TT[idx]; valid = t != NAT
            order = idx[valid][np.argsort(t[valid], kind='stable')]
            first = TT[order[0]]
            s = stored_i.loc[cid]
            end = ns(pd.Series([s.SIM_END_TIME]))[0]
            T = int((end - first) / 1e9)
            keep = order[TT[order] <= end]
            rel = np.round((TT[keep] - first) / 1e9).astype(np.int64)
            ok = (rel >= 0) & (rel <= T); keep, rel = keep[ok], rel[ok]
            info = order[0]
            m = _pp.Eleveld(age=float(AGE[info]), weight=float(WT[info]), height=float(HT[info]), sex=SEXC[info].lower())
            w = float(WT[info]); rate = 0.0; ce = np.empty(T + 1)
            byt = {}
            for i, r_ in zip(keep, rel):
                byt.setdefault(int(r_), []).append(i)
            def app(tt, rate):
                for i in byt.get(tt, []):
                    val = SIG[i] if not np.isnan(SIG[i]) else 0.0     # legacy semantics
                    if KIND[i] == 1: m.give_drug(val)
                    elif KIND[i] == 2: rate = val * w / 60000.0
                return rate
            rate = app(0, rate); ce[0] = m.xeo
            for tt in range(1, T + 1):
                if rate > 0: m.give_drug(rate)
                m.wait_time(1); rate = app(tt, rate); ce[tt] = m.xeo
            errs.append(max(abs(np.median(ce) - s.MEDIAN_CE), abs(ce.max() - s.MAX_CE)))
        errs = np.array(errs)
        n_bad = int((errs > 1e-8).sum())
        print(f'G4 legacy PyTCI reproduction: {len(errs) - n_bad}/{len(errs)} exact (max diff {errs.max():.2e})')
        assert n_bad <= 1, 'Event parsing does not reproduce the earlier run: STOP'   # 1 allowed for a same-second tie
        NUMBERS['G4_pytci_reproduced'] = f'{len(errs) - n_bad}/{len(errs)}'
    else:
        print('G4 skipped (PyTCI unavailable)')

# %% [markdown]
# ## 5. Cohort: earlier eligibility rules (must reproduce the saved 209,406 cohort), then the final eligibility rules

# %%
pre = sims.loc[~sims.OR_CASE_ID.isin(ghost_ids)].copy()
meta_cols = ['OR_CASE_ID', 'AGE', 'SEX', 'ASA_SCORE', 'BMI', 'IM_PROC_NAME', 'ANES_START', 'ANES_STOP']
meta = META
a = pre.merge(meta, on='OR_CASE_ID', how='left', validate='one_to_one')
for c in ['AGE', 'ASA_SCORE', 'BMI']:
    a[c] = pd.to_numeric(a[c], errors='coerce')
a['SEX'] = a['SEX'].astype('string').str.upper().str.strip().str[0]
a.loc[~a['SEX'].isin(['F', 'M']), 'SEX'] = pd.NA
a['IM_PROC_NAME'] = a['IM_PROC_NAME'].fillna('').astype(str)
pl = a.IM_PROC_NAME.str.lower()
cc, ce_ = pl.str.contains('colonoscopy', regex=False), pl.str.contains('endoscopy', regex=False)
a['PROC_GROUP_3'] = np.select([cc & ~ce_, ce_ & ~cc, cc & ce_], ['Colonoscopy only', 'Endoscopy only', 'Both'], default='Other')
a['CE50_AGE'] = 3.08 * np.exp(-0.00635 * (a.AGE - 35))
a['CE_BIS60_AGE'] = BIS60_FACTOR * a.CE50_AGE
a['AGE_BIN'] = pd.cut(a.AGE, bins=AGE_BINS, labels=AGE_LABELS, right=False)

flow = [('Cases with a propofol record (extracted)', N_EXTRACTED),
        ('After dose-plausibility screening (Stage 02)', int(ev.OR_CASE_ID.nunique())),
        ('Reconstructed (complete inputs, positive interval)', int(len(sims))),
        ('After bolus-only 15-min tail rule', int(len(pre)))]
steps = [('Age 18-90 yr', a.AGE.between(18, 90)), ('Recorded sex', a.SEX.isin(['F', 'M'])),
         ('Recorded ASA physical status', a.ASA_SCORE.notna()), ('Recorded BMI', a.BMI.notna()),
         ('Recognized procedure category', a.PROC_GROUP_3.isin(GROUPS)),
         ('Complete concentration summaries', a[['MAX_CE', 'MEDIAN_CE', 'P75_CE']].notna().all(axis=1))]
mask = pd.Series(True, index=a.index)
for name, cond in steps:
    mask &= cond.fillna(False).astype(bool)
    flow.append((name, int(mask.sum())))
base = a.loc[mask].copy()
if STRICT:
    assert len(base) == 209_406
if v1 is not None:
    v1_set_equal = set(base.OR_CASE_ID) == set(v1.OR_CASE_ID)
    print('Pre-exclusion cohort:', len(base), '| identical case set to v1 (209,406):', v1_set_equal)
    REPRO['G5_v1_cohort'] = 'identical' if v1_set_equal else 'DIFFERS'
    assert v1_set_equal, 'Rebuilt cohort differs from v1: STOP'
    print('G5 PASS')

# %% [markdown]
# ### New rule A: anthropometric plausibility

# %%
lo_w, hi_w = PLAUSIBLE['weight']; lo_h, hi_h = PLAUSIBLE['height']; lo_b, hi_b = PLAUSIBLE['bmi']
implaus = (~base.WEIGHT_KG.between(lo_w, hi_w)) | (~base.HEIGHT_CM.between(lo_h, hi_h)) | (~base.BMI.between(lo_b, hi_b))
NUMBERS['excluded_implausible_anthropometrics'] = int(implaus.sum())
step_a = base.loc[~implaus].copy()
flow = [r for r in flow if not r[0].startswith(('Plausible weight', 'Gastrointestinal endoscopy only'))]   # idempotent on re-run
flow.append((f'Plausible weight {lo_w}-{hi_w} kg, height {lo_h}-{hi_h} cm, BMI {lo_b}-{hi_b}', len(step_a)))
print('Excluded implausible anthropometrics:', int(implaus.sum()))

# %% [markdown]
# ### New rule B: procedure scope
# Component-level classifier on the procedure title (semicolon-separated components). A title is **EXCLUDED** if any
# component matches a nongastrointestinal or non-endoscopic surgical term (sleep endoscopy, airway/ENT, urologic,
# gynecologic, thoracic, cardiac, vascular access, laparoscopy, percutaneous liver, anorectal surgery such as
# hemorrhoidectomy, fistulotomy, seton, sphincterotomy, skin-tag excision). GI therapeutic adjuncts (biopsy,
# polypectomy, dilation, banding, PEG, stent, hemostasis, etc.) are kept. These keyword rules also keep POEM/myotomy,
# endoscopic sleeve gastroplasty and fundoplication; the author's overrides file excludes them. Titles with an unrecognized
# component are **REVIEW** (kept in the primary analysis; excluded in a sensitivity analysis). Overrides file wins.

# %%
NON_GI = [r'\bsleep\b', r'\bdise\b', r'bronch', r'laryng', r'trache', r'\bnasal\b', r'\bsinus', r'septo', r'turbinat',
          r'tonsil', r'adenoid', r'myringot', r'\bear\b', r'thyroid', r'bladder', r'cysto', r'uret', r'prostat',
          r'kidney', r'renal\b', r'nephr', r'hystero', r'uter', r'cervi(?!cal esoph)', r'colposc', r'vagin', r'\bd&c\b',
          r'laparoscop', r'laparotom', r'thoraco', r'echocardiog', r'\btee\b', r'cardiover', r'cardiac', r'pacemaker',
          r'\bpicc\b', r'\bport\b', r'central (venous )?(line|catheter)', r'\bliver\b', r'breast', r'dental', r'\btooth',
          r'teeth', r'cataract', r'ophthal', r'hemorrhoidectomy', r'fistul(ectomy|otomy)', r'\bseton', r'sphincterotomy',
          r'skin tag', r'tag excision', r'fissur', r'abscess', r'pilonidal', r'examination under anesth', r'\beua\b',
          r'hernia', r'lymph node', r'bone marrow', r'lumbar puncture', r'arthro', r'\bjoint', r'nerve block',
          r'\bmri\b', r'\bct\b', r'\bpet\b']
GI = [r'colonoscop', r'endoscop', r'sigmoidoscop', r'proctoscop', r'anoscop', r'\begd\b', r'esophag', r'gastr', r'duoden',
      r'jejun', r'ile(o|um|al)', r'\bcolon', r'rect', r'pouch', r'stoma\b', r'ostomy', r'polyp', r'biops', r'dilat',
      r'\bband', r'ligation', r'varic', r'inject', r'hemosta', r'\bclip', r'cauter', r'argon', r'\bapc\b', r'ablat',
      r'radiofrequency', r'barrett', r'\bstent', r'\bpeg\b', r'feeding tube', r'nasogastric', r'nasojejunal',
      r'foreign body', r'removal', r'tattoo', r'mucosal resection', r'\bemr\b', r'\besd\b', r'submucosal', r'capsule',
      r'bravo', r'\bph\b', r'manometr', r'motility', r'botul', r'botox', r'decompress', r'volvulus', r'strictur',
      r'balloon', r'myotomy', r'\bpoem\b', r'upper gi', r'lower gi', r'\bgi\b', r'brush',
      r'snare', r'bleed', r'hemorrhage', r'cec(al|um)', r'small bowel', r'\bbowel', r'enteral', r'transnasal']
# Generic modifiers ("with", "diagnostic", "screening") are deliberately NOT GI evidence: a component made only of
# them is UNCLASSIFIED -> REVIEW (kept in the primary analysis, listed in the private review sheet).
NON_GI_RE = [re.compile(p) for p in NON_GI]
GI_RE = [re.compile(p) for p in GI]
def classify_component(s):
    s = s.lower().strip()
    hit = [p.pattern for p in NON_GI_RE if p.search(s)]
    if hit:
        return 'NON_GI', hit[0]
    if any(p.search(s) for p in GI_RE):
        return 'GI', ''
    return 'UNCLASSIFIED', ''
def classify_title(title):
    comps = [c for c in str(title).split(';') if c.strip()] or [str(title)]
    res = [classify_component(c) for c in comps]
    if any(r[0] == 'NON_GI' for r in res):
        return 'EXCLUDE', next(r[1] for r in res if r[0] == 'NON_GI'), '; '.join(c.strip() for c, r in zip(comps, res) if r[0] != 'GI')
    if any(r[0] == 'UNCLASSIFIED' for r in res):
        return 'REVIEW', '', '; '.join(c.strip() for c, r in zip(comps, res) if r[0] == 'UNCLASSIFIED')
    return 'KEEP', '', ''

titles = step_a.groupby('IM_PROC_NAME', dropna=False).agg(N_CASES=('OR_CASE_ID', 'size')).reset_index()
cls = titles.IM_PROC_NAME.map(classify_title)
titles['AUTO_ACTION'] = [c[0] for c in cls]
titles['MATCHED_NON_GI_TERM'] = [c[1] for c in cls]
titles['FLAGGED_COMPONENTS'] = [c[2] for c in cls]
ov_path = PRIVATE / 'procedure_title_overrides_PRIVATE.csv'
titles['ACTION'] = titles.AUTO_ACTION
OVERRIDES = {}
if ov_path.exists():
    ov = pd.read_csv(ov_path).dropna(subset=['IM_PROC_NAME', 'ACTION'])
    ov['ACTION'] = ov.ACTION.str.upper().str.strip()
    assert ov.ACTION.isin(['KEEP', 'EXCLUDE']).all(), 'Overrides ACTION must be KEEP or EXCLUDE'
    OVERRIDES = dict(zip(ov.IM_PROC_NAME, ov.ACTION))
    titles['ACTION'] = [OVERRIDES.get(t, a_) for t, a_ in zip(titles.IM_PROC_NAME, titles.AUTO_ACTION)]
    unmatched = sorted(set(OVERRIDES) - set(titles.IM_PROC_NAME))
    print(f'Overrides read: {len(OVERRIDES)} | matched a cohort title: {len(OVERRIDES) - len(unmatched)} | '
          f'changed the automatic decision: {int((titles.ACTION != titles.AUTO_ACTION).sum())}')
    if unmatched:
        print('WARNING: these override titles match no title in the cohort (check spelling/spaces):')
        for t in unmatched:
            print('   ', repr(t))
NUMBERS['override_titles_changed'] = int((titles.ACTION != titles.AUTO_ACTION).sum())
NUMBERS['override_cases_changed'] = int(titles.loc[titles.ACTION != titles.AUTO_ACTION, 'N_CASES'].sum())
titles.sort_values(['ACTION', 'N_CASES'], ascending=[True, False]) \
      .to_csv(PRIVATE / 'pipeline_procedure_title_review_PRIVATE.csv', index=False)   # PRIVATE: free-text titles
act = dict(zip(titles.IM_PROC_NAME, titles.ACTION))
step_a['SCOPE_ACTION'] = step_a.IM_PROC_NAME.map(act)
scope_tab = (titles.groupby('ACTION').agg(titles=('IM_PROC_NAME', 'size'), cases=('N_CASES', 'sum')).reset_index())
print(scope_tab.to_string(index=False))
term_tab = (titles.loc[titles.ACTION == 'EXCLUDE'].groupby('MATCHED_NON_GI_TERM').N_CASES.sum()
            .sort_values(ascending=False).reset_index())
term_tab.loc[term_tab.N_CASES < MIN_CELL, 'N_CASES'] = -1
term_tab.to_csv(OUT / 'scope_excluded_by_term_suppressed.csv', index=False)
final = step_a.loc[step_a.SCOPE_ACTION != 'EXCLUDE'].copy()
NUMBERS['excluded_procedure_scope'] = int((step_a.SCOPE_ACTION == 'EXCLUDE').sum())
NUMBERS['review_titles_kept'] = int((titles.ACTION == 'REVIEW').sum())
NUMBERS['review_cases_kept'] = int((step_a.SCOPE_ACTION == 'REVIEW').sum())
flow = [r for r in flow if not r[0].startswith('Gastrointestinal endoscopy only')]   # idempotent on re-run
flow.append(('Gastrointestinal endoscopy only (procedure-scope rule)', len(final)))
flow_df = pd.DataFrame(flow, columns=['step', 'n'])
flow_df['excluded'] = (-flow_df.n.diff()).fillna(0).astype('Int64')
flow_df.to_csv(OUT / 'cohort_flow.csv', index=False)
print(flow_df.to_string(index=False))
NUMBERS['n_final'] = int(len(final))

# patient linkage
key = PATIENT_KEY
final = final.merge(key, on='OR_CASE_ID', how='left', validate='one_to_one')
assert final.PatientKey.notna().all(), 'Unlinked cases'
final['CLUSTER'] = pd.factorize(final.PatientKey)[0]
if PROVIDERS is not None:
    final = final.merge(PROVIDERS, on='OR_CASE_ID', how='left', validate='one_to_one')
else:
    final['PTYPE'], final['PROVIDER'] = 'None recorded', -1
final['PTYPE'] = final.PTYPE.fillna('None recorded')
final['PROVIDER'] = final.PROVIDER.fillna(-1).astype(int)
# procedures without a recorded provider form their own single-procedure clusters
_none = final.PROVIDER < 0
final.loc[_none, 'PROVIDER'] = final.PROVIDER.max() + 1 + np.arange(int(_none.sum()))
NUMBERS['provider_coverage'] = {k: int(v) for k, v in final.PTYPE.value_counts().items()}
pc = final.groupby('PatientKey').size()
NUMBERS.update(n_patients=int(len(pc)), patients_repeat=int((pc > 1).sum()),
               procedures_from_repeat=int(pc[pc > 1].sum()),
               pct_procedures_from_repeat=round(100 * pc[pc > 1].sum() / len(final), 1))
final.to_pickle(CHECK / 'pipeline_primary_cohort_PRIVATE.pkl')       # restricted case-level checkpoint
print('Final cohort', len(final), '| patients', len(pc))

# %% [markdown]
# ## 6. Statistical helpers (own implementation, cross-checked against statsmodels when available)

# %%
def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def prop_ci_cluster(y, g, z=1.959964):
    y = np.asarray(y, float); n = len(y); p = y.mean()
    s = pd.Series(y - p).groupby(np.asarray(g)).sum().to_numpy()
    G = len(s)
    se = math.sqrt(G / (G - 1) * np.sum(s ** 2)) / n
    return p, p - z * se, p + z * se

def _pair(g1, g2):
    return pd.factorize(pd.Series(np.asarray(g1)).astype(str) + '|' + pd.Series(np.asarray(g2)).astype(str))[0]

def prop_ci_twoway(y, g1, g2, z=1.959964):
    """Two-way cluster-robust CI (Cameron, Gelbach and Miller 2011): V = V_patient + V_provider - V_intersection."""
    p = float(np.mean(y))
    def v(g):
        _, lo, _ = prop_ci_cluster(y, g, z=1.0); return (p - lo) ** 2
    var = v(g1) + v(g2) - v(_pair(g1, g2))
    var = max(var, v(g1), v(g2))          # guard against a negative or understated variance
    se = math.sqrt(var)
    return p, p - z * se, p + z * se

def bs_basis(x, knots_inner, lower, upper, degree=3):
    """patsy bs(df=4) equivalent (include_intercept=False): drop the first B-spline column."""
    t = np.r_[[lower] * (degree + 1), knots_inner, [upper] * (degree + 1)]
    B = BSpline.design_matrix(np.clip(np.asarray(x, float), lower, upper), t, degree).toarray()
    return B[:, 1:]

class SplineOLS:
    """y ~ bs(AGE, df=4) + C(SEX) + ASA_SCORE + BMI ; conventional and patient-cluster-robust (CR1) inference."""
    def __init__(self, d, y):
        self.lower, self.upper = float(d.AGE.min()), float(d.AGE.max())
        self.inner = np.percentile(d.AGE.to_numpy(float), [50.0])
        X = self.design(d); yv = d[y].to_numpy(float)
        self.N, self.K = X.shape
        XtX_inv = np.linalg.inv(X.T @ X)
        self.beta = XtX_inv @ X.T @ yv
        u = yv - X @ self.beta
        s2 = (u @ u) / (self.N - self.K)
        self.V_conv = s2 * XtX_inv
        def meat(g):
            S = pd.DataFrame(X * u[:, None]).groupby(np.asarray(g)).sum().to_numpy()
            G = S.shape[0]
            return G / (G - 1) * (self.N - 1) / (self.N - self.K) * XtX_inv @ (S.T @ S) @ XtX_inv
        self.V_clu = meat(d.CLUSTER.to_numpy())                      # patient-clustered (checked against statsmodels)
        if 'PROVIDER' in d:                                           # two-way: patient and provider
            V2 = self.V_clu + meat(d.PROVIDER.to_numpy()) - meat(_pair(d.CLUSTER, d.PROVIDER))
            w, Q = np.linalg.eigh((V2 + V2.T) / 2)
            self.V_two = (Q * np.clip(w, 0, None)) @ Q.T             # nearest positive semidefinite
        else:
            self.V_two = self.V_clu
        self.df = self.N - self.K
        self.cols = ['Intercept', 'bs1', 'bs2', 'bs3', 'bs4', 'SEX_M', 'ASA_SCORE', 'BMI'] + getattr(self, 'extra', [])
    def design(self, d):
        cols = [np.ones(len(d)), bs_basis(d.AGE, self.inner, self.lower, self.upper),
                (d.SEX.astype(str) == 'M').to_numpy(float), d.ASA_SCORE.to_numpy(float), d.BMI.to_numpy(float)]
        for e in getattr(self, 'extra', []):
            cols.append(d[e].to_numpy(float))
        return np.column_stack(cols)
    def predict(self, d, cluster=True, twoway=False):
        from scipy.stats import t as tdist
        X = self.design(d); m = X @ self.beta
        V = self.V_two if twoway else (self.V_clu if cluster else self.V_conv)
        se = np.sqrt(np.einsum('ij,jk,ik->i', X, V, X))
        q = 1.959964 if cluster else tdist.ppf(0.975, self.df)
        return m, m - q * se, m + q * se
    def coef(self, name, cluster=True, twoway=False):
        j = self.cols.index(name); V = self.V_two if twoway else (self.V_clu if cluster else self.V_conv)
        se = math.sqrt(V[j, j]); return self.beta[j], self.beta[j] - 1.959964 * se, self.beta[j] + 1.959964 * se

class SplineOLSExtra(SplineOLS):
    def __init__(self, d, y, extra):
        self.extra = list(extra); super().__init__(d, y)

REF = dict(SEX='F', ASA_SCORE=float(final.ASA_SCORE.mean()), BMI=float(final.BMI.mean()))
NUMBERS['reference_profile'] = dict(sex='female', mean_asa=round(REF['ASA_SCORE'], 2), mean_bmi=round(REF['BMI'], 1))

# statsmodels cross-check (G6)
if importlib.util.find_spec('statsmodels') is not None and importlib.util.find_spec('patsy') is not None:
    import statsmodels.formula.api as smf
    sub = final.loc[final.PROC_GROUP_3 == 'Colonoscopy only'].copy()
    sub['SEX'] = sub.SEX.astype(str)
    mine = SplineOLS(sub, 'MEDIAN_CE')
    grid = pd.DataFrame(dict(AGE=np.linspace(sub.AGE.min(), sub.AGE.max(), 50), SEX='F', ASA_SCORE=REF['ASA_SCORE'], BMI=REF['BMI']))
    sm0 = smf.ols('MEDIAN_CE ~ bs(AGE, df=4) + C(SEX) + ASA_SCORE + BMI', data=sub).fit()
    sm1 = smf.ols('MEDIAN_CE ~ bs(AGE, df=4) + C(SEX) + ASA_SCORE + BMI', data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub.CLUSTER.to_numpy()})
    f0 = sm0.get_prediction(grid).summary_frame(); f1 = sm1.get_prediction(grid).summary_frame()
    m0, l0, u0 = mine.predict(grid, cluster=False); m1, l1, u1 = mine.predict(grid, cluster=True)
    d_ = max(np.abs(m0 - f0['mean']).max(), np.abs(l0 - f0.mean_ci_lower).max(), np.abs(l1 - f1.mean_ci_lower).max())
    print('G6 own spline/CR1 vs statsmodels max abs diff:', d_)
    assert d_ < 1e-6, 'Regression implementation differs from statsmodels: STOP'
else:
    print('G6 skipped (statsmodels/patsy not installed)')

# %% [markdown]
# ## 7. Primary and secondary results

# %%
f = final
N = len(f)
res_rows = []
for metric, col in [('median', 'MEDIAN_CE'), ('max', 'MAX_CE'), ('p25', 'P25_CE')]:
    for ref, rcol in [('ce50', 'CE50_AGE'), ('bis60', 'CE_BIS60_AGE')]:
        y = (f[col] > f[rcol]).to_numpy()
        k = int(y.sum()); wl, wh = wilson(k, N); p, cl, ch = prop_ci_cluster(y, f.CLUSTER)
        _, c2l, c2h = prop_ci_twoway(y, f.CLUSTER, f.PROVIDER)
        res_rows.append(dict(metric=metric, reference=ref, n=k, N=N, pct=100 * k / N, wilson_lo=100 * wl, wilson_hi=100 * wh,
                             cluster_lo=100 * cl, cluster_hi=100 * ch, twoway_lo=100 * c2l, twoway_hi=100 * c2h))
prim = pd.DataFrame(res_rows); prim.to_csv(OUT / 'exceedance_overall.csv', index=False)
print(prim.round(2).to_string(index=False))
def _num(metric, ref):
    r = prim.loc[(prim.metric == metric) & (prim.reference == ref)].iloc[0]
    return dict(n=int(r.n), pct=round(r.pct, 1), ci_cluster=[round(r.cluster_lo, 1), round(r.cluster_hi, 1)],
                ci_twoway=[round(r.twoway_lo, 1), round(r.twoway_hi, 1)])
NUMBERS['median_gt_ce50'] = _num('median', 'ce50'); NUMBERS['max_gt_ce50'] = _num('max', 'ce50')
NUMBERS['p25_gt_ce50'] = _num('p25', 'ce50'); NUMBERS['median_gt_bis60'] = _num('median', 'bis60')
NUMBERS['max_gt_bis60'] = _num('max', 'bis60')

f['R_MEDIAN'] = f.MEDIAN_CE / f.CE50_AGE
q = f.R_MEDIAN.quantile([.25, .5, .75]).round(2)
NUMBERS['ratio_median_iqr'] = [float(q[.5]), float(q[.25]), float(q[.75])]
fa = f.FRAC_ABOVE_CE50.quantile([.25, .5, .75])
NUMBERS['pct_time_above_ce50_median_iqr'] = [round(100 * fa[.5], 1), round(100 * fa[.25], 1), round(100 * fa[.75], 1)]
fb = f.FRAC_ABOVE_BIS60.quantile([.25, .5, .75])
NUMBERS['pct_time_above_bis60_median_iqr'] = [round(100 * fb[.5], 1), round(100 * fb[.25], 1), round(100 * fb[.75], 1)]
NUMBERS['sim_duration_min_median_iqr'] = [round(float(x) / 60, 1) for x in f.SIM_DURATION_SEC.quantile([.5, .25, .75])]
NUMBERS['endpoint_reasons'] = {k: dict(n=int(v), pct=round(100 * v / N, 1)) for k, v in f.SIM_END_REASON.value_counts().items()}
NUMBERS['n_any_infusion'] = None  # filled in section 9

grp = []
for g in GROUPS:
    s = f.loc[f.PROC_GROUP_3 == g]
    y = (s.MEDIAN_CE > s.CE50_AGE).to_numpy(); p, cl, ch = prop_ci_cluster(y, s.CLUSTER)
    _, c2l, c2h = prop_ci_twoway(y, s.CLUSTER, s.PROVIDER)
    ym = (s.MAX_CE > s.CE50_AGE).to_numpy()
    grp.append(dict(group=GROUP_LABEL[g], n=len(s), pct_of_cohort=100 * len(s) / N, median_gt_ce50_pct=100 * p,
                    ci_lo=100 * cl, ci_hi=100 * ch, max_gt_ce50_pct=100 * ym.mean(),
                    ratio_median=s.R_MEDIAN.median(), twoway_lo=100 * c2l, twoway_hi=100 * c2h))
grp = pd.DataFrame(grp); grp.to_csv(OUT / 'exceedance_by_group.csv', index=False)
NUMBERS['by_group'] = {r.group: dict(n=int(r.n), pct_cohort=round(r.pct_of_cohort, 1), median_gt_ce50=round(r.median_gt_ce50_pct, 1),
                                     max_gt_ce50=round(r.max_gt_ce50_pct, 1)) for r in grp.itertuples()}
ag = (f.groupby(['AGE_BIN', 'PROC_GROUP_3'], observed=True)
      .apply(lambda s: pd.Series(dict(n=len(s), median_gt_ce50_pct=100 * (s.MEDIAN_CE > s.CE50_AGE).mean(),
                                      max_gt_ce50_pct=100 * (s.MAX_CE > s.CE50_AGE).mean()))).reset_index())
ag.loc[ag.n < MIN_CELL, ['median_gt_ce50_pct', 'max_gt_ce50_pct']] = np.nan
ag.to_csv(OUT / 'exceedance_by_age_group.csv', index=False)
age_only = (f.groupby('AGE_BIN', observed=True)
            .apply(lambda s: pd.Series(dict(n=len(s), median_gt_ce50_pct=100 * (s.MEDIAN_CE > s.CE50_AGE).mean()))).reset_index())
NUMBERS['median_gt_ce50_by_age'] = {str(r.AGE_BIN): round(r.median_gt_ce50_pct, 1) for r in age_only.itertuples()}

# Individual Ce50 variability + dependence sensitivity
R = f.R_MEDIAN.to_numpy(); om = math.sqrt(OMEGA2_CE50)
NUMBERS['individual_ce50_expected_pct'] = round(100 * float(norm.cdf(np.log(R) / om).mean()), 1)
z = norm.ppf(f.R_MEDIAN.rank(method='average').to_numpy() / (N + 1))
cop = {rho: round(100 * float(norm.cdf((np.log(R) - om * rho * z) / (om * math.sqrt(1 - rho * rho))).mean()), 1)
       for rho in [-0.6, -0.3, 0.0, 0.3, 0.6]}
NUMBERS['individual_ce50_copula_rho_to_pct'] = {str(k): v for k, v in cop.items()}
print('Individual-Ce50 expectation', NUMBERS['individual_ce50_expected_pct'], '| copula', cop)

# Table 1
def med_iqr(s):
    qq = s.quantile([.25, .5, .75]); return f'{qq[.5]:.1f} ({qq[.25]:.1f}-{qq[.75]:.1f})'
t1 = []
for lab, s in [('Overall', f)] + [(GROUP_LABEL[g], f.loc[f.PROC_GROUP_3 == g]) for g in GROUPS]:
    n_ = len(s); asa = s.ASA_SCORE
    t1.append({'Group': lab, 'Procedures, n': f'{n_:,}', 'Patients, n': f'{s.PatientKey.nunique():,}',
               'Age, yr, median (IQR)': med_iqr(s.AGE), 'Female, n (%)': f'{(s.SEX == "F").sum():,} ({100 * (s.SEX == "F").mean():.1f})',
               'BMI, kg/m2, median (IQR)': med_iqr(s.BMI),
               **{f'ASA {lab_}, n (%)': f'{int(m_.sum()):,} ({100 * m_.mean():.1f})' for lab_, m_ in
                  [('I', asa == 1), ('II', asa == 2), ('III', asa == 3), ('IV-V', asa.isin([4, 5]))]},
               'Reconstruction interval, min, median (IQR)': med_iqr(s.SIM_DURATION_SEC / 60),
               'Any charted infusion, n (%)': None})
t1 = pd.DataFrame(t1)

# %% [markdown]
# ## 8. Adjusted concentration-age curves (Figure 2) and ratio distribution (Figure 3)

# %%
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
COL = {'Colonoscopy only': '#1f77b4', 'Endoscopy only': '#ff7f0e', 'Both': '#2ca02c'}
pred_rows, models = [], {}
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=False)
for ax, (yv, lab) in zip(axes, [('MEDIAN_CE', 'A  Within-procedure median'), ('MAX_CE', 'B  Within-procedure maximum')]):
    for g in GROUPS:
        s = f.loc[f.PROC_GROUP_3 == g].copy(); s['SEX'] = s.SEX.astype(str)
        mdl = SplineOLS(s, yv); models[(yv, g)] = mdl
        grid = pd.DataFrame(dict(AGE=np.linspace(max(18, s.AGE.min()), s.AGE.max(), 200), SEX='F', ASA_SCORE=REF['ASA_SCORE'], BMI=REF['BMI']))
        m, lo_, hi_ = mdl.predict(grid, cluster=True); m0, lo0, hi0 = mdl.predict(grid, cluster=False)
        _, lo2, hi2 = mdl.predict(grid, twoway=True)
        pred_rows.append(grid.assign(outcome=yv, group=GROUP_LABEL[g], fit=m, lo=lo_, hi=hi_, lo_conv=lo0, hi_conv=hi0, lo_twoway=lo2, hi_twoway=hi2))
        ax.plot(grid.AGE, m, color=COL[g], lw=1.6, label=GROUP_LABEL[g])
        ax.fill_between(grid.AGE, lo2, hi2, color=COL[g], alpha=.18, lw=0)      # two-way (patient and provider) 95% CI
    ages = np.linspace(18, 90, 200)
    ax.plot(ages, 3.08 * np.exp(-0.00635 * (ages - 35)), 'k--', lw=1.2, label='Ce50 (model BIS ≈ 47)')
    ax.plot(ages, BIS60_FACTOR * 3.08 * np.exp(-0.00635 * (ages - 35)), ':', color='0.45', lw=1.2, label='Model BIS 60')
    ax.set_xlabel('Age (yr)'); ax.set_ylabel('Effect-site concentration (µg/ml)'); ax.set_xlim(18, 90)
    ax.text(0.0, 1.03, lab, transform=ax.transAxes, fontweight='bold')
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, loc='lower center', ncol=5, frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'figure2_adjusted_ce_by_age.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
pred = pd.concat(pred_rows, ignore_index=True); pred.to_csv(OUT / 'figure2_predictions.csv', index=False)
pred['ratio'] = (pred.hi - pred.lo) / (pred.hi_conv - pred.lo_conv)
NUMBERS['cluster_ci_width_ratio_range'] = [round(float(pred.ratio.min()), 2), round(float(pred.ratio.max()), 2)]
at = pred.loc[pred.outcome == 'MEDIAN_CE'].copy()
NUMBERS['adjusted_median_ce_minus_ce50_range'] = [round(float((at.fit - 3.08 * np.exp(-0.00635 * (at.AGE - 35))).min()), 2),
                                                  round(float((at.fit - 3.08 * np.exp(-0.00635 * (at.AGE - 35))).max()), 2)]

fig, ax = plt.subplots(figsize=(4.8, 3.2))
xmax = max(1.6, float(np.percentile(R, 99.5)))
ax.hist(R[R <= xmax], bins=np.linspace(0, xmax, 75), weights=np.full((R <= xmax).sum(), 100 / len(R)), color='#4c78a8', edgecolor='white', lw=.25)
ax.axvline(1, color='#b73538', ls='--', lw=1.4, label='Median Ce = Ce50')
ax.axvline(np.median(R), color='k', lw=1.2, label=f'Cohort median ratio {np.median(R):.2f}')
ax.set_xlabel('Within-procedure median Ce / age-adjusted Ce50'); ax.set_ylabel('Procedures per bin (%)')
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'figure3_ratio_histogram.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
NUMBERS['pct_beyond_hist_axis'] = round(100 * float((R > xmax).mean()), 2)

fig, ax = plt.subplots(figsize=(4.8, 3.2))
for g in GROUPS:
    s = ag.loc[ag.PROC_GROUP_3 == g]
    ax.plot(s.AGE_BIN.astype(str), s.median_gt_ce50_pct, marker='o', color=COL[g], label=GROUP_LABEL[g])
ax.set_ylim(0, 100); ax.set_xlabel('Age (yr)'); ax.set_ylabel('Median Ce > Ce50 (% of procedures)')
ax.legend(frameon=False, fontsize=7.5)
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'figureS1_exceedance_by_age_group.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)

# STROBE flow diagram (Figure 1)
fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(flow_df) + 0.6)); ax.axis('off')
ys = np.linspace(1, 0, len(flow_df))
for i, r in enumerate(flow_df.itertuples()):
    ax.text(0.30, ys[i], f'{r.step}\nn = {int(r.n):,}' if pd.notna(r.n) else r.step, ha='center', va='center', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='0.3', lw=.8))
    if i > 0:
        ax.annotate('', xy=(0.30, ys[i] + 0.035), xytext=(0.30, ys[i - 1] - 0.035), arrowprops=dict(arrowstyle='->', color='0.3', lw=.8))
        if pd.notna(r.excluded) and int(r.excluded) > 0:
            ax.text(0.66, (ys[i] + ys[i - 1]) / 2, f'Excluded n = {int(r.excluded):,}', fontsize=7, va='center', color='0.25')
fig.tight_layout()
for ext in ('png', 'pdf'):
    fig.savefig(FIG / f'figure1_strobe_flow.{ext}', dpi=300, bbox_inches='tight')
plt.close(fig)
print('Figures written to', FIG)

# %% [markdown]
# ## 9. Infusion-rate selection (time-weighted modal rate, final cohort)

# %%
evr = ev.loc[ev.OR_CASE_ID.isin(set(f.OR_CASE_ID)) & (pd.Series(KIND, index=ev.index) == 2), ['OR_CASE_ID', 'TAKEN_TIME', 'SIG']].copy()
evr['_seq'] = np.arange(len(evr))
evr['TAKEN_TIME'] = pd.to_datetime(evr.TAKEN_TIME); evr['RATE'] = pd.to_numeric(evr.SIG, errors='coerce')
evr = evr.merge(f[['OR_CASE_ID', 'FIRST_PROP_TIME', 'SIM_END_TIME']], on='OR_CASE_ID', validate='many_to_one')
evr = evr.loc[evr.TAKEN_TIME.notna() & evr.RATE.notna() & (evr.RATE >= 0)]
evr = evr.loc[(evr.TAKEN_TIME >= evr.FIRST_PROP_TIME) & (evr.TAKEN_TIME <= evr.SIM_END_TIME)]
evr = evr.sort_values(['OR_CASE_ID', 'TAKEN_TIME', '_seq'], kind='stable')
nval = evr.groupby(['OR_CASE_ID', 'TAKEN_TIME'], sort=False).RATE.transform('nunique')
conflict = set(evr.loc[nval > 1, 'OR_CASE_ID'])
evr = evr.drop_duplicates(['OR_CASE_ID', 'TAKEN_TIME'], keep='last')
evr['NEXT'] = evr.groupby('OR_CASE_ID', sort=False).TAKEN_TIME.shift(-1).fillna(evr.SIM_END_TIME)
evr['DUR'] = (evr.NEXT - evr.TAKEN_TIME).dt.total_seconds()
any_pos = set(evr.loc[evr.RATE > 0, 'OR_CASE_ID'])
actv = evr.loc[(evr.RATE > 0) & (evr.DUR > 0)]
tot = actv.groupby(['OR_CASE_ID', 'RATE'], as_index=False).DUR.sum()
mx = tot.groupby('OR_CASE_ID').DUR.transform('max')
ties = tot.loc[np.isclose(tot.DUR, mx, rtol=0, atol=1e-8)].groupby('OR_CASE_ID').RATE.agg(['size', 'min'])
modal = tot.sort_values(['OR_CASE_ID', 'DUR', 'RATE'], ascending=[True, False, True]).drop_duplicates('OR_CASE_ID')
modal = modal.set_index('OR_CASE_ID').join(ties)
modal = modal.loc[~modal.index.isin(conflict)]
uniq_mode = modal.loc[modal['size'] == 1].copy()
fa_ = f.set_index('OR_CASE_ID')
uniq_mode['AGE_BIN'] = fa_.loc[uniq_mode.index, 'AGE_BIN'].astype(str).to_numpy()
modal['AGE_BIN'] = fa_.loc[modal.index, 'AGE_BIN'].astype(str).to_numpy()
rows = []
for lab in AGE_LABELS + ['Total']:
    s = uniq_mode if lab == 'Total' else uniq_mode.loc[uniq_mode.AGE_BIN == lab]
    rows.append({'age': lab, 'n': len(s), **{f'pct_{r}': 100 * (s.RATE == r).mean() for r in (100, 150, 200)},
                 'pct_any': 100 * s.RATE.isin([100, 150, 200]).mean()})
tabS2 = pd.DataFrame(rows); tabS2.to_csv(OUT / 'tableS2_modal_rate_anchors.csv', index=False)
print(tabS2.round(2).to_string(index=False))
NUMBERS['modal'] = dict(n_any_positive_rate=len(any_pos), n_conflicting=len(conflict & any_pos),
                        n_no_positive_duration=len(any_pos - set(tot.OR_CASE_ID) - conflict),
                        n_candidates=len(modal), n_ties=int((modal['size'] > 1).sum()), n_unique=len(uniq_mode),
                        pct_any=round(float(tabS2.iloc[-1].pct_any), 1),
                        pct_100=round(float(tabS2.iloc[-1].pct_100), 1), pct_150=round(float(tabS2.iloc[-1].pct_150), 1),
                        pct_200=round(float(tabS2.iloc[-1].pct_200), 1),
                        age_18_29=dict(p100=round(float(tabS2.iloc[0].pct_100), 1), p200=round(float(tabS2.iloc[0].pct_200), 1)),
                        age_80_90=dict(p100=round(float(tabS2.iloc[6].pct_100), 1), p200=round(float(tabS2.iloc[6].pct_200), 1)),
                        pct_any_ties_to_lower=round(100 * float(np.isin(np.where(modal['size'] > 1, modal['min'], modal.RATE), [100, 150, 200]).mean()), 1))
NUMBERS['n_any_infusion'] = dict(n=len(any_pos), pct=round(100 * len(any_pos) / N, 1))
t1['Any charted infusion, n (%)'] = [f'{len(any_pos & set(s.OR_CASE_ID)):,} ({100 * len(any_pos & set(s.OR_CASE_ID)) / len(s):.1f})'
                                     for s in [f] + [f.loc[f.PROC_GROUP_3 == g] for g in GROUPS]]
t1.to_csv(OUT / 'table1.csv', index=False)
print(t1.T.to_string())
try:
    import seaborn as sns
    sns.set_theme(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})
    kd = uniq_mode.loc[uniq_mode.RATE.between(0, 300)].copy()
    kd['AGE_BIN'] = pd.Categorical(kd.AGE_BIN, categories=AGE_LABELS, ordered=True)
    gF = sns.FacetGrid(kd, row='AGE_BIN', hue='AGE_BIN', row_order=AGE_LABELS, aspect=9, height=.75, palette='flare', sharey=False)
    gF.map(sns.kdeplot, 'RATE', bw_adjust=.35, clip=(0, 310), fill=True, alpha=1, lw=1.2)
    gF.map(sns.kdeplot, 'RATE', bw_adjust=.35, clip=(0, 310), color='w', lw=1.6)
    gF.map(plt.axhline, y=0, lw=1, color='black')
    for axk, lab in zip(gF.axes.flat, AGE_LABELS):
        axk.text(0, .2, lab, fontweight='bold', ha='left', va='center', transform=axk.transAxes, fontsize=8)
    gF.figure.subplots_adjust(hspace=-.25); gF.set_titles(''); gF.set(yticks=[], ylabel='', xlim=(0, 310)); gF.despine(bottom=True, left=True)
    gF.set_xlabels('Time-weighted modal infusion rate (µg · kg⁻¹ · min⁻¹)')
    for ext in ('png', 'pdf'):
        gF.figure.savefig(FIG / f'figureS2_modal_rate_kde.{ext}', dpi=300, bbox_inches='tight')
    plt.close(gF.figure)
    sns.reset_orig()
except Exception as e:
    print('KDE figure skipped:', e)

# %% [markdown]
# ## 10. Selected adjunct sensitivity

# %%
if DRUG_FLAGS is not None:
    fl = DRUG_FLAGS
    mm = f.merge(fl[['OR_CASE_ID', 'PROPOFOL_ONLY_SELECTED', 'ANY_SELECTED_ADJUNCT']], on='OR_CASE_ID', how='inner', validate='one_to_one')
    po = mm.loc[mm.PROPOFOL_ONLY_SELECTED == 1]
    NUMBERS['adjunct'] = dict(matched=len(mm), matched_pct=round(100 * len(mm) / N, 2), none_of_five=len(po),
                              none_pct=round(100 * len(po) / len(mm), 1), any=int(mm.ANY_SELECTED_ADJUNCT.sum()),
                              median_gt_ce50_none=round(100 * float((po.MEDIAN_CE > po.CE50_AGE).mean()), 1))
    rows, fig, axes = [], *plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, yv in zip(axes, ['MEDIAN_CE', 'MAX_CE']):
        for g in GROUPS:
            sa = mm.loc[mm.PROC_GROUP_3 == g].copy(); sa['SEX'] = sa.SEX.astype(str)
            sb = po.loc[po.PROC_GROUP_3 == g].copy(); sb['SEX'] = sb.SEX.astype(str)
            grid = pd.DataFrame(dict(AGE=np.linspace(18, min(sa.AGE.max(), sb.AGE.max()), 120), SEX='F', ASA_SCORE=REF['ASA_SCORE'], BMI=REF['BMI']))
            for lab, s_, ls in [('All matched', sa, '-'), ('No selected adjunct', sb, '--')]:
                mdl = SplineOLS(s_, yv); m, lo_, hi_ = mdl.predict(grid)
                ax.plot(grid.AGE, m, ls, color=COL[g], lw=1.3, label=f'{GROUP_LABEL[g]}, {lab.lower()}')
            ex = SplineOLSExtra(sa, yv, ['ANY_SELECTED_ADJUNCT']); b_, l_, h_ = ex.coef('ANY_SELECTED_ADJUNCT')
            _, l2, h2 = ex.coef('ANY_SELECTED_ADJUNCT', twoway=True)
            rows.append(dict(metric=yv, group=GROUP_LABEL[g], diff=b_, lo=l_, hi=h_, lo_twoway=l2, hi_twoway=h2))
        ax.set_xlabel('Age (yr)'); ax.set_ylabel(('Median' if yv == 'MEDIAN_CE' else 'Maximum') + ' Ce (µg/ml)')
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'figureS3_adjunct_sensitivity.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
    tS3 = pd.DataFrame(rows); tS3.to_csv(OUT / 'tableS3_adjunct_coefficients_clusterCI.csv', index=False)
    print(tS3.round(3).to_string(index=False))
else:
    print('Adjunct flags not available; skipped')

# %% [markdown]
# ## 10b. Optional: full PyTCI 1.1 reconstruction (switch `RUN_PYTCI_FULL`, off by default; takes hours)
# Only needed for the PyTCI row of Table 2 when the saved PyTCI run (`checkpoints/summary_custom_end_v1_full.pkl`) is not
# available. Uses PyTCI 1.1 exactly as the original analysis did (1-s steps, earlier end rule, same event handling as
# gate G4), for the final cohort only, in resumable batches. If the saved run is also present, the two are compared.

# %%
PYTCI_VALUES = None
if RUN_PYTCI_FULL:
    from PyTCI.models import propofol as _pp
    PT_DIR = CHECK / 'pipeline_pytci_batches'; PT_DIR.mkdir(parents=True, exist_ok=True)
    pos_pt = dict(zip(uniq, range(len(uniq))))
    leg_end = dict(zip(sims.OR_CASE_ID, ns(sims.LEGACY_END_TIME)))
    def pytci_case(cid):
        ci = pos_pt[cid]; lo, hi = starts[ci], ends[ci]
        idx = np.arange(lo, hi); t = TT[idx]; valid = t != NAT
        order = idx[valid][np.argsort(t[valid], kind='stable')]
        first = TT[order[0]]; end = leg_end[cid]
        T = int((end - first) / 1e9)
        keep = order[TT[order] <= end]
        rel = np.round((TT[keep] - first) / 1e9).astype(np.int64)
        ok = (rel >= 0) & (rel <= T); keep, rel = keep[ok], rel[ok]
        info = order[0]
        m = _pp.Eleveld(age=float(AGE[info]), weight=float(WT[info]), height=float(HT[info]), sex=SEXC[info].lower())
        w = float(WT[info]); rate = 0.0; ce = np.empty(T + 1); byt = {}
        for i, r_ in zip(keep, rel):
            byt.setdefault(int(r_), []).append(i)
        def app(tt, rate):
            for i in byt.get(tt, []):
                val = SIG[i] if not np.isnan(SIG[i]) else 0.0     # legacy semantics
                if KIND[i] == 1: m.give_drug(val)
                elif KIND[i] == 2: rate = val * w / 60000.0
            return rate
        rate = app(0, rate); ce[0] = m.xeo
        for tt in range(1, T + 1):
            if rate > 0: m.give_drug(rate)
            m.wait_time(1); rate = app(tt, rate); ce[tt] = m.xeo
        return dict(OR_CASE_ID=cid, MEDIAN_CE=float(np.median(ce)), MAX_CE=float(ce.max()))
    ids_pt = f.OR_CASE_ID.to_numpy(); B_PT = 5000; nb_pt = math.ceil(len(ids_pt) / B_PT); tp = time.time()
    for b in range(nb_pt):
        pb = PT_DIR / f'pytci_{b:04d}.pkl'
        if pb.exists():
            continue
        tmp = pb.with_suffix('.tmp')
        pd.DataFrame([pytci_case(c) for c in ids_pt[b * B_PT:(b + 1) * B_PT]]).to_pickle(tmp); os.replace(tmp, pb)
        print(f'PyTCI batch {b + 1}/{nb_pt} ({time.time() - tp:.0f}s)', flush=True)
    PYTCI_VALUES = pd.concat([pd.read_pickle(PT_DIR / f'pytci_{b:04d}.pkl') for b in range(nb_pt)], ignore_index=True)
    if stored is not None:
        cmp_pt = PYTCI_VALUES.merge(stored[['OR_CASE_ID', 'MEDIAN_CE', 'MAX_CE']], on='OR_CASE_ID', suffixes=('', '_saved'))
        dpt = np.maximum((cmp_pt.MEDIAN_CE - cmp_pt.MEDIAN_CE_saved).abs(), (cmp_pt.MAX_CE - cmp_pt.MAX_CE_saved).abs())
        REPRO['pytci_full_vs_saved'] = f'{int((dpt <= 1e-8).sum())}/{len(dpt)} within 1e-8 (others: same-timestamp ordering)'
        print('Full PyTCI rerun vs saved run:', REPRO['pytci_full_vs_saved'])
elif stored is not None:
    PYTCI_VALUES = stored[['OR_CASE_ID', 'MEDIAN_CE', 'MAX_CE']]
    print('PyTCI row: from the saved PyTCI run (checked by gate G4 when PyTCI is installed)')
else:
    print('PyTCI row skipped: no saved PyTCI run and RUN_PYTCI_FULL is off')

# %% [markdown]
# ## 11. Sensitivity analyses

# %%
sens = []
def add_sens(label, d, med_col='MEDIAN_CE', max_col='MAX_CE'):
    sens.append(dict(analysis=label, n=len(d), median_gt_ce50_pct=100 * float((d[med_col] > d.CE50_AGE).mean()),
                     max_gt_ce50_pct=100 * float((d[max_col] > d.CE50_AGE).mean())))
add_sens('Primary', f)
# (a) legacy late-bolus endpoint for the cases whose endpoint changed
chg = f.LEGACYEND_MEDIAN_CE.notna() if 'LEGACYEND_MEDIAN_CE' in f else pd.Series(False, index=f.index)
d_ = f.copy()
d_.loc[chg, 'MEDIAN_CE'] = d_.loc[chg, 'LEGACYEND_MEDIAN_CE']; d_.loc[chg, 'MAX_CE'] = d_.loc[chg, 'LEGACYEND_MAX_CE']
add_sens('Previous late-bolus endpoint rule', d_)
NUMBERS['endpoint_changed_cases'] = int(chg.sum())
if chg.any():
    cc_ = f.loc[chg]
    NUMBERS['endpoint_changed_added_min_median'] = round(float(((cc_.SIM_DURATION_SEC - cc_.LEGACYEND_DURATION_SEC) / 60).median()), 1)
    NUMBERS['endpoint_changed_median_ce_delta_median'] = round(float((cc_.MEDIAN_CE - cc_.LEGACYEND_MEDIAN_CE).median()), 3)
# (b) PyTCI 1.1 values (saved run or full rerun) on the same final cohort, earlier end rule
if PYTCI_VALUES is not None:
    st = f[['OR_CASE_ID', 'CE50_AGE']].merge(PYTCI_VALUES[['OR_CASE_ID', 'MEDIAN_CE', 'MAX_CE']], on='OR_CASE_ID')
    add_sens('PyTCI 1.1 engine and previous endpoint (v1 values)', st)
# (c) include bolus-only tail-rule cases
ga = sims.loc[sims.OR_CASE_ID.isin(ghost_ids)].merge(meta, on='OR_CASE_ID', how='left')
for c in ['AGE', 'ASA_SCORE', 'BMI']:
    ga[c] = pd.to_numeric(ga[c], errors='coerce')
ga['SEX'] = ga.SEX.astype('string').str.upper().str.strip().str[0]
gl = ga.IM_PROC_NAME.fillna('').astype(str).str.lower()
ga_ok = (ga.AGE.between(18, 90) & ga.SEX.isin(['F', 'M']) & ga.ASA_SCORE.notna() & ga.BMI.notna()
         & (gl.str.contains('colonoscopy', regex=False) | gl.str.contains('endoscopy', regex=False))
         & ga.WEIGHT_KG.between(lo_w, hi_w) & ga.HEIGHT_CM.between(lo_h, hi_h) & ga.BMI.between(lo_b, hi_b)
         & (ga.IM_PROC_NAME.fillna('').astype(str).map(lambda t: OVERRIDES.get(t, classify_title(t)[0])) != 'EXCLUDE')).fillna(False)
ga = ga.loc[ga_ok.astype(bool)].copy(); ga['CE50_AGE'] = 3.08 * np.exp(-0.00635 * (ga.AGE - 35))
add_sens('Including bolus-only 15-min tail-rule cases', pd.concat([f[['MEDIAN_CE', 'MAX_CE', 'CE50_AGE']], ga[['MEDIAN_CE', 'MAX_CE', 'CE50_AGE']]]))
NUMBERS['tail_rule_cases_eligible'] = int(len(ga))
add_sens('Excluding age recorded as 90 yr', f.loc[f.AGE < 90])
NUMBERS['age90_n'] = int((f.AGE >= 90).sum())
add_sens('Excluding titles with unrecognized components (REVIEW)', f.loc[f.SCOPE_ACTION != 'REVIEW'])
first_proc = f.sort_values('ANES_START' if 'ANES_START' in f else 'FIRST_PROP_TIME').drop_duplicates('PatientKey')
add_sens("Each patient's first procedure", first_proc)
add_sens('Previous eligibility (v1 case set, exact engine, v2 endpoint)', base)
sens_df = pd.DataFrame(sens); sens_df.to_csv(OUT / 'sensitivity_summary.csv', index=False)
print(sens_df.round(2).to_string(index=False))
NUMBERS['sensitivity'] = {r.analysis: dict(n=int(r.n), median=round(r.median_gt_ce50_pct, 1), max=round(r.max_gt_ce50_pct, 1)) for r in sens_df.itertuples()}

# %% [markdown]
# ### Documented-stop timing error: full cohort, symmetric (exact engine)
# Every procedure is recomputed with its window ending 2 or 5 min earlier (as if the stop entry was charted that much
# late) or extended 2 or 5 min with delivery set to zero (as if charted that much early). Paired comparisons; the
# "earlier" scenarios require at least 1 min of window to remain.

# %%
pos = dict(zip(uniq, range(len(uniq))))
ce50_of = dict(zip(f.OR_CASE_ID, f.CE50_AGE))
SCEN = ['stop_2min_late', 'stop_5min_late', 'stop_2min_early', 'stop_5min_early']
acc = {s_: [0, 0, 0] for s_ in SCEN}       # [n eligible, n above under primary, n above under scenario]
tw = time.time()
for i_, cid in enumerate(f.OR_CASE_ID.to_numpy()):
    rec, (model, rel, k_, v_, T, w) = process_case(pos[cid], want_traj=True)
    ce, post = simulate_exact(model, rel, k_, v_, T, w, post_tail=300)
    c50 = ce50_of[cid]
    base_above = float(np.median(ce)) > c50
    arrs = {'stop_2min_late': ce[:T - 120 + 1] if T - 120 >= 60 else None,
            'stop_5min_late': ce[:T - 300 + 1] if T - 300 >= 60 else None,
            'stop_2min_early': np.r_[ce, post[:120]], 'stop_5min_early': np.r_[ce, post[:300]]}
    for s_, arr in arrs.items():
        if arr is not None:
            acc[s_][0] += 1; acc[s_][1] += base_above; acc[s_][2] += float(np.median(arr)) > c50
    if (i_ + 1) % 50000 == 0:
        print(f'  window analysis {i_ + 1:,}/{N:,} ({time.time() - tw:.0f}s)', flush=True)
win = pd.DataFrame([dict(scenario=s_, n=v[0], primary_pct=100 * v[1] / v[0], scenario_pct=100 * v[2] / v[0]) for s_, v in acc.items()])
win['change_pp'] = win.scenario_pct - win.primary_pct
win.to_csv(OUT / 'tableS1_stop_timing_full_cohort.csv', index=False)
print(win.round(2).to_string(index=False))
NUMBERS['stop_timing'] = {r.scenario: dict(n=int(r.n), pct=round(r.scenario_pct, 1), change=round(r.change_pp, 1)) for r in win.itertuples()}

# %% [markdown]
# ## 12. Save aggregate outputs

# %%
NUMBERS['flow'] = flow_df.assign(n=flow_df.n.astype(float)).to_dict(orient='records')
NUMBERS['scope_table'] = scope_tab.to_dict(orient='records')
NUMBERS['version'] = VERSION
(OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
# LaTeX Table 1
cols = ['Overall', 'Colonoscopy only', 'Upper endoscopy only', 'Combined']
rows_tex = []
for col in [c for c in t1.columns if c != 'Group']:
    rows_tex.append(col.replace('%', r'\%').replace('kg/m2', r'kg/m$^2$') + ' & ' + ' & '.join(str(v).replace('%', r'\%') for v in t1[col]) + r' \\')
(OUT / 'table1.tex').write_text('\\begin{tabular}{lrrrr}\\toprule\n & ' + ' & '.join(cols) + ' \\\\\n\\midrule\n' + '\n'.join(rows_tex) + '\n\\bottomrule\\end{tabular}\n')
print('Analysis outputs written:')
for p in sorted(OUT.glob('*')):
    print('  ', p.name)
print('Figures:', sorted(x.name for x in FIG.glob('*.pdf')))

# %% [markdown]
# ## 13. Software versions, regression check against statsmodels (G6), modal-rate tie bounds

# %%
ADD = {'python': platform.python_version()}
for m_ in ['numpy', 'scipy', 'pandas', 'statsmodels', 'patsy', 'matplotlib']:
    try:
        ADD[m_] = importlib.import_module(m_).__version__
    except Exception:
        ADD[m_] = 'not installed'
print(json.dumps(ADD, indent=1))
G6 = {}
if importlib.util.find_spec('statsmodels') is not None:
    import statsmodels.formula.api as smf
    for grp in GROUPS:
        for yv in ['MEDIAN_CE', 'MAX_CE']:
            sub = final.loc[final.PROC_GROUP_3 == grp].copy(); sub['SEX'] = sub.SEX.astype(str)
            mine = SplineOLS(sub, yv)
            grid = pd.DataFrame(dict(AGE=np.linspace(sub.AGE.min(), sub.AGE.max(), 50), SEX='F', ASA_SCORE=REF['ASA_SCORE'], BMI=REF['BMI']))
            fml = f'{yv} ~ bs(AGE, df=4) + C(SEX) + ASA_SCORE + BMI'
            f0 = smf.ols(fml, data=sub).fit().get_prediction(grid).summary_frame()
            f1 = smf.ols(fml, data=sub).fit(cov_type='cluster', cov_kwds={'groups': sub.CLUSTER.to_numpy()}).get_prediction(grid).summary_frame()
            m0, l0, u0 = mine.predict(grid, cluster=False); m1, l1, u1 = mine.predict(grid, cluster=True)
            d_ = max(np.abs(m0 - f0['mean']).max(), np.abs(l0 - f0.mean_ci_lower).max(), np.abs(u0 - f0.mean_ci_upper).max(),
                     np.abs(l1 - f1.mean_ci_lower).max(), np.abs(u1 - f1.mean_ci_upper).max())
            G6[f'{grp} | {yv}'] = float(d_)
    ADD['G6_max_abs_diff'] = max(G6.values()); ADD['G6_by_model'] = G6
    print(f'G6 max abs difference vs statsmodels: {ADD["G6_max_abs_diff"]:.1e}')
    assert ADD['G6_max_abs_diff'] < 1e-6, 'Regression implementation differs from statsmodels: STOP'
# Procedures per responsible provider (procedures with no recorded provider are left out of this summary)
if 'PROVIDER' in final:
    pp_ = final.loc[final.PTYPE != 'None recorded'].groupby('PROVIDER').size()
    if len(pp_):
        ADD['procedures_per_provider'] = dict(n_providers=int(len(pp_)), median=float(pp_.median()),
            iqr=[float(pp_.quantile(.25)), float(pp_.quantile(.75))], min=int(pp_.min()), max=int(pp_.max()))
        print('Procedures per provider:', ADD['procedures_per_provider'])
# G7: the two-way (patient and provider) clustered intervals against statsmodels (formerly Stage 32)
G7_ERR = None
if importlib.util.find_spec('statsmodels') is not None and 'PROVIDER' in final:
    G7 = {}
    try:
        g2_ = np.column_stack([final.CLUSTER.to_numpy(), final.PROVIDER.to_numpy()])
        for metric_, col_ in [('median', 'MEDIAN_CE'), ('max', 'MAX_CE')]:
            y_ = (final[col_] > final.CE50_AGE).astype(float).to_numpy()
            fit_ = smf.ols('Y ~ 1', data=pd.DataFrame({'Y': y_})).fit(cov_type='cluster', cov_kwds={'groups': g2_})
            lo_s, hi_s = fit_.conf_int().iloc[0]
            _, lo_m, hi_m = prop_ci_twoway(y_, final.CLUSTER, final.PROVIDER)
            G7[f'proportion above Ce50, {metric_}'] = float(max(abs(lo_s - lo_m), abs(hi_s - hi_m)))
        for grp in GROUPS:
            for yv in ['MEDIAN_CE', 'MAX_CE']:
                sub = final.loc[final.PROC_GROUP_3 == grp].copy(); sub['SEX'] = sub.SEX.astype(str)
                mine = SplineOLS(sub, yv)
                grid = pd.DataFrame(dict(AGE=np.linspace(sub.AGE.min(), sub.AGE.max(), 50), SEX='F', ASA_SCORE=REF['ASA_SCORE'], BMI=REF['BMI']))
                gs_ = np.column_stack([sub.CLUSTER.to_numpy(), sub.PROVIDER.to_numpy()])
                f2 = smf.ols(f'{yv} ~ bs(AGE, df=4) + C(SEX) + ASA_SCORE + BMI', data=sub).fit(
                    cov_type='cluster', cov_kwds={'groups': gs_}).get_prediction(grid).summary_frame()
                m2, l2, u2 = mine.predict(grid, twoway=True)
                G7[f'{grp} | {yv}'] = float(max(np.abs(m2 - f2['mean']).max(), np.abs(l2 - f2.mean_ci_lower).max(),
                                                np.abs(u2 - f2.mean_ci_upper).max()))
    except Exception as e:                      # a software problem in the check itself: report it, do not stop the run
        G7_ERR = f'{type(e).__name__}: {e}'
        ADD['G7_error'] = G7_ERR
        print('*** G7 could not be computed:', G7_ERR)
    if G7:
        ADD['G7_max_abs_diff'] = max(G7.values()); ADD['G7_by_model'] = G7
        print(f'G7 two-way intervals, max abs difference vs statsmodels: {ADD["G7_max_abs_diff"]:.1e}')
        assert G7_ERR is not None or ADD['G7_max_abs_diff'] < 1e-6, 'Two-way clustered intervals differ from statsmodels: STOP'
else:
    print('G7 skipped (statsmodels not installed or no provider codes)')
# modal-rate tie bounds (same rate intervals as section 9)
evr_ = ev.loc[ev.OR_CASE_ID.isin(set(f.OR_CASE_ID)) & (ev.UNITS.astype(str).str.upper().str.strip() == 'MCG/KG/MIN'),
              ['OR_CASE_ID', 'TAKEN_TIME', 'SIG']].copy()
evr_['_seq'] = np.arange(len(evr_))
evr_['TAKEN_TIME'] = pd.to_datetime(evr_.TAKEN_TIME); evr_['RATE'] = pd.to_numeric(evr_.SIG, errors='coerce')
evr_ = evr_.merge(f[['OR_CASE_ID', 'FIRST_PROP_TIME', 'SIM_END_TIME']], on='OR_CASE_ID', validate='many_to_one')
evr_ = evr_.loc[evr_.TAKEN_TIME.notna() & evr_.RATE.notna() & (evr_.RATE >= 0)]
evr_ = evr_.loc[(evr_.TAKEN_TIME >= evr_.FIRST_PROP_TIME) & (evr_.TAKEN_TIME <= evr_.SIM_END_TIME)]
evr_ = evr_.sort_values(['OR_CASE_ID', 'TAKEN_TIME', '_seq'], kind='stable')
nval_ = evr_.groupby(['OR_CASE_ID', 'TAKEN_TIME'], sort=False).RATE.transform('nunique')
conflict_ = set(evr_.loc[nval_ > 1, 'OR_CASE_ID'])
evr_ = evr_.drop_duplicates(['OR_CASE_ID', 'TAKEN_TIME'], keep='last')
evr_['NEXT'] = evr_.groupby('OR_CASE_ID', sort=False).TAKEN_TIME.shift(-1).fillna(evr_.SIM_END_TIME)
evr_['DUR'] = (evr_.NEXT - evr_.TAKEN_TIME).dt.total_seconds()
tot_ = evr_.loc[(evr_.RATE > 0) & (evr_.DUR > 0)].groupby(['OR_CASE_ID', 'RATE'], as_index=False).DUR.sum()
tot_ = tot_.loc[~tot_.OR_CASE_ID.isin(conflict_)]
mx_ = tot_.groupby('OR_CASE_ID').DUR.transform('max')
top_ = tot_.loc[np.isclose(tot_.DUR, mx_, rtol=0, atol=1e-8)].assign(a=lambda d: d.RATE.isin([100, 150, 200]))
agg_ = top_.groupby('OR_CASE_ID').agg(n=('RATE', 'size'), any_=('a', 'any'), all_=('a', 'all'))
u_, t_ = agg_.loc[agg_.n == 1], agg_.loc[agg_.n > 1]
k_u = int(u_.any_.sum())
ADD['modal_unique_pct_anchor'] = round(100 * k_u / len(u_), 1)
ADD['modal_ties_n'] = int(len(t_))
ADD['modal_with_ties_bounds_pct'] = [round(100 * (k_u + int(t_.all_.sum())) / len(agg_), 1),
                                     round(100 * (k_u + int(t_.any_.sum())) / len(agg_), 1)]
print(f'Modal rate at 100/150/200: {ADD["modal_unique_pct_anchor"]}% of {len(u_):,} unique; with {len(t_):,} ties '
      f'{ADD["modal_with_ties_bounds_pct"][0]}% to {ADD["modal_with_ties_bounds_pct"][1]}%')
NUMBERS['addendum'] = ADD
(OUT / 'addendum_versions_G6_ties.json').write_text(json.dumps(ADD, indent=2))

# %% [markdown]
# ## 14. Engine validation: step-by-step matrix exponential and LSODA on 400 random histories

# %%
from scipy.integrate import solve_ivp
VAL = {'itiva_max_abs_rel_pct': round(float(g1.rel_pct.abs().max()), 3)}
def random_history(rng):
    age = float(rng.integers(18, 91)); w = float(rng.integers(40, 181)); h = float(rng.integers(145, 201))
    sex = str(rng.choice(['m', 'f'])); T = int(rng.integers(300, 3601))
    ev_ = [(0, 1, float(rng.integers(20, 151)))]; t = 0
    while True:
        t += int(rng.integers(20, 601))
        if t >= T:
            break
        ev_.append((t, 1, float(rng.integers(10, 61))) if rng.random() < 0.4
                   else (t, 2, float(rng.choice([0, 50, 75, 100, 125, 150, 200, 250]))))
    ev_.sort(key=lambda e: e[0])
    return age, w, h, sex, T, ev_

def lsoda_reference(model, ev_, T, w):
    A = model.M[:4, :4]
    by = {}
    for s, k, v in ev_:
        by.setdefault(s, []).append((k, v))
    x = np.zeros(4); rate = 0.0; out = np.zeros(T + 1)
    def apply(s, x, rate):
        for k, v in by.get(s, []):
            if k == 1: x[0] += v
            else: rate = v * w / 60000.0
        return x, rate
    x, rate = apply(0, x, rate); out[0] = x[3]; cur = 0
    for s in sorted(set(by) | {T}):
        if s == 0:
            continue
        sol = solve_ivp(lambda t, y, r=rate: A @ y + np.array([r, 0, 0, 0]), (cur, s), x,
                        t_eval=np.arange(cur, s + 1), method='LSODA', rtol=1e-11, atol=1e-13)
        out[cur:s + 1] = sol.y[3]; x = sol.y[:, -1].copy()
        x, rate = apply(s, x, rate); cur = s
    return out

VAL_SEED, N_HIST = 42, 400
rng_v = np.random.default_rng(VAL_SEED); rows_v = []; tv = time.time()
for i in range(N_HIST):
    age_, w_, h_, sex_, T_, evh = random_history(rng_v)
    rel_ = np.array([e[0] for e in evh]); k_h = np.array([e[1] for e in evh]); v_h = np.array([e[2] for e in evh])
    m_ = ExactEleveld(age_, w_, h_, sex_)
    ce_, _ = simulate_exact(m_, rel_, k_h, v_h, T_, w_)
    ce_step, _ = simulate_exact(m_, rel_, k_h, v_h, T_, w_, loop=True)
    ce_ode = lsoda_reference(m_, evh, T_, w_)
    rows_v.append(dict(history=i, age=age_, weight=w_, height=h_, sex=sex_, duration_s=T_, n_events=len(evh), max_ce=float(ce_.max()),
                       maxdiff_vs_stepwise=float(np.abs(ce_ - ce_step).max()), maxdiff_vs_lsoda=float(np.abs(ce_ - ce_ode).max())))
dv = pd.DataFrame(rows_v); dv.to_csv(OUT / 'engine_validation_random_histories.csv', index=False)
VAL.update(n_histories=N_HIST, seed=VAL_SEED, max_abs_diff_vs_stepwise=float(dv.maxdiff_vs_stepwise.max()),
           max_abs_diff_vs_lsoda=float(dv.maxdiff_vs_lsoda.max()),
           max_rel_diff_vs_lsoda_pct=float(100 * (dv.maxdiff_vs_lsoda / dv.max_ce).max()),
           fallback_procedures=int((~sims.ENGINE_EIG.astype(bool)).sum()))
print(f'{N_HIST} histories in {time.time() - tv:.0f}s | max diff vs stepwise {VAL["max_abs_diff_vs_stepwise"]:.1e} | '
      f'vs LSODA {VAL["max_abs_diff_vs_lsoda"]:.1e} ug/ml | fallback used in {VAL["fallback_procedures"]} procedures')
assert VAL['max_abs_diff_vs_stepwise'] < 1e-8 and VAL['max_abs_diff_vs_lsoda'] < 1e-6, 'Engine validation FAILED'
NUMBERS['engine_validation'] = VAL
(OUT / 'engine_validation_summary.json').write_text(json.dumps(VAL, indent=2))
(OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

# %% [markdown]
# ## 15. Data-handling diagnostics and threshold sensitivity
# Aggregate counts describing how the charted data were handled (units, duplicates, timestamp resolution, same-minute
# stop and bolus, reconstruction start, reasons a case could not be reconstructed), the primary outcome at multiples of
# Ce50 expressed as the model-predicted BIS, results by end-of-reconstruction reason, and the PyTCI-versus-exact
# difference in median Ce across the final cohort.

# %%
def nn(x):
    x = int(x); return '<11' if 0 < x < 11 else x
DIAG = {}
ids_f = set(f.OR_CASE_ID)
e_ = ev.loc[ev.OR_CASE_ID.isin(ids_f), ['OR_CASE_ID', 'TAKEN_TIME', 'UNITS', 'SIG', 'terminal_stop_time']].copy()
e_['TAKEN_TIME'] = pd.to_datetime(e_.TAKEN_TIME)
e_['U'] = e_.UNITS.astype(str).str.upper().str.strip(); e_['V'] = pd.to_numeric(e_.SIG, errors='coerce')
other_ = ~e_.U.isin(['MG', 'MCG/KG/MIN'])
DIAG['rows_other_units'] = dict(rows=nn(other_.sum()), procedures=nn(e_.loc[other_, 'OR_CASE_ID'].nunique()))
dup_ = e_.duplicated(['OR_CASE_ID', 'TAKEN_TIME', 'U', 'V'])
DIAG['exact_duplicate_rows'] = dict(rows=nn(dup_.sum()), procedures=nn(e_.loc[dup_, 'OR_CASE_ID'].nunique()))
DIAG['pct_times_with_nonzero_seconds'] = round(100 * float((e_.TAKEN_TIME.dropna().dt.second != 0).mean()), 3)
pos_ = (e_.V > 0) & ~other_
last_ = e_.loc[pos_].groupby('OR_CASE_ID').TAKEN_TIME.max()
ts_ = f.loc[f.SIM_END_REASON == 'terminal_stop', ['OR_CASE_ID', 'SIM_END_TIME']].copy()
ts_['LAST'] = ts_.OR_CASE_ID.map(last_)
DIAG['final_bolus_same_minute_as_stop'] = nn((ts_.LAST >= ts_.SIM_END_TIME).sum())
first_any_ = e_.groupby('OR_CASE_ID').TAKEN_TIME.min(); first_pos_ = e_.loc[pos_].groupby('OR_CASE_ID').TAKEN_TIME.min()
DIAG['first_entry_precedes_first_dose'] = nn((first_pos_ > first_any_.reindex(first_pos_.index)).sum())
DIAG['reconstruction_skips'] = {str(k): nn(v) for k, v in allrec.SKIP.value_counts().items()} if 'SKIP' in allrec else {}
DIAG['by_end_reason'] = {k: dict(n=int(len(g_)), median_gt_ce50=round(100 * float((g_.MEDIAN_CE > g_.CE50_AGE).mean()), 1))
                         for k, g_ in f.groupby('SIM_END_REASON')}
del e_
def model_bis(r):
    g_ = 1.89 if r <= 1 else 1.47          # Eleveld 2018: Hill coefficient 1.89 below Ce50, 1.47 above
    return 93.0 / (1.0 + r ** g_)
thr = []
for k in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
    ym_ = (f.MEDIAN_CE > k * f.CE50_AGE).to_numpy()
    _, l2, h2 = prop_ci_twoway(ym_, f.CLUSTER, f.PROVIDER)
    thr.append(dict(multiple_of_ce50=k, model_bis=round(model_bis(k), 1), median_gt_pct=100 * ym_.mean(),
                    median_twoway_lo=100 * l2, median_twoway_hi=100 * h2,
                    max_gt_pct=100 * float((f.MAX_CE > k * f.CE50_AGE).mean())))
thr = pd.DataFrame(thr); thr.to_csv(OUT / 'tableS6_threshold_bis.csv', index=False)
print(thr.round(1).to_string(index=False))
if PYTCI_VALUES is not None:
    x_ = f[['OR_CASE_ID', 'MEDIAN_CE'] + [c for c in ['LEGACYEND_MEDIAN_CE'] if c in f]].merge(
        PYTCI_VALUES[['OR_CASE_ID', 'MEDIAN_CE']].rename(columns={'MEDIAN_CE': 'PYT'}), on='OR_CASE_ID')
    ex_ = x_.LEGACYEND_MEDIAN_CE.fillna(x_.MEDIAN_CE) if 'LEGACYEND_MEDIAN_CE' in x_ else x_.MEDIAN_CE
    d_ = 100 * (x_.PYT / ex_ - 1)
    DIAG['pytci_minus_exact_median_ce_pct'] = dict(n=int(len(x_)), median=round(float(d_.median()), 2),
                                                   p05=round(float(d_.quantile(.05)), 2), p95=round(float(d_.quantile(.95)), 2))
print(json.dumps(DIAG, indent=1, default=str))
NUMBERS['diagnostics'] = DIAG
(OUT / 'diagnostics.json').write_text(json.dumps(DIAG, indent=2, default=str))

# %% [markdown]
# ## 16. Provider analysis
# Responsible provider = first CRNA if recorded, otherwise first attending (anonymous codes; names are never output).
# Provider-level summaries use providers with at least 50 procedures. Outputs: coverage by provider type, confidence
# intervals clustered by patient, provider, and both; spread of provider-level exceedance; intraclass correlation and
# shrunken adjusted provider effects for log(median Ce / Ce50); leave-one-provider-out range; modal-rate habits; and the
# adjusted attending-only vs CRNA difference.

# %%
MIN_PROV = int(os.environ.get('PROP_GI_MIN_PROV', '50'))
PROV = {}
fp = f.loc[f.PTYPE != 'None recorded'].copy()
for k_ in sorted(f.PTYPE.unique()):
    g_ = f.loc[f.PTYPE == k_]
    PROV.setdefault('coverage', {})[k_] = dict(procedures=nn(len(g_)), pct=round(100 * len(g_) / len(f), 1),
        providers=(nn(g_.PROVIDER.nunique()) if k_ != 'None recorded' else None),
        median_gt_ce50_pct=round(100 * float((g_.MEDIAN_CE > g_.CE50_AGE).mean()), 1))
for lab_, col_ in [('median', 'MEDIAN_CE'), ('max', 'MAX_CE')]:
    y_ = (fp[col_] > fp.CE50_AGE).to_numpy()
    PROV.setdefault('ci', {})[lab_] = dict(
        patient=[round(100 * v, 2) for v in prop_ci_cluster(y_, fp.CLUSTER)],
        provider=[round(100 * v, 2) for v in prop_ci_cluster(y_, fp.PROVIDER)],
        patient_and_provider=[round(100 * v, 2) for v in prop_ci_twoway(y_, fp.CLUSTER, fp.PROVIDER)])
fp['YM'] = (fp.MEDIAN_CE > fp.CE50_AGE).astype(float)
fp['LR'] = np.log(np.clip(fp.MEDIAN_CE / fp.CE50_AGE, 1e-3, None))       # clip: a few procedures have median Ce = 0
pc_ = fp.groupby('PROVIDER').agg(N=('YM', 'size'), exc=('YM', 'mean'), ptype=('PTYPE', 'first'))
big_ = pc_.loc[pc_.N >= MIN_PROV]
PROV['provider_level'] = dict(min_procedures=MIN_PROV, providers=nn(len(big_)), procedures_covered_pct=round(100 * big_.N.sum() / len(fp), 1),
    providers_by_type={k: nn(v) for k, v in big_.ptype.value_counts().items()},
    exceedance_pct_p10_p25_p50_p75_p90=[round(100 * float(x), 1) for x in big_.exc.quantile([.1, .25, .5, .75, .9])],
    pct_providers_majority_above=round(100 * float((big_.exc > .5).mean()), 1),
    pct_providers_above_25pct=round(100 * float((big_.exc > .25).mean()), 1))
def icc_anova(y, g):
    d = pd.DataFrame({'y': np.asarray(y, float), 'g': np.asarray(g)}); k = d.g.nunique(); N_ = len(d)
    ni = d.groupby('g').size().to_numpy(); gm = d.groupby('g').y.mean(); m_ = d.y.mean()
    msb = (ni * (gm.to_numpy() - m_) ** 2).sum() / (k - 1)
    msw = ((d.y - d.g.map(gm)) ** 2).sum() / (N_ - k)
    n0 = (N_ - (ni ** 2).sum() / N_) / (k - 1)
    s2b = max((msb - msw) / n0, 0.0)
    return s2b / (s2b + msw), s2b, msw
if len(big_) >= 2:
    fb_ = fp.loc[fp.PROVIDER.isin(big_.index)].copy()
    fb_['G_UP'] = (fb_.PROC_GROUP_3 == 'Endoscopy only').astype(float); fb_['G_BOTH'] = (fb_.PROC_GROUP_3 == 'Both').astype(float)
    icc_raw, _, _ = icc_anova(fb_.LR, fb_.PROVIDER)
    mdl_ = SplineOLSExtra(fb_.assign(SEX=fb_.SEX.astype(str)), 'LR', ['G_UP', 'G_BOTH'])
    res_ = fb_.LR.to_numpy() - mdl_.design(fb_.assign(SEX=fb_.SEX.astype(str))) @ mdl_.beta
    icc_adj, s2b_, s2w_ = icc_anova(res_, fb_.PROVIDER)
    eff_ = pd.DataFrame({'r': res_, 'g': fb_.PROVIDER.to_numpy()}).groupby('g').r.agg(['mean', 'size'])
    eb_ = 100 * (np.exp(eff_['mean'] * s2b_ / (s2b_ + s2w_ / eff_['size'])) - 1)
    PROV['provider_variation'] = dict(icc_log_ratio_unadjusted=round(float(icc_raw), 3), icc_log_ratio_adjusted=round(float(icc_adj), 3),
        adjusted_provider_effect_pct_p5_p25_p75_p95=[round(float(x), 1) for x in eb_.quantile([.05, .25, .75, .95])],
        provider_sd_pct=round(100 * (math.exp(math.sqrt(s2b_)) - 1), 1))
S_, N_ = fp.YM.sum(), len(fp)
loo_ = (S_ - pc_.exc * pc_.N) / (N_ - pc_.N)
PROV['leave_one_provider_out_median_gt_ce50_pct'] = [round(100 * float(loo_.min()), 2), round(100 * float(loo_.max()), 2)]
# modal-rate habits (unique modal rates from section 13)
mr_ = u_.reset_index()[['OR_CASE_ID']].merge(top_[['OR_CASE_ID', 'RATE']], on='OR_CASE_ID').merge(fp[['OR_CASE_ID', 'PROVIDER']], on='OR_CASE_ID')
pm_ = mr_.groupby('PROVIDER').agg(N=('RATE', 'size'), top_rate=('RATE', lambda s: s.mode().iloc[0]),
                                  share_top=('RATE', lambda s: s.value_counts(normalize=True).iloc[0]),
                                  anchor=('RATE', lambda s: s.isin([100, 150, 200]).mean()))
pmb_ = pm_.loc[pm_.N >= MIN_PROV]
tr_ = pmb_.top_rate.value_counts()
icc_rate, _, _ = icc_anova(mr_.loc[mr_.PROVIDER.isin(pmb_.index), 'RATE'], mr_.loc[mr_.PROVIDER.isin(pmb_.index), 'PROVIDER'])
PROV['modal_rate_by_provider'] = dict(providers=nn(len(pmb_)),
    providers_most_common_modal_rate_pct={f'{float(k):g}': round(100 * v / len(pmb_), 1) for k, v in tr_.items() if v >= 11},
    share_at_own_most_common_rate_p25_p50_p75=[round(100 * float(x), 1) for x in pmb_.share_top.quantile([.25, .5, .75])],
    anchor_share_p10_p50_p90=[round(100 * float(x), 1) for x in pmb_.anchor.quantile([.1, .5, .9])],
    icc_modal_rate=round(float(icc_rate), 3))
fq_ = fp.assign(SEX=fp.SEX.astype(str), ATT_ONLY=(fp.PTYPE == 'Attending only').astype(float),
                G_UP=(fp.PROC_GROUP_3 == 'Endoscopy only').astype(float), G_BOTH=(fp.PROC_GROUP_3 == 'Both').astype(float))
mq_ = SplineOLSExtra(fq_, 'MEDIAN_CE', ['G_UP', 'G_BOTH', 'ATT_ONLY'])
PROV['attending_only_minus_crna_median_ce_ugml'] = dict(
    patient=[round(float(v), 3) for v in mq_.coef('ATT_ONLY')], patient_and_provider=[round(float(v), 3) for v in mq_.coef('ATT_ONLY', twoway=True)])
print(json.dumps(PROV, indent=1, default=str))
NUMBERS['provider'] = PROV
(OUT / 'provider_analysis.json').write_text(json.dumps(PROV, indent=2, default=str))
(OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

# %% [markdown]
# ## 17. Care setting and resident involvement (sensitivity analyses)
# The primary cohort includes every location. This section repeats the primary outcome after excluding (a) the
# hospital-based procedure unit that serves mainly inpatients, (b) both hospital-based procedure units, and (c)
# procedures with an anesthesia resident (resident minutes > 0), in which either the resident or the attending may have
# chosen the dose. These analyses were added after the distribution of locations was examined (not prespecified).
# Unit names are settings; the outputs contain categories and counts only, never unit names.

# %%
def _units(env, default):
    return [u.strip().upper() for u in os.environ.get(env, default).split(';') if u.strip()]
INPATIENT_UNITS = _units('PROP_GI_INPATIENT_UNITS', '')                          # hospital unit serving mainly inpatients
HOSPITAL_UNITS = sorted(set(_units('PROP_GI_HOSPITAL_UNITS', '')) | set(INPATIENT_UNITS))  # names are site-specific: set both
UNIT_SUFFIX = os.environ.get('PROP_GI_PROCEDURE_UNIT_SUFFIX', ' MPU').upper()    # location names of procedure units end with this
SET, rows_s = {}, []
if SETTING is None:
    print('No location or resident fields in the extract: section 17 skipped')
else:
    assert INPATIENT_UNITS and HOSPITAL_UNITS, 'Set PROP_GI_INPATIENT_UNITS and PROP_GI_HOSPITAL_UNITS (see README, Switches)'
    fs = f[['OR_CASE_ID', 'MEDIAN_CE', 'MAX_CE', 'CE50_AGE', 'CLUSTER', 'PROVIDER', 'PTYPE']].merge(
        SETTING, on='OR_CASE_ID', how='left', validate='one_to_one')
    def _srow(label, keep):
        keep = np.asarray(keep, bool); s_ = fs.loc[keep]
        if len(s_) < 2 or s_.PROVIDER.nunique() < 2 or s_.CLUSTER.nunique() < 2:
            print('  not estimable (too few procedures):', label); return
        y_ = (s_.MEDIAN_CE > s_.CE50_AGE).to_numpy()
        p_, lo_, hi_ = prop_ci_twoway(y_, s_.CLUSTER, s_.PROVIDER)
        rows_s.append(dict(analysis=label, n=int(keep.sum()), median_gt_ce50_pct=100 * p_, twoway_lo=100 * lo_,
                           twoway_hi=100 * hi_, max_gt_ce50_pct=100 * float((s_.MAX_CE > s_.CE50_AGE).mean())))
    def _summ(s_):
        small = len(s_) < 11
        return dict(procedures=nn(len(s_)), pct_of_cohort=round(100 * len(s_) / len(fs), 1),
                    median_gt_ce50_pct=None if small else round(100 * float((s_.MEDIAN_CE > s_.CE50_AGE).mean()), 1))
    res = None
    if 'ANES_RESIDENT_MINUTES' in fs:
        res = (pd.to_numeric(fs.ANES_RESIDENT_MINUTES, errors='coerce').fillna(0) > 0).to_numpy()
        SET['resident'] = dict(procedures=nn(res.sum()), pct_of_cohort=round(100 * float(res.mean()), 1),
                               by_provider_type={str(k): nn(v) for k, v in fs.loc[res, 'PTYPE'].value_counts().items()})
    if 'UCLA_LOC' in fs:
        loc = fs.UCLA_LOC.astype('string').str.upper().str.strip()
        known = loc.notna().to_numpy()
        inpt = loc.isin(INPATIENT_UNITS).to_numpy(); hosp = loc.isin(HOSPITAL_UNITS).to_numpy()
        unit = loc.str.endswith(UNIT_SUFFIX).fillna(False).to_numpy(dtype=bool)
        cat = np.select([~known, inpt, hosp, unit],
                        ['Location not recorded', 'Hospital procedure unit, mainly inpatients',
                         'Hospital procedure unit, mainly outpatients', 'Outpatient procedure unit'],
                        default='Operating room or other location')
        missing_units = [u for u in HOSPITAL_UNITS if not (loc == u).any()]
        if missing_units:
            print(f'*** {len(missing_units)} of the {len(HOSPITAL_UNITS)} configured hospital units were not found among the locations')
        if STRICT:
            assert not missing_units, 'A configured hospital unit name is not present in the extract: check PROP_GI_HOSPITAL_UNITS'
        SET['configured_hospital_units_found'] = len(HOSPITAL_UNITS) - len(missing_units)
        SET['by_setting'] = {}
        for c_ in sorted(set(cat)):
            m_ = cat == c_
            SET['by_setting'][c_] = dict(units=int(loc[m_].nunique()), **_summ(fs.loc[m_]))   # number of locations, not a patient count
            if res is not None:
                SET['by_setting'][c_]['resident_pct'] = None if m_.sum() < 11 else round(100 * float(res[m_].mean()), 1)
        _srow('Excluding the hospital procedure unit serving mainly inpatients', ~inpt)
        _srow('Excluding both hospital-based procedure units', ~hosp)
    if res is not None:
        _srow('Excluding procedures with an anesthesia resident', ~res)
    if rows_s:
        sset = pd.DataFrame(rows_s); sset.to_csv(OUT / 'sensitivity_setting.csv', index=False)
        print(sset.round(1).to_string(index=False))
        SET['sensitivity'] = [dict(analysis=r['analysis'], n=r['n'], median_gt_ce50_pct=round(r['median_gt_ce50_pct'], 1),
                                   ci_twoway=[round(r['twoway_lo'], 1), round(r['twoway_hi'], 1)],
                                   max_gt_ce50_pct=round(r['max_gt_ce50_pct'], 1)) for r in rows_s]
    print(json.dumps({k: v for k, v in SET.items() if k != 'sensitivity'}, indent=1, default=str))
    NUMBERS['setting'] = SET
    (OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
    del fs

# %% [markdown]
# ## 17b. Scheduled anesthesia type (descriptive)
# Distribution of the scheduled (booked) anesthesia type in the analyzed cohort, from an optional case-level file with
# columns `OR_CASE_ID` and `ANES_BOOKING_TYPE` (see README). This is the scheduling label only; no analysis uses it.

# %%
BOOK_PATH = Path(os.environ.get('PROP_GI_BOOKING_PATH', str(IN / 'booking_type_by_case.pkl')))
if not BOOK_PATH.exists():
    print('No scheduled-anesthesia-type file: section 17b skipped')
else:
    def _bkey(s):
        return s.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    bk_ = pd.read_pickle(BOOK_PATH)[['OR_CASE_ID', 'ANES_BOOKING_TYPE']]
    bk_ = bk_.assign(K=_bkey(bk_.OR_CASE_ID)).drop_duplicates('K')
    b_ = pd.DataFrame({'K': _bkey(f.OR_CASE_ID)}).merge(bk_[['K', 'ANES_BOOKING_TYPE']], on='K', how='left')
    cat_ = b_.ANES_BOOKING_TYPE.astype('string').str.upper().str.strip()
    cat_ = cat_.where(~cat_.isin(['', 'N/A', 'NONE']) & cat_.notna(), 'NOT RECORDED')
    vc_ = cat_.value_counts()
    NUMBERS['booking'] = dict(matched=int(b_.ANES_BOOKING_TYPE.notna().sum()), n=int(len(b_)),
                              by_type={k: dict(n=nn(v), pct=None if v < MIN_CELL else round(100 * v / len(b_), 1))
                                       for k, v in vc_.items()})
    print(json.dumps(NUMBERS['booking'], indent=1, default=str))
    (OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

# %% [markdown]
# ## 17c. Results by sex (descriptive)
# Primary and secondary outcomes in female and male patients, with the same two-way clustered CIs. Reads the final cohort
# only; nothing above is changed. Added for the journal's sex-as-a-biological-variable requirement.

# %%
by_sex_ = {}
for s_, g_ in f.groupby(f.SEX.astype(str)):
    ym_ = (g_.MEDIAN_CE > g_.CE50_AGE).to_numpy(); yx_ = (g_.MAX_CE > g_.CE50_AGE).to_numpy()
    pm_, lm_, hm_ = prop_ci_twoway(ym_, g_.CLUSTER, g_.PROVIDER); px_, lx_, hx_ = prop_ci_twoway(yx_, g_.CLUSTER, g_.PROVIDER)
    r_ = g_.MEDIAN_CE / g_.CE50_AGE
    by_sex_[s_] = dict(n=nn(len(g_)), median_gt_ce50=dict(n=nn(ym_.sum()), pct=round(100 * pm_, 1), ci_twoway=[round(100 * lm_, 1), round(100 * hm_, 1)]),
                       max_gt_ce50=dict(n=nn(yx_.sum()), pct=round(100 * px_, 1), ci_twoway=[round(100 * lx_, 1), round(100 * hx_, 1)]),
                       ratio_median_iqr=[round(float(r_.median()), 2), round(float(r_.quantile(0.25)), 2), round(float(r_.quantile(0.75)), 2)])
NUMBERS['by_sex'] = by_sex_
print(json.dumps(by_sex_, indent=1, default=str))
(OUT / 'manuscript_numbers.json').write_text(json.dumps(NUMBERS, indent=2, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

# %% [markdown]
# ## 18. Final reproduction check against the results used in the manuscript
# Compares every number in `manuscript_numbers.json` and every aggregate CSV with `results/stage25/` (plus the Stage 26
# and 27 summaries). Tolerance 1e-9 relative. Stops if anything differs.

# %%
def _flat(o, pfx=''):
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f'{pfx}{k}.'))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            out.update(_flat(v, f'{pfx}{i}.'))
    else:
        out[pfx[:-1]] = o
    return out
def _same(a, b):
    try:
        fa, fb = float(a), float(b)
        return (math.isnan(fa) and math.isnan(fb)) or abs(fa - fb) <= 1e-9 * max(1.0, abs(fb))
    except (TypeError, ValueError):
        return str(a) == str(b)
REF_DIR = ROOT / 'results' / 'stage25'
problems, n_checked = [], 0
if (REF_DIR / 'manuscript_numbers.json').exists():
    old = _flat(json.loads((REF_DIR / 'manuscript_numbers.json').read_text()))
    new = _flat(json.loads((OUT / 'manuscript_numbers.json').read_text()))
    for k, v in old.items():
        if k == 'version':
            continue
        n_checked += 1
        if k not in new or not _same(new[k], v):
            problems.append(f'number {k}: stage25={v} pipeline={new.get(k, "MISSING")}')
    for csv in sorted(REF_DIR.glob('*.csv')):
        if csv.name in ('tableS1_window_pilot.csv',):
            continue                                  # superseded pilot, not produced any more
        n_checked += 1
        if not (OUT / csv.name).exists():
            problems.append(f'file {csv.name}: not produced'); continue
        a_, b_ = pd.read_csv(csv), pd.read_csv(OUT / csv.name)
        if len(a_) != len(b_) or not set(a_.columns) <= set(b_.columns):
            problems.append(f'file {csv.name}: rows or columns differ'); continue
        for c in a_.columns:
            if not all(_same(x, y) for x, y in zip(a_[c], b_[c])):
                problems.append(f'file {csv.name}: column {c} differs'); break
    s26 = REF_DIR / 'stage26_addendum.json'
    if s26.exists():
        for k, v in _flat(json.loads(s26.read_text())).items():
            if k.startswith('G6') or k in ('python', 'numpy', 'scipy', 'pandas', 'statsmodels', 'patsy', 'matplotlib', 'seaborn'):
                continue                              # environment-dependent
            n_checked += 1
            if not _same(_flat(ADD).get(k), v):
                problems.append(f'stage26 {k}: {v} vs {_flat(ADD).get(k)}')
    s27 = ROOT / 'results' / 'stage27_engine_validation' / 'engine_validation_summary.json'
    if s27.exists():
        for k, v in json.loads(s27.read_text()).items():
            n_checked += 1
            if not _same(VAL.get(k), v):
                problems.append(f'stage27 {k}: {v} vs {VAL.get(k)}')
    print(f'Reproduction check: {n_checked} numbers/files compared, {len(problems)} differences')
    for p_ in problems:
        print('  ', p_)
    REPRO['final'] = 'identical' if not problems else f'{len(problems)} differences'
    assert not problems, 'Pipeline does not reproduce the manuscript results: STOP and send this output'
else:
    print('No earlier results/stage25 folder: reproduction check skipped')
(OUT / 'reproduction_checks.json').write_text(json.dumps(REPRO, indent=2))
print('Reproduction checks:', REPRO)
print('PIPELINE COMPLETE. Aggregate outputs are in', OUT)
