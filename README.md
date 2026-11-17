# Propofol effect-site concentrations during GI endoscopy: analysis code

This is the analysis code for *Modeled Propofol Effect-site Concentrations during Gastrointestinal Endoscopy: A Retrospective Cohort Study* (Ershoff B).

**No patient data are included.** The study used de-identified electronic health record data from the University of
California, Los Angeles, and those data cannot be shared. The repository shows exactly how every reported number was
computed. Only someone with authorized access to the study data can run it.

## Contents

| File | Purpose |
| --- | --- |
| `PROPOFOL_GI_PIPELINE.ipynb` | The complete analysis, from raw extract to every number, table and figure in the manuscript (Google Colab notebook) |
| `PROPOFOL_GI_PIPELINE.py` | The same code as a plain Python script (cells marked `# %%`) |
| `manuscript_scripts/make_tables_and_flow.py` | Builds the LaTeX tables and the STROBE flow diagram from the aggregate outputs |
| `manuscript_scripts/make_figure2.py` | Draws Figure 2 from the aggregate model predictions |
| `.gitignore` | Keeps data, checkpoints and private review files out of the repository |

## What the pipeline does

| Section | Step |
| --- | --- |
| A | Reads the propofol extract and interprets each entry as a bolus (mg), an infusion rate (µg·kg⁻¹·min⁻¹) or an infusion stop (rate 0). It then screens out implausible or incomplete dosing histories: no nonzero entry, a bolus over 500 mg, a rate over 500, bolus-only totals of 10 mg or less, or infusion-only administration shorter than 5 min. |
| B | Takes case details (age, sex, ASA physical status, BMI, procedure title, anesthesia times) from each case's first record. It also flags bolus-only cases with more than 15 min between the last dose and the end of the record. |
| C | Flags fentanyl, midazolam, ketamine, remifentanil and etomidate in the medication extract. |
| 1–4 | **Exact Eleveld reconstruction.** Uses the published Eleveld (2018) equations with arterial parameters and no opioid covariate. The model is a linear system, so between dosing events it has an exact solution: a sum of exponentials obtained by eigendecomposition. That solution is evaluated for every second, with no numerical time-stepping. Reconstruction runs from the first propofol entry until whichever comes first: a documented terminal stop; 5 min after the last bolus when no infusion is running; or the end of anesthesia when an infusion is still running. |
| 5 | **Cohort.** Applies the age, sex, ASA, BMI and anthropometric-plausibility rules. Classifies each component of the procedure title, applying the author's manual adjudications from a private overrides file. The keyword rules keep endoscopic myotomy (POEM), endoscopic sleeve gastroplasty and fundoplication; the overrides exclude them, so the published cohort needs the overrides file. |
| 6–12 | **Analyses.** Proportion of procedures whose median or maximum effect-site concentration exceeds the age-adjusted Ce50 and the BIS-60 concentration, with cluster-robust confidence intervals (by patient, and two-way by patient and responsible provider; the two-way intervals are primary). Also: spline models of concentration by age; interindividual variability in Ce50 (with a Gaussian-copula dependence analysis); time-weighted modal infusion rates; the adjunct-drug sensitivity analysis; and all sensitivity analyses in Table 2. |
| 13 | Records software versions; checks the regression code (G6) and the two-way patient-and-provider intervals (G7) against statsmodels; summarizes procedures per provider; computes bounds for tied modal rates. |
| 14 | **Engine validation.** Compares the engine with step-by-step matrix exponentials and with SciPy LSODA on 400 random dosing histories. Also counts the procedures that needed the numerical fallback. |
| 15 | **Data-handling diagnostics.** Unrecognized units, duplicate rows, timestamp resolution, final boluses charted in the same minute as the stop, reasons a case could not be reconstructed, results by end-of-reconstruction reason, the primary outcome at 0.7–1.3 × Ce50 with the corresponding model-predicted BIS, and the PyTCI-versus-exact difference across the final cohort. |
| 16 | **Provider analysis.** The responsible provider is the first CRNA, or the attending when no CRNA is recorded (anonymous codes; names are never output). Reports confidence intervals clustered by patient, by provider and by both; the spread of exceedance across providers; intraclass correlation and shrunken adjusted provider effects; a leave-one-provider-out range; modal-rate habits by provider; and the adjusted attending-only versus CRNA difference. |
| 17 | **Care setting and resident involvement.** Repeats the primary outcome without the hospital-based procedure unit that serves mainly inpatients, without both hospital-based procedure units, and without procedures that had an anesthesia resident. Also counts procedures by setting category. Unit names are settings; outputs contain categories and counts only. |
| 17b | **Scheduled anesthesia type.** Distribution of the booked anesthesia type in the analyzed cohort (descriptive only). |
| 17c | **Results by sex.** Primary and secondary outcomes in female and male patients, with two-way clustered CIs (descriptive). |
| 18 | Compares every result with the results used in the manuscript. |

## Inputs

All inputs are restricted. They stay in the project folder and never go into this repository.

| Path in the project folder | Required? | Content |
| --- | --- | --- |
| `inputs/propall.pkl` | Yes | SQL extract of propofol administrations, with the first CRNA and first attending for each case (`FIRST_CRNA_NAME`, `FIRST_ATT_NAME`; used only as anonymous codes), the care location (`UCLA_LOC`) and anesthesia-resident minutes (`ANES_RESIDENT_MINUTES`; both used only in section 17). Covers anesthesia cases whose title contains "colonoscopy" or "endoscopy" and that have no documented airway event (the event that records tracheal intubation or laryngeal mask placement). ERCP, EUS and enteroscopy were excluded at extraction. |
| `inputs/endo_selected_meds_raw.pkl` | For the adjunct analysis | Medication extract. Its location can be changed with `PROP_GI_MEDS_RAW_PATH`. |
| `results/private_procedure_review/procedure_title_overrides_PRIVATE.csv` | Yes, to reproduce the cohort | The author's procedure-title adjudications, with columns `IM_PROC_NAME` and `ACTION` (KEEP or EXCLUDE). |
| `inputs/booking_type_by_case.pkl` | For section 17b | One row per case: `OR_CASE_ID`, `ANES_BOOKING_TYPE` (the scheduled anesthesia type from the case scheduling table). Its location can be changed with `PROP_GI_BOOKING_PATH`. |
| `checkpoints/summary_custom_end_v1_full.pkl` | Optional | The saved PyTCI 1.1 run from the earlier analysis. See "The PyTCI comparison" below. |

## Running it

1. Open `PROPOFOL_GI_PIPELINE.ipynb` in Google Colab.
2. Point it at the project folder, either by setting `PROP_GI_ROOT` or by editing `ROOT` in the setup cell.
3. Choose **Runtime → Run all**.

A first full run takes about 30–60 minutes. The reconstruction saves its progress in batches, so a rerun after an interruption
continues where it stopped. The saved batches are tied to a fingerprint of the input data and the reconstruction code; if
either changes, the old batches are deleted automatically and the reconstruction starts again.

**Outputs.** Aggregate results go to `results/pipeline/`: tables, figures, `manuscript_numbers.json`,
`reproduction_checks.json` and the validation files. Restricted case-level checkpoints go to `checkpoints/pipeline_*`
and must never be shared. Nothing produced by earlier analyses is modified.

### Switches

You can change these at the top of the setup cell, or set them before running with, for example,
`import os; os.environ['PROP_GI_RUN_PYTCI_FULL'] = '1'`.

| Switch | Default | Effect |
| --- | --- | --- |
| `PROP_GI_ROOT` | the author's Drive folder | Project folder containing `inputs/`, `checkpoints/` and `results/` |
| `PROP_GI_STRICT` | `1` | Stops unless the counts match the analyzed data: 222,936 extracted cases, 222,745 after screening, 214,625 reconstructed, 209,406 before the final exclusions. Set to `0` for any other dataset. |
| `PROP_GI_RUN_PYTCI_FULL` | `0` | Set to `1` to rerun PyTCI 1.1 for every procedure. This takes hours. See below. |
| `PROP_GI_MEDS_RAW_PATH` | `inputs/endo_selected_meds_raw.pkl` | Location of the medication extract |
| `PROP_GI_BOOKING_PATH` | `inputs/booking_type_by_case.pkl` | Location of the scheduled-anesthesia-type file (section 17b) |
| `PROP_GI_MIN_PROV` | `50` | Minimum procedures for a provider to enter the provider-level summaries |
| `PROP_GI_INPATIENT_UNITS` | none | Location name(s), separated by `;`, of the hospital procedure unit serving mainly inpatients (section 17). Required when the extract has location fields. |
| `PROP_GI_HOSPITAL_UNITS` | none | Location names of all hospital-based procedure units, separated by `;` (section 17). Required when the extract has location fields. |
| `PROP_GI_PROCEDURE_UNIT_SUFFIX` | ` MPU` | Ending that identifies procedure-unit locations (section 17) |

## The PyTCI comparison

An earlier version of this analysis reconstructed concentrations with the PyTCI library (version 1.1). That version
implements the Eleveld intercompartmental clearance Q2 differently from the published equation, and it steps the model
forward in 1-s Euler steps. The final analysis uses the exact solution of the published equations. The earlier
reconstruction appears in only one place: a single sensitivity row in Table 2, "PyTCI 1.1 software, earlier end rule."

That row can come from one of two sources:

* **The saved PyTCI run (default).** If `checkpoints/summary_custom_end_v1_full.pkl` is present, the row uses its values.
  First, gate G4 recomputes 300 randomly chosen procedures with PyTCI through this notebook's event handling. They must
  match the saved run within 10⁻⁸ µg/ml.
* **A full rerun.** With `PROP_GI_RUN_PYTCI_FULL=1`, every procedure in the final cohort is recomputed with PyTCI 1.1,
  using the earlier end rule. The rerun is resumable, in batches of 5,000. If the saved run is also present, the two
  are compared case by case.

If neither source is available, that row is left out and every other result is still produced.

## Checks

Every check stops the run if it fails. The **Reference** column shows what a check compares against. Checks that depend
on earlier files run only when those files are present in the project folder, and are skipped otherwise.

| Check | What it verifies | Reference |
| --- | --- | --- |
| A | The cleaned event stream is identical, row for row, to the earlier event-preparation step | `checkpoints/propsig_clean_v1_full.pkl` |
| B1 | The same cases are excluded by the bolus-only 15-min rule | `checkpoints/ghost_case_ids.pkl` |
| B2 | Case details (age, sex, ASA, BMI, title, anesthesia times) are identical for every case | the saved PyTCI run |
| C | The adjunct flags are identical | `checkpoints/case_level_selected_drug_flags.pkl` |
| G1 | The exact engine agrees with iTIVA, an independent Eleveld implementation, within 1% on two synthetic dosing histories (observed: 0.47%) | built in |
| G2 | The same set of cases is reconstructed | the saved PyTCI run |
| G3 | Recomputing the earlier end rule reproduces the saved end time of every case. Any mismatch must be explained by entries that share a timestamp. | the saved PyTCI run |
| G4 | PyTCI, run through this notebook's event handling, reproduces the saved concentrations within 10⁻⁸ µg/ml in 300 random procedures | the saved PyTCI run |
| G5 | The cohort before the anthropometric and procedure-scope exclusions is the same 209,406 procedures | `checkpoints/primary_modeling_cohort_209406.pkl` |
| G6 | The spline regression and cluster-robust code agree with statsmodels within 10⁻⁶ | built in (needs statsmodels) |
| G7 | The two-way (patient and provider) intervals for the primary proportions and the Figure 2 curves agree with statsmodels within 10⁻⁶ | built in (needs statsmodels) |
| Engine validation | Over 400 random dosing histories, the exact solution agrees with step-by-step matrix exponentials within 10⁻⁸ µg/ml and with SciPy LSODA (rtol 10⁻¹¹, atol 10⁻¹³) within 10⁻⁶ µg/ml (observed: 2.7 × 10⁻¹³ and 2.3 × 10⁻¹⁰) | built in |
| Final | Every number in `manuscript_numbers.json` and every aggregate CSV matches the results reported in the manuscript, within a relative tolerance of 10⁻⁹ | `results/stage25/` |

The outcome of each check is written to `results/pipeline/reproduction_checks.json`.

### What runs in which situation

| Project folder contains | Result |
| --- | --- |
| Inputs and the earlier checkpoints and results (the author's environment) | Full analysis, every check, and the PyTCI row from the saved run |
| Inputs only | Full analysis and the built-in checks (G1, G6, G7, engine validation). The PyTCI row needs `PROP_GI_RUN_PYTCI_FULL=1`. |

## Privacy

`inputs/`, `checkpoints/`, the procedure-title review sheet and the overrides file contain restricted data, and
`.gitignore` excludes them. Most aggregate outputs suppress cells with fewer than 11 procedures, but some
diagnostic counts in `results/pipeline` are not suppressed. The results folder is therefore not part of this repository;
check any output for small cells before sharing it.

## Software

Tested with Python 3.13, NumPy 2.4, SciPy 1.16, pandas 2.2, statsmodels 0.15 and matplotlib 3.10, plus PyTCI 1.1 for
gate G4 and the optional full PyTCI rerun.
