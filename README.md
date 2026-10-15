# WM_sEEG — preprocessing and spectral analysis of a cognitive task in stereo-EEG

An MNE-Python pipeline for intracranial depth recordings (sEEG) of a working-memory task. It
takes raw recordings to clean epochs, then to per-condition power spectra and group statistics,
both per frequency and optionally per anatomical region.

* **Two notebooks** for the two stages, with interactive review where human judgement is needed
* **One YAML file** (`config/config.yaml`) holds every parameter
* **A small package** (`seeg_wm/`) holds all the logic, so the notebooks stay short and the same
  steps can run in batch from the command line

The legacy emotion-regulation scripts this repository started from are kept in [`legacy/`](legacy/)
for reference. Their per-subject anatomical dictionaries were ported to
[`config/regions.yaml`](config/regions.yaml).

---

## Repository layout

```
├── config/
│   ├── config.yaml          # all parameters (paths, preprocessing, spectral, statistics, plotting)
│   ├── config_demo.yaml     # inherits config.yaml, points to the synthetic demo dataset
│   └── regions.yaml         # subject -> region -> channel names (anatomical labelling)
├── notebooks/
│   ├── 01_preprocessing.ipynb                    # bad channels, re-referencing, bad intervals, epochs
│   └── 02_spectral_analysis_group_stats.ipynb    # PSD, band power, group stats (per freq / per region)
├── seeg_wm/                 # the package
│   ├── config.py            # YAML loading (extends + local override), path helpers
│   ├── io.py                # raw readers, events (annotations / stim / file), saved decisions
│   ├── channels.py          # contact parsing, shafts, bipolar/Laplacian montages, region assignment
│   ├── preprocessing.py     # SEEGPreprocessor class (one subject)
│   ├── spectral.py          # SpectralAnalyzer, PSD per condition, band power, 1/f correction
│   ├── stats.py             # GroupStats: cluster permutation, band tests, Friedman, mixed models
│   ├── plotting.py          # QC, spectra and statistics figures
│   └── utils.py             # logging, robust z-score, interval helpers, JSON
├── scripts/
│   ├── run_pipeline.py      # non-interactive batch runner (preprocess / spectral / all)
│   └── make_demo_data.py    # synthetic sEEG WM dataset with planted effects
├── legacy/                  # original emotion-regulation scripts
├── requirements.txt
└── README.md
```

---

## Installation

```bash
conda create -n wm_seeg python=3.11 -y
```

```bash
conda activate wm_seeg
```

```bash
pip install -r requirements.txt
```

`venv` works as well: `python -m venv .venv && source .venv/bin/activate`. Python ≥ 3.10 is required.
The pipeline was tested with MNE 1.12, NumPy 2.2, SciPy 1.15, pandas 2.3 and statsmodels 0.15.

---

## Quick start with the synthetic demo

Generate a 6-subject synthetic dataset. Hippocampal theta increases during maintenance (more for
load 4), orbitofrontal high gamma increases during encoding, and each subject has one flat contact,
one noisy contact and a few artefacts. Then run the whole pipeline:

```bash
python scripts/make_demo_data.py
```

```bash
python scripts/run_pipeline.py all --config config/config_demo.yaml
```

Results are written to `demo_data/derivatives/`. To use the notebooks with the demo config, set
`CONFIG_PATH = "../config/config_demo.yaml"` in their first cell, or export
`SEEG_CONFIG=../config/config_demo.yaml` before starting Jupyter.

---

## Using your own data

1. **Paths.** Create `config/config.local.yaml` (git-ignored) with your machine-specific settings.
   It is deep-merged over `config.yaml`:
   ```yaml
   paths:
     raw_root: K:/EXP_DATA/iEEG_data
     raw_fname: "{raw_root}/{subject}_upd2.edf"
   subjects: ["5", "6", "10"]
   ```
2. **Events.** Set `events.source`:
   * `annotations`: annotation descriptions equal to the keys (or numeric codes) of `events.event_id`,
     or map them with `events.annotation_map`
   * `stim`: a trigger channel
   * `file`: an MNE `.eve` file or a CSV with `onset`/`sample` and `code`/`trial_type` columns

   `event_id` uses `/`-separated tags (`maintenance/load4`), so `"maintenance"` selects all loads.
   An optional behavioural CSV (`events.metadata_fname`) becomes `epochs.metadata`. Use
   `metadata_query` to keep, for example, correct trials only.
3. **Anatomy.** Fill [`config/regions.yaml`](config/regions.yaml) (subject → region → channels).
   You can list either bipolar names (`HC1-HC2`) or contacts (`HC1`). A bipolar channel inherits the
   region of its contacts. By convention a prime (`HC'`) denotes the left hemisphere.
4. **Conditions and contrasts.** Edit `spectral.conditions` (name → event tag) and `stats.contrasts`.
5. Run **notebook 01** for each subject, then **notebook 02** once for the group.

---

## Notebook 01 — preprocessing

| Step | What happens | Main config keys |
|---|---|---|
| Load | read EDF/FIF/BrainVision, clean names (`EEG HC'3-Ref` → `HC'3`), drop non-neural channels, keep `<shaft><n>` contacts, find events, resample | `data.*`, `events.*` |
| Bad channels | robust z-scores across contacts on the **monopolar, un-notched** signal: flat (MAD std), high variance, line-noise ratio, HF power, low correlation with shaft neighbours. Merged with `bad_channels.manual` and previously saved bads, then reviewed in the MNE browser | `bad_channels.*` |
| Filter | high/low-pass + notch (line frequency and harmonics) | `filtering.*` |
| Re-reference | `bipolar` (adjacent contacts on a shaft, skipping bads), `laplacian`, `shaft_car`, `car`, `none` | `rereference.*` |
| Bad intervals | per-channel robust z of amplitude, gradient and 20–80 Hz envelope (IED-like transients). Time points where ≥ `min_channels` exceed a threshold become `BAD_auto` annotations (padded and merged), then reviewed in the browser | `bad_segments.*` |
| Epochs | `event` mode (`tmin`/`tmax` around each event) or `block` mode (long events cut into fixed windows, as in the legacy movie design). Epochs overlapping `BAD_*` are dropped | `epochs.*` |
| Save | `derivatives/preprocessing/sub-<ID>/`: `-epo.fif`, `bad_channels.json`, `bad_segments.csv`, `channel_qc.csv`, `preproc_report.json` (every step, parameters and MNE version) | — |

**Interactive review.** With `INTERACTIVE = True`, the notebook opens `raw.plot()` with the Qt
backend. Click a channel name to toggle it as bad. Press `a` and drag to add a `BAD_…` annotation.
Close the window to continue. Your decisions are saved, and later runs (including batch runs)
reload them (`bad_channels.use_saved`, `bad_segments.use_saved`).

> Automatic detection is only a proposal. Contacts in the seizure-onset zone, lesions or white
> matter need clinical and anatomical information. Add them to `bad_channels.manual`. Strong
> task-related activity, such as large high-gamma or theta responses, can also inflate the
> variance and HF metrics, so always check the QC plot.

---

## Notebook 02 — spectral analysis and group statistics

**Spectra.** For each subject, channel and condition, a multitaper PSD (or Welch) is computed in
the analysis window `spectral.tmin`–`tmax`, averaged across epochs (mean or median) and transformed
(`log10` | `db` | `relative` | `none`). An optional 1/f correction subtracts a straight-line fit in
log-log space. Line-noise bins are set to NaN. Band power is the mean of the spectrum within each
band of `spectral.bands`. The tables are cached in `derivatives/spectral/sub-<ID>/`.

> All conditions must use analysis windows of the **same length**, so they share one frequency
> grid. Choose windows that do not overlap the next trial phase. For example, in a Sternberg trial
> a 0–2 s fixation window would include encoding.

**Statistics** (`GroupStats`). The subject is the unit of observation. Channels are averaged within
subject, and within region for the per-location analysis.

| Test | Description | Output |
|---|---|---|
| Per frequency | paired differences a−b; 1-D **cluster-based permutation test** (`mne.stats.permutation_cluster_1samp_test`, adjacency broken at masked line-noise bins), or paired t + FDR | `freq_curves.csv`, `freq_clusters.csv` |
| Per band | paired t-test or Wilcoxon, Cohen's d_z; correction (`fdr_bh`/`holm`/`bonferroni`) across bands × contrasts × regions | `band_tests.csv` |
| Omnibus | Friedman test across ≥3 conditions, Kendall's W | `omnibus.csv` |
| Mixed model (optional) | channel level: `power ~ condition + (1|subject) + (1|subject:channel)` (statsmodels) | `mixed_models.csv` |
| Per location (optional) | all of the above per region covered by ≥ `per_region.min_subjects` subjects, with optional merging of hemispheres | `region` column |

Outputs are written to `derivatives/group/stats/*.csv` and `derivatives/group/figures/`: coverage,
grand-average spectra, contrast curves with clusters, paired band plots and region × band heat maps.

> With *n* subjects the smallest attainable permutation p-value is 1/2^(n−1); for n = 6 that is
> 0.031. sEEG coverage is heterogeneous, so check `coverage.csv` before interpreting region-level
> results.

---

## Batch mode

```bash
python scripts/run_pipeline.py preprocess --subjects 5 6
```

```bash
python scripts/run_pipeline.py spectral --recompute
```

Batch preprocessing runs the same steps as notebook 01 without interaction. It reuses any bad
channels or segments saved during an earlier interactive session.

## Using the package directly

```python
from seeg_wm import load_config, SEEGPreprocessor, SpectralAnalyzer, GroupStats

cfg = load_config("config/config.yaml", overrides={"rereference": {"method": "laplacian"}})
SEEGPreprocessor(cfg, "5").run(interactive=False)
psd, band = SpectralAnalyzer(cfg).run()
results = GroupStats(cfg).run(psd, band)
```

## Configuration tips

* `extends: config.yaml` lets a variant config override only a few keys (see `config_demo.yaml`).
* `load_config(..., overrides={...})` changes parameters from a notebook without editing files.
* Every preprocessing report stores the configuration that produced it.
