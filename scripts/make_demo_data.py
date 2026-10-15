"""Generate a small synthetic sEEG working-memory dataset to test the pipeline.

Each subject has a few depth shafts (HC', HC, OR', OR, NA'), a Sternberg-like
trial structure (fixation -> encoding -> maintenance -> probe, loads 2/4),
1/f background, 50 Hz line noise, one flat and one noisy contact, a few
large artefacts, and *planted effects*:

* hippocampal theta (6 Hz) increases during maintenance, more for load 4;
* orbitofrontal high gamma (70–110 Hz) increases during encoding.

Usage::

    python scripts/make_demo_data.py            # writes demo_data/raw + regions
    python scripts/make_demo_data.py --n-subjects 8 --n-trials 40
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mne
import numpy as np
from scipy.signal import butter, sosfiltfilt

ROOT = Path(__file__).resolve().parents[1]

SHAFTS = {"HC'": 8, "HC": 8, "OR'": 10, "OR": 6, "NA'": 5}
REGIONS = {"HC'": "Left Hippocampus", "HC": "Right Hippocampus",
           "OR'": "Left Orbitofrontal", "OR": "Right Orbitofrontal", "NA'": "Left Amygdala"}
CODES = {"fixation": 1, "encoding/load2": 12, "encoding/load4": 14, "maintenance/load2": 22,
         "maintenance/load4": 24, "probe/load2": 32, "probe/load4": 34}


def pink_noise(n_ch, n_times, rng):
    white = rng.standard_normal((n_ch, n_times))
    f = np.fft.rfftfreq(n_times)
    f[0] = f[1]
    return np.fft.irfft(np.fft.rfft(white, axis=1) / np.sqrt(f), n=n_times, axis=1)


def bandlimited(n_times, sfreq, lo, hi, rng):
    sos = butter(4, [lo, hi], btype="band", fs=sfreq, output="sos")
    x = sosfiltfilt(sos, rng.standard_normal(n_times))
    return x / x.std()


def make_subject(subject: int, sfreq: float, n_trials: int, rng, skip_shafts=()):
    names, shaft_of = [], []
    for shaft, n in SHAFTS.items():
        if shaft in skip_shafts:
            continue
        for k in range(1, n + 1):
            names.append(f"{shaft}{k}")
            shaft_of.append(shaft)
    n_ch = len(names)
    trial_len, iti = 5.0, 1.5
    t0 = 5.0
    duration = t0 + n_trials * (trial_len + iti) + 5
    n_times = int(duration * sfreq)
    t = np.arange(n_times) / sfreq

    # 1/f background, spatially smoothed along each shaft (volume conduction)
    data = pink_noise(n_ch, n_times, rng)
    for shaft in set(shaft_of):
        idx = [i for i, s in enumerate(shaft_of) if s == shaft]
        common = pink_noise(1, n_times, rng)[0]
        for j, i in enumerate(idx):
            data[i] = 0.6 * data[i] + 0.8 * common
    data /= data.std(axis=1, keepdims=True)

    onsets, descs = [], []
    theta_env = np.zeros(n_times)
    gamma_env = np.zeros(n_times)
    for k in range(n_trials):
        load = 2 if k % 2 == 0 else 4
        start = t0 + k * (trial_len + iti)
        for name, off in (("fixation", 0.0), (f"encoding/load{load}", 1.0),
                          (f"maintenance/load{load}", 2.0), (f"probe/load{load}", 4.0)):
            onsets.append(start + off)
            descs.append(name)
        s_m, e_m = int((start + 2.0) * sfreq), int((start + 4.0) * sfreq)
        theta_env[s_m:e_m] = 1.0 if load == 2 else 1.8
        s_e, e_e = int((start + 1.0) * sfreq), int((start + 2.0) * sfreq)
        gamma_env[s_e:e_e] = 1.0

    theta = bandlimited(n_times, sfreq, 5, 7, rng)
    gamma = bandlimited(n_times, sfreq, 70, 110, rng)
    gain_subj = rng.uniform(0.7, 1.3)
    # focal sources: amplitude decays along the shaft away from contact 3, so the
    # effect survives bipolar re-referencing (as a real local generator would)
    contact_num = [int("".join(ch for ch in n if ch.isdigit())) for n in names]
    for i, s in enumerate(shaft_of):
        focal = np.exp(-0.5 * ((contact_num[i] - 3) / 1.0) ** 2)
        if s.startswith("HC"):
            data[i] += gain_subj * 1.5 * focal * theta_env * theta
        if s.startswith("OR"):
            data[i] += gain_subj * 0.8 * focal * gamma_env * gamma

    data *= 50e-6                                       # ~50 µV
    data += 15e-6 * np.sin(2 * np.pi * 50 * t)         # line noise

    # bad contacts
    data[names.index("HC'8")] = 1e-8 * rng.standard_normal(n_times)       # flat
    noisy = names.index("OR'10")
    data[noisy] += 400e-6 * rng.standard_normal(n_times)                   # noisy
    data[noisy] += 200e-6 * np.sin(2 * np.pi * 50 * t)
    # artefacts (large transients on several contacts)
    for _ in range(4):
        c = rng.uniform(t0, duration - 5)
        s, e = int(c * sfreq), int((c + 0.3) * sfreq)
        data[: n_ch // 2, s:e] += 2e-3 * np.hanning(e - s)

    ch_names = [f"EEG {n}" for n in names] + ["ECG"]
    ecg = 1e-3 * np.sin(2 * np.pi * 1.2 * t)[None]
    info = mne.create_info(ch_names, sfreq, ch_types="eeg")
    raw = mne.io.RawArray(np.vstack([data, ecg]), info, verbose="error")
    raw.set_meas_date(1_700_000_000 + subject * 86400)
    raw.set_annotations(mne.Annotations(onsets, [0.0] * len(onsets), descs,
                                        orig_time=raw.info["meas_date"]))
    return raw, names, shaft_of


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "demo_data"))
    ap.add_argument("--n-subjects", type=int, default=6)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--sfreq", type=float, default=1024.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lines = ["# Synthetic demo regions (monopolar contact names)"]
    for s in range(1, args.n_subjects + 1):
        skip = ("NA'",) if s % 3 == 0 else ()             # heterogeneous coverage
        raw, names, shaft_of = make_subject(s, args.sfreq, args.n_trials, rng, skip)
        fname = out / "raw" / f"sub-{s:02d}_task-wm_ieeg.edf"
        mne.export.export_raw(fname, raw, fmt="edf", overwrite=True, verbose="error")
        lines.append(f'"{s:02d}":')
        for shaft, region in REGIONS.items():
            chans = [n for n, sh in zip(names, shaft_of) if sh == shaft]
            if chans:
                quoted = ", ".join(f'"{c}"' for c in chans)
                lines.append(f"  {region}: [{quoted}]")
        print(f"wrote {fname}")
    (out / "regions_demo.yaml").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'regions_demo.yaml'}")


if __name__ == "__main__":
    main()
