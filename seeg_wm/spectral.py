"""Per-subject power spectra per condition and band power (tidy DataFrames).

Output tables (one row per subject x channel x condition x frequency/band)::

    subject  channel  region  hemisphere  condition  freq|band  power  n_epochs
"""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from . import channels as chn
from .config import Config
from .utils import get_logger

log = get_logger()


def _transform(psd: np.ndarray, freqs: np.ndarray, scfg: dict) -> np.ndarray:
    kind = scfg.get("transform", "log10")
    if kind == "log10":
        return np.log10(psd)
    if kind == "db":
        return 10 * np.log10(psd)
    if kind == "relative":
        lo, hi = scfg.get("relative_range", [freqs[0], freqs[-1]])
        m = (freqs >= lo) & (freqs <= hi)
        return psd / psd[..., m].sum(axis=-1, keepdims=True)
    if kind == "none":
        return psd
    raise ValueError(f"Unknown spectral.transform {kind!r}")


def line_noise_mask(freqs: np.ndarray, scfg: dict) -> np.ndarray:
    """True for bins within ±line_noise_width of an excluded line frequency."""
    mask = np.zeros_like(freqs, dtype=bool)
    for f in scfg.get("exclude_line_noise") or []:
        mask |= np.abs(freqs - f) <= scfg.get("line_noise_width", 2.0)
    return mask


def remove_aperiodic(spec: np.ndarray, freqs: np.ndarray, fit_range, exclude: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Subtract a straight-line 1/f fit in log-log space (``spec`` already log-scaled).

    Returns the residual spectrum and the (offset, slope) per channel.
    """
    lf = np.log10(freqs)
    m = (freqs >= fit_range[0]) & (freqs <= fit_range[1]) & ~exclude
    X = np.c_[np.ones(m.sum()), lf[m]]
    flat = spec.reshape(-1, spec.shape[-1])
    coef, *_ = np.linalg.lstsq(X, flat[:, m].T, rcond=None)      # (2, n)
    fit = (np.c_[np.ones_like(lf), lf] @ coef).T                    # (n, n_freqs)
    return (flat - fit).reshape(spec.shape), coef.T.reshape(*spec.shape[:-1], 2)


def compute_condition_spectra(epochs: mne.BaseEpochs, cfg: Config
                              ) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, int]]:
    """PSD averaged across epochs for each configured condition.

    Returns ``({condition: array (n_channels, n_freqs)}, freqs, {condition: n_epochs})``.
    The spectra are transformed (``spectral.transform``), optionally 1/f-corrected
    and line-noise bins are set to NaN.
    """
    scfg = cfg["spectral"]
    method = scfg["method"]
    n_jobs = cfg["project"]["n_jobs"]
    sfreq = epochs.info["sfreq"]
    fmax = min(scfg["fmax"], sfreq / 2)
    windows = {c: (scfg.get("window_per_condition") or {}).get(c, [scfg["tmin"], scfg["tmax"]])
               for c in scfg["conditions"]}
    lengths = {round(w[1] - w[0], 6) for w in windows.values()}
    if len(lengths) > 1:
        raise ValueError(f"spectral windows must all have the same length, got {windows}")
    out, counts, freqs = {}, {}, None
    for cond, selector in scfg["conditions"].items():
        try:
            ep = epochs[selector]
        except KeyError:
            log.warning("Condition %r (%r) not found in epochs - skipped", cond, selector)
            continue
        if len(ep) == 0:
            continue
        tmin, tmax = windows[cond]
        if tmin < ep.tmin - 1e-9 or tmax > ep.tmax + 1e-9:
            raise ValueError(f"window {windows[cond]} for {cond!r} exceeds the epoch "
                             f"limits [{ep.tmin}, {ep.tmax}]")
        kw = dict(method=method, fmin=scfg["fmin"], fmax=fmax, tmin=tmin, tmax=tmax,
                  n_jobs=n_jobs, verbose="error")
        if method == "multitaper":
            kw.update({k: scfg["multitaper"][k] for k in ("bandwidth", "adaptive", "low_bias")})
        elif method == "welch":
            n_samp = int(round((tmax - tmin) * sfreq)) + 1
            n_per_seg = min(scfg["welch"]["n_per_seg"], n_samp)
            kw.update(n_per_seg=n_per_seg,
                      n_fft=scfg["welch"].get("n_fft") or n_per_seg,
                      n_overlap=min(scfg["welch"]["n_overlap"], n_per_seg - 1),
                      average=scfg["welch"].get("average", "mean"))
        spectrum = ep.compute_psd(**kw)
        data = spectrum.get_data()                               # (n_epochs, n_ch, n_freqs)
        avg = np.median(data, 0) if scfg.get("epoch_average") == "median" else data.mean(0)
        freqs = spectrum.freqs
        out[cond] = _transform(avg, freqs, scfg)
        counts[cond] = len(ep)

    if freqs is None:
        raise RuntimeError("No configured condition found in the epochs")
    excl = line_noise_mask(freqs, scfg)
    for cond in out:
        if scfg.get("aperiodic_correction"):
            if scfg.get("transform") not in ("log10", "db"):
                raise ValueError("aperiodic_correction requires transform log10 or db")
            out[cond], _ = remove_aperiodic(out[cond], freqs, scfg["aperiodic_fit_range"], excl)
        out[cond][:, excl] = np.nan
    return out, freqs, counts


def band_power(df_psd: pd.DataFrame, bands: dict[str, list[float]]) -> pd.DataFrame:
    """Average (NaN-aware) the spectrum within each frequency band."""
    keys = [c for c in df_psd.columns if c not in ("freq", "power")]
    parts = []
    for band, (lo, hi) in bands.items():
        sub = df_psd[(df_psd["freq"] >= lo) & (df_psd["freq"] < hi)]
        if sub.empty:
            continue
        g = sub.groupby(keys, dropna=False, observed=True)["power"].mean().reset_index()
        g.insert(len(keys), "band", band)
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["band"] = pd.Categorical(out["band"], categories=list(bands), ordered=True)
    return out


class SpectralAnalyzer:
    """Computes and caches per-subject spectra; assembles the group table."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.regions = cfg.load_regions()

    # ---------------------------------------------------------------- paths
    def _psd_fname(self, subject: str) -> Path:
        return self.cfg.subject_dir(subject, "spectral") / f"sub-{subject}_psd.csv.gz"

    def _band_fname(self, subject: str) -> Path:
        return self.cfg.subject_dir(subject, "spectral") / f"sub-{subject}_bandpower.csv"

    # ------------------------------------------------------------- compute
    def subject_spectra(self, subject: str, epochs: mne.BaseEpochs | None = None,
                        save: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
        subject = str(subject)
        if epochs is None:
            epochs = mne.read_epochs(self.cfg.epochs_fname(subject), preload=True,
                                     verbose="error")
        spectra, freqs, counts = compute_condition_spectra(epochs, self.cfg)
        pr = self.cfg["stats"]["per_region"]
        region_of = chn.assign_regions(epochs.ch_names, self.regions.get(subject, {}),
                                       unassigned=pr.get("unassigned_label", "Unassigned"))
        n_ch, n_f = len(epochs.ch_names), len(freqs)
        rows = []
        for cond, arr in spectra.items():
            rows.append(pd.DataFrame({
                "subject": subject,
                "channel": np.repeat(epochs.ch_names, n_f),
                "region": np.repeat([region_of[c] for c in epochs.ch_names], n_f),
                "hemisphere": np.repeat([chn.hemisphere(c) for c in epochs.ch_names], n_f),
                "condition": cond,
                "n_epochs": counts[cond],
                "freq": np.tile(freqs, n_ch),
                "power": arr.ravel(),
            }))
        df = pd.concat(rows, ignore_index=True)
        bands = band_power(df, self.cfg["spectral"]["bands"])
        if save:
            df.to_csv(self._psd_fname(subject), index=False, compression="gzip")
            bands.to_csv(self._band_fname(subject), index=False)
        log.info("[sub-%s] spectra: %d channels x %d freqs, epochs per condition %s",
                 subject, n_ch, n_f, counts)
        return df, bands

    def run(self, subjects: list[str] | None = None, recompute: bool = False
            ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Spectra for all subjects (cached unless ``recompute``) -> group tables."""
        subjects = subjects or self.cfg.subjects
        psds, bands = [], []
        for s in subjects:
            s = str(s)
            if not recompute and self._psd_fname(s).exists():
                psds.append(pd.read_csv(self._psd_fname(s), dtype={"subject": str}))
                bands.append(pd.read_csv(self._band_fname(s), dtype={"subject": str}))
                continue
            if not self.cfg.epochs_fname(s).exists():
                log.warning("[sub-%s] no epochs file - run preprocessing first; skipped", s)
                continue
            d, b = self.subject_spectra(s)
            psds.append(d)
            bands.append(b)
        if not psds:
            raise RuntimeError("No subject spectra available")
        psd = pd.concat(psds, ignore_index=True)
        band = pd.concat(bands, ignore_index=True)
        band["band"] = pd.Categorical(band["band"], categories=list(self.cfg["spectral"]["bands"]),
                                      ordered=True)
        if self.cfg["stats"]["per_region"].get("merge_hemispheres"):
            for d in (psd, band):
                d["region"] = d["region"].map(chn.merge_hemisphere_label)
        return psd, band
