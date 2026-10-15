"""Figures for QC, spectra and statistics (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import Config
from .stats import subject_average

_DEFAULT_COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


def _color(cfg: Config | None, cond: str, i: int) -> str:
    colors = (cfg or {}).get("plotting", {}).get("condition_colors", {}) if cfg else {}
    return colors.get(cond, _DEFAULT_COLORS[i % len(_DEFAULT_COLORS)])


def savefig(fig, cfg: Config, name: str, kind: str = "group/figures") -> Path:
    p = cfg.group_dir(kind) / f"{name}.{cfg['plotting'].get('save_format', 'png')}"
    fig.savefig(p, dpi=cfg["plotting"].get("dpi", 150), bbox_inches="tight")
    return p


def _ylabel(cfg: Config | None) -> str:
    t = cfg["spectral"].get("transform", "log10") if cfg else "log10"
    lab = {"log10": "log10 power (V²/Hz)", "db": "power (dB)",
           "relative": "relative power", "none": "power (V²/Hz)"}[t]
    if cfg and cfg["spectral"].get("aperiodic_correction"):
        lab += " – 1/f removed"
    return lab


# ------------------------------------------------------------------ QC
def plot_channel_qc(metrics: pd.DataFrame, cfg: Config):
    """Per-contact QC metrics with thresholds; auto-flagged channels in red."""
    acfg = cfg["bad_channels"]["auto"]
    cols = [("std_uv", acfg["flat_std_uv"], "below"), ("variance_z", acfg["variance_z"], "above"),
            ("line_noise_z", acfg["line_noise_z"], "above"),
            ("hf_noise_z", acfg["hf_noise_z"], "above"),
            ("neighbor_corr", acfg["neighbor_corr_min"], "below")]
    n = len(metrics)
    fig, axes = plt.subplots(len(cols), 1, figsize=(max(8, n * 0.12), 2.0 * len(cols)),
                             sharex=True, constrained_layout=True)
    x = np.arange(n)
    c = np.where(metrics["auto_bad"], "tab:red", "tab:gray")
    for ax, (col, thr, side) in zip(axes, cols):
        ax.bar(x, metrics[col], color=c)
        ax.axhline(thr, ls="--", c="k", lw=0.8)
        ax.set_ylabel(col, fontsize=8)
        if col == "std_uv":
            ax.set_yscale("log")
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(metrics["channel"], rotation=90, fontsize=6)
    fig.suptitle("Channel QC (red = automatically flagged)")
    return fig


# ------------------------------------------------------------- spectra
def plot_subject_spectra(psd: pd.DataFrame, cfg: Config, subject: str,
                         conditions: list[str] | None = None, channels: list[str] | None = None):
    """Channel-averaged spectrum per condition for one subject (± SEM across channels)."""
    d = psd[psd["subject"] == str(subject)]
    if channels:
        d = d[d["channel"].isin(channels)]
    conditions = conditions or list(d["condition"].unique())
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for i, cond in enumerate(conditions):
        g = d[d["condition"] == cond].groupby("freq")["power"]
        m, s = g.mean(), g.sem()
        ax.plot(m.index, m.values, color=_color(cfg, cond, i), label=cond)
        ax.fill_between(m.index, m - s, m + s, color=_color(cfg, cond, i), alpha=0.2, lw=0)
    ax.set(xlabel="Frequency (Hz)", ylabel=_ylabel(cfg),
           title=f"sub-{subject} ({d['channel'].nunique()} channels)")
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_group_spectra(psd: pd.DataFrame, cfg: Config, conditions: list[str],
                       region: str | None = None, ax=None, logx: bool = False):
    """Grand-average spectrum (subjects as unit) ± SEM across subjects."""
    d = psd if region in (None, "All") else psd[psd["region"] == region]
    agg = subject_average(d, "freq")
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    else:
        fig = ax.figure
    for i, cond in enumerate(conditions):
        g = agg[agg["condition"] == cond].groupby("freq")["power"]
        m, s = g.mean(), g.sem()
        n = agg[agg["condition"] == cond]["subject"].nunique()
        ax.plot(m.index, m.values, color=_color(cfg, cond, i), label=f"{cond} (n={n})")
        ax.fill_between(m.index, m - s, m + s, color=_color(cfg, cond, i), alpha=0.2, lw=0)
    if logx:
        ax.set_xscale("log")
    ax.set(xlabel="Frequency (Hz)", ylabel=_ylabel(cfg), title=region or "All channels")
    ax.legend(frameon=False, fontsize=8)
    return fig


def plot_contrast_curve(curve: pd.DataFrame, clusters: pd.DataFrame | None = None,
                        title: str = "", cfg: Config | None = None, ax=None):
    """Mean paired difference (a-b) ± SEM with significant frequencies shaded."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
    else:
        fig = ax.figure
    f = curve["freq"].to_numpy()
    ax.plot(f, curve["diff"], c="k", lw=1.2)
    ax.fill_between(f, curve["diff"] - curve["sem"], curve["diff"] + curve["sem"],
                    color="k", alpha=0.15, lw=0)
    ax.axhline(0, c="gray", lw=0.8, ls="--")
    if clusters is not None and len(clusters):
        for _, r in clusters.iterrows():
            ax.axvspan(r["fmin"], r["fmax"], color="tab:red" if r["significant"] else "tab:gray",
                       alpha=0.25 if r["significant"] else 0.08, lw=0)
    if cfg is not None:
        for name, (lo, hi) in cfg["spectral"]["bands"].items():
            ax.axvline(lo, c="gray", lw=0.3)
    n = int(curve["n_subjects"].iloc[0]) if len(curve) else 0
    ax.set(xlabel="Frequency (Hz)", ylabel="Δ power (a − b)", title=f"{title} (n={n})")
    return fig


def plot_band_paired(band: pd.DataFrame, a: str, b: str, region: str | None = None,
                     cfg: Config | None = None, tests: pd.DataFrame | None = None):
    """Per-band paired dot plot (one line per subject)."""
    d = band if region in (None, "All") else band[band["region"] == region]
    agg = subject_average(d, "band")
    bands = list(agg["band"].cat.categories) if hasattr(agg["band"], "cat") \
        else list(agg["band"].unique())
    fig, axes = plt.subplots(1, len(bands), figsize=(2.0 * len(bands), 3.2),
                             constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, bnd in zip(axes, bands):
        piv = agg[agg["band"] == bnd].pivot_table(index="subject", columns="condition",
                                                  values="power")
        if a not in piv or b not in piv:
            ax.set_visible(False)
            continue
        piv = piv[[a, b]].dropna()
        for _, row in piv.iterrows():
            ax.plot([0, 1], row.values, c="gray", alpha=0.6, lw=0.8, marker="o", ms=3)
        ax.plot([0, 1], piv.mean().values, c="k", lw=2.5, marker="o")
        ax.set_xticks([0, 1])
        ax.set_xticklabels([a, b], rotation=30, fontsize=8)
        title = bnd
        if tests is not None and len(tests):
            t = tests[(tests["band"] == bnd)]
            if len(t):
                p = t["p_corrected"].iloc[0]
                title += f"\np={p:.3g}" + (" *" if p < (cfg["stats"]["alpha"] if cfg else .05)
                                           else "")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel(_ylabel(cfg))
    fig.suptitle(f"{a} vs {b} — {region or 'All'}")
    return fig


def plot_region_band_heatmap(band_tests: pd.DataFrame, contrast: str, value: str = "cohen_dz",
                             alpha: float = 0.05, band_order: list[str] | None = None):
    """Regions x bands heat map of an effect metric; * marks corrected significance."""
    d = band_tests[band_tests["contrast"] == contrast]
    piv = d.pivot_table(index="region", columns="band", values=value, observed=False)
    sig = d.pivot_table(index="region", columns="band", values="p_corrected", observed=False)
    if band_order:
        cols = [b for b in band_order if b in piv.columns]
        piv, sig = piv[cols], sig[cols]
    fig, ax = plt.subplots(figsize=(1.1 * piv.shape[1] + 3, 0.5 * piv.shape[0] + 1.5),
                           constrained_layout=True)
    lim = np.nanmax(np.abs(piv.values)) if np.isfinite(piv.values).any() else 1
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=30)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            p = sig.values[i, j]
            if np.isfinite(p) and p < alpha:
                ax.text(j, i, "*", ha="center", va="center", fontsize=14)
    fig.colorbar(im, ax=ax, label=value)
    ax.set_title(f"{contrast}: {value} (* p_corr < {alpha})")
    return fig


def plot_coverage(coverage: pd.DataFrame):
    """Channels per region (rows) and subject (columns)."""
    cov = coverage.set_index("region") if "region" in coverage else coverage
    fig, ax = plt.subplots(figsize=(0.4 * cov.shape[1] + 3, 0.35 * cov.shape[0] + 1.5),
                           constrained_layout=True)
    im = ax.imshow(cov.values, cmap="Greens", aspect="auto")
    ax.set_xticks(range(cov.shape[1]))
    ax.set_xticklabels(cov.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(cov.shape[0]))
    ax.set_yticklabels(cov.index, fontsize=8)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            if cov.values[i, j]:
                ax.text(j, i, int(cov.values[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="# channels")
    ax.set_title("Electrode coverage")
    return fig
