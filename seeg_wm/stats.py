"""Group statistics on spectral power (per frequency, per band, optionally per region).

Unit of observation for the across-subject tests is the **subject**: channels are
averaged within subject (and within region for the per-region analysis), so each
subject contributes one value per condition. The optional mixed model keeps the
channel level (power ~ condition, random intercepts for subject and channel).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .config import Config
from .utils import get_logger

log = get_logger()

ALL_REGIONS = "All"


# ------------------------------------------------------------------ aggregation
def subject_average(df: pd.DataFrame, key: str, by_region: bool = False,
                    min_channels: int = 1) -> pd.DataFrame:
    """Average channels within subject (x region) -> one value per condition x key.

    ``key`` is ``"freq"`` or ``"band"``.
    """
    groups = ["subject", "condition", key] + (["region"] if by_region else [])
    out = (df.groupby(groups, observed=True, dropna=False)
             .agg(power=("power", "mean"), n_channels=("channel", "nunique"))
             .reset_index())
    return out[out["n_channels"] >= min_channels]


def paired_matrix(df_agg: pd.DataFrame, a: str, b: str, key: str
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return (A, B, keys, subjects) with A/B arrays of shape (n_subjects, n_keys)."""
    piv = df_agg.pivot_table(index="subject", columns=["condition", key], values="power",
                             observed=True)
    if a not in piv.columns.get_level_values(0) or b not in piv.columns.get_level_values(0):
        return np.empty((0, 0)), np.empty((0, 0)), np.array([]), []
    A, B = piv[a], piv[b]
    keys = A.columns.intersection(B.columns)
    A, B = A[keys], B[keys]
    ok = A.notna().any(axis=1) & B.notna().any(axis=1)
    return A[ok].to_numpy(), B[ok].to_numpy(), np.asarray(keys), list(A.index[ok])


def _chain_adjacency(valid: np.ndarray) -> sparse.coo_matrix:
    """Adjacency of valid (non-NaN) frequency bins: neighbours only if contiguous."""
    idx = np.where(valid)[0]
    n = len(idx)
    contiguous = np.diff(idx) == 1
    rows = np.r_[np.arange(n - 1)[contiguous], np.arange(1, n)[contiguous]]
    cols = np.r_[np.arange(1, n)[contiguous], np.arange(n - 1)[contiguous]]
    return sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))


def _cohen_dz(d: np.ndarray, axis=0) -> np.ndarray:
    sd = np.nanstd(d, axis=axis, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.nanmean(d, axis=axis) / sd


# ------------------------------------------------------------ per frequency
def frequency_contrast(df_agg: pd.DataFrame, a: str, b: str, method: str = "cluster",
                       alpha: float = 0.05, n_permutations: int = 5000,
                       threshold: float | None = None, tail: int = 0,
                       seed: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired test a-b at every frequency bin across subjects.

    Returns ``(curve, clusters)``: ``curve`` has one row per frequency (mean diff,
    SEM, t, p, significance); ``clusters`` lists frequency clusters (cluster
    method) or contiguous significant runs (fdr method).
    """
    from mne.stats import fdr_correction, permutation_cluster_1samp_test

    A, B, freqs, subjects = paired_matrix(df_agg, a, b, "freq")
    n = len(subjects)
    empty = (pd.DataFrame(), pd.DataFrame())
    if n < 3:
        log.warning("%s vs %s: only %d subjects with both conditions - skipped", a, b, n)
        return empty
    D = A - B
    valid = ~np.isnan(D).any(axis=0)
    t_full = np.full(len(freqs), np.nan)
    p_full = np.full(len(freqs), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        t_val, p_val = stats.ttest_1samp(D[:, valid], 0.0, axis=0)
    t_full[valid], p_full[valid] = t_val, p_val
    sig = np.zeros(len(freqs), dtype=bool)
    cl_rows = []

    if method == "cluster":
        if threshold is None:
            q = 1 - alpha / 2 if tail == 0 else 1 - alpha
            threshold = stats.t.ppf(q, n - 1) * (-1 if tail == -1 else 1)
        t_obs, clusters, cl_p, _ = permutation_cluster_1samp_test(
            D[:, valid], threshold=threshold, n_permutations=n_permutations, tail=tail,
            adjacency=_chain_adjacency(valid), seed=seed, out_type="indices",
            verbose="error")
        vfreqs = freqs[valid]
        vidx = np.where(valid)[0]
        for c, p in zip(clusters, cl_p):
            ii = c[0]
            if p < alpha:
                sig[vidx[ii]] = True
            cl_rows.append({"fmin": vfreqs[ii].min(), "fmax": vfreqs[ii].max(),
                            "n_bins": len(ii), "t_sum": float(t_obs[ii].sum()),
                            "p_cluster": float(p), "significant": bool(p < alpha)})
    elif method == "fdr":
        rej, p_corr = fdr_correction(p_full[valid], alpha=alpha)
        sig[np.where(valid)[0]] = rej
        for on, off in _runs(sig):
            cl_rows.append({"fmin": freqs[on], "fmax": freqs[off - 1], "n_bins": off - on,
                            "t_sum": float(np.nansum(t_full[on:off])),
                            "p_cluster": np.nan, "significant": True})
    else:
        raise ValueError(f"Unknown per_frequency.method {method!r}")

    curve = pd.DataFrame({
        "freq": freqs, "mean_a": np.nanmean(A, 0), "mean_b": np.nanmean(B, 0),
        "diff": np.nanmean(D, 0), "sem": stats.sem(D, axis=0, nan_policy="omit"),
        "t": t_full, "p_uncorrected": p_full, "cohen_dz": _cohen_dz(D),
        "significant": sig, "n_subjects": n})
    return curve, pd.DataFrame(cl_rows)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.where(d == 1)[0], np.where(d == -1)[0]))


# ----------------------------------------------------------------- per band
def band_contrast(df_agg: pd.DataFrame, a: str, b: str, test: str = "ttest") -> pd.DataFrame:
    """Paired test per band (one row per band)."""
    A, B, bands, subjects = paired_matrix(df_agg, a, b, "band")
    rows = []
    for j, band in enumerate(bands):
        ok = ~np.isnan(A[:, j]) & ~np.isnan(B[:, j]) if len(subjects) else []
        x, y = (A[ok, j], B[ok, j]) if len(subjects) else (np.array([]), np.array([]))
        n = len(x)
        stat = p = np.nan
        if n >= 3:
            if test == "ttest":
                stat, p = stats.ttest_rel(x, y)
            elif test == "wilcoxon":
                if np.any(x - y):
                    stat, p = stats.wilcoxon(x, y)
            else:
                raise ValueError(f"Unknown per_band.test {test!r}")
        d = x - y
        rows.append({"band": band, "n_subjects": n, "mean_a": np.mean(x) if n else np.nan,
                     "mean_b": np.mean(y) if n else np.nan,
                     "diff": np.mean(d) if n else np.nan,
                     "cohen_dz": float(_cohen_dz(d)) if n > 1 else np.nan,
                     "statistic": stat, "p": p})
    return pd.DataFrame(rows)


def friedman_omnibus(df_agg: pd.DataFrame, conditions: list[str]) -> pd.DataFrame:
    """Friedman test across >=3 conditions per band (subjects with all conditions)."""
    rows = []
    for band, g in df_agg[df_agg["condition"].isin(conditions)].groupby("band", observed=True):
        piv = g.pivot_table(index="subject", columns="condition", values="power").dropna()
        piv = piv[[c for c in conditions if c in piv.columns]]
        if piv.shape[1] < 3 or len(piv) < 3:
            continue
        chi2, p = stats.friedmanchisquare(*[piv[c].to_numpy() for c in piv.columns])
        k, n = piv.shape[1], len(piv)
        rows.append({"band": band, "n_subjects": n, "conditions": ",".join(piv.columns),
                     "chi2": chi2, "kendall_w": chi2 / (n * (k - 1)), "p": p})
    return pd.DataFrame(rows)


def mixed_model_contrast(df_band: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Channel-level LMM per band: power ~ condition + (1|subject) + (1|subject:channel)."""
    import statsmodels.formula.api as smf
    rows = []
    sub = df_band[df_band["condition"].isin([a, b])].dropna(subset=["power"]).copy()
    sub["is_a"] = (sub["condition"] == a).astype(float)
    sub["channel_id"] = sub["subject"].astype(str) + "_" + sub["channel"].astype(str)
    for band, g in sub.groupby("band", observed=True):
        n_subj = g["subject"].nunique()
        res = {"band": band, "n_subjects": n_subj, "n_channels": g["channel_id"].nunique(),
               "estimate": np.nan, "se": np.nan, "z": np.nan, "p": np.nan, "converged": False}
        if n_subj >= 2 and g["condition"].nunique() == 2:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    fit = smf.mixedlm("power ~ is_a", g, groups=g["subject"],
                                      vc_formula={"channel": "0 + C(channel_id)"}
                                      ).fit(reml=True)
                res.update(estimate=fit.params["is_a"], se=fit.bse["is_a"],
                           z=fit.tvalues["is_a"], p=fit.pvalues["is_a"],
                           converged=bool(fit.converged))
            except Exception as err:  # singular fits etc.
                log.warning("LMM %s vs %s (%s) failed: %s", a, b, band, err)
        rows.append(res)
    return pd.DataFrame(rows)


def correct_pvalues(p: pd.Series, method: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    from statsmodels.stats.multitest import multipletests
    p = p.to_numpy(dtype=float)
    out = np.full_like(p, np.nan)
    rej = np.zeros_like(p, dtype=bool)
    ok = ~np.isnan(p)
    if method == "none" or not ok.any():
        out[ok] = p[ok]
        rej[ok] = p[ok] < alpha
    else:
        rej[ok], out[ok], *_ = multipletests(p[ok], alpha=alpha, method=method)
    return out, rej


# ------------------------------------------------------------- orchestration
class GroupStats:
    """Runs every configured test for the whole cohort and per region."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.scfg = cfg["stats"]
        self.results: dict[str, pd.DataFrame] = {}

    def _region_sets(self, df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
        """[("All", df), (region, df_region), ...] honouring per_region settings."""
        pr = self.scfg["per_region"]
        out = [(ALL_REGIONS, df)]
        if not pr.get("enabled", False):
            return out
        regions = pr.get("regions") or sorted(df["region"].dropna().unique())
        for r in regions:
            if pr.get("drop_unassigned", True) and r == pr.get("unassigned_label"):
                continue
            d = df[df["region"] == r]
            n_subj = d["subject"].nunique()
            if n_subj < pr.get("min_subjects", 3):
                log.info("Region %r: %d subject(s) < min_subjects - skipped", r, n_subj)
                continue
            out.append((r, d))
        return out

    def coverage(self, df: pd.DataFrame) -> pd.DataFrame:
        """Number of channels per subject x region."""
        return (df.groupby(["region", "subject"])["channel"].nunique()
                  .unstack(fill_value=0))

    def run(self, psd: pd.DataFrame, band: pd.DataFrame) -> dict[str, pd.DataFrame]:
        alpha = self.scfg["alpha"]
        minch = self.scfg.get("min_channels_per_subject", 1)
        seed = self.cfg["project"].get("random_seed")
        pf, pb = self.scfg["per_frequency"], self.scfg["per_band"]
        curves, clusters, bands, lmms, omni = [], [], [], [], []

        for region, d_psd in self._region_sets(psd):
            d_band = band[band["region"] == region] if region != ALL_REGIONS else band
            agg_f = subject_average(d_psd, "freq", min_channels=minch)
            agg_b = subject_average(d_band, "band", min_channels=minch)
            for c in self.scfg["contrasts"]:
                tag = {"region": region, "contrast": c["name"], "a": c["a"], "b": c["b"]}
                if pf.get("enabled", True):
                    curve, cl = frequency_contrast(
                        agg_f, c["a"], c["b"], method=pf["method"], alpha=alpha,
                        n_permutations=pf["n_permutations"],
                        threshold=pf.get("cluster_threshold"), tail=pf.get("tail", 0),
                        seed=seed)
                    if len(curve):
                        curves.append(curve.assign(**tag))
                    if len(cl):
                        clusters.append(cl.assign(**tag))
                if pb.get("enabled", True):
                    bands.append(band_contrast(agg_b, c["a"], c["b"], pb["test"]).assign(**tag))
                if self.scfg.get("mixed_model", {}).get("enabled", False):
                    lmms.append(mixed_model_contrast(d_band, c["a"], c["b"]).assign(**tag))
            om = self.scfg.get("omnibus") or {}
            if om.get("enabled", False):
                omni.append(friedman_omnibus(agg_b, om["conditions"]).assign(region=region))

        res = {}
        if curves:
            res["freq_curves"] = pd.concat(curves, ignore_index=True)
        res["freq_clusters"] = pd.concat(clusters, ignore_index=True) if clusters else pd.DataFrame()
        if bands:
            bt = pd.concat(bands, ignore_index=True)
            bt["p_corrected"], bt["significant"] = correct_pvalues(bt["p"], pb["correction"], alpha)
            res["band_tests"] = bt
        if lmms:
            lm = pd.concat(lmms, ignore_index=True)
            lm["p_corrected"], lm["significant"] = correct_pvalues(lm["p"], pb["correction"], alpha)
            res["mixed_models"] = lm
        if omni:
            om = pd.concat(omni, ignore_index=True)
            if len(om):
                om["p_corrected"], om["significant"] = correct_pvalues(om["p"], pb["correction"],
                                                                      alpha)
            res["omnibus"] = om
        res["coverage"] = self.coverage(band).reset_index()
        self.results = res
        return res

    def save(self, outdir=None) -> dict[str, str]:
        outdir = outdir or self.cfg.group_dir("group/stats")
        paths = {}
        for name, df in self.results.items():
            p = outdir / f"{name}.csv"
            df.to_csv(p, index=False)
            paths[name] = str(p)
        log.info("Saved group statistics to %s", outdir)
        return paths
