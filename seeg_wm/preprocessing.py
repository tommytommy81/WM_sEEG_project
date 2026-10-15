"""Single-subject sEEG preprocessing: the :class:`SEEGPreprocessor` class.

Recommended order (as in ``notebooks/01_preprocessing.ipynb``)::

    pp = SEEGPreprocessor(cfg, subject)
    pp.load()                    # read, clean names, set types, find events, resample
    pp.detect_bad_channels()     # automatic proposal (before notch filtering!)
    pp.mark_bad_channels()       # auto + manual (config) + saved
    pp.plot_raw()                # optional interactive review (click channel names)
    pp.filter()                  # high-/low-pass + notch
    pp.rereference()             # bipolar / laplacian / shaft_car / car / none
    pp.detect_bad_segments()     # automatic BAD_auto annotations
    pp.plot_raw()                # optional interactive review (drag = annotate)
    pp.make_epochs()
    pp.save()
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

import mne
import numpy as np
import pandas as pd
from scipy.signal import hilbert

from . import channels as chn
from . import io
from .config import Config
from .utils import get_logger, mask_to_intervals, merge_intervals, robust_zscore, save_json

log = get_logger()


class SEEGPreprocessor:
    """Stateful preprocessing of one subject driven by the YAML config.

    Attributes
    ----------
    raw : mne.io.Raw          current (in-place modified) recording
    events, event_id          events found at load time (sample indices follow ``raw``)
    bad_channel_metrics : pd.DataFrame  output of :meth:`detect_bad_channels`
    epochs : mne.Epochs       after :meth:`make_epochs`
    history : list[dict]      provenance of every step (saved in the report)
    """

    def __init__(self, cfg: Config, subject: str):
        self.cfg = cfg
        self.subject = str(subject)
        self.raw: mne.io.BaseRaw | None = None
        self.events: np.ndarray | None = None
        self.event_id: dict[str, int] = {}
        self.bad_channel_metrics: pd.DataFrame | None = None
        self.auto_bads: dict[str, str] = {}
        self.epochs: mne.Epochs | None = None
        self.reref_method: str | None = None
        self.history: list[dict] = []
        log.setLevel(cfg["project"].get("log_level", "INFO"))

    # ------------------------------------------------------------------ utils
    def _log_step(self, step: str, **info):
        self.history.append({"step": step, **info})
        log.info("[sub-%s] %s %s", self.subject, step,
                 {k: v for k, v in info.items() if not isinstance(v, (list, dict)) or len(v) < 8})

    @property
    def seeg_names(self) -> list[str]:
        picks = mne.pick_types(self.raw.info, seeg=True, exclude=[])
        return [self.raw.ch_names[i] for i in picks]

    # ------------------------------------------------------------------- load
    def load(self) -> mne.io.BaseRaw:
        """Read the raw file, clean channel names, set types, find events, resample."""
        dcfg = self.cfg["data"]
        fname = self.cfg.raw_fname(self.subject)
        raw = io.read_raw(fname, reader=dcfg.get("reader", "auto"))
        n_orig = len(raw.ch_names)

        # 1) clean names (keep originals unique)
        mapping, seen = {}, set()
        for ch in raw.ch_names:
            new = chn.clean_name(ch, dcfg.get("name_strip_patterns", []))
            if new in seen or not new:
                new = ch
            seen.add(new)
            mapping[ch] = new
        raw.rename_channels(mapping)

        # 2) drop non-neural channels
        drop = [ch for ch in raw.ch_names
                if any(re.search(p, ch) for p in dcfg.get("drop_channels_regex", []))]
        stim_ch = self.cfg["events"].get("stim_channel")
        if self.cfg["events"]["source"] == "stim":
            drop = [c for c in drop if c != stim_ch]

        # 3) channel types
        types = {ch: dcfg.get("default_channel_type", "seeg") for ch in raw.ch_names
                 if ch not in drop and mne.channel_type(raw.info, raw.ch_names.index(ch))
                 in ("eeg", "seeg", "ecog", "misc")}
        types.update({k: v for k, v in (dcfg.get("channel_types") or {}).items()
                      if k in raw.ch_names})
        raw.set_channel_types(types, on_unit_change="ignore", verbose="error")

        if dcfg.get("keep_only_parsable_contacts", True):
            unparsable = [ch for ch, t in types.items()
                          if t == "seeg" and chn.parse_contact(ch) is None]
            if unparsable:
                log.info("Dropping %d unparsable channel names: %s",
                         len(unparsable), unparsable[:15])
            drop += unparsable

        # events must be read before dropping a stim channel / resampling
        annot_fname = self.cfg.annotations_fname(self.subject)
        if annot_fname is not None and annot_fname.exists():
            raw.set_annotations(raw.annotations + io.read_annotations_file(annot_fname))
        self.raw = raw
        self.events, self.event_id = io.find_events(raw, self.cfg, self.subject)

        drop = sorted(set(drop) | ({stim_ch} if stim_ch in raw.ch_names else set()))
        raw.drop_channels(drop)

        sfreq = dcfg.get("resample_sfreq")
        if sfreq and sfreq < raw.info["sfreq"]:
            raw, self.events = raw.resample(sfreq, events=self.events,
                                            n_jobs=self.cfg["project"]["n_jobs"],
                                            verbose="error")
        self.raw = raw
        self._log_step("load", fname=str(fname), n_channels_orig=n_orig,
                       n_seeg=len(self.seeg_names), dropped=drop,
                       sfreq=raw.info["sfreq"], duration_s=round(raw.times[-1], 2),
                       n_events=len(self.events))
        return raw

    # ------------------------------------------------------------- filtering
    def filter(self) -> mne.io.BaseRaw:
        fcfg = self.cfg["filtering"]
        nyq = self.raw.info["sfreq"] / 2
        h_freq = fcfg.get("h_freq")
        if h_freq is not None and h_freq >= nyq:
            log.warning("h_freq=%s >= Nyquist (%s); low-pass skipped", h_freq, nyq)
            h_freq = None
        n_jobs = self.cfg["project"]["n_jobs"]
        if fcfg.get("l_freq") or h_freq:
            self.raw.filter(fcfg.get("l_freq"), h_freq, picks="seeg", n_jobs=n_jobs,
                            verbose="error")
        notch = (fcfg.get("notch") or {})
        freqs = [f for f in (notch.get("freqs") or []) if f < nyq]
        if freqs:
            self.raw.notch_filter(freqs, picks="seeg", method=notch.get("method", "fir"),
                                  notch_widths=notch.get("notch_widths"), n_jobs=n_jobs,
                                  verbose="error")
        self._log_step("filter", l_freq=fcfg.get("l_freq"), h_freq=h_freq, notch=freqs)
        return self.raw

    # ---------------------------------------------------------- bad channels
    def detect_bad_channels(self) -> pd.DataFrame:
        """Compute per-contact quality metrics and flag outliers (monopolar data).

        Run *before* notch filtering so that the line-noise metric is meaningful.
        """
        acfg = self.cfg["bad_channels"]["auto"]
        names = self.seeg_names
        raw = self.raw.copy().pick(names)
        if acfg.get("crop_seconds"):
            raw.crop(0, min(acfg["crop_seconds"], raw.times[-1]))
        sfreq = raw.info["sfreq"]
        data_uv = raw.get_data() * 1e6

        # robust (MAD-based) spread: insensitive to transient artefacts / IEDs
        def mad_std(x):
            return 1.4826 * np.median(np.abs(x - np.median(x, axis=1, keepdims=True)), axis=1)

        std = mad_std(data_uv)
        hp = mne.filter.filter_data(data_uv, sfreq, 1.0, None, verbose="error")
        log_var = np.log(mad_std(hp) ** 2 + 1e-12)

        psd, freqs = mne.time_frequency.psd_array_welch(
            data_uv, sfreq, fmin=1, fmax=sfreq / 2, n_fft=int(min(sfreq * 2, data_uv.shape[1])),
            verbose="error")
        total = psd.sum(axis=1) + 1e-20
        line_freqs = [f for f in (self.cfg["filtering"]["notch"].get("freqs") or [])
                      if f < sfreq / 2]
        line_mask = np.zeros_like(freqs, dtype=bool)
        for f in line_freqs:
            line_mask |= np.abs(freqs - f) <= 1.0
        line_ratio = psd[:, line_mask].sum(axis=1) / total if line_mask.any() \
            else np.zeros(len(names))
        lo, hi = acfg.get("hf_band", [100, 200])
        hi = min(hi, sfreq / 2 - 1)
        hf_mask = (freqs >= lo) & (freqs <= hi) & ~line_mask
        hf_pow = np.log(psd[:, hf_mask].mean(axis=1) + 1e-20) if hf_mask.any() \
            else np.zeros(len(names))
        neigh = chn.neighbor_correlation(hp, names)

        df = pd.DataFrame({
            "channel": names,
            "shaft": [(chn.parse_contact(c) or ("?", 0))[0] for c in names],
            "std_uv": std,
            "variance_z": robust_zscore(log_var),
            "line_noise_z": robust_zscore(line_ratio),
            "hf_noise_z": robust_zscore(hf_pow),
            "neighbor_corr": neigh,
        })
        reasons = {}
        for _, row in df.iterrows():
            r = []
            if row.std_uv < acfg["flat_std_uv"]:
                r.append("flat")
            if row.variance_z > acfg["variance_z"]:
                r.append("high_variance")
            if row.line_noise_z > acfg["line_noise_z"]:
                r.append("line_noise")
            if row.hf_noise_z > acfg["hf_noise_z"]:
                r.append("hf_noise")
            if np.isfinite(row.neighbor_corr) and row.neighbor_corr < acfg["neighbor_corr_min"]:
                r.append("low_neighbor_corr")
            if r:
                reasons[row.channel] = ",".join(r)
        df["reason"] = df["channel"].map(reasons).fillna("")
        df["auto_bad"] = df["reason"] != ""
        self.bad_channel_metrics = df
        self.auto_bads = reasons
        self._log_step("detect_bad_channels", n_auto_bads=len(reasons), auto_bads=reasons)
        return df

    def mark_bad_channels(self, extra: list[str] | None = None, use_auto: bool = True,
                          use_manual: bool = True, use_saved: bool | None = None) -> list[str]:
        """Set ``raw.info['bads']`` = auto ∪ manual (config) ∪ saved ∪ extra."""
        bcfg = self.cfg["bad_channels"]
        use_saved = bcfg.get("use_saved", True) if use_saved is None else use_saved
        bads = set(self.raw.info["bads"])
        if use_auto and bcfg["auto"].get("enabled", True):
            bads |= set(self.auto_bads)
        if use_manual:
            bads |= set((bcfg.get("manual") or {}).get(self.subject, []) or [])
        if use_saved:
            bads |= set(io.load_saved_bads(self.cfg, self.subject))
        bads |= set(extra or [])
        bads &= set(self.raw.ch_names)
        self.raw.info["bads"] = sorted(bads)
        self._log_step("mark_bad_channels", bads=self.raw.info["bads"])
        return self.raw.info["bads"]

    # ---------------------------------------------------------- interactive
    def plot_raw(self, block: bool = True, **kwargs):
        """Open the MNE raw browser (click channel names = bad, drag = annotate).

        In Jupyter use ``%matplotlib qt`` (or install ``mne-qt-browser``) for an
        interactive window. Changes are applied to ``self.raw`` in place.
        """
        pcfg = self.cfg["plotting"]["raw_browser"]
        params = dict(duration=pcfg["duration"], n_channels=pcfg["n_channels"],
                      scalings=pcfg["scalings"], block=block,
                      events=self.events if self.events is not None and len(self.events) else None,
                      event_id=self.event_id or None)
        params.update(kwargs)
        fig = self.raw.plot(**params)
        self._log_step("plot_raw", bads=self.raw.info["bads"],
                       n_bad_annotations=int(sum(d.upper().startswith("BAD")
                                                 for d in self.raw.annotations.description)))
        return fig

    # --------------------------------------------------------- re-reference
    def rereference(self, method: str | None = None) -> mne.io.BaseRaw:
        rcfg = self.cfg["rereference"]
        method = method or rcfg["method"]
        bads = list(self.raw.info["bads"])
        names = self.seeg_names
        good = [c for c in names if c not in bads]

        if method == "bipolar":
            anodes, cathodes = chn.bipolar_pairs(names, bads, rcfg["bipolar"]["max_gap"])
            if not anodes:
                raise RuntimeError("No valid bipolar pairs (check contact names / bads)")
            tpl = rcfg.get("bipolar_name_template", "{anode}-{cathode}")
            new_names = [tpl.format(anode=a, cathode=c) for a, c in zip(anodes, cathodes)]
            raw = mne.set_bipolar_reference(self.raw, anodes, cathodes, ch_name=new_names,
                                            drop_refs=True, copy=True, verbose="error")
            raw.info["bads"] = []
            if rcfg["bipolar"].get("drop_monopolar", True):
                leftover = [c for c in raw.ch_names if c in names]
                raw.drop_channels(leftover)
            info = dict(n_pairs=len(new_names))
        elif method in ("laplacian", "shaft_car"):
            raw = self.raw.copy().pick(good)
            data = raw.get_data()
            idx = {c: i for i, c in enumerate(good)}
            new = np.empty_like(data)
            keep = []
            if method == "laplacian":
                neigh = chn.laplacian_neighbors(good)
                for c, nn in neigh.items():
                    new[idx[c]] = data[idx[c]] - data[[idx[n] for n in nn]].mean(axis=0)
                    keep.append(c)
            else:
                for shaft, contacts in chn.group_by_shaft(good).items():
                    ii = [idx[c] for _, c in contacts]
                    new[ii] = data[ii] - data[ii].mean(axis=0, keepdims=True)
                    keep += [c for _, c in contacts] if len(ii) > 1 else []
            raw = mne.io.RawArray(new, raw.info, first_samp=raw.first_samp, verbose="error")
            raw.set_annotations(self.raw.annotations)
            raw.pick(keep)
            info = dict(n_channels=len(keep))
        elif method == "car":
            raw = self.raw.copy()
            raw.set_eeg_reference("average", ch_type="seeg", verbose="error")
            raw.drop_channels(bads)
            info = dict(n_channels=len(raw.ch_names))
        elif method == "none":
            raw = self.raw.copy().drop_channels(bads)
            info = dict(n_channels=len(raw.ch_names))
        else:
            raise ValueError(f"Unknown rereference method {method!r}")

        self.monopolar_bads = bads
        self.raw = raw
        self.reref_method = method
        self._log_step("rereference", method=method, **info)
        return raw

    # --------------------------------------------------------- bad segments
    def _make_annotations(self, intervals, description) -> mne.Annotations:
        meas_date = self.raw.info["meas_date"]
        onsets = np.array([a for a, _ in intervals], dtype=float)
        durations = np.array([b - a for a, b in intervals], dtype=float)
        if meas_date is None:  # onsets relative to first sample
            return mne.Annotations(onsets, durations, [description] * len(onsets))
        return mne.Annotations(onsets + self.raw.first_time, durations,
                               [description] * len(onsets), orig_time=meas_date)

    def detect_bad_segments(self) -> mne.Annotations:
        """Flag artefactual time points (amplitude, gradient, spike-band envelope)."""
        acfg = self.cfg["bad_segments"]["auto"]
        desc = acfg.get("description", "BAD_auto")
        # idempotent: remove previous automatic annotations
        keep = [i for i, d in enumerate(self.raw.annotations.description) if d != desc]
        self.raw.set_annotations(self.raw.annotations[keep])

        sfreq = self.raw.info["sfreq"]
        picks = mne.pick_types(self.raw.info, seeg=True, exclude="bads")
        count = np.zeros(self.raw.n_times, dtype=np.int16)
        env_cfg = acfg.get("hf_envelope") or {}
        do_env = env_cfg.get("enabled", False)
        if do_env:
            lo, hi = env_cfg["band"]
            hi = min(hi, sfreq / 2 - 1)
        per_criterion = {"amplitude": 0, "gradient": 0, "hf_envelope": 0}
        for p in picks:
            x = self.raw.get_data(picks=[p])[0]
            bad = np.zeros_like(x, dtype=bool)
            m = robust_zscore(np.abs(x)) > acfg["amplitude_z"]
            per_criterion["amplitude"] += int(m.sum()); bad |= m
            g = np.abs(np.diff(x, prepend=x[0]))
            m = robust_zscore(g) > acfg["gradient_z"]
            per_criterion["gradient"] += int(m.sum()); bad |= m
            if do_env:
                xf = mne.filter.filter_data(x, sfreq, lo, hi, verbose="error")
                env = np.abs(hilbert(xf))
                m = robust_zscore(env) > env_cfg["z"]
                per_criterion["hf_envelope"] += int(m.sum()); bad |= m
            count += bad
        mask = count >= acfg.get("min_channels", 1)
        intervals = merge_intervals(mask_to_intervals(mask, sfreq), pad=acfg["pad"],
                                    merge_gap=acfg["merge_gap"], t_min=0.0,
                                    t_max=self.raw.times[-1],
                                    min_duration=acfg.get("min_duration", 0.0))
        annot = self._make_annotations(intervals, desc)
        if len(annot):
            self.raw.set_annotations(self.raw.annotations + annot)
        total = float(sum(b - a for a, b in intervals))
        self._log_step("detect_bad_segments", n_segments=len(intervals),
                       total_s=round(total, 2),
                       percent=round(100 * total / self.raw.times[-1], 2),
                       samples_per_criterion=per_criterion)
        return annot

    def load_saved_bad_segments(self) -> int:
        annot = io.load_saved_bad_segments(self.cfg, self.subject)
        if annot is None:
            return 0
        self.raw.set_annotations(self.raw.annotations + annot)
        self._log_step("load_saved_bad_segments", n=len(annot))
        return len(annot)

    def bad_segments_summary(self) -> pd.DataFrame:
        a = self.raw.annotations
        rows = [(d, o, du) for d, o, du in zip(a.description, a.onset, a.duration)
                if d.upper().startswith("BAD")]
        df = pd.DataFrame(rows, columns=["description", "onset", "duration"])
        return df.groupby("description")["duration"].agg(["count", "sum"]) if len(df) else df

    # ---------------------------------------------------------------- epochs
    def _block_events(self) -> np.ndarray:
        bcfg = self.cfg["epochs"]["block"]
        sfreq = self.raw.info["sfreq"]
        step = bcfg["epoch_length"] - bcfg.get("overlap", 0.0)
        offsets = np.arange(0, bcfg["duration"] - bcfg["epoch_length"] + 1e-9, step)
        new = [[int(s + o * sfreq), 0, code] for s, _, code in self.events for o in offsets]
        return np.array(new, dtype=int)

    def make_epochs(self) -> mne.Epochs:
        ecfg = self.cfg["epochs"]
        events = self.events
        if ecfg["mode"] == "block":
            events = self._block_events()
            tmin, tmax = 0.0, ecfg["block"]["epoch_length"] - 1.0 / self.raw.info["sfreq"]
        elif ecfg["mode"] == "event":
            tmin, tmax = ecfg["tmin"], ecfg["tmax"]
        else:
            raise ValueError(f"Unknown epochs.mode {ecfg['mode']!r}")

        metadata = None
        meta_fname = self.cfg.metadata_fname(self.subject)
        if meta_fname is not None and meta_fname.exists():
            metadata = pd.read_csv(meta_fname)
            if len(metadata) != len(events):
                log.warning("Metadata rows (%d) != events (%d): metadata ignored",
                            len(metadata), len(events))
                metadata = None

        baseline = tuple(ecfg["baseline"]) if ecfg.get("baseline") else None
        epochs = mne.Epochs(self.raw, events, event_id=self.event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, picks="seeg", preload=True,
                            reject=ecfg.get("reject_peak_to_peak"), flat=ecfg.get("flat"),
                            reject_by_annotation=ecfg.get("reject_by_annotation", True),
                            decim=ecfg.get("decim", 1), metadata=metadata,
                            event_repeated="drop", verbose="error")
        n_total = len(events)
        if metadata is not None and self.cfg["events"].get("metadata_query"):
            epochs = epochs[self.cfg["events"]["metadata_query"]]
        if ecfg.get("keep_conditions"):
            epochs = epochs[ecfg["keep_conditions"]]
        self.epochs = epochs
        counts = {k: int((epochs.events[:, 2] == v).sum()) for k, v in epochs.event_id.items()}
        self._log_step("make_epochs", mode=ecfg["mode"], tmin=tmin, tmax=tmax,
                       n_events=n_total, n_kept=len(epochs),
                       drop_percent=round(100 * (1 - len(epochs) / max(n_total, 1)), 1),
                       counts=counts)
        return epochs

    # ------------------------------------------------------------------ save
    def save(self, overwrite: bool = True) -> dict[str, str]:
        """Write epochs, bad channels, bad segments and a JSON report to derivatives."""
        out = {}
        if self.epochs is not None:
            fname = self.cfg.epochs_fname(self.subject)
            self.epochs.save(fname, overwrite=overwrite, verbose="error")
            out["epochs"] = str(fname)
        bads = getattr(self, "monopolar_bads", None) or list(self.raw.info["bads"])
        out["bad_channels"] = str(io.save_bads(self.cfg, self.subject, bads,
                                               reasons=self.auto_bads))
        out["bad_segments"] = str(io.save_bad_segments(self.cfg, self.subject, self.raw))
        if self.bad_channel_metrics is not None:
            p = self.cfg.subject_dir(self.subject) / f"sub-{self.subject}_channel_qc.csv"
            self.bad_channel_metrics.to_csv(p, index=False)
            out["channel_qc"] = str(p)
        report = {
            "subject": self.subject,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "mne_version": mne.__version__,
            "config_file": str(self.cfg.path),
            "rereference": self.reref_method,
            "history": self.history,
            "outputs": out,
            "config": {k: self.cfg[k] for k in ("data", "events", "filtering", "bad_channels",
                                                 "rereference", "bad_segments", "epochs")},
        }
        p = self.cfg.subject_dir(self.subject) / f"sub-{self.subject}_preproc_report.json"
        save_json(report, p)
        out["report"] = str(p)
        log.info("[sub-%s] saved: %s", self.subject, out)
        return out

    # ----------------------------------------------------------------- batch
    def run(self, interactive: bool = False) -> mne.Epochs:
        """Full non-interactive (or interactive) pipeline for batch processing."""
        self.load()
        if self.cfg["bad_channels"]["auto"].get("enabled", True):
            self.detect_bad_channels()
        self.mark_bad_channels()
        if interactive:
            self.plot_raw()
        self.filter()
        self.rereference()
        loaded = self.load_saved_bad_segments() if self.cfg["bad_segments"].get("use_saved") else 0
        if self.cfg["bad_segments"]["auto"].get("enabled", True) and not loaded:
            self.detect_bad_segments()
        if interactive:
            self.plot_raw()
        self.make_epochs()
        self.save()
        return self.epochs
