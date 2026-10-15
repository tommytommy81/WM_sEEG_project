"""Reading raw recordings, events and saved preprocessing decisions."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd

from .config import Config
from .utils import get_logger, load_json, save_json

log = get_logger()

_READERS = {
    ".edf": mne.io.read_raw_edf,
    ".bdf": mne.io.read_raw_bdf,
    ".fif": mne.io.read_raw_fif,
    ".vhdr": mne.io.read_raw_brainvision,
    ".set": mne.io.read_raw_eeglab,
}


def read_raw(fname: str | Path, reader: str = "auto") -> mne.io.BaseRaw:
    fname = Path(fname)
    if not fname.exists():
        raise FileNotFoundError(f"Raw file not found: {fname}")
    if reader == "auto":
        ext = ".fif" if fname.name.endswith(".fif.gz") else fname.suffix.lower()
        if ext not in _READERS:
            raise ValueError(f"Cannot infer reader for extension {ext!r}; set data.reader")
        func = _READERS[ext]
    else:
        key = {"edf": ".edf", "edf+": ".edf", "bdf": ".bdf", "fif": ".fif",
               "brainvision": ".vhdr", "eeglab": ".set"}[reader]
        func = _READERS[key]
    return func(fname, preload=True, verbose="error")


def read_events_file(fname: Path, sfreq: float, first_samp: int,
                     event_id: dict[str, int]) -> np.ndarray:
    """Read events from an MNE ``.eve``/``.fif`` file or a CSV.

    CSV must contain either ``sample`` or ``onset`` (s) and either ``code``
    (int) or ``trial_type`` (key of ``event_id``).
    """
    fname = Path(fname)
    if fname.suffix in (".eve", ".fif") or fname.name.endswith("-eve.fif"):
        return mne.read_events(fname)
    df = pd.read_csv(fname)
    if "sample" in df:
        samples = df["sample"].astype(int).to_numpy()
    elif "onset" in df:
        samples = np.round(df["onset"].to_numpy() * sfreq).astype(int) + first_samp
    else:
        raise ValueError(f"{fname}: needs a 'sample' or 'onset' column")
    if "code" in df:
        codes = df["code"].astype(int).to_numpy()
    elif "trial_type" in df:
        codes = df["trial_type"].map(event_id).to_numpy()
        if np.isnan(codes.astype(float)).any():
            raise ValueError(f"{fname}: trial_type values missing from events.event_id")
        codes = codes.astype(int)
    else:
        raise ValueError(f"{fname}: needs a 'code' or 'trial_type' column")
    return np.c_[samples, np.zeros_like(samples), codes]


def detect_threshold_triggers(raw: mne.io.BaseRaw, channel: str, level_uv: float,
                              direction: str, merge_window: float,
                              tmin: float | None = None, tmax: float | None = None
                              ) -> pd.DataFrame:
    """Detect trigger pulses as threshold crossings on an analog marker channel.

    Designed for marker lines such as ``MKR2+``: a ±50 µV square wave on which
    triggers appear as brief excursions beyond the square-wave levels (e.g. to
    -100 µV). Every sample beyond ``level_uv`` (in the given ``direction``) is a
    pulse sample; consecutive samples form a pulse, and pulses closer than
    ``merge_window`` seconds are merged into a single trigger whose onset is the
    first pulse.

    Returns one row per trigger: ``sample`` (absolute, includes ``first_samp``),
    ``onset`` (s from the start of the recording), ``n_pulses``, ``duration_ms``
    (first to last pulse sample), ``peak_uv`` (most extreme value).

    All detection parameters come from ``events.threshold`` in the YAML config;
    use :func:`triggers_from_config` to call this with those values.
    """
    if channel not in raw.ch_names:
        raise ValueError(f"Trigger channel {channel!r} not found. Channels: {raw.ch_names}")
    x = raw.get_data(picks=[channel], tmin=tmin, tmax=tmax)[0] * 1e6
    sfreq = raw.info["sfreq"]
    offset = int(round((tmin or 0.0) * sfreq))
    if direction == "below":
        hit = x < level_uv
    elif direction == "above":
        hit = x > level_uv
    else:
        raise ValueError(f"direction must be 'below' or 'above', got {direction!r}")
    d = np.diff(np.r_[0, hit.astype(np.int8), 0])
    starts, stops = np.where(d == 1)[0], np.where(d == -1)[0]
    cols = ["sample", "onset", "n_pulses", "duration_ms", "peak_uv"]
    if len(starts) == 0:
        return pd.DataFrame(columns=cols)
    gap = int(round(merge_window * sfreq))
    groups, cur = [], [0]
    for i in range(1, len(starts)):
        if starts[i] - starts[cur[0]] <= gap:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    rows = []
    for g in groups:
        s0, s1 = starts[g[0]], stops[g[-1]]
        seg = x[s0:s1]
        rows.append({"sample": raw.first_samp + offset + s0,
                     "onset": (offset + s0) / sfreq,
                     "n_pulses": len(g),
                     "duration_ms": 1000 * (s1 - s0) / sfreq,
                     "peak_uv": seg.min() if direction == "below" else seg.max()})
    return pd.DataFrame(rows, columns=cols)


THRESHOLD_KEYS = ("channel", "level_uv", "direction", "merge_window")


def threshold_config(cfg: Config) -> dict:
    """``events.threshold`` from the YAML, checking that every detection key is set."""
    tcfg = (cfg["events"].get("threshold") or {})
    missing = [k for k in THRESHOLD_KEYS if tcfg.get(k) is None]
    if missing:
        raise KeyError(f"events.threshold is missing {missing} in {cfg.path}")
    return tcfg


def triggers_from_config(raw: mne.io.BaseRaw, cfg: Config, tmin: float | None = None,
                         tmax: float | None = None) -> pd.DataFrame:
    """Run :func:`detect_threshold_triggers` with the parameters of ``events.threshold``."""
    tcfg = threshold_config(cfg)
    return detect_threshold_triggers(raw, tcfg["channel"], float(tcfg["level_uv"]),
                                     tcfg["direction"], float(tcfg["merge_window"]),
                                     tmin=tmin, tmax=tmax)


def find_events(raw: mne.io.BaseRaw, cfg: Config, subject: str
                ) -> tuple[np.ndarray, dict[str, int]]:
    """Return ``(events, event_id)`` restricted to codes declared in the config."""
    ecfg = cfg["events"]
    event_id: dict[str, int] = dict(ecfg["event_id"])
    source = ecfg["source"]
    if source == "threshold":
        tcfg = threshold_config(cfg)
        trig = triggers_from_config(raw, cfg)
        code = int(tcfg.get("code", 1))
        events = np.c_[trig["sample"].to_numpy(int), np.zeros(len(trig), int),
                       np.full(len(trig), code)]
        # triggers carry no condition code: label them all with one name until
        # they are matched to the behavioural log
        event_id = {tcfg.get("event_name", "trigger"): code}
        log.info("Detected %d triggers on %s (threshold %s µV, merge_window %s s)",
                 len(events), tcfg["channel"], tcfg["level_uv"], tcfg["merge_window"])
        return events, event_id
    if source == "annotations":
        mapping = ecfg.get("annotation_map")
        if mapping is None:
            # descriptions equal to a condition name or to its numeric code
            mapping = {**event_id, **{str(v): v for v in event_id.values()}}
        present = set(raw.annotations.description)
        mapping = {k: int(v) for k, v in mapping.items() if k in present}
        if not mapping:
            raise RuntimeError(
                "No annotation description matches events.event_id / annotation_map. "
                f"Found descriptions: {sorted(present)[:20]}")
        events, _ = mne.events_from_annotations(raw, event_id=mapping, verbose="error")
    elif source == "stim":
        events = mne.find_events(raw, stim_channel=ecfg.get("stim_channel"),
                                 min_duration=ecfg.get("min_duration", 0.002),
                                 shortest_event=ecfg.get("shortest_event", 1),
                                 verbose="error")
    elif source == "file":
        events = read_events_file(cfg.events_fname(subject), raw.info["sfreq"],
                                  raw.first_samp, event_id)
    else:
        raise ValueError(f"Unknown events.source {source!r}")

    codes = set(events[:, 2])
    unknown = codes - set(event_id.values())
    if unknown:
        log.info("Ignoring event codes not in events.event_id: %s", sorted(unknown))
    events = events[np.isin(events[:, 2], list(event_id.values()))]
    event_id = {k: v for k, v in event_id.items() if v in codes}
    log.info("Found %d events: %s", len(events),
             {k: int((events[:, 2] == v).sum()) for k, v in event_id.items()})
    return events, event_id


def read_annotations_file(fname: Path, orig_time=None) -> mne.Annotations:
    annot = mne.read_annotations(fname)
    if orig_time is not None and annot.orig_time is None:
        annot = mne.Annotations(annot.onset, annot.duration, annot.description,
                                orig_time=orig_time)
    return annot


# ------------------------------------------------- saved preprocessing decisions
def bads_fname(cfg: Config, subject: str) -> Path:
    return cfg.subject_dir(subject) / f"sub-{subject}_bad_channels.json"


def bad_segments_fname(cfg: Config, subject: str) -> Path:
    return cfg.subject_dir(subject) / f"sub-{subject}_bad_segments.csv"


def save_bads(cfg: Config, subject: str, bads: list[str], reasons: dict | None = None) -> Path:
    return save_json({"bads": sorted(set(bads)), "reasons": reasons or {}},
                     bads_fname(cfg, subject))


def load_saved_bads(cfg: Config, subject: str) -> list[str]:
    path = bads_fname(cfg, subject)
    return list(load_json(path)["bads"]) if path.exists() else []


def save_bad_segments(cfg: Config, subject: str, raw: mne.io.BaseRaw) -> Path:
    """Save the BAD_* annotations of ``raw`` (MNE CSV format, keeps orig_time)."""
    annot = raw.annotations
    keep = [i for i, d in enumerate(annot.description) if d.upper().startswith("BAD")]
    path = bad_segments_fname(cfg, subject)
    annot[keep].save(path, overwrite=True, verbose="error")
    return path


def load_saved_bad_segments(cfg: Config, subject: str, orig_time=None
                            ) -> mne.Annotations | None:
    """Read saved BAD_* annotations, re-expressed relative to ``orig_time``.

    MNE reads a CSV with absolute timestamps back with ``orig_time`` set to the
    first onset, which cannot be added to the recording's annotations. Pass
    ``raw.annotations.orig_time`` so the onsets are shifted onto the same
    reference time.
    """
    path = bad_segments_fname(cfg, subject)
    if not path.exists():
        return None
    annot = mne.read_annotations(path)
    if not len(annot):
        return None
    if orig_time is not None and annot.orig_time is not None and annot.orig_time != orig_time:
        shift = (annot.orig_time - orig_time).total_seconds()
        annot = mne.Annotations(annot.onset + shift, annot.duration, annot.description,
                                orig_time=orig_time)
    return annot
