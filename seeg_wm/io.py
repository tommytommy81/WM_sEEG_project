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


def find_events(raw: mne.io.BaseRaw, cfg: Config, subject: str
                ) -> tuple[np.ndarray, dict[str, int]]:
    """Return ``(events, event_id)`` restricted to codes declared in the config."""
    ecfg = cfg["events"]
    event_id: dict[str, int] = dict(ecfg["event_id"])
    source = ecfg["source"]
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


def load_saved_bad_segments(cfg: Config, subject: str) -> mne.Annotations | None:
    path = bad_segments_fname(cfg, subject)
    if not path.exists():
        return None
    annot = mne.read_annotations(path)
    return annot if len(annot) else None
