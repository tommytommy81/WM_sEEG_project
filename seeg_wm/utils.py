"""Generic helpers: logging, I/O, robust statistics and interval arithmetic."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterable

import numpy as np

LOGGER_NAME = "seeg_wm"


def get_logger(level: str | int = "INFO") -> logging.Logger:
    """Return the package logger, configured once with a compact format."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def set_seed(seed: int | None) -> None:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_json(obj, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=False)
    return path


def load_json(path: str | Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def robust_zscore(x: np.ndarray, axis: int | None = None) -> np.ndarray:
    """Median/MAD z-score (MAD scaled to match the SD of a Gaussian)."""
    x = np.asarray(x, dtype=float)
    med = np.nanmedian(x, axis=axis, keepdims=True)
    mad = 1.4826 * np.nanmedian(np.abs(x - med), axis=axis, keepdims=True)
    mad = np.where(mad == 0, np.finfo(float).eps, mad)
    return (x - med) / mad


def mask_to_intervals(mask: np.ndarray, sfreq: float) -> list[tuple[float, float]]:
    """Convert a boolean sample mask into a list of (onset, offset) in seconds."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any():
        return []
    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    stops = np.where(diff == -1)[0]
    return [(s / sfreq, e / sfreq) for s, e in zip(starts, stops)]


def merge_intervals(
    intervals: Iterable[tuple[float, float]],
    pad: float = 0.0,
    merge_gap: float = 0.0,
    t_min: float = 0.0,
    t_max: float | None = None,
    min_duration: float = 0.0,
) -> list[tuple[float, float]]:
    """Pad, clip, drop short and merge close intervals."""
    out: list[list[float]] = []
    for on, off in sorted(intervals):
        if off - on < min_duration:
            continue
        on, off = max(t_min, on - pad), off + pad
        if t_max is not None:
            off = min(t_max, off)
        if out and on - out[-1][1] <= merge_gap:
            out[-1][1] = max(out[-1][1], off)
        else:
            out.append([on, off])
    return [(a, b) for a, b in out]


def format_path(template: str | None, **kwargs) -> Path | None:
    if template is None:
        return None
    return Path(str(template).format(**kwargs))
