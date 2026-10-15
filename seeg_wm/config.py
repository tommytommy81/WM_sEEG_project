"""Loading of the YAML configuration and of the region (anatomy) file."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import ensure_dir, format_path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in (override or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _read_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_with_extends(path: Path) -> dict:
    cfg = _read_yaml(path)
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = (path.parent / parent).resolve()
        cfg = _deep_merge(_load_with_extends(parent_path), cfg)
    return cfg


class Config(dict):
    """Dictionary holding the configuration plus path helpers.

    Access parameters as ``cfg["spectral"]["fmin"]``. Paths are resolved
    relative to ``cfg.root`` (the repository root by default).
    """

    def __init__(self, data: dict, path: Path, root: Path):
        super().__init__(data)
        self.path = path
        self.root = root

    # ------------------------------------------------------------------ paths
    def resolve(self, p: str | Path | None) -> Path | None:
        if p is None:
            return None
        p = Path(p).expanduser()
        return p if p.is_absolute() else (self.root / p).resolve()

    @property
    def raw_root(self) -> Path:
        return self.resolve(self["paths"]["raw_root"])

    @property
    def derivatives_root(self) -> Path:
        return self.resolve(self["paths"]["derivatives_root"])

    def _fmt(self, key: str, subject: str) -> Path | None:
        tpl = self["paths"].get(key)
        path = format_path(
            tpl,
            raw_root=self.raw_root,
            derivatives_root=self.derivatives_root,
            subject=subject,
        )
        return self.resolve(path) if path is not None else None

    def raw_fname(self, subject: str) -> Path:
        return self._fmt("raw_fname", subject)

    def events_fname(self, subject: str) -> Path | None:
        return self._fmt("events_fname", subject)

    def annotations_fname(self, subject: str) -> Path | None:
        return self._fmt("annotations_fname", subject)

    def metadata_fname(self, subject: str) -> Path | None:
        tpl = self["events"].get("metadata_fname")
        if tpl is None:
            return None
        return self.resolve(
            format_path(tpl, raw_root=self.raw_root, subject=subject,
                        derivatives_root=self.derivatives_root)
        )

    def subject_dir(self, subject: str, kind: str = "preprocessing") -> Path:
        """``derivatives/<kind>/sub-<subject>/`` (created if needed)."""
        return ensure_dir(self.derivatives_root / kind / f"sub-{subject}")

    def group_dir(self, kind: str = "group") -> Path:
        return ensure_dir(self.derivatives_root / kind)

    def epochs_fname(self, subject: str) -> Path:
        return self.subject_dir(subject) / f"sub-{subject}_task-{self['project']['task']}-epo.fif"

    # -------------------------------------------------------------- regions
    def load_regions(self) -> dict[str, dict[str, list[str]]]:
        path = self.resolve(self["paths"].get("regions_file"))
        if path is None or not path.exists():
            return {}
        data = _read_yaml(path)
        return {str(k): {r: list(v or []) for r, v in (regs or {}).items()}
                for k, regs in data.items()}

    @property
    def subjects(self) -> list[str]:
        return [str(s) for s in self.get("subjects", [])]


def load_config(path: str | Path | None = None, root: str | Path | None = None,
                overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config (+ ``extends`` chain, + ``config.local.yaml``, + overrides).

    Parameters
    ----------
    path : path to the YAML file (default: ``config/config.yaml``).
    root : directory relative paths are resolved against (default: repo root).
    overrides : nested dict deep-merged last (handy in notebooks).
    """
    path = Path(path) if path is not None else REPO_ROOT / "config" / "config.yaml"
    path = path.resolve()
    data = _load_with_extends(path)
    local = path.parent / "config.local.yaml"
    if local.exists() and local != path:
        data = _deep_merge(data, _read_yaml(local))
    if overrides:
        data = _deep_merge(data, overrides)
    root = Path(root).resolve() if root is not None else path.parent.parent
    return Config(data, path=path, root=root)
