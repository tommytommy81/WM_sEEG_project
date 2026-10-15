"""Non-interactive batch runner (same steps as the notebooks).

Examples::

    python scripts/run_pipeline.py preprocess --config config/config.yaml
    python scripts/run_pipeline.py preprocess --subjects 5 6
    python scripts/run_pipeline.py spectral --recompute
    python scripts/run_pipeline.py all --config config/config_demo.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from seeg_wm import GroupStats, SEEGPreprocessor, SpectralAnalyzer, load_config  # noqa: E402
from seeg_wm.utils import get_logger, set_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="WM_sEEG batch pipeline")
    ap.add_argument("step", choices=["preprocess", "spectral", "all"])
    ap.add_argument("--config", default=None, help="YAML config (default config/config.yaml)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--recompute", action="store_true", help="ignore cached spectra")
    args = ap.parse_args()

    cfg = load_config(args.config)
    log = get_logger(cfg["project"].get("log_level", "INFO"))
    set_seed(cfg["project"].get("random_seed"))
    subjects = args.subjects or cfg.subjects

    if args.step in ("preprocess", "all"):
        for s in subjects:
            try:
                SEEGPreprocessor(cfg, s).run(interactive=False)
            except Exception as err:
                log.error("[sub-%s] preprocessing failed: %s", s, err)

    if args.step in ("spectral", "all"):
        psd, band = SpectralAnalyzer(cfg).run(subjects, recompute=args.recompute)
        gdir = cfg.group_dir("group")
        psd.to_csv(gdir / "group_psd.csv.gz", index=False, compression="gzip")
        band.to_csv(gdir / "group_bandpower.csv", index=False)
        gs = GroupStats(cfg)
        gs.run(psd, band)
        gs.save()


if __name__ == "__main__":
    main()
