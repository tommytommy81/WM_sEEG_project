"""seeg_wm – preprocessing and spectral analysis of a working-memory task in sEEG (MNE)."""

from .config import Config, load_config
from .preprocessing import SEEGPreprocessor
from .spectral import SpectralAnalyzer, band_power, compute_condition_spectra
from .stats import GroupStats

__all__ = ["Config", "load_config", "SEEGPreprocessor", "SpectralAnalyzer", "GroupStats",
           "band_power", "compute_condition_spectra"]
__version__ = "0.1.0"
