"""State-space cyclical harmonic analysis (SCHA) of tidal time series.

SCHA models a series as a smooth trend (the subtide) plus higher-order stochastic cycles at
tidal harmonic bands (D1, D2, ...) plus observation noise, estimated with Kalman filtering and
smoothing. It follows Ateljevich and Wang (2023) and is the Python counterpart of the R package
``stochCycle``. Cycle frequencies inside the model are in radians per sample; the sampling step
of the data converts them from physical frequencies.

Main entry points:

- :func:`scha.fit.fit` (``scha.fit``): fit a model to a regular series; returns a
  :class:`scha.fit.SCHAFit`.
- :func:`scha.fit.load_fit` (``scha.load_fit``): read a fit written by
  :meth:`scha.fit.SCHAFit.to_yaml`.
- :func:`scha.fit.decompose` (``scha.decompose``): decompose a series with a fit or fit YAML.
- :class:`scha.fit.SCHAFit` (``scha.SCHAFit``): the fit result, with ``decompose``, ``phasors``,
  ``gains``, ``summary`` and related methods.
- :class:`scha.requirements.Requirements` (``scha.Requirements``): gain requirements imposed on
  a constrained fit.
- :func:`scha.phase.amplitude_phase` and :func:`scha.phase.demodulate`
  (``scha.amplitude_phase``, ``scha.demodulate``): amplitude and phase of each band from the
  smoothed phasors.

Lower-level building blocks (:class:`scha.params.SCHAParams`,
:func:`scha.state_space.build_scha_state_space`, the Kalman filter and smoother in
:mod:`scha.ffbs`, the likelihoods in :mod:`scha.fastll`) are also exported.
"""
from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from .params import SCHAParams
from .state_space import build_scha_state_space
from .uncond import cycle_unconditional_cov, cycle_top_variance
from .ffbs import kalman_filter, kalman_smoother, carter_kohn_draw
from .fastll import scha_loglik, scha_loglik_concentrated
from .post import extract_cycle_amplitude_phase
from .requirements import Requirements
from .fit import SCHAFit, decompose, fit, load_fit
from .phase import amplitude_phase, demodulate
__all__ = [
    "fit",
    "load_fit",
    "decompose",
    "SCHAFit",
    "Requirements",
    "amplitude_phase",
    "demodulate",
    "SCHAParams",
    "build_scha_state_space",
    "cycle_unconditional_cov",
    "cycle_top_variance",
    "kalman_filter",
    "kalman_smoother",
    "carter_kohn_draw",
    "scha_loglik",
    "scha_loglik_concentrated",
    "extract_cycle_amplitude_phase",
]
