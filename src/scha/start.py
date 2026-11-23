"""Automatic starting values for SCHA fits.

Each component's variance is set so that its model spectrum carries the excess
periodogram power observed in its own frequency window:

- sigma2_eps from the mean periodogram level above the last band;
- sigma2_kappa for band k from the window cpd_k +/- M1/2;
- sigma2_zeta from the median per-bin ratio of periodogram to trend shape below
  the D1 window.

rho comes from a damping per hour, so the start does not depend on the time step
(``RHO_PER_HOUR ** step_h`` equals 0.985 at 15 min, near the Freeport fits).
Gaps are linearly interpolated for the periodogram only. The start then goes
through the feasibility phase of :func:`scha.mlfit.fit_ml`.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .gains import component_spectra
from .params import SCHAParams
from .tide_registry import rph

RHO_PER_HOUR = 0.985 ** 4
_M1_CPD = rph("M1") * 24.0 / (2.0 * np.pi)
_FLOOR = 0.05


def default_rho(step_h: float) -> float:
    """Return the start value of rho for a sampling step in hours.

    Parameters
    ----------
    step_h : float
        Sampling step in hours.

    Returns
    -------
    float
        ``RHO_PER_HOUR ** step_h``, the damping per sample for a damping of ``RHO_PER_HOUR``
        per hour.
    """
    return float(RHO_PER_HOUR ** float(step_h))


def _periodogram(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Return a one-sided periodogram normalised so that its sum is the variance.

    The series is trimmed to its first and last finite values, gaps are linearly interpolated,
    and a linear trend is removed.

    Parameters
    ----------
    y : np.ndarray
        Observations on a regular grid; NaN marks gaps.

    Returns
    -------
    f : np.ndarray
        Frequencies in cycles per sample, excluding 0 and Nyquist.
    I : np.ndarray
        Periodogram ordinates at ``f``.
    nn : int
        Length of the interpolated span.

    Raises
    ------
    ValueError
        If ``y`` has fewer than 16 finite observations.
    """
    y = np.asarray(y, dtype=float)
    ok = np.flatnonzero(np.isfinite(y))
    if ok.size < 16:
        raise ValueError("Too few observations for an automatic start.")
    y = y[ok[0]:ok[-1] + 1]
    t = np.arange(y.size)
    good = np.isfinite(y)
    y = np.interp(t, t[good], y[good])
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    nn = y.size
    X = np.fft.rfft(y)
    I = 2.0 * np.abs(X) ** 2 / nn**2
    f = np.arange(I.size) / nn
    return f[1:-1], I[1:-1], nn


def auto_start(y: np.ndarray, step_h: float, lambdas: Sequence[float], m: int = 2, n: int = 3,
               rho: Optional[float] = None) -> SCHAParams:
    """Return starting parameters from the periodogram of y.

    Parameters
    ----------
    y : np.ndarray
        Observations on a regular grid; NaN marks gaps.
    step_h : float
        Sampling step in hours.
    lambdas : sequence of float
        Cycle frequencies in radians per sample, increasing.
    m, n : int, optional
        Trend and cycle orders (defaults 2 and 3).
    rho : float, optional
        Start rho; default :func:`default_rho`.

    Returns
    -------
    scha.params.SCHAParams
        Parameters with ``sigma2_eps``, ``sigma2_zeta`` and ``sigma2_kappa`` set from the
        periodogram as described in the module docstring.

    Raises
    ------
    ValueError
        If ``y`` has fewer than 16 finite observations.
    """
    lambdas = list(lambdas)
    rho = default_rho(step_h) if rho is None else float(rho)
    spd = 24.0 / step_h
    f, I, nn = _periodogram(y)
    cpd = f * spd
    band_cpd = np.array(lambdas) * spd / (2.0 * np.pi)
    half = 0.5 * _M1_CPD
    # expected periodogram ordinate of a component with density S is (2 / nn) * S
    shape = component_spectra(SCHAParams(m=m, n=n, rho=rho, lambdas=lambdas, sigma2_eps=1.0, sigma2_zeta=1.0,
                                         sigma2_kappa=[1.0] * len(lambdas)), cpd, step_h, relative=False)
    shape *= 2.0 / nn

    hf = cpd > (band_cpd.max() + half if lambdas else 2.0)
    if hf.sum() < 10:
        hf = cpd >= np.quantile(cpd, 0.9)
    eps = float(I[hf].mean() * nn / 2.0)
    noise = 2.0 * eps / nn

    def level(win: np.ndarray, s: np.ndarray) -> float:
        excess = I[win].sum() - noise * win.sum()
        excess = max(excess, _FLOOR * noise * win.sum())
        return float(excess / s[win].sum())

    lo_edge = (band_cpd.min() - half) if lambdas else 0.5
    # median per-bin ratio: the trend shape is steep, so a power sum would be set by the lowest bins
    low = cpd < lo_edge
    ratio = np.maximum(I[low] - noise, _FLOOR * noise) / shape[0][low]
    zeta = float(np.median(ratio) / np.log(2.0))
    kappa = [level(np.abs(cpd - c) <= half, shape[1 + k]) for k, c in enumerate(band_cpd)]
    return SCHAParams(m=m, n=n, rho=rho, lambdas=lambdas, sigma2_eps=eps, sigma2_zeta=zeta, sigma2_kappa=kappa)
