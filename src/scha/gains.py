"""Steady-state component spectra and smoother gains of the SCHA model.

The gain of component j at frequency f is

    G_j(f) = S_j(f) / (sum_k S_k(f) + sigma2_eps),

with S_j the spectral density (pseudo-spectral for the trend) of component j's
contribution to y. Gains describe the filter implied by the parameters, so they
keep their meaning in nonstationary and gappy records, and they are stated in
cycles per day so they carry over between time steps.

Spectra here use one normalisation for all components: white noise of variance
s2 has density s2. Frequencies are in cpd; the step converts them to radians per
sample.
"""
from __future__ import annotations

from math import comb

import numpy as np

from .params import SCHAParams
from .state_space import build_scha_state_space


def _omega(cpd, step_h: float) -> np.ndarray:
    """Convert frequencies in cycles per day to radians per sample for a step of ``step_h`` hours."""
    return 2.0 * np.pi * np.asarray(cpd, dtype=float) * float(step_h) / 24.0


def component_spectra(p: SCHAParams, cpd, step_h: float, relative: bool = True) -> np.ndarray:
    """Return the closed-form component spectra.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Parameters with lambdas in radians per sample at ``step_h``.
    cpd : array_like
        Frequencies in cycles per day.
    step_h : float
        Sampling step in hours.
    relative : bool, optional
        If True (default), spectra are divided by sigma2_eps (the noise spectrum is then 1).

    Returns
    -------
    np.ndarray
        Shape (1 + Nc, len(cpd)); row 0 is the trend (subtide), rows 1.. the cycles.
    """
    w = _omega(cpd, step_h)
    n, r = p.n, p.rho
    j = np.arange(n + 1)
    d = j[:, None] - j[None, :]
    binom = np.array([comb(n, i) for i in j], dtype=float)
    coef = (-1.0) ** (j[:, None] + j[None, :]) * binom[:, None] * binom[None, :] * r ** (j[:, None] + j[None, :])
    cw = np.cos(w[:, None, None] * d[None])
    scale = 1.0 / p.sigma2_eps if relative else 1.0
    S = np.empty((1 + len(p.lambdas), w.size))
    S[0] = p.sigma2_zeta * scale / (2.0 - 2.0 * np.cos(w)) ** p.m
    for c, lc in enumerate(p.lambdas):
        num = np.einsum("jk,fjk->f", coef * np.cos(lc * d), cw)
        den = (1 + 4 * r**2 * np.cos(lc) ** 2 + r**4 - 4 * r * (1 + r**2) * np.cos(lc) * np.cos(w)
               + 2 * r**2 * np.cos(2 * w)) ** n
        S[1 + c] = p.sigma2_kappa[c] * scale * num / den
    return S


def gains(p: SCHAParams, cpd, step_h: float) -> np.ndarray:
    """Return the steady-state smoother gains, closed form.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Parameters with lambdas in radians per sample at ``step_h``.
    cpd : array_like
        Frequencies in cycles per day.
    step_h : float
        Sampling step in hours.

    Returns
    -------
    np.ndarray
        Shape (1 + Nc, len(cpd)); row 0 subtide, rows 1.. cycles. The noise gain
        is ``1 - gains.sum(axis=0)``.
    """
    S = component_spectra(p, cpd, step_h, relative=True)
    return S / (S.sum(axis=0) + 1.0)


def gains_state_space(p: SCHAParams, cpd, step_h: float) -> np.ndarray:
    """Return the steady-state smoother gains from the state-space transfer functions.

    Slow reference for :func:`gains`; same layout.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Parameters with lambdas in radians per sample at ``step_h``.
    cpd : array_like
        Frequencies in cycles per day.
    step_h : float
        Sampling step in hours.

    Returns
    -------
    np.ndarray
        Shape (1 + Nc, len(cpd)); row 0 subtide, rows 1.. cycles. The noise gain is
        ``1 - gains.sum(axis=0)``.
    """
    ss = build_scha_state_space(p)
    T, R, Z, q = ss["T"], ss["R"], ss["Z"][0], np.diag(ss["Q"])
    eye = np.eye(T.shape[0])
    w = _omega(cpd, step_h)
    S = np.zeros((1 + len(p.lambdas), w.size))
    for i, wi in enumerate(w):
        h = Z @ np.linalg.solve(eye - T * np.exp(-1j * wi), R)
        pw = q * np.abs(h) ** 2
        S[0, i] = pw[0]
        S[1:, i] = pw[1:].reshape(-1, 2).sum(axis=1)
    return S / (S.sum(axis=0) + p.sigma2_eps)


def cosine_lanczos_response(cutoff_h: float, cpd, step_h: float) -> np.ndarray:
    """Return the amplitude response of ``vtools.functions.filter.cosine_lanczos`` at its default length.

    vtools applies the filter with ``filtfilt``, so the response is the squared
    magnitude of the coefficient transform.

    Parameters
    ----------
    cutoff_h : float
        Cutoff period in hours (the ``cutoff_period`` passed to vtools).
    cpd : array_like
        Frequencies in cycles per day.
    step_h : float
        Sampling step in hours.

    Returns
    -------
    np.ndarray
        Response at ``cpd``.
    """
    from vtools.functions.filter import lowpass_lanczos_filter_coef

    cf = 2.0 * float(step_h) / float(cutoff_h)
    m = int(1.25 * 2.0 / cf)
    c = lowpass_lanczos_filter_coef(cf, m, cosine_taper=True)
    k = np.arange(-m, m + 1)
    return np.abs(np.exp(-1j * np.outer(_omega(cpd, step_h), k)) @ c) ** 2


def crossover(p: SCHAParams, step_h: float, lo: float = 0.05, hi: float = None) -> float:
    """Return the lowest frequency (cpd) at which the D1 gain reaches the subtide gain.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Parameters; the first cycle is taken as D1.
    step_h : float
        Sampling step in hours.
    lo, hi : float, optional
        Search range in cpd, searched on 2001 equally spaced points; ``lo`` defaults to 0.05 and
        ``hi`` to the first cycle's line.

    Returns
    -------
    float
        Crossover frequency in cpd (NaN if there are no cycles or none in the range).
    """
    if not p.lambdas:
        return float("nan")
    if hi is None:
        hi = p.lambdas[0] * 24.0 / (2.0 * np.pi * step_h)
    f = np.linspace(lo, hi, 2001)
    g = gains(p, f, step_h)
    diff = g[1] - g[0]
    i = np.flatnonzero(diff >= 0.0)
    if i.size == 0:
        return float("nan")
    k = int(i[0])
    if k == 0:
        return float(f[0])
    return float(np.interp(0.0, diff[k - 1:k + 1], f[k - 1:k + 1]))
