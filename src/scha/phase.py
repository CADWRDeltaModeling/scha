"""Amplitude and phase of SCHA bands from top-pair phasors.

Convention (tidal): a band's contribution is

    x_t = A_t cos(w t - phi_t) = a_t cos(w t) + b_t sin(w t),
    a = A cos(phi),  b = A sin(phi),  phi = atan2(b, a)  (phase lag),

with t in hours since a fixed epoch and w a reference frequency in radians per
hour. The model's top pair (psi, psi*) gives z = psi - i psi* ~ A exp(i(w t - phi))
(psi is the band's contribution to y), so z exp(-i w t) = a - i b.

Fits only produce phasors; everything here is post-processing and works the same
for smoothed means, simulation-smoother draws or MCMC draws (arrays of shape
(..., T)).
"""
from __future__ import annotations

from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd

from .bands import band_cpd
from .tide_registry import rph

DEFAULT_EPOCH = "1970-01-01"


def hours_since(index: pd.DatetimeIndex, epoch=DEFAULT_EPOCH) -> np.ndarray:
    """Return the hours from ``epoch`` to each time in ``index``.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Times to convert.
    epoch : str or Timestamp, optional
        Time origin, default ``DEFAULT_EPOCH`` (``"1970-01-01"``).

    Returns
    -------
    np.ndarray
        Float array of hours since ``epoch``, shape (len(index),).
    """
    return np.asarray((index - pd.Timestamp(epoch)) / pd.Timedelta("1h"), dtype=float)


def demodulate(c, s, t_hours, omega_rph: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute the local cosine and sine coefficients of a band.

    With z = c - i s, the coefficients are a = Re(z exp(-i w t)) and b = -Im(z exp(-i w t)),
    so that x_t = a_t cos(w t) + b_t sin(w t).

    Parameters
    ----------
    c, s : array_like
        Top-pair states psi and psi*, shape (..., T).
    t_hours : array_like
        Times in hours since the epoch, shape (T,).
    omega_rph : float
        Reference frequency in radians per hour.

    Returns
    -------
    a, b : np.ndarray
        Coefficients with x_t = a_t cos(w t) + b_t sin(w t), each of shape (..., T).
    """
    z = np.asarray(c, dtype=float) - 1j * np.asarray(s, dtype=float)
    d = z * np.exp(-1j * omega_rph * np.asarray(t_hours, dtype=float))
    return d.real, -d.imag


def phase_deg(a, b, unwrap: bool = True) -> np.ndarray:
    """Return the phase lag phi = atan2(b, a) in degrees, unwrapped along the last axis by default.

    Parameters
    ----------
    a, b : array_like
        Local cosine and sine coefficients (see :func:`demodulate`), shape (..., T).
    unwrap : bool, optional
        If True (default), unwrap the phase along the last axis (time) with ``np.unwrap``
        before converting to degrees.

    Returns
    -------
    np.ndarray
        Phase lag in degrees, same shape as the broadcast of ``a`` and ``b``.
    """
    ph = np.arctan2(b, a)
    if unwrap:
        ph = np.unwrap(ph, axis=-1)
    return np.degrees(ph)


def _ref_rph(band: str, ref: Union[None, str, float, Mapping]) -> float:
    """Return the reference frequency of ``band`` in radians per hour.

    Parameters
    ----------
    band : str
        Band label such as ``"D1"``.
    ref : None, str, float or mapping
        Reference specification as in :func:`amplitude_phase`. A mapping is looked up with
        ``band`` (a missing key behaves as None). None gives the band's model line
        (:func:`scha.bands.band_cpd`), a string is a tidal constituent name, and a number is
        a frequency in cycles per day.
    """
    r = ref.get(band) if isinstance(ref, Mapping) else ref
    if r is None:
        return band_cpd(band) * 2.0 * np.pi / 24.0
    if isinstance(r, str):
        return rph(r)
    return float(r) * 2.0 * np.pi / 24.0


def amplitude_phase(phasors: pd.DataFrame, ref: Union[None, str, float, Mapping] = None,
                    epoch=DEFAULT_EPOCH, unwrap: bool = True) -> pd.DataFrame:
    """Return local coefficients, amplitude and phase of each band.

    Parameters
    ----------
    phasors : pd.DataFrame
        Output of :meth:`scha.fit.SCHAFit.phasors` (columns ``<band>_c``, ``<band>_s``),
        indexed by a DatetimeIndex.
    ref : None, str, float or mapping, optional
        Reference frequency per band: None (default) uses the band's model line (k times M1);
        a constituent name (e.g. ``"K1"``) or a frequency in cpd; or a mapping band ->
        either of these.
    epoch : str or Timestamp, optional
        Time origin of w t, default ``DEFAULT_EPOCH``.
    unwrap : bool, optional
        Unwrap the phase in time (default True).

    Returns
    -------
    pd.DataFrame
        Columns ``<band>_a``, ``<band>_b``, ``<band>_amp`` and ``<band>_phase``
        (degrees, lag convention).
    """
    bands = [c[:-2] for c in phasors.columns if c.endswith("_c")]
    t = hours_since(phasors.index, epoch)
    out = {}
    for b in bands:
        a, bb = demodulate(phasors[f"{b}_c"].to_numpy(), phasors[f"{b}_s"].to_numpy(), t, _ref_rph(b, ref))
        out[f"{b}_a"] = a
        out[f"{b}_b"] = bb
        out[f"{b}_amp"] = np.hypot(a, bb)
        out[f"{b}_phase"] = phase_deg(a, bb, unwrap=unwrap)
    return pd.DataFrame(out, index=phasors.index)
