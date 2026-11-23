"""Sampling step and conversion of physical frequencies to radians per sample.

The model works in radians per sample. This module is the single place where
that conversion is made, so that physical frequencies (radians per hour or
cycles per day) and the sampling step never meet anywhere else.

Contract for time-series inputs: a naive ``pd.DatetimeIndex`` whose ``freq``
is set to a fixed step (for example ``6min``, ``15min``, ``30min``, ``h``).
Gaps are NaN rows on that grid.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_HOUR = pd.Timedelta("1h")


def freq_hours(freq) -> float:
    """Return the length of a fixed pandas frequency in hours.

    Parameters
    ----------
    freq : str, pandas offset or pd.Timedelta
        Fixed step such as ``"15min"`` or ``"h"``.

    Returns
    -------
    float
        Step in hours.

    Raises
    ------
    ValueError
        If ``freq`` is not a fixed, positive step (for example ``"MS"``).
    """
    try:
        dt = pd.Timedelta(pd.tseries.frequencies.to_offset(freq))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Frequency {freq!r} is not a fixed time step.") from exc
    hours = dt / _HOUR
    if not hours > 0.0:
        raise ValueError(f"Frequency {freq!r} must be a positive step.")
    return float(hours)


def step_hours(index: pd.DatetimeIndex) -> float:
    """Return the sampling step of a regular, naive DatetimeIndex in hours.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Index with ``freq`` set and no time zone.

    Returns
    -------
    float
        Step in hours.

    Raises
    ------
    ValueError
        If the index is not a DatetimeIndex, has no ``freq`` or is tz-aware.
    """
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("Expected a pd.DatetimeIndex.")
    if index.freq is None:
        raise ValueError("Index has no freq; regularise it first, e.g. ts.asfreq('15min').")
    if index.tz is not None:
        raise ValueError("Index must be time-zone naive.")
    return freq_hours(index.freq)


def _check_lambdas(lam: np.ndarray) -> np.ndarray:
    """Return ``lam`` unchanged if every frequency lies strictly inside (0, pi) rad/sample.

    Raises
    ------
    ValueError
        If any frequency is <= 0 or >= pi (at or above Nyquist).
    """
    if np.any(lam <= 0.0) or np.any(lam >= np.pi):
        raise ValueError(
            f"Cycle frequencies {lam} rad/sample are outside (0, pi): a band is at or above "
            "Nyquist for this step, or the frequency units are wrong."
        )
    return lam


def lambdas_from_rph(omega_rph: Iterable[float], step_h: float) -> np.ndarray:
    """Convert frequencies in radians per hour to radians per sample.

    Parameters
    ----------
    omega_rph : iterable of float
        Frequencies in radians per hour.
    step_h : float
        Sampling step in hours (see :func:`step_hours`, :func:`freq_hours`).

    Returns
    -------
    np.ndarray
        Frequencies in radians per sample, each in (0, pi).

    Raises
    ------
    ValueError
        If any frequency is not strictly between 0 and pi radians per sample.
    """
    return _check_lambdas(np.asarray(list(omega_rph), dtype=float) * float(step_h))


def lambdas_from_cpd(cpd: Iterable[float], step_h: float) -> np.ndarray:
    """Convert frequencies in cycles per day to radians per sample.

    Parameters
    ----------
    cpd : iterable of float
        Frequencies in cycles per day.
    step_h : float
        Sampling step in hours.

    Returns
    -------
    np.ndarray
        Frequencies in radians per sample, each in (0, pi).

    Raises
    ------
    ValueError
        If any frequency is not strictly between 0 and pi radians per sample.
    """
    omega_rph =  np.asarray(list(cpd), dtype=float) * (2.0 * np.pi / 24.0)
    return lambdas_from_rph(omega_rph, step_h)
