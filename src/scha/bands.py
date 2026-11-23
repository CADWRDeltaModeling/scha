"""Tidal band registry: model lines and named passband lines in cycles per day.

Band ``Dk`` is modelled by a stochastic cycle at k times the M1 frequency
(half the M2 frequency). Each band also carries a set of named lines, in cpd,
on which its passband requirement is checked (see :mod:`scha.requirements`).
The D1 set is the agreed cusp-room window, 0.83-1.13 cpd in 0.01 cpd steps.
The D2-D8 line choices are provisional (design notes, section 8).
"""
from __future__ import annotations

import re
from typing import Sequence, Union

import numpy as np

from .tide_registry import rph

MAX_BAND = 8


def _cpd(name: str) -> float:
    """Return the frequency of tidal constituent ``name`` in cycles per day."""
    return rph(name) * 24.0 / (2.0 * np.pi)


_M2, _N2, _S2, _K2, _O1, _K1 = (_cpd(k) for k in ("M2", "N2", "S2", "K2", "O1", "K1"))

BAND_LINES: dict[str, np.ndarray] = {
    "D1": np.round(np.arange(0.83, 1.1301, 0.01), 4),
    "D2": np.array([_cpd(k) for k in ("2N2", "N2", "NU2", "M2", "L2", "S2", "K2")]),
    "D3": np.array([_cpd("MO3"), _cpd("M3"), _cpd("MK3")]),
    "D4": np.array([_M2 + _N2, _cpd("M4"), _M2 + _S2, _M2 + _K2]),
    "D5": np.array([2 * _M2 + _O1, 2 * _M2 + _K1]),
    "D6": np.array([2 * _M2 + _N2, _cpd("M6"), 2 * _M2 + _S2, 2 * _M2 + _K2]),
    "D7": np.array([3 * _M2 + _O1, 3 * _M2 + _K1]),
    "D8": np.array([3 * _M2 + _N2, _cpd("M8"), 3 * _M2 + _S2, 3 * _M2 + _K2]),
}

BandSpec = Union[str, Sequence[str]]


def band_number(label: str) -> int:
    """Return k for a band label ``Dk``.

    Parameters
    ----------
    label : str
        Band label ``D1`` .. ``D8`` (``D`` followed by an integer between 1 and ``MAX_BAND``).

    Returns
    -------
    int
        The band number k.

    Raises
    ------
    ValueError
        If ``label`` is not of the form ``Dk`` with 1 <= k <= ``MAX_BAND``.
    """
    m = re.fullmatch(r"D(\d+)", label)
    if m is None or not 1 <= int(m.group(1)) <= MAX_BAND:
        raise ValueError(f"Unknown band {label!r}; expected D1..D{MAX_BAND}.")
    return int(m.group(1))


def band_cpd(label: str) -> float:
    """Return the model line of a band in cpd (k times M1).

    Parameters
    ----------
    label : str
        Band label ``D1`` .. ``D8``.

    Returns
    -------
    float
        Frequency of the band's model line in cycles per day.

    Raises
    ------
    ValueError
        If ``label`` is not a valid band label (see :func:`band_number`).
    """
    return band_number(label) * _cpd("M1")


def parse_bands(spec: BandSpec) -> list[str]:
    """Parse a band specification into an ordered list of labels.

    Parameters
    ----------
    spec : str or sequence of str
        ``"D1-D6"`` (inclusive range), ``"D1,D2,D4"`` or a sequence of labels.

    Returns
    -------
    list of str
        Labels in increasing frequency, without duplicates.

    Raises
    ------
    ValueError
        If a range is empty (``lo > hi``), no bands are given, or a label is not a valid band
        (see :func:`band_number`).
    """
    if isinstance(spec, str):
        s = spec.replace(" ", "")
        m = re.fullmatch(r"D(\d+)-D(\d+)", s)
        if m is not None:
            lo, hi = int(m.group(1)), int(m.group(2))
            if lo > hi:
                raise ValueError(f"Empty band range {spec!r}.")
            labels = [f"D{k}" for k in range(lo, hi + 1)]
        else:
            labels = [x for x in s.split(",") if x]
    else:
        labels = list(spec)
    if not labels:
        raise ValueError("No bands given.")
    ks = sorted({band_number(x) for x in labels})
    return [f"D{k}" for k in ks]


def parse_sacrificial(spec: Union[int, BandSpec, None], bands: Sequence[str]) -> list[str]:
    """Resolve which fitted bands carry no passband requirement.

    Parameters
    ----------
    spec : int, str, sequence of str or None
        An int k selects the highest k fitted bands. A band spec (see
        :func:`parse_bands`) names them. None or 0 means none.
    bands : sequence of str
        Fitted band labels.

    Returns
    -------
    list of str
        Sacrificial band labels.

    Raises
    ------
    ValueError
        If an int ``spec`` is outside 0..``len(bands)``, or a named band is not among ``bands``
        (or ``spec`` is not a valid band spec, see :func:`parse_bands`).
    """
    if spec is None:
        return []
    if isinstance(spec, (int, np.integer)):
        k = int(spec)
        if not 0 <= k <= len(bands):
            raise ValueError(f"Cannot sacrifice {k} of {len(bands)} bands.")
        return list(bands[len(bands) - k:])
    names = parse_bands(spec)
    missing = [b for b in names if b not in bands]
    if missing:
        raise ValueError(f"Sacrificial bands {missing} are not among the fitted bands {list(bands)}.")
    return names
