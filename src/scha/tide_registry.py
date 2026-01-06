
# -*- coding: utf-8 -*-
"""
Tide frequency registry for SCHA (Python)
----------------------------------------
- Provides canonical tidal constituent names with radians-per-hour values,
  following your R `tide_freq_radians_hr()` list.
- Adds D1..D6 aliases mapping to M1..M6 (as requested), while keeping M1..M8, K1, O1, etc.
- Utilities to select subsets and rescale to your sampling rate (e.g., 4 samples/hr for 15-min data).
"""

from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import numpy as np

# Canonical radians per hour (from your R `tide_freq_radians_hr()`), kept verbatim.
# fmt: off
_RPH = {
    "LOW": 2.0 * np.pi / (4 * 2.0 * 12.420612),
    "K1":  2.0 * np.pi / 23.93447213,
    "M2":  2.0 * np.pi / 12.4206012,
    "O1":  2.0 * np.pi / 25.81933871,
    "2Q1": 2.0 * np.pi / 28.00622,
    "P1":  2.0 * np.pi / 24.06589,
    "J1":  2.0 * np.pi / 23.09848,
    "OO1": 2.0 * np.pi / 22.30608,
    "2N2": 2.0 * np.pi / 12.90537,
    "NU2": 2.0 * np.pi / 12.62600,
    "K2":  2.0 * np.pi / 11.96723,
    "S2":  2.0 * np.pi / 12.0,
    "Q1":  2.0 * np.pi / 26.868350,
    "N2":  2.0 * np.pi / 12.65834751,
    "L2":  2.0 * np.pi / 12.19162085,
    "M4":  2.0 * np.pi / 6.210300601,
    "MK3": 2.0 * np.pi / 8.177140247,
    "MO3": 2.0 * np.pi / 8.38630296301,
    "M6":  2.0 * np.pi / 4.140200401,
    "MK5": 2.0 * np.pi / 4.930880215,
    "M05": 2.0 * np.pi / (12.4206012 * 4),
    "M1":  2.0 * np.pi / (12.4206012 * 2),
    "M3":  2.0 * np.pi / (12.4206012 * 2 / 3.0),
    "M5":  2.0 * np.pi / (12.4206012 * 2 / 5.0),
    "M8":  2.0 * np.pi / (12.4206012 * 2 / 8.0),
    "f":   2.0 * np.pi / 25.5,
    "f12": 2.0 * np.pi / 19.0,
}
# fmt: on

# Requested aliases: D1..D6 mirror M1..M6 (introduces redundancy by design).
_ALIASES = {
    "D1": "M1",
    "D2": "M2",
    "D3": "M3",
    "D4": "M4",
    "D5": "M5",
    "D6": "M6",
}

def all_names(include_aliases: bool = True) -> List[str]:
    names = list(_RPH.keys())
    if include_aliases:
        names = names + list(_ALIASES.keys())
    return names

def rph(name: str) -> float:
    """Radians per hour for a given canonical or alias name."""
    if name in _RPH:
        return float(_RPH[name])
    if name in _ALIASES:
        return float(_RPH[_ALIASES[name]])
    raise KeyError(f"Unknown tide name: {name}")

def select(names: Iterable[str], samples_per_hour: float = 1.0) -> Tuple[np.ndarray, List[str]]:
    """
    Select a set of tide names and convert to radians per sample.
    Returns (freqs_rad_per_sample, resolved_names) where aliases are resolved to their canonical names.
    """
    out_vals: List[float] = []
    out_names: List[str] = []
    for nm in names:
        if nm in _ALIASES:
            canon = _ALIASES[nm]
            val = _RPH[canon] / samples_per_hour
            out_vals.append(float(val))
            out_names.append(nm)  # keep the alias label as requested
        else:
            val = _RPH[nm] / samples_per_hour
            out_vals.append(float(val))
            out_names.append(nm)
    return np.array(out_vals, dtype=float), out_names

def label_for_value(val_rph: float, tol: float = 1e-9) -> str:
    """Find the canonical label for a given radians-per-hour value within tolerance."""
    for k, v in _RPH.items():
        if abs(v - val_rph) <= tol:
            return k
    return f"unknown({val_rph:.6g})"
