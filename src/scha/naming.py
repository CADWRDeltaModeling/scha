# -*- coding: utf-8 -*-
"""
Naming helpers for SCHA frequency outputs.
- Build preferred labels for a given list of freqs (radians per sample) and sampling rate.
- Optionally expand with D1..D6 aliases; preferred order is user-controllable.
"""

from __future__ import annotations
from typing import Iterable, List, Dict, Tuple
import numpy as np
from .tide_registry import rph, select, label_for_value, _ALIASES as _D_ALIASES

def preferred_labels(freqs_rad_per_sample: Iterable[float],
                     samples_per_hour: float,
                     use_d_aliases: bool = True) -> List[str]:
    """
    Given numeric freqs (rad/sample) and the sampling rate, return preferred tide labels.
    If a frequency matches an M1..M6, and use_d_aliases is True, return its Dk alias.
    Otherwise return the canonical (M*, K1, O1, ...) name.
    """
    labels: List[str] = []
    freqs_rad_per_sample = list(map(float, freqs_rad_per_sample))
    # Build reverse lookup in rph (rad/hr) space
    rph_values = {nm: rph(nm) for nm in [
        "M1","M2","M3","M4","M5","M6","M8","M05","K1","O1","2Q1","P1","J1","OO1",
        "2N2","NU2","K2","S2","Q1","N2","L2","MK3","MO3","MK5","f","f12","LOW"
    ]}
    for val_rs in freqs_rad_per_sample:
        val_rph = val_rs * samples_per_hour
        # find closest canonical match
        canon = min(rph_values.items(), key=lambda kv: abs(kv[1] - val_rph))[0]
        if use_d_aliases and canon in ["M1","M2","M3","M4","M5","M6"]:
            # map to D1..D6
            inv = {v: k for k, v in _D_ALIASES.items()}
            labels.append(inv[canon])
        else:
            labels.append(canon)
    return labels

def attach_labels_to_gibbs_out(gibbs_out: Dict[str, object],
                               samples_per_hour: float,
                               use_d_aliases: bool = True) -> Dict[str, object]:
    """
    Return a shallow copy of `gibbs_out` adding a `flabels` field derived from `freqs`.
    """
    out = dict(gibbs_out)
    freqs_rs = gibbs_out.get("freqs")
    if freqs_rs is None:
        return out
    out["flabels"] = preferred_labels(freqs_rs, samples_per_hour, use_d_aliases=use_d_aliases)
    return out
