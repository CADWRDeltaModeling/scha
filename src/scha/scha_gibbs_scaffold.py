
"""
Gibbs & Cross-Validation Scaffolding for SCHA (near-reference)

This module is a thin, pandas-friendly wrapper that organizes the Gibbs sampler
and the leave-N-out (tidal-fold) cross-validation as function calls, while
delegating the heavy math to the core linear-Gaussian routines.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, Callable, List, Union
import numpy as np
import pandas as pd

# -----------------------------
# Priors & Sampler configuration
# -----------------------------

@dataclass
class VariancePriors:
    a_eps: float = 1.0
    b_eps: float = 1.0
    a_zeta: float = 1.0
    b_zeta: float = 1.0
    a_kappa: float = 1.0
    b_kappa: float = 1.0

@dataclass
class SamplerConfig:
    n_iter: int = 2000
    burn: int = 1000
    thin: int = 1
    seed: Optional[int] = None
    draw_states_method: str = "ck"  # "ck" or "dk"
    store_states_every: Optional[int] = None

def prepare_observations(y: Union[pd.Series, pd.DataFrame]) -> Tuple[np.ndarray, Optional[pd.Index]]:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame.")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a Series or single-column DataFrame.")
    arr = y.to_numpy(dtype=float, copy=False)
    return arr, y.index

def inv_gamma_sample(a: float, b: float, rng: np.random.Generator) -> float:
    return float(b / rng.gamma(shape=a, scale=1.0))
