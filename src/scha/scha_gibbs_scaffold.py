
"""
Gibbs & Cross-Validation Scaffolding for SCHA (near-reference)

This module is a thin, pandas-friendly wrapper that organizes the Gibbs sampler
and the leave-N-out (tidal-fold) cross-validation as function calls, while
delegating the heavy math to the core linear-Gaussian routines.

Notes
-----
As it stands the module defines only the prior and sampler configuration
dataclasses, :func:`prepare_observations` and :func:`inv_gamma_sample`; it
contains no sampler or cross-validation code.
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
    """Hyperparameters ``(a, b)`` for the variance priors.

    Parameters
    ----------
    a_eps, b_eps : float, optional
        Hyperparameters ``a`` and ``b`` for the observation variance
        ``sigma2_eps`` (default 1.0 each).
    a_zeta, b_zeta : float, optional
        Hyperparameters ``a`` and ``b`` for the trend innovation variance
        ``sigma2_zeta`` (default 1.0 each).
    a_kappa, b_kappa : float, optional
        Hyperparameters ``a`` and ``b`` for the cycle innovation variance
        ``sigma2_kappa`` (default 1.0 each).

    Notes
    -----
    This class only stores the values; nothing in this module uses them.
    """
    a_eps: float = 1.0
    b_eps: float = 1.0
    a_zeta: float = 1.0
    b_zeta: float = 1.0
    a_kappa: float = 1.0
    b_kappa: float = 1.0

@dataclass
class SamplerConfig:
    """Configuration of the Gibbs sampler run.

    Parameters
    ----------
    n_iter : int, optional
        Total number of iterations (default 2000).
    burn : int, optional
        Number of burn-in iterations (default 1000).
    thin : int, optional
        Thinning interval (default 1).
    seed : int or None, optional
        Random seed (default None).
    draw_states_method : str, optional
        State-drawing method; the code comment gives the choices ``"ck"`` or
        ``"dk"`` (default ``"ck"``).
    store_states_every : int or None, optional
        Interval at which states are stored (default None).

    Notes
    -----
    This class only stores the values; nothing in this module uses them.
    """
    n_iter: int = 2000
    burn: int = 1000
    thin: int = 1
    seed: Optional[int] = None
    draw_states_method: str = "ck"  # "ck" or "dk"
    store_states_every: Optional[int] = None

def prepare_observations(y: Union[pd.Series, pd.DataFrame]) -> Tuple[np.ndarray, Optional[pd.Index]]:
    """Convert a series of observations to a float array and its index.

    Parameters
    ----------
    y : pandas.Series or pandas.DataFrame
        Observations. A DataFrame must have exactly one column, which is used.

    Returns
    -------
    values : numpy.ndarray
        Observation values as a 1D float array (a view of the input data
        where possible).
    index : pandas.Index
        Index of the series.

    Raises
    ------
    ValueError
        If ``y`` is a DataFrame with other than one column.
    TypeError
        If ``y`` is neither a Series nor a DataFrame.
    """
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected a single-column DataFrame.")
        y = y.iloc[:, 0]
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a Series or single-column DataFrame.")
    arr = y.to_numpy(dtype=float, copy=False)
    return arr, y.index

def inv_gamma_sample(a: float, b: float, rng: np.random.Generator) -> float:
    """Sample from an inverse-gamma distribution.

    Parameters
    ----------
    a : float
        Shape parameter of the underlying gamma draw.
    b : float
        Numerator of the draw. This is the rate of the inverse-gamma
        distribution (``b / g`` with ``g ~ Gamma(a, scale=1)``, equivalently
        ``1 / Gamma(a, scale=1/b)``).
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    float
        One draw ``b / g``. No argument validation is performed.
    """
    return float(b / rng.gamma(shape=a, scale=1.0))
