"""Gridded hyperparameter cross-validation (Ateljevich and Wang 2023).

For each (rho, sigma2_zeta) on a grid, the remaining parameters (sigma2_eps,
sigma2_kappa) are estimated by maximum likelihood with rho and sigma2_zeta held
fixed. Folds withhold periodic gaps; each withheld value is predicted by the
smoothed signal of a fit to the data without that fold, and the score is the mean
squared error over all withheld observations.

This is the Python counterpart of stochCycle ``calculate_blocks`` and
``example/freeport_mn24_cv``; there the inner fit was a Gibbs sampler.
"""
from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from .fit import SCHAFit, fit
from .timestep import freq_hours, step_hours

logger = logging.getLogger("scha.crossval")

Duration = Union[int, str]


def _samples(d: Duration, step_h: float) -> int:
    """Convert a duration (samples as int, or a duration string such as ``"25h"``) to samples.

    Raises
    ------
    ValueError
        If a duration string is not a whole number of ``step_h`` hours.
    """
    if isinstance(d, (int, np.integer)):
        return int(d)
    k = freq_hours(d) / step_h
    if not np.isclose(k, round(k)):
        raise ValueError(f"Duration {d!r} is not a whole number of {step_h} h steps.")
    return int(round(k))


def cv_blocks(n: int, gap: int, spacing: int, first: int = 0, last: Optional[int] = None) -> list:
    """Return fold masks of periodic gaps (port of stochCycle ``calculate_blocks``).

    Fold k withholds ``[s, s + gap)`` for s = first + k * gap + j * spacing,
    j = 0, 1, ...; folds are generated while their first start is below
    ``first + spacing``. When ``spacing`` is a multiple of ``gap`` the folds tile
    the record after ``first`` exactly; otherwise the last fold overlaps the first
    (as in the stochCycle Freeport design, gap 100 and spacing 1001 samples).

    Parameters
    ----------
    n : int
        Record length (samples).
    gap : int
        Length of each withheld gap (samples).
    spacing : int
        Distance between gap starts within a fold (samples).
    first : int, optional
        First index that may be withheld (default 0).
    last : int, optional
        Gaps must end before this index (default n).

    Returns
    -------
    list of np.ndarray
        Integer positions withheld by each fold.

    Raises
    ------
    ValueError
        If ``gap`` is not positive or ``spacing`` is smaller than ``gap``.
    """
    if gap <= 0 or spacing < gap:
        raise ValueError("Need 0 < gap <= spacing.")
    last = n if last is None else last
    folds = []
    s0 = first
    while s0 < first + spacing:
        starts = np.arange(s0, n, spacing)
        starts = starts[starts + gap <= last]
        if starts.size:
            folds.append(np.concatenate([np.arange(s, s + gap) for s in starts]))
        s0 += gap
    return folds


def blocks(index: pd.DatetimeIndex, gap: Duration = "25h", spacing: Duration = "250h", first: Duration = 0) -> list:
    """Return fold masks with gap, spacing and offset given as durations (or samples).

    See :func:`cv_blocks`. Durations make the design independent of the step.

    Parameters
    ----------
    index : pd.DatetimeIndex
        Regular, naive index of the record (``freq`` set).
    gap : int or str, optional
        Length of each withheld gap: a number of samples or a duration such as ``"25h"``
        (default ``"25h"``).
    spacing : int or str, optional
        Distance between gap starts within a fold, same forms as ``gap`` (default ``"250h"``).
    first : int or str, optional
        First position that may be withheld, same forms as ``gap`` (default 0 samples).

    Returns
    -------
    list of np.ndarray
        Integer positions withheld by each fold, see :func:`cv_blocks`.

    Raises
    ------
    ValueError
        If ``index`` is not a regular naive DatetimeIndex, a duration is not a whole number of
        steps, or the design is invalid (see :func:`cv_blocks`).
    """
    h = step_hours(index)
    return cv_blocks(index.size, _samples(gap, h), _samples(spacing, h), first=_samples(first, h))


def _fold(task):
    """Predict the withheld values of one fold.

    Parameters
    ----------
    task : tuple
        ``(y, mask, kw, start)``: the series, the integer positions to withhold, a dict of
        keyword arguments for :func:`scha.fit.fit` that may contain ``refit`` (default True),
        and the full-record :class:`scha.fit.SCHAFit` used as the start. With ``refit`` the
        model is re-estimated on the data without the fold; otherwise the parameters of
        ``start`` are used unchanged.

    Returns
    -------
    mask : np.ndarray
        The withheld positions.
    np.ndarray
        Smoothed signal (``"fitted"`` of the decomposition) at those positions.
    """
    y, mask, kw, start = task
    yk = y.copy()
    yk.iloc[mask] = np.nan
    if kw.pop("refit", True):
        fk = fit(yk, start=start, workers=1, **kw)
    else:
        fk = SCHAFit(params=start.params, bands=start.bands, fit_freq=start.fit_freq, method=start.method,
                     requirements=start.requirements, fixed=start.fixed, y=yk)
    return mask, fk.decompose(yk)["fitted"].to_numpy()[mask]


def grid_cv(y: pd.Series, rho: Sequence[float], sigma2_zeta: Sequence[float], *, bands="D1-D6",
            order: tuple = (2, 3), folds: Optional[list] = None, gap: Duration = "25h",
            spacing: Duration = "250h", first: Duration = "100h", refit_folds: bool = True,
            requirements=None, maxiter: int = 80, workers: Optional[int] = None) -> pd.DataFrame:
    """Score a (rho, sigma2_zeta) grid by withheld-gap prediction error.

    Parameters
    ----------
    y : pd.Series
        Observations (regular DatetimeIndex with freq; NaN gaps).
    rho, sigma2_zeta : sequence of float
        Grid values (native to the step of y).
    bands : str or sequence of str, optional
        Tidal bands, default ``"D1-D6"``, as in :func:`scha.fit.fit`.
    order : (int, int), optional
        Trend and cycle order, default (2, 3), as in :func:`scha.fit.fit`.
    folds : list of np.ndarray, optional
        Withheld positions per fold; default :func:`blocks` with ``gap``,
        ``spacing`` and ``first``.
    gap, spacing, first : int or str, optional
        Fold design for :func:`blocks` (defaults ``"25h"``, ``"250h"`` and ``"100h"``); ignored
        when ``folds`` is given.
    refit_folds : bool, optional
        Re-estimate sigma2_eps and sigma2_kappa for each fold (as the R version
        did); False smooths each fold with the full-record fit (much faster). Default True.
    requirements : scha.requirements.Requirements or dict, optional
        Used only to report gain margins of each cell; the fits are plain ML.
    maxiter : int, optional
        Optimiser iteration limit of each inner fit (default 80).
    workers : int, optional
        Processes over folds; default min(cpu count, number of folds).

    Returns
    -------
    pd.DataFrame
        One row per cell: rho, sigma2_zeta, cv_mse, n_withheld, loglik, sigma2_eps,
        sigma2_kappa_<band>, min_margin, requirements_met, crossover_cpd.
    """
    if folds is None:
        folds = blocks(y.index, gap=gap, spacing=spacing, first=first)
    yv = y.to_numpy()
    nw = workers or min(os.cpu_count() or 1, len(folds))
    pool = ProcessPoolExecutor(max_workers=nw) if nw > 1 else None
    rows = []
    start = None
    try:
        for r in rho:
            for z in sigma2_zeta:
                fixed = {"rho": float(r), "sigma2_zeta": float(z)}
                kw = dict(bands=bands, order=order, method="ml", requirements=requirements, fixed=fixed,
                          maxiter=maxiter)
                full = fit(y, start=start, workers=1 if pool is not None else None, **kw)
                start = full
                pred = np.full(y.size, np.nan)
                tasks = [(y, mask, dict(kw, refit=refit_folds), full) for mask in folds]
                results = pool.map(_fold, tasks) if pool is not None else map(_fold, tasks)
                for mask, vals in results:
                    pred[mask] = vals
                held = np.isfinite(pred) & np.isfinite(yv)
                mse = float(np.mean((yv[held] - pred[held]) ** 2))
                p, d = full.params, full.diagnostics
                row = dict(rho=r, sigma2_zeta=z, cv_mse=mse, n_withheld=int(held.sum()), loglik=d["loglik"],
                           sigma2_eps=p.sigma2_eps)
                row.update({f"sigma2_kappa_{b}": k for b, k in zip(full.bands, p.sigma2_kappa)})
                row.update(min_margin=d["min_margin"], requirements_met=d["requirements_met"],
                           crossover_cpd=d["crossover_cpd"])
                rows.append(row)
                logger.info("rho=%.5f sigma2_zeta=%.3e cv_mse=%.5g loglik=%.2f", r, z, mse, d["loglik"])
    finally:
        if pool is not None:
            pool.shutdown()
    return pd.DataFrame(rows)
