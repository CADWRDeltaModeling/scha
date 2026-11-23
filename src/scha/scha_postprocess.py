
# -*- coding: utf-8 -*-
"""
Post-processing utilities for SCHA Gibbs output (Python port of sc_postprocess.R)

The functions summarise the traces returned by the Gibbs sampler
(:func:`scha.scha_gibbs.sc_gibbs`): posterior-mean or posterior-mode
parameters and the state-space models built from them, interpretation of state
paths as subtide, harmonic amplitude, phase and frequency, and summaries over
the sampler iterations after burn-in.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from scha import SCHAParams, build_scha_state_space
except Exception:
    SCHAParams = None
    build_scha_state_space = None

def _ensure_burn(n: int, nburn: int) -> int:
    """Return a valid burn-in count for ``n`` iterations.

    Parameters
    ----------
    n : int
        Number of iterations.
    nburn : int
        Requested burn-in. A negative value means ``n // 2``.

    Returns
    -------
    int
        ``nburn`` (or ``n // 2``) clipped to the range ``[0, n - 1]``.
    """
    if nburn < 0:
        nburn = n // 2
    nburn = max(0, min(nburn, n-1))
    return nburn

def _to_1d(x: np.ndarray) -> np.ndarray:
    """Return ``x`` as an array, flattening an ``(n, 1)`` array to ``(n,)``.

    Arrays of any other shape are returned unchanged.
    """
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    return x

def mode_int(arr: np.ndarray) -> Optional[int]:
    """Return the unique most frequent integer value of an array.

    Parameters
    ----------
    arr : numpy.ndarray
        Values to tabulate. Non-finite values are ignored and the rest are
        converted to ``int`` (truncating toward zero).

    Returns
    -------
    int or None
        The single most frequent value, or None if ``arr`` has no finite
        values or the maximum count is shared by more than one value.
    """
    a = np.asarray(arr)
    if a.size == 0:
        return None
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    ai = a.astype(int)
    vals, counts = np.unique(ai, return_counts=True)
    m = counts.max()
    winners = vals[counts == m]
    if winners.size == 1:
        return int(winners[0])
    return None

def col_modes(mat: np.ndarray) -> np.ndarray:
    """Return the :func:`mode_int` of each column of a 2D array.

    Parameters
    ----------
    mat : numpy.ndarray
        2D array of shape ``(n, ncol)``.

    Returns
    -------
    numpy.ndarray
        Object array of length ``ncol``; element ``j`` is ``mode_int`` of
        column ``j`` (an ``int`` or None).
    """
    M = np.asarray(mat)
    out = []
    for j in range(M.shape[1]):
        out.append(mode_int(M[:, j]))
    return np.array(out, dtype=object)

@dataclass
class SCParams:
    """Point estimates of the SCHA model parameters.

    Parameters
    ----------
    sigma2_eps : float
        Observation noise variance.
    sigma2_zeta : float
        Trend innovation variance.
    sigma2_kappa : numpy.ndarray
        Cycle innovation variance for each frequency, length ``len(freqs)``.
    rho : float
        Damping factor of the cycles.
    freqs : numpy.ndarray
        Cycle frequencies, in radians per sample.
    order_trend : int
        Order of the trend component.
    order_cycle : int
        Order of the stochastic cycles.
    filter_prior_val : float or None, optional
        Optional prior value attached to the parameters (default None). It is
        not used by the functions in this module.
    """
    sigma2_eps: float
    sigma2_zeta: float
    sigma2_kappa: np.ndarray
    rho: float
    freqs: np.ndarray
    order_trend: int
    order_cycle: int
    filter_prior_val: Optional[float] = None

def sc_params_from_gibbs(gibbs_out: Dict[str, object], nburn: int = -1,
                         filter_eval: Optional[Dict[str, object]] = None) -> SCParams:
    """Return posterior-mean parameters from Gibbs sampler output.

    Parameters
    ----------
    gibbs_out : dict
        Output of :func:`scha.scha_gibbs.sc_gibbs` (or a dict with the same
        keys). Uses ``"sigma2_epsilon"`` ``(niter,)``, ``"sigma2_zeta"``
        ``(niter,)``, ``"sigma2_kappa"`` ``(niter, nfreq)``, ``"freqs"``,
        ``"order_trend"`` and ``"order_cycle"``, and optionally ``"rho"``
        (``(niter,)`` or ``(niter, 1)``).
    nburn : int, optional
        Number of initial iterations to discard. A negative value (the
        default, -1) discards the first half; the value is limited to
        ``niter - 1``.
    filter_eval : dict, optional
        Accepted for interface compatibility with R; not used.

    Returns
    -------
    SCParams
        Parameters set to the means of the retained iterations. If ``"rho"``
        is absent from ``gibbs_out``, ``rho`` is 1.0. ``filter_prior_val``
        is None.
    """
    s2e = np.asarray(gibbs_out["sigma2_epsilon"])
    n = s2e.shape[0]
    nb = _ensure_burn(n, nburn)
    s2z = np.asarray(gibbs_out["sigma2_zeta"])
    s2k = np.asarray(gibbs_out["sigma2_kappa"])

    sigma2_eps = float(s2e[nb:].mean())
    sigma2_zeta = float(s2z[nb:].mean())
    sigma2_kappa = s2k[nb:, :].mean(axis=0)

    rho_arr = gibbs_out.get("rho", None)
    if rho_arr is None:
        rho = 1.0
    else:
        rho = float(_to_1d(np.asarray(rho_arr))[nb:].mean())

    freqs = np.asarray(gibbs_out["freqs"], dtype=float)
    order_trend = int(gibbs_out["order_trend"])
    order_cycle = int(gibbs_out["order_cycle"])

    prior_val = None
    return SCParams(sigma2_eps, sigma2_zeta, sigma2_kappa, rho, freqs, order_trend, order_cycle, prior_val)

def sc_model_from_parms(parms: SCParams, sigma2_diffuse: float = 1e3):
    """Build the state-space model for a set of parameters.

    Parameters
    ----------
    parms : SCParams
        Parameters of the model.
    sigma2_diffuse : float, optional
        Diffuse prior variance stored in the result (default 1e3).

    Returns
    -------
    dict
        The state-space dictionary returned by
        :func:`scha.state_space.build_scha_state_space` for a
        :class:`scha.params.SCHAParams` built from ``parms`` (with
        ``m=order_trend``, ``n=order_cycle``, ``lambdas=freqs``), with an
        additional key ``"sigma2_diffuse"`` holding ``sigma2_diffuse``.

    Raises
    ------
    ImportError
        If ``scha.SCHAParams`` and ``scha.build_scha_state_space`` could not
        be imported when this module was loaded.
    """
    if build_scha_state_space is None or SCHAParams is None:
        raise ImportError("scha package not available to build the model.")
    p = SCHAParams(m=parms.order_trend, n=parms.order_cycle, rho=parms.rho,
                   lambdas=list(parms.freqs),
                   sigma2_eps=parms.sigma2_eps,
                   sigma2_zeta=parms.sigma2_zeta,
                   sigma2_kappa=list(map(float, parms.sigma2_kappa)))
    ss = build_scha_state_space(p)
    ss["sigma2_diffuse"] = float(sigma2_diffuse)
    return ss

def sc_model_from_gibbs(gibbs_out: Dict[str, object], nburn: int = -1, sigma2_diffuse: float = 1e3):
    """Build the state-space model at the posterior-mean parameters.

    Parameters
    ----------
    gibbs_out : dict
        Output of :func:`scha.scha_gibbs.sc_gibbs`; see
        :func:`sc_params_from_gibbs` for the keys used.
    nburn : int, optional
        Number of initial iterations to discard; negative (default -1) means
        the first half.
    sigma2_diffuse : float, optional
        Diffuse prior variance stored in the result (default 1e3).

    Returns
    -------
    dict
        State-space dictionary as returned by :func:`sc_model_from_parms` for
        the parameters of :func:`sc_params_from_gibbs`.
    """
    parms = sc_params_from_gibbs(gibbs_out, nburn=nburn, filter_eval=None)
    return sc_model_from_parms(parms, sigma2_diffuse=sigma2_diffuse)

def sc_model_from_gibbs_mode(gibbs_out: Dict[str, object], nburn: int = -1):
    """Build the state-space model at the posterior-mode parameters.

    Parameters
    ----------
    gibbs_out : dict
        Output of :func:`scha.scha_gibbs.sc_gibbs`; see
        :func:`sc_params_from_gibbs` for the keys used.
    nburn : int, optional
        Number of initial iterations to discard; negative (default -1) means
        the first half.

    Returns
    -------
    dict
        State-space dictionary as returned by :func:`sc_model_from_parms`
        (with its default ``sigma2_diffuse``).

    Notes
    -----
    Each parameter (``sigma2_epsilon``, ``sigma2_zeta``, each column of
    ``sigma2_kappa`` and, if present, ``rho``) is set to the midpoint of the
    fullest bin of a 50-bin histogram of its retained draws. If ``"rho"`` is
    absent, ``rho`` is 1.0.
    """
    s2e = np.asarray(gibbs_out["sigma2_epsilon"])
    s2z = np.asarray(gibbs_out["sigma2_zeta"])
    s2k = np.asarray(gibbs_out["sigma2_kappa"])
    n = s2e.shape[0]
    nb = _ensure_burn(n, nburn)

    def cont_mode(x: np.ndarray, bins: int = 50) -> float:
        """Return the midpoint of the fullest histogram bin of ``x``."""
        hist, edges = np.histogram(x, bins=bins)
        i = int(np.argmax(hist))
        return float(0.5 * (edges[i] + edges[i+1]))

    sigma2_eps = cont_mode(s2e[nb:])
    sigma2_zeta = cont_mode(s2z[nb:])
    sigma2_kappa = np.apply_along_axis(cont_mode, 0, s2k[nb:, :])

    rho_arr = gibbs_out.get("rho", None)
    if rho_arr is None:
        rho = 1.0
    else:
        rho = cont_mode(_to_1d(np.asarray(rho_arr))[nb:])

    freqs = np.asarray(gibbs_out["freqs"], dtype=float)
    order_trend = int(gibbs_out["order_trend"])
    order_cycle = int(gibbs_out["order_cycle"])

    parms = SCParams(sigma2_eps, sigma2_zeta, sigma2_kappa, rho, freqs, order_trend, order_cycle, None)
    return sc_model_from_parms(parms)

def reconstruct_state(state: np.ndarray, order_trend: int, order_cycle: int, nfreq: int):
    """Reconstruct subtide and full signal from a state path.

    Parameters
    ----------
    state : numpy.ndarray
        State path, shape ``(T, nstate)``, with the trend states first
        followed by ``nfreq`` cycle blocks of ``2 * order_cycle`` states each.
    order_trend : int
        Order of the trend component.
    order_cycle : int
        Order of the stochastic cycles.
    nfreq : int
        Number of cycle frequencies.

    Returns
    -------
    dict
        With keys:

        ``"subtide"``
            ``state[:, 0]``, shape ``(T,)``.
        ``"full"``
            Subtide plus the first state of each cycle block, shape ``(T,)``.
        ``"diurnal"``
            ``state[:, order_trend]``, the first state of the first cycle
            block, shape ``(T,)``.
    """
    state = np.asarray(state)
    subtide = state[:, 0]
    diurnal = state[:, 0 + order_trend]
    recon = subtide.copy()
    for icycle in range(nfreq):
        j0 = order_trend + icycle * 2 * order_cycle
        recon += state[:, j0]
    return {"subtide": subtide, "full": recon, "diurnal": diurnal}

def interpret_state(state: np.ndarray, order_trend: int, order_cycle: int, nfreq: int, flabels: List[str]):
    """Convert a state path to subtide, frequency, amplitude and phase series.

    Parameters
    ----------
    state : numpy.ndarray
        State path, shape ``(T, nstate)``; layout as in
        :func:`reconstruct_state`.
    order_trend : int
        Order of the trend component.
    order_cycle : int
        Order of the stochastic cycles.
    nfreq : int
        Number of cycle frequencies.
    flabels : list of str
        Labels of the frequencies. If empty (or shorter than ``nfreq``), the
        label of frequency ``jj`` (0-based) falls back to ``f{jj + 1}``.

    Returns
    -------
    dict
        With keys:

        ``"fitted"``
            The :func:`reconstruct_state` result.
        ``"subtide"``
            ``state[:, 0]``, shape ``(T,)``.
        ``"freq"``
            Dict label -> first state of the cycle block, shape ``(T,)``.
        ``"amplitude"``
            Dict label -> ``sqrt(psi**2 + psi_star**2)``, shape ``(T,)``,
            where ``psi`` and ``psi_star`` are the first two states of the
            cycle block.
        ``"phase"``
            Dict label -> ``arctan2(psi_star, psi)`` in radians, shape
            ``(T,)``.

    Notes
    -----
    The phase is ``atan2(psi*, psi)``, the same quantity stored by the R
    ``sc_gibbs``. It is the negative of the instantaneous phase of
    ``z = psi - i psi*`` (see section 10 of the design document); for the
    phase-lag convention see :mod:`scha.phase`.
    """
    state = np.asarray(state)
    fitted = reconstruct_state(state, order_trend, order_cycle, nfreq)
    subtide = state[:, 0]
    freq = {}
    amplitude = {}
    phase = {}
    for jj in range(nfreq):
        label = flabels[jj] if flabels and jj < len(flabels) else f"f{jj+1}"
        jndx = order_trend + jj * 2 * order_cycle
        pair = state[:, jndx:jndx+2]
        amplitude[label] = np.sqrt(np.sum(pair**2, axis=1))
        phase[label] = np.arctan2(pair[:, 1], pair[:, 0])
        freq[label] = state[:, jndx]
    return {"fitted": fitted, "subtide": subtide, "freq": freq, "amplitude": amplitude, "phase": phase}

def aggregate_iter(x: np.ndarray, nburn: int = -1, is_angle: bool = False,
                   quantiles: Optional[List[float]] = None) -> np.ndarray:
    """Summarise sampler iterations after burn-in.

    Parameters
    ----------
    x : numpy.ndarray
        Traces, shape ``(niter, ncol)``. A 1D array is treated as a single
        iteration (shape ``(1, len(x))``).
    nburn : int, optional
        Number of initial iterations to discard; negative (default -1) means
        the first half. The value is limited to ``niter - 1``.
    is_angle : bool, optional
        If True, ``x`` holds angles in radians and the circular mean is
        returned (default False).
    quantiles : list of float, optional
        Quantile levels in ``[0, 1]``; if given, quantiles are returned
        instead of the mean.

    Returns
    -------
    numpy.ndarray
        If ``quantiles`` is None: the mean over the retained iterations (the
        circular mean ``angle(mean(exp(1j * x)))`` if ``is_angle``), shape
        ``(ncol,)``. Otherwise the quantiles over the retained iterations,
        shape ``(len(quantiles), ncol)``.

    Raises
    ------
    ValueError
        If ``is_angle`` is True and ``quantiles`` is given.
    """
    X = np.asarray(x)
    if X.ndim == 1:
        X = X[None, :]
    niter = X.shape[0]
    nb = _ensure_burn(niter, nburn)

    if is_angle:
        if quantiles is not None:
            raise ValueError("quantiles not available for angles")
        z = np.exp(1j * X[nb:, :])
        ave = np.angle(np.mean(z, axis=0))
        return ave
    else:
        if quantiles is None:
            return X[nb:, :].mean(axis=0)
        else:
            qs = np.quantile(X[nb:, :], q=quantiles, axis=0)
            return qs

def summarize_scout_result(sc_out: Dict[str, object], interp: Dict[str, object],
                           out_dt: float = 0.25, nburn: int = 10,
                           outfile: Optional[str] = None) -> pd.DataFrame:
    """Tabulate parameter-based and posterior summaries of a Gibbs run.

    Parameters
    ----------
    sc_out : dict
        Gibbs output with thinned traces: ``"subtide_thinned"`` (array
        ``(niter, ncol)``) and the dicts ``"freq_thinned"``,
        ``"amplitude_thinned"`` and ``"phase_thinned"`` mapping each
        frequency label to an array ``(niter, ncol)``. The frequency labels
        are taken from the keys of ``"amplitude_thinned"`` (none if that
        entry is missing or empty).
    interp : dict
        Result of :func:`interpret_state` for the parameter-based fit, with
        ``"subtide"`` of length ``nout + 1`` and ``"freq"``, ``"amplitude"``
        and ``"phase"`` dicts keyed by label; the first element of each
        series is dropped.
    out_dt : float, optional
        Time step of the output (default 0.25); the ``times`` column is
        ``arange(nout) * out_dt``, with ``nout = len(interp["subtide"]) - 1``.
    nburn : int, optional
        Number of initial thinned iterations to discard (default 10), passed
        to :func:`aggregate_iter`.
    outfile : str, optional
        If given, the table is also written to this path as CSV without the
        index.

    Returns
    -------
    pandas.DataFrame
        One row per output time with columns ``times``, ``subtide_parm``,
        ``subtide_postmean``, ``subtide25``, ``subtide50``, ``subtide75`` and,
        for each label ``flab``: the columns ``{flab}_val_`` and ``{flab}_amp_``
        followed by ``parm``, ``postmean``, ``25``, ``50`` or ``75``
        (parameter-based value, posterior mean and quartiles), and
        ``{flab}_phase_parm`` and ``{flab}_phase_postmean`` (the latter a
        circular mean).
    """
    flabels = list((sc_out.get("amplitude_thinned") or {}).keys())

    subtide = np.asarray(interp["subtide"])
    nplusone = subtide.shape[0]
    nout = nplusone - 1
    times = np.arange(nout, dtype=float) * out_dt

    ip1a = aggregate_iter(np.asarray(sc_out["subtide_thinned"]), nburn=nburn)
    ip1b = aggregate_iter(np.asarray(sc_out["subtide_thinned"]), nburn=nburn, quantiles=[0.25, 0.5, 0.75])

    archive = {
        "times": times,
        "subtide_parm": subtide[1:nplusone],
        "subtide_postmean": ip1a,
        "subtide25": ip1b[0, :],
        "subtide50": ip1b[1, :],
        "subtide75": ip1b[2, :],
    }

    for flab in flabels:
        postmean = aggregate_iter(np.asarray(sc_out["freq_thinned"][flab]), nburn=nburn)
        anal = aggregate_iter(np.asarray(sc_out["freq_thinned"][flab]), nburn=nburn, quantiles=[0.25, 0.5, 0.75])
        archive[f"{flab}_val_parm"] = np.asarray(interp["freq"][flab])[1:nplusone]
        archive[f"{flab}_val_postmean"] = postmean[1:nplusone]
        archive[f"{flab}_val_25"] = anal[0, :]
        archive[f"{flab}_val_50"] = anal[1, :]
        archive[f"{flab}_val_75"] = anal[2, :]

        postmean = aggregate_iter(np.asarray(sc_out["amplitude_thinned"][flab]), nburn=nburn)
        anal = aggregate_iter(np.asarray(sc_out["amplitude_thinned"][flab]), nburn=nburn, quantiles=[0.25, 0.5, 0.75])
        archive[f"{flab}_amp_parm"] = np.asarray(interp["amplitude"][flab])[1:nplusone]
        archive[f"{flab}_amp_postmean"] = postmean[1:nplusone]
        archive[f"{flab}_amp_25"] = anal[0, :]
        archive[f"{flab}_amp_50"] = anal[1, :]
        archive[f"{flab}_amp_75"] = anal[2, :]

        postmean = aggregate_iter(np.asarray(sc_out["phase_thinned"][flab]), nburn=nburn, is_angle=True)
        archive[f"{flab}_phase_parm"] = np.asarray(interp["phase"][flab])[1:nplusone]
        archive[f"{flab}_phase_postmean"] = postmean[1:nplusone]

    df = pd.DataFrame(archive)
    if outfile:
        df.to_csv(outfile, index=False)
    return df
