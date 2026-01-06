
# -*- coding: utf-8 -*-
"""
Post-processing utilities for SCHA Gibbs output (Python port of sc_postprocess.R)
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
    if nburn < 0:
        nburn = n // 2
    nburn = max(0, min(nburn, n-1))
    return nburn

def _to_1d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 2 and x.shape[1] == 1:
        return x[:, 0]
    return x

def mode_int(arr: np.ndarray) -> Optional[int]:
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
    M = np.asarray(mat)
    out = []
    for j in range(M.shape[1]):
        out.append(mode_int(M[:, j]))
    return np.array(out, dtype=object)

@dataclass
class SCParams:
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
    parms = sc_params_from_gibbs(gibbs_out, nburn=nburn, filter_eval=None)
    return sc_model_from_parms(parms, sigma2_diffuse=sigma2_diffuse)

def sc_model_from_gibbs_mode(gibbs_out: Dict[str, object], nburn: int = -1):
    s2e = np.asarray(gibbs_out["sigma2_epsilon"])
    s2z = np.asarray(gibbs_out["sigma2_zeta"])
    s2k = np.asarray(gibbs_out["sigma2_kappa"])
    n = s2e.shape[0]
    nb = _ensure_burn(n, nburn)

    def cont_mode(x: np.ndarray, bins: int = 50) -> float:
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
    state = np.asarray(state)
    subtide = state[:, 0]
    diurnal = state[:, 0 + order_trend]
    recon = subtide.copy()
    for icycle in range(nfreq):
        j0 = order_trend + icycle * 2 * order_cycle
        recon += state[:, j0]
    return {"subtide": subtide, "full": recon, "diurnal": diurnal}

def interpret_state(state: np.ndarray, order_trend: int, order_cycle: int, nfreq: int, flabels: List[str]):
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
