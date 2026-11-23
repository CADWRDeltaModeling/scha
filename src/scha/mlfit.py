"""Maximum-likelihood estimation of SCHA parameters, optionally under gain requirements.

The parameters are the original ones (sigma2_eps, sigma2_zeta, sigma2_kappa, rho).
Internally the optimiser works on logs of the variances and log(1 - rho). When no
variance is fixed, sigma2_eps is profiled out and the free variables are the log
ratios to sigma2_eps (concentrated likelihood, exact). When any variance is
fixed, the diffuse likelihood is maximised over the free logs.

The trend prior is centred on the data mean with variance
:func:`scha.fastll.diffuse_variance` (large-variance approximation to the exact
diffuse prior), and the first m observations are left out of the likelihood, so
fitted parameters, the reported loglik (up to n log s) and the states do not
depend on the data units.

Steps: diagonal preconditioning by finite-difference curvature; if the start
violates the requirements, a feasibility phase moves to the nearest feasible
point without the likelihood; then SLSQP (L-BFGS-B when unconstrained).
Gradients are forward differences, evaluated in parallel when ``workers > 1``.

With ``workers > 1`` on Windows, scripts that call this must guard their entry
point with ``if __name__ == "__main__":``.
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy.optimize import minimize

from .fastll import diffuse_variance, scha_loglik_concentrated, scha_loglik_diffuse
from .params import SCHAParams
from .requirements import CompiledRequirements

SIGMA2_DIFFUSE_REL = 1.0e6
FD_STEP = 1.0e-3
_LL_SCALE = 100.0
_RATIO_BOUNDS = (-60.0, 10.0)
_LOG1MRHO_BOUNDS = (np.log(1.0e-5), np.log(0.3))
_ABS_HALFWIDTH = 30.0


def x_from_params(p: SCHAParams) -> np.ndarray:
    """Return the full log parameter vector.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Parameters.

    Returns
    -------
    np.ndarray
        [log sigma2_eps, log sigma2_zeta, log sigma2_kappa..., log(1 - rho)].
    """
    return np.concatenate([[np.log(p.sigma2_eps), np.log(p.sigma2_zeta)], np.log(p.sigma2_kappa),
                           [np.log(1.0 - p.rho)]])


@dataclass
class Problem:
    """Picklable description of the optimisation variables.

    The full log vector x is laid out as in :func:`x_from_params`. ``free`` marks
    the entries the optimiser moves (z = x[free]); in the concentrated case z holds
    log ratios to sigma2_eps and sigma2_eps itself is profiled out.

    Attributes
    ----------
    m, n : int
        Trend and cycle orders.
    lambdas : list of float
        Cycle frequencies in radians per sample.
    x0 : np.ndarray
        Start (and values of fixed entries), full log vector.
    free : np.ndarray of bool
        Entries of x estimated.
    concentrated : bool
        Profile sigma2_eps out (no variance fixed).
    sigma2_diffuse : float
        Trend prior variance for the ordinary (diffuse) likelihood.
    """

    m: int
    n: int
    lambdas: list
    x0: np.ndarray
    free: np.ndarray
    concentrated: bool
    sigma2_diffuse: float = 1.0

    @property
    def nc(self) -> int:
        """Number of cycles."""
        return len(self.lambdas)

    def params(self, x: np.ndarray) -> SCHAParams:
        """Return the parameters from a full log vector.

        Parameters
        ----------
        x : np.ndarray
            Full log vector (:func:`x_from_params` layout).

        Returns
        -------
        scha.params.SCHAParams
        """
        nc = self.nc
        return SCHAParams(m=self.m, n=self.n, rho=float(1.0 - np.exp(x[2 + nc])), lambdas=list(self.lambdas),
                          sigma2_eps=float(np.exp(x[0])), sigma2_zeta=float(np.exp(x[1])),
                          sigma2_kappa=[float(v) for v in np.exp(x[2:2 + nc])])

    def z0(self) -> np.ndarray:
        """Return the start of the optimiser variables.

        Returns
        -------
        np.ndarray
            Free entries of x0 (as log ratios to sigma2_eps when concentrated).
        """
        return self.full_ratio(self.x0)[self.free] if self.concentrated else self.x0[self.free]

    @staticmethod
    def full_ratio(x: np.ndarray) -> np.ndarray:
        """Return the full log vector with the variances divided by sigma2_eps.

        Parameters
        ----------
        x : np.ndarray
            Full log vector.

        Returns
        -------
        np.ndarray
            Same layout with log sigma2_eps = 0 and the other variances as log ratios.
        """
        r = x.copy()
        r[1:-1] -= x[0]
        r[0] = 0.0
        return r

    def x_ratio(self, z: np.ndarray) -> np.ndarray:
        """Return the ratio-scaled full log vector for trial z (the gains depend only on this).

        Parameters
        ----------
        z : np.ndarray
            Optimiser variables.

        Returns
        -------
        np.ndarray
            Full log vector with variances as log ratios to sigma2_eps.
        """
        if self.concentrated:
            r = self.full_ratio(self.x0)
            r[self.free] = z
            return r
        x = self.x0.copy()
        x[self.free] = z
        return self.full_ratio(x)

    def objective(self, z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """Return the objective log-likelihood at z.

        Parameters
        ----------
        z : np.ndarray
            Optimiser variables.
        y : np.ndarray
            Centred observations (NaN at gaps).

        Returns
        -------
        loglik : float
            Concentrated or diffuse log-likelihood.
        sigma2_eps : float
            Profiled (concentrated) or current sigma2_eps.
        """
        if self.concentrated:
            pr = self.params(self.x_ratio(z))
            return scha_loglik_concentrated(pr, y, sigma2_diffuse_rel=SIGMA2_DIFFUSE_REL, scale_rho=pr.rho)
        x = self.x0.copy()
        x[self.free] = z
        p = self.params(x)
        return scha_loglik_diffuse(p, y, self.sigma2_diffuse, scale_rho=p.rho), p.sigma2_eps

    def x_full(self, z: np.ndarray, sigma2_eps: float) -> np.ndarray:
        """Return the full log vector in absolute units for trial z.

        Parameters
        ----------
        z : np.ndarray
            Optimiser variables.
        sigma2_eps : float
            Noise variance (used to undo the ratios when concentrated).

        Returns
        -------
        np.ndarray
            Full log vector in the :func:`x_from_params` layout.
        """
        if self.concentrated:
            x = self.x_ratio(z)
            x[0] = np.log(sigma2_eps)
            x[1:-1] += x[0]
            return x
        x = self.x0.copy()
        x[self.free] = z
        return x

    def bounds(self) -> list:
        """Return the box bounds for the optimiser variables.

        Returns
        -------
        list of (float, float)
            Ratio bounds when concentrated, otherwise +/- 30 log units around the
            start; log(1 - rho) is limited to (log 1e-5, log 0.3).
        """
        nc = self.nc
        if self.concentrated:
            b = [(0.0, 0.0)] + [_RATIO_BOUNDS] * (1 + nc) + [_LOG1MRHO_BOUNDS]
        else:
            b = [(v - _ABS_HALFWIDTH, v + _ABS_HALFWIDTH) for v in self.x0[:-1]] + [_LOG1MRHO_BOUNDS]
        return [bb for bb, f in zip(b, self.free) if f]


_Y: Optional[np.ndarray] = None
_PROB: Optional[Problem] = None


def _init_worker(prob: Problem, y: np.ndarray) -> None:
    """Store the problem and data in module globals (process-pool initializer)."""
    global _Y, _PROB
    _PROB, _Y = prob, y


def _eval_ll(z: np.ndarray) -> float:
    """Return the objective log-likelihood at ``z`` for the stored problem and data.

    Non-finite values are replaced by -1e7.
    """
    ll = _PROB.objective(z, _Y)[0]
    # non-PSD filter at extreme trial points: finite penalty so the line search backs off
    return ll if np.isfinite(ll) else -1.0e7


@dataclass
class MLResult:
    """Outcome of :func:`scha.mlfit.fit_ml`.

    Attributes
    ----------
    params : scha.params.SCHAParams
        Estimated parameters (lambdas in radians per sample).
    loglik : float
        Diffuse log-likelihood at ``params`` (trend prior variance
        :func:`scha.fastll.diffuse_variance` of the centred data).
    loglik_objective : float
        Objective log-likelihood at the optimum: concentrated if ``concentrated``, otherwise
        diffuse.
    concentrated : bool
        True if sigma2_eps was profiled out.
    constrained : bool
        True if gain requirements were imposed.
    success : bool
        Optimiser success flag.
    message : str
        Optimiser termination message.
    nit : int
        Number of optimiser iterations.
    runtime_s : float
        Wall time in seconds.
    min_margin : float
        Smallest requirement margin at ``params`` (log-gain margins, see
        :meth:`scha.requirements.CompiledRequirements.margins`); NaN if unconstrained.
    n_active : int
        Number of requirement margins below 1e-3 at ``params``; 0 if unconstrained.
    history : list of float
        Objective log-likelihood after each optimiser iteration.
    """
    params: SCHAParams
    loglik: float
    loglik_objective: float
    concentrated: bool
    constrained: bool
    success: bool
    message: str
    nit: int
    runtime_s: float
    min_margin: float = float("nan")
    n_active: int = 0
    history: list = field(default_factory=list)


def make_problem(p0: SCHAParams, fixed: Optional[dict] = None) -> Problem:
    """Build the optimisation problem from a start and a set of fixed parameters.

    Parameters
    ----------
    p0 : scha.params.SCHAParams
        Start; fixed entries are overwritten by ``fixed``.
    fixed : dict, optional
        Any of ``rho``, ``sigma2_eps``, ``sigma2_zeta`` (floats) and
        ``sigma2_kappa`` (dict index -> value, or a full list).

    Returns
    -------
    scha.mlfit.Problem
        Concentrated when no variance is fixed.

    Raises
    ------
    ValueError
        For unknown names, out-of-range kappa indices, or when nothing is free.
    """
    fixed = dict(fixed or {})
    unknown = set(fixed) - {"rho", "sigma2_eps", "sigma2_zeta", "sigma2_kappa"}
    if unknown:
        raise ValueError(f"Cannot fix {sorted(unknown)}; allowed: rho, sigma2_eps, sigma2_zeta, sigma2_kappa.")
    nc = len(p0.lambdas)
    x0 = x_from_params(p0)
    free = np.ones(x0.size, dtype=bool)
    if "sigma2_eps" in fixed:
        x0[0], free[0] = np.log(fixed["sigma2_eps"]), False
    if "sigma2_zeta" in fixed:
        x0[1], free[1] = np.log(fixed["sigma2_zeta"]), False
    if "sigma2_kappa" in fixed:
        fk = fixed["sigma2_kappa"]
        items = fk.items() if isinstance(fk, dict) else enumerate(fk)
        for c, v in items:
            if not 0 <= int(c) < nc:
                raise ValueError(f"sigma2_kappa index {c} out of range for {nc} cycles.")
            x0[2 + int(c)], free[2 + int(c)] = np.log(v), False
    if "rho" in fixed:
        x0[-1], free[-1] = np.log(1.0 - float(fixed["rho"])), False
    concentrated = bool(free[:-1].all())
    if concentrated:
        free = free.copy()
        free[0] = False  # eps is profiled out
    if not free.any():
        raise ValueError("All parameters are fixed; nothing to estimate.")
    return Problem(m=p0.m, n=p0.n, lambdas=list(p0.lambdas), x0=x0, free=free, concentrated=concentrated)


def fit_ml(y: np.ndarray, p0: SCHAParams, creq: Optional[CompiledRequirements] = None,
           fixed: Optional[dict] = None, maxiter: int = 80, trust: Optional[float] = None,
           workers: Optional[int] = None, verbose: bool = False,
           callback: Optional[Callable[[int, SCHAParams, float], None]] = None) -> MLResult:
    """Fit parameters by maximum likelihood, optionally subject to gain requirements.

    Parameters
    ----------
    y : np.ndarray
        Observations on a regular grid; NaN marks gaps. The data mean is removed internally.
    p0 : scha.params.SCHAParams
        Start (lambdas in radians per sample).
    creq : scha.requirements.CompiledRequirements, optional
        Gain requirements; None for plain ML.
    fixed : dict, optional
        Parameters held fixed (see :func:`make_problem`).
    maxiter : int, optional
        Optimiser iteration limit (default 80).
    trust : float, optional
        Half-width (log units) of a box around the start.
    workers : int, optional
        Processes for finite-difference gradients; default min(10, cpu count,
        number of free variables + 1). 1 evaluates in-process.
    verbose : bool, optional
        Print one line per iteration (default False).
    callback : callable, optional
        Called as ``callback(iteration, params, loglik_objective)``.

    Returns
    -------
    scha.mlfit.MLResult
        Estimated parameters and optimiser diagnostics.

    Raises
    ------
    ValueError
        From :func:`make_problem`: unknown fixed names, out-of-range kappa indices, or no free
        parameter.
    """
    y = np.asarray(y, dtype=float)
    # level prior centred on the data mean, trend prior variance from the data scale
    y = y - np.nanmean(y)
    prob = make_problem(p0, fixed)
    prob.sigma2_diffuse = diffuse_variance(y)
    z0 = prob.z0()
    nw = workers or min(10, os.cpu_count() or 1, z0.size + 1)
    pool = None
    if nw > 1:
        pool = ProcessPoolExecutor(max_workers=nw, initializer=_init_worker, initargs=(prob, y))
        evaluate = lambda pts: list(pool.map(_eval_ll, pts))
    else:
        _init_worker(prob, y)
        evaluate = lambda pts: [_eval_ll(z) for z in pts]
    cache: dict = {}
    t_start = time.time()
    history: list = []

    def eval_z(z):
        key = z.tobytes()
        if key not in cache:
            pts = [z] + [z + FD_STEP * e for e in np.eye(z.size)]
            lls = evaluate(pts)
            cache.clear()
            cache[key] = (lls[0], (np.array(lls[1:]) - lls[0]) / FD_STEP)
        return cache[key]

    def cons(z):
        return creq.margins(prob.params(prob.x_ratio(z)))

    def cons_jac(z):
        g0 = cons(z)
        return np.array([(cons(z + 1e-6 * e) - g0) / 1e-6 for e in np.eye(z.size)]).T

    def actual_params(z):
        return prob.params(prob.x_full(z, prob.objective(z, y)[1]))

    try:
        bounds = prob.bounds()
        if trust is not None:
            bounds = [(max(lo, v - trust), min(hi, v + trust)) for (lo, hi), v in zip(bounds, z0)]
        h = 0.05
        lls = evaluate([z0 + s * h * e for e in np.eye(z0.size) for s in (1.0, -1.0)])
        ll0 = eval_z(z0)[0]
        curv = np.array([-(lls[2 * i] + lls[2 * i + 1] - 2.0 * ll0) / h**2 / _LL_SCALE for i in range(z0.size)])
        sc = np.sqrt(np.clip(curv, 1e-2, None))
        w0 = z0 * sc
        wb = [(lo * s_, hi * s_) for (lo, hi), s_ in zip(bounds, sc)]
        fun_w = lambda w: -eval_z(w / sc)[0] / _LL_SCALE
        jac_w = lambda w: -eval_z(w / sc)[1] / _LL_SCALE / sc
        it = [0]

        def cb_w(w):
            it[0] += 1
            z = w / sc
            llo = eval_z(z)[0]
            history.append(llo)
            if verbose or callback is not None:
                p = actual_params(z)
                if verbose:
                    line = (f"it {it[0]:3d} ll_obj={llo:.2f} rho={p.rho:.5f} eps={p.sigma2_eps:.3e} "
                            f"zeta={p.sigma2_zeta:.3e} kappa=[{', '.join(f'{k:.2e}' for k in p.sigma2_kappa)}] "
                            f"{time.time() - t_start:.0f} s")
                    if creq is not None:
                        line += "\n        " + creq.report_line(p)
                    print(line, flush=True)
                if callback is not None:
                    callback(it[0], p, llo)

        if creq is not None:
            cons_w = lambda w: cons(w / sc)
            cons_jac_w = lambda w: cons_jac(w / sc) / sc[None, :]
            if cons_w(w0).min() < 0:
                def viol(w):
                    g = np.minimum(cons_w(w), 0.0)
                    return float(g @ g) + 1e-6 * float((w - w0) @ (w - w0))

                def viol_jac(w):
                    g = np.minimum(cons_w(w), 0.0)
                    return 2.0 * cons_jac_w(w).T @ g + 2e-6 * (w - w0)

                w0 = minimize(viol, w0, jac=viol_jac, method="L-BFGS-B", bounds=wb,
                              options=dict(maxiter=500)).x
                if verbose:
                    print(f"feasibility phase: min margin {cons_w(w0).min():.2e}", flush=True)
            res = minimize(fun_w, w0, jac=jac_w, method="SLSQP", bounds=wb,
                           constraints=[dict(type="ineq", fun=cons_w, jac=cons_jac_w)],
                           callback=cb_w, options=dict(maxiter=maxiter, ftol=1e-7))
        else:
            res = minimize(fun_w, w0, jac=jac_w, method="L-BFGS-B", bounds=wb, callback=cb_w,
                           options=dict(maxiter=maxiter))
        z = res.x / sc
        ll_obj, s2 = prob.objective(z, y)
        p = prob.params(prob.x_full(z, s2))
        ll = scha_loglik_diffuse(p, y, prob.sigma2_diffuse, scale_rho=p.rho)
        out = MLResult(params=p, loglik=float(ll), loglik_objective=float(ll_obj), concentrated=prob.concentrated,
                       constrained=creq is not None, success=bool(res.success), message=str(res.message),
                       nit=int(getattr(res, "nit", it[0])), runtime_s=time.time() - t_start, history=history)
        if creq is not None:
            g = creq.margins(p)
            out.min_margin = float(g.min())
            out.n_active = int(np.sum(g < 1e-3))
        return out
    finally:
        if pool is not None:
            pool.shutdown()
