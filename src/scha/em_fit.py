"""
EM variance fitting for SCHA.

This module is intentionally generic (not tropical-specific). It updates:
  - sigma2_zeta (trend driving noise)
  - sigma2_kappa[k] (cycle driving noise, isotropic 2D per cycle)

Optionally it can also update sigma2_eps (observation noise), but in practice
it is often fixed initially.

Key design choice: missing y values (NaN) skip ONLY the observation likelihood
and observation sufficient statistics; process sufficient statistics still use
all times.

The E-step runs the reference filter of :mod:`scha.ffbs` and a Durbin-Koopman
disturbance smoother; the M-step sets each variance to the mean of the
smoothed second moments of its disturbance. :func:`scha_em` is the entry point,
with optional SQUAREM acceleration. rho is not estimated here. Cycle
frequencies are in radians per sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.linalg as npl

from .params import SCHAParams
from .state_space import build_scha_state_space, default_prior
from .ffbs import kalman_filter, kalman_smoother


@dataclass(frozen=True)
class FitStats:
    """Expected sufficient statistics for one EM update.

    Each variance estimate in the M-step is the ratio of a sum of smoothed
    second moments (``sse_*``) to a count (``n_*``).

    Parameters
    ----------
    sse_obs : float
        Sum over observed times of ``E[eps_t^2 | y]``, the observation
        disturbance.
    n_obs : int
        Number of observed (non-NaN) values.
    sse_zeta : float
        Sum over transitions t -> t+1 of ``E[zeta^2 | y]``, the trend shock.
    n_zeta : int
        Number of transitions used for the trend shock (``n_time - 1``).
    sse_kappa : (Nc,) ndarray
        Per cycle, the sum over transitions of ``E[kappa^2 + kappa*^2 | y]``
        for the two shocks of that cycle.
    n_kappa : (Nc,) ndarray
        Per cycle, the number of shock values summed (two per transition).
    """
    sse_obs: float
    n_obs: int
    sse_zeta: float
    n_zeta: int
    sse_kappa: np.ndarray  # (Nc,)
    n_kappa: np.ndarray    # (Nc,)


def _as_1d_y(y: np.ndarray) -> np.ndarray:
    """Return y as a 1-D float array.

    Parameters
    ----------
    y : array_like
        A 1-D series, or a 2-D array of shape (T, 1) or (1, T).

    Returns
    -------
    (T,) ndarray of float
        The flattened series (not copied if already float).

    Raises
    ------
    ValueError
        If `y` cannot be reduced to one dimension.
    """
    y = np.asarray(y)
    if y.ndim == 2 and (y.shape[0] == 1 or y.shape[1] == 1):
        y = y.reshape(-1)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (or (T,1)/(1,T)); got shape {y.shape}")
    return y.astype(float, copy=False)





def _symmetrize(A: np.ndarray) -> np.ndarray:
    """Return the symmetric part, ``(A + A.T) / 2``, of a square matrix."""
    return 0.5 * (A + A.T)


def _loglik_from_filter(v: np.ndarray, F: np.ndarray, obs: np.ndarray) -> float:
    """Gaussian innovations log-likelihood for scalar observations.

    Parameters
    ----------
    v, F : (n,) array_like
        One-step innovations and their variances from the filter.
    obs : (n,) bool ndarray
        True where y was observed; only those steps enter the sum.

    Returns
    -------
    float
        ``-0.5 * sum(log(2 pi F_t) + v_t^2 / F_t)`` over observed steps.

    Raises
    ------
    ValueError
        If the lengths differ, or F is not positive at an observed step.
    """
    v = np.asarray(v).reshape(-1)
    F = np.asarray(F).reshape(-1)
    if v.shape != F.shape or v.shape != obs.shape:
        raise ValueError("v, F, obs must be same length for log-likelihood.")
    if not np.all(F[obs] > 0.0):
        # This should not happen in a correct/stable filter; treat as hard error.
        bad = np.where(obs & ~(F > 0.0))[0][:10]
        raise ValueError(f"Non-positive innovation variance F at observed indices (first few): {bad}")
    ll = -0.5 * np.sum(np.log(2.0 * np.pi * F[obs]) + (v[obs] ** 2) / F[obs])
    return float(ll)


def _disturbance_smoother_rN(
    *,
    v: np.ndarray,
    F: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    obs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Durbin–Koopman disturbance smoother recursions for r_t and N_t (scalar y).

    The arrays have length T, the number of times; the terminal values are
    r[T-1] = 0 and N[T-1] = 0, and ``r[t]``, ``N[t]`` hold the recursion
    quantities accumulated from the observations after time t (see Returns).
    The caller :func:`_expected_fit_stats_from_filter` uses ``r[t]``, ``N[t]``
    for the shock of the transition t -> t+1.

    Parameters
    ----------
    v, F : (T,) array_like
        One-step innovations and their variances from the filter.
    K : (T, d, 1) ndarray
        Filter gain ``P Z' / F`` from :func:`scha.ffbs.kalman_filter`; it is
        multiplied by T here to give the Durbin-Koopman gain.
    T : (d, d) ndarray
        State transition matrix.
    Z : (1, d) ndarray
        Observation matrix.
    obs : (T,) bool array_like
        True where y was observed.

    Returns
    -------
    r : (T, d) ndarray
        Backward recursion vector. ``r[T-1]`` is zero, and
        ``r[t-1] = Z' v_t / F_t + L_t' r[t]`` with ``L_t = T - T K_t Z`` at an
        observed step; at a missing step ``r[t-1] = T' r[t]``.
    N : (T, d, d) ndarray
        The matching covariance recursion, ``N[T-1] = 0`` and
        ``N[t-1] = Z' Z / F_t + L_t' N[t] L_t`` (``T' N[t] T`` at a missing
        step), symmetrised.

    Raises
    ------
    ValueError
        If the input lengths differ, if F is non-finite or at most 1e-14 at an
        observed step, or if N becomes non-finite or its infinity norm exceeds 1e200.
    """
    v = np.asarray(v).reshape(-1)
    F = np.asarray(F).reshape(-1)
    obs = np.asarray(obs, dtype=bool).reshape(-1)

    Tt = int(v.shape[0])
    
    if F.shape[0] != Tt or obs.shape[0] != Tt:
        raise ValueError("v, F, obs must have the same length.")
    d = int(T.shape[0])

    Zt = np.asarray(Z, dtype=float).reshape(1, d)

    r = np.zeros((Tt, d, 1), dtype=float)
    N = np.zeros((Tt, d, d), dtype=float)
    F_min = 1e-14
    N_norm_max = 1e200

    # Backward recursion with terminal conditions r[T]=0, N[T]=0.
    for t in range(Tt - 1, -1, -1):
        if obs[t]:
            Ft = float(F[t])
            if not np.isfinite(Ft) or Ft <= F_min:
                raise ValueError(f"F[t] too small/nonfinite at t={t}: F={Ft:.3e} (H too small / degeneracy).")
            Finv = 1.0 / Ft
            vt = float(v[t])

            # kalman_filter returns the filter gain P Z'/F; Durbin-Koopman needs T P Z'/F
            Kdk = T @ np.asarray(K[t], dtype=float).reshape(d, 1)
            Lt = T - (Kdk @ Zt)
            r_tm1 = (Zt.T * (Finv * vt)) + (Lt.T @ r[t])
            N_tm1 = (Zt.T * Finv @ Zt) + (Lt.T @ N[t] @ Lt)
        else:
            # Missing observation: L_t := T, no measurement term
            r_tm1 = T.T @ r[t]
            N_tm1 = T.T @ N[t] @ T
        N_tm1 = _symmetrize(N_tm1)
        if not np.all(np.isfinite(N_tm1)):
            raise ValueError(f"N blew up to nonfinite at t={t}: F={float(F[t]) if obs[t] else np.nan:.3e}")
        # Quick “is it numerically insane?” check
        nrm = float(npl.norm(N_tm1, ord=np.inf))
        if nrm > N_norm_max:
            # diagnose whether Lt is amplifying
            spec = float(max(abs(npl.eigvals(Lt)))) if obs[t] else float(max(abs(npl.eigvals(T))))
            raise ValueError(
                f"N norm exploded at t={t}: ||N||_inf={nrm:.3e}, "
                f"F={float(F[t]) if obs[t] else np.nan:.3e}, "
                f"spectral_radius(L)≈{spec:.3e}"
            )

        if t > 0:
            r[t - 1] = r_tm1
            N[t - 1] = N_tm1 #_symmetrize(N_tm1)

    return r.squeeze(-1), N


def _expected_fit_stats_from_filter(
    *,
    y: np.ndarray,
    Tmat: np.ndarray,
    Z: np.ndarray,
    H: float,
    R: np.ndarray,
    Q: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    K: np.ndarray,
    dim_tr: int,
    n: int,
    n_cycles: int,
) -> FitStats:
    """Compute expected sufficient statistics using the disturbance smoother.

    Parameters
    ----------
    y : (T,) array_like
        Observations; NaN marks a missing value.
    Tmat : (d, d) ndarray
        State transition matrix.
    Z : (1, d) ndarray
        Observation matrix.
    H : float
        Observation variance.
    R : (d, q) ndarray
        Shock loading matrix.
    Q : (q, q) ndarray
        Shock covariance. Shocks are ordered
        ``[zeta, kappa_1, kappa*_1, ..., kappa_Nc, kappa*_Nc]``.
    v, F : (T,) ndarray
        One-step innovations and their variances from the filter.
    K : (T, d, 1) ndarray
        Filter gain ``P Z' / F`` from :func:`scha.ffbs.kalman_filter`.
    dim_tr, n : int
        Trend dimension and cycle order of the model. Accepted for the
        caller's convenience; not used in the computation.
    n_cycles : int
        Number of cycles, Nc.

    Returns
    -------
    FitStats
        Smoothed second moments of the observation disturbance (observed
        times only) and of the trend and cycle shocks (transitions t -> t+1,
        t = 0..T-2), with their counts.

    Raises
    ------
    ValueError
        If R and Q, or Z and the state dimension, are incompatible; if the
        smoother fails (see :func:`_disturbance_smoother_rN`); or if the
        smoothed shock covariance is not positive semi-definite.

    Notes
    -----
    With ``u_t = v_t / F_t - K_t' r_t`` and ``K_t = T P_t Z' / F_t``, the
    observation disturbance has smoothed mean ``H u_t`` and variance
    ``H - H (1/F_t + K_t' N_t K_t) H``. The shock vector has smoothed mean
    ``Q R' r_t`` and covariance ``Q - Q R' N_t R Q``.
    """
    y = _as_1d_y(y)
    obs = ~np.isnan(y)
    Tt = y.shape[0]

    d = Tmat.shape[0]
    q = Q.shape[0]
    if R.shape[1] != q:
        raise ValueError("R and Q incompatible: R.shape[1] must equal Q.shape[0].")
    if Z.shape != (1, d):
        raise ValueError(f"Z must be (1,d); got {Z.shape}")

    r, N = _disturbance_smoother_rN(v=v, F=F, K=K, T=Tmat, Z=Z, obs=obs)

    # ---- Observation disturbance stats (optional; only uses observed times) ----
    sse_obs = 0.0
    n_obs = 0
    for t in range(Tt):
        if not obs[t]:
            continue
        Ft = float(F[t])
        Finv = 1.0 / Ft
        vt = float(v[t])

        rt = r[t].reshape(d, 1)
        Nt = N[t]
        Kt = Tmat @ np.asarray(K[t], dtype=float).reshape(d, 1)

        # u_t = F^{-1} v_t - K_t' r_t
        u = (Finv * vt) - (Kt.T @ rt).item()
        eps_hat = float(H) * u

        # Var(eps_t|Y) = H - H (F^{-1} + K' N K) H   (scalar)
        KNK = (Kt.T @ Nt @ Kt).item()
        var_eps = float(H) - float(H) * (Finv + KNK) * float(H)

        sse_obs += (var_eps + eps_hat**2)
        n_obs += 1

    # ---- State disturbance stats for transitions t -> t+1 (t=0..Tt-2) ----
    sse_zeta = 0.0
    n_zeta = 0
    sse_kappa = np.zeros((n_cycles,), dtype=float)
    n_kappa = np.zeros((n_cycles,), dtype=int)

    QRt = Q @ R.T  # (q, d)

    for t in range(Tt - 1):
        # Disturbance for transition t -> t+1 uses r[t], N[t] (r[t] accumulates innovations after t)
        rt = r[t].reshape(d, 1)
        Nt = N[t]

        eta_hat = (QRt @ rt).reshape(q, 1)  # (q,1)
        var_eta = Q - (Q @ R.T @ Nt @ R @ Q)
        var_eta = _symmetrize(var_eta)

        # Fail-fast PSD check (this should hold in a correct implementation).
        evals = np.linalg.eigvalsh(var_eta)
        emin = float(evals[0])
        emax = float(evals[-1])
        tol = 1e-10 * max(1.0, emax)
        if emin < -tol:
            raise ValueError(
                f"Var(eta|Y) is not PSD at t={t}: min_eig={emin:.3e}, max_eig={emax:.3e}."
            )

        E_eta = var_eta + (eta_hat @ eta_hat.T)

        # Shock-vector ordering (see build_scha_state_space): [zeta, kappa_1, kappa*_1, ..., kappa_Nc, kappa*_Nc]
        sse_zeta += float(E_eta[0, 0])
        n_zeta += 1

        for c in range(n_cycles):
            j0 = 1 + 2 * c
            block = E_eta[j0:j0 + 2, j0:j0 + 2]
            sse_kappa[c] += float(np.trace(block))
            n_kappa[c] += 2

    return FitStats(
        sse_obs=float(sse_obs),
        n_obs=int(n_obs),
        sse_zeta=float(sse_zeta),
        n_zeta=int(n_zeta),
        sse_kappa=sse_kappa,
        n_kappa=n_kappa,
    )


_LN10 = float(np.log(10.0))


def _with_variances(p: SCHAParams, s2_eps: float, s2_zeta: float, s2_kappa: Sequence[float]) -> SCHAParams:
    """Return a copy of `p` with sigma2_eps, sigma2_zeta and sigma2_kappa replaced.

    m, n, rho and lambdas are copied unchanged.
    """
    return SCHAParams(
        m=p.m,
        n=p.n,
        rho=p.rho,
        lambdas=list(p.lambdas),
        sigma2_eps=float(s2_eps),
        sigma2_zeta=float(s2_zeta),
        sigma2_kappa=[float(k) for k in s2_kappa],
    )


def _log_variances(p: SCHAParams, fix_sigma2_eps: bool, fix_sigma2_zeta: bool = False) -> np.ndarray:
    """Return the log of the free variances of `p` as a vector.

    The order is sigma2_eps (unless `fix_sigma2_eps`), sigma2_zeta (unless
    `fix_sigma2_zeta`), then sigma2_kappa for each cycle.
    """
    v = list(p.sigma2_kappa)
    if not fix_sigma2_zeta:
        v = [p.sigma2_zeta, *v]
    if not fix_sigma2_eps:
        v = [p.sigma2_eps, *v]
    return np.log(v)


def _from_log_variances(
    theta: np.ndarray, p: SCHAParams, fix_sigma2_eps: bool, fix_sigma2_zeta: bool = False
) -> SCHAParams:
    """Invert :func:`_log_variances`.

    Return a copy of `p` whose free variances are ``exp(theta)``, in the order
    of :func:`_log_variances`; fixed variances keep their values from `p`.
    """
    s2 = list(np.exp(theta))
    s2_eps = p.sigma2_eps if fix_sigma2_eps else s2.pop(0)
    s2_zeta = p.sigma2_zeta if fix_sigma2_zeta else s2.pop(0)
    return _with_variances(p, s2_eps, s2_zeta, s2)


def _em_update(
    y: np.ndarray,
    p: SCHAParams,
    *,
    sigma2_diffuse: float,
    fix_sigma2_eps: bool,
    floor: float,
    fix_sigma2_zeta: bool = False,
    scale_rho: float | None = None,
) -> tuple[SCHAParams, float]:
    """One EM update. Returns (updated params, log-likelihood at the input params).

    Parameters
    ----------
    y : (T,) ndarray
        Observations; NaN marks a missing value.
    p : scha.params.SCHAParams
        Current parameters.
    sigma2_diffuse : float
        Prior variance of each trend state (see
        :func:`scha.state_space.default_prior`).
    fix_sigma2_eps : bool
        If True, sigma2_eps is carried over unchanged.
    floor : float
        Lower bound applied to each updated variance.
    fix_sigma2_zeta : bool, default False
        If True, sigma2_zeta is carried over unchanged.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is unscaled.

    Returns
    -------
    params : scha.params.SCHAParams
        Parameters with the updated variances; each is the smoothed second
        moment sum divided by its count (see :class:`FitStats`), bounded
        below by `floor`.
    loglik : float
        Ordinary innovations log-likelihood (all observed values, trend prior
        variance `sigma2_diffuse`) at the input `p`.

    Raises
    ------
    ValueError
        If sigma2_eps is to be estimated and there are no observed values, or
        if the filter or smoother checks fail.
    """
    ss = build_scha_state_space(p, scale_rho=scale_rho)
    a0, P0 = default_prior(p, sigma2_diffuse=sigma2_diffuse, scale_rho=scale_rho)
    filt = kalman_filter(
        y=y,
        T=ss["T"],
        R=ss["R"],
        Q=ss["Q"],
        Z=ss["Z"],
        H=ss["H"],
        a0=a0.reshape(-1, 1),
        P0=P0,
        missing_is_nan=True,
        min_F=None,
        joseph=True,
    )
    v = np.asarray(filt["v"], dtype=float).reshape(-1)
    F = np.asarray(filt["F"], dtype=float).reshape(-1)
    K = np.asarray(filt["K"], dtype=float)
    ll = _loglik_from_filter(v=v, F=F, obs=~np.isnan(y))

    stats = _expected_fit_stats_from_filter(
        y=y,
        Tmat=ss["T"],
        Z=ss["Z"],
        H=float(ss["H"][0, 0]),
        R=ss["R"],
        Q=ss["Q"],
        v=v,
        F=F,
        K=K,
        dim_tr=ss["dim_tr"],
        n=ss["n"],
        n_cycles=ss["Nc"],
    )

    s2_eps_new = p.sigma2_eps
    if not fix_sigma2_eps:
        if stats.n_obs <= 0:
            raise ValueError("No observed y values; cannot estimate sigma2_eps.")
        s2_eps_new = max(stats.sse_obs / stats.n_obs, floor)
    s2_zeta_new = p.sigma2_zeta if fix_sigma2_zeta else max(stats.sse_zeta / stats.n_zeta, floor)
    s2_kappa_new = [max(stats.sse_kappa[k] / stats.n_kappa[k], floor) for k in range(ss["Nc"])]
    return _with_variances(p, s2_eps_new, s2_zeta_new, s2_kappa_new), ll


def scha_em(
    y: np.ndarray,
    params: SCHAParams,
    *,
    fix_rho: float | None = None,
    fix_sigma2_eps: bool = False,
    fix_sigma2_zeta: bool = False,
    sigma2_diffuse: float = 1.0e4,
    scale_rho: float | None = None,
    floor: float = 1e-18,
    max_iter: int = 50,
    tol_log10: float = 1e-3,
    ll_conv_tol: float = 0.01,
    patience: int = 3,
    accelerate: bool = False,
    ll_tol: float = 1.0,
    verbose: bool = True,
) -> SCHAParams:
    """Run EM iterations to fit variance parameters of SCHA.

    Parameters
    ----------
    y : (n_time,) array_like
        Observation time series (NaN allowed; missing obs are skipped). A
        (n_time, 1) or (1, n_time) array is also accepted.
    params : scha.params.SCHAParams
        Initial parameter values. Cycle frequencies (``lambdas``) are in
        radians per sample. The object is not modified.
    fix_rho : float, optional
        If provided, sets rho to this value and does not update it.
    fix_sigma2_eps : bool, default False
        If True, sigma2_eps is held fixed.
    fix_sigma2_zeta : bool, default False
        If True, sigma2_zeta is held fixed at its initial value.
    sigma2_diffuse : float, default 1e4
        Prior variance of the trend states (approximately diffuse).
    scale_rho : float, optional
        If given, run filter/smoother in diagonally scaled state coordinates computed at
        this reference rho (exact reparameterization; improves conditioning).
    floor : float, default 1e-18
        Hard lower bound for variances to prevent invalid negatives.
    max_iter : int, default 50
        Maximum number of EM updates (each SQUAREM cycle uses at least 3).
    tol_log10 : float, default 1e-3
        Tolerance on the estimated remaining distance to the fixed point, in log10-variance
        units: max|plain EM step| / (1 - r), with r the observed contraction rate of EM steps.
    ll_conv_tol : float, default 0.01
        Tolerance on the log-likelihood gain per EM evaluation.
    patience : int, default 3
        Number of consecutive iterations (SQUAREM cycles) the gain must stay below
        ll_conv_tol. Both this and the tol_log10 test must hold to stop.
    accelerate : bool, default False
        If True, use SQUAREM extrapolation (Varadhan & Roland 2008) on the
        log-variances.
    ll_tol : float, default 1.0
        SQUAREM accepts a jump unless it lowers the log-likelihood by more than
        this (R SQUAREM's objfn.inc). Needed because the M-step ignores the
        dependence of the cycle prior P0 on sigma2_kappa, so EM is only
        approximately monotone near the optimum.
    verbose : bool, default True
        If True, print one line per iteration (variances, step size, the
        log-likelihood at the start of the iteration and convergence
        diagnostics).

    Returns
    -------
    scha.params.SCHAParams
        A new parameter object with the fitted variances (sigma2_eps,
        sigma2_zeta, sigma2_kappa); m, n, lambdas and rho are carried over,
        with rho replaced by `fix_rho` if given. Returned when the stopping
        rule is met or `max_iter` EM updates have been used; there is no
        convergence flag.

    Raises
    ------
    ValueError
        If ``params.sigma2_kappa`` is None or its length differs from
        ``len(params.lambdas)``, if `y` cannot be reduced to one dimension, or
        if the filter or smoother checks fail (for example a non-positive
        innovation variance).

    See Also
    --------
    scha.fastll.scha_loglik_concentrated : Likelihood used by the maximum-likelihood fits.

    Notes
    -----
    Each EM update runs the Kalman filter and a disturbance smoother and sets
    sigma2_eps to the mean of ``E[eps_t^2 | y]`` over observed times,
    sigma2_zeta to the mean of ``E[zeta^2 | y]``, and each sigma2_kappa[k] to
    half the mean of ``E[kappa^2 + kappa*^2 | y]`` over the transitions, each
    bounded below by `floor`. Missing values contribute only to the
    process statistics. rho is never estimated; `fix_rho` only sets its
    value.

    With `accelerate`, two EM steps ``th1 = M(th0)``, ``th2 = M(th1)`` give
    ``r = th1 - th0`` and ``v = th2 - 2 th1 + th0`` on the log-variances. The
    SQUAREM jump is ``th0 + 2 a r + a^2 v`` with ``a = |r| / |v|`` clipped to
    ``[1, step_max]`` (``step_max`` starts at 1 and is multiplied by 4 each
    time ``a`` ends up equal to it), followed by one stabilising EM step. If
    the jump lowers the log-likelihood by more than `ll_tol`, ``a`` is moved
    halfway toward 1 until the jump is accepted (``a = 1`` is plain EM).
    Convergence needs both the `tol_log10` and the `ll_conv_tol` tests. The
    printed log-likelihood is the ordinary likelihood, not the diffuse or
    concentrated one.
    """
    y = _as_1d_y(y)

    # Work on a copy (do not mutate caller's object)
    p = SCHAParams(
        m=params.m,
        n=params.n,
        rho=float(params.rho),
        lambdas=list(params.lambdas),
        sigma2_eps=float(params.sigma2_eps),
        sigma2_zeta=float(params.sigma2_zeta),
        sigma2_kappa=list(params.sigma2_kappa) if params.sigma2_kappa is not None else None,
    )
    if fix_rho is not None:
        p.rho = float(fix_rho)

    if p.sigma2_kappa is None or len(p.sigma2_kappa) != len(p.lambdas):
        raise ValueError("params.sigma2_kappa must be a list with length == len(params.lambdas).")

    # Main loop
    em_kw = dict(sigma2_diffuse=sigma2_diffuse, fix_sigma2_eps=fix_sigma2_eps,
                 fix_sigma2_zeta=fix_sigma2_zeta, floor=floor, scale_rho=scale_rho)

    def to_theta(q: SCHAParams) -> np.ndarray:
        """Log of the free variances of `q`."""
        return _log_variances(q, fix_sigma2_eps, fix_sigma2_zeta)

    def report(tag: str, q: SCHAParams, ll: float, dlog: float, extra: str = "") -> None:
        """Print one progress line if `verbose`."""
        if verbose:
            kap = ", ".join(f"{k:.3e}" for k in q.sigma2_kappa)
            print(
                f"{tag}: s2_eps={q.sigma2_eps:.3e}  s2_zeta={q.sigma2_zeta:.3e}  "
                f"s2_kappa=[{kap}]  max_dlog={dlog:.3e}  ll_start={ll:.3f}{extra}",
                flush=True,
            )

    def remaining(step: np.ndarray, prev_step: np.ndarray) -> float:
        """Estimated log10 distance to the EM fixed point from two successive EM steps."""
        rate = npl.norm(step) / max(npl.norm(prev_step), 1e-300)
        if rate >= 1.0:
            return np.inf
        return float(np.max(np.abs(step))) / (1.0 - rate) / _LN10

    gains: list[float] = []

    def converged(dist: float) -> bool:
        """Whether both the distance test and the likelihood-gain test hold."""
        return (dist < tol_log10 and len(gains) >= patience
                and max(gains[-patience:]) < ll_conv_tol)

    if not accelerate:
        prev_step = None
        ll_prev = None
        for it in range(max_iter):
            p_new, ll = _em_update(y, p, **em_kw)
            step = to_theta(p_new) - to_theta(p)
            if ll_prev is not None:
                gains.append(ll - ll_prev)
            dist = np.inf if prev_step is None else remaining(step, prev_step)
            report(f"EM {it:02d}", p_new, ll, float(np.max(np.abs(step))) / _LN10,
                   f"  dist={dist:.2e}  dll/em={gains[-1] if gains else np.nan:.3g}")
            p, prev_step, ll_prev = p_new, step, ll
            if converged(dist):
                break
        return p

    # SQUAREM (Varadhan & Roland 2008, SqS3 step) on log-variances
    log_floor = np.log(floor)
    step_max = 1.0
    n_em = 0
    cycle = 0
    ll_prev, n_prev = None, 0
    while n_em < max_iter:
        th0 = to_theta(p)
        p1, ll0 = _em_update(y, p, **em_kw)
        if ll_prev is not None:
            gains.append((ll0 - ll_prev) / (n_em - n_prev))
        ll_prev, n_prev = ll0, n_em
        p2, ll1 = _em_update(y, p1, **em_kw)
        n_em += 2
        th1, th2 = to_theta(p1), to_theta(p2)
        r = th1 - th0
        vv = (th2 - th1) - r
        dist = remaining(th2 - th1, r)
        if converged(dist) or not np.any(vv):
            report(f"SQ {cycle:02d}", p2, ll1, float(np.max(np.abs(th2 - th1))) / _LN10,
                   f"  dist={dist:.2e}  dll/em={gains[-1] if gains else np.nan:.3g}  n_em={n_em}  converged")
            p = p2
            break

        alpha = float(np.clip(npl.norm(r) / npl.norm(vv), 1.0, step_max))
        while True:
            if alpha == 1.0:
                # alpha=1 reproduces plain EM from p2
                p_jump = p2
                p_new, ll_jump = _em_update(y, p_jump, **em_kw)
                n_em += 1
                break
            p_jump = _from_log_variances(
                np.maximum(th0 + 2.0 * alpha * r + alpha**2 * vv, log_floor), p, fix_sigma2_eps, fix_sigma2_zeta
            )
            try:
                p_new, ll_jump = _em_update(y, p_jump, **em_kw)
                ok = ll_jump >= ll1 - ll_tol
            except ValueError:
                # extreme extrapolations can trip the filter/smoother sanity checks
                ok = False
            n_em += 1
            if ok:
                break
            alpha = (alpha + 1.0) / 2.0
            if alpha < 1.01:
                alpha = 1.0

        if alpha == step_max:
            step_max *= 4.0
        dlog = float(np.max(np.abs(to_theta(p_new) - to_theta(p_jump)))) / _LN10
        report(f"SQ {cycle:02d}", p_new, ll_jump, dlog,
               f"  dist={dist:.2e}  dll/em={gains[-1] if gains else np.nan:.3g}  alpha={alpha:.2f}  n_em={n_em}")
        p = p_new
        cycle += 1

    return p
