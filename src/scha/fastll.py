"""Fast Gaussian log-likelihood for SCHA models (numba-compiled Kalman filter).

The prediction step exploits the block-diagonal transition (trend block, one block per cycle)
and the sparse observation vector. Missing observations (NaN) are skipped as in kalman_filter.

The model is that of :func:`scha.state_space.build_scha_state_space`,
``x_t = T x_{t-1} + R w_t``, ``y_t = Z x_t + e_t``, with a prior
``x_0 ~ N(a0, P0)`` on the pre-sample state from
:func:`scha.state_space.default_prior`: zero mean, the stationary covariance on each
cycle block, and a large variance on each trend state. The likelihood is the
innovations (prediction-error) decomposition. Cycle frequencies are in radians per sample.
When ``scale_rho`` is given the filter runs in scaled state coordinates ``x = D x~``
(exact, with the same likelihood); functions that return states decode them to
unscaled coordinates.

Trend prior ("diffuse" variance). The trend is non-stationary, so it has no
stationary prior; each trend state gets variance ``sigma2_diffuse``, a large-variance
approximation to a diffuse prior (:func:`diffuse_variance`). The first ``m`` observed
values then mostly measure the prior, so :func:`scha_loglik_diffuse` and
:func:`scha_loglik_concentrated` leave them out. :func:`scha_loglik_concentrated`
profiles ``sigma2_eps`` out of the likelihood, with all variances read as ratios to
``sigma2_eps``. These functions expect the data to be centred on its mean.
Larger ``sigma2_diffuse / sigma2_eps`` lose digits in the early filter updates.
"""
from __future__ import annotations

import numpy as np
from numba import njit

from .params import SCHAParams
from .state_space import build_scha_state_space, default_prior


@njit(cache=True, fastmath=True)
def _left_mult(t_ptr, t_col, t_val, X, out):
    """Compute ``out = T @ X`` for a sparse T.

    Parameters
    ----------
    t_ptr, t_col, t_val : ndarray
        T in CSR form: row pointers (d + 1,), column indices and values of the
        nonzeros.
    X : (d, k) ndarray
        Right-hand matrix.
    out : (d, k) ndarray
        Output array, overwritten with the product.
    """
    d = X.shape[0]
    for i in range(d):
        for j in range(X.shape[1]):
            out[i, j] = 0.0
        for q in range(t_ptr[i], t_ptr[i + 1]):
            k = t_col[q]
            tv = t_val[q]
            for j in range(X.shape[1]):
                out[i, j] += tv * X[k, j]


@njit(cache=True)
def _filter_sums(y, t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0, n_skip, e_out):
    """Return (n_used, sum log F, sum v^2/F, n_obs) over observed steps after the first n_skip.

    If e_out has the length of y, standardized innovations v/sqrt(F) are written to it (NaN when missing).

    Parameters
    ----------
    y : (n,) ndarray
        Observations; NaN marks a missing value.
    t_ptr, t_col, t_val : ndarray
        Transition matrix T in CSR form (see :func:`_prepare`).
    RQR : (d, d) ndarray
        State shock covariance ``R Q R'``.
    z_idx, z_val : ndarray
        Indices and values of the nonzeros of the observation row Z.
    H : float
        Observation variance.
    a0 : (d,) ndarray
        Prior mean of the pre-sample state.
    P0 : (d, d) ndarray
        Prior covariance of the pre-sample state.
    n_skip : int
        Number of leading observed (non-NaN) values excluded from the sums
        (they are still filtered).
    e_out : (n,) or (0,) ndarray
        Output array for the standardized innovations, or an empty array to
        skip storing them.

    Returns
    -------
    n_used : int
        Number of observed values included in the sums.
    slogf : float
        Sum of ``log F_t`` over the included values.
    sv2f : float
        Sum of ``v_t^2 / F_t`` over the included values.
    n_obs : int
        Total number of observed values, including the skipped ones.
    """
    store = e_out.size == y.size
    d = a0.size
    m = a0.copy()
    C = P0.copy()
    a = np.empty(d)
    TC = np.empty((d, d))
    P = np.empty((d, d))
    PZ = np.empty(d)
    nz = z_idx.size
    n_obs = 0
    n_used = 0
    slogf = 0.0
    sv2f = 0.0
    for t in range(y.size):
        for i in range(d):
            s = 0.0
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s += t_val[q] * m[t_col[q]]
            a[i] = s
        # P = T C T': TC = T C, then P[i,j] = sum_k T[j,k] TC[i,k]
        _left_mult(t_ptr, t_col, t_val, C, TC)
        for i in range(d):
            for j in range(i, d):
                s = RQR[i, j]
                for q in range(t_ptr[j], t_ptr[j + 1]):
                    s += t_val[q] * TC[i, t_col[q]]
                P[i, j] = s
                P[j, i] = s
        if np.isnan(y[t]):
            if store:
                e_out[t] = np.nan
            for i in range(d):
                m[i] = a[i]
                for j in range(d):
                    C[i, j] = P[i, j]
            continue
        for i in range(d):
            s = 0.0
            for q in range(nz):
                s += P[i, z_idx[q]] * z_val[q]
            PZ[i] = s
        f = H
        za = 0.0
        for q in range(nz):
            f += z_val[q] * PZ[z_idx[q]]
            za += z_val[q] * a[z_idx[q]]
        v = y[t] - za
        if store:
            e_out[t] = v / np.sqrt(f)
        if n_obs >= n_skip:
            slogf += np.log(f)
            sv2f += v * v / f
            n_used += 1
        n_obs += 1
        for i in range(d):
            k_i = PZ[i] / f
            m[i] = a[i] + k_i * v
            for j in range(i, d):
                c = P[i, j] - k_i * PZ[j]
                C[i, j] = c
                C[j, i] = c
    return n_used, slogf, sv2f, n_obs


@njit(cache=True)
def _smooth_means(y, t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0):
    """Smoothed state means (fixed-interval, de Jong backward recursion); stores only predicted covariances.

    Parameters
    ----------
    y, t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0
        As for :func:`_filter_sums`.

    Returns
    -------
    (n, d) ndarray
        Smoothed means E[x_t | y_1:n] in the coordinates of the model passed
        in (scaled coordinates if the model was scaled).
    """
    n = y.size
    d = a0.size
    A = np.empty((n, d))
    PS = np.empty((n, d, d))
    PZS = np.zeros((n, d))
    V = np.zeros(n)
    FV = np.ones(n)
    obs = np.zeros(n, dtype=np.bool_)
    m = a0.copy()
    C = P0.copy()
    a = np.empty(d)
    TC = np.empty((d, d))
    P = np.empty((d, d))
    PZ = np.empty(d)
    nz = z_idx.size
    for t in range(n):
        for i in range(d):
            s = 0.0
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s += t_val[q] * m[t_col[q]]
            a[i] = s
        _left_mult(t_ptr, t_col, t_val, C, TC)
        for i in range(d):
            for j in range(i, d):
                s = RQR[i, j]
                for q in range(t_ptr[j], t_ptr[j + 1]):
                    s += t_val[q] * TC[i, t_col[q]]
                P[i, j] = s
                P[j, i] = s
        A[t] = a
        PS[t] = P
        if np.isnan(y[t]):
            m[:] = a
            C[:, :] = P
            continue
        for i in range(d):
            s = 0.0
            for q in range(nz):
                s += P[i, z_idx[q]] * z_val[q]
            PZ[i] = s
        f = H
        za = 0.0
        for q in range(nz):
            f += z_val[q] * PZ[z_idx[q]]
            za += z_val[q] * a[z_idx[q]]
        v = y[t] - za
        obs[t] = True
        V[t] = v
        FV[t] = f
        PZS[t] = PZ
        for i in range(d):
            k_i = PZ[i] / f
            m[i] = a[i] + k_i * v
            for j in range(i, d):
                c = P[i, j] - k_i * PZ[j]
                C[i, j] = c
                C[j, i] = c
    out = np.empty((n, d))
    r = np.zeros(d)
    s = np.empty(d)
    for t in range(n - 1, -1, -1):
        # r_{t-1} = Z' v/F + L' r_t with L = T - K Z and K = T P Z'/F
        s[:] = 0.0
        for i in range(d):
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s[t_col[q]] += t_val[q] * r[i]
        r[:] = s
        if obs[t]:
            w = V[t]
            for i in range(d):
                w -= PZS[t, i] * s[i]
            w /= FV[t]
            for q in range(nz):
                r[z_idx[q]] += z_val[q] * w
        for i in range(d):
            acc = A[t, i]
            for j in range(d):
                acc += PS[t, i, j] * r[j]
            out[t, i] = acc
    return out


@njit(cache=True)
def _signal_moments(y, t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0):
    """Filter-predicted and smoothed signal mean and variance at every step.

    Returns (pred_mean, pred_var, smooth_mean, smooth_var). The predicted moments
    are E[Z x_t | y_1:t-1] and Var[Z x_t | y_1:t-1]: inside a run of missing values
    they are the one-sided forecast. The smoothed moments condition on all of y
    (Durbin-Koopman r and N recursions). Variances exclude the observation noise H.

    Parameters
    ----------
    y, t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0
        As for :func:`_filter_sums`.

    Returns
    -------
    pred_mean, pred_var, smooth_mean, smooth_var : (n,) ndarray
        Moments of the signal Z x_t, as described above.
    """
    n = y.size
    d = a0.size
    A = np.empty((n, d))
    PS = np.empty((n, d, d))
    G = np.empty((n, d))
    V = np.zeros(n)
    FV = np.ones(n)
    obs = np.zeros(n, dtype=np.bool_)
    pm = np.empty(n)
    pv = np.empty(n)
    m = a0.copy()
    C = P0.copy()
    a = np.empty(d)
    TC = np.empty((d, d))
    P = np.empty((d, d))
    PZ = np.empty(d)
    nz = z_idx.size
    for t in range(n):
        for i in range(d):
            s = 0.0
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s += t_val[q] * m[t_col[q]]
            a[i] = s
        _left_mult(t_ptr, t_col, t_val, C, TC)
        for i in range(d):
            for j in range(i, d):
                s = RQR[i, j]
                for q in range(t_ptr[j], t_ptr[j + 1]):
                    s += t_val[q] * TC[i, t_col[q]]
                P[i, j] = s
                P[j, i] = s
        for i in range(d):
            s = 0.0
            for q in range(nz):
                s += P[i, z_idx[q]] * z_val[q]
            PZ[i] = s
        zpz = 0.0
        za = 0.0
        for q in range(nz):
            zpz += z_val[q] * PZ[z_idx[q]]
            za += z_val[q] * a[z_idx[q]]
        A[t] = a
        PS[t] = P
        G[t] = PZ
        pm[t] = za
        pv[t] = zpz
        if np.isnan(y[t]):
            m[:] = a
            C[:, :] = P
            continue
        f = zpz + H
        v = y[t] - za
        obs[t] = True
        V[t] = v
        FV[t] = f
        for i in range(d):
            k_i = PZ[i] / f
            m[i] = a[i] + k_i * v
            for j in range(i, d):
                c = P[i, j] - k_i * PZ[j]
                C[i, j] = c
                C[j, i] = c
    sm = np.empty(n)
    sv = np.empty(n)
    r = np.zeros(d)
    s = np.empty(d)
    N = np.zeros((d, d))
    TN = np.empty((d, d))
    M = np.empty((d, d))
    u = np.empty(d)
    for t in range(n - 1, -1, -1):
        # r_{t-1} = T' r_t + Z' (v - g' T' r_t) / F,   g = P_t Z'
        s[:] = 0.0
        for i in range(d):
            for q in range(t_ptr[i], t_ptr[i + 1]):
                s[t_col[q]] += t_val[q] * r[i]
        r[:] = s
        # M = T' N_t T: W = N T, then M = T' W (T in CSR: row k holds T[k, c])
        for i in range(d):
            for j in range(d):
                TN[i, j] = 0.0
                M[i, j] = 0.0
        for k in range(d):
            for q in range(t_ptr[k], t_ptr[k + 1]):
                c = t_col[q]
                tv = t_val[q]
                for i in range(d):
                    TN[i, c] += N[i, k] * tv
        for k in range(d):
            for q in range(t_ptr[k], t_ptr[k + 1]):
                c = t_col[q]
                tv = t_val[q]
                for j in range(d):
                    M[c, j] += tv * TN[k, j]
        for i in range(d):
            for j in range(i + 1, d):
                avg = 0.5 * (M[i, j] + M[j, i])
                M[i, j] = avg
                M[j, i] = avg
        g = G[t]
        if obs[t]:
            f = FV[t]
            w = V[t]
            for i in range(d):
                w -= g[i] * s[i]
            w /= f
            for q in range(nz):
                r[z_idx[q]] += z_val[q] * w
            # N_{t-1} = M - (Z'u' + u Z)/F + Z'Z (g'Mg/F^2 + 1/F),  u = M g
            for i in range(d):
                acc = 0.0
                for j in range(d):
                    acc += M[i, j] * g[j]
                u[i] = acc
            gmg = 0.0
            for i in range(d):
                gmg += g[i] * u[i]
            for i in range(d):
                for j in range(d):
                    N[i, j] = M[i, j]
            for q in range(nz):
                zi = z_idx[q]
                zv = z_val[q]
                for j in range(d):
                    N[zi, j] -= zv * u[j] / f
                    N[j, zi] -= zv * u[j] / f
            cz = gmg / (f * f) + 1.0 / f
            for q1 in range(nz):
                for q2 in range(nz):
                    N[z_idx[q1], z_idx[q2]] += z_val[q1] * z_val[q2] * cz
        else:
            for i in range(d):
                for j in range(d):
                    N[i, j] = M[i, j]
        # smoothed signal: Z a + g' r ;  variance: Z P Z' - g' N g
        acc = pm[t]
        for i in range(d):
            acc += g[i] * r[i]
        sm[t] = acc
        q2 = 0.0
        for i in range(d):
            s2 = 0.0
            for j in range(d):
                s2 += N[i, j] * g[j]
            q2 += g[i] * s2
        sv[t] = pv[t] - q2
    return pm, pv, sm, sv


def _prepare(p: SCHAParams, sigma2_diffuse: float, scale_rho: float | None):
    """Assemble the sparse arrays the numba kernels take after ``y``.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters.
    sigma2_diffuse : float
        Prior variance of each trend state (see :func:`scha.state_space.default_prior`).
    scale_rho : float or None
        Reference rho of the state scaling; None for unscaled coordinates.

    Returns
    -------
    tuple
        ``(t_ptr, t_col, t_val, RQR, z_idx, z_val, H, a0, P0)``: the transition
        matrix T in CSR form (int64 pointers and columns, float values),
        ``R Q R'``, the nonzero indices and values of Z, the observation
        variance H as a float, and the prior mean and covariance of the
        pre-sample state. All are in scaled coordinates when `scale_rho` is
        given.
    """
    ss = build_scha_state_space(p, scale_rho=scale_rho)
    a0, P0 = default_prior(p, sigma2_diffuse=sigma2_diffuse, scale_rho=scale_rho)
    T = np.asarray(ss["T"], dtype=float)
    rows, cols = np.nonzero(T)
    t_ptr = np.searchsorted(rows, np.arange(T.shape[0] + 1)).astype(np.int64)
    RQR = ss["R"] @ ss["Q"] @ ss["R"].T
    z = ss["Z"][0]
    z_idx = np.flatnonzero(z).astype(np.int64)
    return (t_ptr, cols.astype(np.int64), T[rows, cols].copy(), np.ascontiguousarray(RQR), z_idx,
            z[z_idx].astype(float), float(ss["H"][0, 0]), np.asarray(a0, dtype=float).ravel(),
            np.ascontiguousarray(P0, dtype=float))


def scha_loglik(p: SCHAParams, y: np.ndarray, sigma2_diffuse: float = 1.0e4,
                scale_rho: float | None = None) -> float:
    """Gaussian log-likelihood of y under p; identical to the innovations likelihood of kalman_filter.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
    y : (n,) array_like
        Observations in time order; NaN marks a missing value.
    sigma2_diffuse : float, default 1e4
        Prior variance of each trend state, in the units of y squared.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is
        unscaled. The likelihood is the same either way, up to round-off.

    Returns
    -------
    float
        ``-0.5 * (n_obs log(2 pi) + sum log F_t + sum v_t^2 / F_t)`` over all
        ``n_obs`` observed values, with ``v_t`` the one-step innovations and
        ``F_t`` their variances.

    See Also
    --------
    scha_loglik_diffuse : Leaves out the first ``m`` observations (diffuse trend).
    scha_loglik_concentrated : Profiles out sigma2_eps.
    scha.ffbs.kalman_filter : Reference implementation.

    Notes
    -----
    All observed values enter, including the first few whose innovations
    mostly reflect the trend prior. The fitting code uses
    :func:`scha_loglik_diffuse` instead.
    """
    n, slogf, sv2f, _ = _filter_sums(np.asarray(y, dtype=float), *_prepare(p, sigma2_diffuse, scale_rho), 0,
                                     np.empty(0))
    return float(-0.5 * (n * np.log(2.0 * np.pi) + slogf + sv2f))


def scha_innovations(p: SCHAParams, y: np.ndarray, sigma2_diffuse: float = 1.0e4,
                     scale_rho: float | None = None) -> np.ndarray:
    """Standardized one-step innovations v_t / sqrt(F_t) (NaN where y is missing).

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
    y : (n,) array_like
        Observations in time order; NaN marks a missing value.
    sigma2_diffuse : float, default 1e4
        Prior variance of each trend state, in the units of y squared.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is unscaled.

    Returns
    -------
    (n,) ndarray
        ``v_t / sqrt(F_t)``, where ``v_t = y_t - Z a_t`` is the one-step
        prediction error and ``F_t`` its variance; NaN at missing values. All
        steps are included, with no leading values dropped.
    """
    y = np.asarray(y, dtype=float)
    e = np.empty(y.size)
    _filter_sums(y, *_prepare(p, sigma2_diffuse, scale_rho), 0, e)
    return e


def scha_smoothed_states(p: SCHAParams, y: np.ndarray, sigma2_diffuse: float = 1.0e4,
                         scale_rho: float | None = None) -> np.ndarray:
    """Smoothed state means E[x_t | y_1:n], shape (n, d), in unscaled state coordinates.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
    y : (n,) array_like
        Observations in time order; NaN marks a missing value.
    sigma2_diffuse : float, default 1e4
        Prior variance of each trend state, in the units of y squared.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is
        unscaled. The smoother runs in the scaled coordinates and the means
        are decoded with ``x = D x_scaled`` before being returned.

    Returns
    -------
    (n, d) ndarray
        Smoothed means, with ``d = m + Nc * 2n`` in the state ordering of
        :func:`scha.state_space.build_scha_state_space` (trend, then each
        cycle's pairs, top pair first).
    """
    D = build_scha_state_space(p, scale_rho=scale_rho)["D"]
    return _smooth_means(np.asarray(y, dtype=float), *_prepare(p, sigma2_diffuse, scale_rho)) * D[None, :]


DIFFUSE_FACTOR = 1.0e3


def diffuse_variance(y: np.ndarray) -> float:
    """Data-based trend prior variance: DIFFUSE_FACTOR * range(y)^2.

    Intended for a series centred on its mean (prior level mean = data mean), so
    the trend's initial deviation is at most of order the range. Proportional to
    the squared data scale, so fits and products do not depend on the data units.
    The factor balances the large-variance approximation (loglik error of order
    1/DIFFUSE_FACTOR) against round-off in the early filter updates, which grows
    with sigma2_diffuse / sigma2_eps (noticeable above about 1e9).

    Parameters
    ----------
    y : (n,) array_like
        Observations, preferably centred on their mean. NaN and infinite values
        are ignored.

    Returns
    -------
    float
        ``DIFFUSE_FACTOR * (max(y) - min(y))**2`` over the finite values, with
        ``DIFFUSE_FACTOR = 1e3``.

    Raises
    ------
    ValueError
        If `y` has no finite values, or they are all equal.
    """
    y = np.asarray(y, dtype=float)
    ok = y[np.isfinite(y)]
    if ok.size == 0:
        raise ValueError("No observations.")
    span = float(ok.max() - ok.min())
    if span <= 0.0:
        raise ValueError("Constant series.")
    return DIFFUSE_FACTOR * span * span


def scha_loglik_diffuse(p: SCHAParams, y: np.ndarray, sigma2_diffuse: float,
                        scale_rho: float | None = None) -> float:
    """Gaussian log-likelihood without the first p.m observations (diffuse trend).

    With ``sigma2_diffuse`` from :func:`diffuse_variance`, rescaling y by s shifts
    the result by exactly -n_used * log(s).

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
    y : (n,) array_like
        Observations in time order, preferably centred on their mean; NaN marks
        a missing value.
    sigma2_diffuse : float
        Prior variance of each trend state, in the units of y squared; see
        :func:`diffuse_variance`.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is unscaled.

    Returns
    -------
    float
        As :func:`scha_loglik`, but summed over the observed values after the
        first ``p.m`` observed (non-NaN) ones, so ``n_used = n_obs - p.m``.

    See Also
    --------
    scha_loglik : Likelihood over all observed values.
    scha_loglik_concentrated : The same construction with sigma2_eps profiled out.

    Notes
    -----
    The first ``m`` innovations have variance of order ``sigma2_diffuse`` and
    mostly measure the large-variance trend prior rather than the parameters,
    so they are dropped. Leading gaps are handled because the count is of
    observed values.
    """
    n, slogf, sv2f, _ = _filter_sums(np.asarray(y, dtype=float), *_prepare(p, sigma2_diffuse, scale_rho), p.m,
                                     np.empty(0))
    return float(-0.5 * (n * np.log(2.0 * np.pi) + slogf + sv2f))


def scha_signal_moments(p: SCHAParams, y: np.ndarray, sigma2_diffuse: float,
                        scale_rho: float | None = None) -> dict:
    """Filter-predicted and smoothed mean and variance of the signal (trend plus cycles).

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
    y : (n,) array_like
        Observations in time order, preferably centred on their mean; NaN marks
        a missing value.
    sigma2_diffuse : float
        Prior variance of each trend state, in the units of y squared; see
        :func:`diffuse_variance`.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is unscaled.
        The signal ``Z x`` is the same in either coordinate system.

    Returns
    -------
    dict
        Each entry is an (n,) ndarray. ``pred_mean``, ``pred_var``: E and Var of the
        signal given y before t; inside a run of NaN they are the one-sided
        (real-time) forecast. ``smooth_mean``,
        ``smooth_var``: given all of y. Variances exclude sigma2_eps; add it for the
        predictive variance of an observation.

    Notes
    -----
    The signal is ``Z x_t``, the trend level plus the real part of each cycle's
    top state.
    """
    pm, pv, sm, sv = _signal_moments(np.asarray(y, dtype=float), *_prepare(p, sigma2_diffuse, scale_rho))
    return dict(pred_mean=pm, pred_var=pv, smooth_mean=sm, smooth_var=sv)


def scha_loglik_concentrated(p: SCHAParams, y: np.ndarray, sigma2_diffuse_rel: float = 1.0e6,
                             scale_rho: float | None = None) -> tuple[float, float]:
    """Log-likelihood with sigma2_eps profiled out.

    The variances in p are read as ratios to sigma2_eps (p.sigma2_eps is ignored). The trend prior is
    diffuse in those relative units and the first p.m observations are dropped (diffuse likelihood),
    so the concentration is exact. Returns (concentrated log-likelihood, sigma2_eps estimate).

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters, with cycle frequencies in radians per sample.
        ``p.sigma2_zeta`` and ``p.sigma2_kappa`` are the variances divided by
        sigma2_eps; ``p.sigma2_eps`` is not used.
    y : (n,) array_like
        Observations in time order, preferably centred on their mean (the
        function does not centre them); NaN marks a missing value.
    sigma2_diffuse_rel : float, default 1e6
        Prior variance of each trend state in units of sigma2_eps, so the
        prior variance is ``sigma2_diffuse_rel * sigma2_eps``. A
        large-variance approximation to a diffuse trend.
    scale_rho : float or None, optional
        Reference rho for scaled state coordinates; None (default) is unscaled.

    Returns
    -------
    loglik : float
        ``-0.5 * (n_used * (log(2 pi) + log(s2) + 1) + sum log F_t)``, where
        ``F_t`` are the innovation variances computed with sigma2_eps = 1 and
        ``n_used`` is the number of observed values after the first ``p.m``.
    s2 : float
        The maximum-likelihood estimate of sigma2_eps given the variance
        ratios, ``sum(v_t^2 / F_t) / n_used`` over the same values.

    See Also
    --------
    scha_loglik_diffuse : Ordinary likelihood with the same dropped values.

    Notes
    -----
    If every variance scales with sigma2_eps (including the stationary prior
    covariance of the cycles and the trend prior), the innovations ``v_t`` are
    unchanged and ``F_t`` is proportional to sigma2_eps, so the likelihood is
    maximised over sigma2_eps in closed form at ``s2``. Inserting that value
    gives the concentrated log-likelihood. As with any large-variance prior,
    about ``log10(sigma2_diffuse_rel)`` significant digits are lost in the
    early filter updates; keep it below about 1e8 (see the design notes).
    """
    pr = SCHAParams(m=p.m, n=p.n, rho=p.rho, lambdas=list(p.lambdas), sigma2_eps=1.0,
                    sigma2_zeta=p.sigma2_zeta, sigma2_kappa=list(p.sigma2_kappa))
    n, slogf, sv2f, _ = _filter_sums(np.asarray(y, dtype=float), *_prepare(pr, sigma2_diffuse_rel, scale_rho), p.m,
                                     np.empty(0))
    s2 = sv2f / n
    return float(-0.5 * (n * (np.log(2.0 * np.pi) + np.log(s2) + 1.0) + slogf)), float(s2)
