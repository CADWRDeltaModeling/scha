"""Reference Kalman filter, smoother and state sampler for SCHA models.

Forward filtering (:func:`kalman_filter`), Rauch-Tung-Striebel backward
smoothing (:func:`kalman_smoother`) and Carter-Kohn backward sampling
(:func:`carter_kohn_draw`) for a time-invariant linear-Gaussian state-space
model with a scalar observation. Missing observations are skipped.

These routines are the straightforward (dense matrix, Python loop) reference
implementations; :mod:`scha.fastll` provides the fast numba likelihood and
smoother that are validated against them.
"""
import numpy as np

def _symmetrize(M: np.ndarray) -> np.ndarray:
    """Return the symmetric part, ``(M + M.T) / 2``, of a square matrix."""
    return 0.5 * (M + M.T)

def _as_11(H) -> np.ndarray:
    """Coerce scalar observation variance/covariance to shape (1,1).

    Parameters
    ----------
    H : float or array_like
        A scalar, or an array of shape (1,) or (1, 1).

    Returns
    -------
    (1, 1) ndarray
        `H` as a float array.

    Raises
    ------
    ValueError
        If `H` is not a scalar, shape (1,) or shape (1, 1).
    """
    H = np.asarray(H, dtype=float)
    if H.ndim == 0:
        return H.reshape(1, 1)
    if H.shape == (1,):
        return H.reshape(1, 1)
    if H.shape != (1, 1):
        raise ValueError(f"H must be scalar or shape (1,1); got {H.shape}.")
    return H

def _as_1dZ(Z, d: int) -> np.ndarray:
    """Coerce the observation matrix to a float array of shape (1, d).

    Parameters
    ----------
    Z : array_like
        Observation matrix, which must already have shape (1, `d`).
    d : int
        Number of state elements.

    Returns
    -------
    (1, d) ndarray
        `Z` as a float array.

    Raises
    ------
    ValueError
        If `Z` does not have shape (1, `d`).
    """
    Z = np.asarray(Z, dtype=float)
    if Z.shape != (1, d):
        raise ValueError(f"Z must be shape (1,{d}); got {Z.shape}.")
    return Z


def kalman_filter(
    y: np.ndarray,
    T: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    Z: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray,
    P0: np.ndarray,
    missing_is_nan: bool = True,
    min_F: float | None = None,
    joseph: bool = True,
):
    """Run the Kalman filter for a time-invariant linear-Gaussian state-space model.

    The model is (see Notes) a latent state driven by shocks and observed
    through a scalar measurement with noise. The filter is run step by step
    from a prior on the pre-sample state x_0, so gaps (NaN observations) are
    handled naturally.

    Parameters
    ----------
    y : (n_time,) or (n_time, 1) array_like
        Observations in time order. Current implementation assumes scalar
        observations (n_obs = 1). Missing observations may be represented
        as NaN if `missing_is_nan=True`.

    T : (d_state, d_state) ndarray
        State transition matrix.

    R : (d_state, d_shock) ndarray
        Process-noise loading matrix.

    Q : (d_shock, d_shock) ndarray
        Covariance of the process shocks w_t.

    Z : (1, d_state) ndarray
        Observation matrix.

    H : (1, 1) ndarray
        Observation-noise covariance.

    a0 : (d_state,) or (d_state, 1) array_like
        Prior mean for x_0.

    P0 : (d_state, d_state) ndarray
        Prior covariance for x_0.

    missing_is_nan : bool, default True
        If True, skip the update step when y_t is NaN (carry prediction forward).
        If False, NaN values are not treated specially and propagate through
        the recursion.

    min_F : float or None, default None
        Optional lower bound applied to the scalar innovation variance F_t
        before inversion. The bounded value is the one stored in ``F`` and
        used in the gain.

    joseph : bool, default True
        If True, update the filtered covariance with the Joseph-stabilised
        form ``(I - K Z) P (I - K Z)' + K H K'``; otherwise use the
        conventional ``P - K Z P``. Both results are symmetrised.

    Returns
    -------
    filt : dict[str, ndarray]
        Arrays (n_time = Tt, d_state = d):

        - a : (Tt, d, 1) predicted means, a[t] = E[x_t | y_{1:t-1}]
        - P : (Tt, d, d) predicted covs,  P[t] = Var[x_t | y_{1:t-1}]
        - m : (Tt, d, 1) filtered means,  m[t] = E[x_t | y_{1:t}]
        - C : (Tt, d, d) filtered covs,   C[t] = Var[x_t | y_{1:t}]
        - v : (Tt, 1, 1) innovations,     v[t] = y_t - Z a[t]
        - F : (Tt, 1, 1) innovation vars, F[t] = Z P[t] Z' + H
        - K : (Tt, d, 1) Kalman gains,    K[t] = P[t] Z' F[t]^{-1}

        At a missing observation, ``m[t]`` and ``C[t]`` equal ``a[t]`` and
        ``P[t]``, and v, F and K are left at zero.

    See Also
    --------
    kalman_smoother : Fixed-interval smoother that consumes this output.
    carter_kohn_draw : Backward sampling of the state path from this output.

    Notes
    -----
    Model:

    State (latent) evolution::

        x_t = T x_{t-1} + R w_t,     w_t ~ N(0, Q)

    Observation::

        y_t = Z x_t + e_t,          e_t ~ N(0, H)

    The critical recursion detail (matching dlm/stats texts)::

        a_t = T m_{t-1}
        P_t = T C_{t-1} T' + R Q R'

    i.e. prediction uses the *filtered* state from the previous step,
    not the previous prediction. For t = 1 the previous filtered state is the
    prior (a0, P0) on the pre-sample state x_0.

    The returned gain is the filter gain ``K = P Z' / F`` that updates the
    predicted state to the filtered state, ``m = a + K v``. It is not the
    Durbin-Koopman prediction gain ``T P Z' / F``.

    In SCHA models the state is the scaled state when the model was built with
    ``scale_rho`` (see :func:`scha.state_space.build_scha_state_space`); this
    function is agnostic to that choice.
    """
    y = np.asarray(y).reshape(-1, 1)
    Tt = y.shape[0]
    d = int(np.asarray(a0).reshape(-1).shape[0])



    a = np.zeros((Tt, d, 1))
    P = np.zeros((Tt, d, d))
    m = np.zeros((Tt, d, 1))
    C = np.zeros((Tt, d, d))
    v = np.zeros((Tt, 1, 1))
    Fm = np.zeros((Tt, 1, 1))
    K = np.zeros((Tt, d, 1))

    # filtered state at t=0 prior (x_0 | y_{1:0})
    m_prev = np.asarray(a0, dtype=float).reshape(d, 1)
    C_prev = np.asarray(P0, dtype=float)

    for t in range(Tt):
        # ---- predict: x_t | y_{1:t-1} ----
        a[t] = T @ m_prev
        P[t] = T @ C_prev @ T.T + R @ Q @ R.T

        yt = y[t : t + 1]  # (1,1)
        if missing_is_nan and np.isnan(yt).any():
            # carry prediction forward as the filtered state when missing
            m[t] = a[t]
            C[t] = P[t]
            m_prev = m[t]
            C_prev = C[t]
            continue

        # ---- update: x_t | y_{1:t} ----
        v[t] = yt - (Z @ a[t])                    # (1,1)
        Ft = (Z @ P[t] @ Z.T) + H                 # (1,1)
        if min_F is not None:
            Ft = np.maximum(Ft, float(min_F))
        Fm[t] = Ft

        Ft_inv = 1.0 / Ft if Ft.shape == (1, 1) else np.linalg.pinv(Ft)
        Kt = (P[t] @ Z.T) @ Ft_inv                # (d,1)
        K[t] = Kt

        m[t] = a[t] + Kt @ v[t]                   # (d,1)
        if joseph:
            # Joseph-stabilized covariance update:
            # C = (I - KZ) P (I - KZ)' + K H K'
            I = np.eye(d)
            IKZ = I - (Kt @ Z)          # (d,d) since Kt(d,1), Z(1,d)
            Cj = IKZ @ P[t] @ IKZ.T + Kt @ H @ Kt.T
            C[t] = _symmetrize(Cj)
        else:
            # Conventional (can be less stable numerically)
            C[t] = _symmetrize(P[t] - Kt @ Z @ P[t])

        m_prev = m[t]
        C_prev = C[t]

    return dict(a=a, P=P, m=m, C=C, v=v, F=Fm, K=K)


def kalman_smoother(filt: dict, T: np.ndarray):
    """Smooth a filtered state path with the Rauch-Tung-Striebel (RTS) smoother.

    Parameters
    ----------
    filt : dict
        Output from :func:`kalman_filter` (must include a,P,m,C).

    T : (d_state, d_state) ndarray
        State transition matrix.

    Returns
    -------
    smooth : dict[str, ndarray]
        - ms : (Tt, d, 1) smoothed means  E[x_t | y_{1:T}]
        - Cs : (Tt, d, d) smoothed covs   Var[x_t | y_{1:T}]
        - J : (Tt, d, d) smoother gains; ``J[-1]`` is unused and zero

    See Also
    --------
    kalman_filter : Produces `filt`.

    Notes
    -----
    Backward recursion from ``ms[-1] = m[-1]``, ``Cs[-1] = C[-1]``, with
    ``J[t] = C[t] T' pinv(P[t+1])``, ``ms[t] = m[t] + J[t] (ms[t+1] - a[t+1])``
    and ``Cs[t] = C[t] + J[t] (Cs[t+1] - P[t+1]) J[t]'`` (symmetrised). The
    Moore-Penrose pseudo-inverse is used for ``P[t+1]``. Gaps need no special
    handling because ``filt`` carries the prediction through them.
    """
    a, P, m, C = filt["a"], filt["P"], filt["m"], filt["C"]
    Tt, d, _ = m.shape

    ms = np.zeros_like(m)          # (Tt, d, 1)
    Cs = np.zeros_like(C)          # (Tt, d, d)
    J  = np.zeros_like(C)          # (Tt, d, d)  smoother gains (J[-1] unused)

    ms[-1] = m[-1]
    Cs[-1] = C[-1]

    for t in range(Tt - 2, -1, -1):
        Pt1 = P[t + 1]  # predicted cov at t+1 given data up to t
        Jt = C[t] @ T.T @ np.linalg.pinv(Pt1)
        J[t] = Jt

        ms[t] = m[t] + Jt @ (ms[t + 1] - a[t + 1])
        Cs[t] = _symmetrize(C[t] + Jt @ (Cs[t + 1] - Pt1) @ Jt.T)

    return {"ms": ms, "Cs": Cs, "J": J}


def carter_kohn_draw(filt, T: np.ndarray, R: np.ndarray, Q: np.ndarray, rng=None):
    """Draw one latent state trajectory by Carter-Kohn backward sampling.

    Parameters
    ----------
    filt : dict
        Output from :func:`kalman_filter` (uses a, P, m, C).
    T : (d_state, d_state) ndarray
        State transition matrix.
    R : (d_state, d_shock) ndarray
        Process-noise loading matrix. Not used in the computation (the
        needed information is already in `filt`).
    Q : (d_shock, d_shock) ndarray
        Shock covariance. Not used in the computation.
    rng : numpy.random.Generator, optional
        Source of standard normal draws (``rng.standard_normal``). If None,
        ``numpy.random.default_rng()`` is used.

    Returns
    -------
    (Tt, d_state) ndarray
        One draw of the state path x_{1:T} given all observations.

    See Also
    --------
    kalman_filter : Produces `filt`.
    kalman_smoother : The mean of the sampled paths.

    Notes
    -----
    The last state is drawn from N(m[T], C[T]). Going backwards,
    ``x_t | x_{t+1}, y_{1:t}`` is normal with mean
    ``m[t] + J[t] (x_{t+1} - a[t+1])`` and covariance
    ``C[t] - J[t] P[t+1] J[t]'``, where ``J[t] = C[t] T' pinv(P[t+1])``. The
    draws use Cholesky factors of these covariances, so they must be
    positive definite; this is the kind of explicit factorisation that is
    sensitive to ill-conditioned state covariances when rho is near 1 (see the
    state-scaling discussion in the design notes).
    """
    if rng is None:
        rng = np.random.default_rng()

    a, P, m, C = filt["a"], filt["P"], filt["m"], filt["C"]
    Tt, d, _ = m.shape

    alpha = np.zeros((Tt, d, 1))
    alpha[-1] = m[-1] + np.linalg.cholesky(C[-1]) @ rng.standard_normal((d, 1))

    for t in range(Tt - 2, -1, -1):
        Pt1 = P[t + 1]
        Jt = C[t] @ T.T @ np.linalg.pinv(Pt1)
        mean_t = m[t] + Jt @ (alpha[t + 1] - a[t + 1])
        Cov_t = _symmetrize(C[t] - Jt @ Pt1 @ Jt.T)
        alpha[t] = mean_t + np.linalg.cholesky(Cov_t) @ rng.standard_normal((d, 1))

    return alpha.squeeze(-1)