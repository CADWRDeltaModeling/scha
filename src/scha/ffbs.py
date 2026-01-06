import numpy as np

def _symmetrize(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)

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
    """Kalman filter for a time-invariant linear-Gaussian state-space model.

    Model
    -----
    State (latent) evolution:
        x_t = T x_{t-1} + R w_t,     w_t ~ N(0, Q)

    Observation:
        y_t = Z x_t + e_t,          e_t ~ N(0, H)

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

    min_F : float or None, default None
        Optional lower bound applied to the scalar innovation variance F_t
        before inversion.

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

    Notes
    -----
    The critical recursion detail (matching dlm/stats texts):

        a_t = T m_{t-1}
        P_t = T C_{t-1} T' + R Q R'

    i.e. prediction uses the *filtered* state from the previous step,
    not the previous prediction.
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
    print("C_prev")
    print(C_prev[:5,:5 ])
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
    """Rauch–Tung–Striebel (RTS) fixed-interval smoother.

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
    """One draw of the full latent state trajectory via Carter–Kohn backward sampling."""
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