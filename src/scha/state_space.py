"""State-space matrices for the SCHA(m, n) model.

Builds the transition, shock-loading, observation and variance matrices of a
smooth trend of order m plus Nc stochastic cycles of order n, and the default
prior (a0, P0) on the pre-sample state. Optionally the model is returned in
scaled state coordinates (see :func:`state_scales`).

State ordering: ``[trend (mu_m, ..., mu_1), cycle 1 (psi_n, psi*_n, ...,
psi_1, psi*_1), ..., cycle Nc]``. Cycle frequencies are in radians per sample.
"""
import numpy as np
from typing import Tuple
from .params import SCHAParams

def _trend_blocks(m: int, sigma2_zeta: float):
    """
    Trend (order m) in 'integrated random walk' companion form.
    State order: [mu_m, mu_{m-1}, ..., mu_1].
    mu_1,t = mu_1,t-1 + zeta_t
    mu_i,t = mu_i,t-1 + mu_{i-1,t-1} for i=2..m
    Observation uses mu_m (top state).

    Parameters
    ----------
    m : int
        Trend order, at least 1.
    sigma2_zeta : float
        Variance of the shock zeta_t entering mu_1.

    Returns
    -------
    T : (m, m) ndarray
        Transition matrix (ones on the diagonal and first superdiagonal).
    R : (m, 1) ndarray
        Shock loading; the shock enters the last state, mu_1.
    Q : (1, 1) ndarray
        Shock variance ``[[sigma2_zeta]]``.
    Z : (1, m) ndarray
        Observation row, selecting the first state, mu_m.

    Raises
    ------
    ValueError
        If ``m < 1``.
    """
    if m < 1:
        raise ValueError("Trend order m must be >= 1")
    T = np.zeros((m, m))
    # Diagonal ones
    for i in range(m):
        T[i, i] = 1.0
    # Superdiagonal ones (shift)
    for i in range(m - 1):
        T[i, i+1] = 1.0
    R = np.zeros((m, 1))
    R[-1, 0] = 1.0  # shock enters the lowest-order state mu_1
    Q = np.array([[sigma2_zeta]])
    Z = np.zeros((1, m))
    Z[0, 0] = 1.0   # observe mu_m
    return T, R, Q, Z

def _cycle_FG(n: int, lam: float, rho: float):
    """
    Build the cycle (order n) state transition F and input matrix G.
    State order per cycle: [psi_n, psi*_n, psi_{n-1}, psi*_{n-1}, ..., psi_1, psi*_1]
    Balanced formulation; only order-1 pair is directly shocked.

    Parameters
    ----------
    n : int
        Cycle order, at least 1.
    lam : float
        Cycle frequency in radians per sample.
    rho : float
        Damping factor multiplying the rotation.

    Returns
    -------
    F : (2n, 2n) ndarray
        Transition matrix: 2 x 2 rotation blocks ``rho [[cos lam, sin lam],
        [-sin lam, cos lam]]`` on the block diagonal, plus a 2 x 2 identity
        in each block above the diagonal, so the pair of level i receives the
        pair of level i-1.
    G : (2n, 2) ndarray
        Shock loading; the identity on the last (order 1) pair, zero elsewhere.

    Raises
    ------
    ValueError
        If ``n < 1``.
    """
    if n < 1:
        raise ValueError("Cycle order n must be >= 1")
    R = rho
    c, s = np.cos(lam), np.sin(lam)
    Trot = np.array([[R*c, R*s],
                     [-R*s, R*c]])
    dim = 2 * n
    F = np.zeros((dim, dim))
    # diagonal Trot blocks
    for k in range(n):
        F[2*k:2*k+2, 2*k:2*k+2] = Trot
    # lower bidiagonal identity links (integrators): state_i += state_{i+1}
    for k in range(n-1):
        F[2*k:2*k+2, 2*(k+1):2*(k+1)+2] += np.eye(2)
    # Input only at the bottom (order-1 pair)
    G = np.zeros((dim, 2))
    G[-2:, :] = np.eye(2)
    return F, G

def state_scales(p: SCHAParams, scale_rho: float | None) -> np.ndarray:
    """Return the diagonal of D for x = D x_scaled.

    D is 1 on the trend states and on each cycle's top pair (psi_n, psi*_n);
    on the lower-order pairs of every cycle it is the per-level scale from
    :func:`scha.uncond.cycle_level_scales` evaluated at `scale_rho`.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters; only ``m``, ``n`` and ``len(lambdas)`` are used.
    scale_rho : float or None
        Reference rho at which the level scales are computed. If None, no
        scaling is applied and D is all ones.

    Returns
    -------
    (m + Nc * 2n,) ndarray
        Diagonal of D, in the state ordering of
        :func:`build_scha_state_space`.

    Raises
    ------
    ValueError
        If `scale_rho` is given and is not in [0, 1) (raised by
        :func:`scha.uncond.cycle_level_scales`).
    """
    from .uncond import cycle_level_scales
    m, n = p.m, p.n
    Nc = len(p.lambdas)
    D = np.ones(m + Nc * 2 * n)
    if scale_rho is not None:
        d = cycle_level_scales(n, scale_rho)
        for c in range(Nc):
            D[m + c * 2 * n : m + (c + 1) * 2 * n] = d
    return D


def build_scha_state_space(p: SCHAParams, scale_rho: float | None = None) -> dict:
    """
    Full SCHA(m,n) with Nc cycles.
    State ordering: [trend (mu_m,...,mu_1), cycle1 (psi_n..psi_1), ..., cycleNc]
    y_t = Z x_t + eps_t,   eps_t ~ N(0, H)
    x_t = T x_{t-1} + R v_t, v_t ~ N(0, Q)

    If scale_rho is given, the model is returned in scaled coordinates x = D x_scaled
    (D from state_scales at that rho). This is exact; trend and top cycle states are unscaled.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters: trend order ``m``, cycle order ``n``, damping
        ``rho``, cycle frequencies ``lambdas`` (radians per sample, one per
        cycle), and variances ``sigma2_eps``, ``sigma2_zeta`` and
        ``sigma2_kappa`` (one per cycle).
    scale_rho : float or None, optional
        Reference rho for the diagonal state scaling. If None (default) the
        model is returned in unscaled coordinates (D = 1).

    Returns
    -------
    dict
        The state-space model, with Nc = ``len(p.lambdas)`` and
        dim = m + Nc * 2n:

        - T : (dim, dim) transition matrix, block diagonal in the trend and
          the cycles
        - R : (dim, 1 + 2 Nc) shock loading; column 0 is the trend shock and
          columns 1 + 2c, 2 + 2c are the two shocks of cycle c
        - Q : (1 + 2 Nc, 1 + 2 Nc) diagonal shock covariance, ``sigma2_zeta``
          then ``sigma2_kappa[c]`` twice per cycle
        - Z : (1, dim) observation row; 1 on mu_m and on the real part
          psi_n of each cycle's top pair
        - H : (1, 1) observation variance ``sigma2_eps``
        - D : (dim,) diagonal of the scaling matrix (all ones if
          `scale_rho` is None)
        - dim, dim_tr (= m), dim_cy (= Nc * 2n), n, m, Nc : sizes

    Raises
    ------
    ValueError
        If ``p.m < 1`` or ``p.n < 1``, or if `scale_rho` is outside [0, 1).

    See Also
    --------
    default_prior : Matching prior (a0, P0).
    state_scales : The scaling diagonal D.

    Notes
    -----
    Observation and state equations are those in the summary. Cycle
    frequencies are in radians per sample.

    With D the scaling returned by :func:`state_scales`, the returned
    matrices are ``T = D^-1 T0 D``, ``R = D^-1 R0`` and ``Z = Z0 D``, where
    the subscript 0 denotes the unscaled matrices; Q and H are unchanged.
    Because the observed states have D = 1, Z is the same as unscaled.
    Decode states and covariances of the scaled model with ``x = D x_scaled``
    and ``C = D C_scaled D``.
    """
    m, n = p.m, p.n
    Nc = len(p.lambdas)
    # Trend blocks
    T_tr, R_tr, Q_tr, Z_tr = _trend_blocks(m, p.sigma2_zeta)
    # Each cycle block
    F_list, G_list = [], []
    for lam in p.lambdas:
        F, G = _cycle_FG(n, lam, p.rho)
        F_list.append(F); G_list.append(G)
    # Assemble block-diagonal T and block-sparse R, Q
    dim_tr = m
    dim_cy = Nc * (2*n)
    dim = dim_tr + dim_cy
    T = np.zeros((dim, dim))
    R = np.zeros((dim, 1 + 2*Nc))  # 1 for trend shock, 2 per cycle
    Q = np.zeros((1 + 2*Nc, 1 + 2*Nc))
    # Place trend
    T[:m, :m] = T_tr
    R[:m, :1] = R_tr
    Q[:1, :1] = Q_tr
    Z = np.zeros((1, dim))
    Z[:, :m] = Z_tr  # picks mu_m
    # Place cycles sequentially
    for c in range(Nc):
        F, G = F_list[c], G_list[c]
        start = m + c*(2*n)
        end = start + 2*n
        T[start:end, start:end] = F
        # cycle innovations enter via 2-dim vector [kappa, kappa*]
        R[start:end, 1 + 2*c : 1 + 2*c + 2] = G
        # observation loads only on psi_n (top pair) of each cycle
        Z[0, start:start+2] += np.array([1.0, 0.0])  # real part only
        # variance for cycle c
        Q[1 + 2*c : 1 + 2*c + 2, 1 + 2*c : 1 + 2*c + 2] = np.eye(2) * p.sigma2_kappa[c]
    H = np.array([[p.sigma2_eps]])
    D = state_scales(p, scale_rho)
    T = T * D[None, :] / D[:, None]
    R = R / D[:, None]
    Z = Z * D[None, :]
    return dict(T=T, R=R, Q=Q, Z=Z, H=H, D=D, dim=dim, dim_tr=m, dim_cy=dim_cy, n=n, m=m, Nc=Nc)


def default_prior(p: SCHAParams, sigma2_diffuse: float = 1e6, scale_rho: float | None = None):
    """Construct a default prior (a0, P0) matching what the R examples use:

    - trend states: diffuse diagonal entries (sigma2_diffuse)
    - each cycle block: unconditional stationary covariance solved from F X F' + G W G'

    With scale_rho, P0 is returned in the scaled coordinates of build_scha_state_space.

    Parameters
    ----------
    p : scha.params.SCHAParams
        Model parameters (``m``, ``n``, ``rho``, ``lambdas`` in radians per
        sample, and ``sigma2_kappa``).
    sigma2_diffuse : float, default 1e6
        Prior variance placed on each trend state (the diagonal of the trend
        block of P0); the trend block has no off-diagonal terms. This is a
        large-variance approximation to a diffuse prior, in the units of the
        data squared.
    scale_rho : float or None, optional
        Reference rho of the state scaling, as in
        :func:`build_scha_state_space`. If None (default), P0 is unscaled.

    Returns
    -------
    a0 : (dim,) ndarray
        Prior mean of the pre-sample state x_0, all zeros.
    P0 : (dim, dim) ndarray
        Prior covariance, block diagonal: ``sigma2_diffuse * I`` on the
        trend, and the stationary cycle covariance (see
        :func:`scha.uncond.cycle_unconditional_cov`) on each cycle. With
        `scale_rho` it is ``D^-1 P0 D^-1``.

    Raises
    ------
    ValueError
        If ``p.rho`` (or `scale_rho`) is outside [0, 1).

    See Also
    --------
    build_scha_state_space : The matching state-space matrices.
    scha.uncond.cycle_unconditional_cov : Stationary cycle covariance.

    Notes
    -----
    The trend is non-stationary, so it has no stationary covariance; the
    diagonal ``sigma2_diffuse`` stands in for a diffuse prior. The default
    1e6 is only a placeholder; ``scha.fit`` passes a value computed from the
    centred data (see :func:`scha.fastll.diffuse_variance`).
    """
    from .uncond import cycle_unconditional_cov
    m, n = p.m, p.n
    Nc = len(p.lambdas)
    dim = m + Nc * (2 * n)
    a0 = np.zeros((dim,))
    P0 = np.zeros((dim, dim))
    # trend block: diffuse diagonal
    for i in range(m):
        P0[i, i] = sigma2_diffuse
    # cycles: unconditional covariance per-cycle
    for c in range(Nc):
        start = m + c*(2*n)
        end = start + 2*n
        X = cycle_unconditional_cov(n, p.lambdas[c], p.rho, p.sigma2_kappa[c])
        P0[start:end, start:end] = X
    D = state_scales(p, scale_rho)
    return a0, P0 / np.outer(D, D)
