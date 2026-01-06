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

def build_scha_state_space(p: SCHAParams) -> dict:
    """
    Full SCHA(m,n) with Nc cycles.
    State ordering: [trend (mu_m,...,mu_1), cycle1 (psi_n..psi_1), ..., cycleNc]
    y_t = Z x_t + eps_t,   eps_t ~ N(0, H)
    x_t = T x_{t-1} + R v_t, v_t ~ N(0, Q)
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
    return dict(T=T, R=R, Q=Q, Z=Z, H=H, dim=dim, dim_tr=m, dim_cy=dim_cy, n=n, m=m, Nc=Nc)


def default_prior(p: SCHAParams, sigma2_diffuse: float = 1e6):
    """Construct a default prior (a0, P0) matching what the R examples use:

    - trend states: diffuse diagonal entries (sigma2_diffuse)
    - each cycle block: unconditional stationary covariance solved from F X F' + G W G'
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
    return a0, P0
