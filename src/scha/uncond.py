import numpy as np
from .state_space import _cycle_FG

def cycle_unconditional_cov(n: int, lam: float, rho: float, sigma2_kappa: float) -> np.ndarray:
    """
    Unconditional covariance of the cycle state psi_t for a single frequency (shape 2*n by 2*n).
    Computed as the solution X to: X = F X F' + G W G', with W = sigma2_kappa * I2.
    Numerical fixed-point iteration is used (spectral radius < 1 ensures convergence).
    """
    F, G = _cycle_FG(n, lam, rho)
    W = sigma2_kappa * np.eye(2)
    Q = G @ W @ G.T
    X = np.zeros_like(Q)
    for _ in range(10000):
        X_new = F @ X @ F.T + Q
        if np.max(np.abs(X_new - X)) < 1e-14:
            X = X_new
            break
        X = X_new
    return X

def cycle_top_variance(n: int, rho: float, sigma2_kappa: float) -> float:
    """
    Closed-form variance of the top (psi_n) real component, Eq. (12) from the paper.
    sigma^2_top = sigma2_kappa * sum_{i=0}^{n-1} C(n-1, i)^2 * rho^{2i} / (1 - rho^2)^{2n-1}
    """
    from math import comb
    denom = (1.0 - rho**2)**(2*n - 1)
    s = 0.0
    for i in range(0, n):
        s += (comb(n-1, i)**2) * (rho**(2*i))
    return sigma2_kappa * s / denom
