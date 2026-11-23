"""Unconditional (stationary) covariance of the SCHA cycle states.

Closed-form (Trimbur) stationary covariance of a cycle of order n, the
per-level state scales derived from it, and the variance of the top
component. Used for the cycle blocks of the prior covariance P0 and for the
scaled-state coordinates of :mod:`scha.state_space`.
"""
import numpy as np
from math import comb


def _trimbur_coeff(i: int, j: int, rho: float) -> float:
    """Return the Trimbur coefficient c_ij(rho) of the stationary cycle covariance.

    Parameters
    ----------
    i, j : int
        Cycle order levels, each at least 1. The coefficient is symmetric in
        (i, j).
    rho : float
        Damping factor, with ``rho < 1``.

    Returns
    -------
    float
        ``sum_r C(i-1, r) C(j-1, r+j-i) rho^(2r) / (1 - rho^2)^(i+j-1)`` for
        i <= j, summed over r = 0..i-1. Per unit ``sigma2_kappa``, c_ii is the
        stationary variance of each element of the level-i pair, and the
        (i, j) block of the covariance is c_ij times a power of the rotation
        (see :func:`cycle_unconditional_cov`).
    """
    if i > j:
        i, j = j, i
    num = sum(comb(i - 1, r) * comb(j - 1, r + j - i) * rho ** (2 * r) for r in range(i))
    return num / (1.0 - rho**2) ** (i + j - 1)


def cycle_unconditional_cov(n: int, lam: float, rho: float, sigma2_kappa: float) -> np.ndarray:
    """
    Unconditional covariance of the cycle state psi_t for a single frequency (shape 2*n by 2*n).
    Closed form of Trimbur (2003), eq. (8), as in stochCycle's calc_cycle_covariance_matrix;
    solves X = F X F' + G W G' with W = sigma2_kappa * I2.
    State order per cycle: [psi_n, psi*_n, ..., psi_1, psi*_1].

    Parameters
    ----------
    n : int
        Cycle order, at least 1.
    lam : float
        Cycle frequency in radians per sample.
    rho : float
        Damping factor, ``0 <= rho < 1``.
    sigma2_kappa : float
        Variance of each of the two shocks driving the cycle.

    Returns
    -------
    (2n, 2n) ndarray
        Stationary covariance of the cycle state, in the order above.

    Raises
    ------
    ValueError
        If `rho` is not in [0, 1).

    See Also
    --------
    cycle_level_scales : Per-level scales derived from the diagonal.
    scha.state_space.default_prior : Uses this for the cycle blocks of P0.

    Notes
    -----
    F is the cycle transition matrix of
    :func:`scha.state_space.build_scha_state_space` and G its shock loading.
    Block (i, j) is ``c_ij(rho)`` times a matrix power of the rotation
    ``rho [[cos lam, sin lam], [-sin lam, cos lam]]`` (or its transpose
    below the diagonal), with ``c_ij`` the Trimbur coefficient; diagonal
    blocks are ``c_ii I``. The closed form works for rho close to 1, where
    fixed-point iteration of the Lyapunov equation fails.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"Stationary covariance requires 0 <= rho < 1; got rho={rho}")
    c, s = np.cos(lam), np.sin(lam)
    Trot = rho * np.array([[c, s], [-s, c]])
    X = np.zeros((2 * n, 2 * n))
    orders = range(n, 0, -1)
    for a, i in enumerate(orders):
        for b, j in enumerate(orders):
            coef = _trimbur_coeff(i, j, rho)
            if i == j:
                B = coef * np.eye(2)
            elif i < j:
                B = coef * np.linalg.matrix_power(Trot, j - i)
            else:
                B = (coef * np.linalg.matrix_power(Trot, i - j)).T
            X[2 * a:2 * a + 2, 2 * b:2 * b + 2] = B
    return sigma2_kappa * X


def cycle_level_scales(n: int, rho: float) -> np.ndarray:
    """Per-state scales d for one cycle, ordered [psi_n, psi*_n, ..., psi_1, psi*_1].

    d_i = sqrt(v_i / v_n) with v_i the stationary variance of order level i, so d_n = 1.
    Depends only on (n, rho).

    Parameters
    ----------
    n : int
        Cycle order, at least 1.
    rho : float
        Reference damping factor, ``0 <= rho < 1``.

    Returns
    -------
    (2n,) ndarray
        The scale of each state element; each level's value is repeated for
        its pair (psi_i, psi*_i), and the first pair (level n) is 1.

    Raises
    ------
    ValueError
        If `rho` is not in [0, 1).

    See Also
    --------
    scha.state_space.state_scales : Assembles these into the full diagonal D.

    Notes
    -----
    These are the diagonal scales D of the scaled-state convention
    ``x = D x_scaled`` (see :func:`scha.state_space.build_scha_state_space`).
    Since v_i / v_n = c_ii(rho) / c_nn(rho) in terms of the Trimbur
    coefficients, the lower-order levels have far larger stationary variance
    than the top pair when rho is near 1, and d falls well below 1 there.
    The result does not depend on lambda or sigma2_kappa.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"Level scales require 0 <= rho < 1; got rho={rho}")
    vn = _trimbur_coeff(n, n, rho)
    d = [np.sqrt(_trimbur_coeff(i, i, rho) / vn) for i in range(n, 0, -1)]
    return np.repeat(d, 2)

def cycle_top_variance(n: int, rho: float, sigma2_kappa: float) -> float:
    """
    Closed-form variance of the top (psi_n) real component, Eq. (12) from the paper.
    sigma^2_top = sigma2_kappa * sum_{i=0}^{n-1} C(n-1, i)^2 * rho^{2i} / (1 - rho^2)^{2n-1}

    Parameters
    ----------
    n : int
        Cycle order, at least 1.
    rho : float
        Damping factor, with ``rho < 1``.
    sigma2_kappa : float
        Variance of each of the two shocks driving the cycle.

    Returns
    -------
    float
        Stationary variance of the real component psi_n of the top pair. It
        does not depend on the cycle frequency.

    See Also
    --------
    cycle_unconditional_cov : The full stationary covariance.

    Notes
    -----
    Equals ``sigma2_kappa * c_nn(rho)``, the top diagonal entry of
    :func:`cycle_unconditional_cov`. Unlike that function it does not check
    that rho is in [0, 1).
    """
    from math import comb
    denom = (1.0 - rho**2)**(2*n - 1)
    s = 0.0
    for i in range(0, n):
        s += (comb(n-1, i)**2) * (rho**(2*i))
    return sigma2_kappa * s / denom
