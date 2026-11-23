"""Parameter container for the SCHA state-space model."""
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SCHAParams:
    """Parameters of an SCHA(m, n) model with one stochastic cycle per entry of ``lambdas``.

    The model is a trend of order ``m`` plus ``len(lambdas)`` higher-order stochastic cycles
    of order ``n`` plus observation noise (see :func:`scha.state_space.build_scha_state_space`).
    ``__post_init__`` fills in ``lambdas`` and ``sigma2_kappa`` when they are None.

    Attributes
    ----------
    m : int
        Order of the trend (an ``m``-fold integrated random walk). Default 2.
    n : int
        Order of each cycle. Default 4.
    rho : float
        Damping factor of the cycles, shared by all cycles. Default 0.99.
    lambdas : list of float
        Central cycle frequencies in radians per sample. None becomes an empty list. The
        user-facing configuration is in radians per hour via the tide registry and is converted
        with the sampling step (see :mod:`scha.io` and :mod:`scha.timestep`).
    sigma2_eps : float
        Observation noise variance. Default 1e-3.
    sigma2_zeta : float
        Variance of the trend shock. Default 1e-6.
    sigma2_kappa : list of float
        Innovation variance of each cycle, same length as ``lambdas``. None becomes
        ``1e-12`` for every cycle.

    Raises
    ------
    ValueError
        If ``sigma2_kappa`` and ``lambdas`` differ in length.
    """
    m: int = 2
    n: int = 4
    rho: float = 0.99
    # Central frequencies used by the state transition, in radians per *sample step*.
    # User-facing configuration is specified in radians per hour via the tide registry
    # (e.g., D1..D6 aliases) and then converted to this representation using the
    # inferred dt from the input time index (see scha.io).
    lambdas: list = None
    sigma2_eps: float = 1e-3
    sigma2_zeta: float = 1e-6
    sigma2_kappa: list = None  # per-cycle variance, length == len(lambdas)
    def __post_init__(self):
        if self.lambdas is None:
            self.lambdas = []
        if self.sigma2_kappa is None:
            self.sigma2_kappa = [1e-12] * len(self.lambdas)
        if len(self.sigma2_kappa) != len(self.lambdas):
            raise ValueError("sigma2_kappa and lambdas must have the same length (Nc).")
