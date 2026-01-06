from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SCHAParams:
    m: int = 2
    n: int = 4
    rho: float = 0.9935
    lambdas: list = None  # list of central frequencies (radians per ts)
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
