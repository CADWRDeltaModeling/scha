import numpy as np
import pytest

from scha.state_space import _cycle_FG
from scha.uncond import cycle_unconditional_cov, cycle_top_variance

scipy_linalg = pytest.importorskip("scipy.linalg")


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("cpd", [1.0, 2.0, 4.0])
@pytest.mark.parametrize("rho", [0.9, 0.9925])
def test_closed_form_matches_lyapunov(n, cpd, rho):
    lam = 2 * np.pi * cpd / 24.0
    s2k = 1.3e-13
    F, G = _cycle_FG(n, lam, rho)
    X_ref = scipy_linalg.solve_discrete_lyapunov(F, s2k * (G @ G.T))
    X = cycle_unconditional_cov(n, lam, rho, s2k)
    np.testing.assert_allclose(X, X_ref, rtol=1e-8, atol=1e-10 * np.abs(X_ref).max())
    assert np.isclose(X[0, 0], cycle_top_variance(n, rho, s2k), rtol=1e-10)


def test_closed_form_is_fixed_point_near_unit_root():
    n, lam, rho = 4, 2 * np.pi / 24.0, 0.9999
    F, G = _cycle_FG(n, lam, rho)
    X = cycle_unconditional_cov(n, lam, rho, 1.0)
    resid = F @ X @ F.T + G @ G.T - X
    assert np.abs(resid).max() < 1e-9 * np.abs(X).max()
