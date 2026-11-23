"""scha.fastll against the reference filter and smoother in scha.ffbs (with and without gaps)."""
import numpy as np
import pytest

from scha.bands import band_cpd
from scha.fastll import (diffuse_variance, scha_loglik, scha_loglik_diffuse, scha_signal_moments,
                         scha_smoothed_states)
from scha.ffbs import kalman_filter, kalman_smoother
from scha.params import SCHAParams
from scha.state_space import build_scha_state_space, default_prior
from scha.timestep import lambdas_from_cpd

DIFFUSE = 1.0e4


def _params(n=3):
    lam = lambdas_from_cpd([band_cpd(f"D{k}") for k in range(1, 5)], 0.25)
    return SCHAParams(m=2, n=n, rho=0.985, lambdas=list(lam), sigma2_eps=2.5e-3, sigma2_zeta=2e-7,
                      sigma2_kappa=[5e-10, 2e-10, 2e-11, 1e-11])


def _y(p, n=1500, seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = 4.0 + 0.002 * t + sum(np.cos(l * t + k) for k, l in enumerate(p.lambdas)) + 0.05 * rng.standard_normal(n)
    return y


def _reference(p, y):
    ss = build_scha_state_space(p)
    a0, P0 = default_prior(p, sigma2_diffuse=DIFFUSE)
    filt = kalman_filter(y, T=ss["T"], R=ss["R"], Q=ss["Q"], Z=ss["Z"], H=ss["H"], a0=a0, P0=P0)
    v, F = np.ravel(filt["v"]), np.ravel(filt["F"])
    obs = np.isfinite(y)
    ll = -0.5 * float(np.sum(np.log(2.0 * np.pi * F[obs]) + v[obs] ** 2 / F[obs]))
    ms = kalman_smoother(filt, T=ss["T"])["ms"].squeeze(-1)
    return ll, ms


@pytest.mark.parametrize("gaps", [False, True])
def test_fastll_matches_reference(gaps):
    p = _params()
    y = _y(p)
    if gaps:
        y[300:420] = np.nan
        y[900:910] = np.nan
    ll_ref, ms_ref = _reference(p, y)
    assert scha_loglik(p, y, sigma2_diffuse=DIFFUSE) == pytest.approx(ll_ref, abs=1e-6)
    assert scha_loglik(p, y, sigma2_diffuse=DIFFUSE, scale_rho=p.rho) == pytest.approx(ll_ref, abs=1e-6)
    ms = scha_smoothed_states(p, y, sigma2_diffuse=DIFFUSE, scale_rho=p.rho)
    obs_pos = [0] + [p.m + c * 2 * p.n for c in range(len(p.lambdas))]
    assert np.abs(ms[:, obs_pos] - ms_ref[:, obs_pos]).max() < 1e-6


def _reference_signal(p, y):
    ss = build_scha_state_space(p)
    a0, P0 = default_prior(p, sigma2_diffuse=DIFFUSE)
    filt = kalman_filter(y, T=ss["T"], R=ss["R"], Q=ss["Q"], Z=ss["Z"], H=ss["H"], a0=a0, P0=P0)
    sm = kalman_smoother(filt, T=ss["T"])
    Z = ss["Z"][0]
    pm = np.einsum("j,tj->t", Z, filt["a"][:, :, 0])
    pv = np.einsum("i,tij,j->t", Z, filt["P"], Z)
    smm = np.einsum("j,tj->t", Z, sm["ms"][:, :, 0])
    smv = np.einsum("i,tij,j->t", Z, sm["Cs"], Z)
    return pm, pv, smm, smv


@pytest.mark.parametrize("scaled", [False, True])
def test_signal_moments_match_reference(scaled):
    p = _params()
    y = _y(p, n=800)
    y[200:290] = np.nan  # a 22.5 h gap at 15 min
    y[600] = np.nan
    ref = _reference_signal(p, y)
    got = scha_signal_moments(p, y, sigma2_diffuse=DIFFUSE, scale_rho=p.rho if scaled else None)
    for key, r in zip(("pred_mean", "pred_var", "smooth_mean", "smooth_var"), ref):
        sl = slice(10, None)  # skip the diffuse start, where the RTS reference loses precision
        assert np.allclose(got[key][sl], r[sl], rtol=1e-6, atol=1e-9), key
    gap = slice(200, 290)
    # inside the gap the filter forecast spreads and the smoother is tighter than the filter
    assert np.all(np.diff(got["pred_var"][gap]) > 0)
    assert np.all(got["smooth_var"][gap] < got["pred_var"][gap])


def test_diffuse_loglik_is_scale_equivariant_and_insensitive():
    p = _params()
    y = _y(p) + 500.0
    y[300:420] = np.nan
    yc = y - np.nanmean(y)  # the fitting layer centres the level prior on the data mean
    s = 37.0
    ps = SCHAParams(m=p.m, n=p.n, rho=p.rho, lambdas=p.lambdas, sigma2_eps=p.sigma2_eps * s * s,
                    sigma2_zeta=p.sigma2_zeta * s * s, sigma2_kappa=[k * s * s for k in p.sigma2_kappa])
    ll = scha_loglik_diffuse(p, yc, diffuse_variance(yc), scale_rho=p.rho)
    lls = scha_loglik_diffuse(ps, yc * s, diffuse_variance(yc * s), scale_rho=p.rho)
    n_used = int(np.isfinite(y).sum()) - p.m
    assert lls == pytest.approx(ll - n_used * np.log(s), abs=1e-6)
    ll100 = scha_loglik_diffuse(p, yc, 100.0 * diffuse_variance(yc), scale_rho=p.rho)
    assert ll100 == pytest.approx(ll, abs=1e-3)
    a = scha_signal_moments(p, yc, diffuse_variance(yc), scale_rho=p.rho)["smooth_mean"]
    b = scha_signal_moments(p, yc, 100.0 * diffuse_variance(yc), scale_rho=p.rho)["smooth_mean"]
    assert np.abs(a - b).max() < 1e-5 * np.ptp(y[np.isfinite(y)])
