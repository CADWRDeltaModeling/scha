import numpy as np
import pandas as pd
import pytest

from scha.bands import band_cpd
from scha.fit import SCHAFit
from scha.params import SCHAParams
from scha.phase import amplitude_phase, demodulate, hours_since
from scha.requirements import Requirements
from scha.timestep import freq_hours, lambdas_from_cpd, step_hours


def test_demodulate_inverts_rotating_pair():
    t = np.arange(0.0, 100.0, 0.25)
    w = 1.4
    A, phi = 2.0, np.deg2rad(35.0)
    c = A * np.cos(w * t - phi)
    s = -A * np.sin(w * t - phi)  # psi* for z = psi - i psi* = A exp(i(w t - phi))
    a, b = demodulate(c, s, t, w)
    assert np.allclose(a, A * np.cos(phi)) and np.allclose(b, A * np.sin(phi))
    # draws axis broadcasts
    a2, _ = demodulate(np.stack([c, 2 * c]), np.stack([s, 2 * s]), t, w)
    assert a2.shape == (2, t.size) and np.allclose(a2[1], 2 * A * np.cos(phi))


@pytest.mark.parametrize("freq", ["15min", "h"])
def test_phase_convention_end_to_end(freq):
    """Smooth A cos(w t - phi) + noise and recover A and the phase lag phi."""
    idx = pd.date_range("2001-03-01", periods=int(40 * 24 / freq_hours(freq)), freq=freq)
    step_h = step_hours(idx)
    t = hours_since(idx, "1970-01-01")
    bands = ["D1", "D2"]
    w = np.array([band_cpd(b) for b in bands]) * 2 * np.pi / 24.0
    amp, lag = [0.6, 1.0], [np.deg2rad(120.0), np.deg2rad(-40.0)]
    rng = np.random.default_rng(1)
    y = 0.3 + sum(A * np.cos(wi * t - ph) for A, wi, ph in zip(amp, w, lag)) + 0.01 * rng.standard_normal(t.size)
    y[200:260] = np.nan
    lam = list(lambdas_from_cpd([band_cpd(b) for b in bands], step_h))
    rho = 0.985 ** (4 * step_h)
    p = SCHAParams(m=2, n=3, rho=rho, lambdas=lam, sigma2_eps=1e-4, sigma2_zeta=1e-10 * step_h ** 3,
                   sigma2_kappa=[1e-9 * step_h ** 5] * 2)
    f = SCHAFit(params=p, bands=bands, fit_freq=freq, method="ml", requirements=Requirements(post_filter=None),
                y=pd.Series(y, index=idx))
    ap = amplitude_phase(f.phasors())
    mid = slice(t.size // 3, 2 * t.size // 3)
    for b, A, ph in zip(bands, amp, lag):
        assert np.median(ap[f"{b}_amp"].to_numpy()[mid]) == pytest.approx(A, rel=0.02)
        phase = np.median(ap[f"{b}_phase"].to_numpy()[mid])
        assert ((phase - np.degrees(ph) + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1.5)
    # the c phasor is the band's contribution in the decomposition
    dec = f.decompose()
    assert np.allclose(dec["D1"], f.phasors()["D1_c"])
