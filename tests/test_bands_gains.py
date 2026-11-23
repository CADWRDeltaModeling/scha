import numpy as np
import pandas as pd
import pytest

from scha.bands import BAND_LINES, band_cpd, parse_bands, parse_sacrificial
from scha.gains import gains, gains_state_space
from scha.params import SCHAParams
from scha.requirements import CompiledRequirements, Requirements, subtide_gap_grid
from scha.timestep import lambdas_from_cpd


def _params(step_h, bands=("D1", "D2", "D3", "D4"), n=3):
    lam = lambdas_from_cpd([band_cpd(b) for b in bands], step_h)
    return SCHAParams(m=2, n=n, rho=0.985 ** (4 * step_h), lambdas=list(lam), sigma2_eps=2.5e-3,
                      sigma2_zeta=2e-7 * step_h ** 3 / 0.25 ** 3,
                      sigma2_kappa=[5e-10, 2e-10, 2e-11, 1e-11][: len(bands)])


def test_parse_bands_and_sacrificial():
    assert parse_bands("D1-D6") == ["D1", "D2", "D3", "D4", "D5", "D6"]
    assert parse_bands("D4,D1,D2") == ["D1", "D2", "D4"]
    assert parse_bands(["D2", "D1"]) == ["D1", "D2"]
    with pytest.raises(ValueError):
        parse_bands("D0-D3")
    with pytest.raises(ValueError):
        parse_bands("D3-D1")
    b = parse_bands("D1-D6")
    assert parse_sacrificial(2, b) == ["D5", "D6"]
    assert parse_sacrificial(0, b) == []
    assert parse_sacrificial(None, b) == []
    assert parse_sacrificial("D5-D6", b) == ["D5", "D6"]
    with pytest.raises(ValueError):
        parse_sacrificial("D7", b)
    assert band_cpd("D2") == pytest.approx(24.0 / 12.4206012)
    assert BAND_LINES["D1"][0] == pytest.approx(0.83) and BAND_LINES["D1"][-1] == pytest.approx(1.13)


@pytest.mark.parametrize("step_h", [0.1, 0.25, 1.0])
def test_closed_form_gains_match_state_space(step_h):
    p = _params(step_h)
    f = np.linspace(0.05, 12.0 / step_h - 0.05, 400)
    g = gains(p, f, step_h)
    assert np.abs(g - gains_state_space(p, f, step_h)).max() < 1e-6
    assert np.all(g.sum(axis=0) <= 1.0 + 1e-12)


def test_gap_grid_and_margins():
    bands = parse_bands("D1-D6")
    grid = subtide_gap_grid(bands, 0.25)
    assert grid.max() == pytest.approx(47.9)
    for b in bands:  # no grid point inside a band's line set
        lines = BAND_LINES[b]
        assert not np.any((grid >= lines.min()) & (grid <= lines.max()))
    p = _params(0.25)
    bands4 = ["D1", "D2", "D3", "D4"]
    creq = CompiledRequirements.build(Requirements(sacrificial=0, post_filter=None), bands4, 0.25)
    rep = creq.report(p)
    g = creq.margins(p)
    n1 = BAND_LINES["D1"].size
    # passband margins are log(G) - log(1 - delta)
    assert np.exp(g[:n1].min()) * 0.99 == pytest.approx(rep["D1_min_gain"])
    assert rep["D1_min_gain"] == pytest.approx(gains(p, BAND_LINES["D1"], 0.25)[1].min())
    assert set(rep) >= {f"{b}_min_gain" for b in bands4}
    # sacrificing D3, D4 removes their passband rows
    creq2 = CompiledRequirements.build(Requirements(sacrificial=2, post_filter=None), bands4, 0.25)
    nlines = BAND_LINES["D3"].size + BAND_LINES["D4"].size
    assert creq2.margins(p).size == g.size - nlines


def test_post_filter_response_matches_vtools_impulse():
    pytest.importorskip("vtools")
    from vtools.functions.filter import cosine_lanczos
    from scha.gains import cosine_lanczos_response
    for freq, step_h in (("15min", 0.25), ("h", 1.0)):
        n = 2 ** 14
        x = np.zeros(n)
        x[n // 2] = 1.0
        h = np.nan_to_num(cosine_lanczos(pd.Series(x, index=pd.date_range("2000-01-01", periods=n, freq=freq)),
                                         cutoff_period="18h").to_numpy())
        f = np.fft.rfftfreq(n, d=step_h / 24.0)
        H = np.abs(np.fft.rfft(h))
        sel = f < 4.0
        assert np.abs(cosine_lanczos_response(18.0, f[sel], step_h) - H[sel]).max() < 1e-8
