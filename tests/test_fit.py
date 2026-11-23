from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import scha
import scha.io
from scha.bands import band_cpd
from scha.phase import hours_since
from scha.timestep import freq_hours

FREEPORT = Path(__file__).resolve().parent / ".." / "examples" / "freeport" / "freeport_2014_select.csv"


def _synthetic(freq="15min", days=40, seed=0):
    idx = pd.date_range("2010-06-01", periods=int(days * 24 / freq_hours(freq)), freq=freq)
    t = hours_since(idx)
    mod = 1.0 + 0.4 * np.cos(2 * np.pi * t / (13.66 * 24))
    w = [band_cpd(b) * 2 * np.pi / 24 for b in ("D1", "D2", "D3", "D4")]
    rng = np.random.default_rng(seed)
    y = (0.5 + 0.2 * np.sin(2 * np.pi * t / (20 * 24))
         + 0.6 * mod * np.cos(w[0] * t - 1.0) + 1.0 * np.cos(w[1] * t + 0.4)
         + 0.15 * mod * np.cos(w[2] * t) + 0.1 * np.cos(w[3] * t - 2.0)
         + 0.02 * rng.standard_normal(t.size))
    s = pd.Series(y, index=idx, name="synthetic")
    s.iloc[1500:1700] = np.nan
    return s


@pytest.fixture(scope="module")
def fitted():
    return scha.fit(_synthetic(), bands="D1-D4", workers=1)


def test_fit_meets_requirements_and_decomposes(fitted):
    d = fitted.diagnostics
    assert d["requirements_met"] and d["success"]
    assert fitted.fit_freq == "15min"
    dec = fitted.decompose()
    assert list(dec.columns) == ["observed", "fitted", "subtide", "subtide_raw", "D1", "D2", "D3", "D4", "residual"]
    ok = dec["subtide"].notna() & dec["observed"].notna()
    total = dec["subtide"] + dec[["D1", "D2", "D3", "D4"]].sum(axis=1) + dec["residual"]
    assert np.allclose(total[ok], dec["observed"][ok])
    filled = fitted.fill()
    assert filled.notna().all() and filled.index.equals(fitted.y.index)
    assert np.isfinite(fitted.innovations().dropna()).all()
    g = fitted.gains([0.1, 1.0, 1.9323])
    assert np.allclose(g[["subtide", "D1", "D2", "D3", "D4", "noise"]].sum(axis=1), 1.0)
    assert "requirements" in fitted.summary()


def test_yaml_round_trip(fitted, tmp_path):
    path = tmp_path / "fit.yaml"
    fitted.to_yaml(path)
    f2 = scha.load_fit(path)
    assert f2.bands == fitted.bands and f2.fit_freq == fitted.fit_freq
    assert np.allclose(f2.params.lambdas, fitted.params.lambdas)
    assert f2.params.sigma2_kappa == pytest.approx(fitted.params.sigma2_kappa)
    dec1 = fitted.decompose()
    dec2 = scha.decompose(fitted.y, path)
    assert np.allclose(dec1["D2"], dec2["D2"])
    with pytest.raises(ValueError):
        f2.decompose(fitted.y.resample("30min").first().asfreq("30min"))


def test_fixed_parameters_and_restart(fitted):
    y = fitted.y
    f = scha.fit(y, bands="D1-D4", fixed={"rho": 0.98, "sigma2_zeta": 1e-8}, start=fitted, workers=1)
    assert f.params.rho == pytest.approx(0.98)
    assert f.params.sigma2_zeta == pytest.approx(1e-8)
    assert f.diagnostics["concentrated"] is False
    assert f.diagnostics["requirements_met"]
    with pytest.raises(ValueError):
        scha.fit(y, bands="D1-D4", fixed={"lambda": 1.0}, workers=1)


def test_unconstrained_and_em_run():
    y = _synthetic(freq="h", days=30)
    f = scha.fit(y, bands="D1-D4", method="ml", requirements={"post_filter": None}, workers=1)
    assert f.diagnostics["success"] and f.fit_freq == "h"
    e = scha.fit(y, bands="D1-D4", method="em", start=f, maxiter=5, fixed={"rho": f.params.rho})
    assert e.params.rho == pytest.approx(f.params.rho)
    assert np.isfinite(e.diagnostics["loglik"])


def test_input_contract():
    y = _synthetic()
    no_freq = pd.Series(y.to_numpy(), index=pd.DatetimeIndex(y.index.values))
    assert scha.io.validate_series(no_freq).index.freqstr == "15min"  # regular: freq inferred
    with pytest.raises(ValueError, match="not regular"):
        scha.fit(y.dropna())  # missing rows instead of NaN rows
    with pytest.raises(ValueError):
        scha.fit(y.to_numpy())
    with pytest.raises(ValueError):
        scha.fit(y, method="bayes")


@pytest.mark.slow
def test_freeport_auto_start_reaches_known_optimum():
    """Automatic start reproduces the stored (2,3) 6-cycle constrained optimum (design notes, section 7).

    The design-note fits were made in units of 100 m3/s. Their ``loglik`` (22690.99) was the
    ordinary likelihood with an absolute trend prior; the reported loglik is now the diffuse one,
    which equals their stored ``loglik_objective`` (22702.04).
    """
    from scha.io import read_series
    y = read_series(FREEPORT) / 100.0
    f = scha.fit(y, requirements={"post_filter": None})
    assert f.diagnostics["loglik"] == pytest.approx(22702.04, abs=0.5)
    assert f.params.rho == pytest.approx(0.98802, abs=2e-4)


def test_units_do_not_matter():
    y = _synthetic(freq="h", days=30) * 0.3 + 5.0
    s = 1000.0
    f1 = scha.fit(y, bands="D1-D4", method="ml", requirements={"post_filter": None}, workers=1)
    f2 = scha.fit(y * s, bands="D1-D4", method="ml", requirements={"post_filter": None}, start=None, workers=1)
    assert f2.params.rho == pytest.approx(f1.params.rho, rel=1e-4)
    assert f2.params.sigma2_eps == pytest.approx(f1.params.sigma2_eps * s * s, rel=1e-3)
    n_used = int(y.notna().sum()) - 2
    assert f2.diagnostics["loglik"] == pytest.approx(f1.diagnostics["loglik"] - n_used * np.log(s), abs=0.05)
    d1, d2 = f1.decompose(), f2.decompose()
    assert np.allclose(d2["D2"] / s, d1["D2"], atol=1e-4 * y.std())


def test_signal_moments_in_gap(fitted):
    mo = fitted.signal_moments()
    gap = fitted.y.isna().to_numpy()
    assert gap.sum() == 200
    sd_in = mo["pred_sd"].to_numpy()[gap]
    assert np.all(np.diff(sd_in) > 0)  # one-sided forecast spreads through the gap
    assert np.all(mo["smooth_sd"].to_numpy()[gap] < sd_in)
    dec = fitted.decompose()
    assert np.allclose(mo["smooth_mean"], dec["fitted"], atol=1e-6)
