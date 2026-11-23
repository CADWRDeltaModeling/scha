import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scha.io import freqs_from_config, load_scha_params_from_yaml, normalize_sigma2_kappa, read_series
from scha.tide_registry import rph
from scha.timestep import freq_hours, lambdas_from_cpd, lambdas_from_rph, step_hours

TROPICAL = Path(__file__).resolve().parent / ".." / "examples" / "tropical"


def _tropical_index():
    return read_series(TROPICAL / "tropical_sample.csv").index


@pytest.mark.parametrize("freq, hours", [("6min", 0.1), ("15min", 0.25), ("30min", 0.5), ("h", 1.0)])
def test_freq_and_step_hours(freq, hours):
    assert freq_hours(freq) == pytest.approx(hours)
    assert step_hours(pd.date_range("2020-01-01", periods=4, freq=freq)) == pytest.approx(hours)


def test_step_hours_rejects_bad_index():
    idx = pd.date_range("2020-01-01", periods=4, freq="15min")
    with pytest.raises(ValueError):
        step_hours(pd.DatetimeIndex(idx.values))  # freq dropped
    with pytest.raises(ValueError):
        step_hours(idx.tz_localize("UTC"))
    with pytest.raises(ValueError):
        step_hours(pd.Index([0.0, 0.25, 0.5]))
    with pytest.raises(ValueError):
        freq_hours("MS")


def test_lambdas_conversion_and_guard():
    # M1 at 15 min is the 0.0632335 rad/sample used throughout the Freeport work
    lam = lambdas_from_rph([rph("M1")], 0.25)
    assert lam[0] == pytest.approx(0.0632335, abs=1e-7)
    assert lambdas_from_cpd([1.0], 1.0)[0] == pytest.approx(2.0 * math.pi / 24.0)
    with pytest.raises(ValueError):
        lambdas_from_cpd([12.0], 1.0)  # at Nyquist for hourly data
    with pytest.raises(ValueError):
        lambdas_from_rph([rph("M1") * 4.0], 4.0)  # rad/hr vs rad/sample mix-up


def test_freqs_from_config_mapping_only():
    omega, labs = freqs_from_config({"A": "D1", "B": 1.0})
    assert labs == ["A", "B"]
    assert omega[0] == pytest.approx(rph("D1"))
    assert omega[1] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        freqs_from_config(["D1", "D2"])


def test_normalize_sigma2_kappa_list_and_scalar_and_dict():
    labels = ["D1", "D2", "D3"]
    with pytest.warns(UserWarning):
        assert normalize_sigma2_kappa(1e-13, 3, labels) == [1e-13, 1e-13, 1e-13]
    with pytest.warns(UserWarning):
        assert normalize_sigma2_kappa([1e-12], 3, labels) == [1e-12, 1e-12, 1e-12]
    assert normalize_sigma2_kappa([1e-12, 2e-12, 3e-12], 3, labels) == [1e-12, 2e-12, 3e-12]
    with pytest.raises(ValueError):
        normalize_sigma2_kappa([1e-12, 2e-12], 3, labels)
    with pytest.raises(ValueError):
        normalize_sigma2_kappa({"D1": 1, "D3": 2}, 3, labels)


def test_load_scha_params_from_yaml_and_missing_sigma(tmp_path):
    cfg = TROPICAL / "tropical_params.yml"
    idx = _tropical_index()
    params, meta = load_scha_params_from_yaml(cfg, index=idx)
    assert params.m == 2
    assert len(params.lambdas) == 4
    assert meta["dt_hours"] == pytest.approx(0.25)
    assert params.lambdas[1] == pytest.approx(rph("M2") * 0.25)

    s2 = "\n".join(line for line in cfg.read_text().splitlines()
                   if not line.strip().startswith("sigma2_kappa"))
    f = tmp_path / "no_s2k.yml"
    f.write_text(s2)
    with pytest.raises(ValueError):
        load_scha_params_from_yaml(str(f), index=idx)

    with pytest.raises(ValueError):
        load_scha_params_from_yaml(cfg, index=pd.DatetimeIndex(idx.values))


def test_load_scha_params_without_freqs(tmp_path):
    f = tmp_path / "trend_only.yml"
    f.write_text("m: 2\nn: 3\nrho: 0.99\nsigma2_eps: 1.0e-3\nsigma2_zeta: 1.0e-8\n")
    params, meta = load_scha_params_from_yaml(f, index=_tropical_index())
    assert params.lambdas == [] and params.sigma2_kappa == []
    assert meta["labels"] == []
