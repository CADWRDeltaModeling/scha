import numpy as np
import pandas as pd
import pytest
import yaml
from click.testing import CliRunner

from scha.cli import main
from scha.io import read_series, validate_series

from test_fit import _synthetic


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    y = _synthetic(days=30)
    path = tmp_path_factory.mktemp("cli") / "series.csv"
    with open(path, "w", encoding="utf-8") as f:
        f.write("# format: dwr-dms-1.0\n# station_id: test\n")
        y.rename("value").to_csv(f, index_label="datetime")
    return path


def test_read_series_validates(csv_path, tmp_path):
    s = read_series(csv_path)
    assert s.index.freqstr == "15min" and s.isna().sum() == 200
    holes = tmp_path / "holes.csv"
    read_series(csv_path).dropna().to_csv(holes, index_label="datetime")
    with pytest.raises(ValueError, match="not regular"):
        read_series(holes)
    hours = tmp_path / "hours.csv"
    hours.write_text("time_hr,value\n0,1.0\n0.25,2.0\n0.5,3.0\n")
    with pytest.raises(ValueError, match="datetimes"):
        read_series(hours)
    two = tmp_path / "two.csv"
    two.write_text("datetime,a,b\n2020-01-01 00:00,1,2\n2020-01-01 00:15,1,2\n2020-01-01 00:30,1,2\n")
    with pytest.raises(ValueError, match="columns"):
        read_series(two)
    assert read_series(two, column="b").iloc[0] == 2.0
    idx = pd.date_range("2020-01-01", periods=4, freq="h")
    with pytest.raises(ValueError, match="naive"):
        validate_series(pd.Series(1.0, index=idx.tz_localize("UTC")))
    with pytest.raises(ValueError, match="Duplicate"):
        validate_series(pd.Series(1.0, index=idx[[0, 1, 1, 2]]))
    assert validate_series(pd.Series(1.0, index=pd.DatetimeIndex(idx.values))).index.freqstr == "h"


def test_cli_help_exits_cleanly():
    r = CliRunner()
    for args in (["-h"], ["fit", "-h"], ["decompose", "-h"], ["ampphase", "-h"], ["gains", "-h"],
                 ["crossval", "-h"]):
        res = r.invoke(main, args)
        assert res.exit_code == 0, res.output


def test_cli_fit_config_and_products(csv_path, tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump({"bands": "D1-D4", "workers": 1, "requirements": {"post_filter": None},
                                   "fixed": {"rho": 0.985}}))
    fit_yaml = tmp_path / "fit.yaml"
    r = CliRunner()
    res = r.invoke(main, ["fit", str(csv_path), "-o", str(fit_yaml), "--config", str(cfg), "--eps", "0.02",
                          "--fix", "sigma2_zeta=2e-8", "-q"])
    assert res.exit_code == 0, res.output
    d = yaml.safe_load(fit_yaml.read_text())
    assert list(d["model"]["bands"]) == ["D1", "D2", "D3", "D4"]
    assert d["requirements"]["eps"] == 0.02 and d["requirements"]["post_filter"] is None
    assert d["fixed"] == {"rho": 0.985, "sigma2_zeta": 2e-8}
    assert d["params"]["rho"] == pytest.approx(0.985)

    for cmd, out in (("decompose", "dec.csv"), ("phasors", "ph.csv"), ("ampphase", "ap.csv")):
        res = r.invoke(main, [cmd, str(csv_path), "--fit", str(fit_yaml), "-o", str(tmp_path / out), "-q"])
        assert res.exit_code == 0, res.output
    dec = pd.read_csv(tmp_path / "dec.csv", index_col=0, parse_dates=True)
    assert {"observed", "subtide", "D1", "D4", "residual"} <= set(dec.columns)
    ap = pd.read_csv(tmp_path / "ap.csv", index_col=0)
    assert {"D2_a", "D2_b", "D2_amp", "D2_phase"} <= set(ap.columns)
    res = r.invoke(main, ["gains", str(fit_yaml), "-o", str(tmp_path / "g.csv"), "-q"])
    assert res.exit_code == 0, res.output
    g = pd.read_csv(tmp_path / "g.csv", index_col=0)
    assert np.allclose(g.sum(axis=1), 1.0)

    bad = tmp_path / "bad.yaml"
    bad.write_text("bandz: D1-D4\n")
    res = r.invoke(main, ["fit", str(csv_path), "-o", str(tmp_path / "x.yaml"), "--config", str(bad)])
    assert res.exit_code != 0
