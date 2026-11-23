"""Basic functional test for the tropical example run script."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from examples.tropical.run_tropical_example import run

HERE = Path(__file__).resolve().parent


def test_run_creates_csv(tmp_path):
    out = tmp_path / "out.csv"
    # call run with sample data and params in the same dir
    run(csv_path=HERE / "tropical_sample.csv", params_path=HERE / "tropical_params.yml", out_path=out)
    assert out.exists()
    data = out.read_text()
    assert "fitted" in data


def test_missing_sigma2_kappa_raises(tmp_path):
    # create a temp params file without sigma2_kappa
    p = PATH = HERE / "tropical_params.yml"
    s = PATH.read_text()
    s2 = "\n".join([line for line in s.splitlines() if not line.strip().startswith('sigma2_kappa')])
    f = tmp_path / "params_missing.yml"
    f.write_text(s2)
    with pytest.raises(ValueError):
        run(csv_path=HERE / "tropical_sample.csv", params_path=f, out_path=tmp_path / "out2.csv")


def test_output_has_regular_datetime_index(tmp_path):
    out = tmp_path / "out.csv"
    run(csv_path=HERE / "tropical_sample.csv", params_path=HERE / "tropical_params.yml", out_path=out)
    df = pd.read_csv(out, parse_dates=["time"])
    assert pd.infer_freq(df["time"]) == "15min"
    assert np.isfinite(df["fitted"]).all()


def test_nonuniform_time_raises(tmp_path):
    csv_path = tmp_path / "irregular.csv"
    csv_path.write_text("datetime,value\n2000-01-01 00:00,1.0\n2000-01-01 00:15,0.5\n"
                        "2000-01-01 00:45,0.0\n2000-01-01 01:00,0.2\n")
    with pytest.raises(ValueError):
        run(csv_path=csv_path, params_path=HERE / "tropical_params.yml", out_path=tmp_path / "out3.csv")


if __name__ == "__main__":
    test_run_creates_csv(Path("."))
    print("tropical_test passed")
