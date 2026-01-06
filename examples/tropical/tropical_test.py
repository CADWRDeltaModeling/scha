"""Basic functional test for the tropical example run script."""
from pathlib import Path
import numpy as np
from pathlib import Path
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


def test_numeric_index_requires_time_unit(tmp_path):
    # build a temporary CSV with numeric index like the R example (hours)
    csv_path = tmp_path / "numeric_tide.csv"
    # create a small numeric-index CSV with decimal values so index is not parsed as datetimes
    csv_path.write_text("0.0,1.0\n1.0,0.5\n2.0,0.0\n")
    # use the normal params file but remove time_unit if present
    params = (HERE / "tropical_params.yml").read_text()
    # ensure no time_unit or dt in params
    params2 = "\n".join([line for line in params.splitlines() if not (line.strip().startswith('time_unit') or line.strip().startswith('dt'))])
    pfile = tmp_path / "params.yml"
    pfile.write_text(params2)
    with pytest.raises(ValueError):
        run(csv_path=csv_path, params_path=pfile, out_path=tmp_path / "out3.csv")


def test_numeric_index_with_time_unit_runs(tmp_path):
    csv_path = tmp_path / "numeric_tide2.csv"
    csv_path.write_text("0.0,1.0\n1.0,0.5\n2.0,0.0\n")
    # add a time_unit to params
    params = (HERE / "tropical_params.yml").read_text()
    params_lines = params.splitlines()
    # add a time_unit: hour
    params_lines.insert(0, "time_unit: hour")
    pfile = tmp_path / "params2.yml"
    pfile.write_text("\n".join(params_lines))
    out = tmp_path / "out4.csv"
    run(csv_path=csv_path, params_path=pfile, out_path=out)
    assert out.exists()


if __name__ == "__main__":
    test_run_creates_csv(Path("."))
    print("tropical_test passed")
