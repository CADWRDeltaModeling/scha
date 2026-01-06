"""Validate Python smoother against R reference outputs in `r_reference/`.

Usage: run the script in the `examples/tropical` directory. It expects the following files in
`examples/tropical/r_reference/`:
 - `params.yml` (YAML parameters matching our schema)
 - `test_tide.csv` (observations; acceptable formats: numeric index or datetime index)
 - `smoothed.csv` (R-produced smoother results with `time,y,fitted` columns)

The script runs our `run_tropical_example.run(...)` to produce a Python smoothed file and compares
`fitted` columns (mean abs / max abs difference). Exits non-zero on failure.
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RREF = HERE / "r_reference"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def compare_smoothed(r_smoothed: pd.DataFrame, py_smoothed: pd.DataFrame, tol_abs: float = 1e-6, tol_rel: float = 1e-4):
    # align on time (allow numeric or datetime index forms)
    # try to align by index or by row order
    if list(r_smoothed.columns)[:3] == ["time", "y", "fitted"]:
        r_f = np.asarray(r_smoothed['fitted'], dtype=float)
    else:
        raise ValueError("R smoothed file must have columns: time,y,fitted")
    py_f = np.asarray(py_smoothed['fitted'], dtype=float)
    if r_f.shape != py_f.shape:
        raise AssertionError(f"Shape mismatch: R fitted {r_f.shape} vs Python fitted {py_f.shape}")
    abs_diff = np.abs(r_f - py_f)
    max_abs = float(np.max(abs_diff))
    mean_abs = float(np.mean(abs_diff))
    # relative metric normalized by typical signal magnitude
    denom = max(1.0, float(np.max(np.abs(r_f))))
    rel = mean_abs / denom
    return {"max_abs": max_abs, "mean_abs": mean_abs, "rel": rel, "ok": (max_abs <= tol_abs or rel <= tol_rel)}


def main():
    params = RREF / "params.yml"
    obs = RREF / "test_tide.csv"
    r_sm = RREF / "smoothed.csv"

    missing = [p for p in [params, obs, r_sm] if not p.exists()]
    if missing:
        print("Missing R reference files:", missing)
        print("Place `params.yml`, `test_tide.csv`, and `smoothed.csv` in examples/tropical/r_reference/ and re-run.")
        sys.exit(2)

    # Run the Python smoother using our runner
    from run_tropical_example import run
    out = HERE / "python_smoothed.csv"
    run(csv_path=obs, params_path=params, out_path=out)

    r_sm_df = pd.read_csv(r_sm)
    py_sm_df = pd.read_csv(out)

    result = compare_smoothed(r_sm_df, py_sm_df)
    print("Comparison result:", result)
    if not result['ok']:
        print("Parity check FAILED: differences exceed tolerance.")
        sys.exit(1)
    print("Parity check PASSED: Python smoother matches R reference within tolerances.")
    sys.exit(0)


if __name__ == "__main__":
    main()
