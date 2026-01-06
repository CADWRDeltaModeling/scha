"""Regenerate the original test_tide.csv from stochCycle.

This reproduces the signal generation logic from `stochCycle/example/tropical/tropical_test.py`.
It writes `test_tide.csv` with a numeric-hour index in the same format as the R reference.

Mapping to the paper:
- D0..D4 are components (subtide and diurnal harmonics); A12 and A22 are quadratic-interaction coefficients that create harmonic content.
- `trop` is the tropical modulation period (hours), used to amplitude-modulate certain components.

Reference:
- Ateljevich, E., & Wang, H. (2023). "Nonstationary analysis of tides using higher order stochastic cycles." This generator implements the deterministic signal-construction used in the SCHA examples and reproduces the quadratic interaction terms described in the paper (the D1×D2 and D2×D2 contributions).

Use::
    python generate_test_tide.py --out examples/tropical/r_reference/test_tide.csv

"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def generate_tide():
    twopi = 2.0 * np.pi
    trop = 2.0 * 13.66 * 24.0  # tropical period in hours
    A12 = 0.75
    A22 = 0.5
    A = np.array([0.2, 0.8, 1.0, 0.2, 0.2])
    A1adj = np.sqrt((0.5 * A12 * A[1] * A[2]) ** 2.0 + A[1] * A[1])
    phi1adj = np.arctan(0.5 * A12 * A[2])

    # frequencies in cycles per hour (c/hr) used in the original example
    w = np.array([0.0, 1.0, 2.0, 3.0, 4.0]) * 0.966137 / 24.0

    t = np.arange(0.0, 800.001, 0.25)  # time in hours, 15-minute samples
    mod = np.cos(twopi * t / trop)

    A[3] = A12 * A[1] * A[2] * 0.5
    A[4] = -A22 * A[2] * A[2] * 0.5

    D0 = mod * A[0]
    D1_orig = mod * A[1] * np.cos(twopi * w[1] * t)
    D1 = mod * A1adj * np.cos(twopi * w[1] * t - phi1adj)
    D2 = A[2] * np.cos(twopi * w[2] * t - np.pi / 2.0)
    D3 = mod * A[3] * np.cos(twopi * w[3] * t - np.pi / 2.0)
    D4 = A[4] * np.cos(twopi * w[4] * t)

    # In the original example, consistent_quadratic==False branch is used (no extra D0 change)
    y1 = D0 + D1 + D2 + D3 + D4
    return t, y1


def main(out_path: Path):
    t, y = generate_tide()
    df = pd.DataFrame({"value": y}, index=t)
    df.index.name = "time"
    df.columns = ["value"]
    # write in the same simple two-column CSV form as the R reference (index, value)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, header=True,float_format="%.6f")
    print(f"Wrote generated tide to {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "tropical_sample.csv")
    args = p.parse_args()
    main(args.out)
