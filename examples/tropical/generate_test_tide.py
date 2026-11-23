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
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from scha.logging_config import configure_logging, resolve_loglevel

logger = logging.getLogger("scha.examples.generate_test_tide")


def generate_components():
    """Return (t, components) where components holds D0..D4 and the tropical modulation `mod`."""
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
    return t, dict(D0=D0, D1=D1, D2=D2, D3=D3, D4=D4, mod=mod)


def generate_tide():
    t, c = generate_components()
    # In the original example, consistent_quadratic==False branch is used (no extra D0 change)
    y1 = c["D0"] + c["D1"] + c["D2"] + c["D3"] + c["D4"]
    return t, y1


def write_test_tide(out_path: Path, noise_sd: float = 0.0, seed: int = 20230, components: bool = False):
    """Write the generated tide (optionally noisy, optionally with true components) to ``out_path``."""
    t, y = generate_tide()
    df = pd.DataFrame({"value": y}, index=t)
    if noise_sd > 0.0:
        df["value"] = y + np.random.default_rng(seed).normal(0.0, noise_sd, size=len(t))
    if components:
        _, c = generate_components()
        df["clean"] = y
        for k in ("D0", "D1", "D2", "D3", "D4"):
            df[k] = c[k]
    df.index.name = "time"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, header=True, float_format="%.10f" if (noise_sd > 0.0 or components) else "%.6f")
    logger.info("Wrote generated tide to %s", out_path)


@click.command()
@click.option("--out", type=click.Path(dir_okay=False, path_type=Path),
              default=Path(__file__).resolve().parent / "tropical_sample.csv", show_default=True,
              help="Output CSV.")
@click.option("--noise-sd", type=float, default=0.0, show_default=True,
              help="SD of additive Gaussian noise (paper intent: 0.01).")
@click.option("--seed", type=int, default=20230, show_default=True, help="Noise seed.")
@click.option("--components", is_flag=True, help="Also write clean signal and true D0..D4 columns.")
@click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
@click.option("--debug", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", is_flag=True, help="No progress messages on the console.")
@click.help_option("-h", "--help")
def main(out, noise_sd, seed, components, logdir, debug, quiet):
    """Regenerate the stochCycle tropical test tide."""
    level, console = resolve_loglevel(debug=debug, quiet=quiet)
    configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                      logfile_prefix="generate_test_tide")
    write_test_tide(out, noise_sd=noise_sd, seed=seed, components=components)


if __name__ == "__main__":
    main()
