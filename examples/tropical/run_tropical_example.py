"""
Run the Tropical example: read CSV of observations and YAML params, build state-space, run Kalman smoother, and save smoothed series.
"""
from __future__ import annotations
import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

from scha.params import SCHAParams
from scha.state_space import build_scha_state_space
from scha.ffbs import kalman_filter, kalman_smoother
from scha.logging_config import configure_logging, resolve_loglevel

HERE = Path(__file__).resolve().parent
logger = logging.getLogger("scha.examples.run_tropical_example")


def read_params(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required to read YAML params. Install with 'pip install pyyaml'.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def run(csv_path: Path | str = None, params_path: Path | str = None, out_path: Path | str = None):
    csv_path = Path(csv_path or (HERE / "tropical_sample.csv"))
    params_path = Path(params_path or (HERE / "tropical_params.yml"))
    out_path = Path(out_path or (HERE / "tropical_smoothed.csv"))

    from scha.io import load_scha_params_from_yaml, read_series

    y = read_series(csv_path)
    p, _ = load_scha_params_from_yaml(params_path, index=y.index)

    ss = build_scha_state_space(p)

    # initial mean and prior covariance — prefer R-style default prior (trend diffuse + stationary cycle cov)
    from scha.state_space import default_prior
    cfg = read_params(params_path)
    sigma2_diffuse = float(cfg.get("sigma2_diffuse", 1e6))
    a0, P0 = default_prior(p, sigma2_diffuse=sigma2_diffuse)

    filt = kalman_filter(y.values, T=ss["T"], R=ss["R"], Q=ss["Q"], Z=ss["Z"], H=ss["H"], a0=a0, P0=P0)
    ks = kalman_smoother(filt, T=ss["T"])
    ms = ks["ms"].squeeze(-1)  # shape (T,d)

    # compute smoothed fitted values: Z @ ms[t]
    fitted = np.array([ (ss["Z"] @ ms[t].reshape(-1,1)).item() for t in range(ms.shape[0]) ])
    out = pd.DataFrame({"time": y.index, "y": y.values, "fitted": fitted})
    out.to_csv(out_path, index=False)
    logger.info("Wrote smoothed result to %s", out_path)


@click.command()
@click.argument("csv_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("params_path", required=False, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
@click.option("--debug", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", is_flag=True, help="No progress messages on the console.")
@click.help_option("-h", "--help")
def main(csv_path, params_path, logdir, debug, quiet):
    """Smooth CSV_PATH with PARAMS_PATH (defaults: tropical_sample.csv, tropical_params.yml)."""
    level, console = resolve_loglevel(debug=debug, quiet=quiet)
    configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                      logfile_prefix="run_tropical_example")
    run(csv_path=csv_path, params_path=params_path)


if __name__ == "__main__":
    main()
