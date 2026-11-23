from __future__ import annotations

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd

from scha.io import load_scha_params_from_yaml
from scha.state_space import build_scha_state_space, default_prior
from scha.ffbs import kalman_filter, kalman_smoother
from scha.em_fit import scha_em
from scha.logging_config import configure_logging, resolve_loglevel

logger = logging.getLogger("scha.examples.fit_em_tropical")


def _write_yaml(path: Path, payload: dict) -> None:
    """Write YAML using PyYAML (fail-fast if missing)."""
    try:
        import yaml
    except Exception as e:
        raise ImportError("PyYAML is required to write YAML") from e

    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False)


def fit_em_tropical() -> None:
    """EM fit of the tropical example (rho fixed); writes params and filter/smoother states to py_em_out/."""
    # Keep the driver tropical-specific; EM logic lives in scha.em_fit
    HERE = Path(__file__).resolve().parent

    yaml_path = HERE / "tropical_params_start.yml"
    csv_path = HERE / "tropical_sample.csv"

    df=pd.read_csv(HERE / 'tropical_sample.csv',index_col=0,header=0)
    rng = np.random.default_rng(12345)
    y=df['value']
    sigma_obs = 0.001
    y = y + rng.normal(0.0, sigma_obs, size=y.shape)


    params, meta = load_scha_params_from_yaml(yaml_path,index=df.index)
    import yaml
    with open(yaml_path, "r", encoding="utf-8") as f:
        sigma2_diffuse = float(yaml.safe_load(f).get("sigma2_diffuse", 1.0e4))

    # Per your guidance: fix rho here (EM does not update rho)
    params.rho = 0.9935

    # Safe starting points
    params.sigma2_zeta = 1.0e-9
    params.sigma2_kappa = [1.35e-13] * len(params.lambdas)

    logger.info("start params: %s", params)
    # -----------------
    # EM fit (variances)
    # -----------------
    fitted = scha_em(
        y,
        params,
        max_iter=50,
        fix_sigma2_eps=False,
        sigma2_diffuse=sigma2_diffuse,
        verbose=True,
    )

    logger.info("fitted params (variances only): rho=%s sigma2_eps=%s sigma2_zeta=%s sigma2_kappa=%s",
                fitted.rho, fitted.sigma2_eps, fitted.sigma2_zeta, fitted.sigma2_kappa)

    # ------------------------------
    # Run KF/RTS with fitted params
    # ------------------------------
    ss = build_scha_state_space(fitted)

    a0, P0 = default_prior(fitted, sigma2_diffuse=sigma2_diffuse)

    filt = kalman_filter(
        y,
        T=ss["T"],
        R=ss["R"],
        Q=ss["Q"],
        Z=ss["Z"],
        H=ss["H"],
        a0=a0,
        P0=P0,
        missing_is_nan=True,
        joseph=True,
    )
    smooth = kalman_smoother(filt, T=ss["T"])

    # Save useful artifacts in the same convention you already used
    out_dir = HERE / "py_em_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoothed states
    ms = smooth["ms"].squeeze(-1)  # (T,d)
    pd.DataFrame(ms).to_csv(out_dir / "smoothed_states_ms.csv", index=False)

    # Smoothed covariance diagonal
    Cs = smooth["Cs"]
    Cs_diag = np.diagonal(Cs, axis1=1, axis2=2)
    pd.DataFrame(Cs_diag).to_csv(out_dir / "smoothed_cov_diag.csv", index=False)

    # Filtered states (m) and predicted (a)
    m = filt["m"].squeeze(-1)
    a = filt["a"].squeeze(-1)
    pd.DataFrame(m).to_csv(out_dir / "filtered_states_m.csv", index=False)
    pd.DataFrame(a).to_csv(out_dir / "predicted_states_a.csv", index=False)

    # Write a minimal fitted YAML (keeps original freqs/dt/m/n etc from input)
    fitted_yaml = {
        "dt": meta.get("dt_days", None),
        "m": int(fitted.m),
        "n": int(fitted.n),
        "rho": float(fitted.rho),
        "freqs": (meta.get("labels", None) if meta.get("labels", None) else None),
        "sigma2_eps": float(fitted.sigma2_eps),
        "sigma2_zeta": float(fitted.sigma2_zeta),
        "sigma2_kappa": [float(v) for v in fitted.sigma2_kappa],
    }
    # Drop Nones so we don't accidentally clobber your schema
    fitted_yaml = {k: v for k, v in fitted_yaml.items() if v is not None}

    _write_yaml(out_dir / "params_fitted.yml", fitted_yaml)

    for name in ("params_fitted.yml", "smoothed_states_ms.csv", "smoothed_cov_diag.csv",
                 "filtered_states_m.csv", "predicted_states_a.csv"):
        logger.info("Wrote %s", out_dir / name)


@click.command()
@click.option("--logdir", type=click.Path(path_type=Path), default=None, help="Directory for a log file.")
@click.option("--debug", is_flag=True, help="Debug-level logging.")
@click.option("--quiet", is_flag=True, help="No progress messages on the console.")
@click.help_option("-h", "--help")
def main(logdir, debug, quiet):
    """EM fit of the tropical example; writes py_em_out/."""
    level, console = resolve_loglevel(debug=debug, quiet=quiet)
    configure_logging(package_name="scha", level=level, console=console, logdir=logdir,
                      logfile_prefix="fit_em_tropical")
    fit_em_tropical()


if __name__ == "__main__":
    main()
