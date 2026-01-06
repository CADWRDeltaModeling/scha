"""
Run the Tropical example: read CSV of observations and YAML params, build state-space, run Kalman smoother, and save smoothed series.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
import numpy as np
import pandas as pd

try:
    import yaml
except Exception:
    yaml = None

from scha.params import SCHAParams
from scha.state_space import build_scha_state_space
from scha.ffbs import kalman_filter, kalman_smoother

HERE = Path(__file__).resolve().parent


def read_params(path: Path) -> dict:
    if yaml is None:
        raise ImportError("PyYAML is required to read YAML params. Install with 'pip install pyyaml'.")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def cpd_to_lambda(cpd: float, dt: float) -> float:
    # convert cycles-per-day to radians per time-step where dt is days per sample
    return 2.0 * math.pi * float(cpd) * float(dt)


def run(csv_path: Path | str = None, params_path: Path | str = None, out_path: Path | str = None):
    csv_path = Path(csv_path or (HERE / "tropical_sample.csv"))
    params_path = Path(params_path or (HERE / "tropical_params.yml"))
    out_path = Path(out_path or (HERE / "tropical_smoothed.csv"))

    # Robust CSV loading: accept either a named time column or index as first column
    df_try = pd.read_csv(csv_path)
    if "time" in df_try.columns:
        df = pd.read_csv(csv_path, parse_dates=["time"]).set_index("time")
    else:
        # assume first column is index
        df = pd.read_csv(csv_path, index_col=0)
        # try to parse index as datetimes
        try:
            idx_dt = pd.to_datetime(df.index, errors='coerce')
            if not idx_dt.isnull().all():
                df.index = idx_dt
        except Exception:
            pass

    # Accept either 'y' (standard) or 'value' (example file) as the observation column
    if 'y' in df.columns:
        y = df['y'].squeeze()
    elif 'value' in df.columns:
        y = df['value'].squeeze()
    else:
        # fallback to the first column if unlabelled
        y = df.iloc[:, 0].squeeze()

    # Decide how to get dt: if datetime index present, infer; if numeric index, require time_unit in YAML
    index_is_datetime = pd.api.types.is_datetime64_any_dtype(y.index)

    # Use strict loader from src.io
    from scha.io import load_scha_params_from_yaml

    if index_is_datetime:
        params_obj, metadata = load_scha_params_from_yaml(params_path, infer_dt_from_index=True, index=y.index)
        p = params_obj
    else:
        # numeric index: read raw params and prefer explicit 'time_unit' but accept 'dt' as well
        params_raw = read_params(params_path)
        time_unit = params_raw.get("time_unit", None)
        if time_unit is None:
            # fall back to 'dt' if present
            time_unit = params_raw.get("dt", None)
        if time_unit is None:
            raise ValueError("Numeric time index detected in CSV; YAML must include 'time_unit' or 'dt' (e.g., '1h' or '15min').")
        # normalize short names like 'hour' -> '1h'
        tu = str(time_unit).strip().lower()
        if tu in ["hour", "hours", "h"]:
            dt_spec = "1h"
        elif tu in ["min", "mins", "minute", "minutes"]:
            dt_spec = "1min"
        elif tu in ["day", "days", "d"]:
            dt_spec = "1d"
        else:
            # allow direct spec like '15min' or numeric days-per-sample
            dt_spec = time_unit
        params_obj, metadata = load_scha_params_from_yaml(params_path, dt_spec=dt_spec, infer_dt_from_index=False)
        p = params_obj

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
    print(f"Wrote smoothed result to {out_path}")


if __name__ == "__main__":
    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    params_path = sys.argv[2] if len(sys.argv) > 2 else None
    run(csv_path=csv_path, params_path=params_path)
