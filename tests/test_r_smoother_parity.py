import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from scha.state_space import build_scha_state_space
from scha.ffbs import kalman_filter, kalman_smoother
from scha.io import load_scha_params_from_yaml

HERE = Path(__file__).resolve().parent.parent / 'examples' / 'tropical'


def _load_obs(csv_path: Path):
    df = pd.read_csv(csv_path)
    # Accept either 'y' or 'value' as the observation column
    if 'y' in df.columns:
        y = df['y']
    elif 'value' in df.columns:
        y = df['value']
    else:
        # try second column
        y = df.iloc[:, 1]
    # preserve index as numeric time if present
    if 'time' in df.columns:
        idx = df['time']
    else:
        idx = np.arange(len(y))
    return y.values, idx


def test_r_smoother_parity():
    r_csv = HERE / 'r_reference' / 'tropical_smoothed_states.csv'
    if not r_csv.exists():
        pytest.skip("R smoothed-state CSV not present in examples/tropical/r_reference")

    # load R smoothed states
    r_df = pd.read_csv(r_csv)
    # Expect first column to be time and remaining to be state_1..state_d
    if 'time' in r_df.columns:
        r_states = r_df.drop(columns=['time']).values
    else:
        # assume all columns are states
        r_states = r_df.values

    # load observations and YAML params
    obs_csv = HERE / 'tropical_sample.csv'
    y, idx = _load_obs(obs_csv)

    params_path = HERE / 'tropical_params.yml'
    params_obj, metadata = load_scha_params_from_yaml(params_path, dt_spec=None, infer_dt_from_index=False)

    ss = build_scha_state_space(params_obj)

    # initial state and prior covariance — use default R-style prior (trend diffuse + stationary cycle cov)
    from scha.state_space import default_prior
    import yaml
    cfg = yaml.safe_load(open(params_path, 'r', encoding='utf-8'))
    sigma2_diffuse = float(cfg['sigma2_diffuse'])
    a0, P0 = default_prior(params_obj, sigma2_diffuse=sigma2_diffuse)

    filt = kalman_filter(y, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0)
    ks = kalman_smoother(filt, T=ss['T'])
    ms = ks['ms'].squeeze(-1)  # shape (T, d)

    T_py, d = ms.shape
    T_r, d_r = r_states.shape

    # Ensure dimension agreement
    assert d_r == d, f"R states columns ({d_r}) != Python state dimension ({d})"

    # Align lengths: if R has extra rows, trim to Python length; if R shorter, fail
    if T_r > T_py:
        r_trim = r_states[:T_py, :]
    elif T_r == T_py:
        r_trim = r_states
    else:
        raise AssertionError(f"R smoothed output has fewer rows ({T_r}) than Python smoother ({T_py})")

    # Compare element-wise, but report max absolute difference on failure for debugging
    ok = np.allclose(r_trim, ms, rtol=1e-6, atol=1e-8)
    diff = np.abs(r_trim - ms)
    maxdiff = float(diff.max())
    assert ok, f"Smoothed states differ beyond tolerance; max abs diff {maxdiff:e}"
