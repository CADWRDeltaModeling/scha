import numpy as np
from pathlib import Path
from scha.state_space import build_scha_state_space, default_prior
from scha.ffbs import kalman_filter, kalman_smoother
from scha.io import load_scha_params_from_yaml
import yaml

HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'


def test_default_prior_avoids_large_spikes():
    params_obj,_=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
    ss=build_scha_state_space(params_obj)
    cfg=yaml.safe_load(open(HERE / 'tropical_params.yml', 'r', encoding='utf-8'))
    sigma2_diffuse = float(cfg.get('sigma2_diffuse', 1e6))
    a0,P0 = default_prior(params_obj, sigma2_diffuse=sigma2_diffuse)

    import pandas as pd
    df=pd.read_csv(HERE / 'tropical_sample.csv')
    if 'y' in df.columns:
        y=df['y']
    elif 'value' in df.columns:
        y=df['value']
    else:
        y=df.iloc[:,1]

    fil = kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0)
    ks = kalman_smoother(fil, T=ss['T'])
    ms = ks['ms'].squeeze(-1)

    spike_threshold = 50.0
    spike_times = np.where(np.sqrt((ms**2).sum(axis=1)) > spike_threshold)[0]
    assert spike_times.size == 0, f"Default prior produced spikes at times: {spike_times[:10]}"