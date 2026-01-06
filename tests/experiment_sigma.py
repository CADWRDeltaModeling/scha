import numpy as np
from pathlib import Path
from scha.io import load_scha_params_from_yaml
from scha.state_space import build_scha_state_space, default_prior
from scha.ffbs import kalman_filter, kalman_smoother
import pandas as pd
HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
params_obj,_=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss=build_scha_state_space(params_obj)
df=pd.read_csv(HERE / 'tropical_sample.csv')
y=df['value']
import yaml
cfg=yaml.safe_load(open(HERE / 'tropical_params.yml', 'r', encoding='utf-8'))
for s in [1e6,1000,100,10,1,0.1,0.01,0.0]:
    a0,P0=default_prior(params_obj, sigma2_diffuse=s)
    fil=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0)
    ks=kalman_smoother(fil, T=ss['T'])
    ms=ks['ms'].squeeze(-1)
    r_df=pd.read_csv(HERE / 'r_reference' / 'tropical_smoothed_states.csv')
    r_states = r_df.drop(columns=['time']).values
    r_trim = r_states[:ms.shape[0],:]
    d = np.abs(ms - r_trim)
    print('sigma2_diffuse',s,'maxdiff',d.max(),'mean',d.mean())
