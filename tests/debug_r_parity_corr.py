import numpy as np
import pandas as pd
from pathlib import Path
from scha.state_space import build_scha_state_space
from scha.ffbs import kalman_filter, kalman_smoother
from scha.io import load_scha_params_from_yaml

HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
r_df=pd.read_csv(HERE / 'r_reference' / 'tropical_smoothed_states.csv')
if 'time' in r_df.columns:
    r_states = r_df.drop(columns=['time']).values
else:
    r_states = r_df.values

# load python smoother
import yaml
params_obj,_=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss=build_scha_state_space(params_obj)
import numpy as np
import pandas as pd

df=pd.read_csv(HERE / 'tropical_sample.csv')
if 'y' in df.columns:
    y=df['y']
elif 'value' in df.columns:
    y=df['value']
else:
    y=df.iloc[:,1]

cfg=yaml.safe_load(open(HERE / 'tropical_params.yml', 'r', encoding='utf-8'))
from scha.state_space import default_prior
sigma2_diffuse = float(cfg.get('sigma2_diffuse',1e6))
a0, P0 = default_prior(params_obj, sigma2_diffuse=sigma2_diffuse)
fil=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0)
ks=kalman_smoother(fil, T=ss['T'])
ms=ks['ms'].squeeze(-1)
if r_states.shape[0] > ms.shape[0]:
    r_trim=r_states[:ms.shape[0],:]
else:
    r_trim=r_states

# compute correlation matrix
D = r_trim.shape[1]
corr = np.zeros((D,D))
for i in range(D):
    for j in range(D):
        ri = r_trim[:,i]
        pj = ms[:,j]
        if np.all(ri == ri[0]) or np.all(pj == pj[0]):
            corr[i,j] = 0.0
        else:
            corr[i,j] = np.corrcoef(ri, pj)[0,1]

# for each R column find best matching Python column
best = corr.argmax(axis=1)
vals = corr.max(axis=1)

print('Rcol -> Pycol (corr):')
for i in range(D):
    print(i, '->', best[i], f'(corr={vals[i]:.3f})')

# show mapping distribution
from collections import Counter
print('\nMapping distribution (Python column matched counts):')
print(Counter(best))

# show some mappings for top-diff columns
print('\nTop-diff columns and their matched Python column:')
max_per_col=np.abs(r_trim-ms).max(axis=0)
order=np.argsort(-max_per_col)
for i in order[:10]:
    print('R col', i, 'matched Py col', best[i], f'corr={vals[i]:.3f}', 'maxdiff=', max_per_col[i])

# try a linear least-squares mapping from Python states -> R states
A, *_ = np.linalg.lstsq(ms, r_trim, rcond=None)
print('\nLeast-squares mapping: A shape', A.shape)
# examine largest-magnitude entries by row
for i in order[:10]:
    row = A[:, i]
    idxs = np.argsort(-np.abs(row))[:6]
    print(f'R col {i} top python contributors:')
    for j in idxs:
        print(f'  Py col {j} coeff {row[j]:.4e} (corr {corr[i,j]:.3f})')
