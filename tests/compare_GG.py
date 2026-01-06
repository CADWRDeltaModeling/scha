from pathlib import Path
import numpy as np
import pandas as pd
from scha.io import load_scha_params_from_yaml
from scha.state_space import build_scha_state_space

HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
# load R GG
gg_path = HERE / 'r_reference' / 'GG.csv'
if not gg_path.exists():
    raise SystemExit(f'GG.csv not found at {gg_path}')

df = pd.read_csv(gg_path, header=0)
# if first column is non-numeric (row names), drop
try:
    gg_mat = df.values.astype(float)
except Exception:
    gg_mat = df.iloc[:,1:].values.astype(float)

# build python T
params_obj,_ = load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss = build_scha_state_space(params_obj)
T = ss['T']

print('Shapes: T', T.shape, 'GG', gg_mat.shape)
if T.shape != gg_mat.shape:
    print('Shape mismatch -> exiting')
    raise SystemExit(0)

# elementwise difference
diff = T - gg_mat
max_abs = np.max(np.abs(diff))
print('Max abs diff:', max_abs)
print('Frobenius norm of diff:', np.linalg.norm(diff))
print('Relative Frobenius:', np.linalg.norm(diff) / (np.linalg.norm(gg_mat) + 1e-30))

# list large differences
tol_small = 1e-12
tol_warn = 1e-8
inds = np.argwhere(np.abs(diff) > tol_warn)
print('Number of entries > 1e-8:', len(inds))
if len(inds) > 0:
    print('Examples (row+1, col+1, python, R, diff):')
    for i, (r,c) in enumerate(inds[:20]):
        print(r+1, c+1, float(T[r,c]), float(gg_mat[r,c]), float(diff[r,c]))

# per-cycle block comparisons
m = ss['m']; n = ss['n']; Nc = ss['Nc']
print('\nPer-block Frobenius diffs:')
print('trend block:', np.linalg.norm(T[:m,:m] - gg_mat[:m,:m]))
for c in range(Nc):
    start = m + c*(2*n)
    end = start + 2*n
    block_diff = np.linalg.norm(T[start:end, start:end] - gg_mat[start:end, start:end])
    print(f' cycle {c} block diff norm:', block_diff)

print('\nIf differences are tiny (<1e-8) they are numerical; otherwise inspect entries shown above.')
