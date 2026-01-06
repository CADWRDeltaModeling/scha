from pathlib import Path
import numpy as np
from scha.io import load_scha_params_from_yaml
from scha.state_space import build_scha_state_space
HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
params_obj,meta=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss=build_scha_state_space(params_obj)
T=ss['T']; Z=ss['Z']
print('dim:', ss['dim'], 'm:', ss['m'], 'n:', ss['n'], 'Nc:', ss['Nc'])
# Z nonzeros
z_idx = np.where(np.abs(Z) > 1e-12)[1]
print('\nZ nonzero indices (0-based):', z_idx)
print('Z values at nonzeros:', Z.flatten()[z_idx])
# show first row - human readable
print('\nZ row (1-based indices where value==1):')
print([i+1 for i in z_idx.tolist()])

# T sparsity summary
nz = np.argwhere(np.abs(T) > 1e-12)
print('\nNonzero entries in T (row,col,value) sample 40:')
for r,c in nz[:40]:
    print(r, c, float(T[r,c]))

# print block structure per cycle
m=ss['m']; n=ss['n']; Nc=ss['Nc']
for c in range(Nc):
    start = m + c*(2*n)
    print(f'\nCycle {c} block start {start+1} (1-based):')
    print(T[start:start+2, start:start+2])
    print('lower bidiagonal link to next pair (if any):')
    if 2*(n-1) > 0:
        print(T[start:start+2, start+2:start+4])

# print top-left trend block
print('\nTrend T block:')
print(T[:m,:m])

# show a small human friendly formatted matrix with rounding
np.set_printoptions(precision=6, suppress=True)
print('\nT (rounded) top-left 20x20):')
print(T[:20,:20])
