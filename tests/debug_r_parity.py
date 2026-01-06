import numpy as np
import pandas as pd
from pathlib import Path
from scha.state_space import build_scha_state_space
from scha.ffbs import kalman_filter, kalman_smoother
from scha.io import load_scha_params_from_yaml

HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
# load R states
r_df=pd.read_csv(HERE / 'r_reference' / 'tropical_smoothed_states.csv')
if 'time' in r_df.columns:
    r_states = r_df.drop(columns=['time']).values
else:
    r_states = r_df.values
# load obs
df=pd.read_csv(HERE / 'tropical_sample.csv')
if 'y' in df.columns:
    y=df['y']
elif 'value' in df.columns:
    y=df['value']
else:
    y=df.iloc[:,1]
# params
params_obj,_=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss=build_scha_state_space(params_obj)
import yaml
cfg=yaml.safe_load(open(HERE / 'tropical_params.yml', 'r', encoding='utf-8'))
from scha.state_space import default_prior
sigma2_diffuse = float(cfg.get('sigma2_diffuse',1e6))
a0, P0 = default_prior(params_obj, sigma2_diffuse=sigma2_diffuse)
fil=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0)
ks=kalman_smoother(fil, T=ss['T'])
ms=ks['ms'].squeeze(-1)
print('ms shape', ms.shape, 'r shape', r_states.shape)
# align
if r_states.shape[0] > ms.shape[0]:
    r_trim=r_states[:ms.shape[0],:]
else:
    r_trim=r_states
# Compare fitted observation reconstructions
fitted_py = np.array([ (ss['Z'] @ ms[t].reshape(-1,1)).item() for t in range(ms.shape[0]) ])
if r_trim.shape[0] == fitted_py.shape[0]:
    fitted_r = np.array([ (ss['Z'] @ r_trim[t].reshape(-1,1)).item() for t in range(r_trim.shape[0]) ])
    print('\nMax abs fitted difference (R vs PY):', np.abs(fitted_r - fitted_py).max(), 'mean abs diff:', np.mean(np.abs(fitted_r - fitted_py)))
    print('\nFirst 10 fitted_R:', fitted_r[:10])
    print('First 10 fitted_PY:', fitted_py[:10])
    print('First 10 fitted diffs:', (fitted_r - fitted_py)[:10])
else:
    print('\nSkipping fitted comparison: shape mismatch')

# diagnostics
# Try small shifts in alignment between R and Python smoothed outputs to detect off-by-one
best = None
for shift in (-1, 0, 1):
    if shift < 0:
        r_try = r_states[-shift: ms.shape[0]-shift, :]
    else:
        r_try = r_states[shift: shift + ms.shape[0], :]
    if r_try.shape[0] != ms.shape[0]:
        continue
    d = np.abs(r_try - ms)
    print(f'Alignment shift {shift}: max abs diff =', d.max())
    if best is None or d.max() < best[0]:
        best = (d.max(), shift)

diff=np.abs(r_trim-ms)
print('max abs diff overall', diff.max())
print('Best alignment (shift, maxdiff):', best)
max_per_col=diff.max(axis=0)
order=np.argsort(-max_per_col)
print('\nTop 10 columns (index, maxdiff):')
for i in order[:10]:
    print(i, max_per_col[i])

print('\nFirst row R vs PY (first 8 states):')
print('R:', r_trim[0,:8])
print('PY:', ms[0,:8])

print('\nColumn-wise means R vs PY (first 10 cols):')
for i in range(10):
    print(i, r_trim[:,i].mean(), ms[:,i].mean())

# print first row differences for top columns
print('\nTop columns sample diffs (first 5 rows):')
for i in order[:5]:
    print('\ncol', i)
    print('R first5:', r_trim[:5,i])
    print('PY first5:', ms[:5,i])
    print('diff first5:', (r_trim[:5,i]-ms[:5,i]))

# compute per-cycle amplitudes (top pair in each cycle block)
m = 2
n = 4
Nc = 4
print('\nPer-cycle amplitude means:')
for c in range(Nc):
    start = m + c*(2*n)
    real_r = r_trim[:,start]
    imag_r = r_trim[:,start+1]
    amp_r = np.sqrt(real_r**2 + imag_r**2)
    real_p = ms[:,start]
    imag_p = ms[:,start+1]
    amp_p = np.sqrt(real_p**2 + imag_p**2)
    print(f'Cycle {c+1} amp mean: R {amp_r.mean():.6f}  PY {amp_p.mean():.6f}')
    print(f'  Cycle {c+1} amp max: R {amp_r.max():.6f}  PY {amp_p.max():.6f}')
    imax = amp_p.argmax()
    print(f'  PY peak at t index {imax}, value {amp_p[imax]:.6f}, corresponding ms values {real_p[imax]:.6f},{imag_p[imax]:.6f}')

# investigate large spikes in Python amplitudes
spike_threshold = 50.0
spike_times = np.where(np.sqrt((ms**2).sum(axis=1)) > spike_threshold)[0]
print('\nSpike times (norm of state vector > 50):', spike_times[:20])

# Identify times where Ft is very small
Ft_array = fil['F'].squeeze()
small_F_threshold = 1e-10
small_F_indices = np.where(Ft_array < small_F_threshold)[0]
print('\nIndices with Ft <', small_F_threshold, ':', small_F_indices[:20])
# Print initial predicted / filtered states to check for differences
print('\nInitial predicted a[0] (prior for t=0):', fil['a'][0].squeeze()[:5])
print('Initial filtered m[0]:', fil['m'][0].squeeze()[:5])
print('Top-left of C0 (from R):', c0_mat[:3,:3] if 'c0_mat' in globals() else 'C0 not loaded')
if small_F_indices.size > 0:
    for idx in small_F_indices[:10]:
        print(f' Ft[{idx}] =', Ft_array[idx], ' v[idx] =', fil['v'][idx].squeeze())

# EXPERIMENT: set initial a0 to R's first smoothed state and re-run filter/smoother
if r_trim.shape[0] >= 1:
    a0_guess = r_trim[0].copy()
    fil_guess = kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0_guess, P0=P0)
    ks_guess = kalman_smoother(fil_guess, T=ss['T'])
    ms_guess = ks_guess['ms'].squeeze(-1)
    diff_guess = np.abs(ms_guess - r_trim)
    print('\nSetting a0=r_trim[0]: max abs diff =', diff_guess.max(), 'mean abs diff =', diff_guess.mean())
    # also compare fitted
    fitted_guess = np.array([ (ss['Z'] @ ms_guess[t].reshape(-1,1)).item() for t in range(ms_guess.shape[0]) ])
    print('Max abs fitted diff after a0 guess:', np.abs(fitted_r - fitted_guess).max(), 'mean abs diff:', np.mean(np.abs(fitted_r - fitted_guess)))

# Unconditional compare: If the R C0 matrix is available, load and compare it to our P0_rstyle
from scha.uncond import cycle_unconditional_cov
sigma2_diffuse = float(cfg.get('sigma2_diffuse', 1e6))
P0_rstyle = np.zeros_like(P0)
# trend block: diffuse diagonal
m = ss['m']
for i in range(m):
    P0_rstyle[i, i] = sigma2_diffuse
# cycles: unconditional covariance per-cycle
n = ss['n']
Nc = ss['Nc']
for c in range(Nc):
    start = m + c*(2*n)
    end = start + 2*n
    X = cycle_unconditional_cov(n, params_obj.lambdas[c], params_obj.rho, params_obj.sigma2_kappa[c])
    P0_rstyle[start:end, start:end] = X

c0_path = HERE / 'r_reference' / 'C0.csv'
if c0_path.exists():
    print('\nFound R C0 at', c0_path)
    df_c0 = pd.read_csv(c0_path, header=0)
    # drop any non-numeric first column (row names) if present
    try:
        c0_mat = df_c0.values.astype(float)
    except Exception:
        df2 = df_c0.iloc[:, 1:]
        c0_mat = df2.values.astype(float)
    print('C0 shape:', c0_mat.shape)
    if c0_mat.shape == P0_rstyle.shape:
        diff_c0 = P0_rstyle - c0_mat
        print('max abs diff P0_rstyle - C0:', np.abs(diff_c0).max())
        print('Frobenius norms: ||P0_rstyle||, ||C0||, ||diff||:', np.linalg.norm(P0_rstyle), np.linalg.norm(c0_mat), np.linalg.norm(diff_c0))
        evals_p = np.linalg.eigvals(P0_rstyle)
        evals_c = np.linalg.eigvals(c0_mat)
        print('Top 6 eigenvalues P0_rstyle (real parts):', np.sort(np.real(evals_p))[-6:])
        print('Top 6 eigenvalues C0 (real parts):', np.sort(np.real(evals_c))[-6:])
        print('Relative Frobenius diff:', np.linalg.norm(diff_c0) / (np.linalg.norm(c0_mat) + 1e-30))
        print('\nPer-block Frobenius norm differences:')
        # trend
        t_block = np.linalg.norm(P0_rstyle[:m, :m] - c0_mat[:m, :m])
        print(' trend block diff norm:', t_block)
        for c in range(Nc):
            start = m + c*(2*n)
            end = start + 2*n
            block_diff = np.linalg.norm(P0_rstyle[start:end, start:end] - c0_mat[start:end, start:end])
            print(f' cycle {c} block diff norm:', block_diff)
    else:
        print('C0 shape mismatch vs expected P0 shape; cannot compare directly')
else:
    print('\nNo C0.csv found in r_reference — skipping C0 comparison')

if spike_times.size > 0:
    # inspect a few spike times
    import numpy.linalg as la
    for t in spike_times[:6]:
        print('\nInspecting filter at time t=', t)
        P_t1 = fil['P'][t+1]
        C_t = fil['C'][t]
        print('norms: ||P_t1||=', la.norm(P_t1), '||C_t||=', la.norm(C_t))
        # eigenvalues of P_t1
        evals = la.eigvals(P_t1)
        print('top eigvals P_t1 (real parts):', [float(ev.real) for ev in evals[:6]])
        # Ft, v, K
        Ft = fil['F'][t].squeeze()
        print('Ft:', Ft, ' innovation v:', fil['v'][t].squeeze())
        Kt = fil['K'][t].squeeze()
        print('Kt norm:', la.norm(Kt), ' max abs Kt:', np.abs(Kt).max())

    # Try constructing an "R-like" prior: diffuse trend but stationary cycle covariances
    from scha.uncond import cycle_unconditional_cov
    sigma2_diffuse = float(cfg.get('sigma2_diffuse', 1e6))
    P0_rstyle = np.zeros_like(P0)
    # trend block: diffuse diagonal
    m = ss['m']
    for i in range(m):
        P0_rstyle[i, i] = sigma2_diffuse
    # cycles: unconditional covariance per-cycle
    n = ss['n']
    Nc = ss['Nc']
    for c in range(Nc):
        start = m + c*(2*n)
        end = start + 2*n
        lam = params_obj.lambdas[c]
        s2k = params_obj.sigma2_kappa[c]
        X = cycle_unconditional_cov(n, lam, params_obj.rho, s2k)
        P0_rstyle[start:end, start:end] = X
    print('\nP0 (diag diffuse) norm:', la.norm(P0), 'P0_rstyle norm:', la.norm(P0_rstyle))

    # If the R C0 matrix is available, load and compare
    c0_path = HERE / 'r_reference' / 'C0.csv'
    if c0_path.exists():
        print('\nFound R C0 at', c0_path)
        df_c0 = pd.read_csv(c0_path, header=0)
        # drop any non-numeric first column (row names) if present
        try:
            c0_mat = df_c0.values.astype(float)
        except Exception:
            # attempt to drop first column
            df2 = df_c0.iloc[:, 1:]
            c0_mat = df2.values.astype(float)
        print('C0 shape:', c0_mat.shape)
        # basic comparisons
        assert c0_mat.shape == P0_rstyle.shape, f'C0 shape {c0_mat.shape} != expected {P0_rstyle.shape}'
        diff_c0 = P0_rstyle - c0_mat
        print('max abs diff P0_rstyle - C0:', np.abs(diff_c0).max())
        print('Frobenius norms: ||P0_rstyle||, ||C0||, ||diff||:', la.norm(P0_rstyle), la.norm(c0_mat), la.norm(diff_c0))
        # eigenvalues
        evals_p = np.linalg.eigvals(P0_rstyle)
        evals_c = np.linalg.eigvals(c0_mat)
        print('Top 6 eigenvalues P0_rstyle (real parts):', np.sort(np.real(evals_p))[-6:])
        print('Top 6 eigenvalues C0 (real parts):', np.sort(np.real(evals_c))[-6:])
        print('Relative Frobenius diff:', la.norm(diff_c0) / (la.norm(c0_mat) + 1e-30))
        # per-block comparison (trend + each cycle)
        m = ss['m']; n = ss['n']; Nc = ss['Nc']
        print('\nPer-block Frobenius norm differences:')
        # trend
        t_block = la.norm(P0_rstyle[:m, :m] - c0_mat[:m, :m])
        print(' trend block diff norm:', t_block)
        for c in range(Nc):
            start = m + c*(2*n)
            end = start + 2*n
            block_diff = la.norm(P0_rstyle[start:end, start:end] - c0_mat[start:end, start:end])
            print(f' cycle {c} block diff norm:', block_diff)
    else:
        print('\nNo C0.csv found in r_reference — skipping C0 comparison')

    # run filter/smoother with R-style prior
    fil_r = kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=np.zeros((ss['dim'],)), P0=P0_rstyle)
    ks_r = kalman_smoother(fil_r, T=ss['T'])
    ms_r = ks_r['ms'].squeeze(-1)
    print('R-style ms shape:', ms_r.shape)
    spike_times_r = np.where(np.sqrt((ms_r**2).sum(axis=1)) > spike_threshold)[0]
    print('Spike times with R-style prior:', spike_times_r[:20])

    # EXPERIMENT: scale cycle sigma2_kappa by dt_days (units-per-day vs per-sample mismatch hypothesis)
    _, meta = load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
    dt_days = meta['dt_days']
    print('\nTesting Q scaled by dt_days =', dt_days)
    ss2 = ss.copy()
    Q2 = ss2['Q'].copy()
    # scale cycle blocks (block starting at 1, then 1+2c..)
    for c in range(Nc):
        i0 = 1 + 2*c
        Q2[i0:i0+2, i0:i0+2] *= dt_days
    ss2['Q'] = Q2
    # recompute unconditional P0 for scaled sigma2_kappa
    P0_scaled = np.zeros_like(P0)
    for i in range(m):
        P0_scaled[i,i] = sigma2_diffuse
    for c in range(Nc):
        start = m + c*(2*n)
        end = start + 2*n
        s2k_scaled = params_obj.sigma2_kappa[c] * dt_days
        X = cycle_unconditional_cov(n, params_obj.lambdas[c], params_obj.rho, s2k_scaled)
        P0_scaled[start:end, start:end] = X
    fil_s = kalman_filter(y.values, T=ss2['T'], R=ss2['R'], Q=ss2['Q'], Z=ss2['Z'], H=ss2['H'], a0=np.zeros((ss2['dim'],)), P0=P0_scaled)
    ks_s = kalman_smoother(fil_s, T=ss2['T'])
    ms_s = ks_s['ms'].squeeze(-1)
    diff_scaled = np.abs(r_trim - ms_s)
    print('max abs diff when Q scaled by dt:', diff_scaled.max())
    # also print first row comparison
    print('\nFirst row R vs PY (scaled Q) (first 8 states):')
    print('R:', r_trim[0,:8])
    print('PY scaled:', ms_s[0,:8])

    # try the regularized filter and smoother (clip Ft)
    min_F = 1e-8
    fil_reg = kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=np.zeros((ss['dim'],)), P0=P0, min_F=min_F)
    ks_reg = kalman_smoother(fil_reg, T=ss['T'])
    ms_reg = ks_reg['ms'].squeeze(-1)
    print('\nRegularized ms shape:', ms_reg.shape)
    # compare to original
    diff_reg = np.abs(ms - ms_reg)
    print('max abs diff between original and regularized ms:', diff_reg.max())
    # check whether spike times persist
    spike_times_reg = np.where(np.sqrt((ms_reg**2).sum(axis=1)) > spike_threshold)[0]
    print('Spike times after regularization:', spike_times_reg[:20])



