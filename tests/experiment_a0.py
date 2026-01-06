from pathlib import Path
from scha.io import load_scha_params_from_yaml
from scha.state_space import build_scha_state_space, default_prior
from scha.ffbs import kalman_filter, kalman_smoother
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent.parent / 'examples' / 'tropical'
params_obj,_=load_scha_params_from_yaml(HERE / 'tropical_params.yml', dt_spec=None, infer_dt_from_index=False)
ss=build_scha_state_space(params_obj)
df=pd.read_csv(HERE / 'tropical_sample.csv')
y=df['value']
from scha.state_space import default_prior
a0_py,P0_py = default_prior(params_obj, sigma2_diffuse=4)

# ---- Initialization to match dlm convention for this test ----
# R prior mean is zero in your setup
d = ss["T"].shape[0]
a0 = np.zeros(d, dtype=float)

# Use R's C0 exactly (already shown to match PY's generated covariance closely)
P0 = pd.read_csv(HERE / 'r_reference' / 'C0.csv', header=0).values
if P0.shape != (d, d):
    raise ValueError(f"C0 shape {P0.shape} does not match state dim {d}")
P0 = 0.5 * (P0 + P0.T)

# ---- Optional diagnostics: compare Python default prior vs R prior ----
P0r = pd.read_csv(HERE / 'r_reference' / 'C0.csv', header=0).values
P0r = 0.5 * (P0r + P0r.T)
P0p = 0.5 * (P0_py + P0_py.T)
print("P0 diag compare (first 10):")
print("  R :", np.diag(P0r)[:10])
print("  PY:", np.diag(P0p)[:10])
print("P0 max abs diff:", np.max(np.abs(P0p - P0r)), "mean abs diff:", np.mean(np.abs(P0p - P0r)))


print("P0")
print(P0[:5,:5 ])
P0[0,0] = 4.
P0[1,1] = 4.

fil=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0,joseph=True)


# Python sequences
a_py = fil["a"].squeeze(-1)  # (T, d)
m_py = fil["m"].squeeze(-1)  # (T, d)

# R filtered m (your existing)
r_m_df = pd.read_csv(HERE / 'r_reference' / 'tropical_filtered_states_m.csv')
r_m_all = r_m_df.drop(columns=['time']).values
r_m = r_m_all[1:, :]  # dlm prepends prior row

# R predicted a (new)
r_a_df = pd.read_csv(HERE / 'r_reference' / 'tropical_filtered_states_a.csv')
r_a_all = r_a_df.drop(columns=['time']).values
r_a = r_a_all[1:, :]  # drop dlm prepended row

Tmin = min(m_py.shape[0], r_m.shape[0], a_py.shape[0], r_a.shape[0])
m_py = m_py[:Tmin]; r_m = r_m[:Tmin]
a_py = a_py[:Tmin]; r_a = r_a[:Tmin]

# Compare predicted a
da = a_py - r_a
print("PREDICTED a parity: max abs diff", np.max(np.abs(da)), "mean abs diff", np.mean(np.abs(da)))
print("first 2 rows R vs PY (predicted a)")
print("R:", r_a[0:3,:8])
print("PY:", a_py[0:3,:8])

# Compare filtered m
dm = m_py - r_m
print("FILTERED m parity: max abs diff", np.max(np.abs(dm)), "mean abs diff", np.mean(np.abs(dm)))
print("first 2 rows R vs PY (filtered m)")
print("R:", r_m[0:3,:8])
print("PY:", m_py[0:3,:8])




fil_j=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0,joseph=True)
fil_n=kalman_filter(y.values, T=ss['T'], R=ss['R'], Q=ss['Q'], Z=ss['Z'], H=ss['H'], a0=a0, P0=P0,joseph=False)


dC0 = fil_j["C"][0] - fil_n["C"][0]
print("Joseph vs non-Joseph: |ΔC[0]| max", np.max(np.abs(dC0)), "mean", np.mean(np.abs(dC0)))

dC1 = fil_j["C"][1] - fil_n["C"][1]
print("Joseph vs non-Joseph: |ΔC[1]| max", np.max(np.abs(dC1)), "mean", np.mean(np.abs(dC1)))

print("y0", float(y.values[0]))
print("v0", float(fil["v"][0,0,0]))
print("F0", float(fil["F"][0,0,0]))
print("K0 first 8", fil["K"][0,:8,0])


# Pick which filtered result you want to smooth
fil = fil_j  # or fil_n

# Run RTS smoother
smooth = kalman_smoother(fil, T=ss["T"])
ms = smooth["ms"].squeeze(-1)   # (T, d)

# Load R reference (timing is pre-aligned per your note)
r_path = HERE / "r_reference" / "tropical_smoothed_states.csv"
r_df = pd.read_csv(r_path)

# Allow one optional time column with a standard name; otherwise fail fast
time_cols = [c for c in r_df.columns if c.lower() in ("time", "timestamp", "datetime", "date")]
if len(time_cols) == 1:
    r_states_df = r_df.drop(columns=time_cols)
elif len(time_cols) == 0:
    r_states_df = r_df
else:
    raise ValueError(f"Expected 0 or 1 time column, found {time_cols}")

r_states = r_states_df.to_numpy(dtype=float)

# Strict shape checks
if r_states.shape[0] != ms.shape[0]:
    raise ValueError(f"Row mismatch: R has {r_states.shape[0]} times, python has {ms.shape[0]}")
if r_states.shape[1] != ms.shape[1]:
    raise ValueError(f"State-dim mismatch: R has {r_states.shape[1]} states, python has {ms.shape[1]}")

diff = ms - r_states
# Where does the worst mismatch occur?
imax = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
t_max, k_max = imax
print("\nWorst point:")
print("  t index =", t_max, " state =", k_max, " abs(diff) =", float(np.abs(diff[imax])))

# If you have time index available in your main dataframe, print it too
# (adjust the variable name to whatever you use for the observation time index)
if "time_cols" in locals() and len(time_cols) == 1:
    # time is in r_df and should align to your internal timeline
    print("  time =", r_df[time_cols[0]].iloc[t_max])


rmse_per_state = np.sqrt(np.mean(diff**2, axis=0))
maxabs_per_state = np.max(np.abs(diff), axis=0)

print("\n=== Smoothed state comparison vs R ===")
for i in range(ms.shape[1]):
    print(f"state[{i:02d}]  rmse={rmse_per_state[i]:.6g}  maxabs={maxabs_per_state[i]:.6g}")

print("overall rmse =", float(np.sqrt(np.mean(diff**2))))
print("overall maxabs =", float(np.max(np.abs(diff))))


# Python-side smoothed covariance diagonals
Cs = smooth["Cs"]                       # (T, d, d)
Cs_diag = np.diagonal(Cs, axis1=1, axis2=2)

# Load R reference covariance diagonals
r_cov_path = HERE / "r_reference" / "tropical_smoothed_cov_diag.csv"
r_cov_df = pd.read_csv(r_cov_path)
r_cov = r_cov_df.to_numpy(dtype=float)

# Strict shape checks
if r_cov.shape != Cs_diag.shape:
    raise ValueError(
        f"Cov diag shape mismatch: R {r_cov.shape} vs python {Cs_diag.shape}"
    )

# Basic sanity (Python side should already be PSD)
if np.min(Cs_diag) < -1e-10:
    raise ValueError(f"Python smoothed cov diag has negative values: min={np.min(Cs_diag)}")

# Differences
diff_cov = Cs_diag - r_cov

rmse_per_state = np.sqrt(np.mean(diff_cov**2, axis=0))
maxabs_per_state = np.max(np.abs(diff_cov), axis=0)

print("\n=== Smoothed covariance diagonal comparison vs R ===")
for i in range(Cs_diag.shape[1]):
    print(
        f"state[{i:02d}]  "
        f"rmse={rmse_per_state[i]:.6g}  "
        f"maxabs={maxabs_per_state[i]:.6g}"
    )

print("overall rmse =", float(np.sqrt(np.mean(diff_cov**2))))
print("overall maxabs =", float(np.max(np.abs(diff_cov))))
imax = np.unravel_index(np.argmax(np.abs(diff_cov)), diff_cov.shape)
print(
    "worst cov diag diff at "
    f"t={imax[0]}, state={imax[1]}, "
    f"abs(diff)={float(np.abs(diff_cov[imax]))}"
)

absdiff = np.abs(diff_cov)

print("\nWorst abs diff by time (top 100):")
t_worst = np.argsort(absdiff.max(axis=1))[::-1][:100]
for t in t_worst:
    k = int(np.argmax(absdiff[t, :]))
    print(f"t={t:5d}  state={k:02d}  maxabs={absdiff[t, k]:.6g}")

print("\nWorst abs diff by state (top 100:")
k_worst = np.argsort(absdiff.max(axis=0))[::-1][:100]
for k in k_worst:
    t = int(np.argmax(absdiff[:, k]))
    print(f"state={k:02d}  t={t:5d}  maxabs={absdiff[t, k]:.6g}")

def rmse(a, b):
    d = a - b
    return float(np.sqrt(np.mean(d * d)))

print("\n=== Cov diag alignment scan (R relative to Python) ===")
for shift in range(-3, 4):
    # compare python[t] vs r[t+shift]
    if shift < 0:
        a = Cs_diag[-shift:, :]
        b = r_cov[: Cs_diag.shape[0] + shift, :]
    elif shift > 0:
        a = Cs_diag[: Cs_diag.shape[0] - shift, :]
        b = r_cov[shift:, :]
    else:
        a = Cs_diag
        b = r_cov
    print(f"shift={shift:+d}  rmse={rmse(a,b):.6g}")    

    r_cov_D  = pd.read_csv(HERE / "r_reference" / "tropical_smoothed_cov_diag_D.csv").to_numpy(float)
r_cov_D2 = pd.read_csv(HERE / "r_reference" / "tropical_smoothed_cov_diag_D2.csv").to_numpy(float)

def overall_rmse(A, B):
    d = A - B
    return float(np.sqrt(np.mean(d*d)))

print("RMSE vs R using U diag(D)  U^T :", overall_rmse(Cs_diag, r_cov_D))
print("RMSE vs R using U diag(D^2)U^T :", overall_rmse(Cs_diag, r_cov_D2))

