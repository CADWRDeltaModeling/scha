# scha package guide

Architecture reference. See [AGENTS.md](../../AGENTS.md) for the rules that
govern changes and [scha_design.md](scha_design.md) for design decisions.

## Model

y_t = trend_t + sum_j cycle_j,t + eps_t. The trend is an m-th order integrated
random walk (subtide); each cycle is an n-th order stochastic cycle with damping
rho at a fixed frequency lambda_j (radians per sample). SCHA(m,n) names the
orders; Freeport work uses m = 2, n = 3 or 4, 6 or 8 cycles at k * M1.

## Using it

```python
import scha
fit = scha.fit(y)                     # pd.Series, naive DatetimeIndex with freq, NaN gaps
print(fit.summary())
dec = fit.decompose()                 # observed, fitted, subtide[, subtide_raw], D1.., residual
ap = scha.amplitude_phase(fit.phasors())   # <band>_a, _b, _amp, _phase (deg, lag)
fit.to_yaml("fit.yaml"); fit2 = scha.load_fit("fit.yaml")
```

Command line (same operations, `scha <command> -h` for options):

```text
scha fit series.csv -o fit.yaml [--config fit_config.yaml] [--bands D1-D6] [--fix rho=0.985]
scha decompose series.csv --fit fit.yaml -o components.csv
scha phasors | ampphase series.csv --fit fit.yaml -o out.csv
scha gains fit.yaml -o gains.csv
scha crossval series.csv --config cv.yaml -o cv.csv
```

Defaults of `scha.fit`: SCHA(2,3), bands D1-D6, constrained method,
`Requirements()` (delta = eps = 0.01, highest two bands sacrificial, 18 h
cosine-Lanczos post-filter on the subtide), automatic start. Parameters are
native to the step they were fitted at (`fit_freq` in the YAML).

## Module map (`src/scha`)

| Module | Purpose |
| --- | --- |
| `fit.py` | `fit`, `SCHAFit` (summary, YAML, decompose, fill, phasors, gains, innovations), `load_fit`, `decompose` |
| `cli.py` | `scha` command group; work functions `fit_file`, `decompose_file`, `phasors_file`, `ampphase_file`, `gains_file`, `crossval_file` |
| `mlfit.py` | `fit_ml`: (constrained) ML, concentrated or ordinary likelihood, fixed subsets, feasibility phase, parallel gradients |
| `requirements.py` | `Requirements` (user settings) and `CompiledRequirements` (margins, report) |
| `gains.py` | Closed-form component spectra and gains, state-space reference gains, cosine-Lanczos response, crossover |
| `bands.py` | Band registry: model lines (k * M1), named passband lines, band and sacrificial specs |
| `start.py` | `auto_start`: periodogram-matched starting values |
| `phase.py` | `demodulate`, `amplitude_phase`: local cos/sin coefficients, amplitude, phase lag |
| `crossval.py` | `cv_blocks`, `blocks`, `grid_cv`: gridded (rho, sigma2_zeta) cross-validation |
| `timestep.py` | `step_hours`, `freq_hours`, `lambdas_from_rph`, `lambdas_from_cpd`: the only place lambdas are computed |
| `params.py` | `SCHAParams` (m, n, rho, lambdas, sigma2_eps, sigma2_zeta, sigma2_kappa) |
| `state_space.py` | `build_scha_state_space`, `default_prior`, `state_scales` (`scale_rho`) |
| `uncond.py` | Unconditional cycle covariance (Trimbur closed form), level scales |
| `ffbs.py` | Reference Kalman filter, RTS smoother, Carter-Kohn draw |
| `fastll.py` | numba filter: `scha_loglik`, `scha_loglik_concentrated`, `scha_innovations`, `scha_smoothed_states` |
| `em_fit.py` | `scha_em` (EM with SQUAREM acceleration, fixed rho) |
| `scha_gibbs.py`, `scha_gibbs_scaffold.py`, `scha_postprocess.py` | Python port of the stochCycle Gibbs sampler and postprocessing |
| `post.py` | Instantaneous cycle amplitude and phase from states (superseded by `phase.py`) |
| `io.py` | `read_series` (`pd.read_csv(index_col=0, parse_dates=True, comment="#")`), `validate_series` (input contract); legacy parameter YAML loading |
| `tide_registry.py`, `naming.py` | Tidal constituent frequencies and labels |
| `logging_config.py` | `configure_logging`, `resolve_loglevel` for CLI drivers |

## Examples

| Folder | Contents |
| --- | --- |
| `examples/freeport` | `fit_freeport.py`: default `scha.fit` on Freeport 2014 discharge (dms-format CSV in m3/s; units do not affect the fit). The research scripts and `em_out/` moved to scha-paper `legacy/freeport` |
| `examples/tropical` | Synthetic tropical test: generator, EM, marginal MH, R parity |

Input files are dms-format CSVs (`datetime,value` plus a `#` YAML header, as
written by `dms_datastore.write_ts_csv`); gaps are NaN rows on a regular grid.
