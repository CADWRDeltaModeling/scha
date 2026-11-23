# SCHA parameter estimation: design notes

Status: working notes, updated 2026-11-22. Test case: Freeport 2014 discharge,
15-min, 18001 samples (about 6 months), units of 100 m3/s.

## 1. Purpose of the constraints

The fit should agree with what we know from outside the data: the tide is forced
in distinct bands (D1, D2, ...), and the subtide is a separate component. Behaving
well as a stationary filter is necessary for good nonstationary behaviour, but
probably not sufficient.

## 2. Requirements are stated as filter gains

- The gains are the steady-state smoother gains implied by the model,
  G_j(f) = S_j(f) / (sum_k S_k(f) + sigma2_eps). They describe the filter itself,
  so they keep their meaning in a nonstationary record, and they are stated in
  cpd, so they carry over between time steps.
- **Tidal bands:** G_j >= 1 - delta on each band's named lines, with cusp room
  where needed (D1: 0.83-1.13 cpd). Because the gains plus the noise share sum
  to 1, this also holds every other component, including noise, to at most delta
  on those lines.
- **Between tidal bands:** no requirement. Neighbouring bands split the gaps
  between themselves.
- **Sacrificial bands (D5 and above at Freeport):** no passband requirement.
  Their role is to absorb high-frequency content that would otherwise go to the
  subtide, the noise, or the main bands.
- **Subtide:** G_sub <= eps at the gap peaks between bands and above the last
  main band.
- **Optional subtide/D1 crossover:** G_sub >= G_D1 at a chosen frequency. Without
  some requirement of this kind nothing stops D1 from taking over the subtide
  (seen as a poor local optimum: crossover at 0.05 cpd).
- **Current settings:** delta = eps = 0.01.

### What cannot be required

A subtide passband edge (G_sub >= 1 - delta up to F) together with the D1 window
(G_D1 >= 1 - delta from 0.83 cpd) needs the subtide/D1 spectral ratio to fall by
((1 - delta)/delta)^2, about 9800, between F and 0.83. That fall depends only on
rho and the spectral shapes, not the variances. At rho near 0.983 the highest
feasible F is about 0.39 cpd (0.435 if the D1 window starts at Q1); F = 0.6 needs
rho > 0.998. A target near 0.6 cpd is therefore realistic only as a crossover.

## 3. Requirements are constraints, not likelihood-weighted priors

The scale of the log-likelihood depends on record length and on the units of y,
so a fixed exchange rate between likelihood and filter quality would mean
different things for different records. The log-likelihood given up is a
diagnostic, not something we tune. A unit-free reading:
exp(2 dll / N) - 1 = relative increase in one-step prediction-error variance.

## 4. Parameterisation

Use the original parameters (sigma2_eps, sigma2_zeta, sigma2_kappa, rho), for
consistency with prior work. Internally the optimiser works on log variance
ratios to sigma2_eps and log(1 - rho), with sigma2_eps profiled out
(concentrated likelihood); results are reported in the original parameters.
No other recast has proved necessary.

## 5. Excluded

- Tolerances in physical units (not realisable in a nonstationary record).
- External benchmark filters.
- Steady-state (time-invariant) filtering: gaps are a feature, so the filter
  must run step by step.

## 6. Implementation

- `scha.fastll`: numba Kalman filter for the likelihood, exploiting the sparse
  transition and observation. Equal to the reference filter to about 1e-8, with
  and without gaps; 0.26 s per evaluation for 6 cycles (was 0.9 s).
  - `scha_loglik`: ordinary likelihood.
  - `scha_loglik_concentrated`: sigma2_eps profiled out; diffuse trend prior in
    relative units and the first m observations dropped, so the concentration
    is exact.
  - `scha_innovations`: standardized one-step innovations.
- `examples/freeport/constrained_fit.py`: SLSQP with closed-form gains (equal to
  the state-space gains to about 1e-9..1e-6).
  - Options: `--n` (cycle order), `--ncycles`, `--free-bands` (repeat the option
    for several bands), `--sub-cross`, `--post-filter`, `--fix-eps`, `--fix-rho`,
    `--no-constraints`, `--trust`; `-h` for the full list.
  - Robustness: NaN guard on the likelihood, diagonal preconditioning, and a
    feasibility phase (nearest point meeting the requirements, no likelihood)
    before SLSQP.
  - Start constrained fits from a constrained solution; crossover requirements
    are raised by continuation (steps of 0.01-0.02 cpd, trust box about 2 log
    units).
- State scaling (`scale_rho`): see below. Used by default in all fitting
  scripts.
- Library form (2026-11): `scha.fit` (`fit.py`) is the user API; `mlfit.fit_ml`
  is `constrained_fit.py` without module globals, extended to any fixed subset
  (fixing a variance switches to the ordinary likelihood over the free logs);
  `requirements.py`, `gains.py` and `bands.py` hold the requirement machinery;
  `start.auto_start` replaces stored starts. `fit_ml` from the stored `constr02`
  start reproduces `params_n3_constr_fromconstr02` (ll 22690.986, both).
- Subtide gap grid: gaps between consecutive fitted bands' line sets plus
  everything above the last band, to 0.1 cpd below Nyquist. Identical to the
  Freeport grid for 6 bands; with 8 bands it no longer includes the D7/D8 line
  windows (19 points, never binding).
- Sacrificial bands default to the highest two fitted. The Freeport 8-cycle runs
  had D5-D8 sacrificial (`sacrificial="D5-D8"` reproduces them).
- Trend prior (2026-11): large-variance approximation to a diffuse trend; see
  "Initial conditions (diffuse trend prior)" below.
- `fastll.scha_signal_moments` / `SCHAFit.signal_moments`: filter-predicted
  (one-sided: inside a gap, the real-time forecast) and smoothed signal mean and
  variance (Durbin-Koopman r and N recursions); matches `ffbs` to 1e-6.

### Initial conditions (diffuse trend prior)

The state is x = [trend (mu_m .. mu_1), cycles]. The filter starts from a prior
on the pre-sample state x_0 ~ N(a0, P0), block diagonal:

- **Cycles:** stationary. Each cycle block of P0 is its unconditional covariance
  (Trimbur closed form, `uncond.cycle_unconditional_cov`), mean zero. This is
  exact; nothing is approximated for the cycles.
- **Trend:** non-stationary (an m-fold integrated random walk), so it has no
  stationary distribution. Its prior should be "diffuse": no information about the
  initial level and slope.

**What scha does (large-variance approximation).**

1. The data are centred: y_c = y - mean(y), and all filtering is on y_c. This is
   the same as giving the level a prior mean of mean(y). The mean is added back
   to the level state (`SCHAFit.states`, `signal_moments`); innovations, gains,
   phasors and the likelihood are unaffected by the shift.
2. Each trend state gets prior variance
   kappa = `fastll.DIFFUSE_FACTOR` * range(y)^2 with DIFFUSE_FACTOR = 1e3
   (`fastll.diffuse_variance`), computed once from the data, not from the
   parameters. The trend's initial deviation from the mean is at most of order
   the range, so this prior is weak, and because kappa scales with the data
   squared, rescaling y by s rescales everything consistently.
3. The likelihood leaves out the first m observed values
   (`fastll.scha_loglik_diffuse`, n_skip = m in `_filter_sums`). Those m
   innovations have variance of order kappa and mostly measure the prior, not
   the parameters. "First m observed" means the first m non-NaN values, so
   leading gaps are handled. This is what `fit` reports as `loglik`, and what the
   ordinary-likelihood objective maximises when any variance is fixed.
4. The concentrated (profiled sigma2_eps) objective uses the same construction in
   units relative to sigma2_eps: kappa_rel = `mlfit.SIGMA2_DIFFUSE_REL` = 1e6 times
   sigma2_eps, also on centred data, also dropping the first m observations. The
   prior must be proportional to sigma2_eps for the profiling to be exact; in the
   diffuse limit it agrees with item 3, and at the optimum the two values agree
   closely (Freeport: reported `loglik` 22702.04; the stored objective of the same
   fit was 22702.036).

Consequences: fitted parameters are unit-free (variances scale with s^2), the
reported loglik shifts by exactly -(n_obs - m) log s, and states and gap fills
scale with s (`tests/test_fit.py::test_units_do_not_matter`,
`tests/test_fastll.py::test_diffuse_loglik_is_scale_equivariant_and_insensitive`).

**Accuracy of the approximation.** Two errors trade off as kappa grows:

- *Approximation error*, of order 1/kappa, from the prior not being infinitely
  wide. It decays as 1/DIFFUSE_FACTOR.
- *Round-off* in the first few covariance updates, P - K F K', which subtracts
  numbers of size kappa to leave numbers of size sigma2_eps. About
  log10(kappa / sigma2_eps) of the 16 significant digits are lost there; they are
  recovered once m observations have pinned the trend down.

Measured on a synthetic SCHA(2,3) series (level 500, 1500 samples at 15 min, a
30 h gap), centred, varying DIFFUSE_FACTOR:

| factor | kappa / sigma2_eps | loglik | scale-equivariance error |
|---|---|---|---|
| 1e0 | 3e4 | 387.70547 | 6e-11 |
| 1e1 | 3e5 | 387.71161 | 3e-10 |
| 1e2 | 3e6 | 387.71222 | 3e-9 |
| 1e3 | 3e7 | 387.71228 | 2e-8 |
| 1e4 | 3e8 | 387.71229 | 1e-6 |
| 1e6 | 3e10 | 387.71226 | 5e-5 |
| 1e8 | 3e12 | 387.71258 | 9e-4 |

The loglik converges like 1/factor (1e3 is within about 1e-5 of the limit) and
round-off appears above a ratio of about 1e8. Uncentred, with the previous
1e4 * (|mean| + range)^2 (ratio about 1e12), the loglik was off by 1.3e-3 and not
scale-equivariant. The smoothed signal changed by at most 2e-5 (data units)
between factors 1e1 and 1e8. A series with very small noise
relative to its range raises kappa / sigma2_eps (sigma2_eps = 1e-6 range^2 gives
1e9 at factor 1e3); expect round-off near 1e-5 in the loglik there.

The previous scheme (absolute kappa = 1e4, mean zero, loglik over all
observations) was not unit-free: for Freeport in m3/s (level about 520) the prior
was informative, the loglik was off by about 4 from the n log s shift and the
smoothed D1 by up to 4.4 m3/s at the start of the record. The design-note tables
in section 7 list that older ordinary loglik (for example 22690.99 for
`params_n3_constr_fromconstr02`); the current `loglik` for the same fit is the
former `loglik_objective` (22702.04).

**Exact alternative (not implemented).** Koopman's exact diffuse initialisation
(Koopman 1997; Durbin and Koopman, *Time Series Analysis by State Space Methods*,
ch. 5) writes the trend prior as kappa P_inf + P_star and takes kappa -> infinity
analytically. The filter carries P_inf and P_star separately through the first d
observations (d = m here, if those observations identify the trend), using
F_inf = Z P_inf Z' and the limiting gains; the diffuse loglik replaces the first d
terms by -1/2 log F_inf,t (constant in the parameters for the trend) and is
otherwise the usual sum. The smoother needs the matching exact recursions
(r0, r1, N0, N1, N2) over the same initial period. This removes both the 1/kappa
error and the round-off and drops the "first m observations" convention. It is a
contained change to the three numba kernels (`_filter_sums`, `_smooth_means`,
`_signal_moments`) and worth doing if a site shows sensitivity to
DIFFUSE_FACTOR.

### State scaling (diagonal reweighting of the state)

In a cycle of order n the lower-order states psi_1 .. psi_{n-1} have stationary
variances far larger than the top pair psi_n (ratios of Trimbur coefficients
c_ii(rho), growing like (1 - rho^2)^-(2i-1)), so the prior covariance spans many
orders of magnitude. Where this got in the way:

- **R Gibbs sampler (dlm filter and backward sampler):** it caused the kappa
  "floor" near 3e-13 and a filter-specific step offset of about +-1.6%. Scaling
  removed both (`gibbs_kappa_step_check_scaled.R`). This was the real obstacle
  to trusting larger models in R.
- **Python filters (`fastll`, `ffbs`):** tolerant. With and without scaling
  (8 cycles, n = 4, 66 states, Freeport) the likelihood, concentrated
  likelihood, finite-difference gradients and smoothed states agree to about
  1e-10 at the fitted rho (0.98-0.99) and at rho 0.995. The error then grows
  quickly: ll difference 5e-8 at rho 0.998, 2e-5 at 0.999, 6e-4 at 0.9995, while
  cond(P0) on the cycles rises from 2e13 to 6e22. Scaled, cond(P0) stays about
  2e5 throughout.
- Untested in Python: sampling and other explicit covariance factorisations
  (for example `carter_kohn_draw`), where the R experience suggests the same
  sensitivity.

So scaling is required for the R sampler, and in Python it is a safeguard
that matters near rho 0.999 and above (the crossover-raising fits reach rho
about 0.995).

The formulation uses a scaled state x = D x~ with diagonal D:

- d = 1 on the trend states and on each cycle's top pair; on cycle level i,
  d_i = sqrt(c_ii(rho_ref) / c_nn(rho_ref)) (`uncond.cycle_level_scales`,
  `state_space.state_scales`). At rho 0.98, n = 4, d reaches about 1e-5.
- System in scaled coordinates: T~ = D^-1 T D, R~ = D^-1 R, Z~ = Z D,
  P0~ = D^-1 P0 D^-1; Q and H are unchanged. Because the observed states have
  d = 1, Z~ = Z.
- Decoding: states x = D x~, covariances D C~ D (`ms * ss["D"]`;
  `fastll.scha_smoothed_states` returns decoded states).
- The transformation is exact: likelihood, gains and decoded states are
  identical. `rho_ref` can be fixed or follow the trial rho
  (`constrained_fit.py` uses the trial rho).
- Measured condition number of the cycle prior covariance, unscaled to scaled:
  3e13 -> 2e5 (Freeport, n = 4, 6 or 8 cycles), 3e12 -> 5e5 (n = 3, 8 cycles),
  1.6e13 -> 300 (tropical test at its reference rho).

Use scaling by default in new code. The R package does not have it yet (see
Pending fixes).

### Estimation methods: history

Times are UTC from the session log.

1. **Plain EM** (`scha.em_fit.scha_em`), inherited, with indexing and gain bugs
   fixed on 2026-11-21. On the tropical test each variance moved about 1% per
   iteration (EM rate about 0.99).
2. **SQUAREM**, adopted 2026-11-21 05:49 because plain EM was too slow
   (`scha_em(accelerate=True)`). Varadhan and Roland (2008), SqS3 step length,
   on the log-variances with rho fixed:
   - two EM steps theta1 = M(theta0), theta2 = M(theta1); r = theta1 - theta0,
     v = theta2 - 2 theta1 + theta0;
   - alpha = |r| / |v|, clipped to [1, step_max]; jump to
     theta0 + 2 alpha r + alpha^2 v, then one stabilising EM step;
   - if the jump lowers the likelihood by more than `ll_tol`, alpha is halved
     toward 1 (alpha = 1 is plain EM), as R SQUAREM's `objfn.inc`. This is
     needed because EM is only approximately monotone (the M-step ignores the
     dependence of P0 on kappa).
   - Stopping test revised 2026-11-21 18:51: rate-corrected step test and small
     likelihood gain per EM evaluation, both held for several cycles.
   - Used for the tropical tests and all Freeport EM fits
     (`examples/freeport/fit_em_freeport.py`: the rho grid, sigma2_zeta free or
     fixed).
3. **Constrained optimiser**, adopted 2026-11-21 23:37, immediately after the
   requirements in sections 1-5 were written down
   (`examples/freeport/constrained_fit.py`; first run `constr01` from the
   rho 0.985 EM fit). Reasons:
   - the gain requirements are inequality constraints, which EM's closed-form
     M-step cannot impose;
   - EM never updated rho; rho had been handled by a grid profile.
   SLSQP with requirements, L-BFGS-B without (`--no-constraints`); parallel
   finite-difference gradients.
4. **Refinements to the optimiser**:
   - 2026-11-22 02:46: numba likelihood (`scha.fastll`) with sigma2_eps
     concentrated out; the optimiser moved from log variances to log ratios to
     sigma2_eps plus log(1 - rho).
   - Later on 2026-11-22: diagonal preconditioning and a feasibility phase
     before SLSQP; continuation with a trust box of about 2 log units for
     crossover sweeps.

D7/D8 had not been attempted before this work because of the conditioning
problem in the R-era implementation (see State scaling); they were first fitted
with the constrained optimiser on 2026-11-22.

## 7. Findings so far (Freeport)

### Unconstrained ML

- With 6 cycles plain ML is degenerate: rho about 0.996, sigma2_zeta about 5.4e-4,
  and the trend absorbs D1 (sigma2_kappa_1 -> 0). This is the original blurring.
  SCHA(2,3) and (2,4) behave the same (ll 26463 vs 26470).
- With 8 cycles (D7, D8 added) unconstrained ML is no longer degenerate:
  sigma2_zeta 8.6e-7, a real D1, crossover at 0.67 cpd; the remaining blur is
  about 18% subtide gain at 1.41 cpd.

### Constrained fits (D5 and above sacrificial)

| model | ll | rho | sigma2_eps | subtide/D1 crossover |
|---|---|---|---|---|
| SCHA(2,4), 6 cycles | 22571.6 | 0.9829 | 2.51e-3 | 0.495 |
| SCHA(2,3), 6 cycles | 22691.0 | 0.9880 | 2.52e-3 | 0.490 |
| SCHA(2,4), 8 cycles | 25584.5 | 0.9817 | 1.55e-3 | 0.500 |
| SCHA(2,3), 8 cycles | 25699.1 | 0.9869 | 1.55e-3 | 0.494 |

- Same optimum from different starts (rho 0.985 fit, R posterior).
- The binding requirement is the subtide in the D1/D2 gap (about 1.46 cpd).
- (2,3) and (2,4) have the same number of parameters, so their likelihoods are
  directly comparable; (2,3) is ahead by about 120 at these settings.
- Requiring D5/D6 passbands cost 41 in ll and made them bind; freeing them does
  not make D6 absorb more at 6 cycles (noise takes 5-6 cpd instead).

### Gap cross-validation, PDC flow (15 min, cms; 2026-11-23)

scha-paper experiment e4 (`gap_cv.py`): one year, withheld gaps of 6-48 h, a
(rho, sigma2_zeta) grid with rho 0.975-0.997 and sigma2_zeta 1e-9 to 1e-3 cms^2.

- Plain ML with D1-D6 or D1-D4 has a runaway optimum. One band takes kappa of
  order sigma2_eps (D6 with D1-D6; D2 or D4 with D1-D4), the neighbouring bands
  collapse, and sigma2_eps falls from about 10 to 0.2 cms^2. Its loglik is about
  22000 above the sensible optimum, so it is the true ML solution. A grid
  warm-started along a path stays in it: smoother RMSE 24-1000 cms.
- The sensible optimum (kappa about 1e-8 to 1e-9 for (2,3), falling from D1 to
  D4; sigma2_eps about 10 cms^2) is local under plain ML. Refits started from a
  constrained fit stay in it.
- The runaway band is absorbing real overtide energy: residuals of the D1-D4
  constrained fit have lag-1 autocorrelation 0.89. About 85% of their variance
  lies above D4 (D5 30%, D6 23%, D7 19%, D8 7%, above D8 15%), about 1.4-1.7 cms
  per band.
- With D1-D6 and D5-D6 sacrificial, the constrained fit still runs away in D6
  (kappa 0.012, D5 near 0, D4 pushed onto its bound), and (2,3) imputes badly
  (smoother RMSE 70 cms). The requirements say nothing about how wide a
  sacrificial band may become.
- D1-D4 with every band required (`sacrificial: 0`), constrained at every grid
  cell, removes the runaway. The exception is the (2,4) column at rho 0.985,
  where a broad D4 meets its passband. Smoother RMSE is 3.5-4.6 cms and
  forecast RMSE 4.3-6.0 cms, against a noise SD of 3.2 cms and tidal persistence
  of 4.8 cms. The free constrained fits (rho 0.9927 for (2,3), 0.9901 for (2,4))
  lie within 0.1-0.5 cms of the best cells, which sit at rho 0.993-0.997. (2,3)
  and (2,4) are within 0.05 cms of each other at this site.
- The same D6 runaway appears in the (2,3) constrained D1-D6 fits at ANH, BIR,
  MSD and PDC, but not at MRZ or MIR. It follows the site's high-frequency
  content, not whether the variable is flow or elevation.

### Subtide/D1 crossover (6 cycles; ll, best run per value)

| required crossover | (2,3) | (2,4) |
|---|---|---|
| none | 22691.0 | 22571.6 |
| 0.54 | 22572.2 | 22449.7 |
| 0.58 | 22246.7 | 22196.0 |
| 0.585 | 22146.9 | 22138.4 |
| 0.60 | 21663.0 | 21835.1 |

- Each order raises the crossover by raising rho; (2,3) has to go further
  (0.988 -> 0.995 vs 0.983 -> 0.988). (2,3) is better up to about 0.585 cpd,
  (2,4) above.
- Cost at 0.58: +5.1% (2,3) / +4.3% (2,4) prediction-error variance; at 0.60:
  +12.1% / +8.5%.
- In practice the 0.4-0.83 cpd range holds only about 5 m3/s RMS at Freeport;
  moving the crossover from 0.495 to 0.58 moves about 1.8 m3/s from D1 to the
  subtide. Envelopes correlate at 0.998 or better.

### D7 and D8

- Adding D7/D8 raised ll by about 3000 for both orders (about 28% lower
  prediction-error variance) and cut sigma2_eps by 38%; the main bands barely
  changed.
- Standardized innovation PSD (white = 1) on the lines: M7 67 -> 1.5, M8 15 -> 0.5;
  M9 and M10 (13, 4 -> 21, 6.5) are now the largest. Off-line background remains
  coloured (excess at 6-12 cpd, deficit at 24-48 cpd).
- Crossover sweep with 8 cycles (ll, best of forward and backward continuation):

| required crossover | (2,3) | (2,4) |
|---|---|---|
| none | 25699.1 | 25584.5 |
| 0.52 | 25664.4 | 25551.0 |
| 0.54 | 25592.1 | 25472.4 |
| 0.56 | 25470.0 | 25318.5 |
| 0.57 | 25334.0 | 25273.0 |
| 0.58 | 25126.8 | 25155.5 |
| 0.585 | 24986.9 | 25079.8 |
| 0.59 | 24818.3 | 24988.8 |
| 0.60 | 24361.3 | 24747.6 |

  - (2,3) is ahead up to about 0.575 cpd, (2,4) above; with 6 cycles the switch
    was near 0.585.
  - Cost against each model's own baseline at 0.58: +6.6% (2,3), +4.9% (2,4)
    prediction-error variance; at 0.60: +16.0%, +9.7%.
  - (2,3) forward and backward sweeps agree exactly; (2,4) forward and backward
    differ by up to 150 in ll, so its likelihood surface has more local optima.

### Subtide post-filter (cosine-Lanczos 18 h)

- Applying the subtide gap requirement to G_sub * H, with H the response of an
  18 h cosine-Lanczos filter applied to the subtide afterwards (H = 0.61 at O1,
  0.17 at 1.45 cpd, below 1e-4 above 2.4 cpd), relaxes the binding requirement.
  The subtidal product is then the post-filtered subtide; G_sub * (1 - H) goes to
  the residual.
- 8 cycles, ll:

| required crossover | (2,3) | (2,4) |
|---|---|---|
| none | 25885.0 (crossover lands at 0.593) | 25761.5 (0.587) |
| 0.60 | 25876.3 | 25723.0 |
| 0.62 | 25745.5 | 25516.5 |
| 0.65 | 24608.4 | 24559.8 |

  - A 0.60 crossover is nearly free (about 0.1% / 0.4% prediction-error
    variance); about 0.62 is the ceiling, where the D1 window edge at 0.83 binds.
  - (2,3) is ahead of (2,4) at every crossover in this mode.
  - The (2,3) baseline is only 52 below unconstrained 8-cycle ML.
  - Raw subtide leakage at 1.45 cpd is about 4.5% (0.8% after the post-filter).
  - 6 cycles (2,3), D5/D6 sacrificial, post-filter, automatic start:
    ll 22922.9 (232 above the same fit without the post-filter).

### Automatic start

- `auto_start` matches each component's model spectrum to the excess
  periodogram power in its window (noise above the last band; band k in
  cpd_k +/- M1/2; trend below the D1 window), rho = 0.985 at 15 min carried as a
  damping per hour.
- Freeport (2,3), D1-D6: reaches the stored constrained optimum
  (ll 22690.99, rho 0.98802) with no stored start, about 30 s.
- The trend level must come from a median per-bin ratio. A power sum is set by
  the lowest bins, where (2 - 2cos w)^-m is steep and the detrend has removed
  power; it started sigma2_zeta 400x low and the fit fell into the D1-swallows-
  subtide optimum (ll 21383, kappa_D1 6e-5, zeta 8e-10).

## 8. Open and deferred

- D8 is the highest band planned; M9/M10 and the coloured background above it
  stay with the noise.
- Imputation study: needs optional constraints and fixed rho and sigma2_eps
  (all available).
- Amplitude/phase error bounds from the smoother covariance (2x2 top-pair
  blocks; linearised or Monte Carlo), and whether parameter uncertainty matters.
  `scha.phase` accepts draws (..., T); a fast simulation smoother is not written
  yet. Amplitude of the smoothed mean is biased low where uncertain
  (|E z| <= E|z|), so bounds should come from draws.
- Gridded CV in Python (`scha.crossval.grid_cv`) uses ML for the inner fit; the
  R version used Gibbs. One cell should be checked against R for continuity.
- Time-step change: lambda * r, rho^r, sigma2_zeta * r^(2m-1),
  sigma2_kappa * r^(2n-1) (or keep the stationary variance); sigma2_eps depends
  on aliasing. Harmonic-aware upsampling: smoother on the fine grid with the
  missing points as NaN (the m = 2 trend smoother is a cubic smoothing spline).
  Proposed check: fit 60-min subsample, compare with withheld 15-min values.
- The choice of named lines for D2-D6 is provisional.

## 9. Durable lessons

### Conventions and units

- Phase: x_t = A cos(w t - phi) = a cos(w t) + b sin(w t), phi = atan2(b, a), t
  in hours since a fixed epoch (default 1970-01-01). The top pair gives
  z = psi - i psi* ~ A exp(i(w t - phi)) and z exp(-i w t) = a - i b
  (`scha.phase.demodulate`); tested end to end in `tests/test_phase.py`.
- Cycle frequencies inside the model are radians per sample:
  `tide_freq_radhr / samples_per_hr` in R, `F.LAMBDAS_ALL` (k * 0.0632335 at
  15 min) in Python. Mixing rad/hr and rad/sample is a 4x error at 15 min and
  has happened more than once; R filter functions take `samples_per_hr`
  explicitly.
- `scha.ffbs.kalman_filter` returns the filter gain K = P Z'/F, not the
  Durbin-Koopman T P Z'/F. State equation x_t = T x_{t-1} + R w_t, prior on the
  pre-sample x_0.
- All component spectra (trend, cycles, noise) must share one normalisation
  before forming gains. Check any new gain code against the exact state-space
  steady-state gains (`examples/freeport/filter_gain_table.py`).
- Python log-likelihood equals R `dlmLL` exactly given the same y and prior; use
  this for parity checks.

### Numerics

- Large-variance diffuse priors lose digits in the early filter updates in
  proportion to sigma2_diffuse / sigma2_eps; centre the data and keep the ratio
  below about 1e8. Measurements and the exact alternative are in "Initial
  conditions (diffuse trend prior)", section 6.
- Use state scaling (`scale_rho`) whenever rho is near 1 and always for
  sampling. Without it the prior covariance has condition number about 1e13 or
  more; the R Gibbs sampler then developed a filter-specific offset (the kappa
  "floor" near 3e-13 was this, not a property of the posterior). The Python
  filters are tolerant up to about rho 0.998 (see State scaling in section 6).
- Unconditional cycle covariance: use the Trimbur closed form; fixed-point
  iteration fails for rho >= 0.999.
- EM on these models converges slowly (rate about 0.99). A step-size stopping
  test alone stops falsely; require both a rate-corrected step test and a small
  likelihood gain per evaluation over several cycles. EM is only approximately
  monotone because the M-step ignores the dependence of P0 on kappa.
- Noise-free deterministic test signals push ML (and samplers) to the kappa
  boundary. Validation needs added noise (`test_tide_noise01.csv`, SD 0.01).
- Marginal Metropolis with sigma2_eps integrated out is a reliable referee for
  Gibbs results (it agreed with the EM optimum when Gibbs did not).

### Estimation behaviour

- Band trading (one component taking another's band) happens in ML and in MCMC
  alike; it is a model-identification issue, not a sampler issue. It motivated
  the gain requirements.
- At fixed rho, the band crossover frequencies and the variance levels are
  bijective, so crossovers are usable fitting and reporting coordinates.
- When rho changes, the cycle stationary variance tau^2 is a poor anchor for
  kappa (M2 tau^2 spans 1e-1 to 3e4 across the rho grid). The data pin the
  spectral level at an offset from the line (log10 S_D1/eps at +0.15 cpd stays
  about 3.0-3.4 for all rho); starts matched at that offset behave better.
- Large steps in a requirement (for example crossover straight to 0.6) or
  unconstrained/R-posterior starts lead to band-swap optima (D1 swallowing the
  subtide, kappa2 inflated). Use continuation from a feasible constrained fit.
- The gain requirements define a set of acceptable filters, not a fit. They
  rule out the degenerate optima that destroy a required band, such as plain-ML
  runaways where neighbouring bands collapse. They do not rule out a poor model
  that still meets the gains: a broad tidal band at low rho can pass its lines
  while absorbing overtide residual. Within the acceptable set, the likelihood
  and out-of-sample prediction do the selecting, and on gap CV they agree. This
  is the intended division of roles, not a defect (PDC gap CV, section 7).
- Bands above the last well-resolved band, required or sacrificial, should not
  be added by default. Weak, transient overtide energy is not well identified;
  a high band with no width limit can grow to absorb it plus the noise. The
  noise term is an acceptable home for it.
- Grid or continuation studies under plain ML must check which optimum each
  cell is in (loglik, sigma2_eps, kappa pattern). Warm-starting along a path
  carries a runaway through the whole grid. Start each cell from one anchor fit.
- Zero-gain "guard band" targets between tidal bands, as in the R filter-prior
  target sets, can only be met by the noise owning the gap. This is why the
  current design places no requirement between bands.

### Interpretation pitfalls

- Do not compare a trend's pseudo-spectrum with a Welch estimate of detrended
  data at low frequency, and do not read innovation notches next to near-unit-
  root components as overprediction; both are expected. An argument that the
  trend was mimicking missing overtides was built this way and rejected.
- Claims about which frequency range "decides" the likelihood need a test, not
  a narrative.

### Classical filter facts (for comparison text)

- `vtools.cosine_lanczos(ts, "40h")`: the 40 h is the sinc cutoff; the default
  half-width is m = 1.25 * period / dt (101 weights at hourly); it is applied
  with `filtfilt`, so the response is the square of the coefficient transform,
  and 2m values are lost at each end.
- Hourly 40 h: single-pass amplitude 0.5 at 40 h; squared 0.5 at 50.8 h
  (m = 50) or 48.6 h (m = 60, the 121 weights of Walters and Heston 1982).
- Time-bandwidth limit for 121 hourly weights is about 0.2 cpd of transition
  width. SCHA faces a comparable limit but places the transition where the
  record has little energy and adapts to the nonstationary tide.

## 10. Pending fixes

Fixed bugs are not listed. R items refer to `f:\projects\stochCycle`.

### Python (scha)

- `tests/test_generate_test_tide.py` uses tolerance 1e-12 against a reference
  written with 6 decimals (observed difference 5e-7).
- Packaging: `environment.yml` names `scha-env`, which does not exist; work uses
  `schism`. The CI matrix entry `3.10` is unquoted YAML and parses as 3.1.
  (`pyproject.toml` dependencies were completed in 2026-11.)
- Diagnostic scripts in `examples/freeport` (`c8_check.py`, `constr_report.py`,
  `decomp_compare.py`, `ll_smoothness.py`) still read `sys.argv` rather than
  following the CLI guide.

### R (stochCycle)

- Port state scaling into the package model build used by `sc_gibbs`; it is
  validated only in
  `example/tropical/tropical_2_4_noise01/gibbs_kappa_step_check_scaled.R`.
  Until then Gibbs has the numerical kappa floor and a filter-specific step
  offset of about +-1.6%.
- `example/freeport_mn24_cv/freeport_m2n4_cv.R` uses 6 cycles with
  `regular_filter_targets_1thru4()`; `filter_prior` now stops on the length
  mismatch. Use the 1thru6 targets. CV `filter_quality` results produced before
  the filter-prior fixes are invalid and need rerunning.
- `example/tropical/tropical_2_3/tropical_2_3.R` and
  `example/tropical/tropical_vmd/tropical_vmd.R` read `test_tide.csv` with
  `header=FALSE`, so the series starts with the string "value" (character
  column). `tropical_2_4.R` already uses `header=TRUE`.
- `tropical_2_4.R` comment says y carries SD 0.01 noise; `test_tide.csv` has
  none (the noisy series is `test_tide_noise01.csv`). The tropical
  `Latest.RData` holds y = c(0, test_tide[1:3200]); the spurious leading 0
  inflates sigma2_eps, so do not use it as a reference without dropping it.
- `R/` contains stale copies (`realistic - Copy.R`, `sc_gibbs.txt`,
  `sc_model.txt`, `sc_postprocess.txt`). The `.R` copy is collated into the
  package and redefines `realistic_tide`.
- Phase sign: `sc_gibbs` and the filter-prior samplers store
  `atan2(psi*, psi)`, the negative of the instantaneous phase of
  z = psi - i psi* (which rotates forward at the band frequency). Python
  (`scha.phase`, `scha.post`) uses atan2(-psi*, psi). Align R before comparing
  phases.
- `freeport_m2n4_cv.R` folds (gap 100, spacing 1001 samples) overlap: 11 folds
  withhold 1100 samples per 1001-sample period, so about 99 samples per period
  are predicted twice and the later fold wins. Use a spacing that is a multiple
  of the gap (`scha.crossval` default: 25 h gaps every 250 h).
