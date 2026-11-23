SCHA YAML schema and R integration
=================================

This document specifies the human-friendly YAML schema for SCHA examples and lists the R artifacts we need to validate ports (R -> Python parity).

Overview
--------
- The YAML file provides inputs to construct a `SCHAParams` instance and to run deterministic/EM/Gibbs analyses.
- Key goals: be flexible about how frequencies are specified, accept user-friendly `dt` formats, and perform *strict* validation (no silent defaults for `sigma2_kappa`).

Schema (fields)
----------------
- `m` (int)
  - Trend order. Default in Python is `2`.
- `n` (int)
  - Cycle order. Default in Python is `4`.
- `freqs` (list or mapping) — *required for cyclostationary models*
  - List form: `freqs: [D1, D2, D3]` (aliases) or `freqs: [1.0, 2.0]` (numeric cycles-per-day)
  - Mapping form: `freqs: {D1: M1, INERT: 1.0123, D2: M2}` (preserves label order)
  - Aliases are resolved via `scha.tide_registry` (canonical names like `M1`,`M2`, and aliases `D1..D6` are supported).

- `rho` (float)
  - State persistence parameter (optional; default shown in examples).
- `sigma2_eps`, `sigma2_zeta` (float)
  - Observation noise and trend innovation variance.
- `sigma2_kappa` (scalar, list, or mapping) — **required when `freqs` provided**
  - Scalar or single-item list: will be replicated to number of frequencies (emits a warning).
  - List: length must equal number of frequencies (otherwise raise ValueError).
  - Mapping: keys must exactly match the declared labels in `freqs` (no missing/extra keys; otherwise raise ValueError).
- `sigma2_diffuse` (float, optional)
  - Diffuse prior variance used when building initial `C0`/`P0`.

Key validation rules
--------------------
- `sigma2_kappa` **must** be present when `freqs` are present — the loader will raise a `ValueError` otherwise.
- Mapping-style `sigma2_kappa` must exactly match `freqs` labels — we do **not** perform silent filling.
- `dt` must be inferrable from the provided CSV index; otherwise loader raises an error when `freqs` are present.

Example YAMLs
-------------
List form (aliases):

.. code-block:: yaml

   m: 2
   n: 4
   freqs: [D1: D2, D2: D2, D3: D3, D4: D4]
   rho: 0.9925
   sigma2_eps: 1e-3
   sigma2_zeta: 1e-10
   sigma2_kappa: 1.35e-13

Mapping form (labels -> alias or numeric cpd):

.. code-block:: yaml

   m: 2
   n: 4
   freqs: {D1: M1, INERT: 1.0123, D2: M2}
   sigma2_kappa: {D1: 1.0e-13, INERT: 2.0e-13, D2: 1.1e-13}

R artifacts needed for Python comparison
---------------------------------------
Please provide the following files (recommended directory: `examples/tropical/r_reference/`):

- `params.yml` — YAML dump of the parameters used in R (frequencies expressed either as aliases or numeric cycles/day; if R stores radians per hour, also include a note about units).
- `obs.csv` — observational time series with a datetime index column named `time` (ISO 8601) and a data column `y` (the series used by the R run).
- `smoothed.csv` — (optional but very useful) Kalman-smoother fitted values produced by R (columns: `time`, `y`, `fitted`). If your test used a scalar/known parameter smoothing run, this gives an immediate equality check on the smoother.
- `gibbs_traces.csv` — (optional) posterior draws for `sigma2_epsilon`, `sigma2_zeta`, `sigma2_kappa` (columns are time/iteration and parameter vectors). Useful to compare posterior summaries.
- `seed.txt` — (optional) RNG seed used in R for reproducibility.
- `diagnostics.json` or `diagnostics.csv` — (optional) small diagnostic values such as the `cal_shape_kappa` quadratic forms and any per-frequency log-determinant pieces used in the `rho` MH step; these are useful for validating the inner algebraic matches.

Small R snippets (examples)
---------------------------
- Write a YAML params file in R (example uses a YAML package):

.. code-block:: r

   library(yaml)
   parms = list(m=2,n=4,freqs=c('M2','S2'),dt='1h',sigma2_kappa=1.35e-13)
   write_yaml(parms, 'examples/tropical/r_reference/params.yml')

- Save smoothed fitted values (using your smoother output):

.. code-block:: r

   write.csv(data.frame(time=index(series), y=series, fitted=smoothed), 'examples/tropical/r_reference/smoothed.csv', row.names=FALSE)

Notes and conventions
---------------------
- Prefer `cpd` (cycles per day) or human-friendly alias names in YAML; if you export numeric frequencies from R use explicit units or document that they are `radians/hour` or `cycles/day`.
- Keep filenames descriptive and place them under `examples/tropical/r_reference/` so tests can pick them up automatically.

How we'll use your artifacts
----------------------------
- Deterministic Kalman-smoother parity: the Python `run_tropical_example.py` will read your `params.yml` + `obs.csv`, run the smoother, and we will compare `fitted` columns for equality/near-equality.
- Gibbs parity: compare posterior summaries (posterior means/modes) from R `gibbs_traces.csv` with Python `sc_gibbs()` outputs — we will compare aggregate quantities (means, quantiles) rather than per-iteration exact equality unless RNG seeds are shared.
- Algebraic diagnostics: use `diagnostics.csv` to match intermediate scalars (quadratic forms, log-dets) from R to the Python computations for `rho` MH acceptance checks.

If you'd like, I can also add a small script in `examples/tropical` that validates a provided `r_reference/` directory and runs the comparison tests automatically.
