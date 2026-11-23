# scha agent guidance

`scha` implements state-space cyclical harmonic analysis (SCHA): a smooth
trend (subtide) plus higher-order stochastic cycles at tidal harmonics plus
noise, estimated with Kalman filtering and smoothing. It is the Python
counterpart of the R package `stochCycle` (`f:\projects\stochCycle`) and the
method of Ateljevich and Wang (2023).

## Package role

- `scha` owns the SCHA model: state-space construction, likelihood, filtering
  and smoothing, parameter estimation, and gain analysis.
- It is a general time-series analysis package. Keep SCHISM-,
  BayDeltaSCHISM- and repository-specific workflow out of it.
- Use `vtools` for reusable time-series algorithms (for example
  cosine-Lanczos filtering) rather than reproducing them locally.
- Station data access belongs to `dms_datastore`; examples here ship their own
  CSV inputs.

## Design source

The design, agreed modelling decisions, findings, durable lessons and pending
fixes are in [.github/docs/scha_design.md](.github/docs/scha_design.md). Read it
before changing estimation code or proposing modelling changes. Key decisions:

- Filter requirements are steady-state gains, imposed as constraints rather
  than likelihood-weighted priors.
- Keep the original parameters (sigma2_eps, sigma2_zeta, sigma2_kappa, rho);
  transforms are internal to optimisers.
- The model is not time-invariant (gaps are a feature); no steady-state filter.
- Work in scaled state coordinates (`scale_rho`, diagonal D, decode x = D x~).
  Required for the R sampler; a safeguard in the Python filters near rho 0.999
  (see the State scaling subsection of the design doc).
- No physical-unit tolerances and no external benchmark filters as fitting
  requirements.

Record new durable lessons and pending fixes there, not in the R package.

## Dependency policy

- Assume the established scientific Python stack (numpy, scipy, pandas, numba,
  pyyaml, click, matplotlib) is available.
- Do not add SCHISM, HEC-DSS, spatial or plotting dependencies without a
  specific need.

## Coding practice

- Plan before coding.
- Keep functions single-purpose and testable.
- Do not refactor outside the scope of the requested work. Alert the user if
  broader refactoring appears warranted.
- Do not contract existing documentation.
- Document everything in `src/scha` with numpydoc (rendered by Sphinx through
  `numpydoc` and `sphinx.ext.apidoc`):
  - Every module has a summary docstring.
  - Every public function, class and method has an imperative one-line summary
    and, as applicable, `Parameters`, `Returns`/`Yields` and `Raises` (only for
    exceptions the code raises explicitly). State units and array shapes.
  - Private helpers get at least a one-line summary.
  - Write cross-references with the fully qualified path
    (`:class:`scha.fit.SCHAFit``): `scha` re-exports names, so short names are
    ambiguous in the build.
  - Add or repair the docstring in the same change as any new or changed
    interface; documentation-only changes must not alter code.
- Preserve NumPy-style documentation; repair it when interfaces change.
- User documentation is the Sphinx project in `docsrc/` (see
  `docsrc/contributing_docs.rst`); `make html` there builds it, including the API
  pages. Update the relevant page when behavior, options or outputs change.
- Prefer explicit errors such as `ValueError` over elaborate recovery from
  invalid arguments.
- Prefer failure to robustification and silent passes.
- Preserve the mathematical semantics of established algorithms. Validate new
  numerical code against a reference implementation (for example `fastll`
  against `ffbs`, closed-form gains against state-space gains, Python against
  R `dlmLL`).

## Command-line interfaces and logging

Follow [.github/docs/CLI_GUIDE.md](.github/docs/CLI_GUIDE.md).

- Command-line interfaces use `click`, with a work function providing the same
  capability programmatically.
- Configure logging at the CLI entry point with
  `scha.logging_config.configure_logging(package_name="scha", ...)` and
  `resolve_loglevel`; offer `--logdir`, `--debug`, `--quiet`.
- Library and driver loggers live under `scha.` (drivers use
  `logging.getLogger("scha.examples.<script>")`) so one configuration covers
  both.
- Progress messages go through logging. Per-iteration optimiser and sampler
  output (EM `verbose`, SLSQP callbacks, MCMC traces) may stay as `print`.

## Time-series conventions

- Cycle frequencies inside the model are radians per sample.
- Use lower-case frequency strings such as `min`, `h`, `d`, `s`.

## Testing

- Use `pytest`.
- Mark tests needing network access as `integration`.
- See [.github/docs/TESTING_GUIDE.md](.github/docs/TESTING_GUIDE.md).

## Specialized knowledge

| Topic | Where |
| --- | --- |
| Design, findings, lessons, pending fixes | [.github/docs/scha_design.md](.github/docs/scha_design.md) |
| Module map and examples | [.github/docs/PACKAGE_GUIDE.md](.github/docs/PACKAGE_GUIDE.md) |
| Key files to read first | [.github/docs/SOURCE_MAP.md](.github/docs/SOURCE_MAP.md) |
| Test layout and invocation | [.github/docs/TESTING_GUIDE.md](.github/docs/TESTING_GUIDE.md) |
| CLI and logging conventions | [.github/docs/CLI_GUIDE.md](.github/docs/CLI_GUIDE.md) |
