# Source map

Files worth reading before making changes.

## Documentation

| File | Contents |
| --- | --- |
| [AGENTS.md](../../AGENTS.md) | Agent rules for this repository |
| [scha_design.md](scha_design.md) | Design decisions, findings, lessons, pending fixes |
| [PACKAGE_GUIDE.md](PACKAGE_GUIDE.md) | Module map and examples |
| [TESTING_GUIDE.md](TESTING_GUIDE.md) | Test layout and invocation |
| [CLI_GUIDE.md](CLI_GUIDE.md) | CLI and logging conventions |
| `docsrc/scha_yaml_schema.rst` | Parameter YAML schema |
| `Ateljevich and Wang_2023_*.pdf` | The method paper |

## Code

| File | Contents |
| --- | --- |
| `src/scha/state_space.py` | State-space construction and prior; read first |
| `src/scha/fit.py` | User API: `fit`, `SCHAFit`, YAML schema |
| `src/scha/mlfit.py` | Constrained ML driver (library form of `constrained_fit.py`) |
| `src/scha/requirements.py`, `src/scha/gains.py` | Gain requirements and closed-form gains |
| `src/scha/fastll.py` | Fast likelihood and smoother used by all fitting scripts |
| `src/scha/em_fit.py` | EM estimation |
| `examples/freeport/constrained_fit.py` | Original constrained ML driver (Freeport-specific; results in `em_out/`) |
| `examples/freeport/filter_gain_table.py` | Fit loader for `em_out/` and exact steady-state gains |

## R reference (`f:\projects\stochCycle`)

| File | Contents |
| --- | --- |
| `R/sc_gibbs.R`, `R/sc_model.R`, `R/sc_uncond_cov.R` | Gibbs sampler and model |
| `R/sc_filter_response.R` | Gains and filter-prior targets |
| `R/sc_postprocess.R` | `sc_params_from_gibbs` |
