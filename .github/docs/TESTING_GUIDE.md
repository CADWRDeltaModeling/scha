# Testing guide

## Environment

Use the `schism` conda environment. New terminals start in `base`; activate first.

```bash
conda activate schism
pip install --no-deps -e .
pytest tests
```

## Layout

| Location | Kind | Notes |
| --- | --- | --- |
| `tests/test_*.py` | Unit and parity tests | Some compare against R output in `examples/tropical/r_reference` or `f:\projects\stochCycle` and skip if absent |
| `tests/test_fit.py`, `test_cli.py`, `test_crossval.py`, `test_phase.py`, `test_bands_gains.py` | User API | Short synthetic records, `workers=1` |
| `tests/` other scripts | Debug and experiment scripts | Not collected by pytest |
| `examples/freeport/*_check.py` | Numerical validation scripts | e.g. `fastll_check.py`, `constrained_fit.py LABEL --check` |

`@pytest.mark.slow` marks reproduction tests (Freeport auto start reaching the
stored optimum, about 30 s). Run them with `pytest -m slow`; skip with
`-m "not slow"`.

Known failing tests are listed under Pending fixes in [scha_design.md](scha_design.md).

## Conventions

- Use `tmp_path` and `monkeypatch` to isolate files and configuration.
- Mark tests needing network access as `integration`.
- Validate new numerical code against a reference implementation.

## pytest configuration

Configured in `[tool.pytest.ini_options]` of [pyproject.toml](../../pyproject.toml): strict markers, JUnit XML output, ignores `setup.py`.
