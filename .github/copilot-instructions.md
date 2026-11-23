# scha — Workspace Instructions

Local project rules in [AGENTS.md](../AGENTS.md) take precedence. Read it first.

Organization defaults are in BayDeltaSCHISM https://raw.githubusercontent.com/CADWRDeltaModeling/BayDeltaSCHISM/refs/heads/master/AGENTS.md and apply only where AGENTS.md is silent.

Deeper material is split out of AGENTS.md:

- [docs/scha_design.md](docs/scha_design.md) — design decisions, findings, durable lessons, pending Python and R fixes
- [docs/PACKAGE_GUIDE.md](docs/PACKAGE_GUIDE.md) — module map and examples
- [docs/SOURCE_MAP.md](docs/SOURCE_MAP.md) — key files to read first
- [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) — test layout and invocation
- [docs/CLI_GUIDE.md](docs/CLI_GUIDE.md) — CLI and logging conventions (shared with dms_datastore)

# Build and Test

Work uses the `schism` conda environment (there is no `scha-env`). New terminals start in `base`, where `scha` is not installed; always activate first.

```bash
conda activate schism
pip install --no-deps -e .
pytest tests
```

R reference: `stochCycle` at `f:\projects\stochCycle`, run with `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`.

# Documentation

Document all code in `src/scha` with numpydoc: a summary docstring on every module, and on every public function, class and method an imperative one-line summary plus `Parameters`, `Returns`/`Yields` and `Raises` as applicable (state units and array shapes). Private helpers get at least a one-line summary. Cross-reference with fully qualified paths (`:class:`scha.fit.SCHAFit``), because `scha` re-exports names. Add or repair the docstring in the same change as any new or changed interface. Exception: click command functions keep a short plain docstring, because click prints it as `--help`; document the options with `help=` and the behaviour on the work function the command calls. Details are in [AGENTS.md](../AGENTS.md), under Coding practice.

User documentation is the Sphinx project in `docsrc/`; `make html` there builds it, including the API pages (see `docsrc/contributing_docs.rst`).
