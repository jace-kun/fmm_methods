# FMM VPM Methods

Clean **Python** rewrite of a 2D **Vortex Panel Method** with a homemade **Fast Multipole Method** for off-body field evaluation.

This is **not** a git fork of [JoshTheEngineer/Panel_Methods](https://github.com/jte0419/Panel_Methods); that project is educational lineage (README attribution). The MIT Google Drive MATLAB folder remains the read-only MATLAB reference — this repo has **no** `.m` files.

## Setup

Requires Python 3.13 (pinned in `.python-version`; `uv` fetches it), [uv](https://github.com/astral-sh/uv), a Fortran compiler (`brew install gcc`) so the benchmark-only `pyfmmlib` can build, and, for validation, [XFOIL 6.99](https://web.mit.edu/drela/Public/web/xfoil/) on `PATH` (on this Mac: `brew install liuyanwpuuci/aerospace/xfoil`; not in core Homebrew).

```bash
uv sync --all-groups
uv run pytest                   # full suite, includes the XFOIL gate
uv run pytest -m "not xfoil"    # without a local XFOIL binary (this is what CI runs)
uv run ruff check .             # lint (also covers import sorting)
uv run ruff format .            # formatter (replaces black + isort)
uv run pyright                  # type check (same engine as Pylance)
```

Plots use [Plotly](https://plotly.com/python/); there is no matplotlib dependency.

XFOIL tests fail (they do not skip) when the binary is missing. The runner sets `XFOIL_HEADLESS=1` itself and runs XFOIL in a temp directory. Set `XFOIL_BIN` to use a binary that is not on `PATH`.

See [docs/xfoil-gate.md](docs/xfoil-gate.md) for what XFOIL is trusted for and the measured tolerances.

## Layout

- `fmm_vpm/` — import package (geometry, vpm, kernels, fmm, `io_xfoil`, `validation`, `bench_ref`, viz)
- `tests/` — analytic (incl. exact Joukowski oracle checks), fmm, regression (XFOIL gate), bench (`pyfmmlib` reference)
- `scripts/` — airfoil fetch, XFOIL golden regeneration

## Status

XFOIL validation gate is in place (Joukowski exact oracle, runner, goldens). Direct VPM is next; FMM after that.
