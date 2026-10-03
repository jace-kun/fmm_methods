# FMM VPM Methods

Clean **Python** rewrite of a 2D **Vortex Panel Method** with a homemade **Fast Multipole Method** for off-body field evaluation.

This is **not** a git fork of [JoshTheEngineer/Panel_Methods](https://github.com/jte0419/Panel_Methods); that project is educational lineage (README attribution). The MIT Google Drive MATLAB folder remains the read-only MATLAB reference — this repo has **no** `.m` files.

## Setup

Requires [uv](https://github.com/astral-sh/uv) and (for validation) [XFOIL 6.99](https://web.mit.edu/drela/Public/web/xfoil/) on `PATH` (on this Mac: `brew install liuyanwpuuci/aerospace/xfoil`).

```bash
uv sync --all-groups
uv run pytest
```

Headless XFOIL for the validation suite:

```bash
XFOIL_HEADLESS=1 uv run pytest
```

## Status

Skeleton only (Checkpoint 0 → Phase 1). Direct VPM + XFOIL surface gates come next; FMM after that.
