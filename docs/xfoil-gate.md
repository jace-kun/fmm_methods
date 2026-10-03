# XFOIL validation gate

How XFOIL is used as an inviscid oracle, what it is trusted for, and the numbers behind the tolerances.
Tests: `tests/regression/test_xfoil_gate.py` (marker `xfoil`). Runner: `fmm_vpm/io_xfoil/runner.py`.

## Recipe (XFOIL 6.99, `liuyanwpuuci/aerospace` tap)

- `XFOIL_HEADLESS=1` is required (plain `xfoil` aborts with "Cannot open display"). The runner sets it.
- Headless XFOIL does **not** print CL. We read `DUMP` (s, x, y, signed `Ue/Vinf` = surface vorticity) and `CPWR` (x, Cp).
- CL is derived two independent ways: circulation (`CL = 2*sum(Ue*ds)/c`) and pressure integration.
- XFOIL writes `plot.ps` into its cwd, so each run uses a temp directory.
- `LOAD` + `PCOP` uses caller-supplied nodes unchanged, so a solver can be compared node-for-node on identical geometry.
- `DUMP`/`CPWR` print 5 decimals: geometry round-trips to ~5e-6 and Cp resolution is ~1e-5. This is the floor for any XFOIL comparison.
- A missing binary raises `XfoilNotFound`; tests fail, they never skip.

## What XFOIL is trusted for

XFOIL is a linear-vorticity panel method; this project uses constant-strength panels. They solve the same physics but are different discretizations, so **XFOIL is not an exact oracle and the 1e-6..1e-8 FMM-vs-direct gates do not apply to it.** To know its error floor we test XFOIL against an exact solution first: potential flow past a Joukowski airfoil (`fmm_vpm/validation/joukowski.py`, closed form, Kutta condition).

## Measured (160 nodes unless noted; Joukowski eps=0.08, delta=0.03, alpha 0/3/-4/8 deg)

| Quantity | Measured | Gate |
|---|---|---|
| XFOIL CL vs exact (circulation and pressure) | <= 3.6e-4 relative (160 nodes); <= 1e-4 at 320 | 1e-3 |
| Cp RMS vs exact, node-for-node, 2 TE nodes skipped each end | 0.003 - 0.011 (worst at 8 deg) | 0.02 |
| Cp RMS ratio when nodes go 160 -> 320 | 0.26 - 0.34 (about second order) | < 0.6 |
| Circulation CL vs pressure CL (NACA 0012/2412/4412) | <= 4.9e-4 relative | 2e-3 |
| Symmetric section, alpha = 0 | CL = 0 to 1e-12 | 1e-6 |
| Negative control: 0.25 deg AoA error | CL off by ~5%, Cp RMS well above clean | must be rejected |
| Golden drift (NACA 0012 @ 0 and 5, 4412 @ 4) | bit-identical | 1e-9 |

Cp **max** error is intentionally not gated: it lands on the leading-edge node (0.011 to 0.083 at 160 nodes) where Cp is steepest, and it is not a stable statistic across node counts.

Goldens: `tests/regression/golden/xfoil_cases.json`, with binary version and tap commit. Regenerate only deliberately: `uv run python scripts/regen_xfoil_goldens.py`.

## Gates for our own direct VPM (provisional, to be enforced when it lands)

Same nodes (`PCOP` route), same angle of attack, compared against **exact Joukowski** first, XFOIL second:

1. Convergence: CL and Cp RMS error vs exact must fall as nodes double (160 -> 320 -> 640), roughly second order.
2. vs exact at 160 nodes: CL relative error and Cp RMS no worse than an agreed multiple of XFOIL's measured values above.
3. vs XFOIL on identical nodes: report the difference; it is bounded by the sum of the two discretization errors against exact, not by a fixed constant.
4. Hold-out: NACA 0012/4412 compared to the committed goldens.

These numbers are provisional. When the direct VPM exists, measure first, then set the gates from data; do not loosen a gate to pass.

## Known limits

- Node-for-node Cp comparison on a cusped TE is ill-defined; the two nodes at each end are excluded.
- Inviscid only, Mach 0, single element.
- XFOIL comes from a single-maintainer third-party tap with patches (double precision, headless). The golden test is the tripwire for a changed binary.
- The tests need a local XFOIL, so they cannot run on a stock CI runner; CI (if added) should run `-m "not xfoil"`.
