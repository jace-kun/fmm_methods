"""XFOIL validation gate (Checkpoint 1 prerequisite).

These tests establish how far XFOIL itself can be trusted as an inviscid oracle,
so later comparisons against our own solver have a known error floor. They fail
(never skip) when the binary is missing; use ``-m "not xfoil"`` to exclude them.

Tolerances come from measurements on XFOIL 6.99 (liuyanwpuuci/aerospace tap),
recorded in docs/xfoil-gate.md. Tighten them if the oracle gets better; never
loosen them just to make a failing run pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fmm_vpm.io_xfoil import XfoilNotFound, find_xfoil, run_inviscid
from fmm_vpm.validation import joukowski_solution

pytestmark = pytest.mark.xfoil

GOLDEN = Path(__file__).parent / "golden" / "xfoil_cases.json"

# XFOIL prints node coordinates and Cp with 5 decimals, so identical geometry
# only round-trips to ~1e-5.
NODE_ROUNDTRIP_TOL = 5e-5
# Joukowski TE is a cusp; skip the two nodes at each end where any panel method is ill-defined.
TE_SKIP = 2


def _interior(n: int) -> slice:
    return slice(TE_SKIP, n - TE_SKIP)


def test_binary_is_found_and_missing_binary_fails_loudly(monkeypatch: pytest.MonkeyPatch) -> None:
    assert Path(find_xfoil()).is_file()
    monkeypatch.setenv("XFOIL_BIN", "/nonexistent/xfoil")
    with pytest.raises(XfoilNotFound):
        run_inviscid(alpha_deg=0.0, naca="0012")


def test_run_leaves_no_files_in_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    run_inviscid(alpha_deg=2.0, naca="0012")
    assert list(tmp_path.iterdir()) == []


def test_symmetric_section_at_zero_alpha_has_zero_lift() -> None:
    sol = run_inviscid(alpha_deg=0.0, naca="0012")
    assert abs(sol.cl_circulation) < 1e-6
    assert abs(sol.cl_pressure) < 1e-6


@pytest.mark.parametrize(
    ("naca", "alpha"),
    [("0012", 5.0), ("0012", -3.0), ("4412", 0.0), ("4412", 4.0), ("2412", 8.0)],
)
def test_two_independent_cl_routes_agree(naca: str, alpha: float) -> None:
    """Circulation and pressure integration are independent; measured gap <= 5e-4."""
    sol = run_inviscid(alpha_deg=alpha, naca=naca)
    assert sol.cl_circulation == pytest.approx(sol.cl_pressure, rel=2e-3, abs=1e-4)


@pytest.mark.parametrize("alpha", [0.0, 3.0, -4.0, 8.0])
def test_xfoil_cl_matches_exact_joukowski(alpha: float) -> None:
    """Measured |dCL|/CL <= 3.6e-4 at 160 nodes; gate at 1e-3."""
    exact = joukowski_solution(alpha_deg=alpha, n_nodes=160)
    sol = run_inviscid(alpha_deg=alpha, coords=(exact.x, exact.y))
    assert sol.cl_circulation == pytest.approx(exact.cl, rel=1e-3)
    assert sol.cl_pressure == pytest.approx(exact.cl, rel=1e-3)


@pytest.mark.parametrize("alpha", [0.0, 3.0, -4.0, 8.0])
def test_xfoil_cp_matches_exact_joukowski_in_rms(alpha: float) -> None:
    """Node-for-node Cp. Measured rms <= 0.0105 at 160 nodes (worst at alpha=8); gate at 0.02.

    Max error is deliberately not gated: it sits on the leading-edge node where
    Cp changes steeply, and it is not a stable statistic across node counts.
    """
    exact = joukowski_solution(alpha_deg=alpha, n_nodes=160)
    sol = run_inviscid(alpha_deg=alpha, coords=(exact.x, exact.y))
    assert np.abs(sol.x - exact.x).max() < NODE_ROUNDTRIP_TOL
    assert np.abs(sol.y - exact.y).max() < NODE_ROUNDTRIP_TOL
    m = _interior(exact.x.size)
    rms = float(np.sqrt(np.mean((sol.cp[m] - exact.cp[m]) ** 2)))
    assert rms < 0.02


@pytest.mark.parametrize("alpha", [3.0, 8.0])
def test_xfoil_cp_error_converges_with_node_count(alpha: float) -> None:
    """Doubling nodes 160 -> 320 must cut Cp rms error; measured ratio 0.26-0.34, gate 0.6."""
    rms = {}
    for n in (160, 320):
        exact = joukowski_solution(alpha_deg=alpha, n_nodes=n)
        sol = run_inviscid(alpha_deg=alpha, coords=(exact.x, exact.y))
        m = _interior(n)
        rms[n] = float(np.sqrt(np.mean((sol.cp[m] - exact.cp[m]) ** 2)))
    assert rms[320] < 0.6 * rms[160]


def test_negative_control_gate_rejects_a_quarter_degree_error() -> None:
    """The tolerances must have teeth: a 0.25 deg AoA slip has to fail both gates."""
    exact = joukowski_solution(alpha_deg=3.0, n_nodes=160)
    skewed = run_inviscid(alpha_deg=3.25, coords=(exact.x, exact.y))
    assert abs(skewed.cl_circulation - exact.cl) / abs(exact.cl) > 1e-3
    m = _interior(exact.x.size)
    rms = float(np.sqrt(np.mean((skewed.cp[m] - exact.cp[m]) ** 2)))
    clean = run_inviscid(alpha_deg=3.0, coords=(exact.x, exact.y))
    clean_rms = float(np.sqrt(np.mean((clean.cp[m] - exact.cp[m]) ** 2)))
    assert rms > 1.5 * clean_rms


def test_golden_outputs_have_not_drifted() -> None:
    """XFOIL is deterministic; any change means the binary (or its tap patches) changed."""
    data = json.loads(GOLDEN.read_text())
    for case in data["cases"]:
        sol = run_inviscid(alpha_deg=case["alpha_deg"], naca=case["naca"])
        label = f"NACA {case['naca']} alpha={case['alpha_deg']}"
        assert sol.cl_circulation == pytest.approx(case["cl_circulation"], abs=1e-9), label
        assert sol.cl_pressure == pytest.approx(case["cl_pressure"], abs=1e-9), label
        np.testing.assert_allclose(sol.x, case["x"], atol=1e-9, err_msg=label)
        np.testing.assert_allclose(sol.cp, case["cp"], atol=1e-9, err_msg=label)
