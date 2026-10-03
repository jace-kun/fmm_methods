"""Self-checks of the exact Joukowski oracle. No XFOIL needed, so these also run in CI."""

from __future__ import annotations

import numpy as np
import pytest

from fmm_vpm.validation import joukowski_solution


@pytest.mark.parametrize("alpha", [-4.0, 0.0, 3.0, 8.0])
def test_pressure_integral_reproduces_closed_form_cl(alpha: float) -> None:
    """Integrating the exact Cp over the surface must give the Kutta-Joukowski CL."""
    sol = joukowski_solution(alpha_deg=alpha, n_nodes=4001)
    dx = np.diff(sol.x)
    dy = np.diff(sol.y)
    cp_mid = 0.5 * (sol.cp[1:] + sol.cp[:-1])
    # Nodes run TE, upper, LE, lower, TE (counter-clockwise): outward normal * ds = (dy, -dx).
    fy = -np.sum(cp_mid * (-dx))
    fx = -np.sum(cp_mid * dy)
    a = np.deg2rad(alpha)
    cl = fy * np.cos(a) - fx * np.sin(a)
    assert cl == pytest.approx(sol.cl, abs=2e-3)


def test_geometry_is_closed_with_unit_chord() -> None:
    sol = joukowski_solution(alpha_deg=2.0, n_nodes=400)
    assert (sol.x[0], sol.y[0]) == (sol.x[-1], sol.y[-1])
    assert sol.x.min() == pytest.approx(0.0, abs=1e-4)  # LE need not be a node
    assert sol.x.max() == pytest.approx(1.0, abs=1e-12)


def test_cp_never_exceeds_stagnation_value() -> None:
    sol = joukowski_solution(alpha_deg=5.0, n_nodes=800)
    assert sol.cp.max() <= 1.0 + 1e-9


def test_cl_increases_with_alpha_at_roughly_two_pi_slope() -> None:
    lo = joukowski_solution(alpha_deg=0.0)
    hi = joukowski_solution(alpha_deg=4.0)
    slope = (hi.cl - lo.cl) / np.deg2rad(4.0)
    assert 2 * np.pi < slope < 2 * np.pi * 1.2


def test_rejects_too_few_nodes() -> None:
    with pytest.raises(ValueError, match="n_nodes"):
        joukowski_solution(alpha_deg=0.0, n_nodes=8)
