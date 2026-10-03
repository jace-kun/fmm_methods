"""Acceptance tests for the Phase 1 direct VPM path."""

from __future__ import annotations

import numpy as np
import pytest

from fmm_vpm.geometry import naca4, panel_geometry
from fmm_vpm.validation import joukowski_solution
from fmm_vpm.vpm import solve


@pytest.mark.parametrize("alpha", [0.0, 3.0, -4.0, 8.0])
def test_joukowski_lift_matches_exact_oracle(alpha: float) -> None:
    exact = joukowski_solution(alpha_deg=alpha, n_nodes=161)
    solution = solve(panel_geometry(exact.x, exact.y), alpha_deg=alpha)
    assert solution.cl_pressure == pytest.approx(exact.cl, abs=7e-3)
    assert solution.cl_kutta_joukowski == pytest.approx(solution.cl_pressure, abs=7e-3)
    assert solution.normal_residual_inf < 7e-3


@pytest.mark.parametrize("alpha", [0.0, 4.0, 8.0])
def test_symmetric_naca_lift_is_antisymmetric_in_alpha(alpha: float) -> None:
    geometry = naca4("0012", n_panels=160)
    positive = solve(geometry, alpha_deg=alpha)
    negative = solve(geometry, alpha_deg=-alpha)
    assert positive.cl_pressure == pytest.approx(-negative.cl_pressure, abs=8e-4)


def test_naca0012_lift_curve_has_two_pi_slope() -> None:
    geometry = naca4("0012", n_panels=160)
    zero = solve(geometry, alpha_deg=0.0)
    four = solve(geometry, alpha_deg=4.0)
    slope = (four.cl_pressure - zero.cl_pressure) / np.deg2rad(4.0)
    assert 6.5 < slope < 7.5


def test_direct_velocity_contains_freestream_far_from_airfoil() -> None:
    solution = solve(naca4("0012", n_panels=160), alpha_deg=4.0)
    velocity = solution.velocity_at(np.array((100.0, 100.0)))
    np.testing.assert_allclose(
        velocity[0],
        np.array((np.cos(np.deg2rad(4.0)), np.sin(np.deg2rad(4.0)))),
        atol=2e-3,
    )


def test_invalid_svd_cutoff_is_rejected() -> None:
    with pytest.raises(ValueError, match="svd_rcond"):
        solve(naca4("0012"), alpha_deg=0.0, svd_rcond=0.0)
