"""End-to-end panel FMM checks against the frozen direct evaluator."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from fmm_vpm.fmm import PanelFMM, evaluate_induced_velocity
from fmm_vpm.geometry import PanelGeometry, contains_points, naca4
from fmm_vpm.kernels import velocity
from fmm_vpm.vpm import solve


@pytest.fixture(scope="module")
def solved_airfoil() -> tuple[PanelGeometry, np.ndarray]:
    geometry = naca4("2412", n_panels=160)
    return geometry, solve(geometry, alpha_deg=4.0).gamma


def _targets(geometry: PanelGeometry) -> np.ndarray:
    x, y = np.meshgrid(np.linspace(-0.75, 1.75, 13), np.linspace(-0.7, 0.7, 11))
    points = np.column_stack((x.ravel(), y.ravel()))
    return points[~contains_points(points, geometry)]


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def test_fmm_p_convergence_against_direct(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    points = _targets(geometry)
    direct = velocity(points, geometry, gamma)
    errors = [
        _relative_error(
            evaluate_induced_velocity(points, geometry, gamma, order=order, theta=0.35).velocity,
            direct,
        )
        for order in (4, 6, 8, 10, 12)
    ]
    assert all(later < earlier for earlier, later in pairwise(errors))
    assert errors[-1] < 1e-6


def test_fmm_uses_both_m2l_and_near_direct_work(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    result = evaluate_induced_velocity(_targets(geometry), geometry, gamma, order=10, theta=0.5)
    assert result.diagnostics.m2l_pair_count > 0
    assert result.diagnostics.near_leaf_pair_count > 0
    assert result.diagnostics.near_panel_target_count > 0


def test_theta_tradeoff_is_visible_without_losing_accuracy(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    points = _targets(geometry)
    direct = velocity(points, geometry, gamma)
    tight = evaluate_induced_velocity(points, geometry, gamma, order=8, theta=0.35)
    permissive = evaluate_induced_velocity(points, geometry, gamma, order=8, theta=0.5)
    assert (
        tight.diagnostics.near_panel_target_count > permissive.diagnostics.near_panel_target_count
    )
    assert _relative_error(tight.velocity, direct) < _relative_error(permissive.velocity, direct)
    assert _relative_error(tight.velocity, direct) < 1e-6


def test_fmm_is_accurate_near_the_body_with_direct_p2p(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    # Offset exterior control points by more than a local panel length, while
    # remaining in the difficult near-wall region for the FMM partition.
    points = (
        geometry.centers[::8] + 2.5 * geometry.lengths[::8, None] * geometry.outward_normals[::8]
    )
    direct = velocity(points, geometry, gamma)
    fmm = evaluate_induced_velocity(points, geometry, gamma, order=12, theta=0.35)
    assert _relative_error(fmm.velocity, direct) < 2e-6


def test_reusable_source_matches_one_shot_evaluation(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    points = _targets(geometry)
    evaluator = PanelFMM(geometry, gamma, order=10, theta=0.35)
    reusable = evaluator.evaluate(points).velocity
    one_shot = evaluate_induced_velocity(points, geometry, gamma, order=10, theta=0.35).velocity
    np.testing.assert_allclose(reusable, one_shot, rtol=0.0, atol=1e-14)


def test_fmm_handles_an_unbalanced_target_tree(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    rng = np.random.default_rng(13)
    clustered = np.column_stack(
        (1.5 + 0.01 * rng.normal(size=300), 0.4 + 0.01 * rng.normal(size=300))
    )
    outliers = np.array(((-1.5, -1.0), (2.5, 1.0), (2.0, -1.5)))
    points = np.vstack((clustered, outliers))
    direct = velocity(points, geometry, gamma)
    fmm = PanelFMM(geometry, gamma, order=12, theta=0.35).evaluate(points)
    assert _relative_error(fmm.velocity, direct) < 1e-6


def test_fmm_handles_targets_very_near_panel_endpoints(
    solved_airfoil: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = solved_airfoil
    panel_ids = np.arange(8, geometry.n_panels - 8, 16)
    points = geometry.starts[panel_ids] + 1e-6 * geometry.outward_normals[panel_ids]
    direct = velocity(points, geometry, gamma)
    fmm = PanelFMM(geometry, gamma, order=12, theta=0.35).evaluate(points)
    assert _relative_error(fmm.velocity, direct) < 2e-6
