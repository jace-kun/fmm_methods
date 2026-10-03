from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from fmm_vpm.fmm import (
    evaluate_local,
    evaluate_multipole,
    multipole_to_local,
    panel_moments,
    translate_local,
    translate_multipole,
)
from fmm_vpm.geometry import PanelGeometry, naca4
from fmm_vpm.kernels import velocity


@pytest.fixture(scope="module")
def panel_sources() -> tuple[PanelGeometry, np.ndarray]:
    geometry = naca4("2412", n_panels=80)
    gamma = np.sin(np.linspace(0.0, 4.0 * np.pi, geometry.n_panels))
    return geometry, gamma


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.linalg.norm(expected))


def test_integrated_panel_p2m_converges_to_direct_velocity(
    panel_sources: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = panel_sources
    center = 0.5 + 0.0j
    targets = np.array(((2.5, -0.2), (2.8, 0.5), (3.1, -0.4)))
    direct = velocity(targets, geometry, gamma)
    errors = []
    for order in (0, 2, 4, 6, 8):
        moments = panel_moments(geometry.starts, geometry.ends, gamma, center, order)
        errors.append(_relative_error(evaluate_multipole(moments, center, targets), direct))
    assert all(later < earlier for earlier, later in pairwise(errors))
    assert errors[-1] < 1e-7


def test_m2m_matches_direct_integrated_moments(
    panel_sources: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = panel_sources
    child_center = 0.25 + 0.1j
    parent_center = 0.75 - 0.2j
    child = panel_moments(geometry.starts, geometry.ends, gamma, child_center, order=12)
    direct_parent = panel_moments(geometry.starts, geometry.ends, gamma, parent_center, order=12)
    np.testing.assert_allclose(
        translate_multipole(child, child_center, parent_center), direct_parent
    )


def test_m2l_then_l2p_converges_to_direct_velocity(
    panel_sources: tuple[PanelGeometry, np.ndarray],
) -> None:
    geometry, gamma = panel_sources
    source_center = 0.5 + 0.0j
    target_center = 2.5 + 0.1j
    targets = np.array(((2.35, 0.05), (2.6, -0.05), (2.7, 0.25)))
    direct = velocity(targets, geometry, gamma)
    errors = []
    for order in (2, 4, 6, 8):
        moments = panel_moments(geometry.starts, geometry.ends, gamma, source_center, order)
        local = multipole_to_local(moments, source_center, target_center, order)
        errors.append(_relative_error(evaluate_local(local, target_center, targets), direct))
    assert all(later < earlier for earlier, later in pairwise(errors))
    assert errors[-1] < 1e-7


def test_l2l_preserves_the_local_polynomial() -> None:
    rng = np.random.default_rng(4)
    local = rng.normal(size=9) + 1j * rng.normal(size=9)
    parent_center = -0.5 + 0.2j
    child_center = -0.3 + 0.05j
    targets = np.array(((-0.25, 0.1), (-0.35, -0.05), (-0.4, 0.2)))
    np.testing.assert_allclose(
        evaluate_local(translate_local(local, parent_center, child_center), child_center, targets),
        evaluate_local(local, parent_center, targets),
        rtol=1e-13,
        atol=1e-13,
    )
