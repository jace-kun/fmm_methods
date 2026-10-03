from __future__ import annotations

import numpy as np
import pytest

from fmm_vpm.geometry import panel_geometry
from fmm_vpm.kernels import panel_velocity_influence


def _quadrature_velocity(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """High-order point-vortex quadrature of a unit-strength panel."""
    roots, weights = np.polynomial.legendre.leggauss(128)
    s = 0.5 * (roots + 1)
    weights = 0.5 * weights
    source = start[None, :] + s[:, None] * (end - start)[None, :]
    delta = point[None, :] - source
    r_sq = np.einsum("ij,ij->i", delta, delta)
    return np.sum(
        weights[:, None]
        * np.column_stack((-delta[:, 1], delta[:, 0]))
        / (2 * np.pi * r_sq[:, None]),
        axis=0,
    )


@pytest.mark.parametrize(
    "point",
    [
        np.array((0.3, 0.7)),
        np.array((-0.4, 1.3)),
        np.array((1.7, -0.8)),
    ],
)
def test_analytic_panel_kernel_matches_high_order_quadrature(point: np.ndarray) -> None:
    geometry = panel_geometry(np.array((0.0, 1.0, 0.5)), np.array((0.0, 0.0, 1.0)))
    actual = panel_velocity_influence(point, geometry)[0, 0]
    expected = _quadrature_velocity(point, geometry.starts[0], geometry.ends[0])
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_kernel_has_counter_clockwise_vortex_sign() -> None:
    geometry = panel_geometry(np.array((0.0, 1.0, 0.5)), np.array((0.0, 0.0, 1.0)))
    # At a point above a left-to-right panel, a positive vortex sheet induces
    # leftward velocity (the point-vortex counter-clockwise convention).
    u, v = panel_velocity_influence(np.array((0.5, 0.5)), geometry)[0, 0]
    assert u < 0
    assert abs(v) < 1e-12
