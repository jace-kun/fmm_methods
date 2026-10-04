"""Exact velocity kernel for constant-strength straight vortex panels."""

from __future__ import annotations

import numpy as np

from fmm_vpm.geometry import PanelGeometry


def panel_velocity_influence(
    points: np.ndarray,
    geometry: PanelGeometry,
) -> np.ndarray:
    """Return velocity per unit vortex-sheet strength for every point/panel.

    The result has shape ``(n_points, n_panels, 2)``.  The panel integral is
    analytic; no point-vortex quadrature is used.  Values directly on a panel
    are the exterior one-sided limit and should only be used deliberately.
    """
    return panel_velocity_influence_segments(points, geometry.starts, geometry.ends)


def panel_velocity_influence_segments(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> np.ndarray:
    """Unit-sheet velocity influence for explicitly supplied panel segments."""
    points = np.asarray(points, dtype=float)
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and ends must have shape (n, 2)")

    edges = ends - starts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0):
        raise ValueError("panels must have nonzero length")
    tangents = edges / lengths[:, None]

    q0 = points[:, None, :] - starts[None, :, :]
    q1 = points[:, None, :] - ends[None, :, :]
    r0_sq = np.einsum("...i,...i->...", q0, q0)
    r1_sq = np.einsum("...i,...i->...", q1, q1)
    if np.any(r0_sq == 0) or np.any(r1_sq == 0):
        raise ValueError("velocity is singular at a panel endpoint")

    # Integral of the source-panel tangential component, before 1/(2*pi).
    source_tangential = 0.5 * np.log(r0_sq / r1_sq)
    # Integral of the source-panel normal component: signed viewing angle of
    # the panel. atan2 preserves the correct branch for points either side.
    cross = q0[..., 0] * q1[..., 1] - q0[..., 1] * q1[..., 0]
    dot = np.einsum("...i,...i->...", q0, q1)
    viewing_angle = np.arctan2(cross, dot)

    left_normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    # A unit vortex sheet is a source sheet rotated counter-clockwise:
    # (u_t, u_n) = (-source_n, source_t).
    return (
        -viewing_angle[..., None] * tangents[None, :, :]
        + source_tangential[..., None] * left_normals[None, :, :]
    ) / (2.0 * np.pi)


def velocity(
    points: np.ndarray,
    geometry: PanelGeometry,
    gamma: np.ndarray,
) -> np.ndarray:
    """Velocity induced by all constant-strength vortex panels."""
    gamma = np.asarray(gamma, dtype=float)
    if gamma.shape != (geometry.n_panels,):
        raise ValueError("gamma must have one entry per panel")
    return np.einsum("pji,j->pi", panel_velocity_influence(points, geometry), gamma)


def velocity_from_segments(
    points: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
    gamma: np.ndarray,
) -> np.ndarray:
    """Velocity from a selected set of constant-strength panel segments."""
    gamma = np.asarray(gamma, dtype=float)
    if gamma.shape != (np.asarray(starts).shape[0],):
        raise ValueError("gamma must have one entry per panel")
    return np.einsum(
        "pji,j->pi",
        panel_velocity_influence_segments(points, starts, ends),
        gamma,
    )
