"""Direct constant-strength vortex panel method for a single sharp-TE airfoil.

This is intentionally the source of truth for Phase 1.  The later FMM only
accelerates its off-body field evaluation and must agree with this path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmm_vpm.geometry import PanelGeometry
from fmm_vpm.kernels import panel_velocity_influence, velocity


@dataclass(frozen=True)
class DirectVPMSolution:
    """Solved surface quantities for one inviscid angle of attack."""

    geometry: PanelGeometry
    alpha_deg: float
    freestream: float
    gamma: np.ndarray
    cp: np.ndarray
    tangential_velocity: np.ndarray
    cl_kutta_joukowski: float
    cl_pressure: float
    normal_residual_inf: float

    def velocity_at(self, points: np.ndarray) -> np.ndarray:
        """Evaluate total velocity at off-body points using the direct kernel."""
        alpha = np.deg2rad(self.alpha_deg)
        freestream = self.freestream * np.array((np.cos(alpha), np.sin(alpha)))
        return velocity(points, self.geometry, self.gamma) + freestream


def solve(
    geometry: PanelGeometry,
    *,
    alpha_deg: float,
    freestream: float = 1.0,
    svd_rcond: float = 1e-5,
) -> DirectVPMSolution:
    """Solve the impermeability equations with the sharp-TE Kutta condition.

    The closed contour must have its first/last panel adjacent to the trailing
    edge.  This is true for :func:`fmm_vpm.geometry.naca4` and for the exact
    Joukowski oracle's node order.  The final normal equation is replaced by
    ``gamma[0] + gamma[-1] = 0``.
    """
    if freestream <= 0:
        raise ValueError("freestream must be positive")

    alpha = np.deg2rad(alpha_deg)
    u_inf = freestream * np.array((np.cos(alpha), np.sin(alpha)))
    influence = panel_velocity_influence(geometry.centers, geometry)
    normal_matrix = np.einsum("ijk,ik->ij", influence, geometry.outward_normals)
    rhs = -geometry.outward_normals @ u_inf

    # Replace the final boundary equation with Kutta.  It is important to do
    # this after generating the complete direct influence matrix: no special
    # panel implementation is hidden in the linear solve.
    system = normal_matrix.copy()
    system[-1, :] = 0.0
    system[-1, 0] = 1.0
    system[-1, -1] = 1.0
    rhs[-1] = 0.0
    if not 0 < svd_rcond < 1:
        raise ValueError("svd_rcond must lie between zero and one")
    # Constant-strength vortex collocation has high-frequency null modes,
    # especially with a cusped TE and cosine-clustered nodes.  A direct solve
    # fits those modes exactly and produces non-physical alternating gamma.
    # Truncated SVD resolves only modes supported by the discretisation; the
    # returned residual is retained as a surface-quality diagnostic.
    gamma = np.linalg.lstsq(system, rhs, rcond=svd_rcond)[0]

    # The analytic self term above has an arbitrary on-panel branch.  Use the
    # Cauchy principal value plus the exterior (right-side for CCW) +gamma/2
    # tangential jump to compute Cp.
    induced = np.einsum("ijk,j->ik", influence, gamma)
    own = np.arange(geometry.n_panels)
    induced[own] -= influence[own, own] * gamma[:, None]
    surface_velocity = induced + u_inf + 0.5 * gamma[:, None] * geometry.tangents
    tangential_velocity = np.einsum("ij,ij->i", surface_velocity, geometry.tangents)
    cp = 1.0 - (tangential_velocity / freestream) ** 2

    # Pressure force coefficient: dC = -Cp * n_out * ds / chord.
    force = -np.einsum("i,ij,i->j", cp, geometry.outward_normals, geometry.lengths) / geometry.chord
    lift_direction = np.array((-np.sin(alpha), np.cos(alpha)))
    cl_pressure = float(force @ lift_direction)
    circulation = float(np.dot(gamma, geometry.lengths))
    # Our kernel takes positive gamma as counter-clockwise vorticity; with
    # freestream to +x that circulation produces negative lift.
    cl_kutta_joukowski = -2.0 * circulation / (freestream * geometry.chord)
    return DirectVPMSolution(
        geometry=geometry,
        alpha_deg=float(alpha_deg),
        freestream=float(freestream),
        gamma=gamma,
        cp=cp,
        tangential_velocity=tangential_velocity,
        cl_kutta_joukowski=cl_kutta_joukowski,
        cl_pressure=cl_pressure,
        normal_residual_inf=float(
            np.max(np.abs(normal_matrix @ gamma + geometry.outward_normals @ u_inf))
        ),
    )
