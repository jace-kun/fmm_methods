"""Exact potential flow past a Joukowski airfoil (Kutta condition).

Closed-form oracle for validating panel solvers, including XFOIL itself. The
airfoil is the image of an offset circle under z = zeta + 1/zeta, so surface
speed and circulation are analytic. Geometry is returned rotated/scaled so the
chord line (trailing edge to the farthest surface point) lies on the x-axis
with unit chord, and the angle of attack is measured from that chord line.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class JoukowskiSolution:
    """Nodes run TE (upper) -> LE -> TE (lower); first and last node coincide."""

    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray
    cl: float
    alpha_deg: float


@dataclass(frozen=True)
class JoukowskiPanelValues:
    """Exact values at midpoint parameters of a uniformly sampled contour."""

    x: np.ndarray
    y: np.ndarray
    cp: np.ndarray


@dataclass(frozen=True)
class _Parameters:
    zeta0: complex
    radius: float
    theta_te: float
    z_le: complex
    z_te: complex
    chord: float
    phi: float
    alpha_z: float
    gamma: float


def _parameters(alpha_deg: float, eps: float, delta: float) -> _Parameters:
    zeta0 = complex(-eps, delta)
    radius = abs(1.0 - zeta0)
    zeta_te = 1.0 + 0.0j
    theta_te = np.angle(zeta_te - zeta0)

    def surface(theta: np.ndarray) -> np.ndarray:
        return zeta0 + radius * np.exp(1j * theta)

    dense = surface(theta_te + np.linspace(0.0, 2.0 * np.pi, 20001))
    z_dense = dense + 1.0 / dense
    z_te = 2.0 + 0.0j
    z_le = z_dense[np.argmax(np.abs(z_dense - z_te))]
    chord_vec = z_te - z_le
    chord = abs(chord_vec)
    phi = np.angle(chord_vec)
    alpha_z = np.deg2rad(alpha_deg) + phi

    zp_a = zeta_te - zeta0
    kutta = -zp_a * (np.exp(-1j * alpha_z) - radius**2 * np.exp(1j * alpha_z) / zp_a**2)
    gamma = 2.0 * np.pi * (kutta / 1j)
    if abs(gamma.imag) > 1e-12:
        raise ArithmeticError("Kutta circulation is not real; check parameters")
    return _Parameters(zeta0, radius, theta_te, z_le, z_te, chord, phi, alpha_z, gamma.real)


def _surface_zeta(theta: np.ndarray, parameters: _Parameters) -> np.ndarray:
    return parameters.zeta0 + parameters.radius * np.exp(1j * theta)


def _normalise(z: np.ndarray, parameters: _Parameters) -> np.ndarray:
    return (z - parameters.z_le) * np.exp(-1j * parameters.phi) / parameters.chord


def _complex_velocity(zeta: np.ndarray, parameters: _Parameters) -> np.ndarray:
    """Complex velocity ``u - i*v`` in chord-aligned, unit-chord coordinates."""
    zp = zeta - parameters.zeta0
    dwd_zeta = (
        np.exp(-1j * parameters.alpha_z)
        - parameters.radius**2 * np.exp(1j * parameters.alpha_z) / zp**2
        + 1j * parameters.gamma / (2.0 * np.pi * zp)
    )
    dwd_z = dwd_zeta / (1.0 - 1.0 / zeta**2)
    # z_normalised = exp(-i*phi) * (z - z_le) / chord.  Complex velocity
    # rotates by the conjugate factor, so far-field velocity becomes
    # exp(-i * alpha_deg), as required by the public coordinates.
    return np.exp(1j * parameters.phi) * dwd_z


def joukowski_solution(
    *,
    alpha_deg: float,
    eps: float = 0.08,
    delta: float = 0.03,
    n_nodes: int = 240,
) -> JoukowskiSolution:
    """Exact Cp at the surface nodes and exact CL for unit freestream.

    ``eps`` offsets the circle centre along -x (thickness), ``delta`` along +y
    (camber). The circle passes through zeta = 1, which maps to the cusped TE.
    """
    if n_nodes < 16:
        raise ValueError("n_nodes must be >= 16")

    parameters = _parameters(alpha_deg, eps, delta)

    t = np.linspace(0.0, 2.0 * np.pi, n_nodes)
    theta = parameters.theta_te + t
    zeta = _surface_zeta(theta, parameters)
    z = zeta + 1.0 / zeta
    z_norm = _normalise(z, parameters)

    # TE is a removable 0/0 singularity of the speed formula: take the limit.
    v = np.empty(n_nodes)
    interior = slice(1, n_nodes - 1)
    v[interior] = np.abs(_complex_velocity(zeta[interior], parameters))
    h = 1e-6
    zeta_te = _surface_zeta(
        np.array((parameters.theta_te + h, parameters.theta_te + 2.0 * np.pi - h)),
        parameters,
    )
    v_te = float(np.abs(_complex_velocity(zeta_te, parameters)).mean())
    v[0] = v[-1] = v_te
    z_norm[0] = z_norm[-1] = _normalise(np.array(parameters.z_te), parameters)

    return JoukowskiSolution(
        x=z_norm.real.copy(),
        y=z_norm.imag.copy(),
        cp=1.0 - v**2,
        cl=2.0 * parameters.gamma / parameters.chord,
        alpha_deg=float(alpha_deg),
    )


def joukowski_surface_panel_values(
    *,
    alpha_deg: float,
    eps: float = 0.08,
    delta: float = 0.03,
    n_panels: int = 160,
) -> JoukowskiPanelValues:
    """Exact surface Cp at parameter midpoints for an ``n_panels`` discretisation."""
    if n_panels < 16:
        raise ValueError("n_panels must be >= 16")
    parameters = _parameters(alpha_deg, eps, delta)
    theta = parameters.theta_te + (np.arange(n_panels) + 0.5) * 2.0 * np.pi / n_panels
    zeta = _surface_zeta(theta, parameters)
    z_norm = _normalise(zeta + 1.0 / zeta, parameters)
    velocity = _complex_velocity(zeta, parameters)
    return JoukowskiPanelValues(
        x=z_norm.real,
        y=z_norm.imag,
        cp=1.0 - np.abs(velocity) ** 2,
    )


def joukowski_velocity(
    points: np.ndarray,
    *,
    alpha_deg: float,
    eps: float = 0.08,
    delta: float = 0.03,
) -> np.ndarray:
    """Exact velocity at exterior points in unit-chord, chord-aligned coordinates."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")

    parameters = _parameters(alpha_deg, eps, delta)
    z = (points[:, 0] + 1j * points[:, 1]) * parameters.chord * np.exp(
        1j * parameters.phi
    ) + parameters.z_le
    root = np.sqrt(z**2 - 4.0)
    zeta_a = 0.5 * (z + root)
    zeta_b = 0.5 * (z - root)
    choose_a = np.abs(zeta_a - parameters.zeta0) > np.abs(zeta_b - parameters.zeta0)
    zeta = np.where(choose_a, zeta_a, zeta_b)
    if np.any(np.abs(zeta - parameters.zeta0) <= parameters.radius * (1.0 + 1e-10)):
        raise ValueError("points must be strictly outside the Joukowski airfoil")
    velocity = _complex_velocity(zeta, parameters)
    return np.column_stack((velocity.real, -velocity.imag))
