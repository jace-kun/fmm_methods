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

    zeta0 = complex(-eps, delta)
    radius = abs(1.0 - zeta0)
    zeta_te = 1.0 + 0.0j
    theta_te = np.angle(zeta_te - zeta0)

    def surface(theta: np.ndarray) -> np.ndarray:
        return zeta0 + radius * np.exp(1j * theta)

    # Chord definition: trailing edge to the farthest surface point.
    dense = surface(theta_te + np.linspace(0.0, 2.0 * np.pi, 20001))
    z_dense = dense + 1.0 / dense
    z_te = 2.0 + 0.0j
    z_le = z_dense[np.argmax(np.abs(z_dense - z_te))]
    chord_vec = z_te - z_le
    chord = abs(chord_vec)
    phi = np.angle(chord_vec)

    # Flow direction in the un-rotated z-plane for the requested chord-relative AoA.
    alpha_z = np.deg2rad(alpha_deg) + phi

    zp_a = zeta_te - zeta0
    kutta = -zp_a * (np.exp(-1j * alpha_z) - radius**2 * np.exp(1j * alpha_z) / zp_a**2)
    gamma = 2.0 * np.pi * (kutta / 1j)
    if abs(gamma.imag) > 1e-12:
        raise ArithmeticError("Kutta circulation is not real; check parameters")
    gamma_r = gamma.real

    def speed(theta: np.ndarray) -> np.ndarray:
        zeta = surface(theta)
        zp = zeta - zeta0
        dw = (
            np.exp(-1j * alpha_z)
            - radius**2 * np.exp(1j * alpha_z) / zp**2
            + 1j * gamma_r / (2.0 * np.pi * zp)
        )
        return np.abs(dw) / np.abs(1.0 - 1.0 / zeta**2)

    t = np.linspace(0.0, 2.0 * np.pi, n_nodes)
    theta = theta_te + t
    zeta = surface(theta)
    z = zeta + 1.0 / zeta
    z_norm = (z - z_le) * np.exp(-1j * phi) / chord

    # TE is a removable 0/0 singularity of the speed formula: take the limit.
    v = np.empty(n_nodes)
    interior = slice(1, n_nodes - 1)
    v[interior] = speed(theta[interior])
    h = 1e-6
    v_te = 0.5 * (
        speed(np.array([theta_te + h]))[0] + speed(np.array([theta_te + 2.0 * np.pi - h]))[0]
    )
    v[0] = v[-1] = v_te
    z_norm[0] = z_norm[-1] = (z_te - z_le) * np.exp(-1j * phi) / chord

    return JoukowskiSolution(
        x=z_norm.real.copy(),
        y=z_norm.imag.copy(),
        cp=1.0 - v**2,
        cl=2.0 * gamma_r / chord,
        alpha_deg=float(alpha_deg),
    )
