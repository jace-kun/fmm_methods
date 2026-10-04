"""Airfoil coordinate generation, loading, and panel geometry.

The solver uses a closed contour ordered counter-clockwise.  That convention
puts the body interior on the left of each panel and makes the stored normal
point outward.  Input contours in the opposite direction are reversed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PanelGeometry:
    """A closed polygon and its constant-strength vortex panels."""

    nodes: np.ndarray
    starts: np.ndarray
    ends: np.ndarray
    centers: np.ndarray
    tangents: np.ndarray
    outward_normals: np.ndarray
    lengths: np.ndarray
    chord: float

    @property
    def n_panels(self) -> int:
        return self.lengths.size


def _signed_area(nodes: np.ndarray) -> float:
    return float(
        0.5
        * np.sum(
            nodes[:-1, 0] * nodes[1:, 1] - nodes[1:, 0] * nodes[:-1, 1],
        ),
    )


def panel_geometry(x: np.ndarray, y: np.ndarray) -> PanelGeometry:
    """Build geometry from a closed airfoil contour.

    The first and last coordinates may coincide.  The returned contour always
    has exactly one duplicate closure node and is counter-clockwise.
    """
    nodes = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise ValueError("expected at least three 2D boundary nodes")
    if not np.isfinite(nodes).all():
        raise ValueError("airfoil coordinates must be finite")
    if np.linalg.norm(nodes[0] - nodes[-1]) > 1e-12:
        nodes = np.vstack((nodes, nodes[0]))
    if _signed_area(nodes) < 0:
        nodes = nodes[::-1].copy()

    starts, ends = nodes[:-1], nodes[1:]
    edges = ends - starts
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("airfoil contour contains a zero-length panel")
    tangents = edges / lengths[:, None]
    # The interior is left of a CCW contour, so its outward normal is right.
    outward_normals = np.column_stack((tangents[:, 1], -tangents[:, 0]))
    chord = float(nodes[:, 0].max() - nodes[:, 0].min())
    if chord <= np.finfo(float).eps:
        raise ValueError("airfoil chord must be positive")
    return PanelGeometry(
        nodes=nodes,
        starts=starts,
        ends=ends,
        centers=0.5 * (starts + ends),
        tangents=tangents,
        outward_normals=outward_normals,
        lengths=lengths,
        chord=chord,
    )


def naca4(code: str, *, n_panels: int = 160) -> PanelGeometry:
    """Return a closed, cosine-spaced NACA four-digit airfoil.

    ``n_panels`` must be even.  The contour begins at the trailing edge on the
    upper surface, travels through the leading edge, and closes on the lower
    trailing edge.  Its final orientation is normalised by :func:`panel_geometry`.
    """
    if len(code) != 4 or not code.isdigit():
        raise ValueError("NACA code must have exactly four digits, e.g. '2412'")
    if n_panels < 20 or n_panels % 2:
        raise ValueError("n_panels must be an even integer >= 20")

    m = int(code[0]) / 100
    p = int(code[1]) / 10
    thickness = int(code[2:]) / 100
    beta = np.linspace(0.0, np.pi, n_panels // 2 + 1)
    xc = 0.5 * (1.0 - np.cos(beta))

    # Closed trailing-edge coefficient (-0.1036), not the historical open-TE
    # coefficient (-0.1015), is required by the panel Kutta condition.
    yt = (
        5
        * thickness
        * (0.2969 * np.sqrt(xc) - 0.1260 * xc - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1036 * xc**4)
    )
    if p == 0:
        yc = np.zeros_like(xc)
        dyc_dx = np.zeros_like(xc)
    else:
        leading = xc < p
        yc = np.where(
            leading,
            m / p**2 * (2 * p * xc - xc**2),
            m / (1 - p) ** 2 * ((1 - 2 * p) + 2 * p * xc - xc**2),
        )
        dyc_dx = np.where(
            leading,
            2 * m / p**2 * (p - xc),
            2 * m / (1 - p) ** 2 * (p - xc),
        )
    theta = np.arctan(dyc_dx)
    xu, yu = xc - yt * np.sin(theta), yc + yt * np.cos(theta)
    xl, yl = xc + yt * np.sin(theta), yc - yt * np.cos(theta)
    # TE upper -> LE -> TE lower; omit duplicate LE and combine duplicate TE.
    x = np.concatenate((xu[::-1], xl[1:]))
    y = np.concatenate((yu[::-1], yl[1:]))
    return panel_geometry(x, y)


def load_dat(path: str | Path) -> PanelGeometry:
    """Load a standard Selig-style DAT file, accepting an optional title line."""
    path = Path(path)
    rows: list[tuple[float, float]] = []
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except ValueError:
            continue
    if len(rows) < 3:
        raise ValueError(f"{path} does not contain at least three coordinate rows")
    coords = np.asarray(rows)
    return panel_geometry(coords[:, 0], coords[:, 1])


def contains_points(points: np.ndarray, geometry: PanelGeometry) -> np.ndarray:
    """Return whether each point lies inside the closed airfoil polygon."""
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    x, y = points[:, 0, None], points[:, 1, None]
    starts, ends = geometry.starts[None, :, :], geometry.ends[None, :, :]
    y0, y1 = starts[..., 1], ends[..., 1]
    x0, x1 = starts[..., 0], ends[..., 0]
    crosses = (y0 > y) != (y1 > y)
    x_intersection = (x1 - x0) * (y - y0) / (y1 - y0 + np.finfo(float).eps) + x0
    return np.count_nonzero(crosses & (x < x_intersection), axis=1) % 2 == 1
