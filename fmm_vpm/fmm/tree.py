"""Iterative quad-tree construction with stable integer node IDs."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadNode:
    """One square box in a quad-tree."""

    id: int
    center: np.ndarray
    half_width: float
    support_radius: float
    depth: int
    source_indices: np.ndarray
    children: tuple[int, ...]

    @property
    def is_leaf(self) -> bool:
        return not self.children


@dataclass(frozen=True)
class QuadTree:
    """An array-backed adaptive tree; children always have stable IDs."""

    points: np.ndarray
    support_radii: np.ndarray
    nodes: tuple[QuadNode, ...]

    @property
    def root(self) -> QuadNode:
        return self.nodes[0]

    @property
    def leaves(self) -> tuple[QuadNode, ...]:
        return tuple(node for node in self.nodes if node.is_leaf)


@dataclass
class _MutableNode:
    center: np.ndarray
    half_width: float
    support_radius: float
    depth: int
    indices: np.ndarray
    children: list[int]


def build_tree(
    points: np.ndarray,
    *,
    support_radii: np.ndarray | None = None,
    leaf_capacity: int = 16,
    max_depth: int = 30,
) -> QuadTree:
    """Build an adaptive quad-tree without recursive Python calls."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points must have shape (n, 2) with n > 0")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if support_radii is None:
        support_radii = np.zeros(points.shape[0])
    support_radii = np.asarray(support_radii, dtype=float)
    if support_radii.shape != (points.shape[0],) or np.any(support_radii < 0):
        raise ValueError("support_radii must be a non-negative value per point")
    if leaf_capacity < 1:
        raise ValueError("leaf_capacity must be positive")
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")

    low = points.min(axis=0)
    high = points.max(axis=0)
    center = 0.5 * (low + high)
    half_width = 0.5 * float(np.max(high - low))
    # A nonzero padding avoids a degenerate root and gives points on the
    # maximum edge a deterministic child assignment.
    half_width = max(half_width * (1.0 + 1e-12), 1e-12)

    def support_radius(indices: np.ndarray, node_center: np.ndarray) -> float:
        return float(
            np.max(np.linalg.norm(points[indices] - node_center, axis=1) + support_radii[indices])
        )

    mutable: list[_MutableNode] = [
        _MutableNode(
            center=center,
            half_width=half_width,
            support_radius=support_radius(np.arange(points.shape[0]), center),
            depth=0,
            indices=np.arange(points.shape[0], dtype=int),
            children=[],
        ),
    ]
    pending = deque([0])
    while pending:
        node_id = pending.popleft()
        node = mutable[node_id]
        indices = node.indices
        depth = node.depth
        if indices.size <= leaf_capacity or depth >= max_depth:
            continue
        node_center = node.center
        child_half = node.half_width / 2.0
        quadrant = (points[indices, 0] >= node_center[0]).astype(int) + 2 * (
            points[indices, 1] >= node_center[1]
        ).astype(int)
        child_ids: list[int] = []
        for quadrant_id in range(4):
            child_indices = indices[quadrant == quadrant_id]
            if child_indices.size == 0:
                continue
            offset = np.array(
                ((1.0 if quadrant_id & 1 else -1.0), (1.0 if quadrant_id & 2 else -1.0)),
            )
            child_ids.append(len(mutable))
            mutable.append(
                _MutableNode(
                    center=node_center + child_half * offset,
                    half_width=child_half,
                    support_radius=support_radius(child_indices, node_center + child_half * offset),
                    depth=depth + 1,
                    indices=child_indices,
                    children=[],
                ),
            )
        # Identical coordinates cannot be subdivided meaningfully. Preserve
        # them as an over-capacity leaf instead of creating a unary chain.
        if len(child_ids) == 1 and mutable[child_ids[0]].indices.size == indices.size:
            del mutable[child_ids[0] :]
            continue
        node.children = child_ids
        pending.extend(child_ids)

    nodes = tuple(
        QuadNode(
            id=node_id,
            center=node.center,
            half_width=node.half_width,
            support_radius=node.support_radius,
            depth=node.depth,
            source_indices=node.indices,
            children=tuple(node.children),
        )
        for node_id, node in enumerate(mutable)
    )
    return QuadTree(points=points, support_radii=support_radii, nodes=nodes)


def build_panel_tree(
    starts: np.ndarray,
    ends: np.ndarray,
    *,
    leaf_capacity: int = 16,
    max_depth: int = 30,
) -> QuadTree:
    """Build a source tree that conservatively bounds each full line panel.

    A panel is assigned by its midpoint, but its half-length is included in
    the node's support radius.  Thus no M2L decision can treat a panel as a
    point located only at its midpoint, even when the panel crosses a child
    box boundary.
    """
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and ends must have shape (n, 2)")
    centers = 0.5 * (starts + ends)
    radii = 0.5 * np.linalg.norm(ends - starts, axis=1)
    return build_tree(
        centers,
        support_radii=radii,
        leaf_capacity=leaf_capacity,
        max_depth=max_depth,
    )
