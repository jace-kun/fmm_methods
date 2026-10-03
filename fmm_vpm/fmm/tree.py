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
    nodes: tuple[QuadNode, ...]

    @property
    def root(self) -> QuadNode:
        return self.nodes[0]

    @property
    def leaves(self) -> tuple[QuadNode, ...]:
        return tuple(node for node in self.nodes if node.is_leaf)


def build_tree(
    points: np.ndarray,
    *,
    leaf_capacity: int = 16,
    max_depth: int = 30,
) -> QuadTree:
    """Build an adaptive quad-tree without recursive Python calls."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError("points must have shape (n, 2) with n > 0")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
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

    mutable: list[dict[str, object]] = [
        {
            "center": center,
            "half_width": half_width,
            "depth": 0,
            "indices": np.arange(points.shape[0], dtype=int),
            "children": [],
        }
    ]
    pending = deque([0])
    while pending:
        node_id = pending.popleft()
        node = mutable[node_id]
        indices = node["indices"]
        assert isinstance(indices, np.ndarray)
        depth = node["depth"]
        assert isinstance(depth, int)
        if indices.size <= leaf_capacity or depth >= max_depth:
            continue
        node_center = node["center"]
        assert isinstance(node_center, np.ndarray)
        child_half = float(node["half_width"]) / 2.0
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
                {
                    "center": node_center + child_half * offset,
                    "half_width": child_half,
                    "depth": depth + 1,
                    "indices": child_indices,
                    "children": [],
                },
            )
        # Identical coordinates cannot be subdivided meaningfully. Preserve
        # them as an over-capacity leaf instead of creating a unary chain.
        if len(child_ids) == 1 and mutable[child_ids[0]]["indices"].size == indices.size:
            del mutable[child_ids[0] :]
            continue
        node["children"] = child_ids
        pending.extend(child_ids)

    nodes = tuple(
        QuadNode(
            id=node_id,
            center=np.asarray(node["center"], dtype=float),
            half_width=float(node["half_width"]),
            depth=int(node["depth"]),
            source_indices=np.asarray(node["indices"], dtype=int),
            children=tuple(node["children"]),  # type: ignore[arg-type]
        )
        for node_id, node in enumerate(mutable)
    )
    return QuadTree(points=points, nodes=nodes)
