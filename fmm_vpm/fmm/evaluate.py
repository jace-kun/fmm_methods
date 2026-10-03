"""End-to-end, panel-aware 2D Laplace FMM velocity evaluation."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from fmm_vpm.fmm.expansions import (
    evaluate_local,
    multipole_to_local,
    panel_moments,
    translate_local,
    translate_multipole,
)
from fmm_vpm.fmm.interactions import DEFAULT_THETA, well_separated
from fmm_vpm.fmm.tree import QuadTree, build_panel_tree, build_tree
from fmm_vpm.geometry import PanelGeometry
from fmm_vpm.kernels import velocity_from_segments


@dataclass(frozen=True)
class FMMDiagnostics:
    """Interaction workload and configuration for one FMM evaluation."""

    order: int
    theta: float
    source_node_count: int
    target_node_count: int
    m2l_pair_count: int
    near_leaf_pair_count: int
    near_panel_target_count: int


@dataclass(frozen=True)
class FMMResult:
    """Induced velocity and diagnostic accounting from one FMM evaluation."""

    velocity: np.ndarray
    diagnostics: FMMDiagnostics


def _as_complex(point: np.ndarray) -> complex:
    return complex(point[0], point[1])


def _upward_moments(
    tree: QuadTree,
    geometry: PanelGeometry,
    gamma: np.ndarray,
    order: int,
) -> list[np.ndarray]:
    """P2M at leaves followed by iterative M2M to the root."""
    moments = [np.zeros(order + 1, dtype=np.complex128) for _ in tree.nodes]
    for node in reversed(tree.nodes):
        center = _as_complex(node.center)
        if node.is_leaf:
            idx = node.source_indices
            moments[node.id] = panel_moments(
                geometry.starts[idx],
                geometry.ends[idx],
                gamma[idx],
                center,
                order,
            )
        else:
            for child_id in node.children:
                child = tree.nodes[child_id]
                moments[node.id] += translate_multipole(
                    moments[child_id],
                    _as_complex(child.center),
                    center,
                )
    return moments


def _partition_dual_tree(
    target_tree: QuadTree,
    source_tree: QuadTree,
    source_moments: list[np.ndarray],
    *,
    order: int,
    theta: float,
) -> tuple[list[np.ndarray], dict[int, list[int]], int]:
    """Iteratively create MAC-valid M2L and leaf-level direct interactions."""
    locals_by_node = [np.zeros(order + 1, dtype=np.complex128) for _ in target_tree.nodes]
    near_by_target_leaf: dict[int, list[int]] = defaultdict(list)
    pending = [(target_tree.root.id, source_tree.root.id)]
    m2l_pair_count = 0
    while pending:
        target_id, source_id = pending.pop()
        target = target_tree.nodes[target_id]
        source = source_tree.nodes[source_id]
        if well_separated(target, source, theta=theta):
            locals_by_node[target_id] += multipole_to_local(
                source_moments[source_id],
                _as_complex(source.center),
                _as_complex(target.center),
                order,
            )
            m2l_pair_count += 1
            continue
        if target.is_leaf and source.is_leaf:
            near_by_target_leaf[target_id].append(source_id)
            continue
        # Split the geometrically larger support whenever possible. This is
        # iterative (no recursion) and terminates at a leaf-leaf direct pair.
        split_target = not target.is_leaf and (
            source.is_leaf or target.support_radius >= source.support_radius
        )
        if split_target:
            pending.extend((child_id, source_id) for child_id in target.children)
        else:
            pending.extend((target_id, child_id) for child_id in source.children)
    return locals_by_node, near_by_target_leaf, m2l_pair_count


def _downward_locals(tree: QuadTree, locals_by_node: list[np.ndarray]) -> None:
    """L2L propagate each accumulated local field to the target leaves."""
    for node in tree.nodes:
        for child_id in node.children:
            child = tree.nodes[child_id]
            locals_by_node[child_id] += translate_local(
                locals_by_node[node.id],
                _as_complex(node.center),
                _as_complex(child.center),
            )


def evaluate_induced_velocity(
    points: np.ndarray,
    geometry: PanelGeometry,
    gamma: np.ndarray,
    *,
    order: int = 10,
    theta: float = DEFAULT_THETA,
    leaf_capacity: int = 16,
) -> FMMResult:
    """Evaluate vortex-panel-induced velocity with a panel-aware FMM.

    The returned velocity excludes freestream. Use the direct solution's
    ``alpha_deg`` and ``freestream`` to add that separately. Source support
    radii enclose the full panels, and the dual-tree pass sends a pair to M2L
    only after it passes the support-radius MAC.
    """
    points = np.asarray(points, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if points.ndim == 1:
        points = points[None, :]
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    if not np.isfinite(points).all():
        raise ValueError("points must be finite")
    if gamma.shape != (geometry.n_panels,):
        raise ValueError("gamma must have one value per panel")
    if order < 1:
        raise ValueError("order must be at least one")

    source_tree = build_panel_tree(
        geometry.starts,
        geometry.ends,
        leaf_capacity=leaf_capacity,
    )
    target_tree = build_tree(points, leaf_capacity=leaf_capacity)
    source_moments = _upward_moments(source_tree, geometry, gamma, order)
    locals_by_node, near_by_target_leaf, m2l_pair_count = _partition_dual_tree(
        target_tree,
        source_tree,
        source_moments,
        order=order,
        theta=theta,
    )
    _downward_locals(target_tree, locals_by_node)

    result = np.zeros_like(points)
    near_leaf_pair_count = 0
    near_panel_target_count = 0
    for target_leaf in target_tree.leaves:
        target_idx = target_leaf.source_indices
        target_points = points[target_idx]
        result[target_idx] += evaluate_local(
            locals_by_node[target_leaf.id],
            _as_complex(target_leaf.center),
            target_points,
        )
        near_ids = near_by_target_leaf[target_leaf.id]
        near_leaf_pair_count += len(near_ids)
        for source_id in near_ids:
            source_idx = source_tree.nodes[source_id].source_indices
            result[target_idx] += velocity_from_segments(
                target_points,
                geometry.starts[source_idx],
                geometry.ends[source_idx],
                gamma[source_idx],
            )
            near_panel_target_count += target_idx.size * source_idx.size

    return FMMResult(
        velocity=result,
        diagnostics=FMMDiagnostics(
            order=order,
            theta=theta,
            source_node_count=len(source_tree.nodes),
            target_node_count=len(target_tree.nodes),
            m2l_pair_count=m2l_pair_count,
            near_leaf_pair_count=near_leaf_pair_count,
            near_panel_target_count=near_panel_target_count,
        ),
    )
