"""Complete near-field and far-field partitions for adaptive quad-tree leaves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fmm_vpm.fmm.tree import QuadNode, QuadTree

DEFAULT_THETA = 0.5
DEFAULT_ORDERS = (4, 6, 8, 10, 12)


@dataclass(frozen=True)
class LeafInteractionLists:
    """Every source leaf is exactly one of a target leaf's near/far entries."""

    target_leaf_id: int
    near_leaf_ids: tuple[int, ...]
    interaction_leaf_ids: tuple[int, ...]


def boxes_touch(a: QuadNode, b: QuadNode) -> bool:
    """Whether closed square boxes touch or overlap (hence need direct P2P)."""
    return bool(
        abs(a.center[0] - b.center[0]) <= a.half_width + b.half_width
        and abs(a.center[1] - b.center[1]) <= a.half_width + b.half_width
    )


def well_separated(target: QuadNode, source: QuadNode, *, theta: float = DEFAULT_THETA) -> bool:
    """Whether a pair satisfies the conservative M2L multipole criterion.

    Each node's support radius contains all assigned source geometry (full
    panels for a panel tree), rather than just its midpoint locations.
    """
    if not 0 < theta < 1:
        raise ValueError("theta must lie between zero and one")
    distance = float(np.linalg.norm(target.center - source.center))
    return distance > 0 and (target.support_radius + source.support_radius) / distance <= theta


def leaf_interaction_lists(
    target_tree: QuadTree,
    source_tree: QuadTree | None = None,
    *,
    theta: float = DEFAULT_THETA,
) -> tuple[LeafInteractionLists, ...]:
    """Return complete direct-neighbor/M2L partitions for every target leaf.

    A far pair must pass :func:`well_separated`; all other pairs are direct
    near interactions. This deliberately enumerates complete leaf pairs in
    Phase 2.5 for correctness and works for arbitrary adaptive depth. Phase 3
    may replace it with a linear-ish dual-tree traversal only if it produces
    an equivalent, gap-free MAC partition.
    """
    if source_tree is None:
        source_tree = target_tree
    target_leaves = target_tree.leaves
    source_leaves = source_tree.leaves
    lists: list[LeafInteractionLists] = []
    for target in target_leaves:
        near: list[int] = []
        far: list[int] = []
        for source in source_leaves:
            (far if well_separated(target, source, theta=theta) else near).append(source.id)
        lists.append(
            LeafInteractionLists(
                target_leaf_id=target.id,
                near_leaf_ids=tuple(near),
                interaction_leaf_ids=tuple(far),
            ),
        )
    assert_complete_partition(target_tree, tuple(lists), source_tree=source_tree, theta=theta)
    return tuple(lists)


def assert_complete_partition(
    target_tree: QuadTree,
    lists: tuple[LeafInteractionLists, ...],
    *,
    source_tree: QuadTree | None = None,
    theta: float = DEFAULT_THETA,
) -> None:
    """Raise when a source leaf is missing, double-counted, or fails its MAC."""
    if source_tree is None:
        source_tree = target_tree
    target_leaves = target_tree.leaves
    source_leaves = source_tree.leaves
    source_ids = {leaf.id for leaf in source_leaves}
    if {item.target_leaf_id for item in lists} != {leaf.id for leaf in target_leaves}:
        raise AssertionError("interaction lists must cover every target leaf exactly once")
    by_target = {item.target_leaf_id: item for item in lists}
    for target in target_leaves:
        item = by_target[target.id]
        near = set(item.near_leaf_ids)
        far = set(item.interaction_leaf_ids)
        if len(near) != len(item.near_leaf_ids) or len(far) != len(item.interaction_leaf_ids):
            raise AssertionError("duplicate source leaf in interaction partition")
        if near & far or near | far != source_ids:
            raise AssertionError("near and far lists must be a complete disjoint partition")
        for source_id in near:
            if well_separated(target, source_tree.nodes[source_id], theta=theta):
                raise AssertionError("well-separated pair omitted from interaction list")
        for source_id in far:
            if not well_separated(target, source_tree.nodes[source_id], theta=theta):
                raise AssertionError("non-admissible pair placed in interaction list")
