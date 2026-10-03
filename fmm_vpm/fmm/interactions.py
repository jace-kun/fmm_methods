"""Complete near-field and far-field partitions for adaptive quad-tree leaves."""

from __future__ import annotations

from dataclasses import dataclass

from fmm_vpm.fmm.tree import QuadNode, QuadTree


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


def leaf_interaction_lists(tree: QuadTree) -> tuple[LeafInteractionLists, ...]:
    """Return complete direct-neighbor/M2L partitions for every target leaf.

    This deliberately evaluates the complete leaf-pair relation in Phase 2.
    It is correctness-first and works for arbitrary adaptive depth. Phase 3
    may replace its all-leaf construction with an O(N) dual-tree traversal,
    but must produce an equivalent, gap-free partition.
    """
    leaves = tree.leaves
    lists: list[LeafInteractionLists] = []
    for target in leaves:
        near: list[int] = []
        far: list[int] = []
        for source in leaves:
            (near if boxes_touch(target, source) else far).append(source.id)
        lists.append(
            LeafInteractionLists(
                target_leaf_id=target.id,
                near_leaf_ids=tuple(near),
                interaction_leaf_ids=tuple(far),
            ),
        )
    assert_complete_partition(tree, tuple(lists))
    return tuple(lists)


def assert_complete_partition(
    tree: QuadTree,
    lists: tuple[LeafInteractionLists, ...],
) -> None:
    """Raise when a source leaf is missing, double-counted, or misclassified."""
    leaves = tree.leaves
    leaf_ids = {leaf.id for leaf in leaves}
    if {item.target_leaf_id for item in lists} != leaf_ids:
        raise AssertionError("interaction lists must cover every target leaf exactly once")
    by_target = {item.target_leaf_id: item for item in lists}
    for target in leaves:
        item = by_target[target.id]
        near = set(item.near_leaf_ids)
        far = set(item.interaction_leaf_ids)
        if len(near) != len(item.near_leaf_ids) or len(far) != len(item.interaction_leaf_ids):
            raise AssertionError("duplicate source leaf in interaction partition")
        if near & far or near | far != leaf_ids:
            raise AssertionError("near and far lists must be a complete disjoint partition")
        for source_id in near:
            if not boxes_touch(target, tree.nodes[source_id]):
                raise AssertionError("non-neighbor placed in near list")
        for source_id in far:
            if boxes_touch(target, tree.nodes[source_id]):
                raise AssertionError("neighbor omitted from near list")
