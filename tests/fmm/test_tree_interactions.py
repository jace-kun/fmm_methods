from __future__ import annotations

import numpy as np

from fmm_vpm.fmm import assert_complete_partition, boxes_touch, build_tree, leaf_interaction_lists


def test_iterative_tree_assigns_every_point_to_one_leaf() -> None:
    rng = np.random.default_rng(12)
    points = np.vstack(
        (
            rng.normal(loc=(-1.0, -1.0), scale=0.08, size=(40, 2)),
            rng.normal(loc=(1.0, 1.0), scale=0.08, size=(40, 2)),
            rng.normal(loc=(1.0, -1.0), scale=0.08, size=(40, 2)),
        ),
    )
    tree = build_tree(points, leaf_capacity=6)
    assigned = np.concatenate([leaf.source_indices for leaf in tree.leaves])
    np.testing.assert_array_equal(np.sort(assigned), np.arange(points.shape[0]))
    assert all(
        tree.nodes[child_id].depth == node.depth + 1
        for node in tree.nodes
        for child_id in node.children
    )


def test_identical_points_stop_without_a_recursive_unary_chain() -> None:
    tree = build_tree(np.zeros((20, 2)), leaf_capacity=1, max_depth=30)
    assert len(tree.nodes) == 1
    assert tree.root.source_indices.size == 20


def test_leaf_pairs_are_a_complete_near_far_partition() -> None:
    points = np.array(
        (
            (-0.9, -0.9),
            (-0.8, -0.8),
            (0.9, -0.9),
            (0.8, -0.8),
            (-0.9, 0.9),
            (-0.8, 0.8),
            (0.9, 0.9),
            (0.8, 0.8),
            (0.1, 0.1),
            (0.11, 0.11),
            (0.12, 0.12),
        ),
    )
    tree = build_tree(points, leaf_capacity=2)
    lists = leaf_interaction_lists(tree)
    assert_complete_partition(tree, lists)
    for item in lists:
        target = tree.nodes[item.target_leaf_id]
        assert all(boxes_touch(target, tree.nodes[source_id]) for source_id in item.near_leaf_ids)
        assert all(
            not boxes_touch(target, tree.nodes[source_id])
            for source_id in item.interaction_leaf_ids
        )
