"""Phase 4 benchmark reference: prove pyfmmlib builds, runs and means what we think.

pyfmmlib is a required dev dependency (``bench`` group), so a failed import fails
the suite rather than skipping it.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from fmm_vpm.bench_ref import IPREC_RELATIVE_ACCURACY, point_vortex_velocity


def _direct(src: np.ndarray, gamma: np.ndarray, tgt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dx = tgt[0][:, None] - src[0][None, :]
    dy = tgt[1][:, None] - src[1][None, :]
    r2 = dx**2 + dy**2
    return (-(gamma / (2 * np.pi)) * dy / r2).sum(axis=1), ((gamma / (2 * np.pi)) * dx / r2).sum(
        axis=1
    )


@pytest.fixture(scope="module")
def cluster() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    src = rng.random((2, 4000))
    tgt = rng.random((2, 1500)) + np.array([[1.5], [0.0]])
    return src, rng.standard_normal(4000), tgt


def _rel_err(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]).max() / np.hypot(*b).max())


def test_single_vortex_sign_and_magnitude() -> None:
    """Counter-clockwise (gamma > 0) vortex at origin: at (1, 0) the flow goes +y."""
    u, v = point_vortex_velocity(
        np.zeros((2, 1)), np.array([2 * np.pi]), np.array([[1.0, 0.0], [0.0, 2.0]]), iprec=5
    )
    np.testing.assert_allclose(u, [0.0, -0.5], atol=1e-13)
    np.testing.assert_allclose(v, [1.0, 0.0], atol=1e-13)


def test_matches_direct_sum_at_high_precision(
    cluster: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    src, gamma, tgt = cluster
    assert (
        _rel_err(point_vortex_velocity(src, gamma, tgt, iprec=5), _direct(src, gamma, tgt)) < 1e-13
    )


def test_error_decreases_with_precision_flag(
    cluster: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """The same monotone p-convergence our own FMM must show in Phase 3."""
    src, gamma, tgt = cluster
    ref = _direct(src, gamma, tgt)
    errs = [_rel_err(point_vortex_velocity(src, gamma, tgt, iprec=i), ref) for i in range(-2, 6)]
    assert all(b < a for a, b in pairwise(errs))
    for iprec, err in zip(range(-2, 6), errs, strict=True):
        assert err < 20 * IPREC_RELATIVE_ACCURACY[iprec]


def test_rejects_unknown_precision_flag(
    cluster: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    src, gamma, tgt = cluster
    with pytest.raises(ValueError, match="iprec"):
        point_vortex_velocity(src, gamma, tgt, iprec=9)
