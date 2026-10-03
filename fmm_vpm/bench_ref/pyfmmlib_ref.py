"""Point-vortex velocity via ``pyfmmlib`` (fmmlib2d) as an external FMM reference.

This is a benchmark and secondary-oracle tool for Phase 4 only; the solver never
imports it. fmmlib2d evaluates point sources, not constant-strength panels, so it
can only be compared against our FMM on an equivalent point-vortex discretisation
(or used as a raw speed reference), never as the truth for panel integrals.

Convention (verified in ``tests/bench/test_pyfmmlib_reference.py``): a vortex of
circulation ``gamma`` at ``z0`` induces ``u = -gamma/(2 pi) * dy / r^2`` and
``v = gamma/(2 pi) * dx / r^2``. fmmlib2d returns ``grad(charge * log r)``, so
``(u, v) = (-fld_y, fld_x) / (2 pi)``.
"""

from __future__ import annotations

import numpy as np
import pyfmmlib

# fmmlib2d precision flag -> approximate relative accuracy.
IPREC_RELATIVE_ACCURACY = {-2: 0.5, -1: 0.5e-1, 0: 0.5e-2, 1: 0.5e-3, 2: 0.5e-6, 3: 0.5e-9,
                           4: 0.5e-12, 5: 0.5e-15}  # fmt: skip


def point_vortex_velocity(
    sources: np.ndarray,
    gamma: np.ndarray,
    targets: np.ndarray,
    *,
    iprec: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Velocity ``(u, v)`` at ``targets`` induced by point vortices.

    ``sources`` and ``targets`` have shape ``(2, n)``; ``gamma`` has shape ``(n_sources,)``.
    A target that coincides with a source is singular and returns a meaningless value.
    """
    if iprec not in IPREC_RELATIVE_ACCURACY:
        raise ValueError(f"iprec must be one of {sorted(IPREC_RELATIVE_ACCURACY)}")
    sources = np.ascontiguousarray(sources, dtype=np.float64)
    targets = np.ascontiguousarray(targets, dtype=np.float64)
    n_src = sources.shape[1]
    n_tgt = targets.shape[1]
    ier, _, _, _, _, fld_targ, _ = pyfmmlib.lfmm2dparttarg(  # pyright: ignore[reportAttributeAccessIssue]
        iprec,
        sources,
        1,
        np.asarray(gamma, dtype=np.complex128),
        0,
        np.zeros(n_src, dtype=np.complex128),
        np.zeros((2, n_src)),
        0,
        0,
        0,
        n_tgt,
        targets,
        0,
        np.zeros(n_tgt, dtype=np.complex128),
        1,
        np.zeros((2, n_tgt), dtype=np.complex128),
        0,
        np.zeros((3, n_tgt), dtype=np.complex128),
    )
    if ier != 0:
        raise RuntimeError(f"fmmlib2d returned error code {ier}")
    return -fld_targ[1].real / (2.0 * np.pi), fld_targ[0].real / (2.0 * np.pi)
