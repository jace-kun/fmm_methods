"""Complex 2D Laplace-FMM operators for constant-strength vortex panels.

Convention
----------
For a positive (counter-clockwise) vortex sheet, complex velocity is
``w = u - i v`` and a panel contributes

``w(z) = -i/(2*pi) * integral(gamma ds / (z - z_source))``.

Moments contain the physical sheet integral, not the ``-i/(2*pi)`` prefactor.
This module deliberately has no tree traversal: every operator is independently
testable against the direct panel kernel before Phase 3 composes them.
"""

from __future__ import annotations

from math import comb

import numpy as np

VORTEX_FACTOR = -1j / (2.0 * np.pi)


def panel_moments(
    starts: np.ndarray,
    ends: np.ndarray,
    gamma: np.ndarray,
    center: complex,
    order: int,
) -> np.ndarray:
    """Integrated P2M moments for constant-strength straight vortex panels.

    ``starts`` and ``ends`` have shape ``(n, 2)`` and ``gamma`` is a
    constant vortex-sheet strength per unit arclength.  Moment ``n`` is
    ``integral gamma * (z_source - center)**n ds`` and is exact for every
    non-negative integer order.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    starts = np.asarray(starts, dtype=float)
    ends = np.asarray(ends, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    if starts.shape != ends.shape or starts.ndim != 2 or starts.shape[1] != 2:
        raise ValueError("starts and ends must have shape (n, 2)")
    if gamma.shape != (starts.shape[0],):
        raise ValueError("gamma must have one value per panel")

    a = starts[:, 0] + 1j * starts[:, 1] - center
    b = ends[:, 0] + 1j * ends[:, 1] - center
    lengths = np.abs(b - a)
    if np.any(lengths == 0):
        raise ValueError("panels must have nonzero length")
    direction = b - a
    moments = np.empty(order + 1, dtype=np.complex128)
    for n in range(order + 1):
        # integral_0^L (a + s/L * (b-a))^n ds
        integral = lengths * (b ** (n + 1) - a ** (n + 1)) / ((n + 1) * direction)
        moments[n] = np.dot(gamma, integral)
    return moments


def translate_multipole(
    moments: np.ndarray, old_center: complex, new_center: complex
) -> np.ndarray:
    """M2M: translate a multipole expansion from ``old_center`` to ``new_center``."""
    moments = np.asarray(moments, dtype=np.complex128)
    order = moments.size - 1
    shift = old_center - new_center
    translated = np.zeros_like(moments)
    for n in range(order + 1):
        translated[n] = sum(comb(n, k) * shift ** (n - k) * moments[k] for k in range(n + 1))
    return translated


def evaluate_multipole(moments: np.ndarray, center: complex, targets: np.ndarray) -> np.ndarray:
    """M2P: evaluate a velocity multipole at targets of shape ``(n, 2)``."""
    moments = np.asarray(moments, dtype=np.complex128)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (n, 2)")
    z = targets[:, 0] + 1j * targets[:, 1] - center
    if np.any(z == 0):
        raise ValueError("target must not equal multipole center")
    powers = np.arange(1, moments.size + 1)
    w = VORTEX_FACTOR * np.sum(moments[None, :] / z[:, None] ** powers, axis=1)
    return np.column_stack((np.real(w), -np.imag(w)))


def multipole_to_local(
    moments: np.ndarray,
    source_center: complex,
    target_center: complex,
    order: int | None = None,
) -> np.ndarray:
    """M2L: translate source moments into a local velocity expansion."""
    moments = np.asarray(moments, dtype=np.complex128)
    if order is None:
        order = moments.size - 1
    if order < 0:
        raise ValueError("order must be non-negative")
    separation = target_center - source_center
    if separation == 0:
        raise ValueError("source and target centers must differ")
    local = np.zeros(order + 1, dtype=np.complex128)
    for n in range(order + 1):
        local[n] = VORTEX_FACTOR * sum(
            (-1) ** n * comb(k + n, n) * moments[k] / separation ** (k + n + 1)
            for k in range(moments.size)
        )
    return local


def translate_local(local: np.ndarray, old_center: complex, new_center: complex) -> np.ndarray:
    """L2L: translate a local velocity expansion to ``new_center``."""
    local = np.asarray(local, dtype=np.complex128)
    order = local.size - 1
    shift = new_center - old_center
    translated = np.zeros_like(local)
    for m in range(order + 1):
        translated[m] = sum(comb(n, m) * local[n] * shift ** (n - m) for n in range(m, order + 1))
    return translated


def evaluate_local(local: np.ndarray, center: complex, targets: np.ndarray) -> np.ndarray:
    """L2P: evaluate a local velocity expansion at targets."""
    local = np.asarray(local, dtype=np.complex128)
    targets = np.asarray(targets, dtype=float)
    if targets.ndim == 1:
        targets = targets[None, :]
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("targets must have shape (n, 2)")
    z = targets[:, 0] + 1j * targets[:, 1] - center
    powers = np.arange(local.size)
    w = np.sum(local[None, :] * z[:, None] ** powers, axis=1)
    return np.column_stack((np.real(w), -np.imag(w)))
