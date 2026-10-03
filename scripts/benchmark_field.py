"""Reproducible, non-CI timing comparison for off-body field evaluation.

The panel-FMM and direct paths are compared on the same integrated vortex
panels.  ``pyfmmlib`` only supports point vortices, so its timing and error are
reported separately for a midpoint point-vortex discretisation.
"""

from __future__ import annotations

import argparse
import platform
import time
from collections.abc import Callable

import numpy as np

from fmm_vpm.bench_ref import point_vortex_velocity
from fmm_vpm.fmm import PanelFMM
from fmm_vpm.geometry import contains_points, naca4
from fmm_vpm.kernels import velocity
from fmm_vpm.vpm import solve


def _median_seconds(call: Callable[[], object], repeats: int) -> float:
    times: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def _point_vortex_direct(
    sources: np.ndarray,
    circulation: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    dx = targets[:, None, 0] - sources[None, :, 0]
    dy = targets[:, None, 1] - sources[None, :, 1]
    strength = circulation[None, :] / (2 * np.pi * (dx**2 + dy**2))
    return np.column_stack(((-strength * dy).sum(axis=1), (strength * dx).sum(axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panels", type=int, default=320)
    parser.add_argument("--targets", type=int, default=12_000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    geometry = naca4("2412", n_panels=args.panels)
    solution = solve(geometry, alpha_deg=4.0)
    rng = np.random.default_rng(20261003)
    candidates = rng.uniform((-0.8, -0.8), (1.8, 0.8), size=(args.targets * 2, 2))
    targets = candidates[~contains_points(candidates, geometry)][: args.targets]
    if targets.shape[0] != args.targets:
        raise RuntimeError("could not produce the requested number of exterior targets")

    direct = velocity(targets, geometry, solution.gamma)
    fmm_setup = _median_seconds(
        lambda: PanelFMM(geometry, solution.gamma, order=10, theta=0.35),
        args.repeats,
    )
    fmm = PanelFMM(geometry, solution.gamma, order=10, theta=0.35)
    fmm_velocity = fmm.evaluate(targets).velocity
    fmm_eval = _median_seconds(lambda: fmm.evaluate(targets), args.repeats)
    direct_time = _median_seconds(lambda: velocity(targets, geometry, solution.gamma), args.repeats)

    sources = geometry.centers
    circulation = solution.gamma * geometry.lengths
    point_direct = _point_vortex_direct(sources, circulation, targets)
    pyfmmlib_velocity = np.column_stack(
        point_vortex_velocity(sources.T, circulation, targets.T, iprec=4)
    )
    pyfmmlib_time = _median_seconds(
        lambda: point_vortex_velocity(sources.T, circulation, targets.T, iprec=4),
        args.repeats,
    )

    fmm_error = np.linalg.norm(fmm_velocity - direct) / np.linalg.norm(direct)
    pyfmmlib_error = np.linalg.norm(pyfmmlib_velocity - point_direct) / np.linalg.norm(point_direct)
    print(f"Python {platform.python_version()} on {platform.platform()}")
    print(f"NACA 2412: {geometry.n_panels} panels, {targets.shape[0]} exterior targets")
    print(f"median direct panel evaluation: {direct_time:.6f} s")
    print(f"median FMM source setup:       {fmm_setup:.6f} s")
    print(f"median FMM target evaluation:  {fmm_eval:.6f} s")
    print(f"FMM/direct panel relative L2:  {fmm_error:.3e}")
    print(f"median pyfmmlib point FMM:     {pyfmmlib_time:.6f} s")
    print(f"pyfmmlib/point-direct L2:      {pyfmmlib_error:.3e}")
    print("pyfmmlib is a point-vortex reference, not a panel-integral accuracy oracle.")


if __name__ == "__main__":
    main()
