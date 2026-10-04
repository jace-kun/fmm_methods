"""Compare the end-to-end panel FMM field with the frozen direct evaluator.

Run with:
    uv run python examples/fmm_vs_direct.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fmm_vpm.fmm import evaluate_induced_velocity
from fmm_vpm.geometry import contains_points, naca4
from fmm_vpm.kernels import velocity
from fmm_vpm.vpm import solve


def main() -> None:
    geometry = naca4("2412", n_panels=160)
    solution = solve(geometry, alpha_deg=4.0)
    x = np.linspace(-0.75, 1.75, 61)
    # Avoid panel-endpoint singularities in the ideal inviscid kernel.
    y = np.linspace(-0.7, 0.7, 49) + 1e-8
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = contains_points(points, geometry)
    exterior = points[~inside]

    direct = velocity(exterior, geometry, solution.gamma)
    fmm = evaluate_induced_velocity(exterior, geometry, solution.gamma, order=10, theta=0.35)
    error = np.linalg.norm(fmm.velocity - direct, axis=1)
    speed = np.linalg.norm(direct, axis=1)
    relative_l2 = np.linalg.norm(fmm.velocity - direct) / np.linalg.norm(direct)

    error_grid = np.full(points.shape[0], np.nan)
    speed_grid = np.full(points.shape[0], np.nan)
    error_grid[~inside] = error
    speed_grid[~inside] = speed
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Direct induced velocity magnitude", "FMM absolute velocity error"),
    )
    for column, values, scale, title in (
        (1, speed_grid, "Viridis", "|Vind|"),
        (2, error_grid, "Magma", "|VFMM - Vdirect|"),
    ):
        figure.add_trace(
            go.Contour(
                x=x,
                y=y,
                z=values.reshape(xx.shape),
                colorscale=scale,
                colorbar={"title": title, "x": 0.45 if column == 1 else 1.0},
            ),
            row=1,
            col=column,
        )
        figure.add_trace(
            go.Scatter(
                x=geometry.nodes[:, 0],
                y=geometry.nodes[:, 1],
                fill="toself",
                fillcolor="white",
                line={"color": "black"},
                showlegend=False,
            ),
            row=1,
            col=column,
        )
        figure.update_yaxes(scaleanchor=f"x{column}", scaleratio=1, row=1, col=column)

    figure.update_layout(
        title=(
            f"NACA 2412 panel FMM vs direct, p = 10, theta = 0.35 (relative L2 = {relative_l2:.2e})"
        ),
        template="plotly_white",
    )
    output = "naca2412_fmm_vs_direct.html"
    figure.write_html(output, include_plotlyjs="cdn")
    print(f"Wrote {output}; relative L2 error = {relative_l2:.3e}")
    print(f"M2L pairs = {fmm.diagnostics.m2l_pair_count}")
    print(f"Near panel-target work = {fmm.diagnostics.near_panel_target_count}")


if __name__ == "__main__":
    main()
