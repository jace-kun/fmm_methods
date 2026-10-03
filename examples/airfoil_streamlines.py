"""Plot airfoil streamlines (filled in after direct VPM lands)."""


def main() -> None:
"""Solve NACA 0012 directly and write an interactive Plotly flow-field HTML file.

Run with:
    uv run python examples/airfoil_streamlines.py
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fmm_vpm.geometry import contains_points, naca4
from fmm_vpm.vpm import solve


def main() -> None:
    geometry = naca4("0012", n_panels=160)
    solution = solve(geometry, alpha_deg=4.0)

    x = np.linspace(-0.5, 1.5, 81)
    y = np.linspace(-0.6, 0.6, 61)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    inside = contains_points(points, geometry).reshape(xx.shape)
    velocity = solution.velocity_at(points).reshape(*xx.shape, 2)
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    speed[inside] = np.nan

    figure = make_subplots(rows=1, cols=2, subplot_titles=("Direct VPM velocity", "Surface pressure"))
    figure.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=speed,
            colorscale="Viridis",
            contours={"showlabels": True},
            colorbar={"title": "|V| / V∞"},
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=geometry.nodes[:, 0],
            y=geometry.nodes[:, 1],
            fill="toself",
            fillcolor="white",
            line={"color": "black"},
            name="NACA 0012",
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=geometry.centers[:, 0],
            y=solution.cp,
            mode="lines",
            name="Cp",
        ),
        row=1,
        col=2,
    )
    figure.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
    figure.update_yaxes(autorange="reversed", title_text="Cp", row=1, col=2)
    figure.update_xaxes(title_text="x / c", row=1, col=1)
    figure.update_xaxes(title_text="x / c", row=1, col=2)
    figure.update_layout(
        title=(
            "NACA 0012 direct vortex-panel solution, α = 4° "
            f"(CL = {solution.cl_pressure:.4f})"
        ),
        template="plotly_white",
    )
    output = "naca0012_direct_vpm.html"
    figure.write_html(output, include_plotlyjs="cdn")
    print(f"Wrote {output}; CL = {solution.cl_pressure:.5f}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
