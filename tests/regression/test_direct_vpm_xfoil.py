"""Direct VPM surface-lift cross-check against the local XFOIL oracle."""

from __future__ import annotations

import pytest

from fmm_vpm.geometry import naca4
from fmm_vpm.io_xfoil import run_inviscid
from fmm_vpm.vpm import solve

pytestmark = pytest.mark.xfoil


@pytest.mark.parametrize(
    ("naca", "alpha"),
    [
        ("0012", 0.0),
        ("0012", 4.0),
        ("0012", 8.0),
        ("2412", 0.0),
        ("4412", 4.0),
    ],
)
def test_direct_vpm_lift_agrees_with_xfoil(naca: str, alpha: float) -> None:
    """Different panel discretisations should still agree in CL to 0.015."""
    direct = solve(naca4(naca, n_panels=160), alpha_deg=alpha)
    xfoil = run_inviscid(naca=naca, alpha_deg=alpha)
    assert direct.cl_pressure == pytest.approx(xfoil.cl_pressure, abs=0.015)
    assert direct.cl_kutta_joukowski == pytest.approx(direct.cl_pressure, abs=0.007)
