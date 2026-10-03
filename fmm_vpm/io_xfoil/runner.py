"""Scripted, headless XFOIL runs for inviscid validation.

Recipe (verified against XFOIL 6.99 from the liuyanwpuuci/aerospace tap):

* ``XFOIL_HEADLESS=1`` is mandatory; without it XFOIL aborts with
  "Cannot open display". We set it ourselves so callers cannot forget.
* Headless XFOIL does not print CL. We read ``DUMP`` (s, x, y, signed Ue/Vinf =
  surface vorticity) and ``CPWR`` (x, Cp) and derive CL two independent ways:
  pressure integration and circulation (Gamma = sum of Ue ds, CL = 2 Gamma / c).
* XFOIL drops ``plot.ps`` in its cwd, so every run happens in a temp directory.
* A missing binary raises ``XfoilNotFound``. Validation must fail loudly, never
  skip silently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_TIMEOUT_S = 30.0


class XfoilError(RuntimeError):
    """XFOIL ran but did not produce usable output."""


class XfoilNotFound(XfoilError):
    """No XFOIL binary could be located."""


@dataclass(frozen=True)
class XfoilSolution:
    """Inviscid XFOIL solution at the panel nodes (TE upper -> LE -> TE lower)."""

    alpha_deg: float
    s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    ue: np.ndarray
    cp: np.ndarray
    chord: float
    cl_circulation: float
    cl_pressure: float


def find_xfoil() -> str:
    """Return the XFOIL executable path (``XFOIL_BIN`` overrides ``PATH``)."""
    override = os.environ.get("XFOIL_BIN")
    candidate = override if override else shutil.which("xfoil")
    if not candidate or not Path(candidate).is_file():
        raise XfoilNotFound(
            "XFOIL binary not found. Install with "
            "`brew install liuyanwpuuci/aerospace/xfoil` or set XFOIL_BIN."
        )
    return candidate


def _write_coords(path: Path, x: np.ndarray, y: np.ndarray) -> None:
    rows = "\n".join(f"{xi:.10f} {yi:.10f}" for xi, yi in zip(x, y, strict=True))
    path.write_text(f"airfoil\n{rows}\n")


def _derive_cl(
    alpha_deg: float, x: np.ndarray, y: np.ndarray, ue: np.ndarray, cp: np.ndarray
) -> tuple[float, float, float]:
    chord = float(x.max() - x.min())
    dx, dy = np.diff(x), np.diff(y)
    ds = np.hypot(dx, dy)
    gamma = float(np.sum(0.5 * (ue[1:] + ue[:-1]) * ds))
    cl_circulation = 2.0 * gamma / chord

    cp_mid = 0.5 * (cp[1:] + cp[:-1])
    fx = -float(np.sum(cp_mid * dy)) / chord
    fy = float(np.sum(cp_mid * dx)) / chord
    a = np.deg2rad(alpha_deg)
    cl_pressure = fy * np.cos(a) - fx * np.sin(a)
    return chord, cl_circulation, cl_pressure


def run_inviscid(
    *,
    alpha_deg: float,
    naca: str | None = None,
    coords: tuple[np.ndarray, np.ndarray] | None = None,
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> XfoilSolution:
    """Solve inviscid flow at ``alpha_deg`` for a NACA section or explicit nodes.

    With ``naca`` (e.g. ``"0012"``) XFOIL generates and panels the section (160
    nodes). With ``coords`` the nodes are used exactly as given (``PCOP``), so a
    panel solver can be compared node-for-node on identical geometry.
    """
    if (naca is None) == (coords is None):
        raise ValueError("pass exactly one of `naca` or `coords`")

    binary = find_xfoil()
    with tempfile.TemporaryDirectory(prefix="xfoil-run-") as tmp:
        work = Path(tmp)
        if naca is not None:
            setup = f"NACA {naca}\nPANE\n"
        else:
            x_in, y_in = coords  # type: ignore[misc]
            _write_coords(work / "airfoil.dat", np.asarray(x_in), np.asarray(y_in))
            setup = "LOAD airfoil.dat\nPCOP\n"

        script = f"{setup}OPER\nALFA {alpha_deg:.6f}\nDUMP dump.txt\nCPWR cp.txt\n\nQUIT\n"
        env = dict(os.environ, XFOIL_HEADLESS="1")
        try:
            proc = subprocess.run(
                [binary],
                input=script,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=work,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise XfoilError(f"XFOIL timed out after {timeout_s:.0f}s") from exc

        dump_file, cp_file = work / "dump.txt", work / "cp.txt"
        if proc.returncode != 0 or not dump_file.exists() or not cp_file.exists():
            tail = "\n".join(proc.stdout.strip().splitlines()[-8:])
            raise XfoilError(
                f"XFOIL failed (exit {proc.returncode}):\n{tail}\n{proc.stderr.strip()[-300:]}"
            )

        dump = np.loadtxt(dump_file, comments="#", usecols=(0, 1, 2, 3))
        cp_table = np.loadtxt(cp_file, comments="#")

    s, x, y, ue = dump.T
    if cp_table.shape[0] != x.size or np.abs(cp_table[:, 0] - x).max() > 1e-9:
        raise XfoilError("DUMP and CPWR node sets disagree")
    cp = cp_table[:, 1]
    if np.abs(cp - (1.0 - ue**2)).max() > 1e-3:
        raise XfoilError("Cp inconsistent with 1 - Ue^2; unexpected XFOIL output format")

    chord, cl_circulation, cl_pressure = _derive_cl(alpha_deg, x, y, ue, cp)
    return XfoilSolution(
        alpha_deg=float(alpha_deg),
        s=s,
        x=x,
        y=y,
        ue=ue,
        cp=cp,
        chord=chord,
        cl_circulation=cl_circulation,
        cl_pressure=cl_pressure,
    )
