"""Regenerate XFOIL golden reference outputs with provenance.

Run only when deliberately updating the oracle (e.g. after an XFOIL upgrade):

    uv run python scripts/regen_xfoil_goldens.py

Review the diff of tests/regression/golden/xfoil_cases.json before committing.
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import date
from pathlib import Path

from fmm_vpm.io_xfoil import find_xfoil, run_inviscid

CASES = [("0012", 0.0), ("0012", 5.0), ("4412", 4.0)]
OUT = Path(__file__).resolve().parents[1] / "tests" / "regression" / "golden" / "xfoil_cases.json"


def _capture(cmd: list[str], stdin: str | None = None) -> str:
    result = subprocess.run(cmd, input=stdin, capture_output=True, text=True, check=False)
    return (result.stdout + result.stderr).strip()


def _provenance() -> dict[str, str]:
    binary = find_xfoil()
    banner = _capture([binary], stdin="QUIT\n")
    version = re.search(r"XFOIL Version\s+(\S+)", banner)
    tap_commit = "unknown"
    tap_repo = _capture(["brew", "--repo", "liuyanwpuuci/aerospace"])
    if tap_repo and Path(tap_repo).is_dir():
        tap_commit = _capture(["git", "-C", tap_repo, "rev-parse", "HEAD"])
    return {
        "generated": date.today().isoformat(),
        "xfoil_binary": binary,
        "xfoil_version": version.group(1) if version else "unknown",
        "homebrew_formula": _capture(["brew", "list", "--versions", "xfoil"]),
        "tap": "liuyanwpuuci/aerospace",
        "tap_commit": tap_commit,
        "command": "NACA <code> / PANE / OPER / ALFA <a> / DUMP / CPWR (XFOIL_HEADLESS=1)",
    }


def main() -> None:
    cases = []
    for naca, alpha in CASES:
        sol = run_inviscid(alpha_deg=alpha, naca=naca)
        cases.append(
            {
                "naca": naca,
                "alpha_deg": alpha,
                "cl_circulation": sol.cl_circulation,
                "cl_pressure": sol.cl_pressure,
                "x": sol.x.tolist(),
                "cp": sol.cp.tolist(),
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"provenance": _provenance(), "cases": cases}, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
