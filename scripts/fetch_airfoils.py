"""Generate NACA sections and optionally fetch a small Selig/UIUC DAT set.

NACA four-digit sections are generated locally: UIUC does not consistently
serve every historical ``nacaXXXX.dat`` filename.  This deliberately does not
download the full Selig archive.
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

from fmm_vpm.geometry import naca4

NACA_CODES = ("0012", "2412", "4412")
# Curated non-NACA sections only — do not vendor the full Selig dump.
SELIG_AIRFOILS = {
    "e216.dat": "https://m-selig.ae.illinois.edu/ads/coord/e216.dat",
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "airfoils"


def _write_dat(name: str, x: list[float], y: list[float]) -> None:
    rows = "\n".join(f"{xi:.10f} {yi:.10f}" for xi, yi in zip(x, y, strict=True))
    (OUT / name).write_text(f"{name.removesuffix('.dat')}\n{rows}\n")


def main(*, fetch_selig: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for code in NACA_CODES:
        geometry = naca4(code)
        _write_dat(f"naca{code}.dat", geometry.nodes[:, 0].tolist(), geometry.nodes[:, 1].tolist())
        print(f"Generated NACA {code} -> {OUT / f'naca{code}.dat'}")
    if not fetch_selig:
        return
    for name, url in SELIG_AIRFOILS.items():
        dest = OUT / name
        print(f"Fetching {name} ...")
        with urllib.request.urlopen(url, timeout=30) as response:
            dest.write_bytes(response.read())
        print(f"  -> {dest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--with-selig",
        action="store_true",
        help="also fetch the curated, non-NACA Selig DAT files",
    )
    main(fetch_selig=parser.parse_args().with_selig)
