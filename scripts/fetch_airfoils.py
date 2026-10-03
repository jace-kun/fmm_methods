"""Download a small curated Selig/UIUC DAT set into data/airfoils/."""

from __future__ import annotations

import urllib.request
from pathlib import Path

# Curated samples only — do not vendor the full Selig dump.
AIRFOILS = {
    "naca0012.dat": "https://m-selig.ae.illinois.edu/ads/coord/naca0012.dat",
    "naca4412.dat": "https://m-selig.ae.illinois.edu/ads/coord/naca4412.dat",
}

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "airfoils"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, url in AIRFOILS.items():
        dest = OUT / name
        print(f"Fetching {name} ...")
        urllib.request.urlretrieve(url, dest)
        print(f"  -> {dest}")


if __name__ == "__main__":
    main()
