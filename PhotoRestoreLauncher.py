#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PhotoRestoreLauncher.py
Single EXE entrypoint to choose between:
  1) ACES/OCIO-based conversion
  2) LUT-based conversion
Then hands off to the corresponding script's interactive flow.
"""

import sys
from pathlib import Path

# Import your two flows as modules
try:
    import ACES_CONVERTION as aces_mod   # your ACES/OCIO script
except Exception as e:
    aces_mod = None
    print("[WARN] Failed to import ACES_CONVERTION.py:", e)

try:
    import LUTs as luts_mod               # your LUT script
except Exception as e:
    luts_mod = None
    print("[WARN] Failed to import LUTs.py:", e)


def ensure_dirs():
    """Create default IO folders if missing."""
    for p in (Path("input"), Path("output"), Path("ACES_Ver"), Path("LUTs")):
        p.mkdir(parents=True, exist_ok=True)


def run_aces():
    """
    Call ACES_CONVERTION.main() with default args:
      --input-dir ./input
      --output-dir ./output
      --aces-dir ./ACES_Ver
      --recursive --keep-exif
    """
    if aces_mod is None:
        print("[ERR] ACES flow is unavailable.")
        return

    argv_bak = sys.argv[:]
    try:
        sys.argv = [
            "ACES_CONVERTION.py",
            "--input-dir", "./input",
            "--output-dir", "./output",
            "--aces-dir", "./ACES_Ver",
            "--quality", "100",
            "--recursive",
            "--keep-exif",
        ]
        aces_mod.main()
    finally:
        sys.argv = argv_bak


def run_luts():
    """
    Call LUTs.main() with default args:
      --input-dir ./input
      --output-dir ./output
      --luts-dir ./LUTs
      --recursive --keep-exif
    """
    if luts_mod is None:
        print("[ERR] LUT flow is unavailable.")
        return

    argv_bak = sys.argv[:]
    try:
        sys.argv = [
            "LUTs.py",
            "--input-dir", "./input",
            "--output-dir", "./output",
            "--luts-dir", "./LUTs",
            "--quality", "100",
            "--recursive",
            "--keep-exif",
        ]
        luts_mod.main()
    finally:
        sys.argv = argv_bak


def main():
    ensure_dirs()

    print("\n=== Photo Restore Launcher ===")
    print("[1] ACES (OCIO) conversion")
    print("[2] LUTs (.cube) conversion")
    print("[Q] Quit")

    while True:
        choice = input("Choose 1/2 (or Q): ").strip().lower()
        if choice == "1":
            run_aces()
            break
        elif choice == "2":
            run_luts()
            break
        elif choice in ("q", "quit", "exit"):
            print("Bye.")
            break
        else:
            print("Invalid choice. Try again.")


if __name__ == "__main__":
    main()
