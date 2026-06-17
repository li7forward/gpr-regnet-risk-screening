#!/usr/bin/env python
"""Command-line entry point for GAFR-RegNet training."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gpr_risk.train import main


if __name__ == "__main__":
    main()
