"""Run from a source checkout without requiring an editable installation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soccer_vision.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
