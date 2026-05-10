from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_parse_wb_reviews_help() -> None:
    script = Path("scripts/parse_wb_reviews.py")

    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert b"parse_wb_reviews.py" in result.stdout
