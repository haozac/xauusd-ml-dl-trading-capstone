from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE2_MODEL_B_MIN_HOLD_INTEGRATION") != "1",
    reason="Set RUN_STAGE2_MODEL_B_MIN_HOLD_INTEGRATION=1 to run the historical Step 2C diagnostic gate",
)
def test_stage2_step2c_script_runs() -> None:
    repo_root = Path.cwd()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/run_model_b_min_hold_comparison.py",
            "--repo-root",
            str(repo_root),
        ],
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert result.returncode == 0, result.stdout
