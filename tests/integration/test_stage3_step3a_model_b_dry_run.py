from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_stage3_step3a_model_b_dry_run_live_gate():
    if os.environ.get("RUN_STAGE3_MODEL_B_DRY_RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_STAGE3_MODEL_B_DRY_RUN_INTEGRATION=1 to run the live Stage 3 Step 3A dry-run gate")
    repo_root = Path.cwd()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/run_model_b_controlled_dry_run.py",
            "--repo-root",
            str(repo_root),
            "--once",
            "--min-new-events",
            "0",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
