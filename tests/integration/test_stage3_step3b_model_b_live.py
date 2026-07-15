from __future__ import annotations

import os
import subprocess
import sys

import pytest


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE3_MODEL_B_LIVE_INTEGRATION") != "1",
    reason="Set RUN_STAGE3_MODEL_B_LIVE_INTEGRATION=1 to run the live Stage 3 Step 3B controlled live gate",
)
def test_stage3_step3b_live_script_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/deployment/run_model_b_controlled_live.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Stage 3 Step 3B" in result.stdout
