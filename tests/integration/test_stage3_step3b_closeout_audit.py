from __future__ import annotations

import os
import subprocess
import sys

import pytest


def test_stage3_step3b_closeout_help_runs():
    result = subprocess.run(
        [sys.executable, "scripts/deployment/audit_model_b_controlled_live_closeout.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "offline" in result.stdout.lower()
    assert "--run-id" in result.stdout


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE3_MODEL_B_CLOSEOUT_INTEGRATION") != "1",
    reason="Set RUN_STAGE3_MODEL_B_CLOSEOUT_INTEGRATION=1 to audit the local Stage 3 Step 3B artefacts",
)
def test_stage3_step3b_closeout_real_artefacts():
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/audit_model_b_controlled_live_closeout.py",
            "--repo-root", ".",
            "--run-id", "stage3_step3b_20260713T114117Z",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + "\n" + result.stderr
    assert "closeout status: PASS" in result.stderr or "closeout status: PASS" in result.stdout
