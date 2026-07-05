from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_stage2_step1_mt5_readiness_gate_runs_when_explicitly_enabled() -> None:
    if os.environ.get("RUN_STAGE2_MT5_READINESS_INTEGRATION") != "1":
        pytest.skip("Set RUN_STAGE2_MT5_READINESS_INTEGRATION=1 to run the live MT5 readiness gate")

    repository_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_mt5_environment_readiness.py"),
            "--repo-root",
            str(repository_root),
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
