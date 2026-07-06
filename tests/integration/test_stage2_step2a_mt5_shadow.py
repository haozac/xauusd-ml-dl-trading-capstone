from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_stage2_step2a_mt5_shadow_logger_runs_when_explicitly_enabled() -> None:
    if os.environ.get("RUN_STAGE2_MT5_SHADOW_INTEGRATION") != "1":
        pytest.skip("Set RUN_STAGE2_MT5_SHADOW_INTEGRATION=1 to run the live MT5 shadow logger gate")

    repository_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        str(repository_root / "scripts" / "deployment" / "run_mt5_shadow_logger.py"),
        "--repo-root",
        str(repository_root),
        "--once",
    ]
    terminal_path = os.environ.get("STAGE2_MT5_TERMINAL_PATH")
    if terminal_path:
        cmd.extend(["--terminal-path", terminal_path])
    completed = subprocess.run(
        cmd,
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
