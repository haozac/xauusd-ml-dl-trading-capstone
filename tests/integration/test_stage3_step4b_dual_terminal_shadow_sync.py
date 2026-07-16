from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _repo_root() -> Path:
    cwd = Path.cwd()
    if (cwd / "scripts/deployment/run_dual_terminal_shadow_sync.py").exists():
        return cwd
    return Path(__file__).resolve().parents[2]


def test_step4b_runner_is_installed() -> None:
    assert (_repo_root() / "scripts/deployment/run_dual_terminal_shadow_sync.py").exists()


@pytest.mark.skipif(
    os.getenv("RUN_STAGE3_DUAL_TERMINAL_SHADOW_SYNC_INTEGRATION") != "1",
    reason=(
        "Set RUN_STAGE3_DUAL_TERMINAL_SHADOW_SYNC_INTEGRATION=1 to run the live "
        "no-order Step 4B gate"
    ),
)
def test_live_dual_terminal_shadow_sync_gate() -> None:
    repo_root = _repo_root()
    result = subprocess.run(
        [
            sys.executable,
            "scripts/deployment/run_dual_terminal_shadow_sync.py",
            "--repo-root",
            str(repo_root),
            "--required-synchronised-events",
            "1",
            "--poll-seconds",
            "30",
            "--max-runtime-minutes",
            "20",
        ],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
        timeout=1500,
    )
    assert result.returncode == 0, result.stdout + result.stderr
