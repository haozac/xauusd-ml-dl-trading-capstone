from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.step6
def test_formal_stage1_step6_gate_passes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    required = repository_root / "runtime" / "reports" / "stage1_step5_model_b_diagnostic_replay.json"
    if not required.exists():
        pytest.skip("Local formal Stage 1 Step 5 report is not available")

    report_name = "runtime/reports/test_stage1_step6_offline_runtime_simulation.json"
    summary_name = "runtime/reports/test_stage1_step6_runtime_summary.csv"
    audit_name = "runtime/reports/test_stage1_step6_runtime_event_audit_1bps.csv"
    resume_name = "runtime/reports/test_stage1_step6_resume_determinism.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_offline_runtime_simulation.py"),
            "--repo-root",
            str(repository_root),
            "--report",
            report_name,
            "--summary-csv",
            summary_name,
            "--audit-csv",
            audit_name,
            "--resume-csv",
            resume_name,
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    payload = json.loads((repository_root / report_name).read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["formal_gate"] is True
    assert payload["mt5_used"] is False
    assert payload["orders_enabled"] is False
    assert payload["checks"]["aggregate_runtime_simulation"]["total_runtime_events"] == 24799
