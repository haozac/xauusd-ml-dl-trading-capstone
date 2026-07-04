from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.step5
def test_formal_stage1_step5_gate_passes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    required = repository_root / "runtime" / "reports" / "stage1_step4_model_a_replay_parity.json"
    if not required.exists():
        pytest.skip("Stage 1 Step 4 PASS report is not available")

    report_name = "runtime/reports/test_stage1_step5_model_b_diagnostic_replay.json"
    metrics_name = "runtime/reports/test_stage1_step5_model_b_metrics_by_cost.csv"
    comparison_name = "runtime/reports/test_stage1_step5_model_b_vs_model_a_comparison.csv"
    diagnostics_name = "runtime/reports/test_stage1_step5_model_b_diagnostics.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_model_b_diagnostic_replay.py"),
            "--repo-root",
            str(repository_root),
            "--report",
            report_name,
            "--metrics-csv",
            metrics_name,
            "--comparison-csv",
            comparison_name,
            "--diagnostics-csv",
            diagnostics_name,
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
    assert payload["diagnostic_only"] is True
    assert payload["checks"]["model_b_invariants"]["passed"] is True
