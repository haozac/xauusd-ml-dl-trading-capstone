from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.tensorflow
@pytest.mark.step3
def test_formal_stage1_step3_gate_passes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    report_name = "runtime/reports/test_stage1_step3_inference_parity.json"
    summary_name = "runtime/reports/test_stage1_step3_probability_summary.csv"
    diagnostic_name = "runtime/reports/test_stage1_step3_probability_diagnostics.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_inference_parity.py"),
            "--repo-root",
            str(repository_root),
            "--report",
            report_name,
            "--summary-csv",
            summary_name,
            "--diagnostic-csv",
            diagnostic_name,
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
    aggregate = payload["checks"]["aggregate_inference_parity"]
    assert aggregate["total_sequences"] == 24799
    assert aggregate["total_threshold_flips"] == 0
    assert aggregate["maximum_absolute_difference"] <= 1e-5
