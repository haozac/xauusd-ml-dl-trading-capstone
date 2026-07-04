from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.step2
def test_formal_stage1_step2_gate_passes(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    report_path = tmp_path / "step2.json"
    column_path = tmp_path / "columns.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_feature_parity.py"),
            "--repo-root",
            str(repository_root),
            "--report",
            str(report_path.relative_to(repository_root))
            if report_path.is_relative_to(repository_root)
            else "runtime/reports/test_stage1_step2.json",
            "--column-report",
            str(column_path.relative_to(repository_root))
            if column_path.is_relative_to(repository_root)
            else "runtime/reports/test_stage1_step2_columns.csv",
        ],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    actual_report = repository_root / "runtime" / "reports" / "test_stage1_step2.json"
    payload = json.loads(actual_report.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["checks"]["feature_parity"]["total_mismatch_count"] == 0
    assert (
        payload["checks"]["sequence_partitions"]["partitions"]["final_holdout"][
            "sequence_count"
        ]
        == 13794
    )
