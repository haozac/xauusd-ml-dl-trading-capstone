from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
@pytest.mark.step4
def test_formal_stage1_step4_gate_passes() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    required = (
        repository_root
        / "notebook_outputs"
        / "07_m15_cnn_lstm_final_holdout_evaluation"
        / "tables"
        / "final_holdout_strategy_bar_log_1bps.csv"
    )
    if not required.exists():
        pytest.skip("Local gitignored Notebook 7 strategy artefacts are not available")

    report_name = "runtime/reports/test_stage1_step4_model_a_replay_parity.json"
    metrics_name = "runtime/reports/test_stage1_step4_model_a_metrics_by_cost.csv"
    bar_name = "runtime/reports/test_stage1_step4_bar_log_comparison.csv"
    completed = subprocess.run(
        [
            sys.executable,
            str(repository_root / "scripts" / "deployment" / "run_model_a_replay_parity.py"),
            "--repo-root",
            str(repository_root),
            "--report",
            report_name,
            "--metrics-csv",
            metrics_name,
            "--bar-comparison-csv",
            bar_name,
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
    assert payload["checks"]["final_holdout_bar_log_1bps_parity"]["comparison_count"] >= 5
    metrics = payload["checks"]["model_a_replay_metrics"]["final_holdout"]["1.0"]
    assert metrics["row_count"] == 13794
    assert metrics["total_stop_triggered"] is True
