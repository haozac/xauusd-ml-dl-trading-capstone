import os
from pathlib import Path

import pytest

from capstone_trading.runtime.broker_execution_controls import run_freeze


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE2_BROKER_EXECUTION_CONTROLS_INTEGRATION") != "1",
    reason="Set RUN_STAGE2_BROKER_EXECUTION_CONTROLS_INTEGRATION=1 to run the Stage 2 Step 3A broker execution controls freeze gate",
)
def test_stage2_step3a_freeze_gate_runs_against_latest_step2a_report():
    repo_root = Path.cwd()
    report = run_freeze(repo_root=repo_root, account_equity_sgd=None)
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["no_order_stage"] is True
    assert report["order_send_called"] is False
