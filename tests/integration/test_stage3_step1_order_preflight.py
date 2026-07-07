from __future__ import annotations

import os
from pathlib import Path

import pytest

from capstone_trading.runtime.order_preflight import run_preflight


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE3_ORDER_PREFLIGHT_INTEGRATION") != "1",
    reason="Set RUN_STAGE3_ORDER_PREFLIGHT_INTEGRATION=1 to run the live Stage 3 Step 1 order_check preflight gate",
)
def test_stage3_step1_order_preflight_live_gate():
    repo_root = Path.cwd()
    terminal_path = os.environ.get("MT5_TERMINAL_PATH")
    report = run_preflight(repo_root=repo_root, terminal_path=terminal_path)
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["order_check_called"] is True
    assert report["order_send_called"] is False
    assert report["orders_executed"] is False
