from __future__ import annotations

import os
from pathlib import Path

import pytest

from capstone_trading.runtime.order_execution_probe import CONFIRM_SEND_TOKEN, run_tiny_order_test


@pytest.mark.skipif(
    os.environ.get("RUN_STAGE3_TINY_ORDER_INTEGRATION") != "1",
    reason="Set RUN_STAGE3_TINY_ORDER_INTEGRATION=1 to run the live Stage 3 Step 2 tiny order gate",
)
def test_stage3_step2_tiny_order_live_gate():
    terminal_path = os.environ.get("STAGE3_MT5_TERMINAL_PATH")
    report = run_tiny_order_test(
        repo_root=Path(".").resolve(),
        terminal_path=terminal_path,
        confirmation=CONFIRM_SEND_TOKEN,
    )
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["validations"]["no_position_after_close"] is True
