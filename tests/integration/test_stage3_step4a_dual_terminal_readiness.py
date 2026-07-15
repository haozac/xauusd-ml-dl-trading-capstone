from __future__ import annotations

import os
from pathlib import Path

import pytest

from capstone_trading.runtime.dual_terminal_readiness import build_dual_terminal_report, load_dual_terminal_config
from capstone_trading.runtime.mt5_readiness import import_metatrader5_module


def test_step4a_config_loads():
    template = Path("config/dual_terminal_runtime_template.yaml")
    if not template.exists():
        pytest.skip("Step 4A template is not installed")
    cfg = load_dual_terminal_config(template)
    assert cfg.model_a.magic_number == 26070101
    assert cfg.model_b.magic_number == 26070102


@pytest.mark.skipif(
    os.getenv("RUN_STAGE3_DUAL_TERMINAL_READINESS_INTEGRATION") != "1",
    reason="Set RUN_STAGE3_DUAL_TERMINAL_READINESS_INTEGRATION=1 to run the live no-order Step 4A gate",
)
def test_live_dual_terminal_gate():
    cfg = load_dual_terminal_config(Path("config/dual_terminal_runtime.yaml"))
    report = build_dual_terminal_report(mt5_module=import_metatrader5_module(), config=cfg)
    assert report["formal_gate"] is True
    assert report["order_send_called"] is False
