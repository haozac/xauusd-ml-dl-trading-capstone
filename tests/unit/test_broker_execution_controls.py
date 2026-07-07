from capstone_trading.runtime.broker_execution_controls import (
    FrozenExecutionControls,
    build_capital_review,
    build_freeze_report,
    is_volume_step_valid,
)


def fake_step2a_report():
    return {
        "status": "PASS",
        "formal_gate": True,
        "shadow_only": True,
        "orders_enabled": False,
        "account": {
            "company": "Dukascopy Bank SA",
            "server": "Dukascopy-demo-mt5-1",
            "trade_mode_name": "DEMO",
            "margin_mode": 2,
        },
        "terminal": {
            "connected": True,
            "trade_allowed": False,
            "tradeapi_disabled": False,
        },
        "rates": {
            "uses_completed_bars_only": True,
            "start_pos": 1,
        },
        "time_normalisation": {
            "conversion_applied": True,
            "mt5_server_time_offset_hours": 3,
            "latest_bar_future_minutes_after_conversion": 0.0,
        },
        "symbol_resolution": {
            "selected_symbol": "XAUUSD",
            "symbol_info": {
                "visible": True,
                "digits": 3,
                "point": 0.001,
                "trade_contract_size": 100.0,
                "volume_min": 0.01,
                "volume_step": 0.01,
                "volume_max": 10000.0,
                "spread": 440,
                "spread_float": True,
                "trade_mode": 4,
                "filling_mode": 2,
                "order_mode": 127,
                "ask": 4147.295,
                "bid": 4146.855,
                "currency_profit": "USD",
            },
        },
    }


def test_volume_step_validation_accepts_0_01():
    assert is_volume_step_valid(0.01, 0.01, 0.01)
    assert is_volume_step_valid(0.02, 0.01, 0.01)


def test_volume_step_validation_rejects_off_grid():
    assert not is_volume_step_valid(0.015, 0.01, 0.01)
    assert not is_volume_step_valid(0.0, 0.01, 0.01)


def test_freeze_report_passes_with_valid_shadow_report():
    report = build_freeze_report(step2a_report=fake_step2a_report(), account_equity_sgd=1000.0)
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["no_order_stage"] is True
    assert report["order_send_called"] is False
    assert report["checks"]["validations"]["volume_step_allows_0_01"] is True


def test_capital_review_flags_small_account_leverage_risk():
    controls = FrozenExecutionControls()
    symbol_info = fake_step2a_report()["symbol_resolution"]["symbol_info"]
    review = build_capital_review(controls=controls, symbol_info=symbol_info, account_equity_sgd=200.0)
    assert review["estimated_notional_usd"] > 4000
    assert review["capstone_leverage_cap_passed_if_equity_supplied"] is False
    assert review["capital_ready_for_final_strategy_execution"] is False


def test_capital_review_accepts_larger_demo_equity():
    controls = FrozenExecutionControls()
    symbol_info = fake_step2a_report()["symbol_resolution"]["symbol_info"]
    review = build_capital_review(controls=controls, symbol_info=symbol_info, account_equity_sgd=1000.0)
    assert review["capstone_leverage_cap_passed_if_equity_supplied"] is True
    assert review["capital_ready_for_final_strategy_execution"] is True
