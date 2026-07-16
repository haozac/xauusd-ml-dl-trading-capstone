from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone_trading.runtime.dual_terminal_shadow_sync import (
    CalculationOnlyMt5Proxy,
    DualTerminalShadowSyncError,
    build_final_report,
    build_worker_command,
    compare_economic_calculations,
    compare_synchronised_event,
    digest_completed_rates,
    digest_feature_frame,
    digest_numpy,
    inspect_flat_state_and_economics,
    load_and_validate_step4a_report,
    synchronised_event_comparisons,
)


def base_row(role: str, event: str = "2026-07-15T14:00:00+00:00") -> dict[str, object]:
    return {
        "run_id": "run-1",
        "role": role,
        "event_time_utc": event,
        "probability_up": 0.5312345,
        "model_a_signal": 1,
        "model_b_from_flat_signal": 0,
        "model_b_entry_condition": False,
        "model_b_hold_condition": True,
        "role_shadow_signal": 1 if role == "MODEL_A" else 0,
        "duplicate_event": False,
        "stale_event_warning": False,
        "rates_digest": "rates",
        "feature_digest": "features",
        "sequence_digest": "sequence",
        "actual_position_count": 0,
        "pending_order_count": 0,
        "forbidden_trade_function_calls": "",
        "order_check_called": False,
        "order_send_called": False,
    }


def economic_snapshot() -> dict[str, object]:
    return {
        "buy_margin_account_currency": 55.10,
        "sell_margin_account_currency": 55.08,
        "buy_profit_for_positive_price_move_account_currency": 1.35,
        "sell_profit_for_positive_price_move_account_currency": 1.35,
        "order_check_called": False,
        "order_send_called": False,
        # Zero metadata is allowed when broker calculations themselves work.
        "trade_tick_value_metadata": 0.0,
    }


def worker_status(role: str) -> dict[str, object]:
    return {
        "status": "PASS",
        "role": role,
        "role_root": f"runtime/{role.lower()}",
        "login_masked": "*****4309" if role == "MODEL_A" else "******9922",
        "fresh_event_count": 4,
        "final_position_count": 0,
        "final_pending_order_count": 0,
        "order_check_called": False,
        "order_send_called": False,
        "economic_sanity": economic_snapshot(),
        "model_artifact_sha256": "model",
        "scaler_artifact_sha256": "scaler",
        "feature_order_sha256": "features",
    }


def step4a_report() -> dict[str, object]:
    return {
        "status": "PASS",
        "formal_gate": True,
        "order_check_called": False,
        "order_send_called": False,
        "model_a": {
            "terminal_executable": "/tmp/a/terminal64.exe",
            "account": {"login_masked": "*****4309"},
            "package": {"terminal_version": "(500, 5836, test)"},
            "terminal": {"name": "Dukascopy MetaTrader 5"},
        },
        "model_b": {
            "terminal_executable": "/tmp/b/terminal64.exe",
            "account": {"login_masked": "******9922"},
            "package": {"terminal_version": "(500, 5836, test)"},
            "terminal": {"name": "Dukascopy MetaTrader 5"},
        },
        "cross_terminal_review": {
            "checks": {
                "terminal_paths_distinct": True,
                "terminal_data_paths_distinct": True,
                "accounts_distinct": True,
                "symbol_contract_matches": True,
                "latest_completed_bar_difference_within_gate": True,
            }
        },
    }


def test_numpy_digest_is_deterministic_and_sensitive() -> None:
    values = np.arange(12, dtype=float).reshape(3, 4)
    assert digest_numpy(values) == digest_numpy(values.copy())
    changed = values.copy()
    changed[0, 0] += 1
    assert digest_numpy(values) != digest_numpy(changed)


def test_feature_frame_digest_includes_index_and_columns() -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="15min", tz="UTC")
    frame = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}, index=index)
    assert digest_feature_frame(frame) == digest_feature_frame(frame.copy())
    moved = frame.copy()
    moved.index = moved.index + pd.Timedelta(minutes=15)
    assert digest_feature_frame(frame) != digest_feature_frame(moved)


def test_completed_rate_digest_is_deterministic() -> None:
    rates = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC"),
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "tick_volume": [10, 12],
        }
    )
    assert digest_completed_rates(rates) == digest_completed_rates(rates.copy())


def test_matching_event_passes() -> None:
    result = compare_synchronised_event(base_row("MODEL_A"), base_row("MODEL_B"))
    assert result.passed is True
    assert result.failures == ()


def test_probability_mismatch_fails() -> None:
    row_b = base_row("MODEL_B")
    row_b["probability_up"] = 0.54
    result = compare_synchronised_event(base_row("MODEL_A"), row_b)
    assert result.passed is False
    assert "probability_mismatch" in result.failures


def test_sequence_digest_mismatch_fails() -> None:
    row_b = base_row("MODEL_B")
    row_b["sequence_digest"] = "different"
    result = compare_synchronised_event(base_row("MODEL_A"), row_b)
    assert result.passed is False
    assert "sequence_digest_mismatch" in result.failures


def test_open_position_fails() -> None:
    row_b = base_row("MODEL_B")
    row_b["actual_position_count"] = 1
    result = compare_synchronised_event(base_row("MODEL_A"), row_b)
    assert result.passed is False
    assert "broker_position_detected" in result.failures


def test_duplicate_and_stale_rows_are_excluded() -> None:
    fresh_a = base_row("MODEL_A")
    fresh_b = base_row("MODEL_B")
    duplicate_a = base_row("MODEL_A", "2026-07-15T14:15:00+00:00")
    duplicate_b = base_row("MODEL_B", "2026-07-15T14:15:00+00:00")
    duplicate_a["duplicate_event"] = True
    duplicate_b["duplicate_event"] = True
    results = synchronised_event_comparisons(
        [fresh_a, duplicate_a], [fresh_b, duplicate_b], run_id="run-1"
    )
    assert len(results) == 1
    assert results[0].event_time_utc == fresh_a["event_time_utc"]


def test_economic_calculations_accept_zero_tick_metadata() -> None:
    result = compare_economic_calculations(economic_snapshot(), economic_snapshot())
    assert result.passed is True
    assert result.values_match is True


def test_economic_calculation_mismatch_fails() -> None:
    model_b = economic_snapshot()
    model_b["buy_margin_account_currency"] = 70.0
    result = compare_economic_calculations(economic_snapshot(), model_b)
    assert result.passed is False
    assert "economic_calculation_mismatch" in result.failures


def test_calculation_proxy_blocks_order_send_and_order_check() -> None:
    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def order_send(self, request):  # pragma: no cover - must never execute
            raise AssertionError

        def order_check(self, request):  # pragma: no cover - must never execute
            raise AssertionError

    proxy = CalculationOnlyMt5Proxy(FakeMt5())
    with pytest.raises(DualTerminalShadowSyncError):
        proxy.order_send({})
    with pytest.raises(DualTerminalShadowSyncError):
        proxy.order_check({})
    assert proxy.forbidden_attempts == ["order_send", "order_check"]


def test_step4a_report_path_and_checks_are_validated(tmp_path: Path) -> None:
    report = step4a_report()
    path = tmp_path / "step4a.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    loaded = load_and_validate_step4a_report(
        report_path=path,
        model_a_terminal_path="/tmp/a/terminal64.exe",
        model_b_terminal_path="/tmp/b/terminal64.exe",
    )
    assert loaded["formal_gate"] is True


def test_step4a_wrong_terminal_path_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "step4a.json"
    path.write_text(json.dumps(step4a_report()), encoding="utf-8")
    with pytest.raises(DualTerminalShadowSyncError):
        load_and_validate_step4a_report(
            report_path=path,
            model_a_terminal_path="/tmp/wrong/terminal64.exe",
            model_b_terminal_path="/tmp/b/terminal64.exe",
        )


def test_worker_command_uses_separate_role_and_exact_paths(tmp_path: Path) -> None:
    command = build_worker_command(
        python_executable="python",
        script_path=tmp_path / "runner.py",
        repo_root=tmp_path,
        run_id="run-1",
        role="MODEL_A",
        config_path=tmp_path / "config.yaml",
        model_a_config=tmp_path / "a.yaml",
        model_b_config=tmp_path / "b.yaml",
        freeze_manifest=tmp_path / "manifest.json",
        step4a_report=tmp_path / "step4a.json",
        worker_root=tmp_path / "runtime/model_a/run-1",
        stop_file=tmp_path / "STOP",
        deadline_utc="2026-07-15T15:00:00+00:00",
        poll_seconds=30,
        server_time_offset_hours=3,
        allow_onednn=False,
    )
    assert command[0] == "python"
    assert command[1] == str(tmp_path / "runner.py")
    assert command[command.index("--worker-role") + 1] == "MODEL_A"
    assert command[command.index("--repo-root") + 1] == str(tmp_path)
    assert command[command.index("--config") + 1] == str(tmp_path / "config.yaml")
    assert command[command.index("--worker-root") + 1] == str(
        tmp_path / "runtime" / "model_a" / "run-1"
    )
    assert command[command.index("--stop-file") + 1] == str(tmp_path / "STOP")



def test_calculation_session_is_order_free_and_returns_positive_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readiness = types.ModuleType("capstone_trading.runtime.mt5_readiness")
    readiness.object_to_plain_dict = lambda value: dict(vars(value))
    readiness.safe_last_error = lambda proxy: (0, "ok")
    monkeypatch.setitem(sys.modules, "capstone_trading.runtime.mt5_readiness", readiness)

    terminal_dir = tmp_path / "terminal_a"
    terminal_dir.mkdir()
    terminal_exe = terminal_dir / "terminal64.exe"
    terminal_exe.write_text("stub", encoding="utf-8")

    class Obj:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class FakeMt5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1

        def __init__(self):
            self.order_send_called = False
            self.order_check_called = False

        def initialize(self, path):
            return Path(path) == terminal_exe

        def shutdown(self):
            return True

        def last_error(self):
            return (0, "ok")

        def terminal_info(self):
            return Obj(path=str(terminal_dir), connected=True)

        def account_info(self):
            return Obj(login=12344309, currency="SGD")

        def symbol_info(self, symbol):
            return Obj(spread=540, trade_tick_value=0.0, trade_tick_value_profit=0.0, trade_tick_value_loss=0.0)

        def symbol_info_tick(self, symbol):
            return Obj(bid=4056.0, ask=4056.54)

        def positions_get(self, symbol):
            return ()

        def orders_get(self, symbol):
            return ()

        def order_calc_margin(self, *args, **kwargs):
            # Reproduce the live failure mode: keyword use returns 0 instead
            # of raising, while the documented positional call works.
            if kwargs:
                return 0.0
            assert args == (self.ORDER_TYPE_BUY, "XAUUSD", 0.01, 4056.54) or args == (
                self.ORDER_TYPE_SELL, "XAUUSD", 0.01, 4056.0
            )
            return 54.8

        def order_calc_profit(self, *args, **kwargs):
            if kwargs:
                return 0.0
            assert len(args) == 5
            return 1.35

    result = inspect_flat_state_and_economics(
        mt5_module=FakeMt5(),
        terminal_path=str(terminal_exe),
        expected_terminal_directory=str(terminal_dir),
        expected_login_masked="****4309",
        symbol="XAUUSD",
        volume=0.01,
    )
    assert result["status"] == "PASS"
    assert result["buy_margin_account_currency"] == pytest.approx(54.8)
    assert result["buy_profit_for_positive_price_move_account_currency"] == pytest.approx(1.35)
    assert result["calculation_call_mode"] == "documented_required_unnamed_positional_parameters"
    assert result["order_check_called"] is False
    assert result["order_send_called"] is False



def test_missing_economic_payload_does_not_invent_forbidden_order_call() -> None:
    comparison = compare_economic_calculations({}, {})
    assert comparison.passed is False
    assert "model_a_economic_calculation_invalid" in comparison.failures
    assert "model_b_economic_calculation_invalid" in comparison.failures
    assert "forbidden_order_call_in_economic_check" not in comparison.failures

def test_final_report_passes_with_four_clean_events(tmp_path: Path) -> None:
    comparisons = []
    for minute in (0, 15, 30, 45):
        event = f"2026-07-15T14:{minute:02d}:00+00:00"
        comparisons.append(
            compare_synchronised_event(base_row("MODEL_A", event), base_row("MODEL_B", event))
        )
    report = build_final_report(
        run_id="run-1",
        started_utc="2026-07-15T14:00:00+00:00",
        completed_utc="2026-07-15T15:00:00+00:00",
        required_synchronised_events=4,
        poll_seconds=30,
        max_runtime_minutes=90,
        comparisons=comparisons,
        worker_status_a=worker_status("MODEL_A"),
        worker_status_b=worker_status("MODEL_B"),
        step4a_report=step4a_report(),
        source_paths={},
    )
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["decision"]["final_14_day_run_authorised"] is False


def test_final_report_fails_with_only_three_events() -> None:
    comparisons = [
        compare_synchronised_event(
            base_row("MODEL_A", f"2026-07-15T14:{minute:02d}:00+00:00"),
            base_row("MODEL_B", f"2026-07-15T14:{minute:02d}:00+00:00"),
        )
        for minute in (0, 15, 30)
    ]
    report = build_final_report(
        run_id="run-1",
        started_utc="2026-07-15T14:00:00+00:00",
        completed_utc="2026-07-15T14:30:00+00:00",
        required_synchronised_events=4,
        poll_seconds=30,
        max_runtime_minutes=90,
        comparisons=comparisons,
        worker_status_a=worker_status("MODEL_A"),
        worker_status_b=worker_status("MODEL_B"),
        step4a_report=step4a_report(),
        source_paths={},
    )
    assert report["status"] == "FAIL"
    assert report["validations"]["minimum_synchronised_events_observed"] is False
