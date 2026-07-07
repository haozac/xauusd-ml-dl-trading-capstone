from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from capstone_trading.runtime.order_preflight import (
    GuardedMt5OrderCheckProxy,
    Stage3OrderPreflightError,
    choose_filling_candidates,
    is_volume_step_valid,
    load_frozen_controls,
    run_preflight,
)


TerminalInfo = namedtuple("TerminalInfo", "connected trade_allowed tradeapi_disabled path build")
AccountInfo = namedtuple(
    "AccountInfo",
    "login company server trade_mode trade_allowed trade_expert currency equity balance margin_free margin_mode",
)
SymbolInfo = namedtuple(
    "SymbolInfo",
    "name visible digits point trade_contract_size volume_min volume_step volume_max spread spread_float trade_mode filling_mode order_mode trade_exemode ask bid currency_profit",
)
TickInfo = namedtuple("TickInfo", "ask bid time flags")
CheckResult = namedtuple(
    "CheckResult",
    "retcode balance equity profit margin margin_free margin_level comment request",
)


FROZEN_YAML = '''# test frozen controls
metadata:
  controls_version: "stage2_step3a_v1_0"
broker:
  broker_company_expected: "Dukascopy Bank SA"
  server_expected: "Dukascopy-demo-mt5-1"
  symbol: "XAUUSD"
  timeframe: "M15"
time:
  canonical_time_basis: "UTC"
  mt5_server_time_policy: "Dukascopy GMT+3 summer / GMT+2 winter, convert to canonical UTC"
  mt5_server_time_offset_hours_current: 3
execution_limits:
  order_volume_lots: 0.01
  max_open_volume_lots_per_model: 0.01
  max_positions_per_model: 1
  max_spread_points_for_entry: 800
  max_deviation_points_for_stage3_order_request: 200
  capstone_leverage_cap: 10.0
  minimum_demo_equity_recommendation_sgd: 1000.0
order_policy:
  stage2_orders_enabled: false
  stage3_order_check_required_before_order_send: true
  stage3_first_order_test_volume_lots: 0.01
identifiers:
  model_a_magic_number: 26070101
  model_b_magic_number: 26070102
  model_a_order_comment: "CAPSTONE_MODEL_A"
  model_b_order_comment: "CAPSTONE_MODEL_B"
model_policy:
  model_b_variant_for_first_controlled_execution: "MODEL_B_V2_CURRENT"
  max_successful_entries_per_model_per_utc_day: 1
review_assumptions:
  review_usd_sgd_rate_assumption: 1.35
'''


class FakeMt5:
    __author__ = "MetaQuotes Ltd."
    __version__ = "5.0.test"

    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    TRADE_ACTION_DEAL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    SYMBOL_TRADE_EXECUTION_MARKET = 2

    def __init__(self, *, trade_allowed: bool = True, check_retcode: int = 0):
        self.trade_allowed = trade_allowed
        self.check_retcode = check_retcode
        self.shutdown_called = False
        self.order_send_called = False
        self.checked_requests = []

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        self.shutdown_called = True
        return True

    def last_error(self):
        return (0, "OK")

    def version(self):
        return (500, 5833, "test")

    def terminal_info(self):
        return TerminalInfo(True, self.trade_allowed, False, "C:/MT5", 5833)

    def account_info(self):
        return AccountInfo(
            123456,
            "Dukascopy Bank SA",
            "Dukascopy-demo-mt5-1",
            0,
            True,
            True,
            "SGD",
            1000.0,
            1000.0,
            900.0,
            2,
        )

    def symbol_info(self, symbol):
        return SymbolInfo(
            symbol,
            True,
            3,
            0.001,
            100.0,
            0.01,
            0.01,
            10000.0,
            440,
            True,
            4,
            2,
            127,
            2,
            4147.295,
            4146.855,
            "USD",
        )

    def symbol_info_tick(self, symbol):
        return TickInfo(4147.295, 4146.855, 1783356853, 2)

    def order_calc_margin(self, *args, **kwargs):
        if kwargs:
            assert set(kwargs) == {"action", "symbol", "volume", "price"}
        else:
            assert len(args) == 4
        return 55.0

    def order_check(self, *args, **kwargs):
        if kwargs:
            assert set(kwargs) == {"request"}
            request = kwargs["request"]
        else:
            assert len(args) == 1
            request = args[0]
        self.checked_requests.append(dict(request))
        return CheckResult(self.check_retcode, 1000.0, 1000.0, 0.0, 55.0, 945.0, 1818.0, "Done" if self.check_retcode == 0 else "Rejected", request)

    def order_send(self, request):  # pragma: no cover - should never be reachable through proxy
        self.order_send_called = True
        raise AssertionError("order_send must not be called")


class KeywordOnlyTradeCheckFakeMt5(FakeMt5):
    def order_calc_margin(self, *args, **kwargs):
        if args:
            raise TypeError("positional arguments rejected by this fake MT5 build")
        assert set(kwargs) == {"action", "symbol", "volume", "price"}
        return 55.0

    def order_check(self, *args, **kwargs):
        if args:
            raise TypeError("positional arguments rejected by this fake MT5 build")
        assert set(kwargs) == {"request"}
        request = kwargs["request"]
        self.checked_requests.append(dict(request))
        return CheckResult(self.check_retcode, 1000.0, 1000.0, 0.0, 55.0, 945.0, 1818.0, "Done", request)


class KeywordOrderCheckBindsEmptyRequestFakeMt5(FakeMt5):
    def order_check(self, *args, **kwargs):
        if set(kwargs) == {"request"}:
            empty_request = (
                "TradeRequest(action=0, magic=0, order=0, symbol='', volume=0.0, "
                "price=0.0, stoplimit=0.0, sl=0.0, tp=0.0, deviation=0, type=0, "
                "type_filling=0, type_time=0, expiration=0, comment='', position=0, position_by=0)"
            )
            return CheckResult(10013, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "Invalid request", empty_request)
        if kwargs:
            self.checked_requests.append(dict(kwargs))
            return CheckResult(0, 1000.0, 1000.0, 0.0, 55.0, 945.0, 1818.0, "Done", kwargs)
        assert len(args) == 1
        request = args[0]
        self.checked_requests.append(dict(request))
        return CheckResult(0, 1000.0, 1000.0, 0.0, 55.0, 945.0, 1818.0, "Done", request)


def write_config(root: Path) -> None:
    path = root / "config" / "broker_execution_controls_frozen.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FROZEN_YAML, encoding="utf-8")


def test_load_frozen_controls_reads_stage2_yaml(tmp_path: Path):
    write_config(tmp_path)
    controls = load_frozen_controls(tmp_path / "config" / "broker_execution_controls_frozen.yaml")
    assert controls.symbol == "XAUUSD"
    assert controls.stage3_first_order_test_volume_lots == 0.01
    assert controls.stage3_order_check_required_before_order_send is True


def test_volume_step_validation():
    assert is_volume_step_valid(0.01, 0.01, 0.01)
    assert not is_volume_step_valid(0.015, 0.01, 0.01)


def test_proxy_blocks_order_send():
    proxy = GuardedMt5OrderCheckProxy(FakeMt5())
    with pytest.raises(Stage3OrderPreflightError):
        proxy.order_send({})
    assert proxy.forbidden_attempts == ["order_send"]


def test_choose_filling_candidates_prefers_ioc_for_dukas_like_symbol():
    fake = FakeMt5()
    symbol_info = {"filling_mode": 2, "trade_exemode": 2}
    candidates = choose_filling_candidates(fake, symbol_info)
    assert candidates[0] == ("ORDER_FILLING_IOC", fake.ORDER_FILLING_IOC)
    assert all(name != "ORDER_FILLING_RETURN" for name, _ in candidates)


def test_order_preflight_passes_and_does_not_send_order(tmp_path: Path):
    write_config(tmp_path)
    fake = FakeMt5(trade_allowed=True, check_retcode=0)
    report = run_preflight(repo_root=tmp_path, terminal_path="C:/MT5/terminal64.exe", mt5_module=fake)
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["order_check_called"] is True
    assert report["order_send_called"] is False
    assert report["shutdown_called"] is True
    assert fake.shutdown_called is True
    assert fake.order_send_called is False
    assert len(fake.checked_requests) == 2
    assert all("_filling_name" not in request for request in fake.checked_requests)
    assert (tmp_path / "runtime" / "reports" / "stage3_step1_order_permission_preflight.json").exists()




def test_order_preflight_supports_keyword_only_mt5_trade_functions(tmp_path: Path):
    write_config(tmp_path)
    fake = KeywordOnlyTradeCheckFakeMt5(trade_allowed=True, check_retcode=0)
    report = run_preflight(repo_root=tmp_path, terminal_path="C:/MT5/terminal64.exe", mt5_module=fake)
    assert report["status"] == "PASS"
    assert report["validations"]["buy_order_check_passed"] is True
    assert report["validations"]["sell_order_check_passed"] is True
    assert len(fake.checked_requests) == 2
    assert all("_filling_name" not in request for request in fake.checked_requests)


def test_order_preflight_retries_positional_when_keyword_binds_empty_request(tmp_path: Path):
    write_config(tmp_path)
    fake = KeywordOrderCheckBindsEmptyRequestFakeMt5(trade_allowed=True, check_retcode=0)
    report = run_preflight(repo_root=tmp_path, terminal_path="C:/MT5/terminal64.exe", mt5_module=fake)
    assert report["status"] == "PASS"
    assert report["validations"]["buy_order_check_passed"] is True
    assert report["validations"]["sell_order_check_passed"] is True
    assert len(fake.checked_requests) == 2
    assert all(request["symbol"] == "XAUUSD" for request in fake.checked_requests)


def test_order_preflight_fails_when_terminal_trading_disabled(tmp_path: Path):
    write_config(tmp_path)
    fake = FakeMt5(trade_allowed=False, check_retcode=0)
    report = run_preflight(repo_root=tmp_path, terminal_path="C:/MT5/terminal64.exe", mt5_module=fake)
    assert report["status"] == "FAIL"
    assert report["validations"]["terminal_trade_allowed"] is False
    assert report["order_send_called"] is False


def test_order_preflight_fails_when_order_check_rejects(tmp_path: Path):
    write_config(tmp_path)
    fake = FakeMt5(trade_allowed=True, check_retcode=10021)
    report = run_preflight(repo_root=tmp_path, terminal_path="C:/MT5/terminal64.exe", mt5_module=fake)
    assert report["status"] == "FAIL"
    assert report["validations"]["buy_order_check_passed"] is False
    assert report["validations"]["sell_order_check_passed"] is False
