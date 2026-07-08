from __future__ import annotations

from collections import namedtuple
from pathlib import Path

import pytest

from capstone_trading.runtime.order_execution_probe import (
    CONFIRM_SEND_TOKEN,
    GuardedMt5TinyOrderProxy,
    Stage3OrderPreflightError,
    run_tiny_order_test,
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
CheckResult = namedtuple("CheckResult", "retcode balance equity profit margin margin_free margin_level comment request")
SendResult = namedtuple("SendResult", "retcode deal order volume price bid ask comment request")
PositionInfo = namedtuple("PositionInfo", "ticket identifier symbol type volume price_open magic comment")
DealInfo = namedtuple("DealInfo", "ticket order position_id symbol type volume price magic comment profit")
OrderInfo = namedtuple("OrderInfo", "ticket symbol type volume price_open magic comment position_id")


FROZEN_YAML = '''metadata:
  controls_version: "stage2_step3a_v1_0"
broker:
  broker_company_expected: "Dukascopy Bank SA"
  server_expected: "Dukascopy-demo-mt5-1"
  symbol: "XAUUSD"
  timeframe: "M15"
time:
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
  stage3_order_check_required_before_order_send: true
  stage3_first_order_test_volume_lots: 0.01
identifiers:
  model_a_magic_number: 26070101
  model_b_magic_number: 26070102
  model_a_order_comment: "CAPSTONE_MODEL_A"
  model_b_order_comment: "CAPSTONE_MODEL_B"
model_policy:
  model_b_variant_for_first_controlled_execution: "MODEL_B_V2_CURRENT"
review_assumptions:
  review_usd_sgd_rate_assumption: 1.35
'''


class FakeMt5Execution:
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
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010

    def __init__(self, *, existing_position: bool = False, reject_close: bool = False):
        self.shutdown_called = False
        self.positions = []
        self.deals = []
        self.orders_history = []
        self.order_send_calls = []
        self.next_ticket = 900001
        self.reject_close = reject_close
        if existing_position:
            self.positions.append(PositionInfo(1, 1, "XAUUSD", 0, 0.01, 4147.0, 26070102, "MANUAL"))

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        self.shutdown_called = True
        return True

    def last_error(self):
        return (1, "Success")

    def version(self):
        return (500, 5833, "test")

    def terminal_info(self):
        return TerminalInfo(True, True, False, "C:/MT5", 5833)

    def account_info(self):
        return AccountInfo(123456, "Dukascopy Bank SA", "Dukascopy-demo-mt5-1", 0, True, True, "SGD", 10000.0, 10000.0, 9900.0, 2)

    def symbol_info(self, symbol):
        return SymbolInfo(symbol, True, 3, 0.001, 100.0, 0.01, 0.01, 10000.0, 440, True, 4, 2, 127, 2, 4147.865, 4147.205, "USD")

    def symbol_info_tick(self, symbol):
        return TickInfo(4147.865, 4147.205, 1783445778, 6)

    def orders_get(self, *args, **kwargs):
        return tuple()

    def orders_total(self):
        return 0

    def positions_get(self, *args, **kwargs):
        symbol = kwargs.get("symbol") if kwargs else None
        if symbol:
            return tuple(position for position in self.positions if position.symbol == symbol)
        return tuple(self.positions)

    def positions_total(self):
        return len(self.positions)

    def order_calc_margin(self, *args, **kwargs):
        return 107.17

    def order_check(self, *args, **kwargs):
        request = kwargs.get("request") if set(kwargs) == {"request"} else (kwargs if kwargs else args[0])
        return CheckResult(0, 10000.0, 10000.0, 0.0, 107.17, 9892.83, 9330.0, "Done", request)

    def order_send(self, *args, **kwargs):
        request = kwargs.get("request") if set(kwargs) == {"request"} else (kwargs if kwargs else args[0])
        request = dict(request)
        self.order_send_calls.append(request)
        order_type = request["type"]
        if order_type == self.ORDER_TYPE_BUY and "position" not in request:
            ticket = self.next_ticket
            self.next_ticket += 1
            self.positions = [PositionInfo(ticket, ticket, request["symbol"], self.POSITION_TYPE_BUY, request["volume"], request["price"], request["magic"], request["comment"])]
            self.deals.append(DealInfo(ticket + 100, ticket, ticket, request["symbol"], order_type, request["volume"], request["price"], request["magic"], request["comment"], 0.0))
            self.orders_history.append(OrderInfo(ticket, request["symbol"], order_type, request["volume"], request["price"], request["magic"], request["comment"], ticket))
            return SendResult(self.TRADE_RETCODE_DONE, ticket + 100, ticket, request["volume"], request["price"], 0.0, 0.0, "Done", request)
        if order_type == self.ORDER_TYPE_SELL and "position" in request:
            if self.reject_close:
                return SendResult(10021, 0, 0, 0.0, request["price"], 0.0, 0.0, "Rejected", request)
            ticket = request["position"]
            self.positions = []
            self.deals.append(DealInfo(ticket + 200, ticket, ticket, request["symbol"], order_type, request["volume"], request["price"], request["magic"], request["comment"], -1.0))
            self.orders_history.append(OrderInfo(ticket + 1, request["symbol"], order_type, request["volume"], request["price"], request["magic"], request["comment"], ticket))
            return SendResult(self.TRADE_RETCODE_DONE, ticket + 200, ticket, request["volume"], request["price"], 0.0, 0.0, "Done", request)
        return SendResult(10013, 0, 0, 0.0, 0.0, 0.0, 0.0, "Invalid request", request)

    def history_deals_get(self, *args, **kwargs):
        return tuple(self.deals)

    def history_orders_get(self, *args, **kwargs):
        if "ticket" in kwargs:
            return tuple(order for order in self.orders_history if order.ticket == kwargs["ticket"])
        return tuple(self.orders_history)


def make_step1_report(root: Path) -> None:
    path = root / "runtime" / "reports" / "stage3_step1_order_permission_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '{"status":"PASS","formal_gate":true,"order_send_called":false,"orders_executed":false,"decision":{"stage3_step2_tiny_order_test_allowed":true}}',
        encoding="utf-8",
    )


def make_config(root: Path) -> None:
    path = root / "config" / "broker_execution_controls_frozen.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(FROZEN_YAML, encoding="utf-8")


def test_guard_blocks_unauthorised_order_send(tmp_path: Path):
    make_config(tmp_path)
    from capstone_trading.runtime.order_preflight import load_frozen_controls

    controls = load_frozen_controls(tmp_path / "config" / "broker_execution_controls_frozen.yaml")
    proxy = GuardedMt5TinyOrderProxy(FakeMt5Execution(), controls)
    with pytest.raises(Stage3OrderPreflightError):
        proxy.order_send({"symbol": "XAUUSD", "volume": 0.01})


def test_tiny_order_test_opens_and_closes_one_demo_position(tmp_path: Path):
    make_config(tmp_path)
    make_step1_report(tmp_path)
    fake = FakeMt5Execution()
    report = run_tiny_order_test(
        repo_root=tmp_path,
        terminal_path="C:/MT5/terminal64.exe",
        confirmation=CONFIRM_SEND_TOKEN,
        mt5_module=fake,
        timeout_seconds=0.01,
        poll_seconds=0.0,
    )
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["order_send_called"] is True
    assert report["orders_executed"] is True
    assert report["validations"]["no_position_after_close"] is True
    assert len(fake.order_send_calls) == 2
    assert fake.order_send_calls[0]["type"] == fake.ORDER_TYPE_BUY
    assert fake.order_send_calls[1]["type"] == fake.ORDER_TYPE_SELL
    assert fake.positions == []
    assert report["validations"]["history_records_recovered"] is True
    assert len(report["history_deals_filtered"]) >= 2
    assert len(report["history_orders_filtered"]) >= 2
    assert (tmp_path / "runtime" / "reports" / "stage3_step2_v1_1_tiny_order_test.json").exists()
    assert (tmp_path / "runtime" / "reports" / "stage3_step2_v1_1_history_orders.csv").exists()


def test_tiny_order_test_requires_explicit_confirmation(tmp_path: Path):
    make_config(tmp_path)
    make_step1_report(tmp_path)
    with pytest.raises(Stage3OrderPreflightError):
        run_tiny_order_test(
            repo_root=tmp_path,
            terminal_path="C:/MT5/terminal64.exe",
            confirmation="wrong",
            mt5_module=FakeMt5Execution(),
            timeout_seconds=0.01,
            poll_seconds=0.0,
        )


def test_tiny_order_test_refuses_existing_position(tmp_path: Path):
    make_config(tmp_path)
    make_step1_report(tmp_path)
    fake = FakeMt5Execution(existing_position=True)
    report = run_tiny_order_test(
        repo_root=tmp_path,
        terminal_path="C:/MT5/terminal64.exe",
        confirmation=CONFIRM_SEND_TOKEN,
        mt5_module=fake,
        timeout_seconds=0.01,
        poll_seconds=0.0,
    )
    assert report["status"] == "FAIL"
    assert report["order_send_called"] is False
    assert fake.order_send_calls == []
