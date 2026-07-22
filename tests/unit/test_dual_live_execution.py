from __future__ import annotations

import pytest

from capstone_trading.runtime.dual_live_execution import (
    DualLiveExecutionError,
    EntrySpreadBlocked,
    execute_transition,
    inspect_broker,
)
from capstone_trading.runtime.order_preflight import FrozenBrokerControls


class FakeMt5:
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    POSITION_TYPE_BUY = 0
    POSITION_TYPE_SELL = 1
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    TRADE_ACTION_DEAL = 1
    ORDER_TIME_GTC = 0
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    SYMBOL_FILLING_FOK = 1
    SYMBOL_FILLING_IOC = 2
    SYMBOL_TRADE_EXECUTION_MARKET = 2
    TRADE_RETCODE_DONE = 10009
    TRADE_RETCODE_DONE_PARTIAL = 10010

    __author__ = "fake"
    __version__ = "0"

    def __init__(self, *, login: int = 1234309, spread: int = 10):
        self.login = login
        self.spread = spread
        self.positions: list[dict] = []
        self.pending: list[dict] = []
        self.next_ticket = 100
        self.shutdown_count = 0

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        self.shutdown_count += 1
        return True

    def last_error(self):
        return (0, "ok")

    def version(self):
        return (5, 0, 6034)

    def terminal_info(self):
        return {
            "connected": True,
            "trade_allowed": True,
            "tradeapi_disabled": False,
        }

    def account_info(self):
        return {
            "login": self.login,
            "trade_mode": self.ACCOUNT_TRADE_MODE_DEMO,
            "company": "Dukascopy Bank SA",
            "server": "Dukascopy-demo-mt5-1",
            "currency": "SGD",
            "balance": 10_000.0,
            "equity": 10_000.0,
            "trade_allowed": True,
            "trade_expert": True,
        }

    def symbol_info(self, symbol):
        return {
            "name": symbol,
            "visible": True,
            "volume_min": 0.01,
            "volume_step": 0.01,
            "volume_max": 100.0,
            "spread": self.spread,
            "filling_mode": self.SYMBOL_FILLING_IOC,
            "trade_exemode": self.SYMBOL_TRADE_EXECUTION_MARKET,
            "trade_contract_size": 100.0,
            "ask": 4000.0,
            "bid": 3999.5,
        }

    def symbol_info_tick(self, symbol):
        return {"symbol": symbol, "ask": 4000.0, "bid": 3999.5}

    def positions_get(self, *, symbol):
        return tuple(item.copy() for item in self.positions if item["symbol"] == symbol)

    def positions_total(self):
        return len(self.positions)

    def orders_get(self, *, symbol):
        return tuple(item.copy() for item in self.pending if item["symbol"] == symbol)

    def orders_total(self):
        return len(self.pending)

    def order_calc_margin(self, *args, **kwargs):
        return 100.0

    def order_check(self, *args, **kwargs):
        request = kwargs.get("request")
        if request is None and kwargs:
            request = kwargs
        if request is None and args:
            request = args[0]
        return {"retcode": 0, "comment": "Done", "request": dict(request)}

    def order_send(self, *args, **kwargs):
        request = kwargs.get("request")
        if request is None and kwargs:
            request = kwargs
        if request is None and args:
            request = args[0]
        request = dict(request)
        self.next_ticket += 1
        if request.get("position"):
            self.positions = [
                item for item in self.positions
                if int(item["ticket"]) != int(request["position"])
            ]
        else:
            position_type = (
                self.POSITION_TYPE_BUY
                if int(request["type"]) == self.ORDER_TYPE_BUY
                else self.POSITION_TYPE_SELL
            )
            self.positions = [
                {
                    "symbol": request["symbol"],
                    "type": position_type,
                    "volume": float(request["volume"]),
                    "ticket": self.next_ticket,
                    "identifier": self.next_ticket + 1000,
                    "order": self.next_ticket,
                    "magic": int(request["magic"]),
                }
            ]
        return {
            "retcode": self.TRADE_RETCODE_DONE,
            "comment": "Done",
            "order": self.next_ticket,
            "deal": self.next_ticket + 5000,
            "request": request,
        }


def controls() -> FrozenBrokerControls:
    return FrozenBrokerControls(
        broker_company_expected="Dukascopy Bank SA",
        server_expected="Dukascopy-demo-mt5-1",
        symbol="XAUUSD",
        timeframe="M15",
        order_volume_lots=0.01,
        max_open_volume_lots_per_model=0.01,
        max_positions_per_model=1,
        max_spread_points_for_entry=800,
        max_deviation_points_for_stage3_order_request=200,
        capstone_leverage_cap=10.0,
        minimum_demo_equity_recommendation_sgd=1000.0,
        stage3_first_order_test_volume_lots=0.01,
        stage3_order_check_required_before_order_send=True,
        model_a_magic_number=26070101,
        model_b_magic_number=26070102,
        model_a_order_comment="CAPSTONE_MODEL_A",
        model_b_order_comment="CAPSTONE_MODEL_B",
        model_b_variant_for_first_controlled_execution="MODEL_B_V2_CURRENT",
        review_usd_sgd_rate_assumption=1.35,
        mt5_server_time_offset_hours_current=3,
        controls_version="test",
    )


def test_inspect_broker_checks_expected_account() -> None:
    fake = FakeMt5(login=1234309)
    result = inspect_broker(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        expected_login_suffix="4309",
        require_trading_permissions=True,
    )
    assert result.snapshot.account_login_masked.endswith("4309")
    assert result.snapshot.positions == ()
    assert fake.shutdown_count == 1


def test_open_long_then_close() -> None:
    fake = FakeMt5(login=1234309)
    opened = execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
    )
    assert opened.broker_position_after == 1
    assert opened.order_send_calls == 1
    assert len(fake.positions) == 1
    closed = execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=0,
        event_time_utc="2026-07-21T16:00:00+00:00",
    )
    assert closed.broker_position_after == 0
    assert closed.order_send_calls == 1
    assert fake.positions == []


def test_model_a_reversal_closes_then_opens() -> None:
    fake = FakeMt5(login=1234309)
    execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
    )
    reversed_result = execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=-1,
        event_time_utc="2026-07-21T16:45:00+00:00",
    )
    assert [leg.purpose for leg in reversed_result.legs] == [
        "CLOSE_LONG",
        "OPEN_SHORT",
    ]
    assert reversed_result.order_send_calls == 2
    assert reversed_result.broker_position_after == -1


def test_model_b_short_is_rejected_before_mt5_send() -> None:
    fake = FakeMt5(login=1230679)
    with pytest.raises(DualLiveExecutionError, match="Model B target cannot be short"):
        execute_transition(
            mt5_module=fake,
            terminal_path="model_b.exe",
            controls=controls(),
            role="model_b",
            expected_login_suffix="0679",
            target_position=-1,
            event_time_utc="2026-07-21T15:45:00+00:00",
        )
    assert fake.positions == []


def test_wide_spread_blocks_entry_but_not_close() -> None:
    fake = FakeMt5(login=1234309, spread=900)
    with pytest.raises(EntrySpreadBlocked):
        execute_transition(
            mt5_module=fake,
            terminal_path="model_a.exe",
            controls=controls(),
            role="model_a",
            expected_login_suffix="4309",
            target_position=1,
            event_time_utc="2026-07-21T15:45:00+00:00",
        )
    fake.spread = 10
    execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=1,
        event_time_utc="2026-07-21T16:00:00+00:00",
    )
    fake.spread = 900
    closed = execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=0,
        event_time_utc="2026-07-21T16:15:00+00:00",
    )
    assert closed.broker_position_after == 0


def test_reversal_spread_jump_after_close_returns_explicit_flat_partial() -> None:
    class JumpingSpreadFake(FakeMt5):
        def __init__(self) -> None:
            super().__init__(login=1234309, spread=10)
            self.symbol_info_calls = 0
            self.jump_on_call: int | None = None

        def symbol_info(self, symbol):
            self.symbol_info_calls += 1
            info = super().symbol_info(symbol)
            if (
                self.jump_on_call is not None
                and self.symbol_info_calls >= self.jump_on_call
            ):
                info["spread"] = 900
            return info

    fake = JumpingSpreadFake()
    execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=1,
        event_time_utc="2026-07-21T15:45:00+00:00",
    )
    fake.symbol_info_calls = 0
    fake.jump_on_call = 4
    partial = execute_transition(
        mt5_module=fake,
        terminal_path="model_a.exe",
        controls=controls(),
        role="model_a",
        expected_login_suffix="4309",
        target_position=-1,
        event_time_utc="2026-07-21T16:45:00+00:00",
    )
    assert partial.completed_target is False
    assert partial.partial_reason == (
        "reversal_entry_spread_blocked_after_confirmed_close"
    )
    assert partial.broker_position_after == 0
    assert [leg.purpose for leg in partial.legs] == ["CLOSE_LONG"]
    assert fake.positions == []
