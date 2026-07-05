from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timezone

import numpy as np
import pytest

from capstone_trading.runtime.mt5_readiness import (
    Mt5ReadinessError,
    Mt5RuntimeConfig,
    SafeMt5Proxy,
    analyse_rates,
    fetch_completed_m15_rates,
    resolve_symbol,
    run_mt5_readiness_check,
)

TerminalInfo = namedtuple("TerminalInfo", "connected trade_allowed path name")
AccountInfo = namedtuple("AccountInfo", "login trade_mode server company currency leverage balance equity")
SymbolInfo = namedtuple("SymbolInfo", "name visible digits point volume_min volume_max volume_step spread trade_mode")
TickInfo = namedtuple("TickInfo", "time bid ask last volume time_msc flags volume_real")


class FakeMt5:
    __author__ = "MetaQuotes Software Corp."
    __version__ = "5.0.test"
    TIMEFRAME_M15 = 15
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    def __init__(self, *, account_mode: int = 0, visible: bool = True, rates_count: int = 200):
        self.account_mode = account_mode
        self.visible = visible
        self.rates_count = rates_count
        self.shutdown_called = False
        self.symbol_select_calls = []

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        self.shutdown_called = True
        return None

    def last_error(self):
        return (0, "ok")

    def version(self):
        return (5000, 4200, "01 Jan 2026")

    def terminal_info(self):
        return TerminalInfo(True, False, "C:/MT5/terminal64.exe", "Fake MT5")

    def account_info(self):
        return AccountInfo(123456789, self.account_mode, "Demo-Server", "Broker", "USD", 100, 10000.0, 10000.0)

    def symbol_info(self, symbol):
        if symbol != "XAUUSDm":
            return None
        return SymbolInfo(symbol, self.visible, 2, 0.01, 0.01, 100.0, 0.01, 25, 0)

    def symbol_select(self, symbol, enable):
        self.symbol_select_calls.append((symbol, enable))
        if symbol == "XAUUSDm" and enable:
            self.visible = True
            return True
        return False

    def symbol_info_tick(self, symbol):
        return TickInfo(1760000000, 2400.0, 2400.2, 2400.1, 1, 1760000000000, 0, 1.0)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        assert start_pos == 1
        assert timeframe == self.TIMEFRAME_M15
        dtype = np.dtype(
            [
                ("time", "i8"),
                ("open", "f8"),
                ("high", "f8"),
                ("low", "f8"),
                ("close", "f8"),
                ("tick_volume", "i8"),
                ("spread", "i8"),
                ("real_volume", "i8"),
            ]
        )
        rows = np.zeros(min(count, self.rates_count), dtype=dtype)
        base = 1760000000 - (len(rows) * 900)
        for idx in range(len(rows)):
            close = 2400.0 + idx * 0.1
            rows[idx] = (
                base + idx * 900,
                close,
                close + 1.0,
                close - 1.0,
                close + 0.2,
                100 + idx,
                20,
                0,
            )
        return rows


def test_safe_proxy_blocks_forbidden_trade_api_access() -> None:
    proxy = SafeMt5Proxy(FakeMt5())

    with pytest.raises(Mt5ReadinessError):
        _ = proxy.order_send

    assert proxy.forbidden_attempts == ["order_send"]


def test_resolve_symbol_uses_candidate_order_and_symbol_select() -> None:
    mt5 = SafeMt5Proxy(FakeMt5(visible=False))
    config = Mt5RuntimeConfig(symbol_candidates=("XAUUSD", "XAUUSDm"))

    resolved = resolve_symbol(mt5, config)

    assert resolved.selected_symbol == "XAUUSDm"
    assert resolved.candidates_checked == ("XAUUSD", "XAUUSDm")
    assert resolved.symbol_select_called is True
    assert resolved.symbol_visible_after_select is True


def test_fetch_completed_m15_rates_uses_start_position_one() -> None:
    mt5 = SafeMt5Proxy(FakeMt5())

    frame = fetch_completed_m15_rates(mt5, symbol="XAUUSDm", timeframe_value=15, count=20)

    assert len(frame) == 20
    assert frame["time"].is_monotonic_increasing
    assert "copy_rates_from_pos" in mt5.calls


def test_analyse_rates_rejects_invalid_ohlc() -> None:
    mt5 = SafeMt5Proxy(FakeMt5())
    frame = fetch_completed_m15_rates(mt5, symbol="XAUUSDm", timeframe_value=15, count=20)
    frame.loc[0, "high"] = frame.loc[0, "low"] - 1.0

    with pytest.raises(Mt5ReadinessError):
        analyse_rates(
            frame,
            symbol="XAUUSDm",
            timeframe_name="M15",
            timeframe_value=15,
            requested_bars=20,
            min_completed_bars=10,
            max_latest_closed_bar_age_minutes_warning=1,
            allow_market_closed_stale_bar=True,
            now_utc=datetime.now(timezone.utc),
        )


def test_run_mt5_readiness_check_passes_with_fake_demo_account() -> None:
    result, rates = run_mt5_readiness_check(
        mt5_module=FakeMt5(),
        config=Mt5RuntimeConfig(symbol_candidates=("XAUUSD", "XAUUSDm"), bars_to_fetch=150, min_completed_bars=120),
        now_utc=datetime.fromtimestamp(1760000000, tz=timezone.utc),
    )

    assert result.status == "PASS"
    assert result.orders_enabled is False
    assert result.shutdown_called is True
    assert result.symbol_resolution["selected_symbol"] == "XAUUSDm"
    assert result.rates["uses_completed_bars_only"] is True
    assert result.forbidden_trade_function_calls == ()
    assert len(rates) == 150


def test_run_mt5_readiness_check_rejects_real_account_when_demo_required() -> None:
    with pytest.raises(Mt5ReadinessError, match="not DEMO"):
        run_mt5_readiness_check(
            mt5_module=FakeMt5(account_mode=2),
            config=Mt5RuntimeConfig(symbol_candidates=("XAUUSDm",), require_demo_account=True),
            now_utc=datetime.fromtimestamp(1760000000, tz=timezone.utc),
        )
