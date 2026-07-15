from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from capstone_trading.runtime.dual_terminal_readiness import (
    DualTerminalConfig,
    DualTerminalReadinessError,
    TerminalRoleConfig,
    build_dual_terminal_report,
)


class NT(SimpleNamespace):
    def _asdict(self):
        return self.__dict__.copy()


class FakeMt5:
    TIMEFRAME_M15 = 15
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2
    __author__ = "MetaQuotes Ltd."
    __version__ = "5.0.test"

    def __init__(self, profiles):
        self.profiles = profiles
        self.current = None
        self.forbidden_called = []

    def initialize(self, path):
        self.current = self.profiles.get(str(path))
        return self.current is not None

    def shutdown(self):
        self.current = None
        return True

    def last_error(self):
        return (1, "ok")

    def version(self):
        return (500, 9999, "test")

    def terminal_info(self):
        p = self.current
        return NT(connected=True, path=str(Path(p["exe"]).parent), data_path=p["data_path"],
                  trade_allowed=True, tradeapi_disabled=False, build=9999)

    def account_info(self):
        p = self.current
        return NT(login=p["login"], trade_mode=0, company="Dukascopy Bank SA",
                  server="Dukascopy-demo-mt5-1", currency="SGD", balance=p["balance"],
                  equity=p["balance"], margin_mode=2, trade_allowed=True, trade_expert=True,
                  leverage=100)

    def symbol_info(self, symbol):
        return NT(name=symbol, visible=True, digits=3, point=0.001, volume_min=0.01,
                  volume_step=0.01, volume_max=10000.0, trade_contract_size=100.0,
                  currency_base="XAU", currency_profit="USD", trade_calc_mode=0,
                  filling_mode=2, ask=4000.8, bid=4000.1, trade_mode=4)

    def symbol_select(self, symbol, selected):
        return True

    def symbol_info_tick(self, symbol):
        return NT(time=1784113200, time_msc=1784113200000, ask=4000.8, bid=4000.1)

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        # Server timestamps are UTC+3. Canonical latest becomes 2026-07-15 10:45 UTC.
        end = int(datetime(2026, 7, 15, 13, 45, tzinfo=timezone.utc).timestamp())
        times = np.arange(end - (count - 1) * 900, end + 1, 900, dtype=np.int64)
        arr = np.zeros(count, dtype=[("time", "i8"), ("open", "f8"), ("high", "f8"),
                                     ("low", "f8"), ("close", "f8"), ("tick_volume", "i8"),
                                     ("spread", "i8"), ("real_volume", "i8")])
        arr["time"] = times
        arr["open"] = 4000.0
        arr["high"] = 4001.0
        arr["low"] = 3999.0
        arr["close"] = 4000.5
        arr["tick_volume"] = 100
        arr["spread"] = 700
        return arr

    def positions_get(self, symbol=None):
        return tuple()

    def orders_get(self, symbol=None):
        return tuple()

    def order_send(self, *args, **kwargs):
        self.forbidden_called.append("order_send")
        raise AssertionError("must not be called")

    def order_check(self, *args, **kwargs):
        self.forbidden_called.append("order_check")
        raise AssertionError("must not be called")


def config(tmp_path: Path, same_path=False, same_login=False):
    a = tmp_path / "A" / "terminal64.exe"
    b = a if same_path else tmp_path / "B" / "terminal64.exe"
    a.parent.mkdir(parents=True, exist_ok=True); a.write_text("x")
    if b != a:
        b.parent.mkdir(parents=True, exist_ok=True); b.write_text("x")
    cfg = DualTerminalConfig(
        model_a=TerminalRoleConfig("MODEL_A", str(a), "runtime/model_a", 26070101, "CAPSTONE_MODEL_A"),
        model_b=TerminalRoleConfig("MODEL_B", str(b), "runtime/model_b", 26070102, "CAPSTONE_MODEL_B"),
    )
    profiles = {
        str(a.resolve()): {"exe": str(a.resolve()), "data_path": str(tmp_path / "dataA"), "login": 111111, "balance": 10000.0},
    }
    if b != a:
        profiles[str(b.resolve())] = {"exe": str(b.resolve()), "data_path": str(tmp_path / "dataB"),
                                      "login": 111111 if same_login else 222222, "balance": 9999.0}
    return cfg, FakeMt5(profiles)


def test_clean_dual_terminal_gate_passes(tmp_path):
    cfg, mt5 = config(tmp_path)
    report = build_dual_terminal_report(
        mt5_module=mt5,
        config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["status"] == "PASS"
    assert report["formal_gate"] is True
    assert report["order_send_called"] is False
    assert report["cross_terminal_review"]["checks"]["accounts_distinct"] is True
    assert mt5.forbidden_called == []


def test_same_terminal_path_fails_before_mt5(tmp_path):
    cfg, mt5 = config(tmp_path, same_path=True)
    with pytest.raises(DualTerminalReadinessError, match="terminal paths must be different"):
        build_dual_terminal_report(mt5_module=mt5, config=cfg)


def test_same_account_fails_formal_gate(tmp_path):
    cfg, mt5 = config(tmp_path, same_login=True)
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["formal_gate"] is False
    assert report["cross_terminal_review"]["checks"]["accounts_distinct"] is False


def test_runtime_roots_must_be_distinct(tmp_path):
    cfg, mt5 = config(tmp_path)
    cfg = DualTerminalConfig(
        model_a=cfg.model_a,
        model_b=TerminalRoleConfig("MODEL_B", cfg.model_b.terminal_path, "runtime/model_a", 26070102, "CAPSTONE_MODEL_B")
    )
    with pytest.raises(DualTerminalReadinessError, match="runtime roots"):
        build_dual_terminal_report(mt5_module=mt5, config=cfg)


def test_magic_numbers_must_be_distinct(tmp_path):
    cfg, mt5 = config(tmp_path)
    cfg = DualTerminalConfig(
        model_a=cfg.model_a,
        model_b=TerminalRoleConfig("MODEL_B", cfg.model_b.terminal_path, "runtime/model_b", 26070101, "CAPSTONE_MODEL_B")
    )
    with pytest.raises(DualTerminalReadinessError, match="magic numbers"):
        build_dual_terminal_report(mt5_module=mt5, config=cfg)


def test_same_data_path_fails_formal_gate(tmp_path):
    cfg, mt5 = config(tmp_path)
    mt5.profiles[str(Path(cfg.model_b.terminal_path).resolve())]["data_path"] = str(tmp_path / "dataA")
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["formal_gate"] is False
    assert report["cross_terminal_review"]["checks"]["terminal_data_paths_distinct"] is False


def test_large_balance_difference_fails_formal_gate(tmp_path):
    cfg, mt5 = config(tmp_path)
    mt5.profiles[str(Path(cfg.model_b.terminal_path).resolve())]["balance"] = 5000.0
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["formal_gate"] is False
    assert report["cross_terminal_review"]["checks"]["starting_balance_difference_within_gate"] is False


def test_low_equity_fails_formal_gate(tmp_path):
    cfg, mt5 = config(tmp_path)
    mt5.profiles[str(Path(cfg.model_b.terminal_path).resolve())]["balance"] = 500.0
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["formal_gate"] is False
    assert report["cross_terminal_review"]["checks"]["model_b_minimum_equity_passed"] is False


def test_open_position_fails_role_gate(tmp_path):
    cfg, mt5 = config(tmp_path)
    original = mt5.positions_get
    def positions_get(symbol=None):
        if mt5.current["login"] == 222222:
            return (NT(ticket=1, symbol=symbol, volume=0.01),)
        return original(symbol=symbol)
    mt5.positions_get = positions_get
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["formal_gate"] is False
    assert report["model_b"]["checks"]["no_open_symbol_positions"] is False


def test_no_forbidden_trade_method_is_used(tmp_path):
    cfg, mt5 = config(tmp_path)
    report = build_dual_terminal_report(
        mt5_module=mt5, config=cfg,
        now_utc=datetime(2026, 7, 15, 11, 0, tzinfo=timezone.utc),
    )
    assert report["model_a"]["order_check_called"] is False
    assert report["model_a"]["order_send_called"] is False
    assert report["model_b"]["order_check_called"] is False
    assert report["model_b"]["order_send_called"] is False
    assert mt5.forbidden_called == []
