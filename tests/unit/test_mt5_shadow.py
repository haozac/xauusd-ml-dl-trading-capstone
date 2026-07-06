from __future__ import annotations

from collections import namedtuple
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from capstone_trading.runtime.mt5_shadow import (
    Mt5ShadowError,
    ShadowRuntimeConfig,
    build_live_feature_frame,
    convert_mt5_rates_to_feature_bars,
    create_latest_shadow_signal,
    load_shadow_state,
    model_a_signal_from_probability,
    normalise_mt5_server_times,
    model_b_from_flat_signal,
    run_shadow_once,
)
from capstone_trading.runtime.mt5_readiness import Mt5RuntimeConfig
from capstone_trading.evaluation.trading_replay import ModelAOverlayRules
from capstone_trading.evaluation.model_b_replay import ModelBOverlayRules

TerminalInfo = namedtuple("TerminalInfo", "connected trade_allowed path name")
AccountInfo = namedtuple("AccountInfo", "login trade_mode server company currency leverage balance equity")
SymbolInfo = namedtuple("SymbolInfo", "name visible digits point volume_min volume_max volume_step spread trade_mode")
TickInfo = namedtuple("TickInfo", "time bid ask last volume time_msc flags volume_real")

FEATURE_ORDER = (
    "minute_of_day_sin",
    "minute_of_day_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
    "is_after_gap",
    "gap_minutes_from_prev_bar",
    "open_rel_prev_close",
    "high_rel_prev_close",
    "low_rel_prev_close",
    "close_rel_prev_close",
    "log_ret_1",
    "log_ret_2",
    "log_ret_3",
    "log_ret_6",
    "log_ret_12",
    "log_ret_24",
    "log_ret_48",
    "open_to_close_pct",
    "high_low_range_pct",
    "upper_wick_pct",
    "lower_wick_pct",
    "close_position_in_bar",
    "close_sma_5_ratio",
    "close_ema_5_ratio",
    "rolling_vol_5",
    "close_sma_10_ratio",
    "close_ema_10_ratio",
    "rolling_vol_10",
    "close_sma_20_ratio",
    "close_ema_20_ratio",
    "rolling_vol_20",
    "close_sma_50_ratio",
    "close_ema_50_ratio",
    "rolling_vol_50",
    "close_sma_100_ratio",
    "close_ema_100_ratio",
    "rolling_vol_100",
    "close_sma_200_ratio",
    "close_ema_200_ratio",
    "rolling_vol_200",
    "roc_5",
    "roc_10",
    "roc_20",
    "rsi_14",
    "true_range_pct",
    "atr_pct_14",
    "bb_width_pct_20",
    "bb_position_20",
    "volume_z20",
)


class FakeScaler:
    def transform(self, frame):
        return np.asarray(frame, dtype=np.float32)


class FakeModelConfig:
    sequence_length = 48
    feature_count = len(FEATURE_ORDER)


class FakeModel:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, batch, training=False):
        return np.asarray([[self.probability]], dtype=np.float32)


class FakeMt5:
    __author__ = "MetaQuotes Software Corp."
    __version__ = "5.0.test"
    TIMEFRAME_M15 = 15
    ACCOUNT_TRADE_MODE_DEMO = 0
    ACCOUNT_TRADE_MODE_CONTEST = 1
    ACCOUNT_TRADE_MODE_REAL = 2

    def __init__(self, *, rates_count: int = 420):
        self.rates_count = rates_count
        self.shutdown_called = False

    def initialize(self, *args, **kwargs):
        return True

    def shutdown(self):
        self.shutdown_called = True
        return None

    def last_error(self):
        return (0, "ok")

    def version(self):
        return (500, 5833, "25 Apr 2026")

    def terminal_info(self):
        return TerminalInfo(True, False, "C:/Program Files/MetaTrader 5", "Fake MT5")

    def account_info(self):
        return AccountInfo(123456789, 0, "Demo-Server", "Broker", "USD", 100, 10000.0, 10000.0)

    def symbol_info(self, symbol):
        if symbol != "XAUUSD":
            return None
        return SymbolInfo(symbol, True, 3, 0.001, 0.01, 10000.0, 0.01, 50, 4)

    def symbol_info_tick(self, symbol):
        return TickInfo(1783000000, 4175.0, 4175.8, 0.0, 0, 1783000000000, 6, 0.0)

    def copy_rates_from_pos(self, symbol, timeframe, start_pos, count):
        assert start_pos == 1
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
        base = 1780000000
        rng = np.random.default_rng(12345)
        closes = 2400.0 + np.cumsum(rng.normal(0.0, 0.35, size=len(rows)))
        for idx in range(len(rows)):
            close = float(closes[idx])
            rows[idx] = (
                base + idx * 900,
                close - 0.1,
                close + 1.0,
                close - 1.0,
                close,
                100 + idx % 17,
                50,
                0,
            )
        return rows


def rules_a() -> ModelAOverlayRules:
    return ModelAOverlayRules(
        long_threshold=0.53,
        short_threshold=0.47,
        minimum_hold_bars=3,
        max_policy_changes_per_day=3,
        daily_loss_log_threshold=-0.020202707317519466,
        total_drawdown_stop=-0.15,
    )


def rules_b() -> ModelBOverlayRules:
    return ModelBOverlayRules(
        entry_threshold=0.55,
        exit_threshold=0.50,
        max_successful_entries_per_day=1,
        daily_loss_log_threshold=-0.020202707317519466,
        total_drawdown_stop=-0.15,
    )


def test_signal_thresholds_are_frozen() -> None:
    assert model_a_signal_from_probability(0.56, rules_a()) == 1
    assert model_a_signal_from_probability(0.44, rules_a()) == -1
    assert model_a_signal_from_probability(0.50, rules_a()) == 0
    assert model_b_from_flat_signal(0.56, rules_b()) == 1
    assert model_b_from_flat_signal(0.54, rules_b()) == 0




def test_normalise_mt5_server_times_converts_dukascopy_summer_offset() -> None:
    canonical_latest = pd.Timestamp("2026-07-06T12:00:00Z")
    raw_server_latest = canonical_latest + pd.Timedelta(hours=3)
    rates = pd.DataFrame(
        {
            "time": [raw_server_latest - pd.Timedelta(minutes=15), raw_server_latest],
            "open": [2400.0, 2401.0],
            "high": [2402.0, 2403.0],
            "low": [2399.0, 2400.0],
            "close": [2401.0, 2402.0],
            "tick_volume": [100, 120],
            "spread": [50, 55],
            "real_volume": [0, 0],
        }
    )
    tick = {"time": int(raw_server_latest.timestamp())}

    normalised, report = normalise_mt5_server_times(
        rates,
        tick,
        server_time_offset_hours=3,
        now_utc=datetime(2026, 7, 6, 12, 30, tzinfo=timezone.utc),
    )

    assert pd.Timestamp(normalised["time"].iloc[-1]) == canonical_latest
    assert report.raw_latest_bar_server_time == raw_server_latest.isoformat()
    assert report.canonical_latest_bar_time_utc == canonical_latest.isoformat()
    assert report.latest_bar_age_minutes_before_conversion == pytest.approx(-150.0)
    assert report.latest_bar_age_minutes_after_conversion == pytest.approx(30.0)
    assert report.latest_bar_future_minutes_after_conversion == 0.0
    assert report.conversion_applied is True

def test_build_live_feature_frame_uses_tick_volume_as_volume() -> None:
    rates = pd.DataFrame(FakeMt5().copy_rates_from_pos("XAUUSD", 15, 1, 420))
    rates["time"] = pd.to_datetime(rates["time"], unit="s", utc=True)
    bars = convert_mt5_rates_to_feature_bars(rates)
    features, report = build_live_feature_frame(bars, FEATURE_ORDER)

    assert len(features) > 180
    assert report.feature_count == len(FEATURE_ORDER)
    assert report.latest_feature_matches_latest_completed_bar is True
    assert "volume_z20" in features.columns


def test_create_latest_shadow_signal_uses_latest_contiguous_window() -> None:
    rates = pd.DataFrame(FakeMt5().copy_rates_from_pos("XAUUSD", 15, 1, 420))
    rates["time"] = pd.to_datetime(rates["time"], unit="s", utc=True)
    bars = convert_mt5_rates_to_feature_bars(rates)
    features, _report = build_live_feature_frame(bars, FEATURE_ORDER)
    scaled = FakeScaler().transform(features)

    signal = create_latest_shadow_signal(
        feature_frame=features,
        scaled_features=scaled,
        model=FakeModel(0.56),
        config=FakeModelConfig(),
        selected_symbol="XAUUSD",
        latest_completed_bar_time=pd.Timestamp(rates["time"].iloc[-1]),
        rules_a=rules_a(),
        rules_b=rules_b(),
    )

    assert signal.probability_up == pytest.approx(0.56, abs=1e-7)
    assert signal.model_a_signal == 1
    assert signal.model_b_from_flat_signal == 1
    assert signal.event_is_latest_completed_bar is True
    assert signal.orders_enabled is False


def test_shadow_state_missing_defaults(tmp_path: Path) -> None:
    state = load_shadow_state(tmp_path / "missing.json")
    assert state["last_event_time_utc"] is None
    assert state["records_written"] == 0


def test_run_shadow_once_appends_first_event_and_skips_duplicate(tmp_path: Path) -> None:
    runtime_config = ShadowRuntimeConfig(
        mt5=Mt5RuntimeConfig(symbol_candidates=("XAUUSD",), bars_to_fetch=420, min_completed_bars=260),
        minimum_feature_rows=180,
        state_path=str(tmp_path / "state.json"),
        signals_csv_path=str(tmp_path / "signals.csv"),
        latest_signal_csv_path=str(tmp_path / "latest.csv"),
    )
    state_path = tmp_path / "state.json"
    signals_path = tmp_path / "signals.csv"

    first_snapshot, _rates, first_signal = run_shadow_once(
        mt5_module=FakeMt5(),
        runtime_config=runtime_config,
        config_a=FakeModelConfig(),
        feature_order=FEATURE_ORDER,
        model=FakeModel(0.56),
        scaler=FakeScaler(),
        rules_a=rules_a(),
        rules_b=rules_b(),
        state_path=state_path,
        signals_csv_path=signals_path,
        mode="once",
        run_id="test",
        now_utc=datetime.fromtimestamp(1784000000, tz=timezone.utc),
    )
    second_snapshot, _rates, second_signal = run_shadow_once(
        mt5_module=FakeMt5(),
        runtime_config=runtime_config,
        config_a=FakeModelConfig(),
        feature_order=FEATURE_ORDER,
        model=FakeModel(0.56),
        scaler=FakeScaler(),
        rules_a=rules_a(),
        rules_b=rules_b(),
        state_path=state_path,
        signals_csv_path=signals_path,
        mode="once",
        run_id="test",
        now_utc=datetime.fromtimestamp(1784000000, tz=timezone.utc),
    )

    assert first_snapshot.status == "PASS"
    assert first_snapshot.shutdown_called is True
    assert first_signal.appended_to_signal_log is True
    assert second_signal.duplicate_event is True
    assert second_signal.appended_to_signal_log is False
    rows = pd.read_csv(signals_path)
    assert len(rows) == 1
