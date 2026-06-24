from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from .config import MT5Config


@dataclass
class MT5PullResult:
    symbol: str
    timeframe_name: str
    start_utc: datetime
    end_utc: datetime
    rows: int


class MT5Client:
    def __init__(self, config: MT5Config):
        self.config = config
        self.mt5 = None

    def connect(self) -> None:
        import MetaTrader5 as mt5  # local import so package is only required at runtime

        self.mt5 = mt5
        kwargs: dict[str, Any] = {
            "login": self.config.login,
            "password": self.config.password,
            "server": self.config.server,
        }
        if self.config.path:
            ok = mt5.initialize(self.config.path, **kwargs)
        else:
            ok = mt5.initialize(**kwargs)

        if not ok:
            raise RuntimeError(f"initialize() failed: {mt5.last_error()}")

        # login() is optional if initialize already used login/server/password,
        # but calling it gives a clearer explicit check.
        if not mt5.login(
            self.config.login,
            password=self.config.password,
            server=self.config.server,
        ):
            raise RuntimeError(f"login() failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        if self.mt5 is not None:
            self.mt5.shutdown()

    def __enter__(self) -> "MT5Client":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()

    def terminal_version(self) -> tuple[int, int, str] | None:
        return self.mt5.version()

    def account_info(self):
        return self.mt5.account_info()

    def symbol_info(self, symbol: str):
        return self.mt5.symbol_info(symbol)

    def ensure_symbol(self, symbol: str) -> None:
        info = self.mt5.symbol_info(symbol)
        if info is None:
            raise ValueError(f"Symbol '{symbol}' was not found in MT5.")
        if not info.visible:
            if not self.mt5.symbol_select(symbol, True):
                raise RuntimeError(f"symbol_select() failed for {symbol}: {self.mt5.last_error()}")

    @staticmethod
    def _to_utc(ts: str | pd.Timestamp | datetime) -> datetime:
        if isinstance(ts, str):
            ts = pd.Timestamp(ts)
        elif isinstance(ts, datetime):
            ts = pd.Timestamp(ts)

        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.to_pydatetime()

    def fetch_rates_range(
        self,
        symbol: str,
        timeframe: int,
        start_utc: str | pd.Timestamp | datetime,
        end_utc: str | pd.Timestamp | datetime,
        chunk_days: int = 90,
    ) -> pd.DataFrame:
        self.ensure_symbol(symbol)

        start = self._to_utc(start_utc)
        end = self._to_utc(end_utc)
        if start >= end:
            raise ValueError("start_utc must be earlier than end_utc")

        all_parts: list[pd.DataFrame] = []
        cursor = start

        while cursor < end:
            chunk_end = min(cursor + timedelta(days=chunk_days), end)

            rates = self.mt5.copy_rates_range(
                symbol,
                timeframe,
                int(cursor.timestamp()),
                int(chunk_end.timestamp()),
            )

            if rates is None:
                raise RuntimeError(
                    f"copy_rates_range() failed for {symbol}, {cursor} -> {chunk_end}: "
                    f"{self.mt5.last_error()}"
                )

            part = pd.DataFrame(rates)
            if not part.empty:
                part["time"] = pd.to_datetime(part["time"], unit="s", utc=True)
                all_parts.append(part)

            cursor = chunk_end + timedelta(minutes=1)

        if not all_parts:
            return pd.DataFrame(
                columns=["open", "high", "low", "close", "tick_volume", "spread", "real_volume"]
            )

        df = pd.concat(all_parts, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
        df = df.set_index("time")
        return df

    def latest_bar(self, symbol: str, timeframe: int, count: int = 5) -> pd.DataFrame:
        self.ensure_symbol(symbol)
        rates = self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)
        if rates is None:
            raise RuntimeError(f"copy_rates_from_pos() failed: {self.mt5.last_error()}")
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        return df.set_index("time").sort_index()
