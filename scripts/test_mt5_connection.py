from __future__ import annotations

from pprint import pprint

from fyp_master.config import get_mt5_config
from fyp_master.mt5_client import MT5Client


def main() -> None:
    cfg = get_mt5_config()
    with MT5Client(cfg) as client:
        print("Connected to MT5 successfully.")
        print("Terminal version:", client.terminal_version())
        print("Account info:")
        pprint(client.account_info()._asdict())

        symbol = cfg.symbol
        info = client.symbol_info(symbol)
        if info is None:
            raise SystemExit(f"Symbol '{symbol}' is not available in this terminal.")

        print(f"\nSymbol info for {symbol}:")
        info_dict = info._asdict()
        for key in ["name", "path", "visible", "trade_mode", "digits", "spread", "point"]:
            print(f"  {key}: {info_dict.get(key)}")

        print(f"\nLatest 5 M1 bars for {symbol}:")
        bars = client.latest_bar(symbol, client.mt5.TIMEFRAME_M1, count=5)
        print(bars[["open", "high", "low", "close", "tick_volume", "spread"]])


if __name__ == "__main__":
    main()
