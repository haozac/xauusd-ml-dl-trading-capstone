from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts\\inspect_one_dukascopy_csv.py data\\external\\dukascopy\\xauusd_m1_2016-02.csv")
        return

    path = Path(sys.argv[1])
    df = pd.read_csv(path)

    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["time"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("time").reset_index(drop=True)

    print("File:", path.name)
    print("Rows:", len(df))
    print("First:", df["time"].min())
    print("Last:", df["time"].max())
    print("Head:")
    print(df[["time", "open", "high", "low", "close", "volume"]].head(10).to_string(index=False))
    print("Tail:")
    print(df[["time", "open", "high", "low", "close", "volume"]].tail(10).to_string(index=False))

    daily_counts = (
        df.assign(day=df["time"].dt.floor("D"))
          .groupby("day")
          .size()
          .rename("rows")
          .reset_index()
    )

    print("\nDaily row counts:")
    print(daily_counts.to_string(index=False))


if __name__ == "__main__":
    main()