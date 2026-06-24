from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_dataframe(df: pd.DataFrame, path_without_suffix: Path) -> Path:
    """Save to parquet when possible, otherwise CSV."""
    ensure_dir(path_without_suffix.parent)
    try:
        out = path_without_suffix.with_suffix(".parquet")
        df.to_parquet(out, index=True)
        return out
    except Exception:
        out = path_without_suffix.with_suffix(".csv")
        df.to_csv(out, index=True)
        return out


def save_json(payload: dict[str, Any], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
