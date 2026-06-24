from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class MT5Config:
    login: int
    password: str
    server: str
    path: str | None = None
    symbol: str = "XAUUSD"


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data_raw: Path
    data_processed: Path
    docs: Path


def get_mt5_config() -> MT5Config:
    login = os.getenv("MT5_LOGIN")
    password = os.getenv("MT5_PASSWORD")
    server = os.getenv("MT5_SERVER")
    path = os.getenv("MT5_PATH") or None
    symbol = os.getenv("MT5_SYMBOL", "XAUUSD")

    missing = [name for name, value in {
        "MT5_LOGIN": login,
        "MT5_PASSWORD": password,
        "MT5_SERVER": server,
    }.items() if not value]

    if missing:
        raise ValueError(
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill it in."
        )

    return MT5Config(
        login=int(login),
        password=password,
        server=server,
        path=path,
        symbol=symbol,
    )


def get_project_paths() -> ProjectPaths:
    root = Path(__file__).resolve().parents[2]
    return ProjectPaths(
        root=root,
        data_raw=root / "data" / "raw",
        data_processed=root / "data" / "processed",
        docs=root / "docs",
    )
