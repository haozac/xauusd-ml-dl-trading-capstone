from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_COLUMNS = ["open", "high", "low", "close", "volume"]
TARGET_COLUMNS = ["target_ret_fwd", "target_dir", "target_class_3"]
RAW_MODEL_EXCLUSIONS = {"open", "high", "low", "close", "volume", "source_m1_bars"}


@dataclass
class Check:
    timeframe: str
    name: str
    status: str
    detail: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Verify M5 and M15 relative-feature datasets before timeframe feasibility modelling."
    )
    parser.add_argument(
        "--m1-master",
        default="data/raw/dukascopy_xauusd_m1_master.parquet",
        help="Path to the aggregated Dukascopy XAUUSD M1 master file.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/capstone_methodology",
        help="Output directory produced by 06_prepare_m5_m15_relative_datasets.py.",
    )
    parser.add_argument("--symbol", default="dukascopy_xauusd")
    parser.add_argument("--timeframes", nargs="+", type=int, default=[5, 15])
    parser.add_argument("--horizon-bars", type=int, default=1)
    parser.add_argument("--neutral-threshold", type=float, default=0.0)
    return parser


def add_check(checks: list[Check], timeframe: str, name: str, passed: bool, detail: str) -> None:
    status = "PASS" if passed else "FAIL"
    checks.append(Check(timeframe=timeframe, name=name, status=status, detail=detail))
    print(f"{status:<5} | {timeframe:<4} | {name} | {detail}")


def load_indexed_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

    if not isinstance(df.index, pd.DatetimeIndex):
        if "time" not in df.columns:
            raise ValueError(f"{path} has no DatetimeIndex or 'time' column.")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time")

    df.index = pd.to_datetime(df.index, utc=True)
    df.index.name = "time"
    return df.sort_index()


def load_m1_master(path: Path) -> pd.DataFrame:
    df = load_indexed_frame(path)
    missing = [column for column in RAW_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"M1 master is missing required columns: {missing}")
    return df[RAW_COLUMNS].sort_index()


def independent_resample(m1: pd.DataFrame, bar_minutes: int) -> pd.DataFrame:
    rule = f"{bar_minutes}min"
    bars = m1.resample(rule, label="right", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    bars["source_m1_bars"] = m1["close"].resample(rule, label="right", closed="left").count()
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars = bars.loc[bars["source_m1_bars"] == bar_minutes].copy()
    bars["source_m1_bars"] = bars["source_m1_bars"].astype("int16")
    return bars


def same_frame(left: pd.DataFrame, right: pd.DataFrame) -> tuple[bool, str]:
    if not left.index.equals(right.index):
        return False, f"timestamp indexes differ: left={len(left):,}, right={len(right):,}"
    if list(left.columns) != list(right.columns):
        return False, f"columns differ: {list(left.columns)} vs {list(right.columns)}"
    try:
        pd.testing.assert_frame_equal(
            left,
            right,
            check_dtype=False,
            check_exact=False,
            rtol=1e-11,
            atol=1e-12,
        )
    except AssertionError as exc:
        return False, str(exc).splitlines()[0][:220]
    return True, "indexes, columns and values match"


def finite_check(df: pd.DataFrame) -> tuple[bool, str]:
    missing_count = int(df.isna().sum().sum())
    numeric = df.select_dtypes(include=["number"]).astype("float64")
    inf_count = int(np.isinf(numeric.to_numpy()).sum())
    return missing_count == 0 and inf_count == 0, f"missing={missing_count}, inf={inf_count}"


def recompute_targets_from_bars(bars: pd.DataFrame, bar_minutes: int, horizon_bars: int) -> pd.Series:
    expected_delta = pd.Timedelta(minutes=bar_minutes * horizon_bars)
    observed_delta = bars.index.to_series().shift(-horizon_bars) - bars.index.to_series()
    target = np.log(bars["close"].shift(-horizon_bars) / bars["close"])
    return target.where(observed_delta == expected_delta)


def verify_timeframe(
    checks: list[Check],
    m1: pd.DataFrame,
    processed_dir: Path,
    symbol: str,
    bar_minutes: int,
    horizon_bars: int,
    neutral_threshold: float,
) -> None:
    timeframe = f"M{bar_minutes}"
    print(f"\n=== {timeframe} ===")

    bars_path = processed_dir / f"{symbol}_m{bar_minutes}_bars_with_volume.parquet"
    price_pre_path = processed_dir / f"{symbol}_m{bar_minutes}_price_only_relative_features_pre_dropna.parquet"
    volume_pre_path = processed_dir / f"{symbol}_m{bar_minutes}_volume_assisted_relative_features_pre_dropna.parquet"
    price_final_path = processed_dir / f"{symbol}_m{bar_minutes}_price_only_relative_dataset.parquet"
    volume_final_path = processed_dir / f"{symbol}_m{bar_minutes}_volume_assisted_relative_dataset.parquet"

    try:
        stored_bars = load_indexed_frame(bars_path)
        price_pre = load_indexed_frame(price_pre_path)
        volume_pre = load_indexed_frame(volume_pre_path)
        price_final = load_indexed_frame(price_final_path)
        volume_final = load_indexed_frame(volume_final_path)
        add_check(checks, timeframe, "outputs_loadable", True, "all required parquet files loaded")
    except Exception as exc:
        add_check(checks, timeframe, "outputs_loadable", False, f"{type(exc).__name__}: {exc}")
        return

    expected_bars = independent_resample(m1, bar_minutes)
    same, detail = same_frame(expected_bars, stored_bars)
    add_check(checks, timeframe, "bars_match_independent_m1_resample", same, detail)

    add_check(
        checks,
        timeframe,
        "all_bars_complete",
        bool((stored_bars["source_m1_bars"] == bar_minutes).all()),
        f"rows={len(stored_bars):,}",
    )

    add_check(
        checks,
        timeframe,
        "variant_indexes_aligned_pre_dropna",
        price_pre.index.equals(volume_pre.index),
        f"price_pre={len(price_pre):,}, volume_pre={len(volume_pre):,}",
    )

    add_check(
        checks,
        timeframe,
        "variant_indexes_aligned_final",
        price_final.index.equals(volume_final.index),
        f"price_final={len(price_final):,}, volume_final={len(volume_final):,}",
    )

    raw_in_price = sorted(RAW_MODEL_EXCLUSIONS.intersection(price_final.columns))
    raw_in_volume = sorted(RAW_MODEL_EXCLUSIONS.intersection(volume_final.columns))
    add_check(
        checks,
        timeframe,
        "raw_ohlcv_excluded_from_model_datasets",
        not raw_in_price and not raw_in_volume,
        f"price_only_raw={raw_in_price}, volume_assisted_raw={raw_in_volume}",
    )

    add_check(
        checks,
        timeframe,
        "volume_feature_policy",
        "volume_z20" not in price_final.columns and "volume_z20" in volume_final.columns,
        f"price_has_volume_z20={'volume_z20' in price_final.columns}, volume_has_volume_z20={'volume_z20' in volume_final.columns}",
    )

    shared_columns = [column for column in price_final.columns if column in volume_final.columns]
    shared_match = all(price_final[column].equals(volume_final[column]) for column in shared_columns)
    add_check(
        checks,
        timeframe,
        "shared_values_match_between_variants",
        shared_match,
        f"shared_columns_checked={len(shared_columns)}",
    )

    price_finite, price_detail = finite_check(price_final)
    volume_finite, volume_detail = finite_check(volume_final)
    add_check(
        checks,
        timeframe,
        "final_datasets_no_nan_or_inf",
        price_finite and volume_finite,
        f"price_only: {price_detail}; volume_assisted: {volume_detail}",
    )

    expected_price_final = price_pre.replace([np.inf, -np.inf], np.nan).dropna().copy()
    expected_volume_final = volume_pre.replace([np.inf, -np.inf], np.nan).dropna().copy()
    # Dtype may differ after target casting, so compare values with relaxed dtype checks.
    same_price, detail_price = same_frame(expected_price_final, price_final)
    same_volume, detail_volume = same_frame(expected_volume_final, volume_final)
    add_check(
        checks,
        timeframe,
        "final_matches_pre_dropna_finalisation",
        same_price and same_volume,
        f"price_only={detail_price}; volume_assisted={detail_volume}",
    )

    expected_targets = recompute_targets_from_bars(stored_bars, bar_minutes, horizon_bars)
    expected_for_final = expected_targets.reindex(price_final.index)
    target_non_missing = expected_for_final.notna().all()
    max_abs_diff = float((price_final["target_ret_fwd"] - expected_for_final).abs().max())
    add_check(
        checks,
        timeframe,
        "targets_do_not_cross_market_gaps",
        bool(target_non_missing),
        f"non_missing_expected_targets={bool(target_non_missing)}",
    )
    add_check(
        checks,
        timeframe,
        "target_returns_recompute_correctly",
        max_abs_diff < 1e-12,
        f"max_abs_diff={max_abs_diff}",
    )

    target_dir_ok = (price_final["target_dir"] == (price_final["target_ret_fwd"] > neutral_threshold).astype("int8")).all()
    add_check(
        checks,
        timeframe,
        "binary_direction_target_consistent",
        bool(target_dir_ok),
        f"target_dir equals target_ret_fwd > neutral_threshold ({neutral_threshold})",
    )


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    processed_dir = output_dir / "processed"
    report_dir = output_dir / "reports" / "data_quality"
    report_dir.mkdir(parents=True, exist_ok=True)
    symbol = args.symbol.lower()

    checks: list[Check] = []

    try:
        m1 = load_m1_master(Path(args.m1_master))
        add_check(checks, "ALL", "m1_master_loadable", True, f"rows={len(m1):,}")
    except Exception as exc:
        add_check(checks, "ALL", "m1_master_loadable", False, f"{type(exc).__name__}: {exc}")
        m1 = None

    if m1 is not None:
        for bar_minutes in args.timeframes:
            verify_timeframe(
                checks=checks,
                m1=m1,
                processed_dir=processed_dir,
                symbol=symbol,
                bar_minutes=bar_minutes,
                horizon_bars=args.horizon_bars,
                neutral_threshold=args.neutral_threshold,
            )

    checklist = pd.DataFrame([asdict(check) for check in checks])
    checklist_path = report_dir / f"{symbol}_m5_m15_relative_pipeline_checklist.csv"
    summary_path = report_dir / f"{symbol}_m5_m15_relative_pipeline_integrity_summary.json"
    checklist.to_csv(checklist_path, index=False)

    failed = checklist[checklist["status"] == "FAIL"]
    final_status = "PASS" if failed.empty else "FAIL"
    summary = {
        "final_status": final_status,
        "checks_run": int(len(checklist)),
        "checks_passed": int((checklist["status"] == "PASS").sum()),
        "checks_failed": int((checklist["status"] == "FAIL").sum()),
        "failed_checks": failed.to_dict(orient="records"),
        "scope": (
            "This verification checks internal consistency from M1 master to M5/M15 complete bars, "
            "relative model datasets, target alignment and feature policy. It does not compare prices "
            "against a second market-data provider."
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n=== FINAL RESULT ===")
    print(f"{final_status}: {summary['checks_passed']}/{summary['checks_run']} checks passed.")
    print(f"Checklist: {checklist_path.resolve()}")
    print(f"Summary:   {summary_path.resolve()}")

    if not failed.empty:
        print("\nFailed checks:")
        print(failed[["timeframe", "name", "detail"]].to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
