"""Safe loading of the audited historical Step 2 parity inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from capstone_trading.artifacts import sha256_file
from capstone_trading.config import safe_repository_path
from capstone_trading.errors import HistoricalDataError, IntegrityError


@dataclass(frozen=True)
class HistoricalFileSpec:
    logical_name: str
    relative_path: Path
    sha256: str
    size_bytes: int
    row_count: int
    column_count: int
    first_timestamp_utc: str
    last_timestamp_utc: str


@dataclass(frozen=True)
class Step2ReferenceManifest:
    source_path: Path
    bar_minutes: int
    sequence_length: int
    feature_tolerance: float
    files: Mapping[str, HistoricalFileSpec]
    partitions: Mapping[str, Mapping[str, Any]]
    prediction_references: Mapping[str, Mapping[str, Any]]


def _load_json_object(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise HistoricalDataError(
            f"Unable to read Step 2 reference manifest {path}: {exc}"
        ) from exc
    if not isinstance(value, Mapping):
        raise HistoricalDataError("Step 2 reference manifest root must be an object")
    return value


def load_step2_reference_manifest(
    repository_root: Path,
    relative_path: str | Path = "config/stage1_step2_reference_manifest.json",
) -> Step2ReferenceManifest:
    path = safe_repository_path(
        repository_root,
        relative_path,
        description="Stage 1 Step 2 reference manifest",
    )
    raw = _load_json_object(path)
    if raw.get("status") != "AUDITED_REFERENCE":
        raise HistoricalDataError("Step 2 reference manifest is not AUDITED_REFERENCE")
    files_raw = raw.get("files")
    if not isinstance(files_raw, Mapping) or not files_raw:
        raise HistoricalDataError("Step 2 reference manifest contains no files")
    files: dict[str, HistoricalFileSpec] = {}
    for logical_name, detail in files_raw.items():
        if not isinstance(logical_name, str) or not isinstance(detail, Mapping):
            raise HistoricalDataError("Invalid Step 2 file entry")
        digest = detail.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise HistoricalDataError(f"Invalid SHA-256 for Step 2 file {logical_name}")
        files[logical_name] = HistoricalFileSpec(
            logical_name=logical_name,
            relative_path=Path(str(detail["path"])),
            sha256=digest.lower(),
            size_bytes=int(detail["size_bytes"]),
            row_count=int(detail["row_count"]),
            column_count=int(detail["column_count"]),
            first_timestamp_utc=str(detail["first_timestamp_utc"]),
            last_timestamp_utc=str(detail["last_timestamp_utc"]),
        )
    partitions = raw.get("partitions")
    predictions = raw.get("prediction_references")
    if not isinstance(partitions, Mapping) or not isinstance(predictions, Mapping):
        raise HistoricalDataError(
            "Step 2 manifest partitions and prediction references are required"
        )
    return Step2ReferenceManifest(
        source_path=path,
        bar_minutes=int(raw["bar_minutes"]),
        sequence_length=int(raw["sequence_length"]),
        feature_tolerance=float(raw["feature_tolerance"]),
        files=files,
        partitions=partitions,
        prediction_references=predictions,
    )


def resolve_and_verify_file(
    repository_root: Path,
    spec: HistoricalFileSpec,
) -> Path:
    path = safe_repository_path(
        repository_root,
        spec.relative_path,
        description=f"Step 2 historical file {spec.logical_name}",
    )
    if not path.is_file():
        raise HistoricalDataError(f"Step 2 historical path is not a file: {path}")
    size = path.stat().st_size
    if size != spec.size_bytes:
        raise IntegrityError(
            f"Step 2 file size mismatch for {spec.logical_name}: "
            f"expected {spec.size_bytes}, found {size}"
        )
    actual = sha256_file(path)
    if actual != spec.sha256:
        raise IntegrityError(
            f"Step 2 hash mismatch for {spec.logical_name}: "
            f"expected {spec.sha256}, found {actual}"
        )
    return path


def load_parquet_reference(path: Path, spec: HistoricalFileSpec) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise HistoricalDataError(f"Unable to read parquet file {path}: {exc}") from exc
    if frame.shape != (spec.row_count, spec.column_count):
        raise HistoricalDataError(
            f"Parquet shape mismatch for {spec.logical_name}: "
            f"expected {(spec.row_count, spec.column_count)}, found {frame.shape}"
        )
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise HistoricalDataError(
            f"{spec.logical_name} must use a timezone-aware DatetimeIndex"
        )
    frame.index = pd.DatetimeIndex(frame.index).tz_convert("UTC")
    frame.index.name = "time"
    if frame.index.min().isoformat() != spec.first_timestamp_utc:
        raise HistoricalDataError(
            f"First timestamp mismatch for {spec.logical_name}: {frame.index.min().isoformat()}"
        )
    if frame.index.max().isoformat() != spec.last_timestamp_utc:
        raise HistoricalDataError(
            f"Last timestamp mismatch for {spec.logical_name}: {frame.index.max().isoformat()}"
        )
    return frame


def load_prediction_reference(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise HistoricalDataError(
            f"Unable to read prediction reference {path}: {exc}"
        ) from exc
    required = ["time", "target_dir", "target_ret_fwd", "p_up"]
    if list(frame.columns) != required:
        raise HistoricalDataError(
            f"Prediction reference columns differ from {required}: {list(frame.columns)}"
        )
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    if frame["time"].isna().any():
        raise HistoricalDataError(
            f"Prediction reference contains invalid timestamps: {path}"
        )
    if frame["time"].duplicated().any() or not frame["time"].is_monotonic_increasing:
        raise HistoricalDataError(
            f"Prediction timestamps are duplicated or unordered: {path}"
        )
    return frame
