from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from capstone_trading.data.historical_adapter import (
    load_step2_reference_manifest,
    resolve_and_verify_file,
)
from capstone_trading.errors import IntegrityError


def test_step2_reference_manifest_loads_from_repository() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    manifest = load_step2_reference_manifest(repository_root)
    assert manifest.bar_minutes == 15
    assert manifest.sequence_length == 48
    assert manifest.files["model_ready_dataset"].row_count == 237001


def test_resolve_and_verify_file_rejects_tampering(tmp_path: Path) -> None:
    data = tmp_path / "data.bin"
    data.write_bytes(b"original")
    digest = hashlib.sha256(b"original").hexdigest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "AUDITED_REFERENCE",
                "bar_minutes": 15,
                "sequence_length": 48,
                "feature_tolerance": 1e-12,
                "files": {
                    "file": {
                        "path": "data.bin",
                        "sha256": digest,
                        "size_bytes": 8,
                        "row_count": 1,
                        "column_count": 1,
                        "first_timestamp_utc": "2025-01-01T00:00:00+00:00",
                        "last_timestamp_utc": "2025-01-01T00:00:00+00:00",
                    }
                },
                "partitions": {"x": {}},
                "prediction_references": {"x": {}},
            }
        ),
        encoding="utf-8",
    )
    manifest = load_step2_reference_manifest(tmp_path, "manifest.json")
    data.write_bytes(b"tampered")
    with pytest.raises(IntegrityError, match="hash mismatch|size mismatch"):
        resolve_and_verify_file(tmp_path, manifest.files["file"])
