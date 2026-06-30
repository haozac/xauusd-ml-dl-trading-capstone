import hashlib
import json
from pathlib import Path

import pytest

from capstone_trading.artifacts import (
    sha256_file,
    verify_model_b_shared_artifacts,
    verify_stage0_freeze_manifest,
)
from capstone_trading.config import load_model_a_config
from capstone_trading.errors import IntegrityError


def test_sha256_file(tmp_path: Path) -> None:
    path = tmp_path / "value.bin"
    path.write_bytes(b"abc")
    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()


def test_stage0_manifest_passes_for_frozen_files() -> None:
    results = verify_stage0_freeze_manifest(Path.cwd())
    assert len(results) == 4
    assert all(item.passed for item in results)


def test_stage0_manifest_detects_tampering(tmp_path: Path) -> None:
    (tmp_path / "config").mkdir()
    target = tmp_path / "config" / "frozen.yaml"
    target.write_text("original", encoding="utf-8")
    manifest = {
        "status": "FROZEN_STAGE_0",
        "hash_algorithm": "SHA-256",
        "files": {
            "config/frozen.yaml": {
                "sha256": hashlib.sha256(b"expected-other-content").hexdigest()
            }
        },
    }
    (tmp_path / "config" / "stage0_freeze_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    with pytest.raises(IntegrityError, match="hash mismatch"):
        verify_stage0_freeze_manifest(tmp_path)


def test_model_b_uses_exact_model_a_shared_artifacts() -> None:
    model_a = load_model_a_config("config/model_a_frozen.yaml")
    verify_model_b_shared_artifacts(Path("config/model_b_v2_frozen.yaml"), model_a)
