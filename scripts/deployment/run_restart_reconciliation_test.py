#!/usr/bin/env python
"""Deliberately terminate one shadow worker and verify supervised recovery.

The first authorised use is orders-disabled shadow mode.  It verifies that the
supervisor starts a new process, the persistent state records a restart, the
broker reconciliation passes, and no order-send counter changes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import signal
import subprocess
import time


class RestartTestError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        raise RestartTestError(f"Unable to read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise RestartTestError(f"Expected JSON object: {path}")
    return raw


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def terminate_pid(pid: int) -> None:
    if os.name == "nt":
        completed = subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RestartTestError(
                f"taskkill failed for PID {pid}: {completed.stdout} {completed.stderr}"
            )
    else:
        os.kill(pid, signal.SIGKILL)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--runtime-root", default="runtime/dual_live_rehearsal_shadow")
    parser.add_argument("--role", choices=("model_a", "model_b"), required=True)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument(
        "--confirm",
        required=True,
        help="Must equal I_UNDERSTAND_THIS_KILLS_ONE_SHADOW_WORKER",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.confirm != "I_UNDERSTAND_THIS_KILLS_ONE_SHADOW_WORKER":
        print("ERROR exact confirmation token not supplied")
        return 2
    repo_root = args.repo_root.expanduser().resolve()
    runtime_root = (repo_root / args.runtime_root).resolve()
    shared = runtime_root / "shared"
    status_path = shared / "supervisor_status.json"
    state_path = runtime_root / args.role / "state.json"
    heartbeat_path = runtime_root / args.role / "heartbeat.json"
    report_path = shared / f"restart_reconciliation_{args.role}.json"
    before_status = load_json(status_path)
    if before_status.get("status") != "RUNNING":
        raise RestartTestError("Supervisor is not RUNNING")
    if before_status.get("orders_enabled") is not False:
        raise RestartTestError(
            "This first restart test is authorised only with orders disabled"
        )
    before_worker = before_status.get("workers", {}).get(args.role, {})
    before_pid = int(before_worker.get("pid", 0) or 0)
    if before_pid <= 0 or before_worker.get("running") is not True:
        raise RestartTestError("Selected worker is not running")
    before_state = load_json(state_path)
    before_restart_count = int(before_state.get("restart_count", 0) or 0)
    before_order_send_calls = int(before_state.get("order_send_calls", 0) or 0)
    before_last_event = before_state.get("last_event_time_utc")
    started_utc = datetime.now(timezone.utc).isoformat()
    terminate_pid(before_pid)
    deadline = time.time() + max(30, args.timeout_seconds)
    after_status: dict[str, Any] | None = None
    after_state: dict[str, Any] | None = None
    while time.time() < deadline:
        time.sleep(5)
        try:
            candidate_status = load_json(status_path)
            candidate_state = load_json(state_path)
            heartbeat = load_json(heartbeat_path)
        except Exception:
            continue
        worker = candidate_status.get("workers", {}).get(args.role, {})
        new_pid = int(worker.get("pid", 0) or 0)
        restarted = new_pid > 0 and new_pid != before_pid and worker.get("running") is True
        count_advanced = int(candidate_state.get("restart_count", 0) or 0) > before_restart_count
        heartbeat_good = heartbeat.get("status") in {"STARTING", "RUNNING"}
        no_order_change = int(candidate_state.get("order_send_calls", 0) or 0) == before_order_send_calls
        if restarted and count_advanced and heartbeat_good and no_order_change:
            after_status = candidate_status
            after_state = candidate_state
            break
    if after_status is None or after_state is None:
        raise RestartTestError("Worker did not satisfy restart checks before timeout")
    after_worker = after_status["workers"][args.role]
    report = {
        "schema_version": "1.0",
        "test": "shadow_worker_restart_and_reconciliation",
        "status": "PASS",
        "formal_gate": True,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "orders_enabled": False,
        "before": {
            "pid": before_pid,
            "restart_count": before_restart_count,
            "order_send_calls": before_order_send_calls,
            "last_event_time_utc": before_last_event,
        },
        "after": {
            "pid": int(after_worker["pid"]),
            "restart_count": int(after_state.get("restart_count", 0) or 0),
            "order_send_calls": int(after_state.get("order_send_calls", 0) or 0),
            "last_event_time_utc": after_state.get("last_event_time_utc"),
            "reconciliation_status": after_state.get("reconciliation_status"),
        },
        "validations": {
            "pid_changed": int(after_worker["pid"]) != before_pid,
            "restart_count_increased": int(after_state.get("restart_count", 0) or 0) > before_restart_count,
            "order_send_count_unchanged": int(after_state.get("order_send_calls", 0) or 0) == before_order_send_calls,
            "supervisor_still_running": after_status.get("status") == "RUNNING",
            "broker_reconciliation_recorded": str(after_state.get("reconciliation_status", "")).startswith("PASS_"),
        },
    }
    if not all(report["validations"].values()):
        report["status"] = "FAIL"
        report["formal_gate"] = False
    write_json(report_path, report)
    print("Restart reconciliation status:", report["status"])
    print("Role:", args.role)
    print("Old PID:", before_pid)
    print("New PID:", report["after"]["pid"])
    print("Restart count:", report["after"]["restart_count"])
    print("Order-send count unchanged:", report["validations"]["order_send_count_unchanged"])
    print("Reconciliation status:", report["after"]["reconciliation_status"])
    print("Report:", report_path)
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
