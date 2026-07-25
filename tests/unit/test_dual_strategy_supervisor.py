from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from capstone_trading.runtime.dual_live_state import write_json_atomic
from capstone_trading.runtime.dual_strategy_supervisor import (
    SupervisorPaths,
    SupervisorSettings,
    WorkerProcess,
    _heartbeat_for_worker,
    _terminate_worker,
    _worker_command,
)


@dataclass
class FakeProcess:
    pid: int
    exit_code: int | None = None

    def poll(self) -> int | None:
        return self.exit_code




@dataclass
class FakeManagedProcess:
    pid: int
    exit_code: int | None = None
    terminate_called: bool = False
    kill_called: bool = False

    def poll(self) -> int | None:
        return self.exit_code

    def wait(self, timeout: int | None = None) -> int:
        self.exit_code = 1
        return self.exit_code

    def terminate(self) -> None:
        self.terminate_called = True
        self.exit_code = 1

    def kill(self) -> None:
        self.kill_called = True
        self.exit_code = 1


def settings(tmp_path: Path, *, orders_enabled: bool = False) -> SupervisorSettings:
    runtime_root = tmp_path / "runtime"
    shared_root = runtime_root / "shared"
    return SupervisorSettings(
        repo_root=tmp_path,
        config_path=tmp_path / "runtime" / "local.yaml",
        execution_mode="live" if orders_enabled else "shadow",
        orders_enabled=orders_enabled,
        duration_hours=1.0,
        poll_seconds=10,
        stale_heartbeat_seconds=180,
        startup_grace_seconds=240,
        shutdown_timeout_seconds=180,
        max_restarts_per_role=3,
        restart_window_minutes=60,
        restart_cooldown_seconds=10,
        worker_script=tmp_path / "scripts" / "worker.py",
        confirmation=(
            "I_UNDERSTAND_DUAL_REHEARSAL_SENDS_DEMO_ORDERS"
            if orders_enabled
            else None
        ),
        paths=SupervisorPaths(
            runtime_root=runtime_root,
            shared_root=shared_root,
            status=shared_root / "supervisor_status.json",
            heartbeat=shared_root / "supervisor_heartbeat.json",
            final_report=shared_root / "supervisor_final_report.json",
            stop_file=shared_root / "STOP",
            kill_switch_file=shared_root / "KILL_SWITCH",
            lock_file=shared_root / "supervisor.lock.json",
            logs_root=runtime_root / "logs",
        ),
    )


def worker(tmp_path: Path, *, role: str, pid: int) -> WorkerProcess:
    return WorkerProcess(
        role=role,
        process=FakeProcess(pid=pid),  # type: ignore[arg-type]
        log_handle=StringIO(),
        log_path=tmp_path / f"{role}.log",
        started_utc="2026-07-21T00:00:00+00:00",
    )


def test_current_generation_heartbeat_accepts_runtime_pid_different_from_launcher(
    tmp_path: Path,
) -> None:
    config = settings(tmp_path)
    current = worker(tmp_path, role="model_a", pid=222)
    current.started_utc = "2026-07-21T00:00:00+00:00"
    heartbeat_path = config.paths.runtime_root / "model_a" / "heartbeat.json"
    write_json_atomic(
        heartbeat_path,
        {
            "run_id": "dual_model_a_current",
            "role": "model_a",
            "pid": 111,
            "status": "RUNNING",
            "started_utc": "2026-07-21T00:00:01+00:00",
            "updated_utc": "2026-07-21T00:00:02+00:00",
        },
    )
    accepted = _heartbeat_for_worker(config, current)
    assert accepted is not None
    assert accepted["pid"] == 111
    assert current.runtime_pid == 111
    assert current.runtime_run_id == "dual_model_a_current"


def test_previous_generation_heartbeat_is_rejected(tmp_path: Path) -> None:
    config = settings(tmp_path)
    current = worker(tmp_path, role="model_a", pid=222)
    current.started_utc = "2026-07-21T00:01:00+00:00"
    heartbeat_path = config.paths.runtime_root / "model_a" / "heartbeat.json"
    write_json_atomic(
        heartbeat_path,
        {
            "run_id": "dual_model_a_previous",
            "role": "model_a",
            "pid": 111,
            "status": "RUNNING",
            "started_utc": "2026-07-21T00:00:00+00:00",
            "updated_utc": "2026-07-21T00:00:30+00:00",
        },
    )
    assert _heartbeat_for_worker(config, current) is None


def test_heartbeat_role_mismatch_is_rejected(tmp_path: Path) -> None:
    config = settings(tmp_path)
    current = worker(tmp_path, role="model_b", pid=333)
    heartbeat_path = config.paths.runtime_root / "model_b" / "heartbeat.json"
    write_json_atomic(
        heartbeat_path,
        {
            "run_id": "dual_model_a_wrong_role",
            "role": "model_a",
            "pid": 444,
            "status": "RUNNING",
            "started_utc": "2026-07-21T00:00:01+00:00",
            "updated_utc": "2026-07-21T00:00:02+00:00",
        },
    )
    assert _heartbeat_for_worker(config, current) is None


def test_shadow_worker_command_cannot_enable_orders(tmp_path: Path) -> None:
    config = settings(tmp_path, orders_enabled=False)
    command = _worker_command(config, "model_a")
    assert "--orders-disabled" in command
    assert "--orders-enabled" not in command
    assert "--confirm-live" not in command


def test_live_worker_command_carries_exact_confirmation(tmp_path: Path) -> None:
    config = settings(tmp_path, orders_enabled=True)
    command = _worker_command(config, "model_b", flatten_only=True)
    assert "--orders-enabled" in command
    assert "--confirm-live" in command
    token_index = command.index("--confirm-live") + 1
    assert command[token_index] == (
        "I_UNDERSTAND_DUAL_REHEARSAL_SENDS_DEMO_ORDERS"
    )
    assert "--flatten-only" in command


def test_shadow_closeout_requires_clean_exit_report_and_broker_flat(
    tmp_path: Path,
) -> None:
    from capstone_trading.runtime.dual_strategy_supervisor import (
        _worker_closeout_review,
    )

    config = settings(tmp_path, orders_enabled=False)
    current = worker(tmp_path, role="model_a", pid=444)
    current.process.exit_code = 0  # type: ignore[attr-defined]
    role_root = config.paths.runtime_root / "model_a"
    write_json_atomic(
        role_root / "final_report.json",
        {
            "status": "PASS",
            "formal_gate": True,
            "state": {
                "broker_position": 0,
                "virtual_position": 1,
            },
        },
    )
    write_json_atomic(
        role_root / "heartbeat.json",
        {
            "run_id": "dual_model_a_closeout",
            "role": "model_a",
            "pid": 777,
            "status": "STOPPED",
            "started_utc": "2026-07-21T00:00:01+00:00",
            "updated_utc": "2026-07-21T00:00:02+00:00",
        },
    )
    review = _worker_closeout_review(config, current)
    assert review["passed"] is True
    assert review["broker_flat"] is True
    assert review["virtual_flat_required"] is False


def test_live_closeout_fails_when_final_state_is_missing(tmp_path: Path) -> None:
    from capstone_trading.runtime.dual_strategy_supervisor import (
        _worker_closeout_review,
    )

    config = settings(tmp_path, orders_enabled=True)
    current = worker(tmp_path, role="model_b", pid=555)
    current.process.exit_code = 0  # type: ignore[attr-defined]
    role_root = config.paths.runtime_root / "model_b"
    write_json_atomic(
        role_root / "final_report.json",
        {
            "status": "PASS",
            "formal_gate": True,
            "state": {},
        },
    )
    write_json_atomic(
        role_root / "heartbeat.json",
        {
            "run_id": "dual_model_b_closeout",
            "role": "model_b",
            "pid": 888,
            "status": "STOPPED",
            "started_utc": "2026-07-21T00:00:01+00:00",
            "updated_utc": "2026-07-21T00:00:02+00:00",
        },
    )
    review = _worker_closeout_review(config, current)
    assert review["passed"] is False
    assert review["broker_flat"] is False
    assert review["virtual_flat"] is False


def test_windows_termination_uses_taskkill_process_tree(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import capstone_trading.runtime.dual_strategy_supervisor as supervisor_module

    process = FakeManagedProcess(pid=999)
    current = WorkerProcess(
        role="model_a",
        process=process,  # type: ignore[arg-type]
        log_handle=StringIO(),
        log_path=tmp_path / "model_a.log",
        started_utc="2026-07-21T00:00:00+00:00",
    )
    calls: list[list[str]] = []

    class Completed:
        returncode = 0
        stdout = "SUCCESS"
        stderr = ""

    def fake_run(command, **kwargs):
        calls.append(list(command))
        process.exit_code = 1
        return Completed()

    monkeypatch.setattr(supervisor_module.os, "name", "nt")
    monkeypatch.setattr(supervisor_module.subprocess, "run", fake_run)

    _terminate_worker(current, timeout_seconds=5)

    assert calls == [["taskkill", "/PID", "999", "/T", "/F"]]
    assert process.terminate_called is False
    assert current.log_handle.closed is True


def test_formal_acceptance_gate_rejects_recovered_restart() -> None:
    from capstone_trading.runtime.dual_strategy_supervisor import (
        formal_acceptance_gate,
    )

    assert formal_acceptance_gate(
        operational_status="PASS",
        restart_events=[],
    ) is True
    assert formal_acceptance_gate(
        operational_status="PASS",
        restart_events=[{"role": "model_b", "reason": "worker_exited_2"}],
    ) is False
