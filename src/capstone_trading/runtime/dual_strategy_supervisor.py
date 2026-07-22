"""Process supervisor for the two independent MT5 rehearsal workers.

The supervisor is intentionally separate from MetaTrader.  It never imports the
MetaTrader5 package and therefore cannot place an order itself.  Its duties are
limited to process lifecycle, heartbeat monitoring, restart budgeting, stop and
kill-switch files, and evidence persistence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import os
import subprocess
import sys
import time
import traceback

from capstone_trading.config import load_yaml_mapping, safe_repository_path
from capstone_trading.runtime.dual_live_state import utc_now_iso, write_json_atomic
LIVE_CONFIRMATION_TOKEN = "I_UNDERSTAND_DUAL_REHEARSAL_SENDS_DEMO_ORDERS"


class DualSupervisorError(RuntimeError):
    """Raised when the dual supervisor cannot continue safely."""


@dataclass(frozen=True)
class SupervisorPaths:
    runtime_root: Path
    shared_root: Path
    status: Path
    heartbeat: Path
    final_report: Path
    stop_file: Path
    kill_switch_file: Path
    lock_file: Path
    logs_root: Path


@dataclass(frozen=True)
class SupervisorSettings:
    repo_root: Path
    config_path: Path
    execution_mode: str
    orders_enabled: bool
    duration_hours: float
    poll_seconds: int
    stale_heartbeat_seconds: int
    startup_grace_seconds: int
    shutdown_timeout_seconds: int
    max_restarts_per_role: int
    restart_window_minutes: int
    restart_cooldown_seconds: int
    worker_script: Path
    confirmation: str | None
    paths: SupervisorPaths


@dataclass
class WorkerProcess:
    role: str
    process: subprocess.Popen[str]
    log_handle: Any
    log_path: Path
    started_utc: str
    starts: int = 1
    restart_timestamps: list[str] = field(default_factory=list)
    last_exit_code: int | None = None
    last_failure_reason: str | None = None
    runtime_pid: int | None = None
    runtime_run_id: str | None = None


def _mapping(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, Mapping):
        raise DualSupervisorError(f"Config section {key!r} must be a mapping")
    return value


def load_supervisor_settings(
    *,
    repo_root: Path,
    config_path: Path,
    execution_mode_override: str | None,
    orders_enabled_override: bool | None,
    duration_hours_override: float | None,
    confirmation: str | None,
) -> SupervisorSettings:
    raw = load_yaml_mapping(config_path)
    runtime = _mapping(raw, "runtime")
    supervisor = _mapping(raw, "supervisor")
    paths = _mapping(raw, "paths")
    execution_mode = str(
        execution_mode_override or runtime.get("execution_mode", "shadow")
    ).lower()
    if execution_mode not in {"shadow", "live"}:
        raise DualSupervisorError("execution_mode must be shadow or live")
    orders_enabled = (
        bool(runtime.get("orders_enabled", False))
        if orders_enabled_override is None
        else bool(orders_enabled_override)
    )
    if execution_mode == "shadow" and orders_enabled:
        raise DualSupervisorError("shadow mode cannot enable orders")
    if orders_enabled and confirmation != LIVE_CONFIRMATION_TOKEN:
        raise DualSupervisorError(
            "Live order mode requires the exact dual-rehearsal confirmation token"
        )
    duration_hours = (
        float(runtime.get("duration_hours", 24.0))
        if duration_hours_override is None
        else float(duration_hours_override)
    )
    if duration_hours <= 0:
        raise DualSupervisorError("duration_hours must be positive")
    runtime_root = safe_repository_path(
        repo_root,
        str(paths.get("runtime_root", "runtime/dual_live_rehearsal")),
        description="dual live runtime root",
        must_exist=False,
    )
    shared = runtime_root / "shared"
    worker_script = safe_repository_path(
        repo_root,
        str(
            paths.get(
                "worker_script",
                "scripts/deployment/run_dual_strategy_worker.py",
            )
        ),
        description="dual strategy worker script",
    )
    return SupervisorSettings(
        repo_root=repo_root,
        config_path=config_path,
        execution_mode=execution_mode,
        orders_enabled=orders_enabled,
        duration_hours=duration_hours,
        poll_seconds=max(2, int(supervisor.get("poll_seconds", 10))),
        stale_heartbeat_seconds=max(
            30,
            int(supervisor.get("stale_heartbeat_seconds", 180)),
        ),
        startup_grace_seconds=max(
            30,
            int(supervisor.get("startup_grace_seconds", 180)),
        ),
        shutdown_timeout_seconds=max(
            15,
            int(supervisor.get("shutdown_timeout_seconds", 120)),
        ),
        max_restarts_per_role=max(
            0,
            int(supervisor.get("max_restarts_per_role", 3)),
        ),
        restart_window_minutes=max(
            1,
            int(supervisor.get("restart_window_minutes", 60)),
        ),
        restart_cooldown_seconds=max(
            1,
            int(supervisor.get("restart_cooldown_seconds", 10)),
        ),
        worker_script=worker_script,
        confirmation=confirmation,
        paths=SupervisorPaths(
            runtime_root=runtime_root,
            shared_root=shared,
            status=shared / "supervisor_status.json",
            heartbeat=shared / "supervisor_heartbeat.json",
            final_report=shared / "supervisor_final_report.json",
            stop_file=shared / "STOP",
            kill_switch_file=shared / "KILL_SWITCH",
            lock_file=shared / "supervisor.lock.json",
            logs_root=runtime_root / "logs",
        ),
    )


def _parse_time(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    return raw if isinstance(raw, dict) else None


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def acquire_lock(settings: SupervisorSettings) -> None:
    settings.paths.shared_root.mkdir(parents=True, exist_ok=True)
    existing = _read_json(settings.paths.lock_file)
    if existing:
        old_pid = int(existing.get("pid", 0) or 0)
        if _pid_alive(old_pid):
            raise DualSupervisorError(
                f"Another supervisor is already running with PID {old_pid}"
            )
    write_json_atomic(
        settings.paths.lock_file,
        {
            "schema_version": "1.0",
            "pid": os.getpid(),
            "created_utc": utc_now_iso(),
            "config_path": str(settings.config_path),
        },
    )


def release_lock(settings: SupervisorSettings) -> None:
    try:
        current = _read_json(settings.paths.lock_file)
        if current and int(current.get("pid", 0) or 0) == os.getpid():
            settings.paths.lock_file.unlink(missing_ok=True)
    except Exception:
        pass


def _worker_command(settings: SupervisorSettings, role: str, *, flatten_only: bool = False) -> list[str]:
    command = [
        sys.executable,
        "-u",
        str(settings.worker_script),
        "--repo-root",
        str(settings.repo_root),
        "--config",
        str(settings.config_path),
        "--role",
        role,
        "--execution-mode",
        settings.execution_mode,
    ]
    if settings.orders_enabled:
        command.extend(
            [
                "--orders-enabled",
                "--confirm-live",
                str(settings.confirmation),
            ]
        )
    else:
        command.append("--orders-disabled")
    if flatten_only:
        command.append("--flatten-only")
    return command


def _launch_worker(settings: SupervisorSettings, role: str, *, starts: int, restarts: list[str]) -> WorkerProcess:
    settings.paths.logs_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = settings.paths.logs_root / f"{role}_{stamp}_start{starts}.log"
    handle = log_path.open("a", encoding="utf-8", buffering=1)
    command = _worker_command(settings, role)
    creationflags = 0
    if os.name == "nt":
        creationflags = int(getattr(subprocess, "CREATE_NO_WINDOW", 0))
    launch_started_utc = utc_now_iso()
    process = subprocess.Popen(
        command,
        cwd=str(settings.repo_root),
        stdin=subprocess.DEVNULL,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        creationflags=creationflags,
    )
    return WorkerProcess(
        role=role,
        process=process,
        log_handle=handle,
        log_path=log_path,
        started_utc=launch_started_utc,
        starts=starts,
        restart_timestamps=list(restarts),
    )


def _close_handle(worker: WorkerProcess) -> None:
    try:
        worker.log_handle.flush()
        worker.log_handle.close()
    except Exception:
        pass


def _terminate_worker(worker: WorkerProcess, timeout_seconds: int) -> None:
    if worker.process.poll() is not None:
        _close_handle(worker)
        return
    try:
        if os.name == "nt":
            completed = subprocess.run(
                [
                    "taskkill",
                    "/PID",
                    str(worker.process.pid),
                    "/T",
                    "/F",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0 and worker.process.poll() is None:
                raise DualSupervisorError(
                    "Unable to terminate Windows worker process tree "
                    f"for {worker.role} launcher PID {worker.process.pid}: "
                    f"{completed.stdout} {completed.stderr}"
                )
        else:
            worker.process.terminate()
        try:
            worker.process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            if os.name == "nt":
                completed = subprocess.run(
                    [
                        "taskkill",
                        "/PID",
                        str(worker.process.pid),
                        "/T",
                        "/F",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if completed.returncode != 0 and worker.process.poll() is None:
                    raise DualSupervisorError(
                        "Windows worker process tree remained alive after timeout "
                        f"for {worker.role} launcher PID {worker.process.pid}: "
                        f"{completed.stdout} {completed.stderr}"
                    )
            else:
                worker.process.kill()
            worker.process.wait(timeout=30)
    finally:
        _close_handle(worker)


def _heartbeat_for(settings: SupervisorSettings, role: str) -> dict[str, Any] | None:
    return _read_json(settings.paths.runtime_root / role / "heartbeat.json")


def _heartbeat_for_worker(
    settings: SupervisorSettings,
    worker: WorkerProcess,
) -> dict[str, Any] | None:
    """Return only a heartbeat from the current worker generation.

    On Windows, ``venv/Scripts/python.exe`` can act as a launcher whose PID
    differs from ``os.getpid()`` inside the actual interpreter.  Therefore the
    supervisor must not require the launcher PID and heartbeat PID to be equal.
    Ownership is established by role plus a heartbeat ``started_utc`` that is
    not older than the supervised launch generation.  Stale heartbeats from a
    previous worker generation remain rejected.
    """

    heartbeat = _heartbeat_for(settings, worker.role)
    if not heartbeat:
        return None
    if str(heartbeat.get("role", "")) != worker.role:
        return None
    heartbeat_started = _parse_time(heartbeat.get("started_utc"))
    worker_started = _parse_time(worker.started_utc)
    if heartbeat_started is None or worker_started is None:
        return None
    if heartbeat_started < worker_started:
        return None
    try:
        runtime_pid = int(heartbeat.get("pid", 0) or 0)
    except (TypeError, ValueError):
        return None
    if runtime_pid <= 0:
        return None
    worker.runtime_pid = runtime_pid
    run_id = str(heartbeat.get("run_id", "") or "").strip()
    worker.runtime_run_id = run_id or None
    return heartbeat


def _heartbeat_age_seconds(payload: Mapping[str, Any] | None) -> float | None:
    if not payload:
        return None
    timestamp = _parse_time(payload.get("updated_utc"))
    if timestamp is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - timestamp).total_seconds())


def _recent_restart_count(worker: WorkerProcess, settings: SupervisorSettings) -> int:
    cutoff = datetime.now(timezone.utc) - timedelta(
        minutes=settings.restart_window_minutes
    )
    recent: list[str] = []
    for value in worker.restart_timestamps:
        parsed = _parse_time(value)
        if parsed is not None and parsed >= cutoff:
            recent.append(value)
    worker.restart_timestamps = recent
    return len(recent)


def _worker_view(settings: SupervisorSettings, worker: WorkerProcess) -> dict[str, Any]:
    heartbeat = _heartbeat_for_worker(settings, worker)
    return {
        "role": worker.role,
        "pid": worker.process.pid,
        "launcher_pid": worker.process.pid,
        "runtime_pid": (
            None if heartbeat is None else int(heartbeat.get("pid", 0) or 0)
        ),
        "runtime_run_id": (
            None if heartbeat is None else heartbeat.get("run_id")
        ),
        "running": worker.process.poll() is None,
        "exit_code": worker.process.poll(),
        "started_utc": worker.started_utc,
        "starts": worker.starts,
        "restart_timestamps": list(worker.restart_timestamps),
        "log_path": str(worker.log_path),
        "heartbeat": heartbeat,
        "heartbeat_age_seconds": _heartbeat_age_seconds(heartbeat),
        "last_failure_reason": worker.last_failure_reason,
    }


def _write_supervisor_status(
    settings: SupervisorSettings,
    *,
    run_id: str,
    started_utc: str,
    status: str,
    message: str,
    workers: Mapping[str, WorkerProcess],
    restart_events: list[Mapping[str, Any]],
) -> None:
    payload = {
        "schema_version": "1.0",
        "run_id": run_id,
        "pid": os.getpid(),
        "status": status,
        "message": message,
        "started_utc": started_utc,
        "updated_utc": utc_now_iso(),
        "execution_mode": settings.execution_mode,
        "orders_enabled": settings.orders_enabled,
        "duration_hours": settings.duration_hours,
        "stop_file_exists": settings.paths.stop_file.exists(),
        "kill_switch_exists": settings.paths.kill_switch_file.exists(),
        "workers": {
            role: _worker_view(settings, worker)
            for role, worker in workers.items()
        },
        "restart_events": restart_events,
    }
    write_json_atomic(settings.paths.status, payload)
    write_json_atomic(
        settings.paths.heartbeat,
        {
            "schema_version": "1.0",
            "run_id": run_id,
            "pid": os.getpid(),
            "status": status,
            "message": message,
            "updated_utc": payload["updated_utc"],
            "orders_enabled": settings.orders_enabled,
        },
    )




def _worker_closeout_review(
    settings: SupervisorSettings,
    worker: WorkerProcess,
) -> dict[str, Any]:
    """Review process exit, final report, heartbeat, and flat broker state."""

    final_path = settings.paths.runtime_root / worker.role / "final_report.json"
    heartbeat_path = settings.paths.runtime_root / worker.role / "heartbeat.json"
    final_report = _read_json(final_path)
    heartbeat = _heartbeat_for_worker(settings, worker)
    exit_code = worker.process.poll()
    state = (final_report or {}).get("state", {})
    if not isinstance(state, Mapping):
        state = {}
    broker_value = state.get("broker_position")
    virtual_value = state.get("virtual_position")
    try:
        broker_flat = broker_value is not None and int(broker_value) == 0
    except (TypeError, ValueError):
        broker_flat = False
    virtual_flat_required = bool(
        settings.execution_mode == "live" and settings.orders_enabled
    )
    try:
        virtual_flat = virtual_value is not None and int(virtual_value) == 0
    except (TypeError, ValueError):
        virtual_flat = False
    heartbeat_stopped = bool(
        heartbeat
        and heartbeat.get("role") == worker.role
        and heartbeat.get("status") == "STOPPED"
    )
    passed = bool(
        exit_code == 0
        and final_report
        and final_report.get("status") == "PASS"
        and final_report.get("formal_gate") is True
        and broker_flat
        and (virtual_flat or not virtual_flat_required)
        and heartbeat_stopped
    )
    return {
        "role": worker.role,
        "passed": passed,
        "process_exit_code": exit_code,
        "final_report_path": str(final_path),
        "final_report_available": final_report is not None,
        "final_report_status": None if final_report is None else final_report.get("status"),
        "final_report_formal_gate": (
            None if final_report is None else final_report.get("formal_gate")
        ),
        "broker_flat": broker_flat,
        "virtual_flat": virtual_flat,
        "virtual_flat_required": virtual_flat_required,
        "heartbeat_path": str(heartbeat_path),
        "heartbeat_stopped": heartbeat_stopped,
    }


def _run_emergency_flatten(settings: SupervisorSettings, role: str) -> dict[str, Any]:
    if not settings.orders_enabled:
        return {"role": role, "attempted": False, "reason": "orders_disabled"}
    log_path = settings.paths.logs_root / f"{role}_emergency_flatten.log"
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        completed = subprocess.run(
            _worker_command(settings, role, flatten_only=True),
            cwd=str(settings.repo_root),
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=180,
            check=False,
        )
    return {
        "role": role,
        "attempted": True,
        "exit_code": completed.returncode,
        "passed": completed.returncode == 0,
        "log_path": str(log_path),
    }


def run_supervisor(settings: SupervisorSettings, *, reset_control_files: bool) -> dict[str, Any]:
    settings.paths.shared_root.mkdir(parents=True, exist_ok=True)
    settings.paths.logs_root.mkdir(parents=True, exist_ok=True)
    if reset_control_files:
        settings.paths.stop_file.unlink(missing_ok=True)
        settings.paths.kill_switch_file.unlink(missing_ok=True)
    elif settings.paths.stop_file.exists() or settings.paths.kill_switch_file.exists():
        raise DualSupervisorError(
            "STOP or KILL_SWITCH exists. Review it, then restart with "
            "--reset-control-files only when both accounts are safe."
        )
    acquire_lock(settings)
    run_id = datetime.now(timezone.utc).strftime("dual_supervisor_%Y%m%dT%H%M%SZ")
    started = datetime.now(timezone.utc)
    started_utc = started.isoformat()
    deadline = started + timedelta(hours=settings.duration_hours)
    workers: dict[str, WorkerProcess] = {}
    restart_events: list[dict[str, Any]] = []
    emergency_flatten: list[dict[str, Any]] = []
    final_status = "FAIL"
    final_message = "supervisor ended unexpectedly"
    try:
        for role in ("model_a", "model_b"):
            workers[role] = _launch_worker(
                settings,
                role,
                starts=1,
                restarts=[],
            )
        while True:
            now = datetime.now(timezone.utc)
            if now >= deadline and not settings.paths.stop_file.exists():
                settings.paths.stop_file.write_text(
                    f"duration_complete_utc={utc_now_iso()}\n",
                    encoding="utf-8",
                )
            if settings.paths.kill_switch_file.exists():
                if not settings.paths.stop_file.exists():
                    settings.paths.stop_file.write_text(
                        (
                            f"created_utc={utc_now_iso()}\n"
                            "reason=kill_switch_detected_by_supervisor\n"
                        ),
                        encoding="utf-8",
                    )
                final_status = "FAIL"
                final_message = "kill switch detected; clean closeout requested"
                break
            if settings.paths.stop_file.exists():
                final_status = "PASS"
                final_message = "stop requested or duration completed"
                break
            failure: tuple[str, str] | None = None
            for role, worker in workers.items():
                exit_code = worker.process.poll()
                if exit_code is not None:
                    failure = (role, f"worker_exited_{exit_code}")
                    worker.last_exit_code = exit_code
                    break
                age = _heartbeat_age_seconds(
                    _heartbeat_for_worker(settings, worker)
                )
                started_time = _parse_time(worker.started_utc) or now
                startup_age = (now - started_time).total_seconds()
                if age is None:
                    if startup_age > settings.startup_grace_seconds:
                        failure = (role, "heartbeat_missing_after_startup_grace")
                        break
                elif age > settings.stale_heartbeat_seconds:
                    failure = (role, f"heartbeat_stale_{age:.1f}_seconds")
                    break
            if failure is not None:
                role, reason = failure
                worker = workers[role]
                worker.last_failure_reason = reason
                _terminate_worker(worker, settings.shutdown_timeout_seconds)
                recent_count = _recent_restart_count(worker, settings)
                if recent_count >= settings.max_restarts_per_role:
                    settings.paths.kill_switch_file.write_text(
                        (
                            f"created_utc={utc_now_iso()}\n"
                            f"reason=restart_budget_exceeded_{role}_{reason}\n"
                        ),
                        encoding="utf-8",
                    )
                    # Ask the still-running peer to perform its normal clean
                    # closeout immediately.  The failed role is flattened by the
                    # dedicated emergency worker after all children are stopped.
                    settings.paths.stop_file.write_text(
                        (
                            f"created_utc={utc_now_iso()}\n"
                            f"reason=restart_budget_exceeded_{role}_{reason}\n"
                        ),
                        encoding="utf-8",
                    )
                    final_status = "FAIL"
                    final_message = (
                        f"restart budget exceeded for {role}: {reason}"
                    )
                    break
                restart_utc = utc_now_iso()
                restarts = list(worker.restart_timestamps) + [restart_utc]
                starts = worker.starts + 1
                restart_events.append(
                    {
                        "role": role,
                        "reason": reason,
                        "restart_utc": restart_utc,
                        "start_number": starts,
                    }
                )
                time.sleep(settings.restart_cooldown_seconds)
                workers[role] = _launch_worker(
                    settings,
                    role,
                    starts=starts,
                    restarts=restarts,
                )
            _write_supervisor_status(
                settings,
                run_id=run_id,
                started_utc=started_utc,
                status="RUNNING",
                message="workers monitored",
                workers=workers,
                restart_events=restart_events,
            )
            time.sleep(settings.poll_seconds)

        # A STOP file asks workers to perform their configured clean closeout.
        stop_deadline = time.time() + settings.shutdown_timeout_seconds
        while time.time() < stop_deadline:
            if all(worker.process.poll() is not None for worker in workers.values()):
                break
            time.sleep(2)
        for worker in workers.values():
            _terminate_worker(worker, 15)

        closeout_reviews = {
            role: _worker_closeout_review(settings, worker)
            for role, worker in workers.items()
        }
        failed_closeouts = [
            role
            for role, review in closeout_reviews.items()
            if review.get("passed") is not True
        ]
        if failed_closeouts:
            final_status = "FAIL"
            final_message += (
                "; worker closeout validation failed for "
                + ",".join(failed_closeouts)
            )
            settings.paths.kill_switch_file.write_text(
                (
                    f"created_utc={utc_now_iso()}\n"
                    f"reason=worker_closeout_validation_failed_{'_'.join(failed_closeouts)}\n"
                ),
                encoding="utf-8",
            )

        if settings.paths.kill_switch_file.exists() or (
            settings.orders_enabled and failed_closeouts
        ):
            for role in ("model_a", "model_b"):
                emergency_flatten.append(_run_emergency_flatten(settings, role))
            if settings.orders_enabled and not all(
                item.get("passed") is True for item in emergency_flatten
            ):
                final_status = "FAIL"
                final_message += "; emergency flatten was not fully confirmed"
        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": final_status,
            "formal_gate": final_status == "PASS",
            "started_utc": started_utc,
            "completed_utc": utc_now_iso(),
            "execution_mode": settings.execution_mode,
            "orders_enabled": settings.orders_enabled,
            "duration_hours": settings.duration_hours,
            "message": final_message,
            "workers": {
                role: _worker_view(settings, worker)
                for role, worker in workers.items()
            },
            "restart_events": restart_events,
            "worker_closeout_reviews": closeout_reviews,
            "emergency_flatten": emergency_flatten,
            "control_files": {
                "stop": str(settings.paths.stop_file),
                "kill_switch": str(settings.paths.kill_switch_file),
            },
        }
        write_json_atomic(settings.paths.final_report, report)
        _write_supervisor_status(
            settings,
            run_id=run_id,
            started_utc=started_utc,
            status=final_status,
            message=final_message,
            workers=workers,
            restart_events=restart_events,
        )
        return report
    except Exception as exc:
        settings.paths.kill_switch_file.write_text(
            f"created_utc={utc_now_iso()}\nreason=supervisor_exception\n",
            encoding="utf-8",
        )
        for worker in workers.values():
            try:
                _terminate_worker(worker, 15)
            except Exception:
                pass
        if settings.orders_enabled:
            for role in ("model_a", "model_b"):
                try:
                    emergency_flatten.append(_run_emergency_flatten(settings, role))
                except Exception as flatten_exc:
                    emergency_flatten.append(
                        {
                            "role": role,
                            "attempted": True,
                            "passed": False,
                            "error": str(flatten_exc),
                        }
                    )
        report = {
            "schema_version": "1.0",
            "run_id": run_id,
            "status": "FAIL",
            "formal_gate": False,
            "started_utc": started_utc,
            "completed_utc": utc_now_iso(),
            "execution_mode": settings.execution_mode,
            "orders_enabled": settings.orders_enabled,
            "message": str(exc),
            "traceback": "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            ),
            "restart_events": restart_events,
            "emergency_flatten": emergency_flatten,
        }
        write_json_atomic(settings.paths.final_report, report)
        raise
    finally:
        release_lock(settings)
