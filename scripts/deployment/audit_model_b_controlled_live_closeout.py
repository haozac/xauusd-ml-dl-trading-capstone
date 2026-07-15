#!/usr/bin/env python
"""Create a formal offline closeout for an interrupted Stage 3 Step 3B run."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from capstone_trading.runtime.model_b_live_closeout import (
    CloseoutInputs,
    CloseoutOutputs,
    DEFAULT_AUDIT_REPORT_PATH,
    DEFAULT_EVENTS_CSV_PATH,
    DEFAULT_FRESH_EVENTS_CSV_PATH,
    DEFAULT_LATEST_DECISION_PATH,
    DEFAULT_LIVE_REPORT_PATH,
    DEFAULT_MIN_FRESH_EVENTS,
    DEFAULT_RUN_EVENTS_CSV_PATH,
    DEFAULT_STATE_PATH,
    DEFAULT_SUMMARY_CSV_PATH,
    Stage3Step3BCloseoutError,
    audit_model_b_live_closeout,
)

LOGGER = logging.getLogger("stage3_step3b_closeout")


def _repo_path(repo_root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = repo_root / candidate
    resolved = candidate.expanduser().resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise Stage3Step3BCloseoutError(f"Path must remain inside repository root: {resolved}") from exc
    return resolved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit incremental Stage 3 Step 3B artefacts after a monitored Ctrl+C stop. "
            "This command is offline and never connects to MT5 or calls order_send."
        )
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-id", required=True, help="Exact Stage 3 Step 3B run_id to isolate.")
    parser.add_argument("--events-csv", default=str(DEFAULT_EVENTS_CSV_PATH))
    parser.add_argument("--state", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--latest-decision-csv", default=str(DEFAULT_LATEST_DECISION_PATH))
    parser.add_argument("--live-report", default=str(DEFAULT_LIVE_REPORT_PATH))
    parser.add_argument("--audit-report", default=str(DEFAULT_AUDIT_REPORT_PATH))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV_PATH))
    parser.add_argument("--run-events-csv", default=str(DEFAULT_RUN_EVENTS_CSV_PATH))
    parser.add_argument("--fresh-events-csv", default=str(DEFAULT_FRESH_EVENTS_CSV_PATH))
    parser.add_argument("--min-fresh-events", type=int, default=DEFAULT_MIN_FRESH_EVENTS)
    parser.add_argument(
        "--termination-reason",
        default="keyboard_interrupt_after_manual_flat_check",
        choices=["keyboard_interrupt_after_manual_flat_check"],
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s %(message)s")
    repo_root = args.repo_root.expanduser().resolve()
    try:
        inputs = CloseoutInputs(
            events_csv=_repo_path(repo_root, args.events_csv),
            state_json=_repo_path(repo_root, args.state),
            latest_decision_csv=_repo_path(repo_root, args.latest_decision_csv),
            live_report_json=_repo_path(repo_root, args.live_report),
        )
        outputs = CloseoutOutputs(
            audit_report_json=_repo_path(repo_root, args.audit_report),
            summary_csv=_repo_path(repo_root, args.summary_csv),
            run_events_csv=_repo_path(repo_root, args.run_events_csv),
            fresh_events_csv=_repo_path(repo_root, args.fresh_events_csv),
        )
        report = audit_model_b_live_closeout(
            run_id=args.run_id,
            inputs=inputs,
            outputs=outputs,
            min_fresh_events=args.min_fresh_events,
            termination_reason=args.termination_reason,
        )
    except Exception as exc:
        LOGGER.exception("Stage 3 Step 3B closeout audit failed: %s", exc)
        return 1

    summary = report["summary"]
    LOGGER.info("Stage 3 Step 3B closeout status: %s", report["status"])
    LOGGER.info("Audited run_id: %s", report["audited_run_id"])
    LOGGER.info("Fresh completed M15 events: %s", summary["fresh_completed_m15_events"])
    LOGGER.info("Maximum p_up: %.6f", summary["maximum_probability_up"])
    LOGGER.info("Wide-spread fresh events: %s", summary["wide_spread_fresh_events"])
    LOGGER.info("order_send called count: %s", summary["order_send_called_count"])
    LOGGER.info("Final position: %s", summary["final_live_position_name"])
    LOGGER.info("Audit JSON: %s", outputs.audit_report_json)
    LOGGER.info("Summary CSV: %s", outputs.summary_csv)
    LOGGER.info("Run-specific events CSV: %s", outputs.run_events_csv)
    LOGGER.info("Fresh-events CSV: %s", outputs.fresh_events_csv)
    return 0


if __name__ == "__main__":
    sys.exit(main())
