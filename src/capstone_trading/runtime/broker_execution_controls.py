"""Stage 2 Step 3A broker-specific execution control freeze utilities.

This module is intentionally read-only/no-order.  It consumes the already
validated Stage 2 Step 2A MT5 shadow report and freezes the broker-specific
execution controls that Stage 3 must obey before any demo order testing.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import csv
import json
import math


DEFAULT_STEP2A_REPORT = Path("runtime/reports/stage2_step2a_v1_1_mt5_shadow_logger.json")
DEFAULT_REPORT_PATH = Path("runtime/reports/stage2_step3a_broker_execution_controls_freeze.json")
DEFAULT_SUMMARY_CSV_PATH = Path("runtime/reports/stage2_step3a_broker_execution_controls_summary.csv")
DEFAULT_FROZEN_CONFIG_PATH = Path("config/broker_execution_controls_frozen.yaml")


@dataclass(frozen=True)
class FrozenExecutionControls:
    """Frozen no-order execution controls for later Stage 3 validation."""

    controls_version: str = "stage2_step3a_v1_0"
    broker_company_expected: str = "Dukascopy Bank SA"
    server_expected: str = "Dukascopy-demo-mt5-1"
    symbol: str = "XAUUSD"
    timeframe: str = "M15"
    canonical_time_basis: str = "UTC"
    mt5_server_time_policy: str = "Dukascopy GMT+3 summer / GMT+2 winter, convert to canonical UTC"
    mt5_server_time_offset_hours_current: int = 3
    order_volume_lots: float = 0.01
    max_open_volume_lots_per_model: float = 0.01
    max_positions_per_model: int = 1
    max_spread_points_for_entry: int = 800
    max_deviation_points_for_stage3_order_request: int = 200
    broker_reported_filling_mode_policy: str = "Use broker-reported symbol_info.filling_mode as candidate; Stage 3 order_check must validate before order_send."
    broker_side_stop_loss_policy: str = "Disabled for tiny plumbing test; strategy/risk engine manages exits. Revisit before unattended final run."
    broker_side_take_profit_policy: str = "Disabled; strategy/risk engine manages exits."
    stage2_orders_enabled: bool = False
    stage3_order_check_required_before_order_send: bool = True
    stage3_first_order_test_volume_lots: float = 0.01
    model_a_magic_number: int = 26070101
    model_b_magic_number: int = 26070102
    model_a_order_comment: str = "CAPSTONE_MODEL_A"
    model_b_order_comment: str = "CAPSTONE_MODEL_B"
    model_b_variant_for_first_controlled_execution: str = "MODEL_B_V2_CURRENT"
    max_successful_entries_per_model_per_utc_day: int = 1
    capstone_leverage_cap: float = 10.0
    review_usd_sgd_rate_assumption: float = 1.35
    minimum_demo_equity_recommendation_sgd: float = 1000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return data


def is_volume_step_valid(volume: float, volume_min: float, volume_step: float, eps: float = 1e-9) -> bool:
    """Return True if volume is >= min and lies on the broker volume grid."""
    if volume_step <= 0 or volume < volume_min - eps:
        return False
    steps = (volume - volume_min) / volume_step
    return abs(steps - round(steps)) <= eps


def _nested_get(data: Mapping[str, Any], path: list[str], default: Any = None) -> Any:
    cur: Any = data
    for key in path:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_capital_review(
    *,
    controls: FrozenExecutionControls,
    symbol_info: Mapping[str, Any],
    account_equity_sgd: float | None,
) -> dict[str, Any]:
    """Estimate notional exposure for 0.01 XAUUSD and check the 10:1 cap.

    MT5 account currency in the user setup is SGD, while XAUUSD profit currency is
    USD.  This function therefore reports both USD notional and an SGD estimate
    based on a documented review assumption, not a live FX conversion.
    """
    contract_size = float(symbol_info.get("trade_contract_size", 100.0) or 100.0)
    ask = float(symbol_info.get("ask", 0.0) or 0.0)
    bid = float(symbol_info.get("bid", 0.0) or 0.0)
    price_reference = ask if ask > 0 else bid
    notional_usd = contract_size * controls.order_volume_lots * price_reference if price_reference > 0 else None
    required_equity_usd_at_cap = (
        notional_usd / controls.capstone_leverage_cap if notional_usd is not None else None
    )
    required_equity_sgd_estimate_at_cap = (
        required_equity_usd_at_cap * controls.review_usd_sgd_rate_assumption
        if required_equity_usd_at_cap is not None
        else None
    )

    effective_leverage_estimate = None
    capstone_leverage_cap_passed = None
    if account_equity_sgd is not None and notional_usd is not None and account_equity_sgd > 0:
        notional_sgd_estimate = notional_usd * controls.review_usd_sgd_rate_assumption
        effective_leverage_estimate = notional_sgd_estimate / account_equity_sgd
        capstone_leverage_cap_passed = effective_leverage_estimate <= controls.capstone_leverage_cap + 1e-12

    return {
        "account_equity_sgd_supplied": account_equity_sgd,
        "account_currency": "SGD",
        "symbol_profit_currency": symbol_info.get("currency_profit", "USD"),
        "review_usd_sgd_rate_assumption": controls.review_usd_sgd_rate_assumption,
        "order_volume_lots": controls.order_volume_lots,
        "contract_size": contract_size,
        "price_reference": price_reference,
        "estimated_notional_usd": notional_usd,
        "required_equity_usd_at_10x_cap": required_equity_usd_at_cap,
        "required_equity_sgd_estimate_at_10x_cap": required_equity_sgd_estimate_at_cap,
        "effective_leverage_estimate_if_equity_supplied": effective_leverage_estimate,
        "capstone_leverage_cap_passed_if_equity_supplied": capstone_leverage_cap_passed,
        "minimum_demo_equity_recommendation_sgd": controls.minimum_demo_equity_recommendation_sgd,
        "capital_ready_for_final_strategy_execution": (
            capstone_leverage_cap_passed is True
            and account_equity_sgd is not None
            and account_equity_sgd >= controls.minimum_demo_equity_recommendation_sgd
        ),
        "tiny_order_plumbing_note": (
            "0.01 lot is the broker minimum used for controlled demo plumbing. "
            "If equity is below the 10:1 project leverage cap requirement, do not use it "
            "for final strategy execution until demo equity is increased."
        ),
    }


def build_freeze_report(
    *,
    step2a_report: Mapping[str, Any],
    account_equity_sgd: float | None = None,
    controls: FrozenExecutionControls | None = None,
) -> dict[str, Any]:
    controls = controls or FrozenExecutionControls()
    account = _nested_get(step2a_report, ["account"], {}) or {}
    symbol_info = _nested_get(step2a_report, ["symbol_resolution", "symbol_info"], {}) or {}
    time_norm = _nested_get(step2a_report, ["time_normalisation"], {}) or {}
    terminal = _nested_get(step2a_report, ["terminal"], {}) or {}
    rates = _nested_get(step2a_report, ["rates"], {}) or {}

    volume_min = float(symbol_info.get("volume_min", math.nan))
    volume_step = float(symbol_info.get("volume_step", math.nan))
    volume_max = float(symbol_info.get("volume_max", math.nan))
    spread = int(symbol_info.get("spread", -1))

    validations = {
        "source_stage2_step2a_passed": step2a_report.get("status") == "PASS" and step2a_report.get("formal_gate") is True,
        "source_stage2_step2a_shadow_only": step2a_report.get("shadow_only") is True,
        "source_stage2_step2a_orders_disabled": step2a_report.get("orders_enabled") is False,
        "broker_company_matches": account.get("company") == controls.broker_company_expected,
        "server_matches": account.get("server") == controls.server_expected,
        "account_is_demo": account.get("trade_mode_name") == "DEMO",
        "account_margin_mode_recorded": "margin_mode" in account,
        "symbol_matches": _nested_get(step2a_report, ["symbol_resolution", "selected_symbol"]) == controls.symbol,
        "symbol_visible": bool(symbol_info.get("visible", False)),
        "symbol_trade_mode_recorded": "trade_mode" in symbol_info,
        "digits_is_3": int(symbol_info.get("digits", -1)) == 3,
        "point_is_0_001": abs(float(symbol_info.get("point", math.nan)) - 0.001) < 1e-12,
        "contract_size_positive": float(symbol_info.get("trade_contract_size", 0.0) or 0.0) > 0,
        "volume_min_allows_0_01": volume_min <= controls.order_volume_lots <= volume_max,
        "volume_step_allows_0_01": is_volume_step_valid(controls.order_volume_lots, volume_min, volume_step),
        "spread_float_recorded": "spread_float" in symbol_info,
        "observed_spread_within_stage3_entry_gate": 0 <= spread <= controls.max_spread_points_for_entry,
        "server_time_conversion_applied": time_norm.get("conversion_applied") is True,
        "server_time_offset_matches_current_freeze": int(time_norm.get("mt5_server_time_offset_hours", 999)) == controls.mt5_server_time_offset_hours_current,
        "latest_bar_not_future_after_conversion": float(time_norm.get("latest_bar_future_minutes_after_conversion", 999.0)) <= 0.0,
        "completed_m15_bars_used": rates.get("uses_completed_bars_only") is True and rates.get("start_pos") == 1,
        "terminal_connected": terminal.get("connected") is True,
        "terminal_order_disabled_for_stage2": terminal.get("trade_allowed") is False,
    }

    hard_gate_keys = [
        "source_stage2_step2a_passed",
        "source_stage2_step2a_shadow_only",
        "source_stage2_step2a_orders_disabled",
        "broker_company_matches",
        "server_matches",
        "account_is_demo",
        "symbol_matches",
        "symbol_visible",
        "digits_is_3",
        "point_is_0_001",
        "contract_size_positive",
        "volume_min_allows_0_01",
        "volume_step_allows_0_01",
        "server_time_conversion_applied",
        "server_time_offset_matches_current_freeze",
        "latest_bar_not_future_after_conversion",
        "completed_m15_bars_used",
        "terminal_connected",
        "terminal_order_disabled_for_stage2",
    ]
    hard_gate_passed = all(validations.get(k) is True for k in hard_gate_keys)

    warnings: list[str] = []
    if spread > controls.max_spread_points_for_entry:
        warnings.append("Observed spread exceeded the frozen Stage 3 entry gate; Stage 3 should block entries until spread normalises.")
    if account_equity_sgd is not None and account_equity_sgd < controls.minimum_demo_equity_recommendation_sgd:
        warnings.append(
            "Supplied demo equity is below the recommended SGD 1000 buffer. 0.01 lot may be acceptable for a tiny plumbing test, but not for unattended final strategy execution."
        )
    if terminal.get("trade_allowed") is False:
        warnings.append("Terminal trading is disabled, which is expected for Stage 2. Stage 3 must explicitly enable/validate this before order_check/order_send.")

    capital_review = build_capital_review(controls=controls, symbol_info=symbol_info, account_equity_sgd=account_equity_sgd)
    if capital_review["capstone_leverage_cap_passed_if_equity_supplied"] is False:
        warnings.append(
            "0.01 lot appears to breach the capstone 10:1 leverage cap at the supplied account equity. Increase demo equity before final strategy execution."
        )

    return {
        "stage": 2,
        "step": "3A",
        "status": "PASS" if hard_gate_passed else "FAIL",
        "formal_gate": hard_gate_passed,
        "no_order_stage": True,
        "mt5_used": False,
        "order_check_called": False,
        "order_send_called": False,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "controls": controls.to_dict(),
        "source_snapshot": {
            "broker_company": account.get("company"),
            "server": account.get("server"),
            "account_trade_mode_name": account.get("trade_mode_name"),
            "account_margin_mode": account.get("margin_mode"),
            "terminal_trade_allowed": terminal.get("trade_allowed"),
            "terminal_tradeapi_disabled": terminal.get("tradeapi_disabled"),
            "selected_symbol": _nested_get(step2a_report, ["symbol_resolution", "selected_symbol"]),
            "digits": symbol_info.get("digits"),
            "point": symbol_info.get("point"),
            "trade_contract_size": symbol_info.get("trade_contract_size"),
            "volume_min": symbol_info.get("volume_min"),
            "volume_step": symbol_info.get("volume_step"),
            "volume_max": symbol_info.get("volume_max"),
            "filling_mode": symbol_info.get("filling_mode"),
            "order_mode": symbol_info.get("order_mode"),
            "spread": symbol_info.get("spread"),
            "spread_float": symbol_info.get("spread_float"),
            "trade_mode": symbol_info.get("trade_mode"),
            "currency_profit": symbol_info.get("currency_profit"),
            "time_normalisation": time_norm,
        },
        "checks": {
            "validations": validations,
            "hard_gate_keys": hard_gate_keys,
            "hard_gate_passed": hard_gate_passed,
            "capital_review": capital_review,
        },
        "decision": {
            "frozen_for_stage3_preflight": hard_gate_passed,
            "first_stage3_order_test_volume_lots": controls.stage3_first_order_test_volume_lots,
            "model_b_for_first_controlled_execution": controls.model_b_variant_for_first_controlled_execution,
            "final_strategy_execution_capital_ready": capital_review["capital_ready_for_final_strategy_execution"],
            "next_step_if_pass": "Stage 3 Step 1 - order permission and order_check preflight; no order_send yet.",
            "do_not_start_final_paper_run_until": [
                "Stage 3 controlled order plumbing passes",
                "Demo equity/leverage review passes for 0.01 lot under capstone 10:1 cap",
                "Dual-terminal setup passes for Model A vs Model B comparison",
            ],
        },
        "warnings": warnings,
    }


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_summary_csv(path: Path, report: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    validations = _nested_get(report, ["checks", "validations"], {}) or {}
    for key, value in validations.items():
        rows.append({"section": "validation", "name": key, "value": value})
    capital = _nested_get(report, ["checks", "capital_review"], {}) or {}
    for key, value in capital.items():
        rows.append({"section": "capital_review", "name": key, "value": value})
    rows.append({"section": "decision", "name": "status", "value": report.get("status")})
    rows.append({"section": "decision", "name": "formal_gate", "value": report.get("formal_gate")})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "name", "value"])
        writer.writeheader()
        writer.writerows(rows)


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).replace('"', '\\"')
    return f'"{text}"'


def write_frozen_yaml(path: Path, controls: FrozenExecutionControls) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    groups = {
        "metadata": ["controls_version"],
        "broker": ["broker_company_expected", "server_expected", "symbol", "timeframe"],
        "time": ["canonical_time_basis", "mt5_server_time_policy", "mt5_server_time_offset_hours_current"],
        "execution_limits": [
            "order_volume_lots",
            "max_open_volume_lots_per_model",
            "max_positions_per_model",
            "max_spread_points_for_entry",
            "max_deviation_points_for_stage3_order_request",
            "capstone_leverage_cap",
            "minimum_demo_equity_recommendation_sgd",
        ],
        "order_policy": [
            "broker_reported_filling_mode_policy",
            "broker_side_stop_loss_policy",
            "broker_side_take_profit_policy",
            "stage2_orders_enabled",
            "stage3_order_check_required_before_order_send",
            "stage3_first_order_test_volume_lots",
        ],
        "identifiers": ["model_a_magic_number", "model_b_magic_number", "model_a_order_comment", "model_b_order_comment"],
        "model_policy": ["model_b_variant_for_first_controlled_execution", "max_successful_entries_per_model_per_utc_day"],
        "review_assumptions": ["review_usd_sgd_rate_assumption"],
    }
    data = controls.to_dict()
    lines = ["# Auto-written by Stage 2 Step 3A. Do not edit casually; use a formal patch.\n"]
    for group, keys in groups.items():
        lines.append(f"{group}:")
        for key in keys:
            lines.append(f"  {key}: {yaml_scalar(data[key])}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_freeze(
    *,
    repo_root: Path,
    step2a_report_path: Path | None = None,
    account_equity_sgd: float | None = None,
    report_path: Path = DEFAULT_REPORT_PATH,
    summary_csv_path: Path = DEFAULT_SUMMARY_CSV_PATH,
    frozen_config_path: Path = DEFAULT_FROZEN_CONFIG_PATH,
) -> dict[str, Any]:
    source_path = repo_root / (step2a_report_path or DEFAULT_STEP2A_REPORT)
    source_report = load_json(source_path)
    controls = FrozenExecutionControls()
    report = build_freeze_report(step2a_report=source_report, account_equity_sgd=account_equity_sgd, controls=controls)
    report["source_report"] = str((step2a_report_path or DEFAULT_STEP2A_REPORT))
    report["report_path"] = str(report_path)
    report["summary_csv_path"] = str(summary_csv_path)
    report["frozen_config_path"] = str(frozen_config_path)
    report["completed_utc"] = datetime.now(timezone.utc).isoformat()

    write_json(repo_root / report_path, report)
    write_summary_csv(repo_root / summary_csv_path, report)
    write_frozen_yaml(repo_root / frozen_config_path, controls)
    return report
