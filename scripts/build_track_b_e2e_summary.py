#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

TRACK_B_E2E_TASKS = [
    "responses-sdk-adapter-cutover/v1-clean-baseline",
    "transcript-merge-regression/v1-clean-baseline",
    "dead-flag-reachability-audit/v1-clean-baseline",
    "sqlalchemy-2-session-modernization/v1-clean-baseline",
    "security-audit-hotfix-remediation/v1-clean-baseline",
    "responsive-checkout-visual-regression/v1-clean-baseline",
    "incident-evidence-synthesis/v1-clean-baseline",
    "policy-aware-request-resolution/v1-clean-baseline",
    "multi-tool-transaction-repair/v1-clean-baseline",
    "skill-router-contract-upgrade/v1-clean-baseline",
    "plugin-scaffold-alignment/v1-clean-baseline",
    "release-note-to-plan-translation/v1-clean-baseline",
    "fanout-fullstack-release-blocker/v1-clean-baseline",
]
SAMPLE_HASH = hashlib.sha256("\n".join(TRACK_B_E2E_TASKS).encode("utf-8")).hexdigest()


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Malformed JSONL at {path}:{line_number}: {exc.msg}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _duration_s(start: Any, end: Any) -> float | None:
    start_ts = _parse_ts(start)
    end_ts = _parse_ts(end)
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, (end_ts - start_ts).total_seconds())


def _p50(values: list[float]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return float(statistics.median(clean))


def _json_float_list(raw: str) -> list[float]:
    if not raw:
        return []
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise RuntimeError("Expected a JSON list of numeric wallclock values")
    values: list[float] = []
    for value in payload:
        if not isinstance(value, (int, float)):
            raise RuntimeError("Expected a JSON list of numeric wallclock values")
        values.append(float(value))
    return values


def _sample_values(samples: list[dict[str, Any]], field: str, start: Any, end: Any) -> list[float]:
    start_ts = _parse_ts(start)
    end_ts = _parse_ts(end)
    values: list[float] = []
    for sample in samples:
        sample_ts = _parse_ts(sample.get("ts"))
        if sample_ts is None:
            continue
        if start_ts is not None and sample_ts < start_ts:
            continue
        if end_ts is not None and sample_ts > end_ts:
            continue
        value = sample.get(field)
        if isinstance(value, (int, float)):
            values.append(float(value))
    return values


def _manifest_workspace_hash(family: str, variant: str) -> str | None:
    path = REPO_ROOT / "benchmark_blueprints" / "families" / family / "manifest.lock.json"
    if not path.is_file():
        return None
    payload = _load_json(path)
    try:
        value = payload["variants"][variant]["workspace_manifest_sha256"]
    except (KeyError, TypeError):
        return None
    return str(value)


def _diagnose(regime_share: dict[str, float], evidence: dict[str, Any], bottleneck_regime: str) -> str:
    dram = evidence.get("regime_dram_active_p50")
    sm = evidence.get("regime_sm_active_p50")
    accepted = evidence.get("regime_accepted_per_draft_p50")
    if regime_share.get("tool-exec-wait", 0.0) >= 0.30:
        return "tool-exec-bound"
    if regime_share.get("prefill", 0.0) >= 0.40 or bottleneck_regime == "prefill":
        return "prefill-dominated"
    if isinstance(dram, (int, float)) and dram >= 0.85:
        return "memory-bw-saturated"
    if isinstance(sm, (int, float)) and sm >= 0.80 and (not isinstance(dram, (int, float)) or dram < 0.70):
        return "sm-bound"
    if isinstance(accepted, (int, float)) and accepted < 0.20:
        return "low-acceptance"
    if (
        isinstance(dram, (int, float))
        and isinstance(sm, (int, float))
        and dram < 0.70
        and sm < 0.50
    ):
        return "memory-bw-headroom"
    return "mixed"


def _vllm_by_request(path: Path) -> dict[str, dict[str, Any]]:
    payload = _load_json(path)
    if isinstance(payload, dict):
        if "requests" in payload and isinstance(payload["requests"], dict):
            return {str(key): value for key, value in payload["requests"].items() if isinstance(value, dict)}
        return {str(key): value for key, value in payload.items() if isinstance(value, dict)}
    if isinstance(payload, list):
        rows = {}
        for row in payload:
            if isinstance(row, dict) and row.get("vllm_request_id"):
                rows[str(row["vllm_request_id"])] = row
        return rows
    raise RuntimeError(f"Unsupported vLLM per-turn JSON shape: {path}")


def _normalize_vllm_request_metrics(row: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    request_id = row.get("request_id") or row.get("vllm_request_id") or row.get("id")
    if not request_id:
        return None
    prompt_tokens = row.get("prompt_tokens")
    completion_tokens = row.get("completion_tokens", row.get("generation_tokens"))
    prefill_sum_s = row.get("prefill_sum_s", row.get("prefill_s"))
    decode_sum_s = row.get("decode_sum_s", row.get("decode_s"))
    accepted = row.get("spec_decode_num_accepted_tokens")
    draft_tokens = row.get("spec_decode_num_draft_tokens")
    normalized = dict(row)
    normalized.update(
        {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "prefill_sum_s": prefill_sum_s,
            "decode_sum_s": decode_sum_s,
            "spec_decode_num_accepted_tokens": accepted,
            "spec_decode_num_draft_tokens": draft_tokens,
        }
    )
    if (
        isinstance(completion_tokens, (int, float))
        and isinstance(decode_sum_s, (int, float))
        and decode_sum_s > 0
        and "decode_tps" not in normalized
    ):
        normalized["decode_tps"] = completion_tokens / decode_sum_s
    if (
        isinstance(accepted, (int, float))
        and isinstance(draft_tokens, (int, float))
        and draft_tokens > 0
        and "accepted_per_draft_token" not in normalized
    ):
        normalized["accepted_per_draft_token"] = accepted / draft_tokens
    return str(request_id), normalized


def _vllm_jsonl_by_request(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for row in _load_jsonl(path):
        normalized = _normalize_vllm_request_metrics(row)
        if normalized is not None:
            request_id, metrics = normalized
            rows[request_id] = metrics
    return rows


def _load_vllm_request_metrics(task_dir: Path) -> dict[str, dict[str, Any]]:
    json_path = task_dir / "vllm_per_turn.json"
    if json_path.is_file():
        return _vllm_by_request(json_path)
    for name in ("vllm_per_turn.jsonl", "vllm_request_metrics.jsonl"):
        jsonl_path = task_dir / name
        if jsonl_path.is_file():
            return _vllm_jsonl_by_request(jsonl_path)
    raise RuntimeError("Task directory is missing vLLM request metrics artifact")


def build_task_summary(args: argparse.Namespace) -> dict[str, Any]:
    task_dir = Path(args.task_dir)
    family = args.family
    variant = args.variant
    task_id = f"{family}/{variant}"
    trace = _load_jsonl(task_dir / "codex_trace.jsonl")
    vllm = _load_vllm_request_metrics(task_dir)
    dcgm = _load_jsonl(task_dir / "dcgm_samples.jsonl") if (task_dir / "dcgm_samples.jsonl").is_file() else []

    starts = [event for event in trace if event.get("event") == "turn_start"]
    ends_by_turn = {event.get("turn"): event for event in trace if event.get("event") == "turn_end"}
    task_start = next((event for event in trace if event.get("event") == "task_start"), None)
    task_end = next((event for event in reversed(trace) if event.get("event") == "task_end"), None)
    if not task_start or not task_end:
        raise RuntimeError("codex_trace.jsonl must contain task_start and task_end")

    turns: list[dict[str, Any]] = []
    regime_duration: dict[str, float] = defaultdict(float)
    regime_dram: dict[str, list[float]] = defaultdict(list)
    regime_sm: dict[str, list[float]] = defaultdict(list)
    regime_acceptance: dict[str, list[float]] = defaultdict(list)
    output_cap_hits = 0
    missing_request_ids = 0
    spec_missing = 0
    silent_fallback = False

    for start in starts:
        turn_index = start.get("turn")
        end = ends_by_turn.get(turn_index, {})
        regime = str(start.get("regime") or "reasoning")
        request_id = start.get("vllm_request_id")
        if not request_id:
            missing_request_ids += 1
        metrics = vllm.get(str(request_id), {}) if request_id else {}
        duration = _duration_s(start.get("ts"), end.get("ts"))
        if duration is None:
            duration = float(metrics.get("decode_sum_s") or 0.0) + float(metrics.get("prefill_sum_s") or 0.0)
        regime_duration[regime] += duration

        completion_tokens = end.get("completion_tokens", metrics.get("completion_tokens"))
        max_tokens = end.get("max_tokens", metrics.get("max_tokens"))
        if isinstance(completion_tokens, (int, float)) and isinstance(max_tokens, (int, float)) and completion_tokens == max_tokens:
            output_cap_hits += 1

        accepted = metrics.get("accepted_per_draft_token", metrics.get("accepted_per_draft"))
        draft_tokens = metrics.get("spec_decode_num_draft_tokens")
        if regime not in {"prefill", "tool-exec-wait"}:
            if "spec_decode_num_accepted_tokens" not in metrics or "spec_decode_num_draft_tokens" not in metrics:
                spec_missing += 1
            if isinstance(draft_tokens, (int, float)) and draft_tokens <= 0:
                silent_fallback = True
        dram_p50 = _p50(_sample_values(dcgm, "dram_active_pct", start.get("ts"), end.get("ts")))
        sm_p50 = _p50(_sample_values(dcgm, "sm_active_pct", start.get("ts"), end.get("ts")))
        if isinstance(dram_p50, (int, float)):
            regime_dram[regime].append(dram_p50)
        if isinstance(sm_p50, (int, float)):
            regime_sm[regime].append(sm_p50)
        if isinstance(accepted, (int, float)):
            regime_acceptance[regime].append(float(accepted))
        turns.append(
            {
                "index": turn_index,
                "regime": regime,
                "duration_s": round(duration, 6),
                "decode_tps": metrics.get("decode_tps"),
                "accepted_per_draft": accepted,
                "dram_active_pct_p50": dram_p50,
                "sm_active_pct_p50": sm_p50,
                "vllm_request_id": request_id,
            }
        )

    for event in trace:
        if event.get("event") == "tool_call":
            wait_s = _duration_s(event.get("ts_codex_emit_end"), event.get("ts_tool_exec_end"))
            if wait_s is not None:
                regime_duration["tool-exec-wait"] += wait_s

    observed_wallclock_s = _duration_s(task_start.get("ts"), task_end.get("ts"))
    if observed_wallclock_s is None:
        raise RuntimeError("task_start/task_end timestamps must be ISO-8601 strings")
    run_wallclocks = _json_float_list(args.run_wallclocks_json) or [observed_wallclock_s]
    wallclock_s = float(statistics.median(run_wallclocks))
    total_regime_s = sum(regime_duration.values()) or wallclock_s
    regime_share = {key: value / total_regime_s for key, value in sorted(regime_duration.items())}
    bottleneck_regime = max(regime_share.items(), key=lambda item: item[1])[0] if regime_share else "reasoning"
    evidence = {
        "regime_share_pct": round(regime_share.get(bottleneck_regime, 0.0) * 100, 3),
        "regime_dram_active_p50": _p50(regime_dram[bottleneck_regime]),
        "regime_sm_active_p50": _p50(regime_sm[bottleneck_regime]),
        "regime_accepted_per_draft_p50": _p50(regime_acceptance[bottleneck_regime]),
    }
    expected_samples = observed_wallclock_s / args.dcgm_interval_s if args.dcgm_interval_s > 0 else 0
    dcgm_dropout_pct = max(0.0, (1.0 - (len(dcgm) / expected_samples)) * 100.0) if expected_samples else 100.0
    dcgm_profile_fields_present = any(
        isinstance(sample.get("dram_active_pct"), (int, float)) and isinstance(sample.get("sm_active_pct"), (int, float))
        for sample in dcgm
    )
    workspace_hash = _manifest_workspace_hash(family, variant)
    baseline_hash = args.baseline_workspace_hash or workspace_hash
    attestation = {
        "rule_1_cold_completion_discarded": bool(args.cold_completion_discarded),
        "rule_2_output_cap_hit_count": output_cap_hits,
        "rule_3_median_of_n_runs": len(run_wallclocks),
        "rule_4_workspace_hash_match": workspace_hash == baseline_hash,
        "rule_5_cache_reset_verified": bool(args.cache_reset_verified),
        "rule_6_dcgm_dropout_pct": round(dcgm_dropout_pct, 6),
        "rule_6_dcgm_profile_fields_present": dcgm_profile_fields_present,
        "rule_7_clock_skew_ms_p99": args.clock_skew_ms_p99,
        "rule_8_task_completed_normally": task_end.get("exit_code") == 0 and task_end.get("task_score") is not None,
        "rule_9_wallclock_wall_to_wall": abs(observed_wallclock_s - (float(task_end.get("wallclock_s", observed_wallclock_s)))) < 0.001,
        "rule_10_protocol_hash_match": bool(args.protocol_hash_match),
        "rule_11_generation_volume_within_band": bool(args.generation_volume_within_band),
        "rule_12_spec_decode_metrics_present": spec_missing == 0,
        "rule_13_silent_fallback_to_vanilla": silent_fallback,
        "rule_14_trace_emitter_correctness_verified_at": args.trace_emitter_correctness_verified_at,
        "rule_15_sample_hash_match": bool(args.sample_hash_match),
    }
    trusted = (
        attestation["rule_1_cold_completion_discarded"]
        and attestation["rule_3_median_of_n_runs"] >= 3
        and attestation["rule_4_workspace_hash_match"]
        and attestation["rule_5_cache_reset_verified"]
        and attestation["rule_6_dcgm_dropout_pct"] < 1.0
        and attestation["rule_6_dcgm_profile_fields_present"]
        and attestation["rule_7_clock_skew_ms_p99"] < 100
        and attestation["rule_8_task_completed_normally"]
        and attestation["rule_9_wallclock_wall_to_wall"]
        and attestation["rule_10_protocol_hash_match"]
        and attestation["rule_11_generation_volume_within_band"]
        and attestation["rule_12_spec_decode_metrics_present"]
        and not attestation["rule_13_silent_fallback_to_vanilla"]
        and bool(attestation["rule_14_trace_emitter_correctness_verified_at"])
        and attestation["rule_15_sample_hash_match"]
        and missing_request_ids == 0
    )
    summary = {
        "schema": "lumo.track_b.e2e_task_summary.v1",
        "round": args.round,
        "task_id": task_id,
        "runtime_config_hash": args.runtime_config_hash,
        "wallclock_s": round(wallclock_s, 6),
        "observed_run_wallclock_s": round(observed_wallclock_s, 6),
        "run_wallclocks_s": run_wallclocks,
        "task_score": task_end.get("task_score"),
        "task_completed": task_end.get("exit_code") == 0,
        "turns": turns,
        "regime_share": regime_share,
        "bottleneck_regime": bottleneck_regime,
        "bottleneck_diagnosis": _diagnose(regime_share, evidence, bottleneck_regime),
        "diagnosis_evidence": evidence,
        "sample_hash": SAMPLE_HASH,
        "workspace_manifest_sha256": workspace_hash,
        "missing_vllm_request_id_turns": missing_request_ids,
        "truthful_measurement_attestation": attestation,
        "trusted_measurement": trusted,
    }
    out = task_dir / "summary.json"
    if trusted or args.write_untrusted_diagnostic:
        out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not trusted and not args.write_untrusted_diagnostic:
        raise RuntimeError("summary failed truthful-measurement attestation; rerun or pass --write-untrusted-diagnostic")
    return summary


def build_round_summary(args: argparse.Namespace) -> dict[str, Any]:
    round_dir = Path(args.round_dir)
    summaries = [_load_json(path) for path in sorted(round_dir.glob("*/summary.json"))]
    trusted = [row for row in summaries if row.get("trusted_measurement")]
    if len(trusted) < 12 and not args.write_untrusted_diagnostic:
        raise RuntimeError(f"Only {len(trusted)} trusted task summaries found; round_summary.json requires at least 12")
    wallclocks = [float(row["wallclock_s"]) for row in trusted]
    diagnosis_distribution = Counter(str(row.get("bottleneck_diagnosis")) for row in trusted)
    regime_totals: dict[str, float] = defaultdict(float)
    for row in trusted:
        for regime, share in row.get("regime_share", {}).items():
            regime_totals[str(regime)] += float(share)
    divisor = max(len(trusted), 1)
    round_summary = {
        "schema": "lumo.track_b.e2e_round_summary.v1",
        "round": args.round,
        "runtime_config_hash": args.runtime_config_hash,
        "config_delta_vs_prior_round": args.config_delta_vs_prior_round,
        "hypothesis": args.hypothesis,
        "median_wallclock_s": statistics.median(wallclocks) if wallclocks else None,
        "aggregate_wallclock_s": sum(wallclocks),
        "wallclock_delta_vs_prior_round_s": args.wallclock_delta_vs_prior_round_s,
        "tasks_completed": sum(1 for row in trusted if row.get("task_completed")),
        "tasks_correctness_passed": sum(1 for row in trusted if row.get("task_completed") and row.get("task_score") is not None),
        "regime_share_aggregate": {key: value / divisor for key, value in sorted(regime_totals.items())},
        "diagnosis_distribution": dict(sorted(diagnosis_distribution.items())),
        "sample_hash": SAMPLE_HASH,
        "trusted_task_count": len(trusted),
        "untrusted_task_count": len(summaries) - len(trusted),
        "auto_research_agent_recommendation": args.auto_research_agent_recommendation,
        "next_round_proposal": args.next_round_proposal,
    }
    if len(trusted) >= 12 or args.write_untrusted_diagnostic:
        (round_dir / "round_summary.json").write_text(
            json.dumps(round_summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return round_summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build truthful Track B E2E task or round summaries.")
    sub = parser.add_subparsers(dest="command", required=True)
    task = sub.add_parser("task")
    task.add_argument("--round", type=int, required=True)
    task.add_argument("--task-dir", required=True)
    task.add_argument("--family", required=True)
    task.add_argument("--variant", default="v1-clean-baseline")
    task.add_argument("--runtime-config-hash", required=True)
    task.add_argument("--baseline-workspace-hash")
    task.add_argument("--run-wallclocks-json", default="")
    task.add_argument("--clock-skew-ms-p99", type=float, default=999999.0)
    task.add_argument("--trace-emitter-correctness-verified-at", default="")
    task.add_argument("--dcgm-interval-s", type=float, default=0.01)
    task.add_argument("--cold-completion-discarded", action="store_true")
    task.add_argument("--cache-reset-verified", action="store_true")
    task.add_argument("--protocol-hash-match", action="store_true")
    task.add_argument("--generation-volume-within-band", action="store_true")
    task.add_argument("--sample-hash-match", action="store_true")
    task.add_argument("--write-untrusted-diagnostic", action="store_true")

    round_parser = sub.add_parser("round")
    round_parser.add_argument("--round", type=int, required=True)
    round_parser.add_argument("--round-dir", required=True)
    round_parser.add_argument("--runtime-config-hash", required=True)
    round_parser.add_argument("--config-delta-vs-prior-round", default="")
    round_parser.add_argument("--hypothesis", default="")
    round_parser.add_argument("--wallclock-delta-vs-prior-round-s", type=float, default=None)
    round_parser.add_argument("--auto-research-agent-recommendation", default="")
    round_parser.add_argument("--next-round-proposal", default="")
    round_parser.add_argument("--write-untrusted-diagnostic", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "task":
            build_task_summary(args)
        else:
            build_round_summary(args)
    except Exception as exc:
        print(f"build_track_b_e2e_summary.py: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
