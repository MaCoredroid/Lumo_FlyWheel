#!/usr/bin/env python3
"""Validate the diagnostic-only fixed32 full-vocabulary small-M head sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fr13_draft_head_m32_pass import (
    EXPECTED_INSTANCE,
    load_json,
    sha256_file,
    validate_chat_traffic_audit,
    validate_live_evidence,
)


LIVE_SCHEMA = "fr13.fixed32.draft_head_full_msweep_live_ab.v1"
EXPECTED_ROWS = [2, 4, 8, 16]
EXPECTED_HEADS = ["root", "mtp1", "mtp2", "mtp3", "mtp4"]
EXPECTED_GRAPH_SIGNATURE = (
    "d9a4ddece41d146e9949b9f8ff7c2603"
    "b8948d157b28ef69244e44469b36150c"
)


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _require_commit(value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError("source_commit is not a lowercase Git commit")
    return value


def validate_live_result(
    payload: dict[str, Any], *, expected_source_sha256: str
) -> dict[str, Any]:
    _require_sha256(expected_source_sha256, "candidate source")
    if (
        payload.get("schema") != LIVE_SCHEMA
        or payload.get("status") != "COMPLETE"
        or payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("candidate_rows") != EXPECTED_ROWS
        or payload.get("candidate_source_sha256") != expected_source_sha256
        or payload.get("served_return") != "reference BF16 logits unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("acceptance_eligible") is not False
        or payload.get("probe_eligible") is not False
        or payload.get("finalized_by_fixed32_flush") is not True
        or payload.get("flush_action") != "final"
    ):
        raise ValueError("small-M sweep live provenance drifted")
    _require_commit(payload.get("source_commit"))
    for key in (
        "events_sha256",
        "boundary_snapshot_sha256",
        "flush_nonce",
    ):
        _require_sha256(payload.get(key), key)
    completed_events = payload.get("completed_events")
    if (
        type(completed_events) is not int
        or completed_events < 1
        or payload.get("complete_work_census_events") != completed_events
        or payload.get("work_census_last_event_index") != completed_events - 1
        or type(payload.get("flush_generation")) is not int
        or payload["flush_generation"] < 1
        or type(payload.get("producer_pid")) is not int
        or payload["producer_pid"] < 1
    ):
        raise ValueError("small-M sweep terminal census drifted")
    geometry = payload.get("geometry")
    if (
        not isinstance(geometry, dict)
        or geometry.get("batch_size") != 1
        or geometry.get("calls_per_diagnostic_event") != 5
        or geometry.get("head_positions") != EXPECTED_HEADS
        or geometry.get("input_shape") != [1, 5120]
        or geometry.get("weight_shape") != [248320, 5120]
        or geometry.get("reference_output_shape") != [1, 248320]
        or geometry.get("hidden_snapshot_shape") != [5, 5120]
        or geometry.get("dtype") != "torch.bfloat16"
    ):
        raise ValueError("small-M sweep geometry drifted")
    event = payload.get("diagnostic_event")
    if (
        not isinstance(event, dict)
        or event.get("batch_size") != 1
        or event.get("measured") is not True
        or event.get("graph_replays") != 1
        or event.get("head_positions_compared") != 5
        or event.get("forward_step_index") != 0
        or type(event.get("graph_id")) is not int
        or event["graph_id"] <= 0
        or event.get("runtime_mode") != "hydra27_fixed32"
        or event.get("graph_signature") != EXPECTED_GRAPH_SIGNATURE
    ):
        raise ValueError("small-M sweep diagnostic event drifted")

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != len(EXPECTED_ROWS):
        raise ValueError("small-M sweep candidate set drifted")
    summary: list[dict[str, Any]] = []
    for row, candidate in zip(EXPECTED_ROWS, candidates, strict=True):
        if (
            not isinstance(candidate, dict)
            or candidate.get("m") != row
            or candidate.get("gemm_mnk") != [row, 248320, 5120]
            or candidate.get("input_shape") != [row, 5120]
            or candidate.get("output_shape") != [row, 248320]
            or candidate.get("served_rows") != 0
            or candidate.get("shadow_compared_rows") != 1
            or candidate.get("head_comparisons") != 5
            or candidate.get("bf16_elements_compared") != 5 * 248320
            or type(candidate.get("raw_bf16_mismatches")) is not int
            or candidate["raw_bf16_mismatches"] < 0
            or candidate.get("byte_exact")
            is not (candidate["raw_bf16_mismatches"] == 0)
            or candidate.get("valid_live_batch_sizes")
            != list(range(1, row + 1))
        ):
            raise ValueError(f"small-M sweep candidate M={row} drifted")
        summary.append(
            {
                "m": row,
                "raw_bf16_mismatches": candidate["raw_bf16_mismatches"],
                "byte_exact": candidate["byte_exact"],
            }
        )
    return {
        "completed_events": completed_events,
        "diagnostic_event": event,
        "candidates": summary,
    }


def validate(args: argparse.Namespace) -> dict[str, Any]:
    live_path = Path(args.live_result)
    live_payload, _ = load_json(live_path)
    expected_source = Path(args.candidate_source)
    source_sha = sha256_file(expected_source)
    if source_sha != args.expected_candidate_source_sha256:
        raise ValueError("candidate source SHA-256 changed")
    live_summary = validate_live_result(
        live_payload,
        expected_source_sha256=args.expected_candidate_source_sha256,
    )
    terminal = validate_live_evidence(
        live_payload=live_payload,
        final_flush_path=Path(args.final_flush),
        boundary_snapshot_path=Path(args.boundary_snapshot),
    )
    traffic = validate_chat_traffic_audit(
        audit_path=Path(args.chat_traffic_audit),
        expected_events=live_payload["completed_events"],
    )
    return {
        "schema": "fr13.fixed32.draft_head_full_msweep_validation.v1",
        "status": "VALID",
        "diagnostic_only": True,
        "performance_measurement": False,
        "acceptance_eligible": False,
        "live_result_sha256": sha256_file(live_path),
        "candidate_source_sha256": source_sha,
        "live": live_summary,
        "terminal": terminal,
        "traffic": traffic,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-result", required=True)
    parser.add_argument("--final-flush", required=True)
    parser.add_argument("--boundary-snapshot", required=True)
    parser.add_argument("--chat-traffic-audit", required=True)
    parser.add_argument("--candidate-source", required=True)
    parser.add_argument("--expected-candidate-source-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
