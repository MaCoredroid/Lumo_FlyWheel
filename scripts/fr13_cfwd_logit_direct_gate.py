#!/usr/bin/env python3
"""Issue and validate the source-bound CFWD logit-direct timing credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Sequence


CANDIDATE = "fixed32_cfwd_logit_direct_packed_physical_slots_v3"
CANDIDATE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3"
CANDIDATE_SOURCE_SHA256 = (
    "5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c"
)
INTEGRATION_SOURCE_SCHEMA = (
    "fr13.fixed32.cfwd_logit_direct.integration_source.v2"
)
INTEGRATION_SOURCE_SHA256 = (
    "a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c"
)
LIVE_SCHEMA = "fr13.fixed32.cfwd_logit_direct_live_ab.v2"
CREDENTIAL_SCHEMA = "fr13.fixed32.cfwd_logit_direct.production_credential.v2"
MODE = "hydra27_fixed32"
BATCH = 1
GATE_TASK_ID = "astropy__astropy-12907"
GATE_SUBSET_SHA256 = (
    "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb"
)
TIMING_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
TIMING_TASK_IDS = [
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
]


class GateError(ValueError):
    pass


def _regular(path: Path, label: str) -> bytes:
    try:
        metadata = path.lstat()
    except OSError as error:
        raise GateError(f"{label} is missing: {path}") from error
    if path.is_symlink() or not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise GateError(f"{label} must be a single-link regular file: {path}")
    raw = path.read_bytes()
    if not raw:
        raise GateError(f"{label} is empty: {path}")
    return raw


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not ASCII JSON") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} must contain one JSON object")
    return payload, raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise GateError(f"{label} is not lowercase SHA-256")
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    with open(temporary, "w", encoding="ascii") as handle:
        handle.write(
            json.dumps(
                payload, ensure_ascii=True, indent=2, sort_keys=True
            )
            + "\n"
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _validate_credential(
    payload: object,
    *,
    expected_source_commit: str,
    expected_subset_sha256: str,
) -> dict[str, Any]:
    keys = {
        "schema",
        "status",
        "candidate",
        "candidate_schema",
        "candidate_source_sha256",
        "integration_source_commit",
        "integration_source_schema",
        "integration_source_sha256",
        "mode",
        "qualified_batch",
        "task_count",
        "task_ids",
        "task_marker",
        "reference_always_served",
        "decision_mismatches",
        "walk_mismatches",
        "candidate_invalid",
        "complete_work_census_events",
        "live_result_sha256",
        "final_flush_sha256",
        "boundary_snapshot_sha256",
        "traffic_audit_sha256",
        "subset_sha256",
        "timing_eligible",
        "floor_acceptance_eligible",
    }
    if not isinstance(payload, dict) or set(payload) != keys:
        raise GateError("CFWD production credential key set drifted")
    task_ids = payload.get("task_ids")
    if (
        payload.get("schema") != CREDENTIAL_SCHEMA
        or payload.get("status") != "production_timing_ready"
        or payload.get("candidate") != CANDIDATE
        or payload.get("candidate_schema") != CANDIDATE_SCHEMA
        or payload.get("candidate_source_sha256") != CANDIDATE_SOURCE_SHA256
        or payload.get("integration_source_commit") != expected_source_commit
        or payload.get("integration_source_schema") != INTEGRATION_SOURCE_SCHEMA
        or payload.get("integration_source_sha256") != INTEGRATION_SOURCE_SHA256
        or payload.get("mode") != MODE
        or payload.get("qualified_batch") != BATCH
        or payload.get("task_count") != 1
        or not isinstance(task_ids, list)
        or len(task_ids) != 1
        or not isinstance(task_ids[0], str)
        or payload.get("task_marker") != "swe_verified:" + task_ids[0]
        or payload.get("reference_always_served") is not True
        or payload.get("decision_mismatches") != [0] * 5
        or payload.get("walk_mismatches") != [0] * 5
        or payload.get("candidate_invalid") != 0
        or type(payload.get("complete_work_census_events")) is not int
        or payload["complete_work_census_events"] < 1
        or payload.get("subset_sha256") != expected_subset_sha256
        or payload.get("timing_eligible") is not False
        or payload.get("floor_acceptance_eligible") is not False
    ):
        raise GateError("CFWD production credential identity drifted")
    for name in (
        "live_result_sha256",
        "final_flush_sha256",
        "boundary_snapshot_sha256",
        "traffic_audit_sha256",
        "subset_sha256",
    ):
        _require_sha(payload.get(name), f"credential {name}")
    return payload


def issue(
    *,
    live_result: Path,
    subset: Path,
    final_flush: Path,
    boundary_snapshot: Path,
    traffic_audit: Path,
    candidate_source: Path,
    source_commit: str,
    output: Path,
) -> dict[str, Any]:
    live, live_raw = _object(live_result, "CFWD live result")
    subset_payload, subset_raw = _object(subset, "one-task subset")
    flush, flush_raw = _object(final_flush, "final flush")
    _boundary, boundary_raw = _object(boundary_snapshot, "boundary snapshot")
    traffic, traffic_raw = _object(traffic_audit, "chat traffic audit")
    candidate_raw = _regular(candidate_source, "candidate source")
    if _sha(candidate_raw) != CANDIDATE_SOURCE_SHA256:
        raise GateError("candidate source SHA-256 drifted")
    if (
        len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
    ):
        raise GateError("integration source commit is invalid")
    task_ids = subset_payload.get("instance_ids")
    subset_sha = _sha(subset_raw)
    if subset_sha != GATE_SUBSET_SHA256 or task_ids != [GATE_TASK_ID]:
        raise GateError("gate subset is not the canonical one-task SWE-Verified gate")
    task_id = task_ids[0]
    marker = "swe_verified:" + task_id
    events = live.get("complete_work_census_events")
    if (
        live.get("schema") != LIVE_SCHEMA
        or live.get("status") != "PASS"
        or live.get("suite") != "SWE-Verified"
        or live.get("candidate") != CANDIDATE
        or live.get("candidate_schema") != CANDIDATE_SCHEMA
        or live.get("candidate_source_sha256") != CANDIDATE_SOURCE_SHA256
        or live.get("integration_source_schema") != INTEGRATION_SOURCE_SCHEMA
        or live.get("integration_source_sha256") != INTEGRATION_SOURCE_SHA256
        or live.get("source_commit") != source_commit
        or live.get("mode") != MODE
        or live.get("instance_id") != task_id
        or live.get("task_marker") != marker
        or live.get("batch_histogram") != {"1": events, "4": 0}
        or type(events) is not int
        or events < 1
        or live.get("counted_graph_replays") != events
        or live.get("decision_mismatches") != [0] * 5
        or live.get("walk_mismatches") != [0] * 5
        or live.get("candidate_invalid") != 0
        or live.get("served_return") != "reference all-parent products unchanged"
        or live.get("performance_measurement") is not False
        or live.get("finalized_by_fixed32_flush") is not True
    ):
        raise GateError("CFWD live result is not a zero-mismatch real B1 gate")
    ack = flush.get("ack") if isinstance(flush, dict) else None
    if (
        flush.get("schema") != "fr13-fixed32-flush-client-result-v1"
        or not isinstance(ack, dict)
        or ack.get("schema") != "fr13-fixed32-flush-ack-v1"
        or ack.get("action") != "final"
        or ack.get("status") != "ok"
        or ack.get("mode") != MODE
        or ack.get("generation") != live.get("flush_generation")
        or ack.get("nonce") != live.get("flush_nonce")
        or ack.get("producer_pid") != live.get("producer_pid")
        or not isinstance(ack.get("counters"), dict)
        or ack["counters"].get("complete_work_census_events") != events
        or _sha(boundary_raw) != live.get("boundary_snapshot_sha256")
    ):
        raise GateError("CFWD live result is not bound to the final flush")
    traffic_subset = traffic.get("subset") if isinstance(traffic, dict) else None
    traffic_checks = traffic.get("checks") if isinstance(traffic, dict) else None
    traffic_stream = traffic.get("complete_stream") if isinstance(traffic, dict) else None
    if (
        not isinstance(traffic_subset, dict)
        or traffic_subset.get("task_ids") != [task_id]
        or traffic_subset.get("task_count") != 1
        or traffic_subset.get("sha256") != subset_sha
        or not isinstance(traffic_checks, dict)
        or not traffic_checks
        or any(value is not True for value in traffic_checks.values())
        or not isinstance(traffic_stream, dict)
        or traffic_stream.get("complete_work_census_events") != events
    ):
        raise GateError("CFWD gate is not bound to authenticated one-task traffic")
    credential = {
        "schema": CREDENTIAL_SCHEMA,
        "status": "production_timing_ready",
        "candidate": CANDIDATE,
        "candidate_schema": CANDIDATE_SCHEMA,
        "candidate_source_sha256": CANDIDATE_SOURCE_SHA256,
        "integration_source_commit": source_commit,
        "integration_source_schema": INTEGRATION_SOURCE_SCHEMA,
        "integration_source_sha256": INTEGRATION_SOURCE_SHA256,
        "mode": MODE,
        "qualified_batch": BATCH,
        "task_count": 1,
        "task_ids": [task_id],
        "task_marker": marker,
        "reference_always_served": True,
        "decision_mismatches": [0] * 5,
        "walk_mismatches": [0] * 5,
        "candidate_invalid": 0,
        "complete_work_census_events": events,
        "live_result_sha256": _sha(live_raw),
        "final_flush_sha256": _sha(flush_raw),
        "boundary_snapshot_sha256": _sha(boundary_raw),
        "traffic_audit_sha256": _sha(traffic_raw),
        "subset_sha256": subset_sha,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
    }
    _validate_credential(
        credential,
        expected_source_commit=source_commit,
        expected_subset_sha256=subset_sha,
    )
    _atomic_json(output, credential)
    return credential


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue_parser = subparsers.add_parser("issue")
    issue_parser.add_argument("--live-result", type=Path, required=True)
    issue_parser.add_argument("--subset", type=Path, required=True)
    issue_parser.add_argument("--final-flush", type=Path, required=True)
    issue_parser.add_argument("--boundary-snapshot", type=Path, required=True)
    issue_parser.add_argument("--traffic-audit", type=Path, required=True)
    issue_parser.add_argument("--candidate-source", type=Path, required=True)
    issue_parser.add_argument("--source-commit", required=True)
    issue_parser.add_argument("--out", type=Path, required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--credential", type=Path, required=True)
    validate_parser.add_argument("--expected-sha256", required=True)
    validate_parser.add_argument("--source-commit", required=True)
    validate_parser.add_argument("--timing-subset", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "issue":
        result = issue(
            live_result=args.live_result,
            subset=args.subset,
            final_flush=args.final_flush,
            boundary_snapshot=args.boundary_snapshot,
            traffic_audit=args.traffic_audit,
            candidate_source=args.candidate_source,
            source_commit=args.source_commit,
            output=args.out,
        )
    else:
        credential, raw = _object(args.credential, "production credential")
        if _sha(raw) != args.expected_sha256:
            raise GateError("production credential raw SHA-256 drifted")
        timing_subset, timing_subset_raw = _object(
            args.timing_subset, "exact4 subset"
        )
        if (
            _sha(timing_subset_raw) != TIMING_SUBSET_SHA256
            or timing_subset.get("instance_ids") != TIMING_TASK_IDS
        ):
            raise GateError("timing subset is not canonical exact4 SWE-Verified")
        result = _validate_credential(
            credential,
            expected_source_commit=args.source_commit,
            expected_subset_sha256=GATE_SUBSET_SHA256,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
