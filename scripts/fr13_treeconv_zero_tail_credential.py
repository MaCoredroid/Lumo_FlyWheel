#!/usr/bin/env python3
"""Issue a fail-closed real-task graph credential for tree-conv zero-tail."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.treeconv_zero_tail.credential.v2"
RECORD_SCHEMA = "fr13.fixed32.treeconv_zero_tail.byte_ab.v2"
TERMINAL_SCHEMA = "fr13.fixed32.treeconv_zero_tail.byte_ab_terminal.v2"
WORK_SCHEMA = "fr13-fixed32-work-census-v12"
WORK_TERMINAL_SCHEMA = "fr13-fixed32-work-census-terminal-v12"
BOUNDARY_SCHEMA = "fr13-fixed32-boundary-snapshot-v4"
TASK_BOUNDARY_SCHEMA = "fr13-fixed32-task-boundary-v1"
FLUSH_RESULT_SCHEMA = "fr13-fixed32-flush-client-result-v1"
FLUSH_ACK_SCHEMA = "fr13-fixed32-flush-ack-v1"
QWEN_SCHEMA = "fr13-fixed32-qwen-campaign-provenance-v1"
SEQUENCE = "scripts/fr13_fixed32_floor_timers_seq.sh"
SOURCE_RELATIVE = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
MODES = {"tail6_fixed32": "Tail23", "hydra27_fixed32": "Hydra27"}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
CANONICAL_SUBSETS = {
    1: {
        "path": "config/fr13_fixed32/subset_b1_diagnostic_one.json",
        "sha256": "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb",
        "task_ids": ("astropy__astropy-12907",),
    },
    4: {
        "path": "config/fr13_fixed32/subset_b4_four.json",
        "sha256": "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5",
        "task_ids": (
            "astropy__astropy-12907",
            "astropy__astropy-13033",
            "astropy__astropy-13236",
            "astropy__astropy-13398",
        ),
    },
}
REQUIRED_CONTAINER_ENV = {
    "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE",
    "ENFORCE_EAGER": "0",
    "FR13_DRAFT_VOCAB_ROOT": "1",
    "FR13_DRAFT_VOCAB_K": "65536",
    "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
    "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
    "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL": "0",
    "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB": "1",
}
COMMIT_BOUND_PATHS = {
    SOURCE_RELATIVE,
    "src/lumo_flywheel_serving/fr13_tree_conv_fused.py",
    "src/lumo_flywheel_serving/inference_proxy.py",
    "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "scripts/fr13_bigdenom_swe_serve_variant.sh",
    "scripts/fr13_fixed32_contract.py",
    "scripts/fr13_fixed32_topology.py",
    "scripts/fr13_fixed32_work_census.py",
    "scripts/fr13_launch_forked_fa2_tree_server.sh",
    "scripts/fr13_run_treeconv_zero_tail_live_gate.sh",
    "scripts/fr13_runtime_manifest.py",
    "scripts/fr13_treeconv_zero_tail_credential.py",
    "scripts/fr13_dvk_subset_blocks.json",
    "scripts/run_swe_bench_q36_a.py",
    SEQUENCE,
}


class CredentialError(RuntimeError):
    """The supplied evidence cannot issue a tree-conv credential."""


def _duplicate_checked(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CredentialError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> object:
    raise CredentialError(f"non-finite JSON constant {value!r}")


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _raw(path: Path, label: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise CredentialError(f"{label} must be a regular non-symlink file")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise CredentialError(f"{label} is unavailable") from error
    if not raw:
        raise CredentialError(f"{label} is empty")
    return raw


def _loads(raw: bytes, label: str) -> object:
    try:
        return json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_duplicate_checked,
            parse_constant=_reject_constant,
        )
    except CredentialError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"{label} is invalid strict JSON") from error


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _raw(path, label)
    value = _loads(raw, label)
    if not isinstance(value, dict):
        raise CredentialError(f"{label} is not a JSON object")
    return value, raw


def _jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = _raw(path, label)
    if not raw.endswith(b"\n") or any(not line for line in raw.splitlines()):
        raise CredentialError(f"{label} is blank or truncated")
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        value = _loads(line, f"{label}:{index}")
        if not isinstance(value, dict):
            raise CredentialError(f"{label}:{index} is not an object")
        rows.append(value)
    return rows, raw


def _identity(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(raw), "bytes": len(raw)}


def _hex64(value: object, label: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise CredentialError(f"{label} is not a SHA-256 digest")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value < 1:
        raise CredentialError(f"{label} is not a positive integer")
    return value


def _git_show(repo: Path, commit: str, relative: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{commit}:{relative}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CredentialError(
            f"source commit cannot provide required path {relative}"
        ) from error
    return result.stdout


def _validate_runtime_manifest(
    path: Path,
    *,
    repo: Path,
    source_commit: str,
    batch_size: int,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    runtime, raw = _json(path, "runtime manifest")
    try:
        from fr13_runtime_manifest import ManifestError, build_manifest

        expected = build_manifest(repo, profile="fixed32", sequence=SEQUENCE)
    except (ImportError, ManifestError, OSError) as error:
        raise CredentialError("cannot rebuild the canonical runtime manifest") from error
    expected_raw = (
        json.dumps(expected, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if runtime != expected or raw != expected_raw:
        raise CredentialError("runtime manifest is not the full canonical live manifest")

    closures = runtime.get("closures")
    records: dict[str, dict[str, Any]] = {}
    if not isinstance(closures, dict):
        raise CredentialError("runtime manifest closures are malformed")
    for section in (
        "host_script_source",
        "python_package_source",
        "runtime_data_and_config",
        "verdict_tools",
    ):
        rows = closures.get(section)
        if not isinstance(rows, list):
            raise CredentialError(f"runtime manifest closure {section} is malformed")
        for record in rows:
            if (
                not isinstance(record, dict)
                or set(record) != {"path", "sha256", "size"}
                or not isinstance(record.get("path"), str)
                or record["path"] in records
            ):
                raise CredentialError("runtime manifest has a duplicate/malformed row")
            records[record["path"]] = record

    required = set(COMMIT_BOUND_PATHS)
    required.add(CANONICAL_SUBSETS[batch_size]["path"])
    committed_digests: dict[str, str] = {}
    for relative in sorted(required):
        record = records.get(relative)
        live_path = repo / relative
        live_raw = _raw(live_path, f"runtime source {relative}")
        committed_raw = _git_show(repo, source_commit, relative)
        if (
            not committed_raw
            or committed_raw != live_raw
            or record
            != {"path": relative, "sha256": _sha(live_raw), "size": len(live_raw)}
        ):
            raise CredentialError(
                f"runtime/source-commit binding differs for {relative}"
            )
        committed_digests[relative] = _sha(live_raw)
    return runtime, raw, committed_digests


def _topology_descriptor(mode: str) -> dict[str, Any]:
    from fr13_fixed32_topology import (
        HYDRA27_VALID_MASK,
        PHYSICAL_PARENT,
        PHYSICAL_PARENT_SHA256,
        TAIL6_VALID_MASK,
    )

    paths: list[list[int]] = []
    for node in range(len(PHYSICAL_PARENT)):
        path: list[int] = []
        cursor = node
        while cursor >= 0:
            path.append(cursor)
            cursor = PHYSICAL_PARENT[cursor]
        paths.append(list(reversed(path)))
    state_src: list[int] = []
    for path in paths:
        for column in range(34):
            position = len(path) + column
            if position < 3:
                state_src.append(position)
            elif column < 3:
                state_src.append(3 + path[position - 3])
            else:
                state_src.append(35)
    valid_mask = (
        TAIL6_VALID_MASK if mode == "tail6_fixed32" else HYDRA27_VALID_MASK
    )
    return {
        "schema": "fr13.fixed32.treeconv_state_descriptor.v1",
        "mode": mode,
        "logical_topology": MODES[mode],
        "valid_mask": valid_mask,
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "conv_width": 4,
        "conv_state_length": 34,
        "source_rows_per_request": 36,
        "live_state_columns": 3,
        "physical_parent_sha256": PHYSICAL_PARENT_SHA256,
        "state_src_sha256": _sha(_canonical(state_src)),
    }


def _validate_work_census(
    path: Path, *, mode: str, batch_size: int
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], bytes]:
    raw = _raw(path, "fixed32 work census")
    try:
        from fr13_fixed32_work_census import (
            CensusError,
            TAW_ROUTE,
            load_jsonl_bytes,
            validate_arm,
        )

        located = load_jsonl_bytes(raw, source=str(path.resolve()))
        report = validate_arm(
            located,
            expected_mode=mode,
            expected_route=TAW_ROUTE,
            required_batches=[batch_size],
        )
    except (ImportError, CensusError) as error:
        raise CredentialError(f"fixed32 work census is invalid: {error}") from error
    events = [row for row, _source in located[:-1]]
    terminal = located[-1][0]
    if (
        not events
        or terminal.get("schema") != WORK_TERMINAL_SCHEMA
        or terminal.get("final") is not True
        or terminal.get("event_count") != len(events)
        or report.get("event_count") != len(events)
        or report.get("producer_pid") != terminal.get("producer_pid")
    ):
        raise CredentialError("fixed32 work census terminal/count binding differs")
    return events, terminal, report, raw


def _validate_comparator(
    path: Path,
    *,
    events: list[dict[str, Any]],
    work_terminal: dict[str, Any],
    descriptor: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], bytes]:
    rows, raw = _jsonl(path, "tree-conv graph comparator")
    if len(rows) != len(events) + 1:
        raise CredentialError("tree-conv comparator is truncated or overcomplete")
    records, terminal = rows[:-1], rows[-1]
    record_keys = {
        "schema", "mode", "event_id", "event_index", "forward_step_index",
        "producer_pid", "batch_size", "request_ids_sha256",
        "request_id_sha256s", "execution_basis", "topology", "conv_layers",
        "conv_channels", "conv_state_length", "source_rows_per_request",
        "candidate_zero_tail", "reference_zero_tail",
        "reference_restored_and_served", "raw_bf16_byte_comparison",
        "compared_bytes", "differing_bytes", "byte_equal", "timing_eligible",
    }
    for index, (record, event) in enumerate(zip(records, events, strict=True)):
        runtime = event["drafter_runtime"]
        batch = event["batch_size"]
        if (
            set(record) != record_keys
            or record.get("schema") != RECORD_SCHEMA
            or record.get("mode") != descriptor["mode"]
            or record.get("event_id") != event["event_id"]
            or record.get("event_index") != index
            or record.get("forward_step_index") != event["forward_step_index"]
            or record.get("producer_pid") != event["producer_pid"]
            or record.get("batch_size") != batch
            or record.get("request_ids_sha256")
            != runtime["request_ids_sha256"]
            or record.get("request_id_sha256s")
            != runtime["request_id_sha256s"]
            or record.get("execution_basis") != "cudagraph_full_replay"
            or record.get("topology") != descriptor
            or record.get("conv_layers") != 48
            or record.get("conv_channels") != 10240
            or record.get("conv_state_length") != 34
            or record.get("source_rows_per_request") != 36
            or record.get("candidate_zero_tail") is not True
            or record.get("reference_zero_tail") is not False
            or record.get("reference_restored_and_served") is not True
            or record.get("raw_bf16_byte_comparison") is not True
            or record.get("compared_bytes") != batch * 48 * 10240 * 34 * 2
            or record.get("differing_bytes") != 0
            or record.get("byte_equal") is not True
            or record.get("timing_eligible") is not False
        ):
            raise CredentialError(f"tree-conv comparator/work join differs at {index}")
    terminal_keys = {
        "schema", "status", "mode", "topology", "complete_work_census_events",
        "first_event_index", "last_event_index", "first_forward_step_index",
        "last_forward_step_index", "producer_pid", "counted_graph_replays",
        "total_compared_bytes", "total_differing_bytes",
        "comparison_records_sha256", "work_census_events_sha256",
        "flush_generation", "flush_nonce", "boundary_snapshot_sha256",
        "flush_action", "finalized_by_fixed32_flush", "reference_always_served",
        "timing_eligible",
    }
    if (
        set(terminal) != terminal_keys
        or terminal.get("schema") != TERMINAL_SCHEMA
        or terminal.get("status") != "PASS"
        or terminal.get("mode") != descriptor["mode"]
        or terminal.get("topology") != descriptor
        or terminal.get("complete_work_census_events") != len(records)
        or terminal.get("first_event_index") != 0
        or terminal.get("last_event_index") != len(records) - 1
        or terminal.get("first_forward_step_index")
        != events[0]["forward_step_index"]
        or terminal.get("last_forward_step_index")
        != events[-1]["forward_step_index"]
        or terminal.get("producer_pid") != work_terminal["producer_pid"]
        or terminal.get("counted_graph_replays") != len(records)
        or terminal.get("total_compared_bytes")
        != sum(record["compared_bytes"] for record in records)
        or terminal.get("total_differing_bytes") != 0
        or terminal.get("comparison_records_sha256") != _sha(_canonical(records))
        or terminal.get("work_census_events_sha256")
        != work_terminal["events_sha256"]
        or terminal.get("flush_action") != "final"
        or terminal.get("finalized_by_fixed32_flush") is not True
        or terminal.get("reference_always_served") is not True
        or terminal.get("timing_eligible") is not False
    ):
        raise CredentialError("tree-conv comparator terminal binding differs")
    _positive_int(terminal.get("flush_generation"), "comparator flush generation")
    _hex64(terminal.get("flush_nonce"), "comparator flush nonce")
    _hex64(
        terminal.get("boundary_snapshot_sha256"),
        "comparator boundary snapshot digest",
    )
    return records, terminal, raw


def _validate_flush(
    result_path: Path,
    boundary_base: Path,
    *,
    mode: str,
    work_terminal: dict[str, Any],
    comparator_terminal: dict[str, Any],
) -> tuple[dict[str, Any], bytes, Path, bytes]:
    result, result_raw = _json(result_path, "final flush result")
    if set(result) != {"schema", "ack"} or result.get("schema") != FLUSH_RESULT_SCHEMA:
        raise CredentialError("final flush result schema differs")
    ack = result.get("ack")
    ack_keys = {
        "schema", "mode", "producer_pid", "generation", "nonce", "action",
        "status", "counters",
    }
    counters = ack.get("counters") if isinstance(ack, dict) else None
    event_count = work_terminal["event_count"]
    if (
        not isinstance(ack, dict)
        or set(ack) != ack_keys
        or ack.get("schema") != FLUSH_ACK_SCHEMA
        or ack.get("mode") != mode
        or ack.get("producer_pid") != work_terminal["producer_pid"]
        or ack.get("generation") != comparator_terminal["flush_generation"]
        or ack.get("nonce") != comparator_terminal["flush_nonce"]
        or ack.get("action") != "final"
        or ack.get("status") != "ok"
        or not isinstance(counters, dict)
        or counters.get("pure_decode_forward_steps") != event_count
        or counters.get("complete_work_census_events") != event_count
        or counters.get("work_census_first_forward_step")
        != work_terminal["first_forward_step_index"]
        or counters.get("work_census_last_forward_step")
        != work_terminal["last_forward_step_index"]
        or any(counters.get(key) != 0 for key in ("sfwd_pending", "dfwd_pending", "cfwd_pending"))
    ):
        raise CredentialError("final flush ack does not close the complete work census")
    generation = _positive_int(ack["generation"], "final flush generation")
    boundary_path = Path(str(boundary_base) + f".{generation}.json")
    boundary, boundary_raw = _json(boundary_path, "final boundary snapshot")
    fixed = (
        boundary.get("metrics", {}).get("fixed32")
        if isinstance(boundary.get("metrics"), dict)
        else None
    )
    if (
        boundary.get("schema") != BOUNDARY_SCHEMA
        or boundary.get("mode") != mode
        or boundary.get("producer_pid") != ack["producer_pid"]
        or boundary.get("generation") != generation
        or boundary.get("nonce") != ack["nonce"]
        or boundary.get("action") != "final"
        or boundary.get("counters") != counters
        or not isinstance(fixed, dict)
        or fixed.get("complete_work_census_events") != event_count
        or fixed.get("pure_decode_forward_steps") != event_count
        or fixed.get("events_sha256") != work_terminal["events_sha256"]
        or _sha(boundary_raw)
        != comparator_terminal["boundary_snapshot_sha256"]
    ):
        raise CredentialError("final boundary snapshot identity differs")
    return result, result_raw, boundary_path, boundary_raw


def _trace(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    events, raw = _jsonl(path, f"Qwen trace {path.parent.name}")
    if (
        not events
        or events[0].get("type") != "system"
        or events[0].get("subtype") != "init"
        or events[0].get("qwen_code_version") != "0.19.4"
    ):
        raise CredentialError(f"{path}: trace lacks the pinned Qwen init event")
    return events, raw


def _validate_qwen_and_tasks(
    *,
    task_root: Path,
    task_ids: list[str],
    mode: str,
    batch_size: int,
    producer_pid: int,
    proxy_rows: list[dict[str, Any]],
    successful_engine_ids: dict[str, str],
    qwen_campaign_path: Path | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any] | None]:
    from fr13_fixed32_contract import (
        ContractError,
        fixed32_trace_session_id,
        validate_fixed32_trace_model_requests,
    )
    from lumo_flywheel_serving.inference_proxy import fixed32_task_key_id

    campaign_identity = None
    campaign_proof = None
    if batch_size == 4:
        if qwen_campaign_path is None:
            raise CredentialError("B4 credential requires a Qwen campaign proof")
        campaign_proof, campaign_raw = _json(qwen_campaign_path, "Qwen campaign proof")
        if (
            campaign_proof.get("schema") != QWEN_SCHEMA
            or campaign_proof.get("task_ids") != task_ids
            or campaign_proof.get("concurrency") != 4
            or campaign_proof.get("metric_scope") != "concurrent_campaign_union"
            or campaign_proof.get("selection", {}).get("basis")
            != "runner_owned_campaign_endpoint_metrics"
            or campaign_proof.get("selection", {}).get("task_boundary_schema")
            != TASK_BOUNDARY_SCHEMA
        ):
            raise CredentialError("Qwen campaign proof identity/selection differs")
        if campaign_raw != _canonical(campaign_proof) + b"\n":
            raise CredentialError("Qwen campaign proof is not canonical JSON")
        campaign_identity = _identity(qwen_campaign_path, campaign_raw)
    elif qwen_campaign_path is not None:
        raise CredentialError("B1 credential forbids a campaign proof")

    task_bindings: dict[str, dict[str, Any]] = {}
    for task_id in task_ids:
        task_dir = task_root / task_id
        metadata, _metadata_raw = _json(
            task_dir / "runner_metadata.json", f"runner metadata {task_id}"
        )
        boundary, _boundary_raw = _json(
            task_dir / "fixed32_task_boundary.json", f"task boundary {task_id}"
        )
        provenance = metadata.get("fixed32_real_task_provenance")
        task_key = fixed32_task_key_id(task_id)
        interval = boundary.get("forward_step_interval")
        pre = boundary.get("pre")
        post = boundary.get("post")
        agent = metadata.get("agent") or metadata.get("codex")
        if (
            metadata.get("instance_id") != task_id
            or metadata.get("fixed32_task_boundary") != boundary
            or boundary.get("schema") != TASK_BOUNDARY_SCHEMA
            or boundary.get("instance_id") != task_id
            or boundary.get("mode") != mode
            or boundary.get("producer_pid") != producer_pid
            or not isinstance(pre, dict)
            or not isinstance(post, dict)
            or any(
                endpoint.get("schema") != FLUSH_ACK_SCHEMA
                or endpoint.get("mode") != mode
                or endpoint.get("producer_pid") != producer_pid
                or endpoint.get("action") != "snapshot"
                or endpoint.get("status") != "ok"
                or not isinstance(endpoint.get("counters"), dict)
                for endpoint in (pre, post)
            )
            or not isinstance(interval, dict)
            or set(interval)
            != {"start_forward_step", "end_forward_step", "expected_complete_events"}
            or type(interval.get("start_forward_step")) is not int
            or type(interval.get("end_forward_step")) is not int
            or interval["start_forward_step"] < 0
            or interval["end_forward_step"] <= interval["start_forward_step"]
            or interval.get("expected_complete_events")
            != interval["end_forward_step"] - interval["start_forward_step"]
            or pre["counters"].get("pure_decode_forward_steps")
            != interval["start_forward_step"]
            or post["counters"].get("pure_decode_forward_steps")
            != interval["end_forward_step"]
            or not isinstance(provenance, dict)
            or provenance.get("schema") != "fr13-fixed32-real-task-provenance-v3"
            or provenance.get("instance_id") != task_id
            or provenance.get("task_key_id") != task_key
            or not isinstance(agent, dict)
            or agent.get("exit_code") != 0
            or agent.get("timed_out") is not False
            or agent.get("offloaded") is not True
            or agent.get("network_drop") is not False
        ):
            raise CredentialError(f"real-task provenance/boundary differs for {task_id}")

        task_proxy = [row for row in proxy_rows if row.get("task_key_id") == task_key]
        expected_evidence = {
            "completed_logical_model_requests": sum(
                row["event"] == "logical_complete" for row in task_proxy
            ),
            "aborted_logical_requests": 0,
            "accepted_attempts": sum(row["event"] == "attempt_begin" for row in task_proxy),
            "completed_attempts": sum(row["event"] == "attempt_result" for row in task_proxy),
            "failed_attempts": 0,
        }
        if any(provenance.get(key) != value for key, value in expected_evidence.items()):
            raise CredentialError(f"task-auth counters differ for {task_id}")
        after_records = provenance.get("task_auth_evidence_after_ledger_records")
        after_head = provenance.get(
            "task_auth_evidence_after_ledger_chain_head_sha256"
        )
        if (
            type(after_records) is not int
            or not 0 < after_records < len(proxy_rows)
            or proxy_rows[after_records - 1]["record_sha256"] != after_head
        ):
            raise CredentialError(f"task-auth ledger prefix differs for {task_id}")
        after_payload = {
            "schema": "fr13-fixed32-task-auth-evidence-v1",
            "task_key_id": task_key,
            **expected_evidence,
            "phase": "campaign",
            "ledger_records": after_records,
            "ledger_chain_head_sha256": after_head,
        }
        if (
            provenance.get("task_auth_evidence_after_sha256")
            != _sha(_canonical(after_payload))
            or HEX64.fullmatch(
                str(provenance.get("task_auth_evidence_before_sha256"))
            )
            is None
        ):
            raise CredentialError(f"task-auth evidence digest differs for {task_id}")

        trace_path = task_dir / "qwen_trace.jsonl"
        events, trace_raw = _trace(trace_path)
        completed = expected_evidence["completed_logical_model_requests"]
        try:
            if batch_size == 1:
                replay = validate_fixed32_trace_model_requests(
                    events,
                    expected_session_id=fixed32_trace_session_id(task_id),
                    expected_completed_logical_model_requests=completed,
                    metrics_pre=_raw(task_dir / "vllm_metrics_pre.txt", "B1 Qwen pre metrics"),
                    metrics_post=_raw(task_dir / "vllm_metrics_post.txt", "B1 Qwen post metrics"),
                )
            else:
                from fr13_floor_gate import GateError, _fixed32_replay_qwen_campaign_proof

                try:
                    replay = _fixed32_replay_qwen_campaign_proof(
                        trace_path,
                        provenance=provenance,
                        expected_task_ids=task_ids,
                    )
                except GateError as error:
                    raise CredentialError(
                        f"Qwen campaign replay failed for {task_id}: {error}"
                    ) from error
        except ContractError as error:
            raise CredentialError(f"Qwen trace/metric replay failed for {task_id}: {error}") from error

        request_ids = replay.get("model_request_ids")
        request_digests = sorted(
            _sha(value.encode("utf-8")) for value in request_ids or []
        )
        request_digest_sha = _sha(_canonical(request_digests))
        if (
            replay.get("completed_logical_model_requests") != completed
            or provenance.get("trace_completed_logical_model_requests") != completed
            or provenance.get("trace_path") != str(trace_path.resolve())
            or provenance.get("trace_sha256") != _sha(trace_raw)
            or provenance.get("trace_bytes") != len(trace_raw)
            or provenance.get("event_count") != len(events)
            or provenance.get("trace_model_request_ids_sha256")
            != request_digest_sha
            or provenance.get("qwen_metric_scope")
            != ("campaign" if batch_size == 4 else "task")
            or provenance.get("hidden_successful_compaction_model_requests")
            != replay.get(
                "hidden_successful_compaction_model_requests",
                replay.get("hidden_compaction_model_requests", 0),
            )
            or provenance.get("hidden_failed_compaction_model_requests")
            != replay.get("hidden_failed_compaction_model_requests", 0)
            or provenance.get("synthetic_compaction_failure_terminal")
            != replay.get("synthetic_compaction_failure_terminal", False)
            or provenance.get("qwen_compaction_metric_evidence")
            != replay.get("qwen_compaction_metric_evidence")
            or not isinstance(provenance.get("agent_terminal"), dict)
            or any(
                provenance["agent_terminal"].get(key) != agent.get(key)
                or type(provenance["agent_terminal"].get(key))
                is not type(agent.get(key))
                for key in ("exit_code", "timed_out", "offloaded", "network_drop")
            )
            or (
                batch_size == 4
                and (
                    metadata.get("fixed32_qwen_campaign_proof") != campaign_identity
                    or provenance.get("qwen_campaign_metric_proof") != campaign_identity
                    or provenance.get("qwen_campaign_metric_evidence_sha256")
                    != campaign_proof.get("metric_evidence_sha256")
                )
            )
        ):
            raise CredentialError(f"Qwen trace provenance differs for {task_id}")
        task_engine_ids = sorted(
            engine_id
            for engine_id, engine_task in successful_engine_ids.items()
            if engine_task == task_id
        )
        if len(task_engine_ids) != completed:
            raise CredentialError(f"trace/engine request count differs for {task_id}")
        if replay.get("engine_id_joinable") is True and request_digests != task_engine_ids:
            raise CredentialError(f"trace/engine request identities differ for {task_id}")
        task_bindings[task_id] = {
            "task_key_id": task_key,
            "start": interval["start_forward_step"],
            "end": interval["end_forward_step"],
            "successful_engine_ids": task_engine_ids,
            "trace_completed_logical_model_requests": completed,
            "trace_sha256": _sha(trace_raw),
        }
    return task_bindings, campaign_identity


def _validate_ledgers(
    proxy_path: Path,
    engine_path: Path,
    *,
    task_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, Any]]:
    from fr13_floor_gate import GateError, load_fixed32_ingress_ledger
    from lumo_flywheel_serving.inference_proxy import (
        fixed32_canonical_task_set_sha256,
        fixed32_task_key_id,
    )

    task_by_key = {fixed32_task_key_id(task_id): task_id for task_id in task_ids}
    task_set_sha = fixed32_canonical_task_set_sha256(tuple(task_ids))
    try:
        proxy_rows, proxy_identity = load_fixed32_ingress_ledger(
            proxy_path,
            role="proxy",
            canonical_task_keys=set(task_by_key),
            canonical_task_set_sha256=task_set_sha,
        )
        engine_rows, engine_identity = load_fixed32_ingress_ledger(
            engine_path,
            role="engine",
            canonical_task_keys=set(task_by_key),
            canonical_task_set_sha256=task_set_sha,
        )
    except GateError as error:
        raise CredentialError(f"fixed32 ingress ledger is invalid: {error}") from error
    proxy_attempts = {
        row["wire_id_sha256"]: row
        for row in proxy_rows
        if row["event"] == "attempt_begin"
    }
    proxy_results = {
        row["wire_id_sha256"]: row
        for row in proxy_rows
        if row["event"] == "attempt_result"
    }
    engine_accepts = {
        row["wire_id_sha256"]: row
        for row in engine_rows
        if row["event"] == "request_accepted"
    }
    engine_completes = {
        row["wire_id_sha256"]: row
        for row in engine_rows
        if row["event"] == "request_complete"
    }
    keys = set(proxy_attempts)
    if (
        not keys
        or len(proxy_attempts) != sum(row["event"] == "attempt_begin" for row in proxy_rows)
        or len(proxy_results) != sum(row["event"] == "attempt_result" for row in proxy_rows)
        or len(engine_accepts) != sum(row["event"] == "request_accepted" for row in engine_rows)
        or len(engine_completes) != sum(row["event"] == "request_complete" for row in engine_rows)
        or keys != set(proxy_results) or keys != set(engine_accepts) or keys != set(engine_completes)
    ):
        raise CredentialError("proxy/engine attempt census is not one-to-one")
    successful: dict[str, str] = {}
    for wire in keys:
        rows = (
            proxy_attempts[wire], proxy_results[wire],
            engine_accepts[wire], engine_completes[wire],
        )
        identities = {
            (
                row["route"], row["task_key_id"], row["wire_id_sha256"],
                row["engine_request_id_sha256"], row["evidence_sha256"],
            )
            for row in rows
        }
        if (
            len(identities) != 1
            or proxy_results[wire].get("status_code") != 200
            or proxy_results[wire].get("outcome") != "response"
            or engine_completes[wire].get("outcome") != "completed"
        ):
            raise CredentialError("proxy/engine request identity or success differs")
        engine_id = proxy_attempts[wire]["engine_request_id_sha256"]
        if engine_id in successful:
            raise CredentialError("engine request digest is duplicated")
        successful[engine_id] = task_by_key[proxy_attempts[wire]["task_key_id"]]
    return proxy_rows, engine_rows, successful, {
        "canonical_task_set_sha256": task_set_sha,
        "proxy": proxy_identity,
        "engine": engine_identity,
    }


def issue_credential(
    *,
    comparator_path: Path,
    subset_path: Path,
    health_path: Path,
    proxy_ledger_path: Path,
    engine_ledger_path: Path,
    work_census_path: Path,
    final_flush_path: Path,
    boundary_snapshot_base: Path,
    runtime_manifest_path: Path,
    source_path: Path,
    repo_path: Path,
    container_env_path: Path,
    task_root: Path,
    source_commit: str,
    mode: str,
    batch_size: int,
    qwen_campaign_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in MODES or batch_size not in CANONICAL_SUBSETS:
        raise CredentialError("mode/batch contract is unsupported")
    if HEX40.fullmatch(source_commit) is None:
        raise CredentialError("source commit is not a full Git SHA-1")
    repo = repo_path.resolve(strict=True)
    if source_path.resolve(strict=True) != repo / SOURCE_RELATIVE:
        raise CredentialError("tree-conv source path is not canonical")

    subset, subset_raw = _json(subset_path, "task subset")
    canonical_subset = CANONICAL_SUBSETS[batch_size]
    task_ids = subset.get("instance_ids")
    if (
        subset_path.resolve(strict=True) != repo / canonical_subset["path"]
        or subset.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or subset.get("split") != "test"
        or task_ids != list(canonical_subset["task_ids"])
        or _sha(subset_raw) != canonical_subset["sha256"]
    ):
        raise CredentialError("task subset is not the pinned SWE-Verified B1/B4 set")

    health, health_raw = _json(health_path, "campaign health")
    health_tasks = health.get("tasks")
    if (
        health.get("swe_orchestrator_rc") != 0
        or not isinstance(health_tasks, list)
        or len(health_tasks) != batch_size
        or {row.get("instance_id") for row in health_tasks if isinstance(row, dict)}
        != set(task_ids)
        or any(
            not isinstance(row, dict)
            or row.get("codex_timed_out") is not False
            or row.get("verdict") not in {"resolved", "failed"}
            for row in health_tasks
        )
    ):
        raise CredentialError("campaign health does not close the exact real task set")

    runtime, runtime_raw, committed = _validate_runtime_manifest(
        runtime_manifest_path,
        repo=repo,
        source_commit=source_commit,
        batch_size=batch_size,
    )
    container_env_raw = _raw(container_env_path, "container environment")
    try:
        env_lines = container_env_raw.decode("utf-8", errors="strict").splitlines()
    except UnicodeDecodeError as error:
        raise CredentialError("container environment is not strict UTF-8") from error
    required_env = {**REQUIRED_CONTAINER_ENV, "FR13_FIXED32_MODE": mode}
    if any(env_lines.count(f"{key}={value}") != 1 for key, value in required_env.items()):
        raise CredentialError("container environment does not bind graph physical32 K64/root1")

    events, work_terminal, work_report, work_raw = _validate_work_census(
        work_census_path, mode=mode, batch_size=batch_size
    )
    descriptor = _topology_descriptor(mode)
    records, comparator_terminal, comparator_raw = _validate_comparator(
        comparator_path,
        events=events,
        work_terminal=work_terminal,
        descriptor=descriptor,
    )
    _result, flush_raw, boundary_path, boundary_raw = _validate_flush(
        final_flush_path,
        boundary_snapshot_base,
        mode=mode,
        work_terminal=work_terminal,
        comparator_terminal=comparator_terminal,
    )
    proxy_rows, engine_rows, successful, ingress = _validate_ledgers(
        proxy_ledger_path, engine_ledger_path, task_ids=task_ids
    )
    task_bindings, campaign_identity = _validate_qwen_and_tasks(
        task_root=task_root,
        task_ids=task_ids,
        mode=mode,
        batch_size=batch_size,
        producer_pid=work_terminal["producer_pid"],
        proxy_rows=proxy_rows,
        successful_engine_ids=successful,
        qwen_campaign_path=qwen_campaign_path,
    )

    memberships = {engine_id: 0 for engine_id in successful}
    task_memberships = {task_id: 0 for task_id in task_ids}
    for event, record in zip(events, records, strict=True):
        for engine_id in event["drafter_runtime"]["request_id_sha256s"]:
            task_id = successful.get(engine_id)
            if task_id is None:
                raise CredentialError("work/comparator request is not authenticated")
            binding = task_bindings[task_id]
            if not binding["start"] <= event["forward_step_index"] < binding["end"]:
                raise CredentialError("work/comparator request lies outside its task boundary")
            if engine_id not in record["request_id_sha256s"]:
                raise CredentialError("comparator request membership differs from work")
            memberships[engine_id] += 1
            task_memberships[task_id] += 1
    if any(count < 1 for count in memberships.values()) or any(
        count < 1 for count in task_memberships.values()
    ):
        raise CredentialError("not every successful real-task engine request reached the comparator")

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "run_classification": (
            "real_swe_verified_exact4_b4_k64_root_treeconv_graph_gate"
            if batch_size == 4
            else "one_real_swe_verified_b1_k64_root_treeconv_graph_gate"
        ),
        "acceptance_valid": False,
        "timing_eligible": False,
        "production_enabled": False,
        "reference_always_served": True,
        "candidate": "physical32_treeconv_zero_tail_v2",
        "mode": mode,
        "topology": descriptor,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "batch_size": batch_size,
        "task_count": batch_size,
        "task_ids": task_ids,
        "task_subset_sha256": _sha(subset_raw),
        "source_commit": source_commit,
        "committed_runtime_sources": committed,
        "container_env_sha256": _sha(container_env_raw),
        "runtime_manifest": _identity(runtime_manifest_path, runtime_raw),
        "runtime_manifest_overall_canonical_sha256": runtime[
            "overall_canonical_sha256"
        ],
        "health": _identity(health_path, health_raw),
        "work_census": _identity(work_census_path, work_raw),
        "work_census_event_count": len(events),
        "work_census_terminal_present": True,
        "work_census_report": work_report,
        "final_flush": _identity(final_flush_path, flush_raw),
        "final_boundary_snapshot": _identity(boundary_path, boundary_raw),
        "ingress": ingress,
        "authenticated_engine_requests": len(successful),
        "all_engine_requests_joined_to_comparator": True,
        "per_task_comparator_memberships": task_memberships,
        "qwen_campaign_proof": campaign_identity,
        "comparison_records": len(records),
        "compared_bytes": sum(record["compared_bytes"] for record in records),
        "comparator": _identity(comparator_path, comparator_raw),
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--proxy-ledger", type=Path, required=True)
    parser.add_argument("--engine-ledger", type=Path, required=True)
    parser.add_argument("--work-census", type=Path, required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot-base", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--container-env", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--batch-size", type=int, choices=(1, 4), required=True)
    parser.add_argument("--qwen-campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = issue_credential(
        comparator_path=args.comparator,
        subset_path=args.subset,
        health_path=args.health,
        proxy_ledger_path=args.proxy_ledger,
        engine_ledger_path=args.engine_ledger,
        work_census_path=args.work_census,
        final_flush_path=args.final_flush,
        boundary_snapshot_base=args.boundary_snapshot_base,
        runtime_manifest_path=args.runtime_manifest,
        source_path=args.source,
        repo_path=args.repo,
        container_env_path=args.container_env,
        task_root=args.task_root,
        source_commit=args.source_commit,
        mode=args.mode,
        batch_size=args.batch_size,
        qwen_campaign_path=args.qwen_campaign,
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
