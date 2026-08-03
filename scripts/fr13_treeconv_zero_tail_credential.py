#!/usr/bin/env python3
"""Issue a source/work-bound real-task tree-conv zero-tail credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.treeconv_zero_tail.credential.v1"
RECORD_SCHEMA = "fr13.fixed32.treeconv_zero_tail.byte_ab.v1"
WORK_SCHEMA = "fr13-fixed32-work-census-v12"
WORK_TERMINAL_SCHEMA = "fr13-fixed32-work-census-terminal-v12"
QWEN_SCHEMA = "fr13-fixed32-qwen-campaign-provenance-v1"
MODES = {"tail6_fixed32": "Tail23", "hydra27_fixed32": "Hydra27"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CredentialError(RuntimeError):
    pass


def _raw(path: Path, label: str) -> bytes:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise CredentialError(f"{label} is unavailable") from error
    if not data:
        raise CredentialError(f"{label} is empty")
    return data


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _raw(path, label)
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"{label} is invalid JSON") from error
    if not isinstance(value, dict):
        raise CredentialError(f"{label} is not an object")
    return value, raw


def _jsonl(path: Path, label: str) -> tuple[list[dict[str, Any]], bytes]:
    raw = _raw(path, label)
    if not raw.endswith(b"\n"):
        raise CredentialError(f"{label} is not newline terminated")
    rows = []
    try:
        for line in raw.splitlines():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise CredentialError(f"{label} row is not an object")
            rows.append(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CredentialError(f"{label} is invalid JSONL") from error
    return rows, raw


def issue_credential(
    *,
    comparator_path: Path,
    subset_path: Path,
    health_path: Path,
    ledger_path: Path,
    work_census_path: Path,
    eager_terminal_path: Path,
    runtime_manifest_path: Path,
    source_path: Path,
    source_commit: str,
    mode: str,
    batch_size: int,
    qwen_campaign_path: Path | None = None,
) -> dict[str, Any]:
    if mode not in MODES or batch_size not in (1, 4):
        raise CredentialError("mode/batch contract is unsupported")
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise CredentialError("source commit is not a full Git SHA-1")

    subset, subset_raw = _json(subset_path, "task subset")
    task_ids = subset.get("instance_ids")
    if (
        subset.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or subset.get("split") != "test"
        or not isinstance(task_ids, list)
        or len(task_ids) != batch_size
        or any(not isinstance(task, str) or not task for task in task_ids)
        or len(set(task_ids)) != batch_size
    ):
        raise CredentialError("task subset is not exact SWE-Verified B1/B4")

    health, health_raw = _json(health_path, "campaign health")
    health_tasks = health.get("tasks")
    if (
        health.get("swe_orchestrator_rc") != 0
        or not isinstance(health_tasks, list)
        or {row.get("instance_id") for row in health_tasks if isinstance(row, dict)}
        != set(task_ids)
        or len(health_tasks) != batch_size
    ):
        raise CredentialError("campaign health does not close the exact task set")

    records, comparator_raw = _jsonl(comparator_path, "tree-conv comparator")
    if not records or len(records) > 320:
        raise CredentialError("tree-conv comparator count is vacuous or over limit")
    state_src_hashes = {row.get("state_src_sha256") for row in records}
    expected_record = {
        "schema": RECORD_SCHEMA,
        "mode": mode,
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "conv_layers": 48,
        "conv_channels": 10240,
        "conv_state_length": 34,
        "source_rows_per_request": 36,
        "live_state_columns": 3,
        "candidate_zero_tail": True,
        "reference_zero_tail": False,
        "reference_restored_and_served": True,
        "timing_eligible": False,
        "byte_equal": True,
        "differing_bytes": 0,
        "first_mismatch_layer": None,
    }
    if any(
        any(row.get(key) != value for key, value in expected_record.items())
        for row in records
    ):
        raise CredentialError("tree-conv comparison contract or bytes differ")
    if [row.get("invocation") for row in records] != list(range(len(records))):
        raise CredentialError("tree-conv comparator invocations are not contiguous")
    if any(
        type(row.get("batch")) is not int
        or not 1 <= row["batch"] <= batch_size
        or type(row.get("compared_bytes")) is not int
        or row["compared_bytes"] != row["batch"] * 48 * 10240 * 34 * 2
        for row in records
    ):
        raise CredentialError("tree-conv comparison byte/batch census is invalid")
    if batch_size not in {row["batch"] for row in records}:
        raise CredentialError("tree-conv comparator never exercised the target batch")
    if len(state_src_hashes) != 1 or HEX64.fullmatch(str(next(iter(state_src_hashes)))) is None:
        raise CredentialError("tree-conv source descriptor binding is invalid")

    work_rows, work_raw = _jsonl(work_census_path, "fixed32 work census")
    terminal_present = work_rows[-1].get("schema") == WORK_TERMINAL_SCHEMA
    work_events = work_rows[:-1] if terminal_present else work_rows
    if (
        not work_events
        or any(row.get("schema") != WORK_SCHEMA for row in work_events)
        or mode not in {row.get("mode") for row in work_events}
        or batch_size not in {row.get("batch_size") for row in work_events}
    ):
        raise CredentialError("fixed32 work census is incomplete or mismatched")
    eager_terminal, eager_terminal_raw = _json(
        eager_terminal_path, "eager diagnostic terminal"
    )
    if eager_terminal != {
        "acceptance_valid": False,
        "flush_protocol_used": False,
        "run_classification": "eager_kernel_byte_diagnostic",
        "schema": "fr13-fixed32-eager-kernel-terminal-v1",
    }:
        raise CredentialError("eager diagnostic terminal contract mismatch")
    if terminal_present:
        raise CredentialError(
            "eager tree-conv diagnostic unexpectedly claimed a graph-census terminal"
        )

    from lumo_flywheel_serving.inference_proxy import (
        fixed32_canonical_task_set_sha256,
        verify_fixed32_ingress_ledger,
    )

    ledger_raw = _raw(ledger_path, "engine ingress ledger")
    ledger = verify_fixed32_ingress_ledger(
        ledger_path, expected_role="engine", require_finalized=True
    )
    ledger_rows = [json.loads(line) for line in ledger_raw.splitlines()]
    expected_set_hash = fixed32_canonical_task_set_sha256(tuple(task_ids))
    if not any(
        row.get("event") == "campaign_begin"
        and row.get("evidence_sha256") == expected_set_hash
        for row in ledger_rows
    ):
        raise CredentialError("engine ingress ledger is not task-set bound")

    source_raw = _raw(source_path, "tree-conv source")
    source_sha256 = _sha(source_raw)
    runtime, runtime_raw = _json(runtime_manifest_path, "runtime manifest")
    closures = runtime.get("closures")
    package_sources = (
        closures.get("python_package_source")
        if isinstance(closures, dict)
        else None
    )
    source_rows = [
        row
        for row in package_sources or []
        if isinstance(row, dict)
        and row.get("path")
        == "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
    ]
    if (
        runtime.get("schema") != "fr13-runtime-manifest-v1"
        or len(source_rows) != 1
        or source_rows[0].get("sha256") != source_sha256
        or source_rows[0].get("size") != len(source_raw)
    ):
        raise CredentialError("runtime manifest does not bind tree-conv source")
    qwen_identity = None
    if batch_size == 4:
        if qwen_campaign_path is None:
            raise CredentialError("B4 credential requires campaign compaction proof")
        qwen, qwen_raw = _json(qwen_campaign_path, "Qwen campaign proof")
        if qwen.get("schema") != QWEN_SCHEMA:
            raise CredentialError("Qwen campaign proof schema mismatch")
        qwen_identity = {"sha256": _sha(qwen_raw), "bytes": len(qwen_raw)}
    elif qwen_campaign_path is not None:
        raise CredentialError("B1 credential forbids a campaign compaction proof")

    state_src_sha256 = str(next(iter(state_src_hashes)))
    return {
        "schema": SCHEMA,
        "status": "pass",
        "run_classification": (
            "real_swe_verified_exact4_b4_k64_root_treeconv_byte_gate"
            if batch_size == 4
            else "one_real_swe_verified_b1_k64_root_treeconv_byte_diagnostic"
        ),
        "acceptance_valid": False,
        "timing_eligible": False,
        "production_enabled": False,
        "reference_always_served": True,
        "candidate": "physical32_treeconv_zero_tail_v1",
        "mode": mode,
        "logical_topology": MODES[mode],
        "physical_drafts": 31,
        "physical_rows_root_inclusive": 32,
        "draft_vocab_root": 1,
        "draft_vocab_k": 65536,
        "batch_size": batch_size,
        "task_count": batch_size,
        "task_ids": task_ids,
        "task_subset_sha256": _sha(subset_raw),
        "source_commit": source_commit,
        "source_file_sha256": source_sha256,
        "state_src_sha256": state_src_sha256,
        "runtime_manifest_sha256": _sha(runtime_raw),
        "work_census_sha256": _sha(work_raw),
        "work_census_event_count": len(work_events),
        "work_census_terminal_present": False,
        "eager_diagnostic_terminal_sha256": _sha(eager_terminal_raw),
        "health_sha256": _sha(health_raw),
        "engine_ingress_ledger_sha256": _sha(ledger_raw),
        "engine_ingress_chain_head_sha256": ledger["chain_head_sha256"],
        "qwen_campaign_proof": qwen_identity,
        "comparison_records": len(records),
        "compared_bytes": sum(int(row["compared_bytes"]) for row in records),
        "comparator_sha256": _sha(comparator_raw),
        "errors": [],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--subset", type=Path, required=True)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--work-census", type=Path, required=True)
    parser.add_argument("--eager-terminal", type=Path, required=True)
    parser.add_argument("--runtime-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
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
        ledger_path=args.ledger,
        work_census_path=args.work_census,
        eager_terminal_path=args.eager_terminal,
        runtime_manifest_path=args.runtime_manifest,
        source_path=args.source,
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
