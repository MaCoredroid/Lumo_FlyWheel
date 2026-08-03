#!/usr/bin/env python3
"""Issue a source/work-bound real-task tree-conv zero-tail credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


SCHEMA = "fr13.fixed32.treeconv_zero_tail.credential.v1"
RECORD_SCHEMA = "fr13.fixed32.treeconv_zero_tail.byte_ab.v1"
WORK_SCHEMA = "fr13-fixed32-work-census-v12"
WORK_TERMINAL_SCHEMA = "fr13-fixed32-work-census-terminal-v12"
QWEN_SCHEMA = "fr13-fixed32-qwen-campaign-provenance-v1"
MODES = {"tail6_fixed32": "Tail23", "hydra27_fixed32": "Hydra27"}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
SOURCE_RELATIVE = "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
CANONICAL_SUBSETS = {
    1: {
        "sha256": "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb",
        "task_ids": ("astropy__astropy-12907",),
    },
    4: {
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
    "FR13_DRAFT_VOCAB_ROOT": "1",
    "FR13_DRAFT_VOCAB_K": "65536",
    "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
    "FR13_FIXED32_PHYSICAL_DRAFTS": "31",
    "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL": "0",
    "FR13_FIXED32_CONV_COMMIT_ZERO_TAIL_BYTE_AB": "1",
}


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


def _artifact_identity(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha(raw), "bytes": len(raw)}


def _validate_artifact_identity(
    identity: object, path: Path, raw: bytes, label: str
) -> None:
    if identity != _artifact_identity(path, raw):
        raise CredentialError(f"{label} identity mismatch")


def _committed_source_bytes(repo: Path, source_commit: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "show", f"{source_commit}:{SOURCE_RELATIVE}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        raise CredentialError("source commit cannot provide the tree-conv source") from error
    if not result.stdout:
        raise CredentialError("source commit tree-conv source is empty")
    return result.stdout


def _validate_qwen_campaign(
    proof_path: Path, task_ids: list[str]
) -> tuple[dict[str, Any], bytes]:
    proof, proof_raw = _json(proof_path, "Qwen campaign proof")
    canonical = (
        json.dumps(proof, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("ascii")
    if proof_raw != canonical:
        raise CredentialError("Qwen campaign proof is not canonical JSON")
    if set(proof) != {
        "schema", "metric_scope", "concurrency", "task_ids", "selection",
        "metrics_pre", "metrics_post", "tasks", "metric_evidence_sha256",
        "metric_evidence",
    }:
        raise CredentialError("Qwen campaign proof keys mismatch")
    if (
        proof.get("schema") != QWEN_SCHEMA
        or proof.get("metric_scope") != "concurrent_campaign_union"
        or proof.get("concurrency") != 4
        or proof.get("task_ids") != task_ids
        or proof.get("selection")
        != {
            "basis": "runner_owned_campaign_endpoint_metrics",
            "task_boundary_schema": "fr13-fixed32-eager-kernel-diagnostic-task-bracket-v1",
            "task_stream_coverage": None,
        }
        or HEX64.fullmatch(str(proof.get("metric_evidence_sha256"))) is None
        or not isinstance(proof.get("metric_evidence"), dict)
    ):
        raise CredentialError("Qwen campaign union contract mismatch")

    dataset_dir = proof_path.resolve().parent
    for key, filename in (
        ("metrics_pre", "fixed32_qwen_campaign_metrics_pre.txt"),
        ("metrics_post", "fixed32_qwen_campaign_metrics_post.txt"),
    ):
        artifact = dataset_dir / filename
        _validate_artifact_identity(
            proof.get(key), artifact, _raw(artifact, f"Qwen {key}"), f"Qwen {key}"
        )

    from lumo_flywheel_serving.inference_proxy import fixed32_task_key_id

    tasks = proof.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != len(task_ids):
        raise CredentialError("Qwen campaign task proof count mismatch")
    proof_identity = _artifact_identity(proof_path, proof_raw)
    for index, task_id in enumerate(task_ids):
        task = tasks[index]
        if (
            not isinstance(task, dict)
            or set(task) != {
                "instance_id", "task_key_id",
                "expected_completed_logical_model_requests", "trace",
            }
            or task.get("instance_id") != task_id
            or task.get("task_key_id") != fixed32_task_key_id(task_id)
            or type(task.get("expected_completed_logical_model_requests")) is not int
            or task["expected_completed_logical_model_requests"] <= 0
        ):
            raise CredentialError("Qwen campaign ordered task contract mismatch")
        task_dir = dataset_dir / "per_task" / task_id
        trace = task_dir / "qwen_trace.jsonl"
        _validate_artifact_identity(
            task.get("trace"), trace, _raw(trace, f"Qwen trace {task_id}"),
            f"Qwen trace {task_id}",
        )
        metadata, _ = _json(task_dir / "runner_metadata.json", f"runner metadata {task_id}")
        provenance = metadata.get("fixed32_real_task_provenance")
        if (
            metadata.get("instance_id") != task_id
            or metadata.get("fixed32_qwen_campaign_proof") != proof_identity
            or not isinstance(provenance, dict)
            or provenance.get("schema") != "fr13-fixed32-real-task-provenance-v3"
            or provenance.get("instance_id") != task_id
            or provenance.get("qwen_metric_scope") != "campaign"
            or provenance.get("qwen_campaign_metric_proof") != proof_identity
            or provenance.get("qwen_campaign_metric_evidence_sha256")
            != proof["metric_evidence_sha256"]
        ):
            raise CredentialError("Qwen per-task provenance binding mismatch")
    return proof, proof_raw


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
    repo_path: Path,
    container_env_path: Path,
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
    canonical_subset = CANONICAL_SUBSETS[batch_size]
    if (
        subset.get("dataset_name") != "princeton-nlp/SWE-bench_Verified"
        or subset.get("split") != "test"
        or task_ids != list(canonical_subset["task_ids"])
        or _sha(subset_raw) != canonical_subset["sha256"]
    ):
        raise CredentialError("task subset is not the pinned SWE-Verified B1/B4 subset")

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
    from fr13_fixed32_work_census import (
        CensusError,
        normalized_work_sha256,
        validate_event,
    )

    try:
        validated = [
            validate_event(row, source=f"fixed32 work census:{index + 1}")
            for index, row in enumerate(work_events)
        ]
    except CensusError as error:
        raise CredentialError(f"fixed32 work census exact-count mismatch: {error}") from error
    signatures = {
        normalized_work_sha256(event.normalized_work) for event in validated
    }
    if (
        not validated
        or any(event.mode != mode for event in validated)
        or batch_size not in {event.batch_size for event in validated}
        or [event.event_index for event in validated] != list(range(len(validated)))
        or any(
            right.forward_step_index <= left.forward_step_index
            for left, right in zip(validated, validated[1:])
        )
        or len({event.producer_pid for event in validated}) != 1
        or len(signatures) != 1
    ):
        raise CredentialError("fixed32 work census sequence or physical work mismatch")
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

    repo = repo_path.resolve()
    expected_source_path = repo / SOURCE_RELATIVE
    if source_path.resolve() != expected_source_path:
        raise CredentialError("tree-conv source is not the repository source path")
    source_raw = _raw(source_path, "tree-conv source")
    if source_raw != _committed_source_bytes(repo, source_commit):
        raise CredentialError("source commit does not bind live tree-conv source bytes")
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
        and row.get("path") == SOURCE_RELATIVE
    ]
    if (
        runtime.get("schema") != "fr13-runtime-manifest-v1"
        or len(source_rows) != 1
        or source_rows[0].get("sha256") != source_sha256
        or source_rows[0].get("size") != len(source_raw)
    ):
        raise CredentialError("runtime manifest does not bind tree-conv source")
    container_env_raw = _raw(container_env_path, "container environment")
    try:
        container_env_lines = container_env_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise CredentialError("container environment is not UTF-8") from error
    required_env = {**REQUIRED_CONTAINER_ENV, "FR13_FIXED32_MODE": mode}
    if any(
        container_env_lines.count(f"{key}={value}") != 1
        for key, value in required_env.items()
    ):
        raise CredentialError("container environment does not bind physical32 K64/root1 gate")

    qwen_identity = None
    if batch_size == 4:
        if qwen_campaign_path is None:
            raise CredentialError("B4 credential requires campaign compaction proof")
        _, qwen_raw = _validate_qwen_campaign(qwen_campaign_path, task_ids)
        qwen_identity = _artifact_identity(qwen_campaign_path, qwen_raw)
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
        "container_env_sha256": _sha(container_env_raw),
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
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--container-env", type=Path, required=True)
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
        repo_path=args.repo,
        container_env_path=args.container_env,
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
