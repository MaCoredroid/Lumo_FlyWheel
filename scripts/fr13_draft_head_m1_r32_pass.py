#!/usr/bin/env python3
"""Issue or verify the authenticated fixed32 R32 five-site byte credential."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any

try:
    from scripts import fr13_draft_head_m32_pass as common
except ImportError:
    import fr13_draft_head_m32_pass as common


LIVE_SCHEMA = "fr13.fixed32.draft_head_m1_r32_live_ab.v1"
CREDENTIAL_SCHEMA = "fr13.fixed32.draft_head_m1_r32_qualification.v1"
EXPECTED_INSTANCE = "astropy__astropy-12907"
EXPECTED_SOURCE_SHA256 = (
    "56d4fcd551dd022efa00dcb926bb4ea5176e6b5a02fa562ccc1886e007001897"
)
EXPECTED_BINARY_SHA256 = (
    "c389bf5e01b942cfe73b2e4fc05db7b158f16b61205c9f3e9988cbd8a82474dd"
)
EXPECTED_BINARY_BYTES = 113648
EXPECTED_GRAPH_SIGNATURE = (
    "d9a4ddece41d146e9949b9f8ff7c2603b8948d157b28ef69244e44469b36150c"
)
SITE_LABELS = (
    "root",
    "mtp_depth_1",
    "mtp_depth_2",
    "mtp_depth_3",
    "mtp_depth_4",
)
EXPECTED_GEOMETRY = {
    "batch_size": 1,
    "calls_per_event": 5,
    "site_labels": list(SITE_LABELS),
    "values_per_site": 65536,
    "input_shape": [1, 5120],
    "input_stride": [5120, 1],
    "weight_shape": [65536, 5120],
    "weight_stride": [5120, 1],
    "output_shape": [1, 65536],
    "output_stride": [65536, 1],
    "dtype": "torch.bfloat16",
}
EXPECTED_CANDIDATE = {
    "operation": "fr13_bf16_k64_head::gemvx_m1_shuffle_r32_out",
    "grid": [2048, 1, 1],
    "block": [16, 32, 1],
    "rows_per_cta": 32,
    "k_partition_lanes": 16,
    "lane_fmas": 320,
    "reduction_strides": [8, 4, 2, 1],
    "binary_bytes": EXPECTED_BINARY_BYTES,
    "binary_sha256": EXPECTED_BINARY_SHA256,
}
LIVE_KEYS = frozenset(
    {
        "batch_size",
        "boundary_snapshot_sha256",
        "candidate",
        "candidate_binary_bytes",
        "candidate_binary_sha256",
        "candidate_source_sha256",
        "compared_bf16_values",
        "complete_work_census_events",
        "completed_events",
        "concurrency",
        "device_counted_without_measured_host_sync",
        "events_sha256",
        "finalized_by_fixed32_flush",
        "flush_action",
        "flush_generation",
        "flush_nonce",
        "full_logit_comparisons",
        "geometry",
        "graph_lifecycle",
        "instance_id",
        "per_site_compared_bf16_values",
        "per_site_full_logit_comparisons",
        "per_site_raw_bf16_mismatches",
        "performance_measurement",
        "producer_pid",
        "raw_bf16_mismatches",
        "runtime_source_sha256",
        "schema",
        "served_return",
        "site_labels",
        "source_commit",
        "status",
        "suite",
        "task_marker",
        "work_census_last_event_index",
    }
)
GRAPH_KEYS = frozenset(
    {
        "capture_origin",
        "captured_loop_calls",
        "drafter_graph_id",
        "drafter_graph_signature",
        "fallback_calls",
        "last_measured_forward_step_index",
        "observed_measured_replays",
        "selected_root_calls",
    }
)
CREDENTIAL_KEYS = frozenset(
    {
        "authenticated_one_task_completion",
        "boundary_snapshot_sha256",
        "candidate",
        "candidate_binary_bytes",
        "candidate_binary_sha256",
        "candidate_source_sha256",
        "canonical_sha256",
        "chat_traffic_audit_sha256",
        "completed_events",
        "events_sha256",
        "final_flush_sha256",
        "flush_generation",
        "geometry",
        "graph_lifecycle",
        "instance_id",
        "live_result_canonical_sha256",
        "live_result_sha256",
        "per_site_compared_bf16_values",
        "per_site_full_logit_comparisons",
        "per_site_raw_bf16_mismatches",
        "performance_measurement",
        "production_admitted",
        "production_default_enabled",
        "runtime_source_sha256",
        "schema",
        "served_return",
        "source_commit",
        "status",
        "trace_completed_logical_model_requests",
    }
)


def _sha256_bytes(raw: bytes) -> str:
    return common._digest_bytes(raw)


def _require_exact_dict(
    value: Any, expected: frozenset[str], label: str
) -> dict[str, Any]:
    return common._require_exact_keys(value, expected, label)


def _validate_live_result(
    payload: dict[str, Any],
    *,
    expected_source_commit: str,
    expected_runtime_source_sha256: str,
) -> dict[str, Any]:
    common._require_commit(expected_source_commit, "expected source commit")
    common._require_sha256(
        expected_runtime_source_sha256, "expected runtime source"
    )
    _require_exact_dict(payload, LIVE_KEYS, "R32 live result")
    if (
        payload["schema"] != LIVE_SCHEMA
        or payload["status"] != "PASS"
        or payload["suite"] != "SWE-Verified"
        or payload["instance_id"] != EXPECTED_INSTANCE
        or payload["task_marker"] != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload["concurrency"] != 1
        or payload["batch_size"] != 1
        or payload["source_commit"] != expected_source_commit
        or payload["runtime_source_sha256"] != expected_runtime_source_sha256
        or payload["candidate_source_sha256"] != EXPECTED_SOURCE_SHA256
        or payload["candidate_binary_sha256"] != EXPECTED_BINARY_SHA256
        or payload["candidate_binary_bytes"] != EXPECTED_BINARY_BYTES
        or payload["geometry"] != EXPECTED_GEOMETRY
        or payload["candidate"] != EXPECTED_CANDIDATE
        or payload["site_labels"] != list(SITE_LABELS)
        or payload["served_return"]
        != "incumbent BF16 K64 reference logits unchanged"
        or payload["performance_measurement"] is not False
        or payload["device_counted_without_measured_host_sync"] is not True
        or payload["finalized_by_fixed32_flush"] is not True
        or payload["flush_action"] != "final"
    ):
        raise ValueError("R32 live result provenance or serving mode drifted")

    completed = common._require_positive_int(
        payload["completed_events"], "R32 completed events"
    )
    census = common._require_positive_int(
        payload["complete_work_census_events"], "R32 work census events"
    )
    if (
        census != completed
        or payload["work_census_last_event_index"] != completed - 1
        or common._require_positive_int(
            payload["flush_generation"], "R32 flush generation"
        )
        < 1
        or common._require_positive_int(
            payload["producer_pid"], "R32 producer PID"
        )
        < 1
    ):
        raise ValueError("R32 live result completion census drifted")
    common._require_sha256(payload["events_sha256"], "R32 event census")
    common._require_sha256(payload["flush_nonce"], "R32 flush nonce")
    common._require_sha256(
        payload["boundary_snapshot_sha256"], "R32 boundary snapshot"
    )

    comparisons = _require_exact_dict(
        payload["per_site_full_logit_comparisons"],
        frozenset(SITE_LABELS),
        "R32 per-site comparisons",
    )
    values = _require_exact_dict(
        payload["per_site_compared_bf16_values"],
        frozenset(SITE_LABELS),
        "R32 per-site values",
    )
    mismatches = _require_exact_dict(
        payload["per_site_raw_bf16_mismatches"],
        frozenset(SITE_LABELS),
        "R32 per-site mismatches",
    )
    for site in SITE_LABELS:
        common._require_positive_int(
            comparisons[site], f"R32 per-site comparisons.{site}"
        )
        common._require_positive_int(values[site], f"R32 per-site values.{site}")
        common._require_nonnegative_int(
            mismatches[site], f"R32 per-site mismatches.{site}"
        )
    common._require_positive_int(
        payload["full_logit_comparisons"], "R32 full-logit comparisons"
    )
    common._require_positive_int(
        payload["compared_bf16_values"], "R32 compared BF16 values"
    )
    common._require_nonnegative_int(
        payload["raw_bf16_mismatches"], "R32 raw BF16 mismatches"
    )
    if (
        comparisons != {site: completed for site in SITE_LABELS}
        or values != {site: completed * 65536 for site in SITE_LABELS}
        or mismatches != {site: 0 for site in SITE_LABELS}
        or payload["full_logit_comparisons"] != completed * len(SITE_LABELS)
        or payload["compared_bf16_values"]
        != completed * len(SITE_LABELS) * 65536
        or payload["raw_bf16_mismatches"] != 0
    ):
        raise ValueError("R32 five-site raw BF16 equality census drifted")

    graph = _require_exact_dict(
        payload["graph_lifecycle"], GRAPH_KEYS, "R32 graph lifecycle"
    )
    for key in (
        "selected_root_calls",
        "captured_loop_calls",
        "fallback_calls",
        "observed_measured_replays",
        "drafter_graph_id",
        "last_measured_forward_step_index",
    ):
        common._require_nonnegative_int(graph[key], f"R32 graph lifecycle.{key}")
    if (
        graph["selected_root_calls"] != 1
        or graph["captured_loop_calls"] != 4
        or graph["fallback_calls"] != 0
        or type(graph["observed_measured_replays"]) is not int
        or graph["observed_measured_replays"] < completed
        or type(graph["drafter_graph_id"]) is not int
        or graph["drafter_graph_id"] < 1
        or graph["drafter_graph_signature"] != EXPECTED_GRAPH_SIGNATURE
        or graph["capture_origin"] not in {"measured", "unmeasured"}
        or type(graph["last_measured_forward_step_index"]) is not int
        or graph["last_measured_forward_step_index"] < completed - 1
    ):
        raise ValueError("R32 graph replay lifecycle drifted")
    return {
        "completed_events": completed,
        "events_sha256": payload["events_sha256"],
        "flush_generation": payload["flush_generation"],
        "graph_lifecycle": graph,
        "per_site_full_logit_comparisons": comparisons,
        "per_site_compared_bf16_values": values,
        "per_site_raw_bf16_mismatches": mismatches,
    }


def _bind_candidate_files(
    *, runtime_source: Path, candidate_source: Path, candidate_binary: Path
) -> tuple[str, str, str]:
    common.require_regular_file(runtime_source, "R32 runtime source")
    common.require_regular_file(candidate_source, "R32 candidate source")
    common.require_regular_file(candidate_binary, "R32 candidate binary")
    runtime_sha = common.sha256_file(runtime_source)
    source_sha = common.sha256_file(candidate_source)
    binary_sha = common.sha256_file(candidate_binary)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError("R32 candidate source SHA-256 mismatch")
    if (
        binary_sha != EXPECTED_BINARY_SHA256
        or candidate_binary.stat().st_size != EXPECTED_BINARY_BYTES
    ):
        raise ValueError("R32 candidate binary identity mismatch")
    return runtime_sha, source_sha, binary_sha


def _bind_repo(repo: Path, expected_source_commit: str) -> Path:
    repo = repo.resolve(strict=True)
    if not repo.is_dir():
        raise ValueError("R32 repository path is not a directory")

    def run_git(*args: str) -> bytes:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode != 0:
            raise ValueError(f"R32 git {' '.join(args)} failed")
        return result.stdout

    try:
        head = run_git("rev-parse", "HEAD").decode("ascii").strip()
    except UnicodeDecodeError as error:
        raise ValueError("R32 repository HEAD is not ASCII") from error
    if head != expected_source_commit:
        raise ValueError("R32 source commit does not equal repository HEAD")
    if run_git("status", "--porcelain=v1", "--untracked-files=no"):
        raise ValueError("R32 repository has tracked source changes")
    return repo


def issue_credential(
    *,
    live_result: Path,
    expected_live_sha256: str,
    final_flush: Path,
    boundary_snapshot: Path,
    chat_traffic_audit: Path,
    runtime_source: Path,
    candidate_source: Path,
    candidate_binary: Path,
    expected_source_commit: str,
    out: Path,
    repo: Path,
) -> dict[str, Any]:
    expected_live_sha256 = common._require_sha256(
        expected_live_sha256, "R32 live result"
    )
    expected_source_commit = common._require_commit(
        expected_source_commit, "R32 source commit"
    )
    repo = _bind_repo(repo, expected_source_commit)
    runtime_sha, source_sha, binary_sha = _bind_candidate_files(
        runtime_source=runtime_source,
        candidate_source=candidate_source,
        candidate_binary=candidate_binary,
    )
    evidence_paths = {
        "final_flush": final_flush,
        "boundary_snapshot": boundary_snapshot,
        "chat_traffic_audit": chat_traffic_audit,
    }
    for label, path in evidence_paths.items():
        common.require_regular_file(path, f"R32 {label}")
    evidence_sha256 = {
        label: common.sha256_file(path) for label, path in evidence_paths.items()
    }
    live, live_raw = common.load_json(live_result)
    if _sha256_bytes(live_raw) != expected_live_sha256:
        raise ValueError("R32 live result raw SHA-256 mismatch")
    summary = _validate_live_result(
        live,
        expected_source_commit=expected_source_commit,
        expected_runtime_source_sha256=runtime_sha,
    )
    terminal = common.validate_live_evidence(
        live_payload=live,
        final_flush_path=final_flush,
        boundary_snapshot_path=boundary_snapshot,
    )
    traffic = common.validate_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        expected_events=summary["completed_events"],
    )
    common.validate_rebuilt_chat_traffic_audit(
        audit_path=chat_traffic_audit,
        repo=repo,
    )
    if (
        any(
            common.sha256_file(path) != evidence_sha256[label]
            for label, path in evidence_paths.items()
        )
        or terminal["completed_events"] != summary["completed_events"]
        or terminal["events_sha256"] != summary["events_sha256"]
        or terminal["boundary_snapshot_sha256"]
        != evidence_sha256["boundary_snapshot"]
        or traffic["completed_events"] != summary["completed_events"]
        or traffic["chat_traffic_audit_sha256"]
        != evidence_sha256["chat_traffic_audit"]
    ):
        raise ValueError("R32 authenticated terminal evidence does not reconcile")

    body = {
        "schema": CREDENTIAL_SCHEMA,
        "status": "PASS",
        "instance_id": EXPECTED_INSTANCE,
        "source_commit": expected_source_commit,
        "runtime_source_sha256": runtime_sha,
        "candidate_source_sha256": source_sha,
        "candidate_binary_sha256": binary_sha,
        "candidate_binary_bytes": EXPECTED_BINARY_BYTES,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _sha256_bytes(
            common.canonical_bytes(live)
        ),
        "completed_events": summary["completed_events"],
        "events_sha256": summary["events_sha256"],
        "flush_generation": summary["flush_generation"],
        "final_flush_sha256": evidence_sha256["final_flush"],
        "boundary_snapshot_sha256": terminal["boundary_snapshot_sha256"],
        "chat_traffic_audit_sha256": traffic["chat_traffic_audit_sha256"],
        "trace_completed_logical_model_requests": traffic[
            "trace_completed_logical_model_requests"
        ],
        "geometry": EXPECTED_GEOMETRY,
        "candidate": EXPECTED_CANDIDATE,
        "graph_lifecycle": summary["graph_lifecycle"],
        "per_site_full_logit_comparisons": summary[
            "per_site_full_logit_comparisons"
        ],
        "per_site_compared_bf16_values": summary[
            "per_site_compared_bf16_values"
        ],
        "per_site_raw_bf16_mismatches": summary[
            "per_site_raw_bf16_mismatches"
        ],
        "served_return": "incumbent BF16 K64 reference logits unchanged",
        "authenticated_one_task_completion": True,
        "performance_measurement": False,
        "production_default_enabled": False,
        "production_admitted": False,
    }
    credential = dict(body)
    credential["canonical_sha256"] = _sha256_bytes(common.canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace R32 qualification credential: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(common.canonical_bytes(credential) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return credential


def verify_credential(
    *,
    credential_path: Path,
    expected_credential_sha256: str,
    runtime_source: Path,
    candidate_source: Path,
    candidate_binary: Path,
    expected_source_commit: str,
    repo: Path,
) -> dict[str, Any]:
    expected_credential_sha256 = common._require_sha256(
        expected_credential_sha256, "R32 qualification credential"
    )
    expected_source_commit = common._require_commit(
        expected_source_commit, "R32 source commit"
    )
    _bind_repo(repo, expected_source_commit)
    runtime_sha, source_sha, binary_sha = _bind_candidate_files(
        runtime_source=runtime_source,
        candidate_source=candidate_source,
        candidate_binary=candidate_binary,
    )
    payload, raw = common.load_json(credential_path)
    if _sha256_bytes(raw) != expected_credential_sha256:
        raise ValueError("R32 qualification credential raw SHA-256 mismatch")
    _require_exact_dict(payload, CREDENTIAL_KEYS, "R32 qualification credential")
    canonical_sha = payload.pop("canonical_sha256", None)
    if common._require_sha256(canonical_sha, "R32 credential canonical") != (
        _sha256_bytes(common.canonical_bytes(payload))
    ):
        raise ValueError("R32 qualification credential canonical digest mismatch")
    completed = common._require_positive_int(
        payload.get("completed_events"), "R32 events"
    )
    comparisons = payload.get("per_site_full_logit_comparisons")
    values = payload.get("per_site_compared_bf16_values")
    mismatches = payload.get("per_site_raw_bf16_mismatches")
    graph = payload.get("graph_lifecycle")
    comparisons = _require_exact_dict(
        comparisons, frozenset(SITE_LABELS), "R32 credential per-site comparisons"
    )
    values = _require_exact_dict(
        values, frozenset(SITE_LABELS), "R32 credential per-site values"
    )
    mismatches = _require_exact_dict(
        mismatches, frozenset(SITE_LABELS), "R32 credential per-site mismatches"
    )
    graph = _require_exact_dict(graph, GRAPH_KEYS, "R32 credential graph lifecycle")
    for site in SITE_LABELS:
        common._require_positive_int(
            comparisons[site], f"R32 credential per-site comparisons.{site}"
        )
        common._require_positive_int(
            values[site], f"R32 credential per-site values.{site}"
        )
        common._require_nonnegative_int(
            mismatches[site], f"R32 credential per-site mismatches.{site}"
        )
    for key in (
        "selected_root_calls",
        "captured_loop_calls",
        "fallback_calls",
        "observed_measured_replays",
        "drafter_graph_id",
        "last_measured_forward_step_index",
    ):
        common._require_nonnegative_int(
            graph[key], f"R32 credential graph lifecycle.{key}"
        )
    if (
        payload.get("schema") != CREDENTIAL_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("source_commit") != expected_source_commit
        or payload.get("runtime_source_sha256") != runtime_sha
        or payload.get("candidate_source_sha256") != source_sha
        or payload.get("candidate_binary_sha256") != binary_sha
        or payload.get("candidate_binary_bytes") != EXPECTED_BINARY_BYTES
        or payload.get("geometry") != EXPECTED_GEOMETRY
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("served_return")
        != "incumbent BF16 K64 reference logits unchanged"
        or payload.get("authenticated_one_task_completion") is not True
        or payload.get("performance_measurement") is not False
        or payload.get("production_default_enabled") is not False
        or payload.get("production_admitted") is not False
        or comparisons != {site: completed for site in SITE_LABELS}
        or values != {site: completed * 65536 for site in SITE_LABELS}
        or mismatches != {site: 0 for site in SITE_LABELS}
        or graph.get("selected_root_calls") != 1
        or graph.get("captured_loop_calls") != 4
        or graph.get("fallback_calls") != 0
        or type(graph.get("observed_measured_replays")) is not int
        or graph["observed_measured_replays"] < completed
        or type(graph.get("drafter_graph_id")) is not int
        or graph["drafter_graph_id"] < 1
        or graph.get("drafter_graph_signature") != EXPECTED_GRAPH_SIGNATURE
        or graph.get("capture_origin") not in {"measured", "unmeasured"}
        or type(graph.get("last_measured_forward_step_index")) is not int
        or graph["last_measured_forward_step_index"] < completed - 1
    ):
        raise ValueError("R32 qualification credential contract drifted")
    for key in (
        "live_result_sha256",
        "live_result_canonical_sha256",
        "events_sha256",
        "final_flush_sha256",
        "boundary_snapshot_sha256",
        "chat_traffic_audit_sha256",
    ):
        common._require_sha256(payload.get(key), f"R32 credential {key}")
    common._require_positive_int(
        payload.get("trace_completed_logical_model_requests"),
        "R32 authenticated requests",
    )
    common._require_positive_int(
        payload.get("flush_generation"), "R32 flush generation"
    )
    payload["canonical_sha256"] = canonical_sha
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--live-result", required=True, type=Path)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--final-flush", required=True, type=Path)
    issue.add_argument("--boundary-snapshot", required=True, type=Path)
    issue.add_argument("--chat-traffic-audit", required=True, type=Path)
    issue.add_argument("--runtime-source", required=True, type=Path)
    issue.add_argument("--candidate-source", required=True, type=Path)
    issue.add_argument("--candidate-binary", required=True, type=Path)
    issue.add_argument("--expected-source-commit", required=True)
    issue.add_argument("--out", required=True, type=Path)
    issue.add_argument("--repo", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--credential", required=True, type=Path)
    verify.add_argument("--expected-credential-sha256", required=True)
    verify.add_argument("--runtime-source", required=True, type=Path)
    verify.add_argument("--candidate-source", required=True, type=Path)
    verify.add_argument("--candidate-binary", required=True, type=Path)
    verify.add_argument("--expected-source-commit", required=True)
    verify.add_argument("--repo", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "issue":
        result = issue_credential(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            final_flush=args.final_flush,
            boundary_snapshot=args.boundary_snapshot,
            chat_traffic_audit=args.chat_traffic_audit,
            runtime_source=args.runtime_source,
            candidate_source=args.candidate_source,
            candidate_binary=args.candidate_binary,
            expected_source_commit=args.expected_source_commit,
            out=args.out,
            repo=args.repo,
        )
    else:
        result = verify_credential(
            credential_path=args.credential,
            expected_credential_sha256=args.expected_credential_sha256,
            runtime_source=args.runtime_source,
            candidate_source=args.candidate_source,
            candidate_binary=args.candidate_binary,
            expected_source_commit=args.expected_source_commit,
            repo=args.repo,
        )
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
