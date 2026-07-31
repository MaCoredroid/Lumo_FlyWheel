#!/usr/bin/env python3
"""Issue and verify the canonical unified-attention BM8 production pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.dfwd_unified_bm8_live_ab.v1"
IDENTITY_SCHEMA = "fr13.fixed32.dfwd_unified_bm8.identity.v1"
SIDECAR_SCHEMA = "fr13.fixed32.dfwd_unified_bm8_production_pass.v1"
HEX = frozenset("0123456789abcdef")
EXPECTED_INSTANCE = "astropy__astropy-12907"
EXPECTED_CANDIDATE = {
    "kernel": "kernel_unified_attention_2d",
    "stock_block_m": 16,
    "stock_block_q": 2,
    "candidate_block_m": 8,
    "candidate_block_q": 1,
    "required_calls": 4,
}
EXPECTED_FILES = {
    "patcher": "/workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py",
    "unified_attention": (
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/attention/ops/"
        "triton_unified_attention.py"
    ),
    "eagle_replay_hook": (
        "/usr/local/lib/python3.12/dist-packages/vllm/v1/spec_decode/eagle.py"
    ),
}


def _digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path: Path) -> tuple[dict[str, Any], bytes]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"not a regular non-symlink file: {path}")
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("JSON root must be an object")
    return payload, raw


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _validate_identity(
    identity: Any,
    *,
    expected_candidate_source_sha256: str,
) -> dict[str, str]:
    if not isinstance(identity, dict):
        raise ValueError("BM8 live candidate identity is not an object")
    source_commit = identity.get("source_commit")
    if (
        identity.get("schema") != IDENTITY_SCHEMA
        or identity.get("production_enabled") is not False
        or identity.get("candidate") != EXPECTED_CANDIDATE
        or not isinstance(source_commit, str)
        or len(source_commit) != 40
        or any(c not in HEX for c in source_commit)
    ):
        raise ValueError("BM8 live candidate identity drifted")
    files = identity.get("files")
    if not isinstance(files, dict) or set(files) != set(EXPECTED_FILES):
        raise ValueError("BM8 live candidate source set drifted")
    digests: dict[str, str] = {}
    for label, expected_path in EXPECTED_FILES.items():
        row = files.get(label)
        if not isinstance(row, dict) or row.get("path") != expected_path:
            raise ValueError(f"BM8 live candidate path drifted for {label}")
        digests[label] = _require_sha256(
            row.get("sha256"), f"BM8 live {label}"
        )
    if digests["unified_attention"] != expected_candidate_source_sha256:
        raise ValueError("BM8 qualified candidate source SHA-256 mismatch")
    digests["source_commit"] = source_commit
    return digests


def validate_live_result(
    payload: dict[str, Any],
    *,
    expected_candidate_source_sha256: str,
) -> dict[str, str]:
    expected_candidate_source_sha256 = _require_sha256(
        expected_candidate_source_sha256, "qualified candidate source"
    )
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("BM8 live result is not a PASS record")
    identity = payload.get("candidate_identity")
    digests = _validate_identity(
        identity,
        expected_candidate_source_sha256=expected_candidate_source_sha256,
    )
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("task_marker") != f"swe_verified:{EXPECTED_INSTANCE}"
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("candidate_dispatch")
        != "launcher-private BM8 exact B1 selector"
        or payload.get("candidate_dispatches") != 4
        or payload.get("served_return")
        != "stock captured drafter graph unchanged"
        or payload.get("performance_measurement") is not False
    ):
        raise ValueError("BM8 live result provenance drifted")
    expected_geometry = {
        "query_shape": [1, 24, 256],
        "kv_heads": 4,
        "stock_block_m": 16,
        "stock_block_q": 2,
        "candidate_block_m": 8,
        "candidate_block_q": 1,
        "valid_query_heads_per_kv": 6,
    }
    if payload.get("geometry") != expected_geometry:
        raise ValueError("BM8 live operand geometry drifted")
    calls = payload.get("calls")
    if not isinstance(calls, list) or len(calls) != 4:
        raise ValueError("BM8 live result does not contain four calls")
    for index, row in enumerate(calls):
        if not isinstance(row, dict):
            raise ValueError(f"BM8 live call {index} is not an object")
        stock = _require_sha256(row.get("stock_sha256"), f"call {index} stock")
        candidate = _require_sha256(
            row.get("candidate_sha256"), f"call {index} candidate"
        )
        if (
            row.get("call_index") != index
            or type(row.get("seq_len")) is not int
            or row["seq_len"] <= 0
            or row.get("bytes") != 12288
            or row.get("raw_byte_mismatches") != 0
            or candidate != stock
        ):
            raise ValueError(f"BM8 live call {index} is not an exact byte PASS")
    return digests


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    expected_candidate_source_sha256: str,
    out: Path,
) -> dict[str, Any]:
    expected_live_sha256 = _require_sha256(expected_live_sha256, "live result")
    expected_candidate_source_sha256 = _require_sha256(
        expected_candidate_source_sha256, "qualified candidate source"
    )
    live_payload, live_raw = load_json(live_result)
    if _digest_bytes(live_raw) != expected_live_sha256:
        raise ValueError("BM8 live result raw SHA-256 mismatch")
    digests = validate_live_result(
        live_payload,
        expected_candidate_source_sha256=expected_candidate_source_sha256,
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "live_gate_schema": LIVE_SCHEMA,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest_bytes(
            canonical_bytes(live_payload)
        ),
        "instance_id": EXPECTED_INSTANCE,
        "qualified_source_commit": digests["source_commit"],
        "qualified_patcher_sha256": digests["patcher"],
        "qualified_unified_attention_sha256": digests["unified_attention"],
        "qualified_eagle_replay_hook_sha256": digests["eagle_replay_hook"],
        "candidate": EXPECTED_CANDIDATE,
        "candidate_artifact_kind": "triton_jit_source",
        "required_runtime": "fixed32 B1 FULL",
        "production_scope": "four exact B1 MTP unified-attention calls",
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest_bytes(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace BM8 pass sidecar: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_bytes(sidecar) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return sidecar


def verify_sidecar(
    *,
    sidecar_path: Path,
    expected_sidecar_sha256: str,
    candidate_source: Path,
    expected_candidate_source_sha256: str,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "BM8 pass sidecar"
    )
    expected_candidate_source_sha256 = _require_sha256(
        expected_candidate_source_sha256, "qualified candidate source"
    )
    payload, raw = load_json(sidecar_path)
    if _digest_bytes(raw) != expected_sidecar_sha256:
        raise ValueError("BM8 pass sidecar raw SHA-256 mismatch")
    required = {
        "schema",
        "status",
        "live_gate_schema",
        "live_result_sha256",
        "live_result_canonical_sha256",
        "instance_id",
        "qualified_source_commit",
        "qualified_patcher_sha256",
        "qualified_unified_attention_sha256",
        "qualified_eagle_replay_hook_sha256",
        "candidate",
        "candidate_artifact_kind",
        "required_runtime",
        "production_scope",
        "canonical_sha256",
    }
    if set(payload) != required:
        raise ValueError("BM8 pass sidecar key set drifted")
    canonical_sha256 = payload.pop("canonical_sha256")
    if _require_sha256(canonical_sha256, "sidecar canonical") != _digest_bytes(
        canonical_bytes(payload)
    ):
        raise ValueError("BM8 pass sidecar canonical digest mismatch")
    if (
        payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("instance_id") != EXPECTED_INSTANCE
        or payload.get("qualified_unified_attention_sha256")
        != expected_candidate_source_sha256
        or payload.get("candidate") != EXPECTED_CANDIDATE
        or payload.get("candidate_artifact_kind") != "triton_jit_source"
        or payload.get("required_runtime") != "fixed32 B1 FULL"
        or payload.get("production_scope")
        != "four exact B1 MTP unified-attention calls"
    ):
        raise ValueError("BM8 pass sidecar contract drifted")
    for key in (
        "live_result_sha256",
        "live_result_canonical_sha256",
        "qualified_patcher_sha256",
        "qualified_unified_attention_sha256",
        "qualified_eagle_replay_hook_sha256",
    ):
        _require_sha256(payload.get(key), key)
    commit = payload.get("qualified_source_commit")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in HEX for c in commit)
    ):
        raise ValueError("BM8 qualified source commit drifted")
    info = candidate_source.lstat()
    if not stat.S_ISREG(info.st_mode) or candidate_source.is_symlink():
        raise ValueError("BM8 candidate source must be a regular non-symlink file")
    if sha256_file(candidate_source) != expected_candidate_source_sha256:
        raise ValueError("attested BM8 candidate source SHA-256 mismatch")
    payload["canonical_sha256"] = canonical_sha256
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--live-result", required=True, type=Path)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--expected-candidate-source-sha256", required=True)
    issue.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--sidecar", required=True, type=Path)
    verify.add_argument("--expected-sidecar-sha256", required=True)
    verify.add_argument("--candidate-source", required=True, type=Path)
    verify.add_argument("--expected-candidate-source-sha256", required=True)
    args = parser.parse_args()
    if args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            expected_candidate_source_sha256=(
                args.expected_candidate_source_sha256
            ),
            out=args.out,
        )
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_source=args.candidate_source,
            expected_candidate_source_sha256=(
                args.expected_candidate_source_sha256
            ),
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
