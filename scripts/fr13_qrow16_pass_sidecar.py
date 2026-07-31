#!/usr/bin/env python3
"""Issue and verify the canonical qrow16 production-pass sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.fa2_qrow16_live_paged_ab.v1"
SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow16_production_pass.v1"
HEX = frozenset("0123456789abcdef")


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


def validate_live_result(
    payload: dict[str, Any],
    *,
    candidate_sha256: str,
) -> dict[str, Any]:
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("qrow16 live result is not a PASS record")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("concurrency") != 1
        or payload.get("physical_rows") != 32
        or payload.get("runtime_mode") != "FULL"
        or not isinstance(payload.get("instance_id"), str)
        or not payload["instance_id"]
        or payload.get("candidate_so_sha256") != candidate_sha256
        or payload.get("candidate_dispatch")
        != "qrow16 internal exact-geometry require"
        or payload.get("served_return") != "stock captured graph output unchanged"
        or payload.get("performance_measurement") is not False
    ):
        raise ValueError("qrow16 live result provenance drifted")
    operands = payload.get("operands")
    if not isinstance(operands, dict) or (
        operands.get("query_shape") != [32, 24, 256]
        or operands.get("value_cache_shape") != operands.get("key_cache_shape")
        or not isinstance(operands.get("key_cache_shape"), list)
        or operands["key_cache_shape"][1:] != [1024, 4, 256]
        or operands.get("block_table_shape", [None])[0] != 1
        or operands.get("query_start_loc") != [0, 32]
        or not isinstance(operands.get("seq_lens"), list)
        or len(operands["seq_lens"]) != 1
        or int(operands["seq_lens"][0]) <= 0
        or operands.get("tree_bias_shape") not in ([32, 32], [1, 32, 32])
    ):
        raise ValueError("qrow16 live operand geometry drifted")
    for label, dtype in (("output", "torch.bfloat16"), ("lse", "torch.float32")):
        row = payload.get(label)
        if not isinstance(row, dict) or (
            row.get("dtype") != dtype
            or not isinstance(row.get("bytes"), int)
            or int(row["bytes"]) <= 0
            or row.get("raw_byte_mismatches") != 0
            or _require_sha256(row.get("stock_sha256"), f"{label} stock")
            != _require_sha256(row.get("candidate_sha256"), f"{label} candidate")
        ):
            raise ValueError(f"qrow16 live {label} comparison drifted")
    return {
        "instance_id": payload["instance_id"],
        "output_sha256": payload["output"]["stock_sha256"],
        "lse_sha256": payload["lse"]["stock_sha256"],
    }


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    out: Path,
) -> dict[str, Any]:
    expected_live_sha256 = _require_sha256(expected_live_sha256, "live result")
    expected_candidate_sha256 = _require_sha256(
        expected_candidate_sha256, "candidate SO"
    )
    candidate_info = candidate_so.lstat()
    if not stat.S_ISREG(candidate_info.st_mode) or candidate_so.is_symlink():
        raise ValueError("candidate SO must be a regular non-symlink file")
    if sha256_file(candidate_so) != expected_candidate_sha256:
        raise ValueError("candidate SO SHA-256 mismatch")
    live_payload, live_raw = load_json(live_result)
    if _digest_bytes(live_raw) != expected_live_sha256:
        raise ValueError("live result raw SHA-256 mismatch")
    summary = validate_live_result(
        live_payload,
        candidate_sha256=expected_candidate_sha256,
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "candidate_so_sha256": expected_candidate_sha256,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest_bytes(
            canonical_bytes(live_payload)
        ),
        "live_gate_schema": LIVE_SCHEMA,
        "instance_id": summary["instance_id"],
        "output_sha256": summary["output_sha256"],
        "lse_sha256": summary["lse_sha256"],
        "required_runtime": "fixed32 B1 FULL",
        "production_scope": "qrow16 exact target tree attention only",
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest_bytes(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace qrow16 pass sidecar: {out}")
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
    candidate_so: Path,
    expected_candidate_sha256: str,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _require_sha256(
        expected_sidecar_sha256, "pass sidecar"
    )
    expected_candidate_sha256 = _require_sha256(
        expected_candidate_sha256, "candidate SO"
    )
    payload, raw = load_json(sidecar_path)
    if _digest_bytes(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    required = {
        "schema",
        "status",
        "candidate_so_sha256",
        "live_result_sha256",
        "live_result_canonical_sha256",
        "live_gate_schema",
        "instance_id",
        "output_sha256",
        "lse_sha256",
        "required_runtime",
        "production_scope",
        "canonical_sha256",
    }
    if set(payload) != required:
        raise ValueError("pass sidecar key set drifted")
    canonical_sha256 = payload.pop("canonical_sha256")
    if _require_sha256(canonical_sha256, "sidecar canonical") != _digest_bytes(
        canonical_bytes(payload)
    ):
        raise ValueError("pass sidecar canonical digest mismatch")
    if (
        payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("candidate_so_sha256") != expected_candidate_sha256
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("required_runtime") != "fixed32 B1 FULL"
        or payload.get("production_scope")
        != "qrow16 exact target tree attention only"
    ):
        raise ValueError("pass sidecar contract drifted")
    for key in (
        "live_result_sha256",
        "live_result_canonical_sha256",
        "output_sha256",
        "lse_sha256",
    ):
        _require_sha256(payload.get(key), key)
    if sha256_file(candidate_so) != expected_candidate_sha256:
        raise ValueError("attested candidate SO SHA-256 mismatch")
    payload["canonical_sha256"] = canonical_sha256
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--live-result", required=True, type=Path)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--candidate-so", required=True, type=Path)
    issue.add_argument("--expected-candidate-sha256", required=True)
    issue.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--sidecar", required=True, type=Path)
    verify.add_argument("--expected-sidecar-sha256", required=True)
    verify.add_argument("--candidate-so", required=True, type=Path)
    verify.add_argument("--expected-candidate-sha256", required=True)
    args = parser.parse_args()
    if args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            out=args.out,
        )
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
