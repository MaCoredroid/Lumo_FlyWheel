#!/usr/bin/env python3
"""Issue or verify an arm-bound qrow32 B1 production credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_live_paged_ab.v1"
SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_production_pass.v1"
ARMS = {
    "no_split": {"selector_sentinel": 1179791668, "num_splits": 0},
    "split2": {"selector_sentinel": 1179791669, "num_splits": 2},
}
HEX = frozenset("0123456789abcdef")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
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


def _sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase SHA-256")
    return value


def _arm(value: str) -> str:
    if value not in ARMS:
        raise ValueError("arm must be no_split or split2")
    return value


def _validate_comparison(value: Any, label: str) -> str:
    if not isinstance(value, dict) or (
        not isinstance(value.get("bytes"), int)
        or value["bytes"] <= 0
        or value.get("raw_byte_mismatches") != 0
    ):
        raise ValueError(f"{label} byte comparison drifted")
    stock = _sha256(value.get("stock_sha256"), f"{label} stock")
    candidate = _sha256(value.get("candidate_sha256"), f"{label} candidate")
    if stock != candidate:
        raise ValueError(f"{label} digest comparison drifted")
    return stock


def validate_live_result(
    payload: dict[str, Any], *, candidate_sha256: str, arm: str
) -> dict[str, Any]:
    arm = _arm(arm)
    config = ARMS[arm]
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError("qrow32 B1 live result is not a PASS")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != "astropy__astropy-12907"
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("physical_rows") != 32
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("runtime_mode") != "FULL"
        or payload.get("candidate_so_sha256") != candidate_sha256
        or payload.get("arm") != arm
        or payload.get("selector_sentinel") != config["selector_sentinel"]
        or payload.get("candidate_num_splits") != config["num_splits"]
        or payload.get("layer_count") != 16
        or payload.get("fallback_allowed") is not False
        or payload.get("served_return") != "stock captured graph output unchanged"
        or payload.get("performance_measurement") is not False
    ):
        raise ValueError("qrow32 B1 live provenance drifted")
    expected_scratch = (
        "stock FA2 set_params_splitkv via num_splits=2"
        if arm == "split2"
        else "not applicable"
    )
    if payload.get("split_scratch_allocation") != expected_scratch:
        raise ValueError("qrow32 B1 split scratch provenance drifted")
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != 16:
        raise ValueError("qrow32 B1 live layer set drifted")
    output_digests = []
    lse_digests = []
    names = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_name"), str):
            raise ValueError("qrow32 B1 live layer record drifted")
        names.add(layer["layer_name"])
        output_digests.append(
            _validate_comparison(layer.get("output"), f"layer {index} output")
        )
        lse_digests.append(
            _validate_comparison(layer.get("lse"), f"layer {index} lse")
        )
    if len(names) != 16:
        raise ValueError("qrow32 B1 live layer identities are not unique")
    if (
        payload.get("output_raw_byte_mismatches") != 0
        or payload.get("lse_raw_byte_mismatches") != 0
    ):
        raise ValueError("qrow32 B1 live aggregate mismatch drifted")
    return {
        "instance_id": payload["instance_id"],
        "layers_sha256": _digest(canonical_bytes({
            "names": sorted(names),
            "output": output_digests,
            "lse": lse_digests,
        })),
    }


def issue_sidecar(
    *, live_result: Path, expected_live_sha256: str, candidate_so: Path,
    expected_candidate_sha256: str, arm: str, out: Path,
) -> dict[str, Any]:
    arm = _arm(arm)
    expected_live_sha256 = _sha256(expected_live_sha256, "live result")
    expected_candidate_sha256 = _sha256(
        expected_candidate_sha256, "candidate SO"
    )
    info = candidate_so.lstat()
    if not stat.S_ISREG(info.st_mode) or candidate_so.is_symlink():
        raise ValueError("candidate SO must be a regular non-symlink file")
    if sha256_file(candidate_so) != expected_candidate_sha256:
        raise ValueError("candidate SO SHA-256 mismatch")
    live, raw = load_json(live_result)
    if _digest(raw) != expected_live_sha256:
        raise ValueError("live result raw SHA-256 mismatch")
    summary = validate_live_result(
        live, candidate_sha256=expected_candidate_sha256, arm=arm
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "arm": arm,
        "selector_sentinel": ARMS[arm]["selector_sentinel"],
        "num_splits": ARMS[arm]["num_splits"],
        "candidate_so_sha256": expected_candidate_sha256,
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest(canonical_bytes(live)),
        "live_gate_schema": LIVE_SCHEMA,
        "instance_id": summary["instance_id"],
        "layers_sha256": summary["layers_sha256"],
        "required_runtime": "fixed32 K64 ROOT=1 B1",
        "production_scope": f"qrow32 B1 {arm} exact tree attention only",
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace qrow32 B1 pass sidecar: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name + f".tmp.{os.getpid()}")
    temporary.write_bytes(canonical_bytes(sidecar) + b"\n")
    os.chmod(temporary, 0o600)
    os.replace(temporary, out)
    return sidecar


def verify_sidecar(
    *, sidecar_path: Path, expected_sidecar_sha256: str, candidate_so: Path,
    expected_candidate_sha256: str, arm: str,
) -> dict[str, Any]:
    arm = _arm(arm)
    expected_sidecar_sha256 = _sha256(expected_sidecar_sha256, "pass sidecar")
    expected_candidate_sha256 = _sha256(
        expected_candidate_sha256, "candidate SO"
    )
    candidate_info = candidate_so.lstat()
    if not stat.S_ISREG(candidate_info.st_mode) or candidate_so.is_symlink():
        raise ValueError("candidate SO must be a regular non-symlink file")
    payload, raw = load_json(sidecar_path)
    if _digest(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise ValueError("pass sidecar canonical digest mismatch")
    config = ARMS[arm]
    if (
        payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("arm") != arm
        or payload.get("selector_sentinel") != config["selector_sentinel"]
        or payload.get("num_splits") != config["num_splits"]
        or payload.get("candidate_so_sha256") != expected_candidate_sha256
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("required_runtime") != "fixed32 K64 ROOT=1 B1"
        or payload.get("production_scope")
        != f"qrow32 B1 {arm} exact tree attention only"
    ):
        raise ValueError("pass sidecar contract drifted")
    for key in (
        "live_result_sha256", "live_result_canonical_sha256", "layers_sha256"
    ):
        _sha256(payload.get(key), key)
    if sha256_file(candidate_so) != expected_candidate_sha256:
        raise ValueError("attested candidate SO SHA-256 mismatch")
    payload["canonical_sha256"] = canonical
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    issue = subparsers.add_parser("issue")
    issue.add_argument("--live-result", required=True, type=Path)
    issue.add_argument("--expected-live-sha256", required=True)
    issue.add_argument("--candidate-so", required=True, type=Path)
    issue.add_argument("--expected-candidate-sha256", required=True)
    issue.add_argument("--arm", required=True, choices=sorted(ARMS))
    issue.add_argument("--out", required=True, type=Path)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--sidecar", required=True, type=Path)
    verify.add_argument("--expected-sidecar-sha256", required=True)
    verify.add_argument("--candidate-so", required=True, type=Path)
    verify.add_argument("--expected-candidate-sha256", required=True)
    verify.add_argument("--arm", required=True, choices=sorted(ARMS))
    args = parser.parse_args()
    if args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            out=args.out,
        )
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
