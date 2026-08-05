#!/usr/bin/env python3
"""Validate qrow32 B1 live gates and issue the split2 production credential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


LIVE_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_live_paged_ab.v2"
SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow32_b1_production_pass.v2"
ARM = "split2"
SELECTOR_SENTINEL = 1179791669
QROW16_REFERENCE_SENTINEL = 1179791667
NUM_SPLITS = 2
LIVE_ARMS = {
    "nosplit": {
        "selector_sentinel": 1179791668,
        "num_splits": 0,
        "split_scratch_allocation": "not used; num_splits=0",
        "candidate_dispatch": "qrow32 B1 nosplit exact geometry; no fallback",
    },
    ARM: {
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "split_scratch_allocation": (
            "stock FA2 set_params_splitkv via num_splits=2"
        ),
        "candidate_dispatch": "qrow32 B1 split2 exact geometry; no fallback",
    },
}
CANDIDATE_SHA256 = (
    "ec36c5d26635fead8f626539ff98ab055a756af1e568dbadf88905a41f61862a"
)
CANDIDATE_SIZE = 300_153_584
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "c01882eec7e4333ca775f7d64cb0b88634c61bc615cbdc01895f33216dadb158"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu": (
        "00f38d6ed68580d0e9c957ed0620d9655b1e4bf24bf16e970e693341bfa60cc1"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu": (
        "223542ecf9bcc8837022aaceeca7468e4a8c866b528c4327c68f924dc4ab344d"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "5829b5c6832ce962fc6864f491901ed68e8a24b165d5cff914e94c38b0de8177"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
SOURCE_CLOSURE_SHA256 = (
    "3c559d80c65573932c5c7bfd5ef7081df6c3f1a3f6c888bc36a04ccc264d394b"
)
SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_hdim256_bf16_sm80.cu",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_b1_split2_hdim256_bf16_sm80.cu",
)
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
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


def _commit(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 40 or any(c not in HEX for c in value):
        raise ValueError(f"{label} is not a lowercase commit")
    return value


def _regular(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError(f"{label} must be a regular non-symlink file")
    return info


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout.rstrip("\n")


def validate_candidate(candidate_so: Path, expected_sha256: str) -> dict[str, Any]:
    expected_sha256 = _sha256(expected_sha256, "candidate SO")
    if expected_sha256 != CANDIDATE_SHA256:
        raise ValueError("candidate SO is not the pinned qrow32 split2 binary")
    info = _regular(candidate_so, "candidate SO")
    if info.st_size != CANDIDATE_SIZE or sha256_file(candidate_so) != CANDIDATE_SHA256:
        raise ValueError("candidate SO identity mismatch")
    return {"size": CANDIDATE_SIZE, "sha256": CANDIDATE_SHA256}


def validate_source_closure(source_root: Path) -> dict[str, Any]:
    if not source_root.is_dir() or source_root.is_symlink():
        raise ValueError("FA2 source root must be a non-symlink directory")
    head = _git(source_root, "rev-parse", "HEAD")
    if head != FA2_HEAD:
        raise ValueError("FA2 source head drifted")
    status = tuple(
        line
        for line in _git(
            source_root, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    if status != SOURCE_STATUS:
        raise ValueError("FA2 modified source set drifted")
    records: dict[str, str] = {}
    for relative, expected in SOURCE_FILES.items():
        path = source_root / relative
        _regular(path, f"FA2 source {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"FA2 source hash drifted: {relative}")
        records[relative] = actual
    closure = {"fa2_head": head, "files": records}
    if _digest(canonical_bytes(closure)) != SOURCE_CLOSURE_SHA256:
        raise ValueError("FA2 source closure digest drifted")
    closure["canonical_sha256"] = SOURCE_CLOSURE_SHA256
    return closure


def validate_patch_source(
    patch_source: Path, *, expected_source_commit: str
) -> dict[str, str]:
    _regular(patch_source, "patch source")
    expected_source_commit = _commit(expected_source_commit, "source commit")
    repo = Path(_git(patch_source.parent, "rev-parse", "--show-toplevel"))
    head = _git(repo, "rev-parse", "HEAD")
    if head != expected_source_commit:
        raise ValueError("patch source commit drifted")
    relative = patch_source.resolve().relative_to(repo.resolve()).as_posix()
    if relative != "scripts/fr13_patch_fa2_tree_bias.py":
        raise ValueError("patch source path drifted")
    return {"source_commit": head, "patch_source_sha256": sha256_file(patch_source)}


def _validate_comparison(value: Any, label: str) -> str:
    if not isinstance(value, dict) or (
        not isinstance(value.get("bytes"), int)
        or value["bytes"] <= 0
        or value.get("raw_byte_mismatches") != 0
    ):
        raise ValueError(f"{label} byte comparison drifted")
    reference = _sha256(value.get("reference_sha256"), f"{label} reference")
    candidate = _sha256(value.get("candidate_sha256"), f"{label} candidate")
    if reference != candidate:
        raise ValueError(f"{label} digest comparison drifted")
    return reference


def validate_live_result(
    payload: dict[str, Any],
    *,
    candidate_sha256: str,
    arm: str,
    source_commit: str | None = None,
    patch_source_sha256: str | None = None,
) -> dict[str, Any]:
    arm_contract = LIVE_ARMS.get(arm)
    if arm_contract is None:
        raise ValueError("qrow32 live arm must be nosplit or split2")
    if candidate_sha256 != CANDIDATE_SHA256:
        raise ValueError("qrow32 live candidate is not pinned")
    if payload.get("schema") != LIVE_SCHEMA or payload.get("status") != "PASS":
        raise ValueError(f"qrow32 {arm} live result is not a PASS")
    if (
        payload.get("suite") != "SWE-Verified"
        or payload.get("instance_id") != EXACT4_TASK_IDS[0]
        or payload.get("concurrency") != 1
        or payload.get("batch_size") != 1
        or payload.get("physical_rows") != 32
        or payload.get("draft_vocab_root") != 1
        or payload.get("draft_vocab_k") != 65536
        or payload.get("runtime_mode") != "FULL"
        or payload.get("candidate_so_sha256") != CANDIDATE_SHA256
        or payload.get("candidate_so_size") != CANDIDATE_SIZE
        or payload.get("arm") != arm
        or payload.get("selector_sentinel") != arm_contract["selector_sentinel"]
        or payload.get("candidate_num_splits") != arm_contract["num_splits"]
        or payload.get("reference_selector_sentinel") != QROW16_REFERENCE_SENTINEL
        or payload.get("reference_dispatch")
        != "qrow16 incumbent exact geometry; no fallback"
        or payload.get("candidate_dispatch")
        != arm_contract["candidate_dispatch"]
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256") != SOURCE_CLOSURE_SHA256
        or payload.get("layer_count") != 16
        or payload.get("fallback_allowed") is not False
        or payload.get("served_return") != "qrow16 captured graph output unchanged"
        or payload.get("performance_measurement") is not False
        or payload.get("split_scratch_allocation")
        != arm_contract["split_scratch_allocation"]
    ):
        raise ValueError(f"qrow32 {arm} live provenance drifted")
    live_source_commit = _commit(payload.get("source_commit"), "live source commit")
    live_patch_sha = _sha256(
        payload.get("patch_source_sha256"), "live patch source"
    )
    if source_commit is not None and live_source_commit != source_commit:
        raise ValueError("qrow32 live source commit drifted")
    if patch_source_sha256 is not None and live_patch_sha != patch_source_sha256:
        raise ValueError("qrow32 live patch source drifted")
    layers = payload.get("layers")
    if not isinstance(layers, list) or len(layers) != 16:
        raise ValueError(f"qrow32 {arm} live layer set drifted")
    output_digests = []
    lse_digests = []
    names = set()
    for index, layer in enumerate(layers):
        if not isinstance(layer, dict) or not isinstance(layer.get("layer_name"), str):
            raise ValueError(f"qrow32 {arm} live layer record drifted")
        names.add(layer["layer_name"])
        output_digests.append(
            _validate_comparison(layer.get("output"), f"layer {index} output")
        )
        lse_digests.append(
            _validate_comparison(layer.get("lse"), f"layer {index} lse")
        )
    expected_names = {
        f"language_model.model.layers.{index}.self_attn.attn"
        for index in range(3, 64, 4)
    }
    if names != expected_names:
        raise ValueError(f"qrow32 {arm} live layer identities drifted")
    if (
        payload.get("output_raw_byte_mismatches") != 0
        or payload.get("lse_raw_byte_mismatches") != 0
    ):
        raise ValueError(f"qrow32 {arm} live aggregate mismatch drifted")
    return {
        "arm": arm,
        "instance_id": payload["instance_id"],
        "source_commit": live_source_commit,
        "patch_source_sha256": live_patch_sha,
        "layers_sha256": _digest(
            canonical_bytes(
                {
                    "names": sorted(names),
                    "output": output_digests,
                    "lse": lse_digests,
                }
            )
        ),
    }


def issue_sidecar(
    *,
    live_result: Path,
    expected_live_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    out: Path,
) -> dict[str, Any]:
    if arm != ARM:
        raise ValueError("qrow32 production arm must be split2")
    expected_live_sha256 = _sha256(expected_live_sha256, "live result")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    live, raw = load_json(live_result)
    if _digest(raw) != expected_live_sha256:
        raise ValueError("live result raw SHA-256 mismatch")
    summary = validate_live_result(
        live,
        candidate_sha256=expected_candidate_sha256,
        arm=arm,
        source_commit=patch["source_commit"],
        patch_source_sha256=patch["patch_source_sha256"],
    )
    body = {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "arm": ARM,
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "reference": "qrow16 incumbent exact geometry",
        "reference_selector_sentinel": QROW16_REFERENCE_SENTINEL,
        "candidate_so_size": CANDIDATE_SIZE,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "source_commit": patch["source_commit"],
        "patch_source_sha256": patch["patch_source_sha256"],
        "live_result_sha256": expected_live_sha256,
        "live_result_canonical_sha256": _digest(canonical_bytes(live)),
        "live_gate_schema": LIVE_SCHEMA,
        "instance_id": summary["instance_id"],
        "layers_sha256": summary["layers_sha256"],
        "required_runtime": "Hydra27 fixed32 K64 ROOT=1 B1 physical32 FULL graph",
        "production_scope": "qrow32 B1 split2 exact tree attention only",
        "fallback_allowed": False,
    }
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise ValueError(f"refusing to replace qrow32 split2 pass sidecar: {out}")
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
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    expected_sidecar_sha256 = _sha256(expected_sidecar_sha256, "pass sidecar")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    payload, raw = load_json(sidecar_path)
    if _digest(raw) != expected_sidecar_sha256:
        raise ValueError("pass sidecar raw SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise ValueError("pass sidecar canonical digest mismatch")
    if (
        arm != ARM
        or payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("arm") != ARM
        or payload.get("selector_sentinel") != SELECTOR_SENTINEL
        or payload.get("num_splits") != NUM_SPLITS
        or payload.get("reference") != "qrow16 incumbent exact geometry"
        or payload.get("reference_selector_sentinel") != QROW16_REFERENCE_SENTINEL
        or payload.get("candidate_so_size") != CANDIDATE_SIZE
        or payload.get("candidate_so_sha256") != CANDIDATE_SHA256
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256") != SOURCE_CLOSURE_SHA256
        or payload.get("source_commit") != patch["source_commit"]
        or payload.get("patch_source_sha256") != patch["patch_source_sha256"]
        or payload.get("live_gate_schema") != LIVE_SCHEMA
        or payload.get("required_runtime")
        != "Hydra27 fixed32 K64 ROOT=1 B1 physical32 FULL graph"
        or payload.get("production_scope")
        != "qrow32 B1 split2 exact tree attention only"
        or payload.get("fallback_allowed") is not False
    ):
        raise ValueError("pass sidecar contract drifted")
    for key in (
        "live_result_sha256",
        "live_result_canonical_sha256",
        "layers_sha256",
    ):
        _sha256(payload.get(key), key)
    payload["canonical_sha256"] = canonical
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("validate-source")
    source.add_argument("--source-root", required=True, type=Path)
    for name in ("issue", "verify"):
        command = subparsers.add_parser(name)
        if name == "issue":
            command.add_argument("--live-result", required=True, type=Path)
            command.add_argument("--expected-live-sha256", required=True)
            command.add_argument("--out", required=True, type=Path)
        else:
            command.add_argument("--sidecar", required=True, type=Path)
            command.add_argument("--expected-sidecar-sha256", required=True)
        command.add_argument("--candidate-so", required=True, type=Path)
        command.add_argument("--expected-candidate-sha256", required=True)
        command.add_argument("--arm", required=True, choices=(ARM,))
        command.add_argument("--patch-source", required=True, type=Path)
        command.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    if args.command == "validate-source":
        result = validate_source_closure(args.source_root)
    elif args.command == "issue":
        result = issue_sidecar(
            live_result=args.live_result,
            expected_live_sha256=args.expected_live_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
            out=args.out,
        )
    else:
        result = verify_sidecar(
            sidecar_path=args.sidecar,
            expected_sidecar_sha256=args.expected_sidecar_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
        )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
