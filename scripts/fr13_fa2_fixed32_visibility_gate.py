#!/usr/bin/env python3
"""Authenticate and verify the fixed32 FA2 visibility-mask B4 candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import fr13_fa2_qrow32_gate as qrow32_gate


CANDIDATE_ARM = "visibility"
CANDIDATE_SHA256 = (
    "805635d6881dbf73287d66c10541880b7cf93bcb6bf7b04e50efd3d32728b0aa"
)
CANDIDATE_SIZE = 299_810_632
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SOURCE_FILES = {
    "csrc/flash_attn/flash_api.cpp": (
        "f6ed4164e181521c55d167e0cf7af04dc10069a6af909481450ae4eb9236d11c"
    ),
    "csrc/flash_attn/flash_api_torch_lib.cpp": (
        "c575d9f02ba44bf7022c77b80fdf12173da0ecae8a4d7599934c2cc9fa52e121"
    ),
    "csrc/flash_attn/src/flash.h": (
        "e4c7875a72c0bc5f8ed3e0661ef956ca24b38c8f4758ae2a89f5e58b88671c5a"
    ),
    "csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu": (
        "07fbf58dc774d399d511e46d411caedd8f5b8952f6d61876b3fb190a6cef4a17"
    ),
    "csrc/flash_attn/src/flash_fwd_kernel.h": (
        "5e48444ff68c75dee9227570735b15a4273c064c574d21a4c1953019bb9eb876"
    ),
    "csrc/flash_attn/src/utils.h": (
        "5887df63c79a3e42fb9ddad93f64fe3c0625dbee4c547af68b6f2108b7beeb5f"
    ),
}
SOURCE_STATUS = (
    " M csrc/flash_attn/flash_api.cpp",
    " M csrc/flash_attn/flash_api_torch_lib.cpp",
    " M csrc/flash_attn/src/flash.h",
    " M csrc/flash_attn/src/flash_fwd_kernel.h",
    " M csrc/flash_attn/src/utils.h",
    "?? csrc/flash_attn/src/flash_fwd_fr13_qrow32_hdim256_bf16_sm80.cu",
)
SOURCE_CLOSURE_SHA256 = (
    "1dac8f7fd910a564c5c3b792770029f0013e2df48c25c89376e4d5e7da949ced"
)
SELECTOR_SENTINEL = 131092
DISPATCH = "qrow32 fixed32 visibility-mask exact B4 geometry; no fallback"


class GateError(ValueError):
    pass


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except FileNotFoundError as error:
        raise GateError(f"{label} is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise GateError(f"{label} must be a regular non-symlink file")
    return info


def _git(repo: Path, *args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.rstrip("\n")
    except subprocess.CalledProcessError as error:
        raise GateError(f"git {' '.join(args)} failed for {repo}") from error


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")


def validate_b4(candidate_so: Path, fa2_source: Path) -> dict[str, Any]:
    info = _regular(candidate_so, "visibility candidate SO")
    if info.st_size != CANDIDATE_SIZE or _sha256_file(candidate_so) != CANDIDATE_SHA256:
        raise GateError("visibility candidate SO identity drifted")
    if not fa2_source.is_dir() or fa2_source.is_symlink():
        raise GateError("FA2 source root must be a non-symlink directory")
    head = _git(fa2_source, "rev-parse", "HEAD")
    if head != FA2_HEAD:
        raise GateError("FA2 source HEAD drifted")
    status = tuple(
        line
        for line in _git(
            fa2_source, "status", "--porcelain=v1", "--untracked-files=all"
        ).splitlines()
        if line
    )
    if status != SOURCE_STATUS:
        raise GateError("FA2 regenerated source set drifted")
    files: dict[str, str] = {}
    for relative, expected in SOURCE_FILES.items():
        path = fa2_source / relative
        _regular(path, f"FA2 source {relative}")
        actual = _sha256_file(path)
        if actual != expected:
            raise GateError(f"FA2 source hash drifted: {relative}")
        files[relative] = actual
    closure = {"fa2_head": head, "files": files}
    if hashlib.sha256(_canonical_bytes(closure)).hexdigest() != SOURCE_CLOSURE_SHA256:
        raise GateError("FA2 source closure digest drifted")
    return {
        "candidate_arm": CANDIDATE_ARM,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "selector_sentinel": SELECTOR_SENTINEL,
    }


def verify_b4(args: argparse.Namespace) -> dict[str, Any]:
    if args.fixed32_mode != "hydra27_fixed32":
        raise GateError("visibility candidate is scoped to Hydra27 B4")
    identity = validate_b4(args.candidate_so, args.fa2_source)
    if not re.fullmatch(r"[0-9a-f]{40}", args.source_commit):
        raise GateError("source commit is not a full lowercase commit")
    try:
        base = qrow32_gate.verify_live(args)
    except qrow32_gate.GateError as error:
        raise GateError(str(error)) from error
    result = json.loads(args.result.read_text(encoding="ascii"))
    expected = {
        **identity,
        "candidate_dispatch": DISPATCH,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "incumbent_dispatch": "stock FA2 exact geometry; no fallback",
        "source_commit": args.source_commit,
    }
    for key, value in expected.items():
        if result.get(key) != value:
            raise GateError(f"visibility B4 live result {key} drifted")
    return {
        "schema": "fr13.fixed32.fa2_visibility_b4_live_verification.v1",
        "status": "PASS",
        "fixed32_mode": args.fixed32_mode,
        **identity,
        "source_commit": args.source_commit,
        "task_ids": list(qrow32_gate.TASK_IDS),
        "layer_count": base["layer_count"],
        "slot_coverage": base["slot_coverage"],
        "output_raw_byte_mismatches": 0,
        "lse_raw_byte_mismatches": 0,
        "fallback_allowed": False,
        "performance_measurement": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    candidate = commands.add_parser("validate-b4")
    candidate.add_argument("--candidate-so", type=Path, required=True)
    candidate.add_argument("--fa2-source", type=Path, required=True)
    verify = commands.add_parser("verify-b4")
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--campaign-arm", type=Path, required=True)
    verify.add_argument("--campaign-provenance", type=Path, required=True)
    verify.add_argument("--candidate-so", type=Path, required=True)
    verify.add_argument("--fa2-source", type=Path, required=True)
    verify.add_argument("--fixed32-mode", choices=("hydra27_fixed32",), required=True)
    verify.add_argument("--source-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = (
        validate_b4(args.candidate_so, args.fa2_source)
        if args.command == "validate-b4"
        else verify_b4(args)
    )
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
