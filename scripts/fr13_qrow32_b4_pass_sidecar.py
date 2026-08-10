#!/usr/bin/env python3
"""Bind a B4 GQA-pair dual raw-byte gate PASS to the production credential.

The credential this issues is what ``FR13_FA2_QROW32_B4_PRODUCTION_ARM``
consumes, and it is the ONLY thing that lets the patched tree_attn selector
inject the GQA-pair batch-stride sentinel on the served decode call.

The dual-gate PASS is supplied by path, never by a hardcoded runroot: the
credential must be issued against a dual gate re-run at the SAME source commit
as the production plumbing, so any previously produced gate artifact is
rejected by the commit binding rather than accidentally reused.

The gate artifact's own ``timing_eligible``/``production_eligible`` fields are
recorded verbatim and required to be false -- they are the byte gate's true
statement that THAT run measured nothing and served nothing. What this
credential asserts is narrower and is the only thing the gate established:
raw-byte output/LSE equality between the stock and GQA-pair dispatches of one
binary, on both qualified topologies, at this commit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path
from typing import Any


DUAL_GATE_SCHEMA = "fr13.fixed32.fa2_qrow32_gqa_pair_b4_dual_gate.v1"
SIDECAR_SCHEMA = "fr13.fixed32.fa2_qrow32_b4_production_pass.v1"
ARM = "gqa_pair"
SELECTOR_SENTINEL = 0x20014
NUM_SPLITS = 0
CANDIDATE_SHA256 = (
    "af9e9f24335db899468032f5b5a3eba100febe294932533cb9b87163ce2b3fdb"
)
CANDIDATE_SIZE = 299_813_360
FA2_HEAD = "29210221863736a08f71a866459e368ad1ac4a95"
SOURCE_CLOSURE_SHA256 = (
    "9c3f9e751da7b783e9d07d8e40d5bc2234b99e719a1048668bd6c82244ed2d81"
)
QUALIFIED_TOPOLOGIES = ("Tail23", "Hydra27")
EXACT4_TASK_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
)
EXACT4_SUBSET_SHA256 = (
    "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5"
)
PATCH_SOURCE_RELATIVE = "scripts/fr13_patch_fa2_tree_bias.py"
REQUIRED_RUNTIME = (
    "fixed32 K64 ROOT=1 exact4 B4 physical32 FULL graph on Tail23 or Hydra27"
)
PRODUCTION_SCOPE = "qrow32 GQA-pair B4 exact tree attention only"
CREDENTIAL_BASIS = (
    "dual-topology raw-byte output/LSE equality between the stock and "
    "GQA-pair dispatches of the pinned binary at this source commit"
)
HEX = frozenset("0123456789abcdef")


class SidecarError(ValueError):
    """The credential inputs are not the bound, pinned, exact4 evidence."""


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
            raise SidecarError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _regular(path: Path, label: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise SidecarError(f"{label} must be a regular non-symlink file")
    return info


def load_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    _regular(path, label)
    raw = path.read_bytes()
    payload = json.loads(
        raw,
        object_pairs_hook=_reject_duplicates,
        parse_constant=lambda value: (_ for _ in ()).throw(
            SidecarError(f"non-finite JSON value: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise SidecarError(f"{label} JSON root must be an object")
    return payload, raw


def _sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in HEX for char in value)
    ):
        raise SidecarError(f"{label} is not a lowercase SHA-256")
    return value


def _commit(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(char not in HEX for char in value)
    ):
        raise SidecarError(f"{label} is not a lowercase commit")
    return value


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
        raise SidecarError(f"git {' '.join(args)} failed for {repo}") from error


def validate_candidate(candidate_so: Path, expected_sha256: str) -> dict[str, Any]:
    """Fail closed unless the binary IS the dual-gate-qualified GQA-pair .so."""
    expected_sha256 = _sha256(expected_sha256, "candidate SO")
    if expected_sha256 != CANDIDATE_SHA256:
        raise SidecarError("candidate SO is not the pinned GQA-pair binary")
    info = _regular(candidate_so, "candidate SO")
    if info.st_size != CANDIDATE_SIZE or sha256_file(candidate_so) != CANDIDATE_SHA256:
        raise SidecarError("candidate SO identity mismatch")
    return {
        "candidate_so_sha256": CANDIDATE_SHA256,
        "candidate_so_size": CANDIDATE_SIZE,
    }


def validate_patch_source(
    patch_source: Path, *, expected_source_commit: str
) -> dict[str, str]:
    """The plumbing commit itself: the credential is void off this commit."""
    _regular(patch_source, "patch source")
    expected_source_commit = _commit(expected_source_commit, "source commit")
    repo = Path(_git(patch_source.parent, "rev-parse", "--show-toplevel"))
    head = _git(repo, "rev-parse", "HEAD")
    if head != expected_source_commit:
        raise SidecarError("patch source commit drifted")
    relative = patch_source.resolve().relative_to(repo.resolve()).as_posix()
    if relative != PATCH_SOURCE_RELATIVE:
        raise SidecarError("patch source path drifted")
    return {
        "source_commit": head,
        "patch_source_sha256": sha256_file(patch_source),
    }


def validate_dual_gate(
    payload: dict[str, Any], *, source_commit: str
) -> dict[str, Any]:
    if payload.get("schema") != DUAL_GATE_SCHEMA:
        raise SidecarError("dual gate schema drifted")
    if payload.get("status") != "PASS":
        raise SidecarError("dual gate result is not a PASS")
    if (
        payload.get("candidate_arm") != ARM
        or payload.get("selector_sentinel") != SELECTOR_SENTINEL
        or payload.get("candidate_so_sha256") != CANDIDATE_SHA256
        or payload.get("candidate_so_size") != CANDIDATE_SIZE
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256") != SOURCE_CLOSURE_SHA256
    ):
        raise SidecarError("dual gate candidate identity drifted")
    if (
        payload.get("task_ids") != list(EXACT4_TASK_IDS)
        or payload.get("subset_sha256") != EXACT4_SUBSET_SHA256
        or payload.get("qualified_topologies") != list(QUALIFIED_TOPOLOGIES)
        or payload.get("layer_count_per_topology") != 16
    ):
        raise SidecarError("dual gate exact4 scope drifted")
    if (
        payload.get("output_raw_byte_mismatches") != 0
        or payload.get("lse_raw_byte_mismatches") != 0
        or payload.get("fallback_allowed") is not False
        or payload.get("performance_measurement") is not False
    ):
        raise SidecarError("dual gate raw-byte equality drifted")
    # The byte gate served stock and measured nothing; those are properties of
    # THAT run, and a gate artifact claiming otherwise has been edited.
    if (
        payload.get("timing_eligible") is not False
        or payload.get("production_eligible") is not False
    ):
        raise SidecarError("dual gate eligibility fields were rewritten")
    gate_commit = _commit(payload.get("source_commit"), "dual gate source commit")
    if gate_commit != source_commit:
        raise SidecarError(
            "dual gate was not produced at the production plumbing commit"
        )
    return {
        "source_commit": gate_commit,
        "tail23_verification_sha256": _sha256(
            payload.get("tail23_verification_sha256"), "Tail23 verification"
        ),
        "hydra27_verification_sha256": _sha256(
            payload.get("hydra27_verification_sha256"), "Hydra27 verification"
        ),
    }


def _body(
    *,
    patch: dict[str, str],
    dual_gate_sha256: str,
    dual_gate_canonical_sha256: str,
    summary: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": SIDECAR_SCHEMA,
        "status": "PASS",
        "arm": ARM,
        "selector_sentinel": SELECTOR_SENTINEL,
        "num_splits": NUM_SPLITS,
        "reference": "stock FA2 dispatch of the same pinned binary",
        "candidate_so_size": CANDIDATE_SIZE,
        "candidate_so_sha256": CANDIDATE_SHA256,
        "fa2_head": FA2_HEAD,
        "fa2_source_closure_sha256": SOURCE_CLOSURE_SHA256,
        "source_commit": patch["source_commit"],
        "patch_source_sha256": patch["patch_source_sha256"],
        "dual_gate_schema": DUAL_GATE_SCHEMA,
        "dual_gate_sha256": dual_gate_sha256,
        "dual_gate_canonical_sha256": dual_gate_canonical_sha256,
        "dual_gate_timing_eligible": False,
        "dual_gate_production_eligible": False,
        "tail23_verification_sha256": summary["tail23_verification_sha256"],
        "hydra27_verification_sha256": summary["hydra27_verification_sha256"],
        "qualified_topologies": list(QUALIFIED_TOPOLOGIES),
        "task_ids": list(EXACT4_TASK_IDS),
        "subset_sha256": EXACT4_SUBSET_SHA256,
        "credential_basis": CREDENTIAL_BASIS,
        "required_runtime": REQUIRED_RUNTIME,
        "production_scope": PRODUCTION_SCOPE,
        "fallback_allowed": False,
    }


def validate_binding(
    *,
    dual_gate: Path,
    expected_dual_gate_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
) -> dict[str, Any]:
    """The credential body, without writing it: a pre-flight for runners."""
    if arm != ARM:
        raise SidecarError("qrow32 B4 production arm must be gqa_pair")
    expected_dual_gate_sha256 = _sha256(expected_dual_gate_sha256, "dual gate")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    payload, raw = load_json(dual_gate, "dual gate")
    if _digest(raw) != expected_dual_gate_sha256:
        raise SidecarError("dual gate raw SHA-256 mismatch")
    summary = validate_dual_gate(payload, source_commit=patch["source_commit"])
    return _body(
        patch=patch,
        dual_gate_sha256=expected_dual_gate_sha256,
        dual_gate_canonical_sha256=_digest(canonical_bytes(payload)),
        summary=summary,
    )


def issue_sidecar(
    *,
    dual_gate: Path,
    expected_dual_gate_sha256: str,
    candidate_so: Path,
    expected_candidate_sha256: str,
    arm: str,
    patch_source: Path,
    expected_source_commit: str,
    out: Path,
) -> dict[str, Any]:
    body = validate_binding(
        dual_gate=dual_gate,
        expected_dual_gate_sha256=expected_dual_gate_sha256,
        candidate_so=candidate_so,
        expected_candidate_sha256=expected_candidate_sha256,
        arm=arm,
        patch_source=patch_source,
        expected_source_commit=expected_source_commit,
    )
    sidecar = dict(body)
    sidecar["canonical_sha256"] = _digest(canonical_bytes(body))
    if out.exists() or out.is_symlink():
        raise SidecarError(f"refusing to replace qrow32 B4 pass sidecar: {out}")
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
    if arm != ARM:
        raise SidecarError("qrow32 B4 production arm must be gqa_pair")
    expected_sidecar_sha256 = _sha256(expected_sidecar_sha256, "pass sidecar")
    validate_candidate(candidate_so, expected_candidate_sha256)
    patch = validate_patch_source(
        patch_source, expected_source_commit=expected_source_commit
    )
    payload, raw = load_json(sidecar_path, "pass sidecar")
    if _digest(raw) != expected_sidecar_sha256:
        raise SidecarError("pass sidecar raw SHA-256 mismatch")
    canonical = payload.pop("canonical_sha256", None)
    if _sha256(canonical, "sidecar canonical") != _digest(canonical_bytes(payload)):
        raise SidecarError("pass sidecar canonical digest mismatch")
    if (
        payload.get("schema") != SIDECAR_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("arm") != ARM
        or payload.get("selector_sentinel") != SELECTOR_SENTINEL
        or payload.get("num_splits") != NUM_SPLITS
        or payload.get("reference") != "stock FA2 dispatch of the same pinned binary"
        or payload.get("candidate_so_size") != CANDIDATE_SIZE
        or payload.get("candidate_so_sha256") != CANDIDATE_SHA256
        or payload.get("fa2_head") != FA2_HEAD
        or payload.get("fa2_source_closure_sha256") != SOURCE_CLOSURE_SHA256
        or payload.get("source_commit") != patch["source_commit"]
        or payload.get("patch_source_sha256") != patch["patch_source_sha256"]
        or payload.get("dual_gate_schema") != DUAL_GATE_SCHEMA
        or payload.get("dual_gate_timing_eligible") is not False
        or payload.get("dual_gate_production_eligible") is not False
        or payload.get("qualified_topologies") != list(QUALIFIED_TOPOLOGIES)
        or payload.get("task_ids") != list(EXACT4_TASK_IDS)
        or payload.get("subset_sha256") != EXACT4_SUBSET_SHA256
        or payload.get("credential_basis") != CREDENTIAL_BASIS
        or payload.get("required_runtime") != REQUIRED_RUNTIME
        or payload.get("production_scope") != PRODUCTION_SCOPE
        or payload.get("fallback_allowed") is not False
    ):
        raise SidecarError("pass sidecar contract drifted")
    for key in (
        "dual_gate_sha256",
        "dual_gate_canonical_sha256",
        "tail23_verification_sha256",
        "hydra27_verification_sha256",
    ):
        _sha256(payload.get(key), key)
    payload["canonical_sha256"] = canonical
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "issue", "verify"):
        command = subparsers.add_parser(name)
        if name in ("validate", "issue"):
            # By PATH, never by runroot: a credential must be issued against
            # the dual gate re-run at the production plumbing commit.
            command.add_argument("--dual-gate", required=True, type=Path)
            command.add_argument("--expected-dual-gate-sha256", required=True)
            if name == "issue":
                command.add_argument("--out", required=True, type=Path)
        else:
            command.add_argument("--sidecar", required=True, type=Path)
            command.add_argument("--expected-sidecar-sha256", required=True)
        command.add_argument("--candidate-so", required=True, type=Path)
        command.add_argument("--expected-candidate-sha256", required=True)
        command.add_argument("--arm", required=True, choices=(ARM,))
        command.add_argument("--patch-source", required=True, type=Path)
        command.add_argument("--expected-source-commit", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "validate":
        result = validate_binding(
            dual_gate=args.dual_gate,
            expected_dual_gate_sha256=args.expected_dual_gate_sha256,
            candidate_so=args.candidate_so,
            expected_candidate_sha256=args.expected_candidate_sha256,
            arm=args.arm,
            patch_source=args.patch_source,
            expected_source_commit=args.expected_source_commit,
        )
    elif args.command == "issue":
        result = issue_sidecar(
            dual_gate=args.dual_gate,
            expected_dual_gate_sha256=args.expected_dual_gate_sha256,
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
