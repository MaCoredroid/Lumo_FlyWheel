#!/usr/bin/env python3
"""Validate one real B1 boot through CFWD and candidate-served DFWD U8 gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import fr13_cfwd_logit_direct_gate as cfwd
from scripts import fr13_dfwd_k64_m1_r64_u8_gate as dfwd


SCHEMA = "fr13.fixed32.cfwd_dfwd_u8.composed_real_b1_gate.v2"
TASK_ID = "astropy__astropy-12907"
SUBSET = Path("config/fr13_fixed32/subset_b1_diagnostic_one.json")
U8_SOURCE = Path("csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu")
U8_BUILD = Path(
    "results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/"
    "build_attestation.json"
)
PATCH_SOURCE = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")
RUNNER = Path("scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh")
VOCAB_BLOCKS = Path("scripts/fr13_dvk_subset_blocks.json")
CFWD_SOURCE = Path("scripts/fr13_cfwd_logit_direct_decision_kernel.py")


class GateError(ValueError):
    pass


def _regular(path: Path, label: str) -> bytes:
    try:
        info = path.lstat()
        raw = path.read_bytes()
    except OSError as error:
        raise GateError(f"{label} is unavailable: {path}") from error
    if (
        path.is_symlink()
        or not stat.S_ISREG(info.st_mode)
        or info.st_nlink != 1
        or not raw
    ):
        raise GateError(f"{label} must be one nonempty regular file: {path}")
    return raw


def _object(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _regular(path, label)
    try:
        payload = json.loads(raw.decode("ascii"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise GateError(f"{label} is not ASCII JSON") from error
    if not isinstance(payload, dict):
        raise GateError(f"{label} must contain one JSON object")
    return payload, raw


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def validate_composed_gate(
    *,
    repo: Path,
    source_commit: str,
    cfwd_credential: Path,
    cfwd_live_result: Path,
    dfwd_gate: Path,
    dfwd_live_result: Path,
    candidate_so: Path,
    fa2_so: Path,
    final_flush: Path,
    boundary_snapshot: Path,
    traffic_audit: Path,
) -> dict[str, Any]:
    repo = repo.resolve()
    paths = {
        "cfwd_credential": cfwd_credential.resolve(),
        "cfwd_live_result": cfwd_live_result.resolve(),
        "dfwd_gate": dfwd_gate.resolve(),
        "dfwd_live_result": dfwd_live_result.resolve(),
        "candidate_so": candidate_so.resolve(),
        "fa2_so": fa2_so.resolve(),
        "final_flush": final_flush.resolve(),
        "boundary_snapshot": boundary_snapshot.resolve(),
        "traffic_audit": traffic_audit.resolve(),
    }
    cfwd_payload, cfwd_raw = _object(paths["cfwd_credential"], "CFWD credential")
    dfwd_payload, dfwd_raw = _object(paths["dfwd_gate"], "DFWD U8 gate")

    with tempfile.TemporaryDirectory(prefix="fr13-cfwd-u8-compose-") as scratch:
        expected_cfwd = cfwd.issue(
            live_result=paths["cfwd_live_result"],
            subset=repo / SUBSET,
            final_flush=paths["final_flush"],
            boundary_snapshot=paths["boundary_snapshot"],
            traffic_audit=paths["traffic_audit"],
            candidate_source=repo / CFWD_SOURCE,
            source_commit=source_commit,
            output=Path(scratch) / "cfwd.json",
        )
    if not dfwd._json_exact(cfwd_payload, expected_cfwd):
        raise GateError("recorded CFWD credential differs from unchanged validator")

    expected_dfwd = dfwd.validate_gate(
        live_result=paths["dfwd_live_result"],
        candidate_so=paths["candidate_so"],
        candidate_source=repo / U8_SOURCE,
        build_attestation=repo / U8_BUILD,
        patch_source=repo / PATCH_SOURCE,
        runner=repo / RUNNER,
        subset=repo / SUBSET,
        vocab_blocks=repo / VOCAB_BLOCKS,
        fa2_so=paths["fa2_so"],
        expected_source_commit=source_commit,
        final_flush=paths["final_flush"],
        boundary_snapshot=paths["boundary_snapshot"],
        chat_traffic_audit=paths["traffic_audit"],
        repo=repo,
    )
    if not dfwd._json_exact(dfwd_payload, expected_dfwd):
        raise GateError("recorded DFWD U8 result differs from unchanged validator")

    final_raw = _regular(paths["final_flush"], "shared final flush")
    boundary_raw = _regular(paths["boundary_snapshot"], "shared boundary snapshot")
    traffic_raw = _regular(paths["traffic_audit"], "shared traffic audit")
    events = cfwd_payload["complete_work_census_events"]
    if (
        cfwd_payload["integration_source_commit"] != source_commit
        or dfwd_payload["source_commit"] != source_commit
        or cfwd_payload["task_ids"] != [TASK_ID]
        or dfwd_payload["completed_events"] != events
        or cfwd_payload["final_flush_sha256"] != _sha(final_raw)
        or dfwd_payload["final_flush_sha256"] != _sha(final_raw)
        or cfwd_payload["boundary_snapshot_sha256"] != _sha(boundary_raw)
        or dfwd_payload["boundary_snapshot_sha256"] != _sha(boundary_raw)
        or cfwd_payload["traffic_audit_sha256"] != _sha(traffic_raw)
        or dfwd_payload["chat_traffic_audit_sha256"] != _sha(traffic_raw)
        or cfwd_payload["reference_always_served"] is not True
        or dfwd_payload["reference_always_served"] is not False
        or dfwd_payload["candidate_returned"] is not True
        or dfwd_payload["nonfinite_logits"] != 0
        or dfwd_payload["qualification_policy"]
        != "lossless_deterministic_proposal_v1"
        or not dfwd._json_exact(
            dfwd_payload["proposal_distribution"],
            dfwd.EXPECTED_PROPOSAL_DISTRIBUTION,
        )
        or cfwd_payload["timing_eligible"] is not False
        or dfwd_payload["timing_eligible"] is not False
    ):
        raise GateError("CFWD and DFWD U8 results do not bind one shared real event stream")

    return {
        "schema": SCHEMA,
        "status": "PASS",
        "source_commit": source_commit,
        "suite": "SWE-Verified",
        "task_ids": [TASK_ID],
        "batch_size": 1,
        "concurrency": 1,
        "mode": "hydra27_fixed32",
        "physical_rows": 32,
        "draft_vocab_k": 65_536,
        "draft_vocab_root": 1,
        "execution_basis": "FULL_AND_PIECEWISE_graph_replay",
        "shared_complete_work_census_events": events,
        "shared_final_flush_sha256": _sha(final_raw),
        "shared_boundary_snapshot_sha256": _sha(boundary_raw),
        "shared_traffic_audit_sha256": _sha(traffic_raw),
        "cfwd_credential_sha256": _sha(cfwd_raw),
        "dfwd_u8_gate_sha256": _sha(dfwd_raw),
        "component_validators_reexecuted": True,
        "cfwd_reference_served": True,
        "dfwd_u8_candidate_served": True,
        "reference_always_served": False,
        "candidates_returned": True,
        "dfwd_u8_proposal_distribution": dfwd.EXPECTED_PROPOSAL_DISTRIBUTION,
        "performance_measurement": False,
        "timing_eligible": False,
        "floor_acceptance_eligible": False,
        "production_eligible": False,
        "sfwd_gate_pack_qualified": False,
        "sfwd_requires_separate_eager_qrow16_boot": True,
    }


def _write(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise GateError(f"refusing to replace composed gate result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        + "\n",
        encoding="ascii",
    )
    temporary.chmod(0o600)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cfwd-credential", type=Path, required=True)
    parser.add_argument("--cfwd-live-result", type=Path, required=True)
    parser.add_argument("--dfwd-gate", type=Path, required=True)
    parser.add_argument("--dfwd-live-result", type=Path, required=True)
    parser.add_argument("--candidate-so", type=Path, required=True)
    parser.add_argument("--fa2-so", type=Path, required=True)
    parser.add_argument("--final-flush", type=Path, required=True)
    parser.add_argument("--boundary-snapshot", type=Path, required=True)
    parser.add_argument("--traffic-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = validate_composed_gate(
        repo=args.repo,
        source_commit=args.source_commit,
        cfwd_credential=args.cfwd_credential,
        cfwd_live_result=args.cfwd_live_result,
        dfwd_gate=args.dfwd_gate,
        dfwd_live_result=args.dfwd_live_result,
        candidate_so=args.candidate_so,
        fa2_so=args.fa2_so,
        final_flush=args.final_flush,
        boundary_snapshot=args.boundary_snapshot,
        traffic_audit=args.traffic_audit,
    )
    _write(args.out.resolve(), result)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
