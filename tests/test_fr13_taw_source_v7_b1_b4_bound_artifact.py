from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results" / "fr13_fixed32_taw_source_v7_b1_b4_bound_20260805"


def _load(name: str) -> dict[str, object]:
    return json.loads((ARTIFACT / name).read_text(encoding="ascii"))


def _sha256(name: str) -> str:
    return hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest()


def test_bound_artifact_matches_credential_chain() -> None:
    manifest = _load("manifest.json")
    b1 = _load("b1_credential.json")
    b4 = _load("b4_byte_gate.json")
    merged = _load("merged_production_pass.json")
    binding = _load("merge_binding.json")

    assert manifest["status"] == "production_ready"
    assert b1["status"] == b4["status"] == "pass"
    assert b1["source_file_sha256"] == b4["source_file_sha256"]
    assert b1["source_contract_sha256"] == b4["source_contract_sha256"]
    assert merged["status"] == "production_ready"
    assert merged["qualified_batches"] == [1, 2, 3, 4]
    assert binding["status"] == "bound"
    assert binding["preserved_reviewed_batches"] == [2, 3, 4]
    assert _sha256("b1_credential.json") == manifest["b1"]["credential_sha256"]
    assert _sha256("b1_live_bundle.json") == manifest["b1"]["live_bundle_sha256"]
    assert _sha256("b4_production_pass.json") == manifest["b4"]["production_bundle_sha256"]
    assert _sha256("b4_byte_gate.json") == manifest["b4"]["gate_verdict_sha256"]
    assert _sha256("merged_production_pass.json") == manifest["merge"]["production_bundle_sha256"]
    assert _sha256("merge_binding.json") == manifest["merge"]["binding_sha256"]


def test_all_replay_rows_are_exact_and_shadow_only() -> None:
    b1 = _load("b1_live_bundle.json")
    b4 = _load("b4_production_pass.json")
    merged = _load("merged_production_pass.json")

    assert b1["qualified_batches"] == [1]
    assert b4["qualified_batches"] == [1, 2, 3, 4]
    for batch in (1, 2, 3, 4):
        record = merged["batch_passes"][str(batch)]
        assert record["status"] == "pass"
        assert record["evidence_route"] == "full_graph_replay"
        assert record["probability_mismatches"] == 0
        assert record["product_mismatches"] == 0
        assert record["reference_returned"] is True
        assert record["candidate_returned"] is False


def test_preboot_failure_cannot_be_mistaken_for_evidence() -> None:
    failure = _load("failed_preboot_attempt.json")
    assert failure["status"] == "failed_preboot"
    assert failure["container_created"] is False
    assert failure["authenticated_task_started"] is False
    assert failure["gpu_work_started"] is False
    assert failure["credential_issued"] is False
    assert failure["timing_eligible"] is False
    assert failure["floor_acceptance_eligible"] is False

