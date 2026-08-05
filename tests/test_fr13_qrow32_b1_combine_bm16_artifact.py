from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_fa2_qrow32_b1_split2_combine_bm16_sm121a_20260805"
)
PATCHER = ROOT / "scripts" / "fr13_patch_fa2_tree_bias.py"


def _patcher_module():
    spec = importlib.util.spec_from_file_location("fr13_fa2_bm16", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _json(name: str) -> dict:
    return json.loads((ARTIFACT / name).read_text(encoding="ascii"))


def test_bm16_artifact_is_host_only_and_fail_closed() -> None:
    manifest = _json("manifest.json")

    assert manifest["status"] == "HOST_CODEGEN_PASS_ADMIT_REAL_SWE_VERIFIED_AB"
    assert manifest["classification"] == {
        "synthetic_probe_used": False,
        "real_swe_task_used": False,
        "byte_parity_claim": False,
        "performance_claim": False,
        "production_eligible": False,
        "safe_to_merge_over_pinned_generator": False,
    }
    assert manifest["admission"] == {
        "verdict": "ADMIT_TO_REAL_SWE_VERIFIED_AB",
        "real_b1_byte_gate": "PENDING",
        "real_b1_full_step_timing": "PENDING",
        "production": "NOT_AUTHORIZED",
    }


def test_bm16_source_and_closure_are_bound_to_the_generator() -> None:
    patcher = _patcher_module()
    manifest = _json("manifest.json")
    closure = _json("source_closure.json")
    translation_unit = patcher.FIXED32_QUERY_TILE32_B1_SPLIT2_TRANSLATION_UNIT

    assert "kCombineBlockM = 16" in translation_unit
    assert "kCombineRows / kCombineBlockM == 48" in translation_unit
    assert hashlib.sha256(translation_unit.encode("utf-8")).hexdigest() == (
        manifest["source"]["split2_translation_unit_sha256"]
    )
    assert hashlib.sha256(PATCHER.read_bytes()).hexdigest() == (
        manifest["source"]["generator_sha256"]
    )
    canonical = {"fa2_head": closure["fa2_head"], "files": closure["files"]}
    raw = json.dumps(
        canonical,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    assert hashlib.sha256(raw).hexdigest() == closure["canonical_sha256"]
    assert closure["canonical_sha256"] == (
        manifest["source"]["source_closure_sha256"]
    )


def test_bm16_economics_preserve_work_while_reducing_ctas() -> None:
    manifest = _json("manifest.json")
    control = manifest["control_bm4"]
    candidate = manifest["candidate_bm16"]
    equivalence = manifest["equivalence"]
    economics = manifest["economics"]

    assert control["ctas_per_layer"] == 768 // control["block_m"] == 192
    assert candidate["ctas_per_layer"] == 768 // candidate["block_m"] == 48
    assert candidate["stack_bytes"] == candidate["local_bytes"] == 0
    assert equivalence["attention_main_sass_identical"] is True
    assert equivalence["logical_output_coverage_identical"] is True
    assert equivalence["lse_reduction_lanes_control"] == 2
    assert equivalence["lse_reduction_lanes_candidate"] == 2
    assert equivalence["per_output_split_accumulation_order"] == [0, 1]
    assert equivalence["real_task_raw_byte_comparison"] == "PENDING"
    assert economics["combine_cta_reduction_percent"] == 75.0
    assert economics["combine_bytes_per_layer"] == 1_975_296
    assert economics["combine_bytes_per_16_layer_step"] == 31_604_736
    assert economics["kernel_launches_removed_per_layer"] == 0
    assert economics["measured_performance"] is False
