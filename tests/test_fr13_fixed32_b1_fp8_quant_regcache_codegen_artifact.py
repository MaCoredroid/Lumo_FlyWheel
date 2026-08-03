from __future__ import annotations

import hashlib
import json
from pathlib import Path


ARTIFACT = Path(
    "results/fr13_fixed32_b1_fp8_quant_regcache_sm121a_20260803"
)


def test_manifest_records_default_off_static_boundary_and_removed_work() -> None:
    payload = json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))
    assert payload["status"] == "sm121a_codegen_pass_default_off"
    assert payload["acceptance_valid"] is False
    assert payload["performance_claim"] is False
    assert payload["candidate"]["default_enabled"] is False
    assert payload["candidate"]["phase"] == "sfwd"
    assert payload["candidate"]["input_shape"] == [32, 5120]
    assert payload["candidate"]["group_size"] == 128
    assert payload["semantic_invariants"]["kernel_launch_count_changed"] is False
    removed = payload["removed_work"]
    assert removed["shared_traffic_bytes_per_quant_call"] == 655360
    assert removed["known_target_quant_calls_per_forward"] == 128
    assert removed["shared_traffic_bytes_per_target_forward"] == 83886080
    assert removed["cta_barriers_per_target_forward"] == 10240
    assert removed["quant_launches_removed_per_target_forward"] == 0


def test_manifest_records_spill_free_candidate_and_stock_identity() -> None:
    payload = json.loads((ARTIFACT / "manifest.json").read_text(encoding="ascii"))
    resource = payload["resource_audit"]
    assert resource["stock_registers_per_thread"] == 48
    assert resource["candidate_registers_per_thread"] == 26
    assert resource["candidate_stack_bytes_per_thread"] == 0
    assert resource["candidate_local_bytes_per_thread"] == 0
    assert resource["candidate_bar"] == 0
    assert resource["candidate_lds"] == 0
    assert resource["candidate_sts"] == 0
    assert resource["detected_spills"] is False
    assert payload["sass_audit"]["untouched_stock_sass_identical"] is True
    assert payload["sass_audit"]["performance_inference_allowed"] is False


def test_published_hash_manifest_is_complete() -> None:
    entries = {}
    for line in (ARTIFACT / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        digest, name = line.split("  ", 1)
        entries[name] = digest
    published = {
        path.name
        for path in ARTIFACT.iterdir()
        if path.is_file() and path.name != "SHA256SUMS"
    }
    assert set(entries) == published
    for name, expected in entries.items():
        assert hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest() == expected
