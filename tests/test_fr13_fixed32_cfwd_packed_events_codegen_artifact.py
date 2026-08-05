from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "results/fr13_fixed32_cfwd_packed_events_sm121a_codegen_20260805"
RESOURCE_FIELDS = ("stack_bytes", "local_bytes", "ldl", "stl", "calls")


def _clean(build: dict) -> bool:
    return all(build[name] == 0 for name in RESOURCE_FIELDS)


def test_packed_event_codegen_summary_has_exact_v2_deltas() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["schema"] == "fr13.fixed32.cfwd_packed_events.sm121a.codegen.v1"
    assert summary["status"] == "pass"
    assert summary["base_revision"] == "1f7485ade5ec6bfacf51dde7afa514531effcbcd"
    assert summary["candidate_revision"] == (
        "103030ea88ad7da28a4bcab187a57200be70756d"
    )
    assert summary["claim_scope"] == (
        "static_sm121a_codegen_and_exact_cpu_semantics_no_runtime_claim"
    )
    assert summary["exact_work"] == {
        "commit_programs_per_request_after": 1,
        "commit_programs_per_request_before": 1,
        "decision_programs_per_request_after": 30,
        "decision_programs_per_request_before": 30,
        "decision_values_stored_per_request_after": 30,
        "decision_values_stored_per_request_before": 81,
        "decision_workspace_bytes_per_request_after": 504,
        "decision_workspace_bytes_per_request_before": 1_048,
        "physical_rows": 32,
        "tree_metadata_loads_per_request_after": 0,
        "tree_metadata_loads_per_request_before": 24,
        "walk_levels": 12,
    }
    direct = "_fr13_cfwd_logit_direct_decision_kernel"
    before = summary["producer"]["base"][direct]
    after = summary["producer"]["candidate"][direct]
    assert (before["registers"], after["registers"]) == (80, 80)
    assert (before["ldg"], after["ldg"]) == (51, 51)
    assert (before["stg"], after["stg"]) == (5, 2)
    assert (
        before["static_noncontrol_sass_instructions"],
        after["static_noncontrol_sass_instructions"],
    ) == (2_558, 2_565)
    assert _clean(before) and _clean(after)
    for batch in ("b1", "b4"):
        before = summary["commit"]["base"][batch]
        after = summary["commit"]["candidate"][batch]
        assert (before["registers"], after["registers"]) == (64, 46)
        assert (before["ldg"], after["ldg"]) == (95, 35)
        assert (before["stg"], after["stg"]) == (41, 41)
        assert (
            before["static_noncontrol_sass_instructions"],
            after["static_noncontrol_sass_instructions"],
        ) == (684, 509)
        assert (before["encoded_sass_instructions"], after["encoded_sass_instructions"]) == (696, 520)
        assert (before["cubin_bytes"], after["cubin_bytes"]) == (59_656, 46_176)
        assert _clean(before) and _clean(after)
    assert summary["comparator"]["b1"]["registers"] == 40
    assert summary["comparator"]["b4"]["registers"] == 38
    assert all(_clean(summary["comparator"][batch]) for batch in ("b1", "b4"))


def test_packed_event_codegen_summary_pins_contracts() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["source_contracts"] == {
        "candidate": {
            "name": "fixed32_cfwd_logit_direct_packed_physical_slots_v3",
            "schema": "fr13.fixed32.cfwd_logit_direct_packed_physical_slots.v3",
            "sha256": "5a9107306bdc37200448a6a5add2b84dfd839dc377b11009f218662c63abcc1c",
        },
        "cfwd_integration": {
            "schema": "fr13.fixed32.cfwd_logit_direct.integration_source.v2",
            "sha256": "a82ce3f5e526792ca45bb444212e5440e8444778f174fd0650accc4bb5f8558c",
        },
        "taw": {
            "schema": "fr13-fixed32-taw-all-parent-v7",
            "sha256": "998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290",
            "unchanged_by_candidate": True,
        },
    }
    assert summary["packed_event_contract"] == {
        "accepted_node_zero_row": 1,
        "accepted_row_mask": 31,
        "accepted_row_shift": 18,
        "parent_mask": 8_388_608,
        "rejection_accepted_row": 0,
        "token_mask": 262_143,
        "verifier_vocab_fits_token_bits": True,
        "verifier_vocab_size": 248_320,
    }


def test_packed_event_artifact_source_checksums_match_candidate() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    revision = summary["candidate_revision"]
    entries = (ARTIFACT / "source_checksums.sha256").read_text().splitlines()
    assert len(entries) == 3
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        historical = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{revision}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected


def test_packed_event_artifact_states_unrun_real_gates() -> None:
    readme = (ARTIFACT / "README.md").read_text()
    results = (ARTIFACT / "test_results.txt").read_text()
    assert "runtime speedup" in readme
    assert "GPU execution was not performed" in readme
    assert "default-off" in readme
    assert "one-task" in readme
    assert "4-task" in readme
    assert "16-task" in readme
    assert "gpu_execution=NOT_RUN" in results
    assert "real_swe_verified_one_task_byte_gate=REQUIRED" in results
    assert "runtime_speedup_claim=NONE" in results


def test_packed_event_artifact_manifest_matches() -> None:
    entries = (ARTIFACT / "SHA256SUMS").read_text().splitlines()
    assert len(entries) == 7
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        assert hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest() == expected
