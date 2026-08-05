from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = (
    ROOT
    / "results/fr13_fixed32_cfwd_physical_slots_sm121a_codegen_20260805"
)


def test_physical_slot_codegen_summary_is_narrow_and_spill_free() -> None:
    summary = json.loads((ARTIFACT / "codegen_summary.json").read_text())
    assert summary["status"] == "pass"
    assert summary["base_revision"] == (
        "3bdd984c2408467d321c35e758af3776983aaf38"
    )
    assert summary["candidate_revision"] == (
        "1a53d3a01f73d89a6725ac0de94c65ef62bd5fef"
    )
    assert summary["schema"] == (
        "fr13.fixed32.cfwd_physical_slots_preseeded.sm121a.codegen.v1"
    )
    assert summary["claim_scope"] == (
        "static_sm121a_codegen_and_exact_work_only_no_runtime_speed_claim"
    )
    assert summary["logical_work"] == {
        "commit_launches_per_event_after": 1,
        "commit_launches_per_event_before": 1,
        "commit_programs_per_request_after": 1,
        "commit_programs_per_request_before": 1,
        "decision_programs_per_request_after": 30,
        "decision_programs_per_request_before": 30,
        "decision_values_stored_per_request_after": 81,
        "decision_values_stored_per_request_before": 81,
        "decision_workspace_bytes_per_request_after": 1_048,
        "decision_workspace_bytes_per_request_before": 529,
        "topology_index_scalar_loads_per_request_after": 0,
        "topology_index_scalar_loads_per_request_before": 24,
    }
    assert summary["safety_contract"] == {
        "cfwd_integration_source_schema": (
            "fr13.fixed32.cfwd_logit_direct.integration_source.v1"
        ),
        "cfwd_integration_source_sha256": (
            "cc266bd4468c78193ef63701489eba666ec14b91530443a92439051796a6cc09"
        ),
        "decision_padding_initialization_stores_per_event": 0,
        "decision_workspace_zero_seeded_once": True,
        "hot_walk_dynamic_load_masks_added": 0,
        "incumbent_taw_source_sha256": (
            "998bc6331177469d6890f97f3e066e1d07c2ca2d8ab4bff723f32d5229fef290"
        ),
        "leaf_child_table_source_zero_value": -1,
        "leaf_unwritten_source_value": 0,
        "runtime_source_contract_attests_physical_committer": True,
    }
    direct = "_fr13_cfwd_logit_direct_decision_kernel"
    assert summary["producer"]["incumbent"][direct]["registers"] == 80
    assert summary["producer"]["candidate"][direct]["registers"] == 80
    for batch in ("b1", "b4"):
        incumbent = summary["commit"]["incumbent"][batch]
        candidate = summary["commit"]["candidate"][batch]
        assert (incumbent["registers"], candidate["registers"]) == (66, 64)
        assert (incumbent["ldg"], candidate["ldg"]) == (118, 95)
        assert (
            incumbent["static_noncontrol_sass_instructions"],
            candidate["static_noncontrol_sass_instructions"],
        ) == (747, 684)
        assert incumbent["stg"] == candidate["stg"] == 41
        for build in (incumbent, candidate):
            assert all(
                build[name] == 0
                for name in (
                    "stack_bytes",
                    "local_bytes",
                    "ldl",
                    "stl",
                    "calls",
                )
            )
    assert summary["conclusion"] == {
        "committer_static_improves": True,
        "producer_registers_preserved": True,
        "real_swe_verified_gate_required": True,
        "resource_clean": True,
        "runtime_speedup_claimed": False,
    }


def test_physical_slot_artifact_source_checksums_match() -> None:
    artifact_commit = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "log",
            "-1",
            "--format=%H",
            "--",
            str(ARTIFACT.relative_to(ROOT) / "source_checksums.sha256"),
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    assert artifact_commit
    entries = (ARTIFACT / "source_checksums.sha256").read_text().splitlines()
    assert len(entries) == 3
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        historical = subprocess.run(
            ["git", "-C", str(ROOT), "show", f"{artifact_commit}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        observed = hashlib.sha256(historical).hexdigest()
        assert observed == expected


def test_physical_slot_artifact_states_unrun_real_gates() -> None:
    readme = (ARTIFACT / "README.md").read_text()
    results = (ARTIFACT / "test_results.txt").read_text()
    assert "not claim a runtime speedup" in readme
    assert "zero-seeded once" in readme
    assert "supersedes the unsafe" in readme
    assert "separate CFWD integration source contract" in readme
    assert "998bc633" in readme
    assert "one-task" in readme
    assert "4-task" in readme
    assert "16-task" in readme
    assert "gpu_execution=NOT_RUN" in results
    assert "real_swe_verified_one_task_byte_gate=REQUIRED" in results
    assert "runtime_speedup_claim=NONE" in results


def test_physical_slot_artifact_manifest_matches() -> None:
    entries = (ARTIFACT / "SHA256SUMS").read_text().splitlines()
    assert len(entries) == 7
    for entry in entries:
        expected, relative = entry.split("  ", 1)
        observed = hashlib.sha256((ARTIFACT / relative).read_bytes()).hexdigest()
        assert observed == expected
