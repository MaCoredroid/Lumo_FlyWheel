from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
BLOCK_MAP = SCRIPTS / "fr13_dvk_subset_blocks.json"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_cutlass_b4_pass.py"
    spec = importlib.util.spec_from_file_location(
        "fr13_cutlass_b4_k64_profile_test", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load()
    candidate_bytes = b"persistent b4 m128 candidate\n"
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    patch_bytes = b"cutlass patch\n"
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    monkeypatch.setattr(module.binary, "B4_M128_CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module.binary, "B4_M128_CANDIDATE_SHA256", candidate_sha256)
    monkeypatch.setattr(module, "PATCH_SOURCE_SHA256", patch_sha256)
    profile = module.QUALIFICATION_PROFILES["k64_root"]
    task_marker = f"swe_verified:{module.EXPECTED_TASK_IDS[0]}"
    live = {
        "schema": profile["live_schema"],
        "status": "pass",
        "run_classification": profile["run_classification"],
        "acceptance_valid": False,
        "task_count": 4,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "topology": "hydra27_fixed32",
        "task_marker": task_marker,
        "qualification_profile": "k64_root",
        "draft_vocab_root": 1,
        "draft_vocab_k": 65_536,
        "draft_vocab_blocks": module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
        "draft_vocab_blocks_sha256": module.DRAFT_VOCAB_BLOCKS_SHA256,
        "mandatory_weight_bytes": module.K64_ROOT_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": module.K64_ROOT_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": module.K64_ROOT_SLO_CAP_MS,
        "comparator_timing_eligible": False,
        "batch_size": 4,
        "concurrency": 4,
        "fixed_rows": 128,
        "eager_builder_capacity": 128,
        "candidate": "persistent_b4_m128",
        "diagnostic_selector": "persistent_b4_m128_byte_ab",
        "served_result": "stock",
        "production_enabled": False,
        "comparison_call_limit": module.MAX_COMPARISONS,
        "comparisons": 5,
        "observed_m_values": [128],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
        "candidate_family": "persistent_b4_m128",
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": patch_sha256,
        "vllm_base_commit": module.VLLM_BASE_COMMIT,
        "patched_dispatch_sha256": module.PATCHED_DISPATCH_SHA256,
        "source_commit": "c" * 40,
        "binary_attestation_sha256": "d" * 64,
        "real_task_arm_sha256": "e" * 64,
        "container_env_sha256": "f" * 64,
        "errors": [],
    }
    live_path = tmp_path / "live.json"
    live_path.write_text(json.dumps(live, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live_path.read_bytes()).hexdigest()
    return module, candidate, patch_source, live_path, live_sha256


def test_k64_root_profile_issues_and_verifies_distinct_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="persistent_b4_m128",
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert issued == verified
    assert issued["schema"] == module.K64_ROOT_SIDECAR_SCHEMA
    assert issued["qualification_profile"] == "k64_root"
    assert issued["qualified_draft_vocab_root"] == 1
    assert issued["qualified_draft_vocab_k"] == 65_536
    assert (
        issued["qualified_draft_vocab_blocks"]
        == module.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH
    )
    assert (
        issued["qualified_draft_vocab_blocks_sha256"]
        == module.DRAFT_VOCAB_BLOCKS_SHA256
    )
    assert issued["mandatory_weight_bytes"] == 32_666_638_208
    assert issued["mandatory_weight_floor_ms"] == 119.658015414
    assert issued["one_sided_u95_cap_ms"] == 137.6067177261


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("qualification_profile", "full_vocab"),
        ("draft_vocab_root", 0),
        ("draft_vocab_k", 0),
        ("draft_vocab_blocks", "/tmp/unpinned.json"),
        ("draft_vocab_blocks_sha256", "0" * 64),
        ("mandatory_weight_bytes", 42_025_179_008),
        ("mandatory_weight_floor_ms", 153.9383846446886),
        ("one_sided_u95_cap_ms", 177.0291423413919),
    ),
)
def test_k64_root_profile_rejects_workload_or_floor_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    (
        module,
        candidate,
        patch_source,
        live,
        _,
    ) = _fixture(tmp_path, monkeypatch)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload[field] = value
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()

    with pytest.raises(module.QualificationError):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            expected_source_commit="c" * 40,
            qualification_profile="k64_root",
            draft_vocab_blocks=BLOCK_MAP,
        )


def test_full_vocab_default_cannot_consume_k64_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _fixture(tmp_path, monkeypatch)

    with pytest.raises(module.QualificationError):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            expected_source_commit="c" * 40,
        )


def test_k64_root_binary_attestation_preserves_profile_and_block_map(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _fixture(tmp_path, monkeypatch)
    sidecar = tmp_path / "sidecar.json"
    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    identity = {
        "bytes": candidate.stat().st_size,
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "regular": True,
        "symlink": False,
    }
    qualification_keys = (
        "live_result_sha256",
        "candidate_sha256",
        "patch_source_sha256",
        "qualification_source_commit",
        "qualification_task_marker",
        "real_task_arm_sha256",
        "container_env_sha256",
        "qualified_draft_vocab_root",
        "qualified_draft_vocab_k",
        "qualified_eager_builder_capacity",
        "qualified_topology",
        "qualified_comparison_call_limit",
        "mandatory_weight_bytes",
        "mandatory_weight_floor_ms",
        "one_sided_u95_cap_ms",
        "qualification_profile",
        "qualified_draft_vocab_blocks",
        "qualified_draft_vocab_blocks_sha256",
        "qualified_fixed_rows",
        "qualified_projection_nk",
    )
    attestation_payload = {
        "schema": module.ATTESTATION_SCHEMA,
        "selector": "persistent_b4_m128",
        "source": {"path": str(module.binary.CONTAINER_SOURCE), **identity},
        "destination": {
            "path": str(module.binary.CONTAINER_DESTINATION),
            **identity,
        },
        "installed_mode": "0555",
        "production_enabled": True,
        "qualification": {
            "sidecar_sha256": sidecar_sha256,
            **{key: issued[key] for key in qualification_keys},
        },
    }
    attestation = tmp_path / "attestation.json"
    attestation.write_text(
        json.dumps(attestation_payload, sort_keys=True) + "\n", encoding="ascii"
    )

    binding = module.validate_production_attestation(
        attestation,
        sidecar_sha256,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
    )

    assert binding["schema"] == (
        "fr13.fixed32.cutlass_b4.k64_root.production_binding.v1"
    )
    assert binding["qualification_profile"] == "k64_root"
    assert binding["qualified_fixed_rows"] == 128
    assert (
        binding["qualified_draft_vocab_blocks_sha256"]
        == module.DRAFT_VOCAB_BLOCKS_SHA256
    )


def test_k64_root_profile_is_bound_through_gate_timing_and_launcher() -> None:
    gate = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_live_gate.sh").read_text(
        encoding="utf-8"
    )
    timing = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh").read_text(
        encoding="utf-8"
    )
    timing_math = (SCRIPTS / "fr13_b4_timing_math.py").read_text(encoding="utf-8")
    launcher = (SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    binary = (SCRIPTS / "fr13_cutlass_wave_binary.py").read_text(encoding="utf-8")

    for source in (gate, timing, launcher):
        assert "k64_root" in source
        assert "65536" in source
        assert (
            "85dffa58703e42aaf7e248fe022c52c76b10364f67532ff724621ba3fce242ff" in source
        )
    for source in (gate, timing):
        assert "32666638208" in source
        assert "119.658015414" in source
        assert "137.6067177261" in source
        assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in source
        assert '"$ARM" "$FIXED32_MODE" "$SUBSET"' in source or (
            '"$arm" "$TIMING_KIND" "$SUBSET"' in source
        )
    assert '--qualification-profile "$QUALIFICATION_PROFILE"' in gate
    assert timing.count('FR13_DRAFT_VOCAB_ROOT="$DRAFT_VOCAB_ROOT"') >= 2
    assert timing.count('FR13_DRAFT_VOCAB_K="$DRAFT_VOCAB_K"') >= 2
    assert '"target_verifier_vocabulary": "full"' in timing
    assert '"sfwd_gpu_ms_per_step"' in timing_math
    assert '"dfwd_gpu_ms_per_step"' in timing_math
    assert '"cfwd_gpu_ms_per_step"' in timing_math
    assert 'positive(record, "s_per_fwd_gpu_per_forward")' in timing_math
    assert 'positive(record, "s_per_fwd_gpu") * 1000.0' not in timing
    assert '"gpu_component_ms_per_step"' in timing_math
    assert '"other_wall_ms_per_step"' in timing_math
    assert "phase components exceed full-step wall time" in timing_math
    assert "phase breakdown does not reconcile" in timing_math
    assert "qualified_draft_vocab_blocks_sha256" in binary
    assert (
        "-e FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="
        '"$FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE"'
    ) in launcher
    assert '--fixed32-mode "\\$FR13_FIXED32_MODE"' in launcher


def test_prepared_campaign_runs_one_gate_and_one_paired_screen() -> None:
    prepared = (
        REPO
        / "results"
        / "fr13_fixed32_cutlass_b4_persistent_m128_k64_route_20260801"
        / "prepared_campaign.sh"
    ).read_text(encoding="utf-8")

    assert (
        prepared.count("bash scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh")
        == 1
    )
    assert (
        prepared.count("bash scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh")
        == 1
    )
    assert "fr13_bigdenom_swe_serve_variant.sh" not in prepared
    assert prepared.count("CUTLASS_B4_QUALIFICATION_PROFILE=k64_root") == 2
    assert 'CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT="$gate_source_commit"' in prepared
    assert (
        "EXPECTED_CANDIDATE_SHA256="
        "895495fe82cb0e0278d3b0a39b8e57e1281aa73a10bbba01a94085733c81d64f"
    ) in prepared


def test_k64_root_profile_binds_tail23_without_reusing_hydra_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (
        module,
        candidate,
        patch_source,
        live,
        _,
    ) = _fixture(tmp_path, monkeypatch)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload["topology"] = "tail6_fixed32"
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "tail23-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
        qualification_profile="k64_root",
        draft_vocab_blocks=BLOCK_MAP,
        fixed32_mode="tail6_fixed32",
    )
    assert issued["qualified_topology"] == "tail6_fixed32"

    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    with pytest.raises(module.QualificationError, match="qualified_topology"):
        module.verify_sidecar(
            sidecar,
            sidecar_sha256,
            candidate,
            patch_source,
            qualification_profile="k64_root",
            draft_vocab_blocks=BLOCK_MAP,
        )


def test_tail23_hydra27_stack_route_is_real_exact4_and_fixed_k64() -> None:
    route = (SCRIPTS / "fr13_run_b4_tail23_hydra27_k64_m128_stack.sh").read_text(
        encoding="utf-8"
    )
    manifest = (SCRIPTS / "fr13_runtime_manifest.py").read_text(encoding="utf-8")

    assert route.count("run_topology tail6_fixed32 tail23") == 1
    assert route.count("run_topology hydra27_fixed32 hydra27") == 1
    assert route.index("run_topology tail6_fixed32 tail23") < route.index(
        "run_topology hydra27_fixed32 hydra27"
    )
    assert "CUTLASS_B4_QUALIFICATION_PROFILE=k64_root" in route
    assert 'FR13_FIXED32_ALL_PARENT_PASS_JSON="$taw_pass"' in route
    assert 'FR13_FIXED32_ALL_PARENT_VERDICT_JSON="$taw_verdict"' in route
    assert route.count("fr13_run_b4_tail23_all_parent_live_gate.sh") == 1
    assert route.count("fr13_run_b4_cutlass_persistent_m128_live_gate.sh") == 1
    assert route.count("fr13_run_b4_cutlass_persistent_m128_timing.sh") == 1
    assert "subset_b4_four" not in route  # Canonical exact4 is owned by gate runners.
    assert '"physical_rows_per_request": 32' in route
    assert '"sfwd_projection_rows": 128' in route
    assert '"draft_vocab_k": 65536' in route
    assert '"target_verifier_vocabulary": "full"' in route
    assert '"qwen_turn_tool_call_cap": qwen_derivation.DERIVED_CAP' in route
    assert (
        'timing.get("all_parent_verdict_sha256") != taw_sha256'
        in route
    )
    assert (
        'timing.get("candidate", {}).get("live_result_sha256") != m128_sha256'
        in route
    )
    for field in (
        "accepted_drafts_per_event",
        "committed_tokens_per_event",
        "measured_tps_fullstep_wall",
        "sfwd_gpu_ms_per_step",
        "dfwd_gpu_ms_per_step",
        "cfwd_gpu_ms_per_step",
        "other_wall_ms_per_step",
        "step_wall_to_optimistic_floor_ratio",
    ):
        assert field in route
    assert '"scripts/fr13_run_b4_tail23_hydra27_k64_m128_stack.sh"' in manifest
    assert '"scripts/fr13_b4_timing_math.py"' in manifest


def test_b4_timing_binds_all_parent_verdict_and_uses_per_step_sfwd() -> None:
    timing = (
        SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh"
    ).read_text(encoding="utf-8")

    assert "FR13_FIXED32_ALL_PARENT_VERDICT_JSON" in timing
    assert "all-parent exact4 credential is scoped to k64_root" in timing
    assert 'verdict.get("production_bundle_sha256") != pass_sha256' in timing
    assert 'verdict.get("source_commit") != source_commit' in timing
    assert (
        'verdict.get("task_marker") != f"swe_verified:campaign4_{subset_sha256}"'
        in timing
    )
    assert '"$ALL_PARENT_VERDICT_SHA256"' in timing
    assert "fixed32_native_precompute_production_candidate_return" in timing
    assert "all-parent production did not engage on every measured event" in timing
    assert "from fr13_b4_timing_math import phase_breakdown, positive" in timing
    assert 'positive(record, "s_per_fwd_gpu_per_forward")' not in timing
