from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load():
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / "fr13_cutlass_b4_pass.py"
    spec = importlib.util.spec_from_file_location("fr13_cutlass_b4_pass_test", path)
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
    task_marker = f"swe_verified:{module.EXPECTED_TASK_IDS[1]}"
    live = {
        "schema": module.LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "real_swe_verified_exact4_b4_byte_diagnostic",
        "acceptance_valid": False,
        "task_count": 4,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "topology": "hydra27_fixed32",
        "task_marker": task_marker,
        "draft_vocab_root": 0,
        "draft_vocab_k": 0,
        "mandatory_weight_bytes": module.EXPECTED_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": module.EXPECTED_MANDATORY_WEIGHT_FLOOR_MS,
        "one_sided_u95_cap_ms": module.EXPECTED_SLO_CAP_MS,
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
    return module, candidate, patch_source, live_path, live_sha256, task_marker


def test_b4_projection_contract_uses_packed_full_attention_qkv() -> None:
    module = _load()

    assert (14336, 5120) in module.EXPECTED_PROJECTION_NK
    assert (8192, 5120) not in module.EXPECTED_PROJECTION_NK
    assert module.MAX_COMPARISONS == 320


def test_exact4_b4_pass_issues_and_verifies_production_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256, task_marker = _fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        expected_source_commit="c" * 40,
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="persistent_b4_m128",
    )

    assert issued == verified
    assert issued["schema"] == module.SIDECAR_SCHEMA
    assert "qualification_profile" not in issued
    assert "qualified_draft_vocab_blocks" not in issued
    assert issued["qualification_task_marker"] == task_marker
    assert issued["qualified_fixed_rows"] == 128
    assert issued["qualified_draft_vocab_root"] == 0
    assert issued["qualified_draft_vocab_k"] == 0
    assert issued["qualified_eager_builder_capacity"] == 128
    assert issued["qualified_topology"] == "hydra27_fixed32"
    assert issued["qualified_comparison_call_limit"] == 320


def test_b4_production_attestation_preserves_exact4_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256, task_marker = _fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"
    issued = module.issue_sidecar(live, live_sha256, candidate, sidecar, patch_source)
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    candidate_sha256 = hashlib.sha256(candidate.read_bytes()).hexdigest()
    identity = {
        "bytes": candidate.stat().st_size,
        "sha256": candidate_sha256,
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

    binding = module.validate_production_attestation(attestation, sidecar_sha256)

    assert binding["status"] == "BOUND"
    assert binding["qualification_task_marker"] == task_marker
    assert binding["qualification_task_ids"] == list(module.EXPECTED_TASK_IDS)
    assert binding["qualified_fixed_rows"] == 128
    assert binding["qualified_eager_builder_capacity"] == 128
    assert binding["qualified_topology"] == "hydra27_fixed32"
    assert binding["qualified_comparison_call_limit"] == 320
    assert (
        binding["mandatory_weight_floor_ms"]
        == module.EXPECTED_MANDATORY_WEIGHT_FLOOR_MS
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_count", 1),
        ("batch_size", 1),
        ("fixed_rows", 32),
        ("draft_vocab_root", 1),
        ("draft_vocab_k", 65_536),
        ("eager_builder_capacity", 32),
        ("topology", "tail6_fixed32"),
        ("comparison_call_limit", 256),
        ("task_marker", "swe_verified:django__django-10097"),
    ],
)
def test_b4_pass_rejects_noncanonical_workload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    module, candidate, patch_source, live, _, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload[field] = value
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()

    with pytest.raises(module.QualificationError):
        module.validate_live_result(
            live, live_sha256, candidate, patch_source, expected_source_commit="c" * 40
        )


def test_b4_gate_and_timing_are_closed_over_by_runtime_manifest() -> None:
    manifest = (SCRIPTS / "fr13_runtime_manifest.py").read_text(encoding="utf-8")
    gate = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_live_gate.sh").read_text(
        encoding="utf-8"
    )
    timing = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh").read_text(
        encoding="utf-8"
    )

    for path in (
        "scripts/fr13_cutlass_b4_pass.py",
        "scripts/fr13_run_b4_cutlass_persistent_m128_live_gate.sh",
        "scripts/fr13_run_b4_cutlass_persistent_m128_timing.sh",
    ):
        assert f'"{path}"' in manifest
    assert "subset_b4_four.json" in gate
    assert "persistent_b4_m128_byte_ab" in gate
    assert "fixed32_cutlass_b4_byte_ab.real_event.arm" in gate
    assert 'sudo -n -- "$PYTHON_BIN" - \\' in gate
    assert '"task_count": 4' in gate
    assert '"fixed_rows": 128' in gate
    assert '"eager_builder_capacity": 128' in gate
    assert (
        "QUALIFICATION_PROFILE=${CUTLASS_B4_QUALIFICATION_PROFILE:-full_vocab}" in gate
    )
    assert "DRAFT_VOCAB_ROOT=0" in gate
    assert "DRAFT_VOCAB_K=0" in gate
    assert "NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0" in gate
    assert "37335563648" in gate
    assert "136.7603064029304" in gate
    assert "157.27435236336996" in gate
    assert "f51e23c5c84f7256c99ccc36d7b049e464d5ef81b1ab095bf5629c28ad45f19d" in gate
    assert "STOCK_FA2_BYTES=299183936" in gate
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in gate
    assert "AGENT_WALL_S=5400" in gate
    assert '"$ARM" "$FIXED32_MODE" "$SUBSET"' in gate
    assert "CUTLASS_B4_FIXED32_MODE" in gate
    assert 'f"FR13_FIXED32_MODE={fixed32_mode}"' in gate
    assert "tail6_fixed32" in gate
    assert "COMPARISON_CALL_LIMIT=320" in gate
    assert "comparisons > MAX_COMPARISONS" in (
        SCRIPTS / "fr13_cutlass_b4_pass.py"
    ).read_text(encoding="utf-8")
    assert "persistent_b4_m128" in timing
    assert "--batch-size 4" in timing
    assert (
        "QUALIFICATION_PROFILE=${CUTLASS_B4_QUALIFICATION_PROFILE:-full_vocab}"
        in timing
    )
    assert "DRAFT_VOCAB_ROOT=0" in timing
    assert "DRAFT_VOCAB_K=0" in timing
    assert "NEEDS_ALLOW=FR13_DRAFT_VOCAB_K=0" in timing
    assert "37335563648" in timing
    assert "136.7603064029304" in timing
    assert "157.27435236336996" in timing
    assert "STOCK_FA2_BYTES=299183936" in timing
    assert "MAX_NUM_SEQS_OVR=4 SWE_CONCURRENCY=4" in timing
    assert "AGENT_WALL_S=5400" in timing
    assert "only_arm_delta=CUTLASS_stock_to_persistent_b4_m128" in timing
    assert 'binding.get("qualified_topology") != fixed32_mode' in timing
    assert 'binding.get("qualified_comparison_call_limit") != 320' in timing
    assert '"optimistic_floor_is_full_step_hardware_floor": False' in timing
    assert 'record.get("floor_is_full_step_hardware_floor") is not False' in timing


def test_b4_timing_keeps_live_qualification_separate_from_harness_commit() -> None:
    timing = (SCRIPTS / "fr13_run_b4_cutlass_persistent_m128_timing.sh").read_text(
        encoding="utf-8"
    )
    launcher = (SCRIPTS / "fr13_launch_forked_fa2_tree_server.sh").read_text(
        encoding="utf-8"
    )
    qualification_commit = "0f2a31ed298758cba72fad7e77fc3e13e27d545a"
    patch_sha256 = "656c53b20497fc08cc7fdfb18256235b07cfad9868fde2faa70e6b0b9dfca41a"

    assert (
        "QUALIFICATION_SOURCE_COMMIT="
        "${CUTLASS_B4_QUALIFICATION_SOURCE_COMMIT:-"
        f"{qualification_commit}}}"
    ) in timing
    assert "TIMING_HARNESS_COMMIT=$(git rev-parse HEAD)" in timing
    assert f"QUALIFIED_PATCH_SOURCE_SHA256={patch_sha256}" in timing
    assert 'git show "${QUALIFICATION_SOURCE_COMMIT}:${PATCH_SOURCE}"' in timing
    assert '--expected-source-commit "$QUALIFICATION_SOURCE_COMMIT"' in timing
    assert "cutlass_b4_live_pass_binding.at_launch.json" in timing
    assert "qualification_source_commit=%s" in timing
    assert "timing_harness_commit=%s" in timing
    assert "qualified_patch_source_sha256=%s" in timing
    assert (
        'FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT="${qualification_source_commit:-}"'
        in timing
    )
    assert (
        'binding.get("qualification_source_commit") != qualification_source_commit'
        in timing
    )
    assert 'binding.get("patch_source_sha256") != patch_source_sha256' in timing

    assert (
        "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT=${FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT:-}"
        in launcher
    )
    assert (
        "_fr13_cutlass_streamk_source_commit=$FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_SOURCE_COMMIT"
        in launcher
    )
    assert (
        launcher.count(
            '--expected-source-commit "$_fr13_cutlass_streamk_source_commit"'
        )
        == 2
    )
    assert (
        "CUTLASS persistent M128 production requires a pinned qualification source commit"
        in launcher
    )
    assert (
        "CUTLASS B1 historical qualification is restricted to the pinned "
        "cooperative target" in launcher
    )
    assert (
        '--runtime-source-commit "$_fr13_cutlass_runtime_source_commit"'
        in launcher
    )


def test_b4_selector_reaches_eager_process_attestation() -> None:
    serve = (SCRIPTS / "fr13_bigdenom_swe_serve_variant.sh").read_text(encoding="utf-8")
    orchestrator = (SCRIPTS / "run_swe_bench_q36_a.py").read_text(encoding="utf-8")

    assert '"persistent_b4_m128_byte_ab",' in serve
    assert '"persistent_b4_m128_static_byte_ab",' in serve
    assert '"persistent_b4_m128"' in serve
    assert '"persistent_b4_m128_byte_ab"' in serve
    assert 'batch_gdn_byte_ab_text == "1"\n            or cutlass_wave in {' in serve
    assert '== "persistent_b4_m128_static_byte_ab" ]]; then' in serve
    assert "fr13-fixed32-eager-kernel-terminal-v1" in serve
    assert "fr13-fixed32-eager-kernel-traffic-audit-skip-v1" in serve
    assert "fixed32 eager kernel diagnostic: graph-census needles" in serve
    assert "_Fixed32EagerKernelDiagnosticTaskBracket" in orchestrator
    assert "or fixed32_cutlass_b4_diagnostic" in orchestrator


def test_b4_pass_accepts_320_comparisons_and_rejects_321(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _, _ = _fixture(tmp_path, monkeypatch)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload["comparisons"] = module.MAX_COMPARISONS
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()

    module.validate_live_result(live, live_sha256, candidate, patch_source)

    payload["comparisons"] = module.MAX_COMPARISONS + 1
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    with pytest.raises(module.QualificationError, match="comparison count"):
        module.validate_live_result(live, live_sha256, candidate, patch_source)
