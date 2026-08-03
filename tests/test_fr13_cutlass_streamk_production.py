from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"


def _load(name: str, filename: str):
    sys.path.insert(0, str(SCRIPTS))
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _qualified_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = _load("fr13_cutlass_streamk_pass_test", "fr13_cutlass_streamk_pass.py")
    candidate_bytes = b"candidate\n"
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidate = tmp_path / "candidate.so"
    candidate.write_bytes(candidate_bytes)
    patch_bytes = b"patch source\n"
    patch_sha256 = hashlib.sha256(patch_bytes).hexdigest()
    patch_source = tmp_path / "patch.py"
    patch_source.write_bytes(patch_bytes)
    monkeypatch.setattr(module.binary, "CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module.binary, "CANDIDATE_SHA256", candidate_sha256)
    monkeypatch.setattr(module, "PATCH_SOURCE_SHA256", patch_sha256)
    live = {
        "schema": module.LIVE_SCHEMA,
        "status": "pass",
        "run_classification": "one_real_swe_verified_b1_byte_diagnostic",
        "acceptance_valid": False,
        "task_count": 1,
        "task_ids": list(module.EXPECTED_TASK_IDS),
        "task_marker": module.EXPECTED_TASK_MARKER,
        "draft_vocab_k": 0,
        "draft_vocab_root": 0,
        "mandatory_weight_bytes": module.floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES,
        "mandatory_weight_floor_ms": (
            module.floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
        ),
        "one_sided_u95_cap_ms": module.floor.FULL_VOCAB_SLO_CAP_MS,
        "comparator_timing_eligible": False,
        "batch_size": 1,
        "concurrency": 1,
        "fixed_rows": 32,
        "candidate": "streamk_coop128",
        "diagnostic_selector": "streamk_coop128_byte_ab",
        "served_result": "stock",
        "production_enabled": False,
        "comparisons": 5,
        "observed_m_values": [32],
        "observed_projection_nk": [
            list(shape) for shape in module.EXPECTED_PROJECTION_NK
        ],
        "mismatching_comparisons": 0,
        "differing_bytes": 0,
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


def test_live_pass_issues_and_verifies_source_binary_bound_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _qualified_fixture(
        tmp_path, monkeypatch
    )
    sidecar = tmp_path / "sidecar.json"

    issued = module.issue_sidecar(live, live_sha256, candidate, sidecar, patch_source)
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(sidecar, sidecar_sha256, candidate, patch_source)

    assert issued == verified
    assert issued["status"] == "QUALIFIED"
    assert (
        issued["candidate_sha256"] == hashlib.sha256(candidate.read_bytes()).hexdigest()
    )
    assert (
        issued["patch_source_sha256"]
        == hashlib.sha256(patch_source.read_bytes()).hexdigest()
    )
    assert issued["live_result_sha256"] == live_sha256


def test_wide256_live_pass_is_bound_to_wide_binary_and_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, _ = _qualified_fixture(
        tmp_path, monkeypatch
    )
    candidate_bytes = candidate.read_bytes()
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SIZE", len(candidate_bytes))
    monkeypatch.setattr(module.binary, "WIDE256_CANDIDATE_SHA256", candidate_sha256)
    payload = json.loads(live.read_text(encoding="ascii"))
    payload.update(
        {
            "schema": module.WIDE256_LIVE_SCHEMA,
            "candidate": "streamk_force_wide256",
            "candidate_family": "streamk_force_wide256",
            "diagnostic_selector": "streamk_force_wide256_byte_ab",
        }
    )
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    live_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()
    sidecar = tmp_path / "wide256-sidecar.json"

    issued = module.issue_sidecar(
        live,
        live_sha256,
        candidate,
        sidecar,
        patch_source,
        candidate_selector="streamk_force_wide256",
    )
    sidecar_sha256 = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    verified = module.verify_sidecar(
        sidecar,
        sidecar_sha256,
        candidate,
        patch_source,
        candidate_selector="streamk_force_wide256",
    )

    assert verified == issued
    assert issued["candidate_selector"] == "streamk_force_wide256"
    assert issued["diagnostic_selector"] == "streamk_force_wide256_byte_ab"
    assert issued["candidate_family"] == "streamk_force_wide256"


def test_live_pass_rejects_mismatch_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _qualified_fixture(
        tmp_path, monkeypatch
    )
    payload = json.loads(live.read_text(encoding="ascii"))
    payload["differing_bytes"] = 1
    live.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    changed_sha256 = hashlib.sha256(live.read_bytes()).hexdigest()

    with pytest.raises(module.QualificationError, match="differing_bytes mismatch"):
        module.validate_live_result(live, changed_sha256, candidate, patch_source)
    alias = tmp_path / "live-link.json"
    alias.symlink_to(live)
    with pytest.raises(module.QualificationError, match="non-symlink"):
        module.validate_live_result(alias, live_sha256, candidate, patch_source)


def test_production_attestation_requires_exact_selector_and_sidecar_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load(
        "fr13_cutlass_streamk_pass_attestation", "fr13_cutlass_streamk_pass.py"
    )
    sidecar_sha256 = "a" * 64
    payload = {
        "schema": module.ATTESTATION_SCHEMA,
        "selector": "streamk_coop128",
        "source": {
            "path": str(module.binary.CONTAINER_SOURCE),
            "bytes": module.binary.CANDIDATE_SIZE,
            "sha256": module.binary.CANDIDATE_SHA256,
            "regular": True,
            "symlink": False,
        },
        "destination": {
            "path": str(module.binary.CONTAINER_DESTINATION),
            "bytes": module.binary.CANDIDATE_SIZE,
            "sha256": module.binary.CANDIDATE_SHA256,
            "regular": True,
            "symlink": False,
        },
        "installed_mode": "0555",
        "production_enabled": True,
        "qualification": {
            "sidecar_sha256": sidecar_sha256,
            "live_result_sha256": "b" * 64,
            "candidate_sha256": module.binary.CANDIDATE_SHA256,
            "patch_source_sha256": module.PATCH_SOURCE_SHA256,
            "qualification_source_commit": "c" * 40,
            "qualification_task_marker": module.EXPECTED_TASK_MARKER,
            "real_task_arm_sha256": "d" * 64,
            "container_env_sha256": "e" * 64,
            "qualified_draft_vocab_root": 0,
            "qualified_draft_vocab_k": 0,
            "mandatory_weight_bytes": (
                module.floor.FULL_VOCAB_MANDATORY_WEIGHT_BYTES
            ),
            "mandatory_weight_floor_ms": (
                module.floor.FULL_VOCAB_MANDATORY_WEIGHT_FLOOR_MS
            ),
            "one_sided_u95_cap_ms": module.floor.FULL_VOCAB_SLO_CAP_MS,
        },
    }
    attestation = tmp_path / "attestation.json"
    attestation.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")

    result = module.validate_production_attestation(attestation, sidecar_sha256)
    assert result["status"] == "BOUND"
    payload["qualification"]["qualification_task_ids"] = [
        "astropy__astropy-13236"
    ]
    attestation.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(module.QualificationError, match="qualification_task_ids"):
        module.validate_production_attestation(attestation, sidecar_sha256)
    payload["qualification"].pop("qualification_task_ids")
    payload["selector"] = "stock"
    attestation.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(module.QualificationError, match="selector mismatch"):
        module.validate_production_attestation(attestation, sidecar_sha256)


def test_live_pass_rejects_stale_source_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, candidate, patch_source, live, live_sha256 = _qualified_fixture(
        tmp_path, monkeypatch
    )

    with pytest.raises(module.QualificationError, match="source commit is stale"):
        module.validate_live_result(
            live,
            live_sha256,
            candidate,
            patch_source,
            expected_source_commit="a" * 40,
        )


def _measure(
    module, task_ids=None, qualification_profile="full_vocab"
) -> dict[str, object]:
    if task_ids is None:
        task_ids = module.EXPECTED_TASK_IDS
    profile = module.qualification.QUALIFICATION_PROFILES[qualification_profile]
    floor_ms = profile["mandatory_weight_floor_ms"]
    return {
        "schema": module.MEASURE_SCHEMA,
        "regime": "deployment",
        "instrument": "OFF",
        "batch_size": 1,
        "n_tasks": len(task_ids),
        "task_instance_ids": list(task_ids),
        "draft_vocab_k": profile["draft_vocab_k"],
        "draft_vocab_root": profile["draft_vocab_root"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "floor_is_full_step_hardware_floor": False,
        "measured_tps_fullstep_wall": 25.0,
        "step_wall_ms": 230.0,
        "accept_per_event": 4.75,
        "committed_per_event": 5.75,
        "wall_steps_measured": 100.0,
        "events_per_step": 1.0,
        "s_per_fwd_gpu": 0.16,
        "drafter_gpu_ms_per_step": 37.0,
        "committer_gpu_ms_per_step": 20.0,
        "weight_floor_ms": floor_ms,
        "floor_ms": floor_ms,
        "floor_ratio": 230.0 / floor_ms,
    }


@pytest.mark.parametrize(
    (
        "candidate_selector",
        "task_set",
        "qualification_profile",
        "diagnostic_task_profile",
    ),
    (
        ("streamk_coop128", "exact4", "full_vocab", "astropy12907"),
        ("streamk_force_wide256", "one", "full_vocab", "astropy12907"),
        ("identity_onen_b1", "exact4", "k64_root", "astropy12907"),
        (
            "identity_onen_n5120_single_b1",
            "exact4",
            "k64_root",
            "astropy12907",
        ),
        (
            "identity_onen_n5120_fullgrid_b1",
            "exact4",
            "k64_root",
            "astropy13236",
        ),
    ),
)
def test_timing_reducer_requires_pinned_task_set_profile_and_current_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    candidate_selector: str,
    task_set: str,
    qualification_profile: str,
    diagnostic_task_profile: str,
) -> None:
    module = _load("fr13_cutlass_streamk_timing_test", "fr13_cutlass_streamk_timing.py")
    profile = module.qualification.QUALIFICATION_PROFILES[qualification_profile]
    diagnostic_profile = module.qualification.DIAGNOSTIC_TASK_PROFILES[
        diagnostic_task_profile
    ]
    candidate_bytes = b"candidate\n"
    candidate_sha256 = hashlib.sha256(candidate_bytes).hexdigest()
    candidate_so = tmp_path / "candidate.so"
    candidate_so.write_bytes(candidate_bytes)
    if candidate_selector == "streamk_coop128":
        monkeypatch.setattr(module.binary, "CANDIDATE_SIZE", len(candidate_bytes))
        monkeypatch.setattr(module.binary, "CANDIDATE_SHA256", candidate_sha256)
        diagnostic_selector = "streamk_coop128_byte_ab"
    elif candidate_selector == "streamk_force_wide256":
        monkeypatch.setattr(
            module.binary, "WIDE256_CANDIDATE_SIZE", len(candidate_bytes)
        )
        monkeypatch.setattr(
            module.binary, "WIDE256_CANDIDATE_SHA256", candidate_sha256
        )
        diagnostic_selector = "streamk_force_wide256_byte_ab"
    elif candidate_selector == "identity_onen_b1":
        monkeypatch.setattr(
            module.binary, "IDENTITY_ONEN_B1_CANDIDATE_SIZE", len(candidate_bytes)
        )
        monkeypatch.setattr(
            module.binary, "IDENTITY_ONEN_B1_CANDIDATE_SHA256", candidate_sha256
        )
        diagnostic_selector = "identity_onen_b1_byte_ab"
    elif candidate_selector == "identity_onen_n5120_single_b1":
        monkeypatch.setattr(
            module.binary,
            "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SIZE",
            len(candidate_bytes),
        )
        monkeypatch.setattr(
            module.binary,
            "IDENTITY_ONEN_N5120_SINGLE_B1_CANDIDATE_SHA256",
            candidate_sha256,
        )
        diagnostic_selector = "identity_onen_n5120_single_b1_byte_ab"
    else:
        monkeypatch.setattr(
            module.binary,
            "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SIZE",
            len(candidate_bytes),
        )
        monkeypatch.setattr(
            module.binary,
            "IDENTITY_ONEN_N5120_FULLGRID_B1_CANDIDATE_SHA256",
            candidate_sha256,
        )
        diagnostic_selector = "identity_onen_n5120_fullgrid_b1_byte_ab"

    if qualification_profile == "k64_root":
        monkeypatch.setattr(
            module.qualification,
            "validate_source_commit_binding",
            lambda source_commit, *_args, **_kwargs: {
                "source_commit": source_commit
            },
        )
    monkeypatch.setattr(
        module.qualification,
        "_source_contract",
        lambda _candidate_selector: {
            "patch_source_sha256": "e" * 64,
            "patched_dispatch_sha256": "f" * 64,
        },
    )
    qrow16_bytes = b"qrow16 candidate\n"
    qrow16_sha256 = hashlib.sha256(qrow16_bytes).hexdigest()
    qrow16_so = tmp_path / "qrow16.so"
    qrow16_so.write_bytes(qrow16_bytes)
    monkeypatch.setattr(module, "QROW16_BYTES", len(qrow16_bytes))
    monkeypatch.setattr(module, "QROW16_SHA256", qrow16_sha256)
    monkeypatch.setattr(module, "QROW16_LIVE_RESULT_SHA256", "8" * 64)
    monkeypatch.setattr(
        module.qrow,
        "verify_sidecar",
        lambda **_kwargs: {"live_result_sha256": "8" * 64},
    )
    task_ids = tuple(module.TASK_SET_CONTRACTS[task_set]["task_ids"])
    subset = tmp_path / "subset.json"
    subset.write_text(
        json.dumps({"instance_ids": list(task_ids)}) + "\n",
        encoding="ascii",
    )
    monkeypatch.setitem(
        module.TASK_SET_CONTRACTS[task_set],
        "subset_sha256",
        hashlib.sha256(subset.read_bytes()).hexdigest(),
    )
    stock = tmp_path / "stock.json"
    candidate = tmp_path / "candidate.json"
    stock.write_text(
        json.dumps(_measure(module, task_ids, qualification_profile)) + "\n",
        encoding="ascii",
    )
    candidate_payload = _measure(module, task_ids, qualification_profile)
    candidate_payload["step_wall_ms"] = 220.0
    candidate_payload["measured_tps_fullstep_wall"] = 26.0
    candidate_payload["floor_ratio"] = (
        220.0
        / module.qualification.QUALIFICATION_PROFILES[qualification_profile][
            "mandatory_weight_floor_ms"
        ]
    )
    candidate.write_text(json.dumps(candidate_payload) + "\n", encoding="ascii")
    stock_env = tmp_path / "stock.env"
    candidate_env = tmp_path / "candidate.env"
    for path, selector, production in (
        (stock_env, "stock", 0),
        (candidate_env, candidate_selector, 1),
    ):
        path.write_text(
            "\n".join(
                (
                        (
                            "FR13_DRAFT_VOCAB_ROOT="
                            f"{profile['draft_vocab_root']}"
                        ),
                        (
                            "FR13_DRAFT_VOCAB_K="
                            f"{profile['draft_vocab_k']}"
                        ),
                        *module.COMMON_EXPECTED_ENV,
                        "FR13_FIXED32_MODE=hydra27_fixed32",
                        "FR13_FA2_QROW16_LIVE_PAGED_AB=0",
                        f"FR13_FIXED32_CUTLASS_WAVE={selector}",
                        f"FR13_FIXED32_CUTLASS_WAVE_PRODUCTION={production}",
                        (
                            "FR13_FIXED32_CUTLASS_WAVE_QUALIFICATION_PROFILE="
                            f"{qualification_profile}"
                        ),
                        (
                            "FR13_FIXED32_CUTLASS_WAVE_DIAGNOSTIC_TASK_PROFILE="
                            f"{diagnostic_task_profile}"
                        ),
                        *(
                            (
                                "FR13_DRAFT_VOCAB_BLOCKS="
                                + module.qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH,
                            )
                            if qualification_profile == "k64_root"
                            else ()
                        ),
                    )
            )
            + "\n",
            encoding="ascii",
        )
    stock_qrow16_sidecar = tmp_path / "stock-qrow16-sidecar.json"
    candidate_qrow16_sidecar = tmp_path / "candidate-qrow16-sidecar.json"
    for path in (stock_qrow16_sidecar, candidate_qrow16_sidecar):
        path.write_text("{}\n", encoding="ascii")
    qrow16_sidecar_sha256 = hashlib.sha256(
        stock_qrow16_sidecar.read_bytes()
    ).hexdigest()

    def qrow16_capture(path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "schema": "fr13.fixed32.fa2_qrow16_production_capture.v1",
                    "status": "ENGAGED",
                    "graph_id": 17,
                    "graph_signature": "7" * 64,
                    "runtime_mode": "FULL",
                    "batch_size": 1,
                    "layers": [f"model.layers.{index}" for index in range(16)],
                    "layer_count": 16,
                    "candidate_so_sha256": qrow16_sha256,
                    "pass_sidecar_sha256": qrow16_sidecar_sha256,
                    "dispatch": "qrow16 exact geometry; no fallback",
                }
            )
            + "\n",
            encoding="ascii",
        )

    stock_qrow16_capture = tmp_path / "stock-qrow16-capture.json"
    candidate_qrow16_capture = tmp_path / "candidate-qrow16-capture.json"
    qrow16_capture(stock_qrow16_capture)
    qrow16_capture(candidate_qrow16_capture)
    source_commit = "c" * 40
    binding = tmp_path / "binding.json"
    binding_payload = {
        "schema": profile["binding_schema"],
        "status": "BOUND",
        "selector": candidate_selector,
        "diagnostic_selector": diagnostic_selector,
        "candidate_family": candidate_selector,
        "candidate_sha256": candidate_sha256,
        "candidate_bytes": len(candidate_bytes),
        "patch_source_sha256": "e" * 64,
        "production_sidecar_sha256": "f" * 64,
        "live_result_sha256": "a" * 64,
        "binary_attestation_sha256": "b" * 64,
        "real_task_arm_sha256": "d" * 64,
        "container_env_sha256": "9" * 64,
        "qualification_source_commit": source_commit,
        "qualification_task_profile": diagnostic_task_profile,
        "qualification_task_ids": list(diagnostic_profile["task_ids"]),
        "qualification_task_marker": diagnostic_profile["task_marker"],
        "qualified_draft_vocab_root": profile["draft_vocab_root"],
        "qualified_draft_vocab_k": profile["draft_vocab_k"],
        "mandatory_weight_bytes": profile["mandatory_weight_bytes"],
        "mandatory_weight_floor_ms": profile["mandatory_weight_floor_ms"],
        "one_sided_u95_cap_ms": profile["one_sided_u95_cap_ms"],
        "production_default_enabled": False,
    }
    if qualification_profile == "k64_root":
        binding_payload.update(
            {
                "qualification_profile": qualification_profile,
                "qualified_draft_vocab_blocks": (
                    module.qualification.DRAFT_VOCAB_BLOCKS_CONTAINER_PATH
                ),
                "qualified_draft_vocab_blocks_sha256": (
                    module.qualification.DRAFT_VOCAB_BLOCKS_SHA256
                ),
                "qualified_comparison_call_limit": module.qualification.MAX_COMPARISONS,
                "qualification_source_identity": {"source_commit": source_commit},
            }
        )
    binding.write_text(
        json.dumps(binding_payload) + "\n",
        encoding="ascii",
    )

    result = module.reduce_pair(
        subset,
        stock,
        candidate,
        stock_env,
        candidate_env,
        stock_qrow16_sidecar,
        candidate_qrow16_sidecar,
        stock_qrow16_capture,
        candidate_qrow16_capture,
        qrow16_so,
        binding,
        candidate_so,
        source_commit,
        candidate_selector=candidate_selector,
        qualification_profile=qualification_profile,
        diagnostic_task_profile=diagnostic_task_profile,
        task_set=task_set,
    )

    assert result["run_classification"] == module.TASK_SET_CONTRACTS[task_set][
        "run_classification"
    ]
    assert result["topology"] == "hydra27_fixed32"
    assert result["lineage"] == "successor_to_legacy_hydra23_not_same_topology"
    assert result["only_arm_delta"] == f"CUTLASS stock to {candidate_selector}"
    assert result["common_kernel_stack"]["identical_in_both_arms"] is True
    assert (
        result["common_kernel_stack"]["stock_arm"]["candidate_so_sha256"]
        == qrow16_sha256
    )
    assert result["task_count"] == len(task_ids)
    assert result["qualification_profile"] == qualification_profile
    assert result["qualification_task_profile"] == diagnostic_task_profile
    assert result["draft_vocab_k"] == profile["draft_vocab_k"]
    assert result["mandatory_weight_bytes"] == profile["mandatory_weight_bytes"]
    assert result["mandatory_weight_floor_ms"] == profile["mandatory_weight_floor_ms"]
    assert result["one_sided_u95_cap_ms"] == profile["one_sided_u95_cap_ms"]
    assert result["comparator_gate_timing_eligible"] is False
    assert result["timing_eligible"] is (task_set == "exact4")
    assert result["floor_acceptance_eligible"] is False
    assert result["candidate_to_stock_full_wall_tps_ratio"] == pytest.approx(
        26.0 / 25.0
    )

    drifted_qrow16 = json.loads(candidate_qrow16_capture.read_text(encoding="ascii"))
    drifted_qrow16["graph_signature"] = "6" * 64
    candidate_qrow16_capture.write_text(
        json.dumps(drifted_qrow16) + "\n", encoding="ascii"
    )
    with pytest.raises(module.TimingError, match="graph_signature differs"):
        module.reduce_pair(
            subset,
            stock,
            candidate,
            stock_env,
            candidate_env,
            stock_qrow16_sidecar,
            candidate_qrow16_sidecar,
            stock_qrow16_capture,
            candidate_qrow16_capture,
            qrow16_so,
            binding,
            candidate_so,
            source_commit,
            candidate_selector=candidate_selector,
            qualification_profile=qualification_profile,
            diagnostic_task_profile=diagnostic_task_profile,
            task_set=task_set,
        )
    qrow16_capture(candidate_qrow16_capture)

    stale = json.loads(binding.read_text(encoding="ascii"))
    stale["qualification_source_commit"] = "0" * 40
    binding.write_text(json.dumps(stale) + "\n", encoding="ascii")
    with pytest.raises(module.TimingError, match="qualification_source_commit"):
        module.reduce_pair(
            subset,
            stock,
            candidate,
            stock_env,
            candidate_env,
            stock_qrow16_sidecar,
            candidate_qrow16_sidecar,
            stock_qrow16_capture,
            candidate_qrow16_capture,
            qrow16_so,
            binding,
            candidate_so,
            source_commit,
            candidate_selector=candidate_selector,
            qualification_profile=qualification_profile,
            diagnostic_task_profile=diagnostic_task_profile,
            task_set=task_set,
        )
