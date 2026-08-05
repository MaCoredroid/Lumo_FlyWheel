from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
CREDENTIAL = REPO / "scripts" / "fr13_dfwd_k64_m1_r64_u8_production_credential.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_dfwd_k64_m1_r64_u8_timing.sh"
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module():
    return _load(CREDENTIAL, "fr13_u8_production_credential")


def _patcher():
    return _load(PATCHER, "fr13_u8_production_patcher")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input_files(tmp_path: Path) -> dict[str, Path]:
    names = {
        "candidate_so_sha256": "candidate.so",
        "candidate_source_sha256": "candidate.cu",
        "build_attestation_sha256": "build.json",
        "patch_source_sha256": "patch.py",
        "runner_sha256": "qualification.sh",
        "subset_sha256": "subset.json",
        "vocab_blocks_sha256": "blocks.json",
        "fa2_sha256": "fa2.so",
    }
    paths = {}
    for index, (key, name) in enumerate(names.items(), start=1):
        path = tmp_path / name
        path.write_bytes((f"{key}:{index}\n").encode("ascii"))
        paths[key] = path
    return paths


def _pin_temp_inputs(module, monkeypatch, paths: dict[str, Path]) -> dict[str, object]:
    gate = module.gate
    monkeypatch.setattr(
        gate, "EXPECTED_SO_BYTES", paths["candidate_so_sha256"].stat().st_size
    )
    monkeypatch.setattr(gate, "EXPECTED_SO_SHA256", _sha(paths["candidate_so_sha256"]))
    monkeypatch.setattr(
        gate, "EXPECTED_SOURCE_SHA256", _sha(paths["candidate_source_sha256"])
    )
    monkeypatch.setattr(
        gate,
        "EXPECTED_BUILD_ATTESTATION_SHA256",
        _sha(paths["build_attestation_sha256"]),
    )
    monkeypatch.setattr(gate, "EXPECTED_SUBSET_SHA256", _sha(paths["subset_sha256"]))
    monkeypatch.setattr(
        gate, "EXPECTED_VOCAB_BLOCKS_SHA256", _sha(paths["vocab_blocks_sha256"])
    )
    monkeypatch.setattr(gate, "EXPECTED_FA2_SHA256", _sha(paths["fa2_sha256"]))
    return {
        **{key: _sha(path) for key, path in paths.items()},
        "candidate_so_bytes": paths["candidate_so_sha256"].stat().st_size,
    }


def _gate_result(module, inputs: dict[str, object]) -> dict[str, object]:
    events = 3
    return {
        "schema": module.gate.GATE_SCHEMA,
        "status": "PASS",
        "source_commit": "b" * 40,
        "candidate": module.gate.EXPECTED_CANDIDATE,
        "geometry": module.gate.EXPECTED_GEOMETRY,
        "topology": module.gate.EXPECTED_TOPOLOGY,
        "inputs": inputs,
        "live_result_sha256": "1" * 64,
        "completed_events": events,
        "root_forward_steps": list(range(events)),
        "captured_mtp_depths": [1, 2, 3, 4],
        "comparison_scope": module.gate.COMPARISON_SCOPE,
        "worker_env_bridge": {},
        "per_depth_full_logit_comparisons": {
            label: events for label in module.gate.DEPTH_LABELS
        },
        "compared_elements": events * 5 * 65536,
        "compared_bytes": events * 5 * 65536 * 2,
        "raw_bf16_mismatches": 0,
        "reference_always_served": True,
        "candidate_returned": False,
        "task_resolved": True,
        "events_sha256": "2" * 64,
        "final_flush_sha256": "3" * 64,
        "boundary_snapshot_sha256": "4" * 64,
        "chat_traffic_audit_sha256": "5" * 64,
        "performance_measurement": False,
        "timing_eligible": False,
        "production_eligible": False,
    }


def _write_canonical(module, path: Path, payload: dict[str, object]) -> str:
    path.write_bytes(module.terminal.canonical_bytes(payload) + b"\n")
    return _sha(path)


def test_credential_requires_exact_shadow_pass_and_all_real_depths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    paths = _input_files(tmp_path)
    inputs = _pin_temp_inputs(module, monkeypatch, paths)
    gate_result = _gate_result(module, inputs)
    credential = module._credential_from_gate(gate_result)
    assert credential["captured_mtp_depths"] == [1, 2, 3, 4]
    assert credential["production_default_enabled"] is False
    assert credential["performance_claim"] is False
    assert credential["serve_policy"] == "candidate_only_after_internal_attestation"

    partial = dict(gate_result)
    partial["captured_mtp_depths"] = [1, 2, 3]
    with pytest.raises(ValueError, match="exact shadow PASS"):
        module._credential_from_gate(partial)
    mismatch = dict(gate_result)
    mismatch["raw_bf16_mismatches"] = 1
    with pytest.raises(ValueError, match="exact shadow PASS"):
        module._credential_from_gate(mismatch)
    returned = dict(gate_result)
    returned["candidate_returned"] = True
    with pytest.raises(ValueError, match="exact shadow PASS"):
        module._credential_from_gate(returned)


def test_credential_verifier_binds_canonical_payload_inputs_and_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    paths = _input_files(tmp_path)
    inputs = _pin_temp_inputs(module, monkeypatch, paths)
    payload = module._credential_from_gate(_gate_result(module, inputs))
    credential_path = tmp_path / "credential.json"
    digest = _write_canonical(module, credential_path, payload)
    result = module.validate_credential(
        credential_path=credential_path,
        expected_credential_sha256=digest,
        candidate_so=paths["candidate_so_sha256"],
        candidate_source=paths["candidate_source_sha256"],
        build_attestation=paths["build_attestation_sha256"],
        patch_source=paths["patch_source_sha256"],
        qualification_runner=paths["runner_sha256"],
        subset=paths["subset_sha256"],
        vocab_blocks=paths["vocab_blocks_sha256"],
        fa2_so=paths["fa2_sha256"],
        expected_source_commit="b" * 40,
    )
    assert result["status"] == "PASS"
    assert result["graph_contract"] == module.GRAPH_CONTRACT

    tampered = json.loads(json.dumps(payload))
    tampered["captured_mtp_depths"] = [1, 2, 3]
    tampered_digest = _write_canonical(module, credential_path, tampered)
    with pytest.raises(ValueError, match="provenance drifted"):
        module.validate_credential(
            credential_path=credential_path,
            expected_credential_sha256=tampered_digest,
            candidate_so=paths["candidate_so_sha256"],
            candidate_source=paths["candidate_source_sha256"],
            build_attestation=paths["build_attestation_sha256"],
            patch_source=paths["patch_source_sha256"],
            qualification_runner=paths["runner_sha256"],
            subset=paths["subset_sha256"],
            vocab_blocks=paths["vocab_blocks_sha256"],
            fa2_so=paths["fa2_sha256"],
            expected_source_commit="b" * 40,
        )

    forged_type = json.loads(json.dumps(payload))
    forged_type["graph_contract"]["root_calls"] = True
    forged_digest = _write_canonical(module, credential_path, forged_type)
    with pytest.raises(ValueError, match="provenance drifted"):
        module.validate_credential(
            credential_path=credential_path,
            expected_credential_sha256=forged_digest,
            candidate_so=paths["candidate_so_sha256"],
            candidate_source=paths["candidate_source_sha256"],
            build_attestation=paths["build_attestation_sha256"],
            patch_source=paths["patch_source_sha256"],
            qualification_runner=paths["runner_sha256"],
            subset=paths["subset_sha256"],
            vocab_blocks=paths["vocab_blocks_sha256"],
            fa2_so=paths["fa2_sha256"],
            expected_source_commit="b" * 40,
        )


def test_production_worker_bridge_requires_validator_attestation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _patcher()
    payload = {key: "" for key in patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS}
    payload.update(
        {
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB": "0",
            "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION": "1",
            "FR13_DRAFT_HEAD_M1_R64_U8_SO": "/tmp/fr13_bf16_k64_m1_r64_u8.abi3.so",
            "FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID": "astropy__astropy-12907",
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON": "/logs/fr13_dfwd_k64_m1_r64_u8.live.json",
            "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR": "/logs/fr13_dfwd_k64_m1_r64_u8.production_credential.json",
            "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_PASS_SIDECAR_SHA256": "c" * 64,
            "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION_ENGAGEMENT_JSON": "/logs/fr13_dfwd_k64_m1_r64_u8.production_engagement.json",
            "FR13_DRAFT_VOCAB_BLOCKS": "/workspace/scripts/fr13_dvk_subset_blocks.json",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_ROOT": "1",
            "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT": "b" * 40,
        }
    )
    for key in payload:
        if key.endswith("SHA256") and not key.endswith(
            "PRODUCTION_PASS_SIDECAR_SHA256"
        ):
            payload[key] = "a" * 64
    for key, value in payload.items():
        monkeypatch.setenv(key, value)
    sidecar = tmp_path / "worker.json"
    with pytest.raises(RuntimeError, match="inputs drifted"):
        patcher._fr13_write_draft_head_u8_worker_env_sidecar(sidecar)

    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED", "1")
    record = patcher._fr13_write_draft_head_u8_worker_env_sidecar(sidecar)
    assert record is not None
    assert sidecar.stat().st_mode & 0o777 == 0o400
    for key in payload:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.delenv(
        "FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED", raising=False
    )

    source = patcher._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    body = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_fr13_draft_head_u8_worker_env_bridge"
        )
    ]
    namespace = {"_FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED": True}
    exec(compile(ast.Module(body=body, type_ignores=[]), "<bridge>", "exec"), namespace)
    bridge = namespace["_fr13_draft_head_u8_worker_env_bridge"](str(sidecar))
    assert bridge["hydrated_keys"] == list(patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS)
    assert os.environ["FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION"] == "1"
    assert os.environ["FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED"] == "1"
    for key in patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_u8_production_selector_tracks_exact_root_and_four_capture_calls() -> None:
    source = _patcher()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_fixed32_drafter_u8_head_selection"
    )
    namespace = {
        "_FR13_FIXED32_MODE": "hydra27_fixed32",
        "_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT": {
            "batch_size": 1,
            "mode": "hydra27_fixed32",
            "measured": True,
            "mtp_execution_basis": "unbound",
            "mtp_forward_calls": 0,
            "mtp_forward_rows": 0,
        },
        "_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT": None,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "<selector>", "exec"),
        namespace,
    )
    select = namespace["_fr13_fixed32_drafter_u8_head_selection"]
    assert select(1) is False

    context = {
        "capturing": True,
        "batch_size": 1,
        "mode": "hydra27_fixed32",
        "mtp_forward_calls": 1,
        "mtp_forward_rows": 1,
        "draft_head_u8_calls": 0,
        "draft_head_u8_rows": 0,
    }
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"] = context
    for depth in range(1, 5):
        context["mtp_forward_calls"] = depth
        context["mtp_forward_rows"] = depth
        assert select(1) is True
    assert context["draft_head_u8_calls"] == 4
    context["mtp_forward_calls"] = 5
    context["mtp_forward_rows"] = 5
    with pytest.raises(RuntimeError, match="left capture lifecycle"):
        select(1)


def test_engagement_validator_requires_graph_identity_and_zero_fallbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    monkeypatch.setattr(module.gate, "EXPECTED_SO_SHA256", "1" * 64)
    monkeypatch.setattr(module.gate, "EXPECTED_SOURCE_SHA256", "2" * 64)
    payload = {
        "schema": module.ENGAGEMENT_SCHEMA,
        "status": "ENGAGED",
        "source_commit": "b" * 40,
        "candidate_so_sha256": "1" * 64,
        "candidate_source_sha256": "2" * 64,
        "production_credential_sha256": "3" * 64,
        "geometry": module.gate.EXPECTED_GEOMETRY,
        "qualification_candidate": module.gate.EXPECTED_CANDIDATE,
        "selector": module.SELECTOR,
        "selected_root_calls": 1,
        "captured_loop_calls": 4,
        "fallback_calls": 0,
        "drafter_graph_id": 41,
        "drafter_graph_signature": module.GRAPH_SIGNATURE,
        "observed_measured_replays_at_least": 1,
        "capture_origin": "unmeasured",
        "execution_basis": "cudagraph_replay",
        "forward_step_index": 0,
        "runtime_mode": "FULL",
        "candidate_served": True,
        "incumbent_head_calls": 0,
        "steady_state_synchronizations": 0,
        "performance_claim": False,
    }
    path = tmp_path / "engagement.json"
    _write_canonical(module, path, payload)
    assert (
        module.validate_engagement(
            engagement_path=path,
            expected_credential_sha256="3" * 64,
            expected_source_commit="b" * 40,
        )
        == payload
    )
    payload["fallback_calls"] = 1
    _write_canonical(module, path, payload)
    with pytest.raises(ValueError, match="engagement drifted"):
        module.validate_engagement(
            engagement_path=path,
            expected_credential_sha256="3" * 64,
            expected_source_commit="b" * 40,
        )

    payload["fallback_calls"] = 0
    payload["selected_root_calls"] = True
    _write_canonical(module, path, payload)
    with pytest.raises(ValueError, match="engagement drifted"):
        module.validate_engagement(
            engagement_path=path,
            expected_credential_sha256="3" * 64,
            expected_source_commit="b" * 40,
        )


def test_launcher_and_runner_keep_incumbent_default_and_fixed_task_sets() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    assert 'FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION", "0"' in patcher
    assert "candidate_only_after_internal_attestation" in CREDENTIAL.read_text(
        encoding="utf-8"
    )
    assert "return _fr13_dh_u8_reference" in patcher
    assert "return self._fr13_dh_u8_output" in patcher
    assert "_fr13_dh_u8_note_production_replay(" in patcher
    assert (
        "FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION=${FR13_DRAFT_HEAD_M1_R64_U8_PRODUCTION:-0}"
        in launcher
    )
    assert "production_credential.py issue" in launcher
    assert "production_credential.py verify" in launcher
    clean_guard = '[[ -z "$(git status --porcelain=v1 --untracked-files=no)" ]]'
    assert clean_guard in launcher
    assert launcher.index(clean_guard) < launcher.index("production_credential.py issue")
    assert launcher.index(
        "export FR13_DRAFT_HEAD_M1_R64_U8_INTERNAL_PRODUCTION_ATTESTED=1"
    ) < launcher.index("python3 /workspace/scripts/fr10_phase4_patch_vllm_tree_gdn.py")
    assert "TASK_SET=${TASK_SET:-exact4}" in runner
    assert "TASK_SET must be exactly exact4 or exact16" in runner
    assert "subset_b4_four.json" in runner
    assert "subset_b4_sixteen.json" in runner
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in runner
    assert "production_default_enabled=0" in runner
    assert '"performance_claim": False' in runner
    assert "fr13_dfwd_k64_m1_r64_u8_production_credential.py" in manifest
    assert "fr13_run_b1_dfwd_k64_m1_r64_u8_timing.sh" in manifest
