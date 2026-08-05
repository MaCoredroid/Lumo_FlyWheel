from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
GATE = REPO / "scripts" / "fr13_dfwd_k64_m1_r64_u8_gate.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
GENERIC_RUNNER = REPO / "scripts" / "fr13_run_b1_kernel_live_gate.sh"
LIVE_RUNNER = REPO / "scripts" / "fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh"
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
READINESS = (
    REPO
    / "results"
    / "fr13_fixed32_dfwd_k64_m1_r64_u8_shadow_ready_20260805"
    / "qualification_manifest.json"
)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _patcher():
    return _load(PATCHER, "fr13_u8_patcher")


def _gate():
    return _load(GATE, "fr13_u8_gate")


def _runtime_namespace(*function_names: str) -> dict[str, object]:
    source = _patcher()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    wanted = set(function_names)
    body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in body} == wanted
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<u8-runtime>", "exec"),
        namespace,
    )
    return namespace


def _worker_bridge_namespace() -> dict[str, object]:
    source = _patcher()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    tree = ast.parse(source)
    body = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name)
                and target.id == "_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS"
                for target in node.targets
            )
        ) or (
            isinstance(node, ast.FunctionDef)
            and node.name == "_fr13_draft_head_u8_worker_env_bridge"
        ):
            body.append(node)
    namespace: dict[str, object] = {
        "_FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED": True,
    }
    exec(
        compile(ast.Module(body=body, type_ignores=[]), "<u8-bridge>", "exec"),
        namespace,
    )
    return namespace


def _worker_payload(module) -> dict[str, str]:
    payload = {key: "" for key in module._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS}
    payload.update(
        {
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB": "1",
            "FR13_DRAFT_HEAD_M1_R64_U8_SO": (
                "/tmp/fr13_bf16_k64_m1_r64_u8.abi3.so"
            ),
            "FR13_DRAFT_HEAD_M1_R64_U8_INSTANCE_ID": (
                "astropy__astropy-12907"
            ),
            "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON": (
                "/logs/fr13_dfwd_k64_m1_r64_u8.live.json"
            ),
            "FR13_DRAFT_VOCAB_BLOCKS": (
                "/workspace/scripts/fr13_dvk_subset_blocks.json"
            ),
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_ROOT": "1",
            "FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT": "b" * 40,
        }
    )
    for key in payload:
        if key.endswith("SHA256"):
            payload[key] = "a" * 64
    return payload


def _identities(gate) -> dict[str, object]:
    return {
        "build_attestation_sha256": gate.EXPECTED_BUILD_ATTESTATION_SHA256,
        "candidate_so_bytes": gate.EXPECTED_SO_BYTES,
        "candidate_so_sha256": gate.EXPECTED_SO_SHA256,
        "candidate_source_sha256": gate.EXPECTED_SOURCE_SHA256,
        "fa2_sha256": gate.EXPECTED_FA2_SHA256,
        "instance_id": gate.EXPECTED_INSTANCE,
        "patch_source_sha256": "1" * 64,
        "runner_sha256": "2" * 64,
        "source_commit": "b" * 40,
        "subset_sha256": gate.EXPECTED_SUBSET_SHA256,
        "vocab_blocks_sha256": gate.EXPECTED_VOCAB_BLOCKS_SHA256,
    }


def _bridge(gate) -> dict[str, object]:
    return {
        "schema": "fr13.fixed32.dfwd_k64_m1_r64_u8_worker_env_bridge.v1",
        "sidecar": "/logs/fr13_draft_head_m1_r64_u8.worker_env.json",
        "sidecar_sha256": "3" * 64,
        "payload_sha256": "4" * 64,
        "hydrated_keys": list(gate.WORKER_ENV_KEYS),
    }


class _Counter:
    def __init__(self, values: list[int]) -> None:
        self.values = values

    def tolist(self) -> list[int]:
        return self.values


def test_worker_sidecar_restores_exact_curated_enginecore_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _patcher()
    assert len(patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS) == 16
    assert (
        "FR13_DRAFT_HEAD_PAD_ROWS"
        not in patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS
    )
    assert (
        "FR13_DRAFT_HEAD_PAD_ALL_BYTE_AB"
        not in patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS
    )
    payload = _worker_payload(patcher)
    for key, value in payload.items():
        monkeypatch.setenv(key, value)
    sidecar = tmp_path / "worker_env.json"
    record = patcher._fr13_write_draft_head_u8_worker_env_sidecar(sidecar)
    assert record is not None
    assert sidecar.stat().st_mode & 0o777 == 0o400
    assert record["payload"] == payload

    for key in payload:
        monkeypatch.delenv(key, raising=False)
    namespace = _worker_bridge_namespace()
    bridge = namespace["_fr13_draft_head_u8_worker_env_bridge"](str(sidecar))
    assert bridge["hydrated_keys"] == list(patcher._FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS)
    assert bridge["sidecar_sha256"] == hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert {key: os.environ[key] for key in payload} == payload
    for key in payload:
        monkeypatch.delenv(key, raising=False)


def test_worker_sidecar_fails_closed_on_tamper_missing_and_off_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    patcher = _patcher()
    payload = _worker_payload(patcher)
    for key, value in payload.items():
        monkeypatch.setenv(key, value)
    sidecar = tmp_path / "worker_env.json"
    patcher._fr13_write_draft_head_u8_worker_env_sidecar(sidecar)
    decoded = json.loads(sidecar.read_text(encoding="ascii"))
    decoded["payload"]["FR13_DRAFT_VOCAB_K"] = "32768"
    sidecar.chmod(0o600)
    sidecar.write_text(json.dumps(decoded), encoding="ascii")
    sidecar.chmod(0o400)
    for key in payload:
        monkeypatch.delenv(key, raising=False)
    namespace = _worker_bridge_namespace()
    bridge = namespace["_fr13_draft_head_u8_worker_env_bridge"]
    with pytest.raises(RuntimeError, match="digest/K64-root drifted"):
        bridge(str(sidecar))
    with pytest.raises(RuntimeError, match="missing"):
        bridge(str(tmp_path / "missing.json"))
    namespace["_FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED"] = False
    with pytest.raises(RuntimeError, match="leaked while off"):
        bridge(str(sidecar))

    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "0")
    patcher._fr13_write_draft_head_u8_worker_env_sidecar(sidecar)
    assert not sidecar.exists()


def test_depth_classifier_proves_root_and_each_real_mtp_depth() -> None:
    namespace = _runtime_namespace("_fr13_draft_head_u8_live_depth")
    namespace["_FR13_FIXED32_MODE"] = "hydra27_fixed32"
    namespace["_FR13_DRAFT_HEAD_U8_LIVE_STATE"] = {
        "root_forward_steps": [],
        "captured_depths": [],
    }
    namespace["_FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT"] = {
        "batch_size": 1,
        "mode": "hydra27_fixed32",
        "measured": True,
        "mtp_execution_basis": "unbound",
        "mtp_forward_calls": 0,
        "mtp_forward_rows": 0,
        "forward_step_index": 0,
    }
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"] = None
    classify = namespace["_fr13_draft_head_u8_live_depth"]
    assert classify(1) == 0

    context = {
        "capturing": True,
        "batch_size": 1,
        "mode": "hydra27_fixed32",
        "draft_head_u8_calls": 0,
        "draft_head_u8_rows": 0,
        "mtp_forward_calls": 1,
        "mtp_forward_rows": 1,
    }
    namespace["_FR13_FIXED32_DRAFTER_GRAPH_CAPTURE_CONTEXT"] = context
    for depth in range(1, 5):
        context["mtp_forward_calls"] = depth
        context["mtp_forward_rows"] = depth
        assert classify(1) == depth
    assert namespace["_FR13_DRAFT_HEAD_U8_LIVE_STATE"]["captured_depths"] == [
        1,
        2,
        3,
        4,
    ]

    context["mtp_forward_calls"] = 5
    context["mtp_forward_rows"] = 5
    with pytest.raises(RuntimeError, match="depth lifecycle drifted"):
        classify(1)
    namespace["_FR13_DRAFT_HEAD_U8_LIVE_STATE"]["captured_depths"] = []
    context.update(
        draft_head_u8_calls=1,
        draft_head_u8_rows=1,
        mtp_forward_calls=2,
        mtp_forward_rows=2,
    )
    with pytest.raises(RuntimeError, match="depth lifecycle drifted"):
        classify(1)


def test_finalizer_and_gate_require_every_depth_per_authenticated_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate = _gate()
    namespace = _runtime_namespace("_fr13_draft_head_u8_live_finalize")
    live_path = tmp_path / "live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_R64_U8_LIVE_JSON", str(live_path))
    events = 3
    namespace["_FR13_DRAFT_HEAD_U8_LIVE_STATE"] = {
        "compares": _Counter([events] * 5),
        "mismatches": _Counter([0] * 5),
        "geometry": copy.deepcopy(gate.EXPECTED_GEOMETRY),
        "candidate": copy.deepcopy(gate.EXPECTED_CANDIDATE),
        "identities": _identities(gate),
        "worker_env_bridge": _bridge(gate),
        "root_forward_steps": list(range(events)),
        "captured_depths": [1, 2, 3, 4],
    }
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "5" * 64,
        "complete_work_census_events": events,
        "events_sha256": "6" * 64,
        "generation": 2,
        "nonce": "7" * 64,
        "producer_pid": 321,
    }
    rows = [{"batch_size": 1, "forward_step_index": index} for index in range(events)]
    namespace["_fr13_draft_head_u8_live_finalize"](rows, binding)
    live = json.loads(live_path.read_text(encoding="ascii"))
    assert gate.validate_live_result(live, expected_source_commit="b" * 40) == live
    assert live["per_depth_full_logit_comparisons"] == {
        label: events for label in gate.DEPTH_LABELS
    }

    aggregate_preserved = copy.deepcopy(live)
    aggregate_preserved["per_depth_full_logit_comparisons"]["mtp_depth_1"] += 1
    aggregate_preserved["per_depth_full_logit_comparisons"]["mtp_depth_2"] -= 1
    assert aggregate_preserved["full_logit_comparisons"] == events * 5
    with pytest.raises(ValueError, match="per-depth comparison/event census"):
        gate.validate_live_result(
            aggregate_preserved, expected_source_commit="b" * 40
        )

    namespace["_FR13_DRAFT_HEAD_U8_LIVE_STATE"]["compares"].values[4] -= 1
    with pytest.raises(RuntimeError, match="depth/event comparison mismatch"):
        namespace["_fr13_draft_head_u8_live_finalize"](rows, binding)
    assert json.loads(live_path.read_text(encoding="ascii"))["status"] == "FAIL"


def test_wiring_is_shadow_only_default_off_and_fully_pinned() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")
    generic = GENERIC_RUNNER.read_text(encoding="utf-8")
    runner = LIVE_RUNNER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "return _fr13_dh_u8_reference" in patcher
    assert "self._fr13_dh_u8_op(" in patcher
    assert "view(torch.int16)" in patcher
    assert "_FR13_DRAFT_HEAD_U8_WORKER_ENV_REQUIRED" in patcher
    assert (
        "FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB="
        "${FR13_DRAFT_HEAD_M1_R64_U8_LIVE_AB:-0}"
    ) in launcher
    assert ":/tmp/fr13_bf16_k64_m1_r64_u8.abi3.so:ro" in launcher
    assert '-e FR13_DRAFT_HEAD_M1_R64_U8_SOURCE_COMMIT=' in launcher
    assert "RUNTIME_DRAFT_HEAD_M32=0" in generic
    assert "FR13_GATE_DRAFT_HEAD_U8=1" in runner
    assert "FR13_B1_DIAGNOSTIC_TASK_PROFILE=astropy12907" in runner
    assert "CUDAGRAPH_MODE=FULL_AND_PIECEWISE" in runner
    assert "TIMING_ARM" not in runner
    for relative in (
        "scripts/fr13_dfwd_k64_m1_r64_u8_gate.py",
        "scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh",
        "csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu",
        "results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/"
        "build_attestation.json",
    ):
        assert relative in manifest


def test_runtime_source_compiles_with_default_off_bridge() -> None:
    source = _patcher()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    namespace: dict[str, object] = {}
    exec(compile(source, "<u8-runtime-full>", "exec"), namespace)
    assert namespace["_FR13_DRAFT_HEAD_U8_WORKER_ENV_BRIDGE"] is None
    assert namespace["_FR13_DRAFT_HEAD_U8_WORKER_ENV_KEYS"] == tuple(
        _gate().WORKER_ENV_KEYS
    )


def test_readiness_manifest_binds_original_qualification_sources() -> None:
    payload = json.loads(READINESS.read_text(encoding="ascii"))
    assert payload["status"] == "SHADOW_READY_UNMEASURED"
    assert payload["source_tip_commit"] == (
        "674f574a0346b4f7b2bc96a30a4ad403841c41d4"
    )
    assert payload["execution"] == {
        "gpu_run": False,
        "docker_run": False,
        "real_swe_verified_run": False,
        "correctness_claim": False,
        "performance_claim": False,
        "timing_eligible": False,
        "production_eligible": False,
    }
    assert payload["comparison_contract"][
        "exhaustive_within_fixed_k64_root1_head"
    ] is True
    assert payload["comparison_contract"][
        "exhaustive_full_model_vocabulary"
    ] is False
    tracked = payload["tracked_inputs"]
    assert set(tracked) == {
        "scripts/fr10_phase4_patch_vllm_tree_gdn.py",
        "scripts/fr13_launch_forked_fa2_tree_server.sh",
        "scripts/fr13_run_b1_kernel_live_gate.sh",
        "scripts/fr13_run_b1_dfwd_k64_m1_r64_u8_live_gate.sh",
        "scripts/fr13_dfwd_k64_m1_r64_u8_gate.py",
        "scripts/fr13_runtime_manifest.py",
        "tests/test_fr13_dfwd_k64_m1_r64_u8_gate.py",
        "tests/test_fr13_draft_head_m32_production.py",
        "csrc/fr13_bf16_gemvx_k64_m1_shuffle_r64_u8.cu",
        "results/fr13_fixed32_dfwd_k64_m1_r64_u8_linked_build_20260805/"
        "build_attestation.json",
        "config/fr13_fixed32/subset_b1_diagnostic_one.json",
        "scripts/fr13_dvk_subset_blocks.json",
    }
    artifact_commit = "2aee844d2b4ce62b901764ce6455bd06914f387b"
    for relative, expected in tracked.items():
        historical = subprocess.run(
            ["git", "-C", str(REPO), "show", f"{artifact_commit}:{relative}"],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        assert hashlib.sha256(historical).hexdigest() == expected
