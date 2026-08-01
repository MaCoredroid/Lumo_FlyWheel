from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import textwrap
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m1_live.sh"
TIMING_RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m1_timing.sh"
VALIDATOR = REPO / "scripts" / "fr13_draft_head_m1_validate.py"
EXACT4_SUBSET = REPO / "config" / "fr13_fixed32" / "subset_b4_four.json"
RECOVERED_STOCK = (
    REPO
    / "results"
    / "fr13_fixed32_bf16_gemvx_m1_exact4_stock_recovered_20260801"
    / "deploy_speed_fullwall.json"
)
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"
PREPARED = (
    REPO
    / "results"
    / "fr13_fixed32_bf16_gemvx_m1_b1_ready_20260801"
    / "prepared_command.sh"
)


def _patcher_module():
    spec = importlib.util.spec_from_file_location("m1_patcher", PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator_module():
    scripts = str(REPO / "scripts")
    sys.path.insert(0, scripts)
    try:
        spec = importlib.util.spec_from_file_location("m1_validator", VALIDATOR)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(scripts)


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == (
            "_patch_eagle_tree_consumption_verify"
        ):
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_M1_LIVE_AB" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head M1 runtime snippet not found")


def _snippet_function(name: str) -> ast.FunctionDef:
    functions = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(_eagle_snippet())))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1
    return functions[0]


def test_m1_runtime_is_default_off_strict_b1_full_vocab() -> None:
    snippet = _eagle_snippet()
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert '"FR13_DRAFT_HEAD_M1_LIVE_AB", "0"' in snippet
    assert "FR13 full-head M1 live A/B requires exact fixed32 B1" in snippet
    assert "_fr13_dvk_root" in snippet
    assert "_fr13_dvk_configured != 0" in snippet
    assert 'os.environ.get("FR13_DRAFT_VOCAB_BLOCKS", "")' in snippet
    assert "tuple(_fr13_dh_w.shape) != (248320, 5120)" in snippet
    assert 'type(_fr13_dh_sh).__name__ != "ParallelLMHead"' in snippet
    assert '!= "UnquantizedEmbeddingMethod"' in snippet
    assert "FR13_DRAFT_HEAD_M1_SOURCE_SHA256" in snippet
    assert "FR13_DRAFT_HEAD_M1_PATCHER_SHA256" in snippet
    assert "FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION_SHA256" in snippet
    assert "FR13_DRAFT_HEAD_M1_SO_SHA256" in snippet

    assert "FR13_DRAFT_HEAD_M1_LIVE_AB=${FR13_DRAFT_HEAD_M1_LIVE_AB:-0}" in launcher
    assert "FR13 draft-head M1 live A/B requires its pinned SO" in launcher
    assert ':ro"' in launcher
    assert '"$_v" == "FR13_DRAFT_HEAD_M1_SO"' in launcher
    assert "FR13_DRAFT_HEAD_M1_RUNTIME_SO=/tmp/fr13_bf16_gemvx_m1.abi3.so" in launcher
    mount_index = launcher.index(
        'FR13_DRAFT_HEAD_M1_RUNTIME_SO=/tmp/fr13_bf16_gemvx_m1.abi3.so'
    )
    ingress_index = launcher.index(
        'FR13_FIXED32_CONTAINER_INGRESS_SECRET_FILE=/run/fr13_fixed32_ingress_secret'
    )
    assert mount_index < ingress_index
    assert "FR13_FIXED32_DOCKER_ARGS+=(" in launcher[ingress_index:]


def test_m1_contract_and_shadow_order_cover_all_five_heads() -> None:
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(
                body=[_snippet_function("_fr13_dh_m1_contract")],
                type_ignores=[],
            ),
            "<m1-contract>",
            "exec",
        ),
        namespace,
    )
    contract = namespace["_fr13_dh_m1_contract"]()
    validator = _validator_module()
    assert contract["geometry"] == validator.EXPECTED_GEOMETRY
    assert contract["candidate"] == validator.EXPECTED_CANDIDATE

    snippet = _eagle_snippet()
    branch_start = snippet.index("if _fr13_dh_m1_live_on:")
    branch_end = snippet.index("elif _fr13_dh_m32_on:", branch_start)
    branch = snippet[branch_start:branch_end]
    assert branch.index("_sh.quant_method.apply") < branch.index(
        "_fr13_dh_m1_logits"
    )
    assert "_logits = _fr13_dh_reference" in branch
    assert "_fr13_dh_candidate.view(torch.int16)" in branch
    assert "_fr13_dh_reference.view(torch.int16)" in branch
    assert "self._fr13_dh_m1_capture_position" in branch
    assert "not 1 <= _fr13_dh_position <= 4" in branch
    assert "_fr13_dh_position = 0" in branch
    assert snippet.count("_fr13_dvk_logits(") >= 3


def test_m1_production_is_candidate_only_and_fail_closed() -> None:
    snippet = _eagle_snippet()
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert '"FR13_DRAFT_HEAD_M1_PRODUCTION", "0"' in snippet
    assert "FR13_DRAFT_HEAD_M1_INTERNAL_PRODUCTION_ATTESTED" in snippet
    branch_start = snippet.index("elif _fr13_dh_m1_prod_on:")
    branch_end = snippet.index("elif _fr13_dh_m32_on:", branch_start)
    branch = snippet[branch_start:branch_end]
    assert "_sh.quant_method.apply" not in branch
    assert "_fr13_dh_candidate = _fr13_dh_m1_logits(_h)" in branch
    assert "_fr13_dh_m1_note_production(_fr13_dh_capturing)" in branch
    assert "_logits = _fr13_dh_candidate" in branch
    assert "_fr13_dh_m1_note_production_replay" in snippet

    assert "FR13_DRAFT_HEAD_M1_PRODUCTION=${FR13_DRAFT_HEAD_M1_PRODUCTION:-0}" in launcher
    assert "draft-head M1 live A/B and production are mutually exclusive" in launcher
    assert "fr13_draft_head_m1_validate.py issue" in launcher
    assert "fr13_draft_head_m1_validate.py verify" in launcher
    assert "FR13_DRAFT_HEAD_M1_INTERNAL_PRODUCTION_ATTESTED=1" in launcher
    assert "M1 timing arm permits only stock or M1 production" in launcher
    assert 'if [[ "$FR13_DRAFT_HEAD_M1_LIVE_AB" == "1" \\' in launcher
    assert '|| "$FR13_DRAFT_HEAD_M1_PRODUCTION" == "1" ]]; then' in launcher


def test_m1_production_contract_reports_the_served_candidate() -> None:
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(
                body=[_snippet_function("_fr13_dh_m1_contract")],
                type_ignores=[],
            ),
            "<m1-production-contract>",
            "exec",
        ),
        namespace,
    )
    validator = _validator_module()
    contract = namespace["_fr13_dh_m1_contract"](True)
    assert contract["geometry"] == validator.EXPECTED_GEOMETRY
    assert contract["candidate"] == validator.EXPECTED_PRODUCTION_CANDIDATE
    assert contract["candidate"]["served_rows"] == 1
    assert contract["candidate"]["shadow_compared_rows"] == 0


def test_m1_finalizer_requires_exact_per_position_event_census(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _patcher_module()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_draft_head_m1_live_finalize"
    ]
    assert len(selected) == 1
    validator = _validator_module()
    out = tmp_path / "live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_LIVE_JSON", str(out))

    class Counter:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return self.values

    state = {
        "compares": Counter([7, 7, 7, 7, 7]),
        "mismatches": Counter([0, 0, 0, 0, 0]),
        "geometry": validator.EXPECTED_GEOMETRY,
        "candidate": validator.EXPECTED_CANDIDATE,
        "binary": {
            "path": "/tmp/fr13_bf16_gemvx_m1.abi3.so",
            "sha256": "b" * 64,
            "bytes": 162160,
        },
        "source_commit": "a" * 40,
        "candidate_source_sha256": "c" * 64,
        "patcher_sha256": "d" * 64,
        "build_attestation_sha256": "e" * 64,
        "instance_id": validator.EXPECTED_INSTANCE,
    }
    namespace = {"_FR13_DRAFT_HEAD_M1_LIVE_STATE": state}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<m1-live>", "exec"),
        namespace,
    )
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "f" * 64,
        "complete_work_census_events": 7,
        "events_sha256": "1" * 64,
        "generation": 3,
        "nonce": "2" * 64,
        "producer_pid": 257,
    }
    namespace["_fr13_draft_head_m1_live_finalize"](
        [{"batch_size": 1} for _ in range(7)], binding
    )
    payload = json.loads(out.read_text(encoding="ascii"))
    validator.validate_live_result(
        payload,
        expected_source_sha256="c" * 64,
        expected_patcher_sha256="d" * 64,
        expected_build_attestation_sha256="e" * 64,
        expected_so_sha256="b" * 64,
        expected_so_bytes=162160,
    )
    assert [row["position"] for row in payload["per_head"]] == list(
        validator.POSITIONS
    )
    assert payload["bf16_elements_compared"] == 7 * 5 * 248320

    state["compares"].values[4] = 6
    with pytest.raises(RuntimeError, match="comparison/event census mismatch"):
        namespace["_fr13_draft_head_m1_live_finalize"](
            [{"batch_size": 1} for _ in range(7)], binding
        )
    assert json.loads(out.read_text(encoding="ascii"))["status"] == "FAIL"


def test_build_attestation_binds_source_so_and_pinned_toolchain() -> None:
    validator = _validator_module()
    payload = {
        "schema": "fr13.fixed32.bf16_gemvx_m1_build.v1",
        "status": "BUILT_UNQUALIFIED",
        "performance_measurement": False,
        "byte_equality_claim": False,
        "production_default_enabled": False,
        "torch_version": "2.10.0+cu130",
        "cuda_release": "13.0",
        "cuda_arch": "12.1a",
        "source": {
            "path": "csrc/fr13_bf16_gemvx_m1.cu",
            "sha256": "a" * 64,
        },
        "binary": {
            "path": "results/candidate.abi3.so",
            "sha256": "b" * 64,
            "bytes": 162160,
            "mode": "0555",
        },
        "kernel_contract": validator.EXPECTED_BUILD_CONTRACT,
    }
    validator.validate_build_attestation(
        payload,
        expected_source_sha256="a" * 64,
        expected_so_sha256="b" * 64,
        expected_so_bytes=162160,
    )
    payload["cuda_arch"] = "12.0"
    with pytest.raises(ValueError, match="contract drifted"):
        validator.validate_build_attestation(
            payload,
            expected_source_sha256="a" * 64,
            expected_so_sha256="b" * 64,
            expected_so_bytes=162160,
        )


def test_m1_production_sidecar_binds_all_candidate_identities(
    tmp_path: Path,
) -> None:
    validator = _validator_module()
    source = tmp_path / "candidate.cu"
    patcher = tmp_path / "patcher.py"
    candidate_so = tmp_path / "candidate.so"
    source.write_bytes(b"source")
    patcher.write_bytes(b"patcher")
    candidate_so.write_bytes(b"binary")
    source_sha = validator.sha256_file(source)
    patcher_sha = validator.sha256_file(patcher)
    so_sha = validator.sha256_file(candidate_so)
    build_sha = "b" * 64
    live_sha = "c" * 64
    body = {
        "schema": validator.SIDECAR_SCHEMA,
        "status": "PASS",
        "live_gate_schema": validator.LIVE_SCHEMA,
        "validation_schema": validator.VALIDATION_SCHEMA,
        "live_result_sha256": live_sha,
        "live_result_canonical_sha256": "d" * 64,
        "instance_id": validator.EXPECTED_INSTANCE,
        "qualified_source_commit": "a" * 40,
        "qualified_candidate_source_sha256": source_sha,
        "qualified_patcher_sha256": patcher_sha,
        "qualified_build_attestation_sha256": build_sha,
        "qualified_candidate_so_sha256": so_sha,
        "qualified_candidate_so_bytes": candidate_so.stat().st_size,
        "qualified_completed_events": 7,
        "qualified_events_sha256": "e" * 64,
        "qualified_flush_generation": 3,
        "final_flush_sha256": "f" * 64,
        "boundary_snapshot_sha256": "1" * 64,
        "chat_traffic_audit_sha256": "2" * 64,
        "qualified_trace_completed_logical_model_requests": 1,
        "candidate": validator.EXPECTED_PRODUCTION_CANDIDATE,
        "geometry": validator.EXPECTED_GEOMETRY,
        "required_runtime": "fixed32 B1 full drafter graph, K0/root0",
        "production_scope": (
            "five exact full-vocabulary BF16 M1 GEMV calls per event"
        ),
    }
    payload = dict(body)
    payload["canonical_sha256"] = validator._digest_bytes(
        validator.canonical_bytes(body)
    )
    sidecar = tmp_path / "production_pass.json"
    sidecar.write_bytes(validator.canonical_bytes(payload) + b"\n")
    sidecar_sha = validator.sha256_file(sidecar)

    assert validator.verify_sidecar(
        sidecar_path=sidecar,
        expected_sidecar_sha256=sidecar_sha,
        expected_live_sha256=live_sha,
        candidate_source=source,
        expected_candidate_source_sha256=source_sha,
        patcher=patcher,
        expected_patcher_sha256=patcher_sha,
        candidate_so=candidate_so,
        expected_candidate_so_sha256=so_sha,
        expected_build_attestation_sha256=build_sha,
    ) == payload
    with pytest.raises(ValueError, match="contract drifted"):
        validator.verify_sidecar(
            sidecar_path=sidecar,
            expected_sidecar_sha256=sidecar_sha,
            expected_live_sha256=live_sha,
            candidate_source=source,
            expected_candidate_source_sha256=source_sha,
            patcher=patcher,
            expected_patcher_sha256="3" * 64,
            candidate_so=candidate_so,
            expected_candidate_so_sha256=so_sha,
            expected_build_attestation_sha256=build_sha,
        )


def test_m1_production_engages_only_after_measured_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    validator = _validator_module()
    graph_id = 123
    signature = validator.EXPECTED_GRAPH_SIGNATURE
    source_sha = "a" * 64
    patcher_sha = "b" * 64
    build_sha = "c" * 64
    so_sha = "d" * 64
    sidecar_sha = "e" * 64
    state = SimpleNamespace(
        _fr13_dh_m1_production_active=True,
        _fr13_dh_m1_engagement_written=False,
        _fr13_dh_m1_selected_root_calls=0,
        _fr13_dh_m1_selected_capture_calls=0,
        _fr13_dh_m1_fallback_calls=0,
        _fr13_dh_m1_graph_attestation=None,
    )
    proposal = {
        "batch_size": 1,
        "mode": "hydra27_fixed32",
        "graph_id": graph_id,
        "graph_signature": signature,
        "graph_replays": 1,
        "measured": False,
        "forward_step_index": 0,
    }
    lifecycle = {
        "captures": 1,
        "batch_size": 1,
        "graph_signature": signature,
        "capture_origin": "unmeasured",
        "measured_replays": 0,
    }
    gdn = types.ModuleType("gdn_linear_attn")
    gdn._FR13_FIXED32_DRAFTER_PROPOSAL_CURRENT = proposal
    gdn._FR13_FIXED32_DRAFTER_GRAPH_LIFECYCLE = {graph_id: lifecycle}
    packages = {
        "vllm": types.ModuleType("vllm"),
        "vllm.model_executor": types.ModuleType("vllm.model_executor"),
        "vllm.model_executor.layers": types.ModuleType(
            "vllm.model_executor.layers"
        ),
        "vllm.model_executor.layers.mamba": types.ModuleType(
            "vllm.model_executor.layers.mamba"
        ),
    }
    packages["vllm.model_executor.layers.mamba"].gdn_linear_attn = gdn
    for name, package in packages.items():
        monkeypatch.setitem(sys.modules, name, package)
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_SOURCE_COMMIT", "f" * 40)
    monkeypatch.setenv(
        "FR13_DRAFT_HEAD_M1_PRODUCTION_PASS_SIDECAR_SHA256", sidecar_sha
    )
    writes: list[dict[str, object]] = []
    namespace = {
        "self": state,
        "os": os,
        "_fr13_dh_m1_source_sha": source_sha,
        "_fr13_dh_m1_patcher_sha": patcher_sha,
        "_fr13_dh_m1_build_attestation_sha": build_sha,
        "_fr13_dh_m1_so_sha": so_sha,
        "_fr13_dh_m1_contract": lambda _production=False: {
            "geometry": validator.EXPECTED_GEOMETRY,
            "candidate": validator.EXPECTED_PRODUCTION_CANDIDATE,
        },
        "_fr13_dh_m32_atomic_json": lambda _path, payload: writes.append(
            payload
        ),
    }
    functions = [
        _snippet_function("_fr13_dh_m1_note_production"),
        _snippet_function("_fr13_dh_m1_note_production_replay"),
    ]
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            "<m1-production-lifecycle>",
            "exec",
        ),
        namespace,
    )
    note = namespace["_fr13_dh_m1_note_production"]
    replay = namespace["_fr13_dh_m1_note_production_replay"]
    note(False)
    for _ in range(4):
        note(True)
    replay(graph_id, signature, 1)
    assert writes == []

    proposal["measured"] = True
    proposal["forward_step_index"] = 1
    lifecycle["measured_replays"] = 1
    note(False)
    replay(graph_id, signature, 1)
    assert len(writes) == 1
    engagement = tmp_path / "engagement.json"
    engagement.write_bytes(validator.canonical_bytes(writes[0]) + b"\n")
    assert validator.validate_engagement(
        engagement_path=engagement,
        expected_source_sha256=source_sha,
        expected_patcher_sha256=patcher_sha,
        expected_build_attestation_sha256=build_sha,
        expected_so_sha256=so_sha,
        expected_sidecar_sha256=sidecar_sha,
    ) == writes[0]


def test_real_b1_runner_is_pinned_nonprobe_and_manifested() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")
    prepared = PREPARED.read_text(encoding="utf-8")

    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in runner
    assert "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb" in runner
    assert "astropy__astropy-12907" in runner
    assert "CANONICAL_FA2_SHA256=f51e23c5" in runner
    assert "CANONICAL_FA2_SIZE=299183936" in runner
    assert 'CANONICAL_FA2="$REPO/output/' not in runner
    assert 'stat -c %s "$FORKED_FA2_SO"' in runner
    assert 'sha256sum "$FORKED_FA2_SO"' in runner
    assert "FR13_DRAFT_HEAD_M1_EXPECTED_SOURCE_COMMIT" in runner
    assert "git status --porcelain=v1" in runner
    assert "--untracked-files=no" not in runner
    assert "validate-build" in runner
    assert "FR13_DRAFT_HEAD_M1_BUILD_ATTESTATION_SHA256" in runner
    assert "FR13_DRAFT_VOCAB_ROOT=0" in runner
    assert "FR13_DRAFT_VOCAB_K=0" in runner
    assert 'FR13_NEEDS_ALLOW="FR13_DRAFT_VOCAB_K=0"' in runner
    assert "FR13_MANDATORY_WEIGHT_BYTES=42025179008" in runner
    assert "FR13_WEIGHT_FLOOR_MS=153.938384645" in runner
    assert "bash scripts/fr13_bigdenom_swe_serve_variant.sh" in runner
    assert "classification=real_swe_verified_b1_kernel_byte_diagnostic" in runner
    assert "diagnostic_only=1" in runner
    assert "performance_measurement=0" in runner
    assert "probe_eligible=0" in runner
    assert "floor_acceptance_eligible=0" in runner
    assert "fr13_draft_head_m1_validate.py" in runner
    assert 'FORKED_FA2_SO:?set an absolute path' in prepared
    assert "PYTHON_BIN=${PYTHON_BIN:-/usr/bin/python3}" in prepared
    assert 'PYTHON_BIN="$PYTHON_BIN"' in prepared
    assert 'FORKED_FA2_SO="$FORKED_FA2_SO"' in prepared
    assert "CANONICAL_FA2_SHA256=f51e23c5" in prepared
    assert "CANONICAL_FA2_SIZE=299183936" in prepared
    assert "$REPO/output/auto_research" not in prepared
    for path in (
        "csrc/fr13_bf16_gemvx_m1.cu",
        "scripts/fr13_build_bf16_gemvx_m1.py",
        "scripts/fr13_draft_head_m1_validate.py",
        "scripts/fr13_run_b1_draft_head_m1_live.sh",
        "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    ):
        assert f'"{path}"' in manifest


def test_m1_timing_runner_is_exact4_b1_full_wall_and_isolated() -> None:
    runner = TIMING_RUNNER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "config/fr13_fixed32/subset_b4_four.json" in runner
    assert "0e37b7137115332372ef76ba7c8db0db4a46ebad5db777c5b999bf797ae853f5" in runner
    assert "MAX_NUM_SEQS_OVR=1 SWE_CONCURRENCY=1" in runner
    assert "fr13_draft_head_m1_validate.py validate-live" in runner
    assert 'FR13_DRAFT_HEAD_M1_PRODUCTION="$production"' in runner
    assert "FR13_DRAFT_HEAD_M1_TIMING_ARM=1" in runner
    assert 'FR13_DRAFT_HEAD_M1_SO="$candidate_so"' in runner
    assert "FR13_DRAFT_HEAD_M32_PRODUCTION=0" in runner
    assert "export FR13_DRAFT_VOCAB_ROOT=0" in runner
    assert "export FR13_DRAFT_VOCAB_K=0" in runner
    assert "scripts/fr13_measure.py deploy-speed" in runner
    assert 'record.get("n_tasks") != 4' in runner
    assert '"measured_tps_fullstep_wall"' in runner
    assert "MIN_RETAINED_WALL_FRACTION = 0.99" in runner
    assert "MIN_TASK_COUNTER_STEPS = 64" in runner
    assert "floor_acceptance_eligible=0" in runner
    assert "stock arm emitted M1 production sidecar" in runner
    assert '"qualified_patcher_sha256"' in runner
    assert '"qualified_build_attestation_sha256"' in runner
    assert '"qualified_candidate_so_sha256"' in runner
    assert '"scripts/fr13_run_b1_draft_head_m1_timing.sh"' in manifest


def test_m1_recovered_stock_route_is_exact4_candidate_only_and_fail_closed(
    tmp_path: Path,
) -> None:
    runner = TIMING_RUNNER.read_text(encoding="utf-8")
    validator = _validator_module()
    stock_sha = validator.sha256_file(RECOVERED_STOCK)
    result = validator.validate_recovered_stock(
        deploy_speed=RECOVERED_STOCK,
        expected_deploy_speed_sha256=stock_sha,
        exact4_subset=EXACT4_SUBSET,
        expected_exact4_subset_sha256=(
            validator.EXPECTED_EXACT4_SUBSET_SHA256
        ),
    )

    assert result["status"] == "PASS"
    assert result["stock_arm"].startswith("hydra27_fixed32_head_stock_")
    assert result["instance_ids"] == list(
        validator.EXPECTED_EXACT4_INSTANCE_IDS
    )
    assert result["timing_pair_eligible"] is False
    assert result["retained_wall_fraction"] >= 0.95
    assert "RECOVERED_STOCK_JSON and RECOVERED_STOCK_SHA256 must be set together" in runner
    assert "fr13_draft_head_m1_validate.py recovered-stock" in runner
    assert "recovered exact4 stock validated; launching candidate arm only" in runner
    assert "real_swe_verified_exact4_b1_recovered_cross_run_diagnostic" in runner
    assert '"diagnostic_only": True' in runner
    assert '"timing_eligible": False' in runner
    assert '"candidate_source_sha256": source_sha' in runner
    assert '"patcher_sha256": patcher_sha' in runner
    assert "git status --porcelain=v1" in runner
    assert "--untracked-files=no" not in runner

    tampered = tmp_path / "candidate.json"
    payload = json.loads(RECOVERED_STOCK.read_text(encoding="utf-8"))
    payload["arm"] = payload["arm"].replace("head_stock", "head_m1")
    tampered.write_text(json.dumps(payload) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="not canonical exact4 B1"):
        validator.validate_recovered_stock(
            deploy_speed=tampered,
            expected_deploy_speed_sha256=validator.sha256_file(tampered),
            exact4_subset=EXACT4_SUBSET,
            expected_exact4_subset_sha256=(
                validator.EXPECTED_EXACT4_SUBSET_SHA256
            ),
        )


def test_fixed32_flush_calls_m1_finalizer() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    assert "_gdn._fr13_draft_head_m1_live_finalize(" in patcher
    assert "_gdn._fr13_draft_head_m32_live_finalize(" in patcher
