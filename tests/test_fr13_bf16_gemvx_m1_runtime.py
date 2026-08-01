from __future__ import annotations

import ast
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_m1_live.sh"
VALIDATOR = REPO / "scripts" / "fr13_draft_head_m1_validate.py"
MANIFEST = REPO / "scripts" / "fr13_runtime_manifest.py"


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


def test_real_b1_runner_is_pinned_nonprobe_and_manifested() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    manifest = MANIFEST.read_text(encoding="utf-8")

    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in runner
    assert "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb" in runner
    assert "astropy__astropy-12907" in runner
    assert "CANONICAL_FA2_SHA256=f51e23c5" in runner
    assert "CANONICAL_FA2_SIZE=299183936" in runner
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
    for path in (
        "csrc/fr13_bf16_gemvx_m1.cu",
        "scripts/fr13_build_bf16_gemvx_m1.py",
        "scripts/fr13_draft_head_m1_validate.py",
        "scripts/fr13_run_b1_draft_head_m1_live.sh",
        "config/fr13_fixed32/subset_b1_diagnostic_one.json",
    ):
        assert f'"{path}"' in manifest


def test_fixed32_flush_calls_m1_finalizer() -> None:
    patcher = PATCHER.read_text(encoding="utf-8")
    assert "_gdn._fr13_draft_head_m1_live_finalize(" in patcher
    assert "_gdn._fr13_draft_head_m32_live_finalize(" in patcher
