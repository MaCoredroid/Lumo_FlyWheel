from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import textwrap
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
M1_CUDA = REPO / "csrc" / "fr13_bf16_gemvx_m1.cu"
M1_PATCHER = REPO / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
B4_PATCHER = REPO / "scripts" / "fr13_phase4_patch_vllm_tree_gdn_b1_b4.py"
LAUNCHER = REPO / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _patcher_module():
    spec = importlib.util.spec_from_file_location("b1_b4_patcher", B4_PATCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eagle_snippet() -> str:
    tree = ast.parse(B4_PATCHER.read_text(encoding="utf-8"))
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
                    and "FR13_DRAFT_HEAD_M1_MAX_BATCH" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("B1-B4 runtime snippet not found")


def _snippet_function(name: str) -> ast.FunctionDef:
    functions = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(_eagle_snippet())))
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(functions) == 1
    return functions[0]


def test_qualified_b1_source_and_patcher_are_byte_identical() -> None:
    assert _sha256(M1_CUDA) == (
        "26ea8aad9f891b5e758a39464209d6f82008a10fac8da4c02ee052e839218a54"
    )
    assert _sha256(M1_PATCHER) == (
        "c4b5550cac2bbb5b213d76de3551e3ea61c1a0b5e5db93064404711f6313332d"
    )
    assert "FR13_DRAFT_HEAD_M1_MAX_BATCH" not in M1_PATCHER.read_text(
        encoding="utf-8"
    )


def test_b1_b4_runtime_is_exact4_shadow_only() -> None:
    snippet = _eagle_snippet()
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert '"FR13_DRAFT_HEAD_M1_MAX_BATCH", "1"' in snippet
    assert '"gemvx_b1_b4_out"' in snippet
    assert "torch.ops.fr13_bf16_head.gemvx_b1_b4_out(" in snippet
    assert "_fr13_dh_reference = _sh.quant_method.apply(" in snippet
    assert "_logits = _fr13_dh_reference" in snippet
    assert "B1-B4 kernel is shadow-only until its " in snippet
    assert "exact4 byte gate passes" in snippet
    assert 'FR13_DRAFT_HEAD_M1_MAX_BATCH" == "4"' in launcher
    assert 'FR13_FIXED32_MODE:-}" == "hydra27_fixed32"' in launcher
    assert 'MAX_NUM_SEQS_OVR:-}" == "4"' in launcher
    assert 'SWE_CONCURRENCY:-}" == "4"' in launcher
    assert "B1-B4 kernel is shadow-only until an exact4 byte gate passes" in launcher


def test_b1_b4_contract_is_one_launch_for_each_actual_batch() -> None:
    namespace: dict[str, object] = {"_fr13_dh_m1_max_batch": 4}
    exec(
        compile(
            ast.Module(
                body=[_snippet_function("_fr13_dh_m1_contract")],
                type_ignores=[],
            ),
            "<b1-b4-contract>",
            "exec",
        ),
        namespace,
    )
    contract = namespace["_fr13_dh_m1_contract"]
    for batch_size in (1, 2, 3, 4):
        payload = contract(False, batch_size)
        assert payload["geometry"]["supported_batch_sizes"] == [1, 2, 3, 4]
        assert payload["geometry"]["input_shape"] == [batch_size, 5120]
        assert payload["geometry"]["output_shape"] == [batch_size, 248320]
        assert payload["candidate"]["gemv_mnk"] == [
            batch_size,
            248320,
            5120,
        ]
        assert payload["candidate"]["candidate_launches_per_head"] == 1
        assert payload["candidate"]["shadow_compared_rows"] == batch_size
    with pytest.raises(RuntimeError, match="contract batch drifted"):
        contract(False, 5)


def test_b1_b4_finalizer_counts_rows_and_requires_a_real_b4(
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
    out = tmp_path / "b4-live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_MAX_BATCH", "4")
    monkeypatch.setenv("FR13_DRAFT_HEAD_M1_LIVE_JSON", str(out))

    class Counter:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return self.values

    logical_rows = 10
    state = {
        "compares": Counter([logical_rows] * 5),
        "mismatches": Counter([0] * 5),
        "geometry": {"supported_batch_sizes": [1, 2, 3, 4]},
        "candidate": {"candidate_launches_per_head": 1},
        "binary": {"path": "/tmp/candidate.so", "sha256": "b" * 64, "bytes": 1},
        "source_commit": "a" * 40,
        "candidate_source_sha256": "c" * 64,
        "patcher_sha256": "d" * 64,
        "build_attestation_sha256": "e" * 64,
        "instance_id": "",
        "max_batch": 4,
    }
    namespace = {"_FR13_DRAFT_HEAD_M1_LIVE_STATE": state}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<b4-live>", "exec"),
        namespace,
    )
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "f" * 64,
        "complete_work_census_events": 4,
        "events_sha256": "1" * 64,
        "generation": 3,
        "nonce": "2" * 64,
        "producer_pid": 257,
    }
    namespace["_fr13_draft_head_m1_live_finalize"](
        [{"batch_size": value} for value in (4, 3, 2, 1)], binding
    )
    payload = json.loads(out.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13.fixed32.draft_head_full_b1_b4_live_ab.v1"
    assert payload["status"] == "PASS"
    assert payload["concurrency"] == 4
    assert payload["completed_events"] == logical_rows
    assert payload["full_logit_comparisons"] == logical_rows * 5
    assert payload["raw_bf16_mismatches"] == 0

    binding["complete_work_census_events"] = 3
    with pytest.raises(RuntimeError, match="live finalization drifted"):
        namespace["_fr13_draft_head_m1_live_finalize"](
            [{"batch_size": value} for value in (3, 2, 1)], binding
        )


def test_b4_patcher_and_so_mount_survive_ingress_array_append() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    assert "sha256sum csrc/fr13_bf16_gemvx_b1_b4.cu" in launcher
    assert "sha256sum scripts/fr13_phase4_patch_vllm_tree_gdn_b1_b4.py" in launcher
    assert "python3 /workspace/scripts/fr13_phase4_patch_vllm_tree_gdn_b1_b4.py" in launcher
    mount_index = launcher.index(
        "FR13_DRAFT_HEAD_M1_RUNTIME_SO=/tmp/fr13_bf16_gemvx_b1_b4.abi3.so"
    )
    ingress_index = launcher.index(
        "FR13_FIXED32_CONTAINER_INGRESS_SECRET_FILE=/run/fr13_fixed32_ingress_secret"
    )
    docker_run_index = launcher.index("docker run -d --pull=never")
    assert mount_index < ingress_index < docker_run_index
    assert "FR13_FIXED32_DOCKER_ARGS+=(" in launcher[ingress_index:docker_run_index]
    assert '"${FR13_FIXED32_DOCKER_ARGS[@]}"' in launcher[docker_run_index:]
