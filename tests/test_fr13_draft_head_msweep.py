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
RUNNER = REPO / "scripts" / "fr13_run_b1_draft_head_msweep_live.sh"
VALIDATOR = REPO / "scripts" / "fr13_draft_head_msweep_validate.py"


def _patcher_module():
    spec = importlib.util.spec_from_file_location("msweep_patcher", PATCHER)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _validator_module():
    sys.path.insert(0, str(REPO / "scripts"))
    try:
        spec = importlib.util.spec_from_file_location("msweep_validator", VALIDATOR)
        assert spec is not None
        assert spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def _eagle_snippet() -> str:
    tree = ast.parse(PATCHER.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_patch_eagle_tree_consumption_verify":
            for statement in ast.walk(node):
                if (
                    isinstance(statement, ast.Assign)
                    and len(statement.targets) == 1
                    and isinstance(statement.targets[0], ast.Name)
                    and statement.targets[0].id == "new"
                    and isinstance(statement.value, ast.Constant)
                    and isinstance(statement.value.value, str)
                    and "FR13_DRAFT_HEAD_MSWEEP_LIVE_AB" in statement.value.value
                ):
                    return statement.value.value
    raise AssertionError("draft-head small-M sweep snippet not found")


def _snippet_functions(*names: str) -> list[ast.FunctionDef]:
    wanted = set(names)
    selected = [
        node
        for node in ast.walk(ast.parse(textwrap.dedent(_eagle_snippet())))
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in selected} == wanted
    return selected


def _contract() -> dict[str, object]:
    namespace: dict[str, object] = {}
    exec(
        compile(
            ast.Module(
                body=_snippet_functions("_fr13_dh_msweep_contract"),
                type_ignores=[],
            ),
            "<small-m-contract>",
            "exec",
        ),
        namespace,
    )
    return namespace["_fr13_dh_msweep_contract"]()


def _live_payload(source_sha: str) -> dict[str, object]:
    contract = _contract()
    candidates = []
    for index, candidate in enumerate(contract["candidates"]):
        candidates.append(
            {
                **candidate,
                "head_comparisons": 5,
                "bf16_elements_compared": 5 * 248320,
                "raw_bf16_mismatches": index,
                "byte_exact": index == 0,
            }
        )
    return {
        "schema": "fr13.fixed32.draft_head_full_msweep_live_ab.v1",
        "status": "COMPLETE",
        "suite": "SWE-Verified",
        "instance_id": "astropy__astropy-12907",
        "task_marker": "swe_verified:astropy__astropy-12907",
        "concurrency": 1,
        "batch_size": 1,
        "source_commit": "a" * 40,
        "candidate_source_sha256": source_sha,
        "geometry": contract["geometry"],
        "candidate_rows": [2, 4, 8, 16],
        "candidates": candidates,
        "diagnostic_event": {
            "batch_size": 1,
            "forward_step_index": 0,
            "graph_id": 41,
            "graph_signature": (
                "d9a4ddece41d146e9949b9f8ff7c2603"
                "b8948d157b28ef69244e44469b36150c"
            ),
            "graph_replays": 1,
            "measured": True,
            "runtime_mode": "hydra27_fixed32",
            "head_positions_compared": 5,
        },
        "completed_events": 7,
        "complete_work_census_events": 7,
        "work_census_last_event_index": 6,
        "events_sha256": "c" * 64,
        "flush_generation": 2,
        "flush_nonce": "d" * 64,
        "producer_pid": 42,
        "boundary_snapshot_sha256": "e" * 64,
        "served_return": "reference BF16 logits unchanged",
        "performance_measurement": False,
        "acceptance_eligible": False,
        "probe_eligible": False,
        "finalized_by_fixed32_flush": True,
        "flush_action": "final",
    }


def test_contract_sweeps_only_small_valid_rows() -> None:
    contract = _contract()

    assert contract["geometry"]["head_positions"] == [
        "root",
        "mtp1",
        "mtp2",
        "mtp3",
        "mtp4",
    ]
    assert contract["geometry"]["weight_shape"] == [248320, 5120]
    assert [candidate["m"] for candidate in contract["candidates"]] == [2, 4, 8, 16]
    assert all(candidate["m"] != 32 for candidate in contract["candidates"])
    for candidate in contract["candidates"]:
        rows = candidate["m"]
        assert candidate["gemm_mnk"] == [rows, 248320, 5120]
        assert candidate["served_rows"] == 0
        assert candidate["shadow_compared_rows"] == 1
        assert candidate["valid_live_batch_sizes"] == list(range(1, rows + 1))


def test_runtime_snapshots_graph_heads_then_runs_one_measured_event() -> None:
    source = textwrap.dedent(_eagle_snippet())
    tree = ast.parse(source)
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }

    run_source = ast.unparse(functions["_fr13_dh_msweep_run_once"])
    reference_source = ast.unparse(functions["_fr13_dh_msweep_reference"])
    assert "if _fr13_dh_run_state['completed']" in run_source
    assert "_fr13_dh_proposal.get('measured') is not True" in run_source
    assert "range(5)" in run_source
    assert "(2, 4, 8, 16)" in run_source
    assert "torch.cuda.synchronize" not in run_source
    assert "torch.cuda.is_current_stream_capturing()" in reference_source
    assert "_fr13_dh_msweep_capture_calls" in reference_source
    assert source.count("_fr13_dh_msweep_run_once(") == 3
    assert "_logits = _fr13_dh_msweep_reference(_h)" in source


def test_finalizer_emits_per_m_mismatches_without_serving_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _patcher_module()._FR13_FIXED32_OBSERVED_RUNTIME_SOURCE
    selected = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_draft_head_msweep_live_finalize"
    ]
    assert len(selected) == 1

    class Counter:
        def __init__(self, values: list[int]) -> None:
            self.values = values

        def tolist(self) -> list[int]:
            return self.values

    out = tmp_path / "live.json"
    monkeypatch.setenv("FR13_DRAFT_HEAD_MSWEEP_LIVE_AB", "1")
    monkeypatch.setenv("FR13_DRAFT_HEAD_MSWEEP_LIVE_JSON", str(out))
    contract = _contract()
    state = {
        "compares": Counter([5, 5, 5, 5]),
        "mismatches": Counter([0, 2, 3, 4]),
        "geometry": contract["geometry"],
        "candidates": contract["candidates"],
        "run_state": {
            "completed": True,
            "event": {
                "batch_size": 1,
                "forward_step_index": 0,
                "graph_id": 7,
                "graph_signature": "f" * 64,
                "graph_replays": 1,
                "measured": True,
                "runtime_mode": "hydra27_fixed32",
                "head_positions_compared": 5,
            },
        },
        "source_commit": "a" * 40,
        "candidate_source_sha256": "b" * 64,
        "instance_id": "astropy__astropy-12907",
    }
    namespace = {"_FR13_DRAFT_HEAD_MSWEEP_LIVE_STATE": state}
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "<msweep-final>", "exec"),
        namespace,
    )
    binding = {
        "action": "final",
        "boundary_snapshot_sha256": "c" * 64,
        "complete_work_census_events": 7,
        "events_sha256": "d" * 64,
        "generation": 3,
        "nonce": "e" * 64,
        "producer_pid": 257,
    }
    namespace["_fr13_draft_head_msweep_live_finalize"](
        [{"batch_size": 1} for _ in range(7)], binding
    )
    payload = json.loads(out.read_text(encoding="ascii"))
    assert payload["status"] == "COMPLETE"
    assert payload["candidate_rows"] == [2, 4, 8, 16]
    assert [row["raw_bf16_mismatches"] for row in payload["candidates"]] == [
        0,
        2,
        3,
        4,
    ]
    assert [row["byte_exact"] for row in payload["candidates"]] == [
        True,
        False,
        False,
        False,
    ]
    assert payload["served_return"] == "reference BF16 logits unchanged"
    assert payload["acceptance_eligible"] is False

    state["compares"].values[2] = 4
    with pytest.raises(RuntimeError, match="comparison/event census drifted"):
        namespace["_fr13_draft_head_msweep_live_finalize"](
            [{"batch_size": 1} for _ in range(7)], binding
        )
    assert json.loads(out.read_text(encoding="ascii"))["status"] == "FAIL"


def test_validator_accepts_mismatch_results_but_rejects_m32() -> None:
    module = _validator_module()
    source_sha = "1" * 64
    payload = _live_payload(source_sha)

    summary = module.validate_live_result(
        payload, expected_source_sha256=source_sha
    )
    assert summary["candidates"][0] == {
        "m": 2,
        "raw_bf16_mismatches": 0,
        "byte_exact": True,
    }
    assert summary["candidates"][3]["raw_bf16_mismatches"] == 3

    payload["candidate_rows"] = [2, 4, 8, 32]
    with pytest.raises(ValueError, match="provenance drifted"):
        module.validate_live_result(payload, expected_source_sha256=source_sha)


def test_launcher_and_runner_are_real_b1_reference_served_only() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    runner = RUNNER.read_text(encoding="utf-8")

    assert "FR13 draft-head M32 is retired after real-B1 byte rejection" in launcher
    assert '-e FR13_DRAFT_HEAD_MSWEEP_LIVE_AB="$FR13_DRAFT_HEAD_MSWEEP_LIVE_AB"' in launcher
    assert '-e FR13_DRAFT_HEAD_MSWEEP_SOURCE_SHA256="$FR13_DRAFT_HEAD_MSWEEP_SOURCE_SHA256"' in launcher
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in runner
    assert "cc0264dbeab51847000bea7d14e9ada1d3a7c0d49182d423554c15e88417fefb" in runner
    assert "astropy__astropy-12907" in runner
    assert "classification=real_swe_verified_b1_full_vocab_small_m_byte_diagnostic" in runner
    assert "diagnostic_only=1" in runner
    assert "performance_measurement=0" in runner
    assert "acceptance_eligible=0" in runner
    assert "probe_eligible=0" in runner
    assert "reference_served=1" in runner
    assert "candidate_rows=2,4,8,16" in runner
    assert "FR13_DRAFT_VOCAB_ROOT=0" in runner
    assert "FR13_DRAFT_VOCAB_K=0" in runner
    assert "FR13_DRAFT_HEAD_M32_LIVE_AB=0" in runner
    assert "FR13_DRAFT_HEAD_MSWEEP_LIVE_AB=1" in runner
    assert "scripts/fr13_draft_head_msweep_validate.py" in runner
    assert '[[ "$(docker ps -aq | wc -l)" -eq 0 ]]' in runner
    assert 'cmp -s "$RUNROOT_ABS/runtime_manifest.at_launch.json"' in runner
