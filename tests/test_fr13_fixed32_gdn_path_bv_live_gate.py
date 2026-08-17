from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"


def _kernel_gate_namespace() -> dict[str, object]:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id
            in {
                "_FR13_FIXED32_MODES",
                "_FR13_FIXED32_GDN_PATH_BV_SIDECARS",
                "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_SIDECARS",
                "_FR13_FIXED32_GDN_BV_SURFACES",
                "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE_ID",
                "_FR13_DRAFT_VOCAB_PROFILES",
                "_FR13_DRAFT_VOCAB_CREDENTIAL_FIELDS",
                "_FR13_GDN_ORDERED_CANDIDATES",
            }
            for target in node.targets
        )
    ]
    wanted = {
        "_fr13_resolve_fixed32_gdn_path_bv_candidate",
        "_fr13_tensor_byte_equal",
        "_fr13_resolve_fixed32_gdn_path_bv_production",
        "_fr13_fixed32_gdn_bv_compare_records",
        "fixed32_gdn_bv_live_gate_on_replay",
        # FR14 declared draft-vocabulary identity.
        "_fr13_draft_vocab_profile",
        "_fr13_draft_vocab_env_matches",
        "_fr13_draft_vocab_credential_matches",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    assert {node.name for node in definitions} == wanted
    namespace: dict[str, object] = {"json": json, "os": os, "torch": torch}
    exec(
        compile(
            ast.Module(body=[*assignments, *definitions], type_ignores=[]),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace


def test_byte_comparator_handles_scalar_integer_surfaces() -> None:
    namespace = _kernel_gate_namespace()
    byte_equal = namespace["_fr13_tensor_byte_equal"]

    assert byte_equal(
        torch.tensor(7, dtype=torch.int32),
        torch.tensor(7, dtype=torch.int32),
    )
    assert not byte_equal(
        torch.tensor(7, dtype=torch.int32),
        torch.tensor(8, dtype=torch.int32),
    )


@pytest.mark.parametrize("candidate", (16, 32, 64, 128))
def test_live_gate_runs_distinct_bv_then_restores_served_bytes(
    candidate: int,
) -> None:
    namespace = _kernel_gate_namespace()
    surfaces = namespace["_FR13_FIXED32_GDN_BV_SURFACES"]
    state = {name: f"served:{name}".encode("ascii") for name in surfaces}
    served = dict(state)
    calls: list[int] = []

    def snapshot():
        return dict(state)

    def restore(value):
        state.clear()
        state.update(value)

    def run(block_v: int):
        calls.append(block_v)
        for name in surfaces:
            state[name] = f"gate:{name}".encode("ascii")
        return {
            "block_v": block_v,
            "launch_key": ("tree_gdn_path", block_v),
            "output": b"identical-output",
        }

    record = {
        "snapshot": snapshot,
        "restore": restore,
        "run": run,
        "byte_equal": lambda left, right: left == right,
        "surface_names": surfaces,
    }
    result = namespace["_fr13_fixed32_gdn_bv_compare_records"](
        (record,), candidate
    )

    assert result == {
        "records": 1,
        "reference_bv": 8,
        "candidate_bv": candidate,
    }
    assert calls == [8, candidate]
    assert state == served


def test_live_gate_fails_mismatch_after_restoring_served_bytes() -> None:
    namespace = _kernel_gate_namespace()
    surfaces = namespace["_FR13_FIXED32_GDN_BV_SURFACES"]
    state = {name: f"served:{name}".encode("ascii") for name in surfaces}
    served = dict(state)

    def snapshot():
        return dict(state)

    def restore(value):
        state.clear()
        state.update(value)

    def run(block_v: int):
        for name in surfaces:
            state[name] = f"gate:{name}".encode("ascii")
        return {
            "block_v": block_v,
            "launch_key": ("tree_gdn_path", block_v),
            "output": b"reference" if block_v == 8 else b"candidate",
        }

    record = {
        "snapshot": snapshot,
        "restore": restore,
        "run": run,
        "byte_equal": lambda left, right: left == right,
        "surface_names": surfaces,
    }
    with pytest.raises(RuntimeError, match="byte mismatch.*output"):
        namespace["_fr13_fixed32_gdn_bv_compare_records"]((record,), 16)
    assert state == served


def test_live_gate_rejects_false_stock_vs_stock() -> None:
    namespace = _kernel_gate_namespace()
    surfaces = namespace["_FR13_FIXED32_GDN_BV_SURFACES"]
    state = {name: b"served" for name in surfaces}

    def run(_block_v: int):
        return {
            "block_v": 8,
            "launch_key": ("tree_gdn_path", 8),
            "output": b"same",
        }

    record = {
        "snapshot": lambda: dict(state),
        "restore": lambda value: state.update(value),
        "run": run,
        "byte_equal": lambda left, right: left == right,
        "surface_names": surfaces,
    }
    with pytest.raises(RuntimeError, match="false stock-vs-stock"):
        namespace["_fr13_fixed32_gdn_bv_compare_records"]((record,), 16)
    with pytest.raises(RuntimeError, match="refused stock-vs-stock"):
        namespace["_fr13_fixed32_gdn_bv_compare_records"]((record,), 8)


def test_live_gate_executes_only_on_first_measured_graph_replay() -> None:
    namespace = _kernel_gate_namespace()
    calls: list[tuple[int, int]] = []
    records = tuple(object() for _ in range(48))
    graph_signature = "a" * 64
    census_graph_signature = "b" * 64
    namespace.update(
        {
            "torch": SimpleNamespace(
                cuda=SimpleNamespace(
                    is_available=lambda: False,
                    is_current_stream_capturing=lambda: False,
                )
            ),
            "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE": 64,
            "_FR13_FIXED32_GDN_BV_CAPTURES": {
                (1, 101, graph_signature): {
                    "batch_size": 1,
                    "graph_id": 101,
                    "graph_signature": graph_signature,
                    "records": records,
                }
            },
            "_FR13_FIXED32_GDN_BV_LIVE_STATE": {
                "status": "armed",
                "candidate_bv": 64,
                "graph_id": None,
                "batch_size": None,
                "records": 0,
            },
        }
    )

    def compare(actual_records, candidate_bv: int):
        calls.append((len(actual_records), candidate_bv))
        return {
            "records": len(actual_records),
            "reference_bv": 8,
            "candidate_bv": candidate_bv,
        }

    namespace["_fr13_fixed32_gdn_bv_compare_records"] = compare
    namespace["_fr13_fixed32_gdn_bv_real_event_marker"] = (
        lambda: "swe_verified:django__django-12345"
    )
    emitted = []
    namespace["_fr13_fixed32_gdn_bv_live_pass_emit"] = (
        lambda **payload: emitted.append(payload)
    )
    gate = namespace["fixed32_gdn_bv_live_gate_on_replay"]

    first = gate(101, graph_signature, census_graph_signature, 1, 48)
    second = gate(101, graph_signature, census_graph_signature, 1, 48)

    assert calls == [(48, 64)]
    assert emitted == [
        {
            "task_marker": "swe_verified:django__django-12345",
            "batch_size": 1,
            "graph_signature": census_graph_signature,
            "result": {
                "records": 48,
                "reference_bv": 8,
                "candidate_bv": 64,
            },
        }
    ]
    assert first == second == {
        "status": "passed",
        "candidate_bv": 64,
        "graph_id": 101,
        "graph_signature": graph_signature,
        "batch_size": 1,
        "records": 48,
    }


def test_candidate_resolver_is_exact_fixed32_bv8_only(tmp_path: Path) -> None:
    namespace = _kernel_gate_namespace()
    resolve = namespace["_fr13_resolve_fixed32_gdn_path_bv_candidate"]

    for candidate in (16, 32, 64, 128):
        assert (
            resolve(
                "tail6_fixed32",
                environ={
                    "FR13_FIXED32_GDN_PATH_BV_CANDIDATE": str(candidate)
                },
                sidecars=(),
                geom_override={"BV": 8},
            )
            == candidate
        )
    assert resolve(
        "tail6_fixed32",
        environ={
            "FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "gqa_group3_bv16",
            "FR13_DRAFT_VOCAB_K": "65536",
            "FR13_DRAFT_VOCAB_ROOT": "1",
        },
        sidecars=(),
        geom_override={"BV": 8},
    ) == "gqa_group3_bv16"
    with pytest.raises(RuntimeError, match="expected one of"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "8"},
            sidecars=(),
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="requires an exact fixed32 mode"):
        resolve(
            None,
            environ={"FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "16"},
            sidecars=(),
            geom_override={"BV": 8},
        )
    with pytest.raises(RuntimeError, match="pinned exactly"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "16"},
            sidecars=(),
            geom_override={"BV": 8, "num_warps": 4},
        )

    sidecar = tmp_path / "candidate.flag"
    sidecar.write_text("32\n", encoding="ascii")
    with pytest.raises(RuntimeError, match="conflicting sources"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "16"},
            sidecars=(str(sidecar),),
            geom_override={"BV": 8},
        )


@pytest.mark.parametrize("candidate", (16, 32, 64, 128))
def test_production_resolver_requires_same_candidate_source_and_geometry(
    tmp_path: Path,
    candidate: int,
) -> None:
    namespace = _kernel_gate_namespace()
    resolve = namespace["_fr13_resolve_fixed32_gdn_path_bv_production"]
    live_pass = tmp_path / "pass.json"
    live_pass.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.gdn_path_bv.live_pass.v1",
                "status": "pass",
                "candidate": "fixed32_gdn_path_bv_v1",
                "source_sha256": "a" * 64,
                "task_marker": "swe_verified:django__django-12345",
                "mode": "tail6_fixed32",
                "batch_size": 4,
                "covered_batches": [1, 2, 3, 4],
                "records": 192,
                "physical_rows": 32,
                "reference_bv": 8,
                "candidate_bv": candidate,
                "raw_byte_equal": True,
                "reference_served": True,
                "state_restored": True,
            }
        ),
        encoding="ascii",
    )
    payload = resolve(
        "tail6_fixed32",
        environ={"FR13_FIXED32_GDN_PATH_BV_PRODUCTION": str(candidate)},
        sidecars=(),
        geom_override={"BV": candidate},
        pass_path=str(live_pass),
        source_sha256="a" * 64,
    )
    assert payload is not None
    assert payload["candidate_bv"] == candidate
    assert payload["covered_batches"] == [1, 2, 3, 4]

    with pytest.raises(RuntimeError, match="geometry pinned exactly"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_PRODUCTION": str(candidate)},
            sidecars=(),
            geom_override={"BV": 8},
            pass_path=str(live_pass),
            source_sha256="a" * 64,
        )
    with pytest.raises(RuntimeError, match="different candidate/source"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_PRODUCTION": str(candidate)},
            sidecars=(),
            geom_override={"BV": candidate},
            pass_path=str(live_pass),
            source_sha256="b" * 64,
        )

    invalid = json.loads(live_pass.read_text(encoding="ascii"))
    invalid["covered_batches"] = [4]
    live_pass.write_text(json.dumps(invalid), encoding="ascii")
    with pytest.raises(RuntimeError, match="different candidate/source"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_PRODUCTION": str(candidate)},
            sidecars=(),
            geom_override={"BV": candidate},
            pass_path=str(live_pass),
            source_sha256="a" * 64,
        )


def test_served_launch_stays_bv8_and_live_gate_is_first_replay_hooked() -> None:
    kernel = KERNEL_PATH.read_text(encoding="utf-8")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")

    launch_start = kernel.index("def launch_tree_gdn_prepared(")
    launch = kernel[launch_start:]
    assert "_path_block_v=_bv" in launch
    assert "triton.cdiv(dim_v, _path_block_v)" in launch
    assert "BLOCK_V=_path_block_v" in launch
    assert "_launch_paths(out)" in launch
    assert "_path_block_v=_gate_bv" in launch
    assert "reference = run(8)" in kernel
    assert "candidate_result = run(candidate)" in kernel

    assert "fixed32_gdn_bv_live_capture_begin(identity, batch)" in patcher
    assert "fixed32_gdn_bv_live_capture_end(" in patcher
    assert "fixed32_gdn_bv_live_gate_on_replay(" in patcher
    replay = patcher.index("fixed32_gdn_bv_live_gate_on_replay(")
    observed = patcher.index("def _fr13_fixed32_observed_graph_replay(")
    assert observed < replay

    assert "must be exactly 16, 32, 64, or 128" in launcher
    assert "requires FR13_FIXED32_MODE" in launcher
    assert "fr13_fixed32_gdn_path_bv_candidate.flag" in launcher
    assert "_fr13_fixed32_expected_gdn_geom" in launcher
    assert "FR13_FIXED32_GDN_PATH_BV_PRODUCTION" in launcher
    assert "requires a regular live PASS JSON" in launcher
    assert "path-BV diagnostic and production selectors" in launcher
    assert "FR13 fixed32 GDN BV production geometry/PASS contract drift" in kernel
    assert "no fallback is permitted" in kernel
    assert "_fixed32_path_bv_production" in launch
    assert (
        "_bv_cap <= 8 or _fixed32_path_bv_production" in launch
    )
    assert 'int(staging_rows) not in production_pass["covered_batches"]' in kernel
    assert "_launch_paths(out)" in launch
