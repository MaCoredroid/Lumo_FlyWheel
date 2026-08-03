from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
KERNEL = ROOT / "src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py"
PATCHER = ROOT / "scripts/fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER = ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"


def _namespace(*function_names: str) -> dict[str, object]:
    tree = ast.parse(KERNEL.read_text(encoding="utf-8"))
    assignment_names = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_PATH_BV_SIDECARS",
        "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION_SIDECARS",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_GATE_VALUE",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_CANDIDATE_ID",
        "_FR13_FIXED32_GDN_PATH_BV_LIVE_PASS",
        "_FR13_FIXED32_GDN_BV_SURFACES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_SURFACES",
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES",
    }
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in assignment_names
            for target in node.targets
        )
    ]
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in definitions} == set(function_names)
    namespace: dict[str, object] = {"json": json, "os": os}
    exec(
        compile(
            ast.Module(body=[*assignments, *definitions], type_ignores=[]),
            KERNEL,
            "exec",
        ),
        namespace,
    )
    return namespace


def _record(*, mismatch: str | None = None, identity_tamper: bool = False):
    namespace = _namespace("_fr13_fixed32_gdn_single_launch_compare_records")
    state_names = namespace[
        "_FR13_FIXED32_GDN_SINGLE_LAUNCH_STATE_SURFACES"
    ]
    state = {name: f"served:{name}".encode("ascii") for name in state_names}
    served = dict(state)
    calls: list[str] = []

    def snapshot():
        return dict(state)

    def restore(value):
        state.clear()
        state.update(value)

    def run(candidate: str):
        calls.append(candidate)
        single = candidate != "reference"
        for name in state_names:
            state[name] = (
                b"candidate-mismatch"
                if single and name == mismatch
                else f"equal:{name}".encode("ascii")
            )
        return {
            "candidate": (
                "tampered"
                if identity_tamper and single
                else (
                    "fixed32_gdn_single_launch_tree_v2"
                    if single
                    else "fixed32_gdn_two_launch_reference_v1"
                )
            ),
            "physical_launches": 1 if single else 2,
            "output": (
                b"candidate-mismatch"
                if single and mismatch == "direct_output"
                else b"equal-output"
            ),
        }

    return namespace, state, served, calls, {
        "snapshot": snapshot,
        "restore": restore,
        "run": run,
        "byte_equal": lambda left, right: left == right,
        "surface_names": state_names,
    }


def test_single_launch_comparator_runs_distinct_arms_and_restores_stock() -> None:
    namespace, state, served, calls, record = _record()
    result = namespace["_fr13_fixed32_gdn_single_launch_compare_records"](
        (record,)
    )

    assert calls == ["reference", "fixed32_gdn_single_launch_tree_v2"]
    assert result == {
        "records": 1,
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches": 2,
        "candidate_physical_launches": 1,
    }
    assert state == served


@pytest.mark.parametrize(
    ("mismatch", "message"),
    (("direct_output", "byte mismatch.*output"), ("ring_k", "byte mismatch.*ring_k")),
)
def test_single_launch_comparator_fails_closed_and_restores(
    mismatch: str, message: str
) -> None:
    namespace, state, served, _calls, record = _record(mismatch=mismatch)
    with pytest.raises(RuntimeError, match=message):
        namespace["_fr13_fixed32_gdn_single_launch_compare_records"]((record,))
    assert state == served


def test_single_launch_comparator_rejects_candidate_identity_tamper() -> None:
    namespace, state, served, _calls, record = _record(identity_tamper=True)
    with pytest.raises(RuntimeError, match="launch identity drift"):
        namespace["_fr13_fixed32_gdn_single_launch_compare_records"]((record,))
    assert state == served


@pytest.mark.parametrize("mode", ("tail6_fixed32", "hydra27_fixed32"))
def test_gate_resolver_is_exact_physical32_bv8_k64_root1(mode: str) -> None:
    namespace = _namespace("_fr13_resolve_fixed32_gdn_path_bv_candidate")
    resolve = namespace["_fr13_resolve_fixed32_gdn_path_bv_candidate"]
    exact = {
        "FR13_FIXED32_GDN_PATH_BV_CANDIDATE": "single_launch",
        "FR13_DRAFT_VOCAB_K": "65536",
        "FR13_DRAFT_VOCAB_ROOT": "1",
    }
    assert resolve(
        mode, environ=exact, sidecars=(), geom_override={"BV": 8}
    ) == "single_launch"

    for field, value in (
        ("FR13_DRAFT_VOCAB_K", "32768"),
        ("FR13_DRAFT_VOCAB_ROOT", "0"),
    ):
        tampered = dict(exact)
        tampered[field] = value
        with pytest.raises(RuntimeError, match="K64/root1"):
            resolve(
                mode,
                environ=tampered,
                sidecars=(),
                geom_override={"BV": 8},
            )
    with pytest.raises(RuntimeError, match="pinned exactly"):
        resolve(mode, environ=exact, sidecars=(), geom_override={"BV": 16})
    with pytest.raises(RuntimeError, match="exact fixed32 mode"):
        resolve(None, environ=exact, sidecars=(), geom_override={"BV": 8})


def test_b1_b4_capture_counts_are_closed_over_real_graph_records() -> None:
    namespace = _namespace("fixed32_gdn_bv_live_capture_end")
    end = namespace["fixed32_gdn_bv_live_capture_end"]
    namespace["_FR13_FIXED32_GDN_PATH_BV_CANDIDATE"] = "single_launch"
    namespace["_FR13_FIXED32_GDN_BV_CAPTURES"] = {}

    for graph_id, batch in ((101, 1), (404, 4)):
        records = [object() for _ in range(48)]
        namespace["_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT"] = {
            "graph_id": graph_id,
            "batch_size": batch,
            "records": records,
        }
        end(graph_id, batch, 48)
        assert len(namespace["_FR13_FIXED32_GDN_BV_CAPTURES"][graph_id]["records"]) == 48

    namespace["_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT"] = {
        "graph_id": 405,
        "batch_size": 4,
        "records": [object() for _ in range(192)],
    }
    with pytest.raises(RuntimeError, match="capture end drift"):
        end(405, 4, 192)


def test_single_launch_gate_ignores_unqualified_b2_b3_graph_shapes() -> None:
    namespace = _namespace(
        "fixed32_gdn_bv_live_capture_begin",
        "fixed32_gdn_bv_live_capture_end",
    )
    namespace["_FR13_FIXED32_MODE"] = "hydra27_fixed32"
    namespace["_FR13_FIXED32_GDN_PATH_BV_CANDIDATE"] = "single_launch"
    namespace["_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT"] = None
    namespace["_FR13_FIXED32_GDN_BV_CAPTURES"] = {}

    for batch in (2, 3):
        namespace["fixed32_gdn_bv_live_capture_begin"](100 + batch, batch)
        assert namespace["_FR13_FIXED32_GDN_BV_CAPTURE_CONTEXT"] is None
        namespace["fixed32_gdn_bv_live_capture_end"](100 + batch, batch, 48)
    assert namespace["_FR13_FIXED32_GDN_BV_CAPTURES"] == {}


@pytest.mark.parametrize(
    ("mode", "batch", "topology"),
    (
        ("tail6_fixed32", 1, "Tail23"),
        ("hydra27_fixed32", 4, "Hydra27"),
    ),
)
def test_live_pass_is_source_mode_batch_and_reference_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    batch: int,
    topology: str,
) -> None:
    namespace = _namespace("_fr13_fixed32_gdn_bv_live_pass_emit")
    namespace["_FR13_FIXED32_MODE"] = mode
    namespace["_fr13_fixed32_gdn_path_bv_source_sha256"] = lambda: "a" * 64
    output = tmp_path / "single-launch-pass.json"
    monkeypatch.setenv("FR13_FIXED32_GDN_PATH_BV_LIVE_JSON", os.fspath(output))

    namespace["_fr13_fixed32_gdn_bv_live_pass_emit"](
        task_marker="swe_verified:astropy__astropy-12907",
        batch_size=batch,
        result={
            "records": 48,
            "candidate": "fixed32_gdn_single_launch_tree_v2",
            "reference_bv": 8,
            "candidate_bv": 8,
            "reference_physical_launches": 2,
            "candidate_physical_launches": 1,
        },
    )
    payload = json.loads(output.read_text(encoding="ascii"))
    assert payload["schema"] == "fr13.fixed32.gdn_single_launch.live_pass.v1"
    assert payload["candidate"] == "fixed32_gdn_single_launch_tree_v2"
    assert payload["source_sha256"] == "a" * 64
    assert payload["mode"] == mode
    assert payload["logical_topology"] == topology
    assert payload["covered_batches"] == [batch]
    assert payload["physical_rows"] == 32
    assert payload["draft_vocab_k"] == 65536
    assert payload["draft_vocab_root"] == 1
    assert payload["reference_served"] is True
    assert payload["state_restored"] is True
    assert payload["production_eligible"] is False


def test_route_is_stock_serving_and_production_cannot_accept_gate_value() -> None:
    kernel = KERNEL.read_text(encoding="utf-8")
    patcher = PATCHER.read_text(encoding="utf-8")
    launcher = LAUNCHER.read_text(encoding="utf-8")

    assert 'return "single_launch_gate" if batch == 4 else None' in kernel
    assert '_launch_reference(collect_export=False)\n        return out, None' in kernel
    assert '_single_launch_override=True' in kernel
    assert '"reference_served": True' in kernel
    assert '"state_restored": True' in kernel
    assert '"fixed32_gdn_single_launch_tree_v2"' in kernel
    assert 'candidate_raw == "single_launch"' in patcher
    assert 'candidate == "single_launch"' in patcher
    assert '"FR13_TREE_GDN_GEOM_OVERRIDE": "BV=8"' in patcher
    assert '|| "$_fr13_gdn_path_bv_candidate" == "single_launch"' in launcher
    assert "single-launch GDN live gate requires exact K64/root1" in launcher
    assert launcher.count("fr13_fixed32_gdn_single_launch_tree.arm") == 2
    assert "-e FR13_FIXED32_GDN_SINGLE_LAUNCH_TREE=" not in launcher

    production_section = kernel[
        kernel.index("def _fr13_resolve_fixed32_gdn_path_bv_production(") :
        kernel.index("def _fr13_fixed32_gdn_bv_real_event_marker(")
    ]
    assert 'if value not in ("16", "32", "64", "128")' in production_section
    assert "single_launch" not in production_section

    namespace = _namespace("_fr13_resolve_fixed32_gdn_path_bv_production")
    with pytest.raises(RuntimeError, match="requires one of"):
        namespace["_fr13_resolve_fixed32_gdn_path_bv_production"](
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_PATH_BV_PRODUCTION": "single_launch"},
            sidecars=(),
            geom_override={"BV": 8},
            pass_path="/does/not/matter",
            source_sha256="a" * 64,
        )
