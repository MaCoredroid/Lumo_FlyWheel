from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
KERNEL_PATH = (
    ROOT / "src" / "lumo_flywheel_serving" / "fr10_gdn_tree_kernel.py"
)
PATCHER_PATH = ROOT / "scripts" / "fr10_phase4_patch_vllm_tree_gdn.py"
LAUNCHER_PATH = ROOT / "scripts" / "fr13_launch_forked_fa2_tree_server.sh"
BIGDENOM_PATH = ROOT / "scripts" / "fr13_bigdenom_swe_serve_variant.sh"
RUNNER_PATH = ROOT / "scripts" / "run_swe_bench_q36_a.py"
TIMING_PATH = ROOT / "scripts" / "fr13_run_b1_gdn_level0_coeff_timing.sh"
BUILD_PATH = ROOT / "scripts" / "fr13_build_gdn_level0_coeff_sm121.py"
MANIFEST_PATH = ROOT / "scripts" / "fr13_runtime_manifest.py"
BUILD_ARTIFACT = (
    ROOT
    / "results"
    / "fr13_fixed32_gdn_level0_coeff_sm121_build_review_20260801"
)


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _load_layout_function():
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = _function(tree, "_fr13_fixed32_level0_coeff_layout")
    module = ast.Module(
        body=[
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Constant(value=value),
            )
            for name, value in (
                ("QK_HEADS", 16),
                ("V_HEADS", 48),
                ("V", 128),
                ("K", 128),
                ("_FR13_FIXED32_COEFF_SCRATCH_ROWS", 32),
                ("_FR13_FIXED32_EXPORT_SLOTS", 5),
            )
        ]
        + [function],
        type_ignores=[],
    )
    namespace: dict[str, object] = {}
    exec(
        compile(ast.fix_missing_locations(module), "<level0-coeff-layout>", "exec"),
        namespace,
    )
    return namespace["_fr13_fixed32_level0_coeff_layout"]


def _gate_namespace() -> dict[str, object]:
    tree = ast.parse(KERNEL_PATH.read_text(encoding="utf-8"))
    assignment_names = {
        "_FR13_FIXED32_MODES",
        "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_SIDECARS",
        "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_PASS_PATH",
        "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_ENGAGEMENT",
        "_FR13_FIXED32_GDN_LEVEL0_COEFF_CANDIDATE_ID",
        "_FR13_FIXED32_GDN_BV_SURFACES",
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
    function_names = {
        "_fr13_resolve_fixed32_gdn_level0_coeff_production",
        "_fr13_fixed32_gdn_level0_coeff_compare_records",
        "fixed32_gdn_level0_coeff_production_on_replay",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in function_names
    ]
    assert {node.name for node in functions} == function_names
    namespace: dict[str, object] = {
        "Path": Path,
        "hashlib": hashlib,
        "json": json,
        "os": os,
    }
    exec(
        compile(
            ast.Module(body=[*assignments, *functions], type_ignores=[]),
            KERNEL_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace


def _build_audit_namespace() -> dict[str, object]:
    tree = ast.parse(BUILD_PATH.read_text(encoding="utf-8"))
    names = {"_sass_spill_instructions", "_resource_bytes"}
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in functions} == names
    namespace = {"re": re}
    exec(
        compile(
            ast.Module(body=functions, type_ignores=[]),
            BUILD_PATH,
            "exec",
        ),
        namespace,
    )
    return namespace


def _timing_closeout_source() -> str:
    source = TIMING_PATH.read_text(encoding="utf-8")
    code = source.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(code, TIMING_PATH, "exec")
    return code


def _load_pass_module():
    path = ROOT / "scripts" / "fr13_gdn_level0_coeff_pass.py"
    spec = importlib.util.spec_from_file_location("fr13_gdn_level0_coeff_pass", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_level0_coefficient_scratch_is_disjoint_for_b1_and_b4() -> None:
    layout_fn = _load_layout_function()
    common = {
        "n_actual": 32,
        "num_kh": 16,
        "num_vh": 48,
        "dim_v": 128,
        "dim_k": 128,
        "export_or_mask": 16915,
    }
    b1 = layout_fn(batch_size=1, compact_export=False, **common)
    b4 = layout_fn(batch_size=4, compact_export=True, **common)

    row_elems = 48 * 128 * 128
    payload_elems = 2 * 32 * 16 * 128 + 2 * 32 * 48
    assert b1 == {
        "scratch_row_start": 31,
        "scratch_rows": 1,
        "row_elems": row_elems,
        "q_offset": 0,
        "k_offset": 32 * 16 * 128,
        "decay_offset": 2 * 32 * 16 * 128,
        "beta_offset": 2 * 32 * 16 * 128 + 32 * 48,
        "payload_elems": payload_elems,
        "capacity_elems": row_elems,
    }
    assert b4["scratch_row_start"] == 28
    assert b4["scratch_rows"] == 4
    assert b4["payload_elems"] == payload_elems
    assert b4["scratch_row_start"] >= 4 * 5
    assert not (16915 & (1 << b1["scratch_row_start"]))

    with pytest.raises(ValueError, match="B1-B4 production geometry"):
        layout_fn(batch_size=5, compact_export=True, **common)


def test_level0_coefficient_staging_keeps_two_launches_and_fixed32_nodes() -> None:
    source = KERNEL_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    stage = ast.get_source_segment(
        source, _function(tree, "_fixed32_gdn_stage_coefficients")
    )
    node_step = ast.get_source_segment(source, _function(tree, "_gdn_node_step"))
    path = ast.get_source_segment(source, _function(tree, "_tree_gdn_path_kernel"))
    batch_path = ast.get_source_segment(
        source, _function(tree, "_tree_gdn_path_kernel_fixed32_batch")
    )
    launch = ast.get_source_segment(
        source, _function(tree, "launch_tree_gdn_prepared")
    )
    batch_launch = ast.get_source_segment(
        source, _function(tree, "launch_tree_gdn_prepared_fixed32_batch")
    )
    assert all(
        segment is not None
        for segment in (stage, node_step, path, batch_path, launch, batch_launch)
    )

    programs = 48 * (128 // 8)
    qk_pairs = 32 * 16
    scalar_pairs = 32 * 48
    assert programs == 768
    assert qk_pairs == 512
    assert scalar_pairs == 2 * programs

    baseline_node_programs = 32 * programs
    candidate_qk_programs = 5 * programs + qk_pairs
    candidate_gate_programs = 5 * programs + scalar_pairs
    assert baseline_node_programs == 24576
    assert candidate_qk_programs == 4352
    assert candidate_gate_programs == 5376
    assert 1 - candidate_qk_programs / baseline_node_programs > 0.82
    assert 1 - candidate_gate_programs / baseline_node_programs > 0.78

    assert "b_q = b_q * tl.rsqrt(tl.sum(b_q * b_q) + 1e-6)" in stage
    assert "b_k = b_k * tl.rsqrt(tl.sum(b_k * b_k) + 1e-6)" in stage
    assert "decay = tl.exp(-tl.exp(b_a_log) * softplus_x)" in stage
    assert "b_beta = tl.sigmoid(b_raw_b)" in stage
    assert "if RAW_GATING and not COEFFICIENTS_PRECOMPUTED" in node_step
    assert "PRECOMPUTE_LEVEL1" in path
    assert "LOAD_PRECOMPUTED" in path
    assert "PRECOMPUTE_LEVEL1" in batch_path
    assert "LOAD_PRECOMPUTED" in batch_path
    assert "PRECOMPUTE_LEVEL1=_use_level0_coeff and (_li == 0)" in launch
    assert "LOAD_PRECOMPUTED=_use_level0_coeff and (_li == 1)" in launch
    assert "for level_index" in batch_launch
    assert "launches_per_layer=2" in launch
    assert "launches_per_layer=2" in batch_launch


def test_live_gate_compares_every_non_scratch_surface_and_restores() -> None:
    namespace = _gate_namespace()
    surfaces = namespace["_FR13_FIXED32_GDN_BV_SURFACES"]
    state = {
        "export": torch.zeros((32, 1, 1, 1), dtype=torch.float32),
        **{
            name: torch.zeros((), dtype=torch.int32)
            for name in surfaces
            if name != "export"
        },
    }
    baseline = {name: tensor.clone() for name, tensor in state.items()}

    def snapshot():
        return {name: tensor.clone() for name, tensor in state.items()}

    def restore(value):
        for name, tensor in value.items():
            state[name].copy_(tensor)

    def run(_block_v: int, candidate: bool):
        state["export"][:31].fill_(3)
        state["export"][31].fill_(9 if candidate else 5)
        for name in surfaces[1:]:
            state[name].fill_(7)
        return {
            "block_v": 8,
            "count_invocation": False,
            "launch_key": ("tree_gdn_path", 8, candidate),
            "output": torch.full((), 11, dtype=torch.int32),
        }

    result = namespace["_fr13_fixed32_gdn_level0_coeff_compare_records"](
        (
            {
                "snapshot": snapshot,
                "restore": restore,
                "run": run,
                "byte_equal": torch.equal,
                "surface_names": surfaces,
            },
        )
    )
    assert result == {
        "records": 1,
        "compared_bytes": 152,
        "scratch_row_start": 31,
        "scratch_rows": 1,
        "count_invocation": False,
    }
    assert all(torch.equal(state[name], baseline[name]) for name in surfaces)


def test_live_gate_fails_non_scratch_mismatch_after_restore() -> None:
    namespace = _gate_namespace()
    surfaces = namespace["_FR13_FIXED32_GDN_BV_SURFACES"]
    state = {
        "export": torch.zeros((32, 1, 1, 1), dtype=torch.float32),
        **{
            name: torch.zeros((), dtype=torch.int32)
            for name in surfaces
            if name != "export"
        },
    }
    baseline = {name: tensor.clone() for name, tensor in state.items()}

    def snapshot():
        return {name: tensor.clone() for name, tensor in state.items()}

    def restore(value):
        for name, tensor in value.items():
            state[name].copy_(tensor)

    def run(_block_v: int, candidate: bool):
        state["export"].fill_(1)
        if candidate:
            state["export"][0].fill_(2)
        return {
            "block_v": 8,
            "count_invocation": False,
            "launch_key": ("tree_gdn_path", 8, candidate),
            "output": torch.ones((), dtype=torch.int32),
        }

    record = {
        "snapshot": snapshot,
        "restore": restore,
        "run": run,
        "byte_equal": torch.equal,
        "surface_names": surfaces,
    }
    with pytest.raises(RuntimeError, match="export_non_scratch_rows"):
        namespace["_fr13_fixed32_gdn_level0_coeff_compare_records"]((record,))
    assert all(torch.equal(state[name], baseline[name]) for name in surfaces)


def test_production_resolver_is_source_mode_and_pass_bound(
    tmp_path: Path,
) -> None:
    namespace = _gate_namespace()
    live_pass = tmp_path / "live.json"
    payload = {
        "schema": "fr13.fixed32.gdn_level0_coeff.live_pass.v1",
        "status": "pass",
        "candidate": "fixed32_gdn_level0_coeff_v1",
        "source_sha256": "a" * 64,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "mode": "hydra27_fixed32",
        "batch_size": 1,
        "covered_batches": [1],
        "records": 48,
        "physical_rows": 32,
        "path_lengths": [5, 7],
        "launches_per_layer": 2,
        "scratch_row_start": 31,
        "scratch_rows": 1,
        "count_invocation": False,
        "compared_bytes": 1234,
        "raw_byte_equal": True,
        "scratch_contained": True,
        "reference_served": True,
        "state_restored": True,
    }
    raw = json.dumps(payload, sort_keys=True).encode("ascii")
    live_pass.write_bytes(raw)
    resolve = namespace[
        "_fr13_resolve_fixed32_gdn_level0_coeff_production"
    ]
    result = resolve(
        "hydra27_fixed32",
        environ={"FR13_FIXED32_GDN_LEVEL0_COEFF": "1"},
        sidecars=(),
        geom_override={"BV": 8},
        pass_path=str(live_pass),
        source_sha256="a" * 64,
    )
    assert result["production_pass_sha256"] == hashlib.sha256(raw).hexdigest()

    with pytest.raises(RuntimeError, match="different candidate/source/mode"):
        resolve(
            "tail6_fixed32",
            environ={"FR13_FIXED32_GDN_LEVEL0_COEFF": "1"},
            sidecars=(),
            geom_override={"BV": 8},
            pass_path=str(live_pass),
            source_sha256="a" * 64,
        )
    with pytest.raises(RuntimeError, match="geometry pinned exactly"):
        resolve(
            "hydra27_fixed32",
            environ={"FR13_FIXED32_GDN_LEVEL0_COEFF": "1"},
            sidecars=(),
            geom_override={"BV": 16},
            pass_path=str(live_pass),
            source_sha256="a" * 64,
        )


def test_host_pass_requires_the_complete_b1_compared_byte_closure() -> None:
    pass_module = _load_pass_module()
    payload = {
        "schema": pass_module.SCHEMA,
        "status": "pass",
        "candidate": pass_module.CANDIDATE,
        "source_sha256": "a" * 64,
        "task_marker": "swe_verified:astropy__astropy-12907",
        "mode": "hydra27_fixed32",
        "batch_size": 1,
        "covered_batches": [1],
        "records": 48,
        "physical_rows": 32,
        "path_lengths": [5, 7],
        "launches_per_layer": 2,
        "scratch_row_start": 31,
        "scratch_rows": 1,
        "count_invocation": False,
        "non_scratch_export_rows_compared": 31,
        "surfaces": pass_module.SURFACES,
        "raw_byte_equal": True,
        "scratch_contained": True,
        "reference_served": True,
        "state_restored": True,
        "compared_bytes": pass_module.EXPECTED_COMPARED_BYTES,
    }
    assert pass_module.validate(
        payload,
        source_sha256="a" * 64,
        expected_task_id="astropy__astropy-12907",
        expected_mode="hydra27_fixed32",
    ) is payload
    payload["compared_bytes"] -= 4
    with pytest.raises(pass_module.PassError, match="compared-byte closure drift"):
        pass_module.validate(
            payload,
            source_sha256="a" * 64,
            expected_task_id="astropy__astropy-12907",
            expected_mode="hydra27_fixed32",
        )


def test_production_engagement_is_first_replay_source_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _gate_namespace()
    engagement_path = tmp_path / "engagement.json"
    monkeypatch.setenv(
        "FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_ENGAGEMENT_PATH",
        str(engagement_path),
    )
    namespace.update(
        {
            "_FR13_FIXED32_MODE": "hydra27_fixed32",
            "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_PASS": {
                "source_sha256": "a" * 64,
                "production_pass_sha256": "b" * 64,
                "task_marker": "swe_verified:astropy__astropy-12907",
                "mode": "hydra27_fixed32",
                "covered_batches": [1],
                "count_invocation": False,
            },
            "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_STATE": {
                "status": "armed",
                "graph_id": None,
                "graph_signature": None,
                "batch_size": None,
                "records": 0,
            },
            "_fr13_fixed32_gdn_path_bv_source_sha256": lambda: "a" * 64,
        }
    )
    engage = namespace["fixed32_gdn_level0_coeff_production_on_replay"]
    first = engage(17, "c" * 64, 1, 48)
    second = engage(17, "c" * 64, 1, 48)
    assert first == second
    assert first["status"] == "engaged"
    artifact = json.loads(engagement_path.read_text(encoding="ascii"))
    assert artifact["status"] == "ENGAGED"
    assert artifact["production_pass_sha256"] == "b" * 64
    assert artifact["count_invocation"] is False
    assert artifact["fallback"] == 0


def test_gate_and_timing_hooks_cross_the_real_swe_runner_boundary() -> None:
    kernel = KERNEL_PATH.read_text(encoding="utf-8")
    patcher = PATCHER_PATH.read_text(encoding="utf-8")
    launcher = LAUNCHER_PATH.read_text(encoding="utf-8")
    bigdenom = BIGDENOM_PATH.read_text(encoding="utf-8")
    runner = RUNNER_PATH.read_text(encoding="utf-8")
    timing = TIMING_PATH.read_text(encoding="utf-8")
    manifest = MANIFEST_PATH.read_text(encoding="utf-8")

    assert "reference = run(8, False)" in kernel
    assert "candidate = run(8, True)" in kernel
    assert '"count_invocation": bool(_count)' in kernel
    assert "reference_export[:31]" in kernel
    assert "fixed32_gdn_level0_coeff_live_gate_on_replay(" in patcher
    assert "fixed32_gdn_level0_coeff_production_on_replay(" in patcher
    assert "fr13_gdn_level0_coeff_pass.py" in launcher
    assert "fr13_fixed32_gdn_level0_coeff.production_pass.json" in launcher
    assert "fr13_fixed32_gdn_level0_coeff.real_event.arm" in bigdenom
    assert "--fixed32-gdn-coeff-real-event-arm" in runner
    assert "subset_b4_four.json" in timing
    assert "FR13_DRAFT_VOCAB_K=65536" in timing
    assert "measured_tps_fullstep_wall" in timing
    assert "sfwd_gpu_ms_per_step" in timing
    assert "dfwd_gpu_ms_per_step" in timing
    assert "cfwd_gpu_ms_per_step" in timing
    assert "other_wall_ms_per_step" in timing
    assert "tps_from_committed_tokens_and_wall" in timing
    assert "runtime_manifest_sha256" in timing
    assert "production_engagement_sha256" in timing
    build = BUILD_PATH.read_text(encoding="utf-8")
    assert '"builder_sha256"' in build
    assert 'artifact_hashes["resources"]' in build
    assert 'artifact_hashes["sass"]' in build
    assert '"COUNT_INVOCATION": False' in build
    for path in (
        "scripts/fr13_gdn_level0_coeff_pass.py",
        "scripts/fr13_run_b1_gdn_level0_coeff_live_gate.sh",
        "scripts/fr13_run_b1_gdn_level0_coeff_timing.sh",
    ):
        assert repr(path) in manifest or f'"{path}"' in manifest


def test_offline_resource_audit_parses_addressed_sass_and_stack() -> None:
    namespace = _build_audit_namespace()
    spill_lines = namespace["_sass_spill_instructions"]
    resource_bytes = namespace["_resource_bytes"]
    sample = """
        /*0010*/                   LDL R3, [R2] ;
        /*0020*/              @!P0 STL.64 [R4], R6 ;
        /*0030*/                   LDG.E R7, desc[UR4][R8.64] ;
    """
    assert len(spill_lines(sample)) == 2
    assert resource_bytes("REG:48 STACK:0 LOCAL:0", "STACK") == [0]
    assert resource_bytes("REG:48 STACK:0 LOCAL:0", "LOCAL") == [0]

    sass_paths = list(BUILD_ARTIFACT.glob("*.sass.txt"))
    resource_paths = list(BUILD_ARTIFACT.glob("*.resources.txt"))
    assert len(sass_paths) == 8
    assert len(resource_paths) == 8
    for sass_path in sass_paths:
        assert not spill_lines(sass_path.read_text(encoding="ascii")), sass_path.name
    for resource_path in resource_paths:
        resources = resource_path.read_text(encoding="ascii")
        assert not any(resource_bytes(resources, "STACK")), resource_path.name
        assert not any(resource_bytes(resources, "LOCAL")), resource_path.name

    manifest = json.loads(
        (BUILD_ARTIFACT / "build_manifest.json").read_text(encoding="ascii")
    )
    assert manifest["status"] == "pass_zero_spill"
    assert manifest["production_count_invocation"] is False
    assert manifest["source_sha256"] == hashlib.sha256(
        KERNEL_PATH.read_bytes()
    ).hexdigest()
    assert manifest["builder_sha256"] == hashlib.sha256(
        BUILD_PATH.read_bytes()
    ).hexdigest()
    records = {record["name"]: record for record in manifest["specializations"]}
    assert set(records) == {
        f"b{batch}_{arm}_level{level}"
        for batch in (1, 4)
        for arm in ("stock", "candidate")
        for level in (0, 1)
    }
    assert records["b1_stock_level0"]["artifact_sha256"]["cubin"] != records[
        "b1_candidate_level0"
    ]["artifact_sha256"]["cubin"]


def test_timing_closeout_reconciles_units_phases_tps_and_provenance(
    tmp_path: Path,
) -> None:
    task_ids = [
        "astropy__astropy-12907",
        "astropy__astropy-13033",
        "astropy__astropy-13236",
        "astropy__astropy-13398",
    ]
    subset = tmp_path / "subset.json"
    stock_path = tmp_path / "stock.json"
    candidate_path = tmp_path / "candidate.json"
    engagement_path = tmp_path / "engagement.json"
    out_path = tmp_path / "summary.json"
    subset.write_text(json.dumps({"instance_ids": task_ids}), encoding="ascii")

    def measurement(wall_ms: float, sfwd_seconds: float) -> dict[str, object]:
        accepted = 3.0
        return {
            "schema": "fr13.measure.deploy_speed.v1",
            "regime": "deployment",
            "instrument": "OFF",
            "batch_size": 1,
            "n_tasks": 4,
            "task_instance_ids": task_ids,
            "step_wall_ms": wall_ms,
            "wall_s_per_event": wall_ms / 1000.0,
            "events_per_step": 1.0,
            "rows_per_step": 32.0,
            "s_per_fwd_gpu": sfwd_seconds,
            "s_per_fwd_gpu_per_forward": sfwd_seconds,
            "drafter_gpu_ms_per_step": 50.0,
            "committer_gpu_ms_per_step": 20.0,
            "accept_per_event": accepted,
            "measured_tps_fullstep_wall": (accepted + 1.0) * 1000.0 / wall_ms,
            "wall_steps_measured": 100.0,
        }

    stock = measurement(200.0, 0.1)
    candidate = measurement(190.0, 0.09)
    stock_path.write_text(json.dumps(stock), encoding="utf-8")
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    source_sha = "1" * 64
    pass_sha = "2" * 64
    engagement_path.write_text(
        json.dumps(
            {
                "schema": "fr13.fixed32.gdn_level0_coeff.production_engagement.v1",
                "status": "ENGAGED",
                "candidate": "fixed32_gdn_level0_coeff_v1",
                "source_sha256": source_sha,
                "production_pass_sha256": pass_sha,
                "task_marker": "swe_verified:astropy__astropy-12907",
                "mode": "hydra27_fixed32",
                "graph_signature": "3" * 64,
                "batch_size": 1,
                "records": 48,
                "physical_rows": 32,
                "path_lengths": [5, 7],
                "launches_per_layer": 2,
                "scratch_row_start": 31,
                "count_invocation": False,
                "fallback": 0,
                "observed_full_graph_replays_at_least": 1,
            }
        ),
        encoding="ascii",
    )
    arguments = [
        str(subset),
        str(stock_path),
        str(candidate_path),
        str(engagement_path),
        str(out_path),
        "hydra27_fixed32",
        "a" * 40,
        source_sha,
        "4" * 64,
        "5" * 64,
        "6" * 64,
        "7" * 64,
        pass_sha,
        "8" * 64,
        "9" * 64,
        "b" * 64,
        "119.658015414",
        "137.6067177261",
    ]
    completed = subprocess.run(
        [sys.executable, "-", *arguments],
        input=_timing_closeout_source(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    summary = json.loads(out_path.read_text(encoding="ascii"))
    assert summary["stock"]["phase_reconciliation_abs_error_ms"] == 0.0
    assert summary["candidate"]["tps_reconciliation_abs_error"] == 0.0
    assert summary["candidate"]["committed_tokens_per_event"] == 4.0
    assert summary["candidate_id"] == "fixed32_gdn_level0_coeff_v1"
    assert summary["runtime_manifest_sha256"] == "8" * 64

    candidate["measured_tps_fullstep_wall"] += 1.0
    candidate_path.write_text(json.dumps(candidate), encoding="utf-8")
    failed = subprocess.run(
        [sys.executable, "-", *arguments],
        input=_timing_closeout_source(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "full-wall TPS does not reconcile" in failed.stderr


try:
    import triton  # noqa: F401

    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only source hosts
    _TRITON_OK = False


def _byte_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().reshape(-1).view(torch.uint8),
        right.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.mark.skipif(
    not (_TRITON_OK and torch.cuda.is_available()),
    reason="fixed32 level-0 coefficient byte gate requires CUDA + Triton",
)
@pytest.mark.parametrize("batch", [1, 4])
def test_level0_coefficient_route_is_byte_identical(
    monkeypatch: pytest.MonkeyPatch, batch: int
) -> None:
    sys.path.insert(0, str(ROOT / "src"))
    from lumo_flywheel_serving import fr10_gdn_tree_kernel as kernel

    monkeypatch.setenv("FR13_TREE_GDN_GEOM_OVERRIDE", "BV=8")
    monkeypatch.delenv("FR13_SCAN_ALIGN", raising=False)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_MODE", "tail6_fixed32")
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_ROUTE_REQUESTED", True)
    monkeypatch.setattr(kernel, "_FR13_SUBTREE_SELFCHECK_REQUESTED", False)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_CANDIDATE", None)
    monkeypatch.setattr(kernel, "_FR13_FIXED32_GDN_PATH_BV_PRODUCTION", None)
    monkeypatch.setattr(
        kernel,
        "_FR13_FIXED32_GDN_LEVEL0_COEFF_PRODUCTION_PASS",
        {"covered_batches": [batch]},
    )
    monkeypatch.setattr(kernel, "_FR13_FIXED32_BATCH_GDN_BV_PRODUCTION", 8)
    monkeypatch.setattr(kernel, "fixed32_batch_gdn_selector", lambda _batch: "production")
    monkeypatch.setattr(
        kernel,
        "_fr13_fixed32_batch_gdn_bv8_production_capture_register",
        lambda **_kwargs: None,
    )
    armed = {"value": False}
    monkeypatch.setattr(
        kernel, "fixed32_gdn_level0_coeff_on", lambda: armed["value"]
    )
    kernel._FR13_SUBTREE_CACHE.clear()

    device = torch.device("cuda", torch.cuda.current_device())
    n_actual = n_pad = 32
    num_kh, num_vh = 16, 48
    dim_k = dim_v = 128
    rows = batch * n_actual
    torch.manual_seed(13050 + batch)
    kernel.subtree_preseed(
        kernel._FR13_FIXED32_PARENT,
        n_actual,
        num_vh,
        dim_v,
        dim_k,
        device,
    )
    state = kernel.subtree_get(n_actual, num_vh, dim_v, dim_k, device)

    q = torch.randn((rows, num_kh, dim_k), device=device, dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn((rows, num_vh, dim_v), device=device, dtype=torch.bfloat16)
    raw_a = torch.randn((rows, num_vh), device=device, dtype=torch.bfloat16)
    raw_b = torch.randn_like(raw_a)
    g = torch.zeros((rows, num_vh), device=device, dtype=torch.float32)
    beta = torch.zeros_like(g)
    a_log = -torch.rand((num_vh,), device=device, dtype=torch.float32)
    dt_bias = torch.randn((num_vh,), device=device, dtype=torch.float32)
    h0 = torch.randn(
        (batch, num_vh, dim_v, dim_k), device=device, dtype=torch.float32
    )
    h0_indices = torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1)
    accepted = torch.zeros((batch,), device=device, dtype=torch.int32)
    descriptor = torch.zeros((n_pad, n_pad), device=device, dtype=torch.int32)

    def run(candidate: bool) -> dict[str, torch.Tensor]:
        armed["value"] = candidate
        state["export"].zero_()
        out = torch.zeros((rows, num_vh, dim_v), device=device, dtype=torch.bfloat16)
        ring_k = torch.zeros_like(q)
        ring_v = torch.zeros_like(v)
        ring_a = torch.zeros_like(raw_a)
        ring_b = torch.zeros_like(raw_b)
        flags = torch.zeros((2,), device=device, dtype=torch.int32)
        counter = torch.zeros((), device=device, dtype=torch.int32)
        common = dict(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            raw_a=raw_a,
            raw_b=raw_b,
            A_log=a_log,
            dt_bias=dt_bias,
            h0=h0,
            h0_indices=h0_indices,
            h0_num_accepted_tokens=accepted,
            h0_use_accepted_column=False,
            n_actual=n_actual,
            n_pad=n_pad,
            strict_mask=descriptor,
            visible_mask=descriptor,
            out=out,
            output_scale=dim_k**-0.5,
            use_qk_l2norm_in_kernel=True,
            invocation_counter=counter,
            ring_k=ring_k,
            ring_v=ring_v,
            ring_a=ring_a,
            ring_b=ring_b,
            staging_flags=flags,
            staging_rows=batch,
        )
        if batch == 1:
            kernel.launch_tree_gdn_prepared(
                **common,
                h0_is_bank=True,
                h0_index_row=0,
                h0_batch_index=0,
            )
        else:
            kernel.launch_tree_gdn_prepared_fixed32_batch(
                **common,
                batch_size=batch,
            )
        torch.cuda.synchronize()
        return {
            "out": out.clone(),
            "compact_export": torch.stack(
                tuple(
                    state["export"][
                        request * 5 + slot if batch > 1 else node
                    ]
                    for request in range(batch)
                    for slot, node in enumerate(kernel._FR13_FIXED32_EXPORT_NODES)
                ),
                dim=0,
            ),
            "ring_k": ring_k.clone(),
            "ring_v": ring_v.clone(),
            "ring_a": ring_a.clone(),
            "ring_b": ring_b.clone(),
            "flags": flags.clone(),
            "counter": counter.clone(),
        }

    reference = run(False)
    candidate = run(True)
    mismatches = [
        name
        for name in reference
        if not _byte_equal(reference[name], candidate[name])
    ]
    assert not mismatches, f"B{batch} level-0 coefficient mismatches: {mismatches}"
