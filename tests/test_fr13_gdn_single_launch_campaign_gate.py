from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REDUCER = ROOT / "scripts/fr13_gdn_single_launch_gate.py"
COMMON = ROOT / "scripts/fr13_run_gdn_single_launch_live_gate.sh"
ENTRYPOINTS = {
    "scripts/fr13_run_b1_gdn_single_launch_live_gate.sh": (
        "hydra27_fixed32",
        "1",
    ),
    "scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh": (
        "tail6_fixed32",
        "4",
    ),
    "scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh": (
        "hydra27_fixed32",
        "4",
    ),
}


def _load_reducer():
    spec = importlib.util.spec_from_file_location("fr13_gdn_single_launch_gate", REDUCER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _live_payload(*, mode: str, batch: int) -> dict[str, object]:
    contract = {
        "tail6_fixed32": ("Tail23", "tail23", 23, 0x7A9CE7FF),
        "hydra27_fixed32": ("Hydra27", "hydra27", 27, 0x7ABDFFFF),
    }[mode]
    topology, slug, drafts, mask = contract
    return {
        "schema": "fr13.fixed32.gdn_single_launch.live_pass.v1",
        "status": "pass",
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "source_sha256": "a" * 64,
        "task_marker": "swe_verified:marker",
        "mode": mode,
        "graph_id": 404,
        "graph_signature": "b" * 64,
        "batch_size": batch,
        "expected_batch": batch,
        "covered_batches": [batch],
        "records": 48,
        "physical_rows": 32,
        "reference_bv": 8,
        "candidate_bv": 8,
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "compared_byte_surfaces": [
            "output",
            "ring_k",
            "ring_v",
            "ring_a",
            "ring_b",
            "flags",
            "counter",
        ],
        "raw_byte_equal": True,
        "reference_served": True,
        "state_restored": True,
        "real_task_authenticated": True,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "logical_topology": topology,
        "logical_drafts": drafts,
        "valid_mask": mask,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "gate_mode": "post_first_measured_full_graph_replay",
        "diagnostic_identity": (
            f"fixed32_gdn_single_launch_tree_v2:{slug}:b{batch}"
        ),
    }


def test_three_entrypoints_bake_disjoint_mode_batch_scopes() -> None:
    assert len(ENTRYPOINTS) == 3
    for relative, (mode, batch) in ENTRYPOINTS.items():
        text = (ROOT / relative).read_text(encoding="ascii")
        assert f"export FR13_GDN_GATE_MODE={mode}" in text
        assert f"export FR13_GDN_GATE_BATCH={batch}" in text
        assert f"export FR13_GDN_GATE_ENTRYPOINT={relative}" in text
        assert "fr13_run_gdn_single_launch_live_gate.sh" in text
    common = COMMON.read_text(encoding="ascii")
    assert 'FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH="$BATCH"' in common
    assert 'MAX_NUM_SEQS_OVR="$BATCH" SWE_CONCURRENCY="$BATCH"' in common
    assert 'if [[ "$FR13_GDN_GATE_BATCH" == "4" ]]; then' in common
    assert 'KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES"' in common
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in common
    assert "config/fr13_fixed32/subset_b4_four.json" in common


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_batch", 1),
        ("covered_batches", [1]),
        ("diagnostic_identity", "fixed32_gdn_single_launch_tree_v2:tail23:b4"),
        ("logical_topology", "Tail23"),
        ("graph_signature", "c" * 64),
        ("reference_served", False),
    ),
)
def test_live_pass_rejects_batch_topology_graph_and_served_state_tamper(
    field: str, value: object
) -> None:
    reducer = _load_reducer()
    payload = _live_payload(mode="hydra27_fixed32", batch=4)
    payload[field] = value
    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        reducer._validate_live_pass(
            payload,
            mode="hydra27_fixed32",
            batch=4,
            task_markers=frozenset({"swe_verified:marker"}),
            kernel_sha256="a" * 64,
            graph_signature="b" * 64,
        )


def test_live_pass_accepts_only_its_exact_scope() -> None:
    reducer = _load_reducer()
    payload = _live_payload(mode="tail6_fixed32", batch=4)
    reducer._validate_live_pass(
        payload,
        mode="tail6_fixed32",
        batch=4,
        task_markers=frozenset({"swe_verified:marker"}),
        kernel_sha256="a" * 64,
        graph_signature="b" * 64,
    )
    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        reducer._validate_live_pass(
            payload,
            mode="hydra27_fixed32",
            batch=4,
            task_markers=frozenset({"swe_verified:marker"}),
            kernel_sha256="a" * 64,
            graph_signature="b" * 64,
        )


def test_b4_live_pass_accepts_only_one_of_the_exact4_trigger_tasks() -> None:
    reducer = _load_reducer()
    payload = _live_payload(mode="hydra27_fixed32", batch=4)
    allowed = frozenset(
        f"swe_verified:{task_id}" for task_id in reducer.EXACT4_TASK_IDS
    )
    payload["task_marker"] = next(iter(allowed))
    reducer._validate_live_pass(
        payload,
        mode="hydra27_fixed32",
        batch=4,
        task_markers=allowed,
        kernel_sha256="a" * 64,
        graph_signature="b" * 64,
    )
    payload["task_marker"] = "swe_verified:not-in-exact4"
    with pytest.raises(reducer.GateError, match="not a canonical task"):
        reducer._validate_live_pass(
            payload,
            mode="hydra27_fixed32",
            batch=4,
            task_markers=allowed,
            kernel_sha256="a" * 64,
            graph_signature="b" * 64,
        )


def test_runtime_manifest_rejects_source_closure_tamper() -> None:
    reducer = _load_reducer()
    payload = {
        "schema": "fr13-runtime-manifest-v1",
        "profile": "fixed32",
        "sequence": "scripts/fr13_fixed32_floor_timers_seq.sh",
        "closures": {
            "host_script_source": [
                {"path": "scripts/runner.sh", "sha256": "a" * 64, "size": 1}
            ]
        },
    }
    unsigned = dict(payload)
    payload["overall_canonical_sha256"] = hashlib.sha256(
        json.dumps(
            unsigned,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    assert reducer._validate_runtime_manifest(
        payload, required_closure={"scripts/runner.sh": "a" * 64}
    ) == payload["overall_canonical_sha256"]
    with pytest.raises(reducer.GateError, match="does not bind"):
        reducer._validate_runtime_manifest(
            payload, required_closure={"scripts/runner.sh": "b" * 64}
        )


def test_reducer_rebuilds_qwen_ingress_and_graph_evidence() -> None:
    reducer = _load_reducer()
    source = REDUCER.read_text(encoding="ascii")
    assert "floor_gate.build_fixed32_chat_traffic_audit(" in source
    assert "work_census.validate_arm(" in source
    assert "fixed32_contract.validate_external_manifest(" in source
    assert 'runtime_launch_raw != runtime_end_raw' in source
    assert 'external_launch_raw != external_end_raw' in source
    assert '"qwen_compaction_algebra_replayed": True' in source
    assert '"qwen_per_task_binding_verified": True' in source
    assert '"finalized_ingress_verified": True' in source
    assert '"batch_specific_pass_verified": True' in source
    assert set(reducer.VALIDATOR_SOURCES) == {
        "scripts/fr13_fixed32_contract.py",
        "scripts/fr13_fixed32_work_census.py",
        "scripts/fr13_floor_gate.py",
        "scripts/fr13_runtime_manifest.py",
    }


def test_runtime_manifest_closes_over_all_gdn_gate_sources() -> None:
    manifest = (ROOT / "scripts/fr13_runtime_manifest.py").read_text(encoding="ascii")
    for relative in (*ENTRYPOINTS, "scripts/fr13_run_gdn_single_launch_live_gate.sh"):
        assert f'"{relative}"' in manifest
    assert '"scripts/fr13_gdn_single_launch_gate.py"' in manifest


def test_server_contract_admits_single_launch_metrics() -> None:
    launcher = (
        ROOT / "scripts/fr13_launch_forked_fa2_tree_server.sh"
    ).read_text(encoding="utf-8")
    start = launcher.index("_fixed32_expected_metrics=0")
    end = launcher.index(
        'if [[ "$_fr13_fixed32_batch_gdn_diagnostic" == "1" ]]; then',
        start,
    )
    expected_metrics = launcher[start:end]
    assert '$_fr13_gdn_path_bv_candidate" == "single_launch"' in expected_metrics
