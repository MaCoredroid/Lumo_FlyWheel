from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


sys.path.insert(0, str(Path("scripts")))
ORCHESTRATOR_PATH = Path("scripts/run_swe_bench_q36_a.py")
ORCHESTRATOR_SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_task_boundary_orchestrator",
    ORCHESTRATOR_PATH,
)
assert ORCHESTRATOR_SPEC is not None and ORCHESTRATOR_SPEC.loader is not None
orchestrator = importlib.util.module_from_spec(ORCHESTRATOR_SPEC)
ORCHESTRATOR_SPEC.loader.exec_module(orchestrator)
FLOOR_PATH = Path("scripts/fr13_floor_gate.py")
FLOOR_SPEC = importlib.util.spec_from_file_location(
    "fr13_fixed32_task_boundary_floor_gate",
    FLOOR_PATH,
)
assert FLOOR_SPEC is not None and FLOOR_SPEC.loader is not None
floor_gate = importlib.util.module_from_spec(FLOOR_SPEC)
FLOOR_SPEC.loader.exec_module(floor_gate)
PATCHER_PATH = Path("scripts/fr10_phase4_patch_vllm_tree_gdn.py")


def _snapshot(
    *,
    server_capacity: int,
) -> tuple[dict[str, object], SimpleNamespace]:
    histogram = (
        {"1": 5, "2": 0, "3": 0, "4": 0}
        if server_capacity == 1
        else {"1": 2, "2": 1, "3": 1, "4": 1}
    )
    events = sum(histogram.values())
    census_records = [{"fixture_event": index} for index in range(events)]
    events_sha256 = hashlib.sha256(
        json.dumps(
            census_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    ).hexdigest()
    spec_drafts = sum(
        int(batch) * count for batch, count in histogram.items()
    )
    counters = {
        "pure_decode_forward_steps": events,
        "complete_work_census_events": events,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": events - 1,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    by_batch = dict(histogram)
    nonpure_by_batch = (
        {"1": 0, "2": 0, "3": 0, "4": 0}
        if server_capacity == 1
        else {"1": 0, "2": 1, "3": 0, "4": 0}
    )
    raw_by_batch = {
        batch: by_batch[batch] + nonpure_by_batch[batch]
        for batch in by_batch
    }
    nonpure_replays = sum(nonpure_by_batch.values())
    nonpure_dispatch = (
        {
            "guarded_steps": 0,
            "piecewise_steps": 0,
            "none_steps": 0,
            "forbidden_full_steps": 0,
        }
        if server_capacity == 1
        else {
            "guarded_steps": 2,
            "piecewise_steps": 1,
            "none_steps": 1,
            "forbidden_full_steps": 0,
        }
    )
    capture_by_batch = {
        str(batch): int(batch <= server_capacity) for batch in range(1, 5)
    }
    zero_by_batch = {str(batch): 0 for batch in range(1, 5)}
    payload: dict[str, object] = {
        "schema": "fr13-fixed32-boundary-snapshot-v3",
        "mode": "tail6_fixed32",
        "producer_pid": 123,
        "generation": 7,
        "nonce": "a" * 64,
        "action": "snapshot",
        "counters": counters,
        "metrics": {
            "fixed32": {
                "pure_decode_forward_steps": events,
                "complete_work_census_events": events,
                "complete_spec_rows": spec_drafts,
                "spec_drafts": spec_drafts,
                "spec_tokens": 31 * spec_drafts,
                "batch_histogram": histogram,
                "first_forward_step": 0,
                "last_forward_step": events - 1,
                "events_sha256": events_sha256,
            },
            "sfwd": {
                "gpu_seconds": 1.0,
                "steps": events,
                "drafts": spec_drafts,
                "wall_seconds": 2.0,
                "wall_drafts": events,
                "wall_steps": events,
                "wall_rejected": 0,
            },
            "dfwd": {"gpu_seconds": 0.5, "spans": events},
            "cfwd": {"gpu_seconds": 0.25, "spans": events},
            "committer": {
                "actual_replays_by_batch": raw_by_batch,
                "actual_replays_enqueued": events + nonpure_replays,
                "all_batches_ready": True,
                "captures": server_capacity,
                "fast_route_ready": True,
                "maximum_ready_capacity": server_capacity,
                "nonpure_committer_replays_by_batch": nonpure_by_batch,
                "nonpure_committer_replays_enqueued": nonpure_replays,
                "nonpure_dispatch": nonpure_dispatch,
                "preseeded_batches": list(
                    range(1, server_capacity + 1)
                ),
                "preseeded_graphs": server_capacity,
                "ready_capacities": {
                    str(batch): server_capacity
                    for batch in range(1, server_capacity + 1)
                },
                "required_capacity": server_capacity,
            },
            "conv_pregather": {
                "actual_stages": 0,
                "actual_stages_by_batch": zero_by_batch,
                "aux_capture_stages": 0,
                "graph_capture_stages": server_capacity,
                "graph_capture_stages_by_batch": capture_by_batch,
                "graph_replay_stages": events,
                "graph_replay_stages_by_batch": by_batch,
                "max_batch_size": server_capacity,
                "pointer_entries": 48,
                "preseeded": True,
                "preseeded_batches": list(
                    range(1, server_capacity + 1)
                ),
                "profile_capture_stages": 0,
            },
        },
    }
    ack = SimpleNamespace(
        mode=payload["mode"],
        producer_pid=payload["producer_pid"],
        generation=payload["generation"],
        nonce=payload["nonce"],
        action=payload["action"],
        counters=counters,
    )
    return payload, ack


def _write_snapshot(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    base_path = tmp_path / "fr13_fixed32_boundary_snapshot"
    path = Path(f"{base_path}.{payload['generation']}.json")
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="ascii",
    )
    return base_path


def _write_census(
    tmp_path: Path,
    payload: dict[str, object],
) -> Path:
    events = payload["metrics"]["fixed32"][
        "complete_work_census_events"
    ]
    records = [{"fixture_event": index} for index in range(events)]
    path = tmp_path / "fr13_fixed32_work_census.jsonl"
    path.write_text(
        "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":"))
            + "\n"
            for record in [*records, {"fixture_terminal": True}]
        ),
        encoding="ascii",
    )
    return path


def _ack_dict(ack: SimpleNamespace) -> dict[str, object]:
    return {
        "schema": "fr13-fixed32-flush-ack-v1",
        "status": "ok",
        **{
            key: getattr(ack, key)
            for key in (
                "mode",
                "producer_pid",
                "generation",
                "nonce",
                "action",
                "counters",
            )
        },
    }


def _assert_both_validators_reject(
    tmp_path: Path,
    payload: dict[str, object],
    ack: SimpleNamespace,
    *,
    server_capacity: int,
) -> None:
    base_path = _write_snapshot(tmp_path, payload)
    snapshot_path = Path(f"{base_path}.{ack.generation}.json")
    census_path = _write_census(tmp_path, payload)
    with pytest.raises(orchestrator.Fixed32BoundaryError):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=server_capacity,
        )
    with pytest.raises(floor_gate.GateError):
        floor_gate.validate_runtime_boundary_snapshot(
            snapshot_path,
            ack=_ack_dict(ack),
            server_capacity=server_capacity,
            metrics_path=None,
            metric_values=None,
            reference=None,
            census_path=census_path,
        )


@pytest.mark.parametrize("server_capacity", (1, 4))
def test_task_boundary_accepts_in_graph_pregather_counts(
    tmp_path: Path,
    server_capacity: int,
) -> None:
    payload, ack = _snapshot(server_capacity=server_capacity)
    base_path = _write_snapshot(tmp_path, payload)

    loaded, path, digest = orchestrator._load_fixed32_boundary_snapshot(
        base_path=base_path,
        ack=ack,
        server_capacity=server_capacity,
    )

    assert loaded == payload
    assert path == Path(f"{base_path}.{ack.generation}.json")
    assert len(digest) == 64
    floor_report = floor_gate.validate_runtime_boundary_snapshot(
        path,
        ack=_ack_dict(ack),
        server_capacity=server_capacity,
        metrics_path=None,
        metric_values=None,
        reference=None,
        census_path=_write_census(tmp_path, payload),
    )
    assert floor_report["committer"][
        "nonpure_committer_replays_enqueued"
    ] == (0 if server_capacity == 1 else 1)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("actual_stages", 5),
        ("graph_replay_stages", 0),
        ("graph_capture_stages", 0),
    ),
)
def test_task_boundary_rejects_wrong_pregather_stage_class(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    payload, ack = _snapshot(server_capacity=1)
    tampered = copy.deepcopy(payload)
    tampered["metrics"]["conv_pregather"][field] = value
    base_path = _write_snapshot(tmp_path, tampered)

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="committer/nonpure/in-graph pregather counters do not reconcile",
    ):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=1,
        )


def test_task_boundary_rejects_wrong_in_graph_batch_count(
    tmp_path: Path,
) -> None:
    payload, ack = _snapshot(server_capacity=4)
    tampered = copy.deepcopy(payload)
    tampered["metrics"]["conv_pregather"][
        "graph_replay_stages_by_batch"
    ]["4"] -= 1
    base_path = _write_snapshot(tmp_path, tampered)

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="committer/nonpure/in-graph pregather counters do not reconcile",
    ):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=4,
        )


def test_task_boundary_rejects_row_weighted_b4_replay_counts(
    tmp_path: Path,
) -> None:
    payload, ack = _snapshot(server_capacity=4)
    tampered = copy.deepcopy(payload)
    fixed = tampered["metrics"]["fixed32"]
    row_weighted = {
        batch: int(batch) * count
        for batch, count in fixed["batch_histogram"].items()
    }
    for owner in ("committer", "conv_pregather"):
        metrics = tampered["metrics"][owner]
        scalar = (
            "actual_replays_enqueued"
            if owner == "committer"
            else "graph_replay_stages"
        )
        by_batch = (
            "actual_replays_by_batch"
            if owner == "committer"
            else "graph_replay_stages_by_batch"
        )
        metrics[scalar] = fixed["spec_drafts"]
        metrics[by_batch] = row_weighted
    base_path = _write_snapshot(tmp_path, tampered)

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="committer/nonpure/in-graph pregather counters do not reconcile",
    ):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=4,
        )


def test_task_boundary_rejects_boolean_nested_counter(
    tmp_path: Path,
) -> None:
    payload, ack = _snapshot(server_capacity=4)
    tampered = copy.deepcopy(payload)
    tampered["metrics"]["conv_pregather"][
        "graph_replay_stages_by_batch"
    ]["4"] = True
    base_path = _write_snapshot(tmp_path, tampered)

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="must be a nonnegative integer",
    ):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=4,
        )


def test_task_boundary_rejects_uncontracted_pregather_field(
    tmp_path: Path,
) -> None:
    payload, ack = _snapshot(server_capacity=1)
    tampered = copy.deepcopy(payload)
    tampered["metrics"]["conv_pregather"]["unexpected"] = 0
    base_path = _write_snapshot(tmp_path, tampered)

    with pytest.raises(
        orchestrator.Fixed32BoundaryError,
        match="conv_pregather keys mismatch",
    ):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=1,
        )


def test_both_validators_reject_nonpure_reconciliation_tampers(
    tmp_path: Path,
) -> None:
    def raw_scalar(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"]["actual_replays_enqueued"] += 1

    def raw_bucket(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"]["actual_replays_by_batch"]["2"] += 1

    def nonpure_scalar(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"][
            "nonpure_committer_replays_enqueued"
        ] += 1

    def nonpure_bucket(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"][
            "nonpure_committer_replays_by_batch"
        ]["2"] += 1

    def dispatch_decomposition(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"]["nonpure_dispatch"][
            "guarded_steps"
        ] += 1

    def forbidden_full(payload: dict[str, object]) -> None:
        dispatch = payload["metrics"]["committer"]["nonpure_dispatch"]
        dispatch["guarded_steps"] += 1
        dispatch["forbidden_full_steps"] = 1

    def nonpure_exceeds_guard(payload: dict[str, object]) -> None:
        dispatch = payload["metrics"]["committer"]["nonpure_dispatch"]
        dispatch.update(
            {
                "guarded_steps": 0,
                "piecewise_steps": 0,
                "none_steps": 0,
                "forbidden_full_steps": 0,
            }
        )

    def impossible_capacity_bucket(payload: dict[str, object]) -> None:
        committer = payload["metrics"]["committer"]
        nonpure = committer["nonpure_committer_replays_by_batch"]
        raw = committer["actual_replays_by_batch"]
        nonpure["2"] -= 1
        raw["2"] -= 1
        nonpure["4"] += 1
        raw["4"] += 1

    def pregather_counts_nonpure(payload: dict[str, object]) -> None:
        pregather = payload["metrics"]["conv_pregather"]
        pregather["graph_replay_stages"] += 1
        pregather["graph_replay_stages_by_batch"]["2"] += 1

    mutations = (
        raw_scalar,
        raw_bucket,
        nonpure_scalar,
        nonpure_bucket,
        dispatch_decomposition,
        forbidden_full,
        nonpure_exceeds_guard,
        impossible_capacity_bucket,
        pregather_counts_nonpure,
    )
    for index, mutate in enumerate(mutations):
        payload, ack = _snapshot(server_capacity=4)
        mutate(payload)
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        _assert_both_validators_reject(
            case_dir,
            payload,
            ack,
            server_capacity=4,
        )


def test_both_validators_reject_nonpure_replay_at_b1(
    tmp_path: Path,
) -> None:
    payload, ack = _snapshot(server_capacity=1)
    committer = payload["metrics"]["committer"]
    committer["nonpure_committer_replays_enqueued"] = 1
    committer["nonpure_committer_replays_by_batch"]["1"] = 1
    committer["nonpure_dispatch"]["guarded_steps"] = 1
    committer["nonpure_dispatch"]["none_steps"] = 1
    committer["actual_replays_enqueued"] += 1
    committer["actual_replays_by_batch"]["1"] += 1

    _assert_both_validators_reject(
        tmp_path,
        payload,
        ack,
        server_capacity=1,
    )


def test_both_validators_reject_numeric_aliases_and_v2(
    tmp_path: Path,
) -> None:
    def producer_float(payload: dict[str, object]) -> None:
        payload["producer_pid"] = 123.0

    def pending_bool(payload: dict[str, object]) -> None:
        payload["counters"]["sfwd_pending"] = False

    def fixed_float(payload: dict[str, object]) -> None:
        fixed = payload["metrics"]["fixed32"]
        fixed["complete_spec_rows"] = float(fixed["complete_spec_rows"])

    def sfwd_bool(payload: dict[str, object]) -> None:
        payload["metrics"]["sfwd"]["wall_rejected"] = False

    def span_string(payload: dict[str, object]) -> None:
        spans = payload["metrics"]["dfwd"]["spans"]
        payload["metrics"]["dfwd"]["spans"] = str(spans)

    def committer_bool(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"]["captures"] = True

    def capacity_float(payload: dict[str, object]) -> None:
        payload["metrics"]["committer"]["ready_capacities"]["1"] = 1.0

    def legacy_schema(payload: dict[str, object]) -> None:
        payload["schema"] = "fr13-fixed32-boundary-snapshot-v2"

    mutations = (
        producer_float,
        pending_bool,
        fixed_float,
        sfwd_bool,
        span_string,
        committer_bool,
        capacity_float,
        legacy_schema,
    )
    for index, mutate in enumerate(mutations):
        payload, ack = _snapshot(server_capacity=1)
        mutate(payload)
        case_dir = tmp_path / str(index)
        case_dir.mkdir()
        _assert_both_validators_reject(
            case_dir,
            payload,
            ack,
            server_capacity=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "coordinate_snapshot"),
    (
        ("producer_pid", 123.0, False),
        ("generation", 7.0, False),
        ("mode", 17, True),
        ("nonce", 17, True),
        ("action", 17, True),
    ),
)
def test_both_validators_reject_malformed_ack_identity(
    tmp_path: Path,
    field: str,
    value: object,
    coordinate_snapshot: bool,
) -> None:
    payload, ack = _snapshot(server_capacity=1)
    if coordinate_snapshot:
        payload[field] = value
    setattr(ack, field, value)
    base_path = _write_snapshot(tmp_path, payload)
    snapshot_path = Path(f"{base_path}.{payload['generation']}.json")
    census_path = _write_census(tmp_path, payload)

    with pytest.raises(orchestrator.Fixed32BoundaryError):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=1,
        )
    with pytest.raises(floor_gate.GateError):
        floor_gate.validate_runtime_boundary_snapshot(
            snapshot_path,
            ack=_ack_dict(ack),
            server_capacity=1,
            metrics_path=None,
            metric_values=None,
            reference=None,
            census_path=census_path,
        )


@pytest.mark.parametrize(
    ("counter", "value"),
    (
        ("pure_decode_forward_steps", 5.0),
        ("sfwd_pending", False),
    ),
)
def test_both_validators_reject_malformed_ack_counter(
    tmp_path: Path,
    counter: str,
    value: object,
) -> None:
    payload, ack = _snapshot(server_capacity=1)
    ack.counters = copy.deepcopy(ack.counters)
    ack.counters[counter] = value
    base_path = _write_snapshot(tmp_path, payload)
    snapshot_path = Path(f"{base_path}.{payload['generation']}.json")
    census_path = _write_census(tmp_path, payload)

    with pytest.raises(orchestrator.Fixed32BoundaryError):
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=1,
        )
    with pytest.raises(floor_gate.GateError):
        floor_gate.validate_runtime_boundary_snapshot(
            snapshot_path,
            ack=_ack_dict(ack),
            server_capacity=1,
            metrics_path=None,
            metric_values=None,
            reference=None,
            census_path=census_path,
        )


def test_runtime_writer_serializes_mixed_b4_v3_for_both_validators(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    patcher_tree = ast.parse(PATCHER_PATH.read_text(encoding="utf-8"))
    fixed_flush_sources = [
        node.value.value
        for node in ast.walk(patcher_tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "fixed_flush"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]
    assert len(fixed_flush_sources) == 1
    tree = ast.parse(fixed_flush_sources[0])
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_fr13_f32_flush_write_boundary"
    ]
    assert len(definitions) == 1

    events = [{"batch_size": 4}]
    nonpure_by_batch = {1: 0, 2: 1, 3: 0, 4: 0}
    nonpure_dispatch = {
        "guarded_steps": 2,
        "piecewise_steps": 1,
        "none_steps": 1,
        "forbidden_full_steps": 0,
    }
    gdn = SimpleNamespace(
        _fr13_fixed32_nonpure_commit_replays_by_batch=(
            lambda: dict(nonpure_by_batch)
        ),
        _fr13_fixed32_nonpure_dispatch_counters=(
            lambda: dict(nonpure_dispatch)
        ),
    )
    timer = SimpleNamespace(
        _accum_s=1.0,
        _n_steps=1,
        _n_drafts=4,
        _wall_accum_s=2.0,
        _wall_drafts=1,
        _wall_steps=1,
        _wall_rejected=0,
        _n_spans=1,
    )
    raw_by_batch = {1: 0, 2: 1, 3: 0, 4: 1}
    commit_counters = {
        "actual_replays_by_batch": raw_by_batch,
        "actual_replays_enqueued": 2,
        "all_batches_ready": True,
        "captures": 4,
        "fast_route_ready": True,
        "maximum_ready_capacity": 4,
        "preseeded_batches": [1, 2, 3, 4],
        "preseeded_graphs": 4,
        "ready_capacities": {1: 4, 2: 4, 3: 4, 4: 4},
        "required_capacity": 4,
    }
    pregather_counters = {
        "actual_stages": 0,
        "actual_stages_by_batch": {1: 0, 2: 0, 3: 0, 4: 0},
        "aux_capture_stages": 0,
        "graph_capture_stages": 4,
        "graph_capture_stages_by_batch": {1: 1, 2: 1, 3: 1, 4: 1},
        "max_batch_size": 4,
        "pointer_entries": 48,
        "preseeded": True,
        "preseeded_batches": [1, 2, 3, 4],
        "profile_capture_stages": 0,
    }
    kernel_module = ModuleType(
        "lumo_flywheel_serving.fr10_gdn_tree_kernel"
    )
    kernel_module.fixed32_committer_counters = lambda: copy.deepcopy(
        commit_counters
    )
    kernel_module.fixed32_conv_col0_pregather_counters = (
        lambda: copy.deepcopy(pregather_counters)
    )
    package = ModuleType("lumo_flywheel_serving")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "lumo_flywheel_serving", package)
    monkeypatch.setitem(
        sys.modules,
        "lumo_flywheel_serving.fr10_gdn_tree_kernel",
        kernel_module,
    )

    base_path = tmp_path / "runtime_boundary"
    writes: list[tuple[Path, str]] = []
    namespace = {
        "_fr13_f32_flush_runtime_state": lambda: (
            gdn,
            events,
            1,
            0,
            0,
            timer,
            timer,
            timer,
        ),
        "_fr13_f32_flush_json": json,
        "_fr13_f32_flush_hashlib": hashlib,
        "_FR13_FIXED32_FLUSH_MODE": "tail6_fixed32",
        "_FR13_FIXED32_FLUSH_PID": 123,
        "_FR13_FIXED32_FLUSH_BOUNDARY_PATH": base_path,
        "_Fr13F32FlushPath": Path,
        "_fr13_f32_flush_atomic_text": (
            lambda path, body: writes.append((path, body))
        ),
    }
    exec(
        compile(
            ast.Module(body=definitions, type_ignores=[]),
            "<fixed32-runtime-boundary-writer>",
            "exec",
        ),
        namespace,
    )
    counters = {
        "pure_decode_forward_steps": 1,
        "complete_work_census_events": 1,
        "work_census_first_forward_step": 0,
        "work_census_last_forward_step": 0,
        "sfwd_pending": 0,
        "dfwd_pending": 0,
        "cfwd_pending": 0,
    }
    request = {
        "generation": 7,
        "nonce": "a" * 64,
        "action": "snapshot",
    }
    namespace["_fr13_f32_flush_write_boundary"](request, counters)

    assert len(writes) == 1
    snapshot_path, body = writes[0]
    assert snapshot_path == Path(f"{base_path}.7.json")
    snapshot_path.write_text(body, encoding="ascii")
    snapshot = json.loads(body)
    assert snapshot["schema"] == "fr13-fixed32-boundary-snapshot-v3"
    assert snapshot["metrics"]["fixed32"]["batch_histogram"] == {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 1,
    }
    assert snapshot["metrics"]["committer"][
        "nonpure_committer_replays_by_batch"
    ] == {"1": 0, "2": 1, "3": 0, "4": 0}
    assert snapshot["metrics"]["conv_pregather"][
        "graph_replay_stages"
    ] == 1

    ack = SimpleNamespace(
        mode="tail6_fixed32",
        producer_pid=123,
        generation=7,
        nonce="a" * 64,
        action="snapshot",
        counters=counters,
    )
    loaded, _path, _digest = (
        orchestrator._load_fixed32_boundary_snapshot(
            base_path=base_path,
            ack=ack,
            server_capacity=4,
        )
    )
    assert loaded == snapshot
    census_path = tmp_path / "runtime_census.jsonl"
    census_path.write_text(
        json.dumps(events[0], sort_keys=True, separators=(",", ":"))
        + "\n{}\n",
        encoding="ascii",
    )
    floor_report = floor_gate.validate_runtime_boundary_snapshot(
        snapshot_path,
        ack=_ack_dict(ack),
        server_capacity=4,
        metrics_path=None,
        metric_values=None,
        reference=None,
        census_path=census_path,
    )
    assert floor_report["committer"][
        "nonpure_committer_replays_enqueued"
    ] == 1
