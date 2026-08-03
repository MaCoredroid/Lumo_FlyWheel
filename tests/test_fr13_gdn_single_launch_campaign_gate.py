from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REDUCER = ROOT / "scripts/fr13_gdn_single_launch_gate.py"
COMMON = ROOT / "scripts/fr13_run_gdn_single_launch_live_gate.sh"
ENTRYPOINTS = {
    "scripts/fr13_run_b1_gdn_single_launch_live_gate.sh": (
        "hydra27_fixed32",
        "1",
        None,
    ),
    "scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh": (
        "hydra27_fixed32",
        "1",
        "gqa_group3",
    ),
    "scripts/fr13_run_b4_tail23_gdn_single_launch_live_gate.sh": (
        "tail6_fixed32",
        "4",
        None,
    ),
    "scripts/fr13_run_b4_hydra27_gdn_single_launch_live_gate.sh": (
        "hydra27_fixed32",
        "4",
        None,
    ),
}


def _load_reducer():
    spec = importlib.util.spec_from_file_location("fr13_gdn_single_launch_gate", REDUCER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _git_fixture(tmp_path: Path) -> tuple[Path, str, bytes]:
    repo = tmp_path / "repo"
    source = b"#!/usr/bin/env bash\necho bound\n"
    path = repo / "scripts/runner.sh"
    path.parent.mkdir(parents=True)
    path.write_bytes(source)
    (repo / "scripts/sequence.sh").write_bytes(b"#!/usr/bin/env bash\n")
    for command in (
        ["git", "init", "-q"],
        ["git", "config", "user.email", "gate@example.invalid"],
        ["git", "config", "user.name", "Gate Test"],
        ["git", "add", "scripts/runner.sh", "scripts/sequence.sh"],
        ["git", "commit", "-qm", "fixture"],
    ):
        subprocess.run(command, cwd=repo, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    return repo, commit, source


def _runtime_payload(
    *, commit: str, source: bytes, digest: str
) -> dict[str, object]:
    source_record = {
        "path": "scripts/runner.sh",
        "sha256": digest,
        "size": len(source),
    }
    closures = {
        "host_script_source": [source_record],
        "python_package_source": [],
        "runtime_data_and_config": [],
        "verdict_tools": [],
    }
    payload: dict[str, object] = {
        "schema": "fr13-runtime-manifest-v1",
        "profile": "fixed32",
        "sequence": "scripts/fr13_fixed32_floor_timers_seq.sh",
        "canonical_format": "utf8-json-sort-keys-compact-v1",
        "closures": closures,
        "required_absence": [],
        "summary": {
            "file_count": 1,
            "python_package_file_count": 0,
            "total_size": len(source),
        },
        "source_identity": {
            "schema": "fr13-runtime-source-identity-v1",
            "git_commit": commit,
            "source_file_count": 1,
            "source_closure_sha256": hashlib.sha256(
                json.dumps(
                    [source_record],
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest(),
        },
    }
    payload["overall_canonical_sha256"] = hashlib.sha256(
        json.dumps(
            payload,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return payload


def _comparator_event(
    *,
    mode: str,
    batch: int,
    event_index: int,
    request_digests: list[str],
    marker_task: str,
    graph_signature: str = "b" * 64,
    capture_signature: str = "c" * 64,
) -> dict[str, object]:
    return {
        "schema": "fr13.fixed32.gdn_single_launch.comparator_event.v1",
        "mode": mode,
        "batch_size": batch,
        "runtime_capture_manifest_sha256": capture_signature,
        "structural_graph_signature": graph_signature,
        "reference": "fixed32_gdn_two_launch_reference_v1",
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "reference_physical_launches_per_request_layer": 2,
        "candidate_physical_launches_per_request_layer": 1,
        "records": 48,
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
        "state_restored": True,
        "reference_served": True,
        "candidate_served": False,
        "comparison_order": [
            "reference",
            "restore_baseline",
            "candidate",
            "restore_baseline_in_finally",
        ],
        "census_event_id": f"{mode}:4242:{event_index}",
        "census_event_index": event_index,
        "census_forward_step_index": event_index,
        "request_id_sha256s": request_digests,
        "observed_task_marker": f"swe_verified:{marker_task}",
    }


def _live_payload(
    *, mode: str, batch: int, comparator_events: list[dict[str, object]]
) -> dict[str, object]:
    contract = {
        "tail6_fixed32": ("Tail23", "tail23", 23, 0x7A9CE7FF),
        "hydra27_fixed32": ("Hydra27", "hydra27", 27, 0x7ABDFFFF),
    }[mode]
    topology, slug, drafts, mask = contract
    canonical_events = json.dumps(
        comparator_events,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {
        "schema": "fr13.fixed32.gdn_single_launch.live_observation.v2",
        "status": "observed_pending_authenticated_coverage_join",
        "candidate": "fixed32_gdn_single_launch_tree_v2",
        "source_sha256": "a" * 64,
        "mode": mode,
        "batch_size": batch,
        "expected_batch": batch,
        "covered_batches": [batch],
        "records_per_comparator_event": 48,
        "comparator_event_count": len(comparator_events),
        "comparator_events_sha256": hashlib.sha256(canonical_events).hexdigest(),
        "comparator_events": comparator_events,
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
        "reference_served": True,
        "candidate_served": False,
        "production_eligible": False,
        "performance_measurement": False,
        "acceptance_valid": False,
        "logical_topology": topology,
        "logical_drafts": drafts,
        "valid_mask": mask,
        "draft_vocab_k": 65536,
        "draft_vocab_root": 1,
        "gate_mode": "post_measured_replay_distinct_request_tuple",
        "coverage_authority": "authenticated_proxy_engine_request_join",
        "diagnostic_identity": (
            f"fixed32_gdn_single_launch_tree_v2:{slug}:b{batch}"
        ),
    }


def _coverage_fixture(
    reducer, *, mode: str, batch: int
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, str],
    list[str],
]:
    tasks = list(
        reducer.B1_TASK_IDS if batch == 1 else reducer.EXACT4_TASK_IDS
    )
    digests = [hashlib.sha256(task.encode("ascii")).hexdigest() for task in tasks]
    comparator = _comparator_event(
        mode=mode,
        batch=batch,
        event_index=0,
        request_digests=digests,
        marker_task=tasks[0],
    )
    payload = _live_payload(
        mode=mode,
        batch=batch,
        comparator_events=[comparator],
    )
    return (
        payload,
        [{"gdn_comparator": comparator}],
        dict(zip(digests, tasks, strict=True)),
        tasks,
    )


def _validate_fixture(
    reducer,
    payload: dict[str, object],
    census_events: list[dict[str, object]],
    request_task_map: dict[str, str],
    tasks: list[str],
    *,
    mode: str,
    batch: int,
):
    return reducer._validate_live_pass(
        payload,
        mode=mode,
        batch=batch,
        expected_tasks=tasks,
        kernel_sha256="a" * 64,
        graph_signature="b" * 64,
        census_events=census_events,
        request_task_map=request_task_map,
    )


def test_entrypoints_bake_disjoint_mode_batch_candidate_scopes() -> None:
    assert len(ENTRYPOINTS) == 4
    for relative, (mode, batch, candidate) in ENTRYPOINTS.items():
        text = (ROOT / relative).read_text(encoding="ascii")
        assert f"export FR13_GDN_GATE_MODE={mode}" in text
        assert f"export FR13_GDN_GATE_BATCH={batch}" in text
        assert f"export FR13_GDN_GATE_ENTRYPOINT={relative}" in text
        if candidate is not None:
            assert f"export FR13_GDN_GATE_CANDIDATE={candidate}" in text
        assert "fr13_run_gdn_single_launch_live_gate.sh" in text
    common = COMMON.read_text(encoding="ascii")
    assert 'FR13_FIXED32_GDN_SINGLE_LAUNCH_EXPECTED_BATCH="$BATCH"' in common
    assert 'MAX_NUM_SEQS_OVR="$BATCH" SWE_CONCURRENCY="$BATCH"' in common
    assert 'if [[ "$FR13_GDN_GATE_BATCH" == "4" ]]; then' in common
    assert 'KV_CACHE_MEMORY_BYTES="$KV_CACHE_MEMORY_BYTES"' in common
    assert common.count('--source-commit "$SOURCE_COMMIT"') == 3
    assert "config/fr13_fixed32/subset_b1_diagnostic_one.json" in common
    assert "config/fr13_fixed32/subset_b4_four.json" in common
    assert "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION=0" in common
    assert "FR13_FIXED32_GDN_GQA_GROUP3_PRODUCTION_BATCH=" in common
    assert "FR13_FIXED32_GDN_GQA_GROUP3_PASS_JSON=" in common


def test_b1_gqa_group3_entrypoint_cannot_be_reused_for_another_candidate(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        ["bash", os.fspath(COMMON)],
        cwd=ROOT,
        env={
            **os.environ,
            "RUNROOT": os.fspath(tmp_path / "unused"),
            "TAG": "candidate-scope-test",
            "FORKED_FA2_SO": "/dev/null",
            "FR13_GDN_GATE_MODE": "hydra27_fixed32",
            "FR13_GDN_GATE_BATCH": "1",
            "FR13_GDN_GATE_CANDIDATE": "single_launch",
            "FR13_GDN_GATE_ENTRYPOINT": (
                "scripts/fr13_run_b1_gdn_gqa_group3_live_gate.sh"
            ),
        },
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 2
    assert "wrapper mode/batch/entrypoint identity is invalid" in result.stderr


def test_gqa_group3_b1_live_pass_is_candidate_source_bound() -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode="hydra27_fixed32", batch=1
    )
    candidate_id = reducer.GQA_GROUP3_CANDIDATE
    source_sha256 = "e" * 64
    comparator = events[0]["gdn_comparator"]
    comparator["candidate"] = candidate_id
    comparator_events = [comparator]
    payload["candidate"] = candidate_id
    payload["source_sha256"] = source_sha256
    payload["diagnostic_identity"] = f"{candidate_id}:hydra27:b1"
    payload["comparator_events_sha256"] = hashlib.sha256(
        json.dumps(
            comparator_events,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()

    report = reducer._validate_live_pass(
        payload,
        mode="hydra27_fixed32",
        batch=1,
        expected_tasks=tasks,
        candidate_id=candidate_id,
        candidate_source_sha256=source_sha256,
        graph_signature="b" * 64,
        census_events=events,
        request_task_map=request_map,
    )
    assert report["authenticated_task_ids"] == tasks

    payload["source_sha256"] = "f" * 64
    with pytest.raises(reducer.GateError, match="source_sha256"):
        reducer._validate_live_pass(
            payload,
            mode="hydra27_fixed32",
            batch=1,
            expected_tasks=tasks,
            candidate_id=candidate_id,
            candidate_source_sha256=source_sha256,
            graph_signature="b" * 64,
            census_events=events,
            request_task_map=request_map,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("expected_batch", 1),
        ("covered_batches", [1]),
        ("diagnostic_identity", "fixed32_gdn_single_launch_tree_v2:tail23:b4"),
        ("logical_topology", "Tail23"),
        ("reference_served", False),
    ),
)
def test_live_observation_rejects_scope_and_served_state_tamper(
    field: str, value: object
) -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode="hydra27_fixed32", batch=4
    )
    payload[field] = value
    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        _validate_fixture(
            reducer,
            payload,
            events,
            request_map,
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )


@pytest.mark.parametrize(
    ("mode", "batch"),
    (("hydra27_fixed32", 1), ("tail6_fixed32", 4), ("hydra27_fixed32", 4)),
)
def test_live_observation_accepts_exact_authenticated_scope(
    mode: str, batch: int
) -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode=mode, batch=batch
    )
    report = _validate_fixture(
        reducer,
        payload,
        events,
        request_map,
        tasks,
        mode=mode,
        batch=batch,
    )
    assert report["authenticated_task_ids"] == tasks
    assert report["comparator_event_count"] == 1


def test_stale_same_source_observation_and_truncation_are_rejected() -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode="hydra27_fixed32", batch=4
    )
    stale = dict(events[0]["gdn_comparator"])
    stale["census_event_id"] = "hydra27_fixed32:4242:9"
    stale["census_event_index"] = 9
    stale["census_forward_step_index"] = 9
    stale_payload = _live_payload(
        mode="hydra27_fixed32", batch=4, comparator_events=[stale]
    )
    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        _validate_fixture(
            reducer,
            stale_payload,
            events,
            request_map,
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )

    second = dict(events[0]["gdn_comparator"])
    second["census_event_id"] = "hydra27_fixed32:4242:1"
    second["census_event_index"] = 1
    second["census_forward_step_index"] = 1
    second["request_id_sha256s"] = ["d" * 64, *second["request_id_sha256s"][1:]]
    expanded_events = [events[0], {"gdn_comparator": second}]
    truncated = _live_payload(
        mode="hydra27_fixed32",
        batch=4,
        comparator_events=[events[0]["gdn_comparator"]],
    )
    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        _validate_fixture(
            reducer,
            truncated,
            expanded_events,
            request_map | {"d" * 64: tasks[0]},
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )


def test_missing_and_duplicate_exact4_comparator_coverage_are_rejected() -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode="hydra27_fixed32", batch=4
    )
    missing_map = dict(request_map)
    missing_digest = next(
        digest for digest, task in request_map.items() if task == tasks[-1]
    )
    missing_map[missing_digest] = tasks[0]
    with pytest.raises(reducer.GateError, match="task union is not the exact"):
        _validate_fixture(
            reducer,
            payload,
            events,
            missing_map,
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )

    duplicate = dict(events[0]["gdn_comparator"])
    duplicate["census_event_id"] = "hydra27_fixed32:4242:1"
    duplicate["census_event_index"] = 1
    duplicate["census_forward_step_index"] = 1
    duplicated_events = [events[0], {"gdn_comparator": duplicate}]
    duplicated_payload = _live_payload(
        mode="hydra27_fixed32",
        batch=4,
        comparator_events=[events[0]["gdn_comparator"], duplicate],
    )
    with pytest.raises(reducer.GateError, match="tuple is duplicated"):
        _validate_fixture(
            reducer,
            duplicated_payload,
            duplicated_events,
            request_map,
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )


def test_request_task_relabel_and_scope_swap_are_rejected() -> None:
    reducer = _load_reducer()
    payload, events, request_map, tasks = _coverage_fixture(
        reducer, mode="hydra27_fixed32", batch=4
    )
    relabeled = dict(events[0]["gdn_comparator"])
    relabeled["observed_task_marker"] = "swe_verified:not_the_request_task"
    relabeled_payload = _live_payload(
        mode="hydra27_fixed32", batch=4, comparator_events=[relabeled]
    )
    with pytest.raises(reducer.GateError, match="marker was relabeled"):
        _validate_fixture(
            reducer,
            relabeled_payload,
            [{"gdn_comparator": relabeled}],
            request_map,
            tasks,
            mode="hydra27_fixed32",
            batch=4,
        )

    with pytest.raises(reducer.GateError, match="live PASS field drifted"):
        _validate_fixture(
            reducer,
            payload,
            events,
            request_map,
            tasks,
            mode="tail6_fixed32",
            batch=4,
        )


def test_work_census_seals_comparator_event_index_and_request_digests() -> None:
    reducer = _load_reducer()
    census = reducer.work_census
    mode = "tail6_fixed32"
    event = census.reference_event(
        mode,
        4,
        f"{mode}:4242:0",
        event_index=0,
        forward_step_index=0,
    )
    request_digests = list(event["drafter_runtime"]["request_id_sha256s"])
    event["gdn_comparator"] = _comparator_event(
        mode=mode,
        batch=4,
        event_index=0,
        request_digests=request_digests,
        marker_task="astropy__astropy-12907",
        graph_signature=census.forward_graph_structural_signature(4),
    )
    census.validate_event(event, source="sealed-event")

    swapped = json.loads(json.dumps(event))
    swapped["gdn_comparator"]["census_event_index"] = 1
    with pytest.raises(census.CensusError, match="census_event_index"):
        census.validate_event(swapped, source="index-swap")

    relabeled = json.loads(json.dumps(event))
    relabeled["gdn_comparator"]["request_id_sha256s"][0] = "f" * 64
    with pytest.raises(census.CensusError, match="request_id_sha256s"):
        census.validate_event(relabeled, source="request-relabel")

    event_relabel = json.loads(json.dumps(event))
    event_relabel["event_id"] = "hydra27_fixed32:9999:0"
    event_relabel["gdn_comparator"]["census_event_id"] = event_relabel["event_id"]
    with pytest.raises(census.CensusError, match="event_id"):
        census.validate_event(event_relabel, source="event-relabel")


def test_authenticated_request_map_requires_proxy_engine_identity_parity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reducer = _load_reducer()
    task_id = reducer.B1_TASK_IDS[0]
    task_key = reducer.floor_gate.fixed32_task_key_id(task_id)
    request_digest = "a" * 64
    evidence_digest = "b" * 64
    proxy = [
        {
            "event": "attempt_result",
            "status_code": 200,
            "outcome": "response",
            "wire_id_sha256": "c" * 64,
            "task_key_id": task_key,
            "engine_request_id_sha256": request_digest,
            "evidence_sha256": evidence_digest,
        }
    ]
    engine = [
        {
            "event": event,
            "outcome": "completed" if event == "request_complete" else None,
            "wire_id_sha256": "c" * 64,
            "task_key_id": task_key,
            "engine_request_id_sha256": request_digest,
            "evidence_sha256": evidence_digest,
        }
        for event in ("request_accepted", "request_complete")
    ]

    def load(_path, *, role, **_kwargs):
        return (proxy if role == "proxy" else engine), {"role": role}

    monkeypatch.setattr(reducer.floor_gate, "load_fixed32_ingress_ledger", load)
    assert reducer._authenticated_request_task_map(
        tmp_path, [task_id]
    ) == {request_digest: task_id}

    engine[-1] = dict(engine[-1], task_key_id="0" * 64)
    with pytest.raises(reducer.GateError, match="identity parity failed"):
        reducer._authenticated_request_task_map(tmp_path, [task_id])


def test_arm_git_head_is_exact_source_commit_and_digest(tmp_path: Path) -> None:
    reducer = _load_reducer()
    commit = "a" * 40
    path = tmp_path / "git_head.txt"
    path.write_bytes(f"{commit}\n".encode("ascii"))
    assert reducer._validate_arm_git_head(tmp_path, commit) == {
        "path": "git_head.txt",
        "sha256": hashlib.sha256(f"{commit}\n".encode("ascii")).hexdigest(),
        "bytes": 41,
        "source_commit": commit,
    }


@pytest.mark.parametrize(
    "raw",
    (b"a" * 40, b"a" * 40 + b"\n\n", b"b" * 40 + b"\n", b"not-a-commit\n"),
)
def test_arm_git_head_rejects_malformed_or_mismatched_bytes(
    tmp_path: Path, raw: bytes
) -> None:
    reducer = _load_reducer()
    (tmp_path / "git_head.txt").write_bytes(raw)
    with pytest.raises(reducer.GateError, match="exact source commit"):
        reducer._validate_arm_git_head(tmp_path, "a" * 40)


def test_arm_git_head_rejects_links_and_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reducer = _load_reducer()
    commit = "a" * 40
    expected = f"{commit}\n".encode("ascii")
    target = tmp_path / "target"
    target.write_bytes(expected)
    arm = tmp_path / "arm"
    arm.mkdir()
    (arm / "git_head.txt").symlink_to(target)
    with pytest.raises(reducer.GateError, match="securely open"):
        reducer._validate_arm_git_head(arm, commit)

    (arm / "git_head.txt").unlink()
    os.link(target, arm / "git_head.txt")
    with pytest.raises(reducer.GateError, match="singly-linked"):
        reducer._validate_arm_git_head(arm, commit)

    (arm / "git_head.txt").unlink()
    original = arm / "git_head.txt"
    replacement = tmp_path / "replacement"
    original.write_bytes(expected)
    replacement.write_bytes(expected)
    calls = 0

    def swapped_open(_base: Path, _relative: str, _label: str) -> int:
        nonlocal calls
        calls += 1
        selected = original if calls == 1 else replacement
        return os.open(selected, os.O_RDONLY)

    monkeypatch.setattr(reducer, "_open_no_symlinks", swapped_open)
    with pytest.raises(reducer.GateError, match="changed while being read"):
        reducer._validate_arm_git_head(arm, commit)


def test_runtime_manifest_rejects_source_closure_tamper(tmp_path: Path) -> None:
    reducer = _load_reducer()
    repo, commit, source = _git_fixture(tmp_path)
    digest = hashlib.sha256(source).hexdigest()
    payload = _runtime_payload(commit=commit, source=source, digest=digest)
    assert reducer._validate_runtime_manifest(
        payload,
        repo=repo,
        source_commit=commit,
        git_closure={"scripts/runner.sh": digest},
        required_paths=("scripts/runner.sh",),
    ) == payload["overall_canonical_sha256"]
    with pytest.raises(reducer.GateError, match="does not bind"):
        reducer._validate_runtime_manifest(
            payload,
            repo=repo,
            source_commit=commit,
            git_closure={"scripts/runner.sh": "b" * 64},
            required_paths=("scripts/runner.sh",),
        )
    forged_identity = _runtime_payload(
        commit="0" * 40, source=source, digest=digest
    )
    with pytest.raises(reducer.GateError, match="Git/source identity"):
        reducer._validate_runtime_manifest(
            forged_identity,
            repo=repo,
            source_commit=commit,
            git_closure={"scripts/runner.sh": digest},
            required_paths=("scripts/runner.sh",),
        )


def test_git_closure_rejects_forged_commit_and_changed_worktree(
    tmp_path: Path,
) -> None:
    reducer = _load_reducer()
    repo, commit, source = _git_fixture(tmp_path)
    assert reducer._git_bound_closure(
        repo, commit, {"scripts/runner.sh"}
    ) == {"scripts/runner.sh": hashlib.sha256(source).hexdigest()}
    with pytest.raises(reducer.GateError, match="does not contain"):
        reducer._git_bound_closure(
            repo, "0" * 40, {"scripts/runner.sh"}
        )
    (repo / "scripts/runner.sh").write_bytes(source + b"# changed\n")
    with pytest.raises(reducer.GateError, match="differs from"):
        reducer._git_bound_closure(repo, commit, {"scripts/runner.sh"})


def test_runtime_manifest_source_identity_rejects_dirty_source(
    tmp_path: Path,
) -> None:
    reducer = _load_reducer()
    repo, commit, source = _git_fixture(tmp_path)
    runtime = reducer.runtime_manifest
    spec = runtime.ProfileSpec(
        host_script_source=("scripts/runner.sh",),
        python_package_source=(),
        runtime_data_and_config=(),
        required_absence=(),
        verdict_tools=(),
        package_dir="src/empty",
        package_name="empty",
        package_file_count=0,
    )
    manifest = runtime.build_manifest(
        repo,
        profile="fixed32",
        sequence="scripts/sequence.sh",
        source_commit=commit,
        spec_override=spec,
    )
    assert manifest["source_identity"]["git_commit"] == commit
    assert manifest["source_identity"]["source_file_count"] == 2
    (repo / "scripts/runner.sh").write_bytes(source + b"# dirty\n")
    with pytest.raises(runtime.ManifestError, match="current source differs"):
        runtime.build_manifest(
            repo,
            profile="fixed32",
            sequence="scripts/sequence.sh",
            source_commit=commit,
            spec_override=spec,
        )


def test_self_consistent_metadata_dataset_forgery_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reducer = _load_reducer()
    task_id = reducer.B1_TASK_IDS[0]
    fake_digest = "b" * 64
    pinned_digest = "a" * 64
    arm = tmp_path / "arm"
    task_dir = arm / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "runner_metadata.json").write_text(
        json.dumps({"fixed32_dataset_record_sha256": fake_digest}) + "\n",
        encoding="ascii",
    )
    (arm / "fixed32_chat_traffic_audit.json").write_text(
        json.dumps({"dataset_record_sha256": fake_digest}) + "\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        reducer.floor_gate,
        "pinned_dataset_record_digests",
        lambda _repo: {task_id: pinned_digest, "unrelated": "c" * 64},
    )

    def rebuild(
        _arm: Path,
        *,
        mode: str,
        subset: dict[str, object],
        dataset_record_digests: dict[str, str],
        concurrency: int,
    ) -> dict[str, str]:
        assert mode == "hydra27_fixed32"
        assert subset["task_ids"] == [task_id]
        assert concurrency == 1
        assert dataset_record_digests == {task_id: pinned_digest}
        return {"dataset_record_sha256": dataset_record_digests[task_id]}

    monkeypatch.setattr(
        reducer.floor_gate, "build_fixed32_chat_traffic_audit", rebuild
    )
    with pytest.raises(reducer.GateError, match="differs from raw task"):
        reducer._rebuild_traffic_audit(
            arm,
            mode="hydra27_fixed32",
            subset={"task_ids": [task_id]},
            concurrency=1,
        )


def test_reducer_rebuilds_qwen_ingress_and_graph_evidence() -> None:
    reducer = _load_reducer()
    source = REDUCER.read_text(encoding="ascii")
    assert "floor_gate.build_fixed32_chat_traffic_audit(" in source
    assert "floor_gate.pinned_dataset_record_digests(str(REPO))" in source
    assert "git" in source and "show" in source
    assert "runtime_manifest.build_manifest(" in source
    assert '_validate_arm_git_head(arm, args.source_commit)' in source
    assert '"arm_git_head": arm_git_head' in source
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
