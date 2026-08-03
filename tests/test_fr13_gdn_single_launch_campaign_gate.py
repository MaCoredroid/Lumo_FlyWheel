from __future__ import annotations

import hashlib
import importlib.util
import json
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
    assert common.count('--source-commit "$SOURCE_COMMIT"') == 3
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

    payload["graph_id"] = 404
    with pytest.raises(reducer.GateError, match="record shape drifted"):
        reducer._validate_live_pass(
            payload,
            mode="tail6_fixed32",
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
