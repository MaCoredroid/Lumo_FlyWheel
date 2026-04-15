from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from lumo_flywheel_serving.data_pool import (
    CodexLongGradingArtifacts,
    CodexLongLaunchArtifacts,
    DataPoolManager,
    DispatchDecision,
    IntegrityError,
    TrainingAccessViolation,
    _find_manifest_variant,
)


SCENARIO_TYPES = [
    "feature_evolution",
    "migration_refactor",
    "build_ci_breakage",
    "investigate_then_fix",
    "cross_layer_changes",
]


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _fixture_files(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, list[str]]]:
    pools_path = tmp_path / "swe_bench_pools.yaml"
    split_path = tmp_path / "split_assignment.yaml"
    manifest_path = tmp_path / "benchmark_manifest.lock"

    _write_yaml(
        pools_path,
        {
            "upstream_commit": "princeton-nlp/SWE-bench@abc1234",
            "generation_seed": 42,
            "generation_date": "2026-05-01",
            "pools": {
                "dev_bench": {
                    "tasks": [{"instance_id": "dev-1"}, {"instance_id": "dev-2"}],
                    "total": 2,
                },
                "bench_control": {
                    "tasks": [{"instance_id": "ctrl-1"}, {"instance_id": "ctrl-2"}],
                    "total": 2,
                },
                "final_test": {
                    "tasks": [{"instance_id": "final-1"}, {"instance_id": "final-2"}],
                    "total": 2,
                },
            },
        },
    )

    families_by_split = {
        "train_long": [
            "train-feature",
            "train-migration",
            "train-build",
            "train-investigate",
            "train-cross",
        ],
        "val_long": [
            "val-feature",
            "val-migration",
            "val-build",
            "val-investigate",
            "val-cross",
        ],
        "test_long": [
            "test-feature",
            "test-migration",
            "test-build",
            "test-investigate",
            "test-cross",
        ],
        "public_dev": ["public-feature", "public-migration"],
    }

    assignment = {"splits": {}}
    manifest_variants: list[dict] = []
    for split_name, family_ids in families_by_split.items():
        assignment["splits"][split_name] = {"families": []}
        for index, family_id in enumerate(family_ids):
            scenario_type = SCENARIO_TYPES[index % len(SCENARIO_TYPES)]
            variant_ids = ["v1", "v2"] if family_id == "train-feature" else ["v1"]
            assignment["splits"][split_name]["families"].append(
                {
                    "family_id": family_id,
                    "scenario_type": scenario_type,
                    "variant_ids": variant_ids,
                    "variant_count": len(variant_ids),
                }
            )
            for variant_id in variant_ids:
                manifest_variants.append(
                    {
                        "family_id": family_id,
                        "variant_id": variant_id,
                        "split": split_name,
                        "scenario_type": scenario_type,
                        "family_spec_hash": _sha256(f"family-{family_id}"),
                        "image_digest": _sha256(f"{family_id}-{variant_id}"),
                        "verifier_hash": _sha256(f"verifier-{family_id}-{variant_id}"),
                        "milestone_hashes": {"m1": _sha256(f"m1-{family_id}-{variant_id}")},
                        "agents_md_hash": _sha256(f"agents-{family_id}"),
                        "verifier_data_hash": _sha256(f"data-{family_id}"),
                    }
                )

    _write_yaml(split_path, assignment)
    split_hash = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(
        manifest_path,
        {
            "manifest_version": 3,
            "split_assignment_hash": split_hash,
            "grader_image_digest": _sha256("codex-long-grader"),
            "variants": manifest_variants,
            "change_log": [],
        },
    )
    return pools_path, split_path, manifest_path, families_by_split


def _manager(tmp_path: Path) -> tuple[DataPoolManager, dict[str, list[str]]]:
    pools_path, split_path, manifest_path, families_by_split = _fixture_files(tmp_path)
    manager = DataPoolManager(
        swe_bench_pools_path=pools_path,
        split_assignment_path=split_path,
        manifest_path=manifest_path,
        db_path=tmp_path / "run_state.db",
    )
    return manager, families_by_split


def test_load_codex_long_splits_and_public_dev_carve_out(tmp_path: Path) -> None:
    manager, families_by_split = _manager(tmp_path)
    try:
        assert manager.swe_bench_metadata["upstream_commit"] == "princeton-nlp/SWE-bench@abc1234"
        assert manager.b1_viable is False
        assert manager.codex_long_env_index["train-feature/v1"].image_digest == _sha256("train-feature-v1")
        assert {family.family_id for family in manager.list_families("train_long")} == set(families_by_split["train_long"])
        assert len(manager.list_codex_long_envs("public_dev", exclude_finished=False)) == 2
    finally:
        manager.close()


def test_find_manifest_variant_raises_on_missing_entry_and_fields(tmp_path: Path) -> None:
    _, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())

    with pytest.raises(IntegrityError, match="has no entry"):
        _find_manifest_variant(manifest, "missing-family", "v1")

    broken_manifest = yaml.safe_load(manifest_path.read_text())
    broken_manifest["variants"][0].pop("image_digest")
    with pytest.raises(IntegrityError, match="missing required fields"):
        _find_manifest_variant(broken_manifest, broken_manifest["variants"][0]["family_id"], "v1")

    metadata_broken_manifest = yaml.safe_load(manifest_path.read_text())
    metadata_broken_manifest["variants"][0].pop("split")
    with pytest.raises(IntegrityError, match="missing required fields"):
        _find_manifest_variant(metadata_broken_manifest, metadata_broken_manifest["variants"][0]["family_id"], "v1")

    duplicate_manifest = yaml.safe_load(manifest_path.read_text())
    duplicate_manifest["variants"].append(dict(duplicate_manifest["variants"][0]))
    with pytest.raises(IntegrityError, match="multiple entries"):
        _find_manifest_variant(duplicate_manifest, duplicate_manifest["variants"][0]["family_id"], "v1")

    malformed_manifest = yaml.safe_load(manifest_path.read_text())
    malformed_manifest["variants"][0].pop("family_id")
    with pytest.raises(IntegrityError, match="family_id and variant_id"):
        _find_manifest_variant(malformed_manifest, "train-feature", "v1")

    non_list_manifest = yaml.safe_load(manifest_path.read_text())
    non_list_manifest["variants"] = {"broken": True}
    with pytest.raises(IntegrityError, match="must contain a 'variants' list"):
        _find_manifest_variant(non_list_manifest, "train-feature", "v1")

    malformed_hash_manifest = yaml.safe_load(manifest_path.read_text())
    malformed_hash_manifest["variants"][0]["family_spec_hash"] = "not-a-digest"
    with pytest.raises(IntegrityError, match="family_spec_hash"):
        _find_manifest_variant(malformed_hash_manifest, malformed_hash_manifest["variants"][0]["family_id"], "v1")

    malformed_prefixed_hash_manifest = yaml.safe_load(manifest_path.read_text())
    malformed_prefixed_hash_manifest["variants"][0]["family_spec_hash"] = "sha256:not-a-real-digest"
    with pytest.raises(IntegrityError, match="64-character sha256 hex digest"):
        _find_manifest_variant(
            malformed_prefixed_hash_manifest,
            malformed_prefixed_hash_manifest["variants"][0]["family_id"],
            "v1",
        )

    # Sanity check that the split loader is actually using the frozen hash.
    split_path.write_text(split_path.read_text() + "\n# drift\n", encoding="utf-8")
    with pytest.raises(IntegrityError, match="hash mismatch"):
        DataPoolManager(
            swe_bench_pools_path=tmp_path / "swe_bench_pools.yaml",
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "broken.db",
        )


def test_manager_requires_manifest_version(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest.pop("manifest_version")
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="manifest_version"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "broken-version.db",
        )


def test_manager_requires_prefixed_split_assignment_hash(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = hashlib.sha256(split_path.read_bytes()).hexdigest()
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="split_assignment_hash"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "bare-split-hash.db",
        )


def test_manager_requires_grader_digest_and_exposes_phase_artifacts(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        launch = manager.get_codex_long_launch_artifacts("train-feature/v1")
        grading = manager.get_codex_long_grading_artifacts("train-feature/v1")

        assert launch == CodexLongLaunchArtifacts(
            scenario_id="train-feature/v1",
            family_id="train-feature",
            variant_id="v1",
            split="train_long",
            scenario_type="feature_evolution",
            manifest_version=3,
            image_digest=_sha256("train-feature-v1"),
            agents_md_hash=_sha256("agents-train-feature"),
            family_spec_hash=_sha256("family-train-feature"),
        )
        assert grading == CodexLongGradingArtifacts(
            scenario_id="train-feature/v1",
            family_id="train-feature",
            variant_id="v1",
            split="train_long",
            scenario_type="feature_evolution",
            manifest_version=3,
            grader_image_digest=_sha256("codex-long-grader"),
            verifier_hash=_sha256("verifier-train-feature-v1"),
            milestone_hashes={"m1": _sha256("m1-train-feature-v1")},
            verifier_data_hash=_sha256("data-train-feature"),
        )
    finally:
        manager.close()

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest.pop("grader_image_digest")
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="grader_image_digest"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "broken-grader.db",
        )


def test_claim_finish_retry_and_superseded_by(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        scenario_id = "train-feature/v1"
        assert manager.claim_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        assert not manager.claim_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "crash",
            grading_manifest_ver=3,
        )
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RETRY

        assert manager.claim_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            attempt=2,
            launch_manifest_ver=4,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        runs = manager._query_runs("codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1)
        assert runs[0].superseded_by == 2

        manager.finish_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            2,
            "resolved",
            grading_manifest_ver=4,
            codex_long_pass=True,
            milestone_results={"m1": True},
            snapshot_image_ref="codex-long-snapshot/train-feature/v1",
        )
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.SKIP
    finally:
        manager.close()


def test_claim_run_rejects_codex_long_metadata_drift(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        with pytest.raises(IntegrityError, match="belongs to split 'train_long'"):
            manager.claim_run(
                "codex_long",
                "val_long",
                "train-feature/v1",
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=3,
            )

        with pytest.raises(IntegrityError, match="belongs to family 'train-feature'"):
            manager.claim_run(
                "codex_long",
                "train_long",
                "train-feature/v1",
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=3,
                family_id="wrong-family",
            )

        with pytest.raises(IntegrityError, match="scenario_type 'feature_evolution'"):
            manager.claim_run(
                "codex_long",
                "train_long",
                "train-feature/v1",
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=3,
                scenario_type="migration_refactor",
            )

        with pytest.raises(IntegrityError, match="requires launch_manifest_ver"):
            manager.claim_run(
                "codex_long",
                "train_long",
                "train-feature/v1",
                "qwen3.5-27b",
                "codex",
                1,
            )
    finally:
        manager.close()


def test_invalidation_distinguishes_regrade_and_rerun(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        for scenario_id, snapshot in [("train-feature/v1", "snap-1"), ("train-feature/v2", None)]:
            assert manager.claim_run(
                "codex_long",
                "train_long",
                scenario_id,
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=3,
                family_id="train-feature",
                scenario_type="feature_evolution",
            )
            manager.finish_run(
                "codex_long",
                "train_long",
                scenario_id,
                "qwen3.5-27b",
                "codex",
                1,
                1,
                "resolved",
                grading_manifest_ver=3,
                codex_long_pass=True,
                snapshot_image_ref=snapshot,
            )

        count = manager.invalidate_stale_runs(
            family_id="train-feature",
            new_manifest_version=4,
            affected_artifact="verifier",
            reason="verifier bugfix",
            affected_variant_ids=["v1"],
            re_gate_required=True,
        )
        assert count == 1
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", "train-feature/v1", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.REGRADE_NEEDED
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", "train-feature/v2", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.SKIP

        count = manager.invalidate_stale_runs(
            family_id="train-feature",
            new_manifest_version=5,
            affected_artifact="image",
            reason="image refresh",
            affected_variant_ids=["v2"],
        )
        assert count == 1
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", "train-feature/v2", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RERUN_NEEDED
    finally:
        manager.close()


def test_invalidation_marks_running_attempts_non_current_before_finish(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        scenario_id = "train-feature/v1"
        assert manager.claim_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )

        count = manager.invalidate_stale_runs(
            family_id="train-feature",
            new_manifest_version=4,
            affected_artifact="image",
            reason="image refresh during in-flight run",
            affected_variant_ids=["v1"],
        )
        assert count == 1
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.DUPLICATE

        manager.finish_run(
            "codex_long",
            "train_long",
            scenario_id,
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "resolved",
            grading_manifest_ver=3,
            codex_long_pass=True,
            snapshot_image_ref="snap-1",
        )

        latest = manager._query_runs("codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1)[0]
        assert latest.is_current is False
        assert latest.recovery_action == "rerun_full"
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RERUN_NEEDED
    finally:
        manager.close()


def test_seal_enforcement_and_unseal(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        assert manager.list_swe_bench_tasks("final_test") == []
        assert manager.list_codex_long_envs("test_long") == []
        assert manager.check_dispatch_eligible(
            "swe_bench", "final_test", "final-1", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.BLOCKED

        manager.unseal("final_test", operator="benchmark_runner", reason="Sprint 3 B2 eval start")
        manager.unseal("test_long", operator="benchmark_runner", reason="Sprint 3 B1 eval start")
        assert [task["instance_id"] for task in manager.list_swe_bench_tasks("final_test")] == ["final-1", "final-2"]
        assert len(manager.list_codex_long_envs("test_long")) == 5
        assert len(manager.seal_state.unseal_log) == 2
    finally:
        manager.close()


def test_unseal_state_persists_across_manager_restart(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    db_path = tmp_path / "run_state.db"

    manager = DataPoolManager(
        swe_bench_pools_path=pools_path,
        split_assignment_path=split_path,
        manifest_path=manifest_path,
        db_path=db_path,
    )
    try:
        manager.unseal("final_test", operator="benchmark_runner", reason="Sprint 3 B2 eval start")
        manager.unseal("test_long", operator="benchmark_runner", reason="Sprint 3 B1 eval start")
    finally:
        manager.close()

    reloaded = DataPoolManager(
        swe_bench_pools_path=pools_path,
        split_assignment_path=split_path,
        manifest_path=manifest_path,
        db_path=db_path,
    )
    try:
        assert reloaded.seal_state.is_sealed("final_test") is False
        assert reloaded.seal_state.is_sealed("test_long") is False
        assert [task["instance_id"] for task in reloaded.list_swe_bench_tasks("final_test")] == ["final-1", "final-2"]
        assert len(reloaded.list_codex_long_envs("test_long")) == 5
        assert len(reloaded.seal_state.unseal_log) == 2
    finally:
        reloaded.close()


def test_training_access_progress_family_summary_and_matching(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        # SWE-bench training-eligible run
        assert manager.claim_run("swe_bench", "bench_control", "ctrl-1", "qwen3.5-27b", "codex", 1)
        manager.finish_run(
            "swe_bench",
            "bench_control",
            "ctrl-1",
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "resolved",
            trajectory_path="/tmp/ctrl-1.jsonl",
        )
        assert manager.claim_run("swe_bench", "dev_bench", "dev-1", "qwen3.5-27b", "codex", 1)
        manager.finish_run("swe_bench", "dev_bench", "dev-1", "qwen3.5-27b", "codex", 1, 1, "resolved")

        # Train-Long seed 1 crash then retry resolves; seed 2 resolves; second variant fails.
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "crash",
            grading_manifest_ver=3,
        )
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            attempt=2,
            launch_manifest_ver=4,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            2,
            "resolved",
            trajectory_path="/tmp/train-feature-v1-s1.jsonl",
            grading_manifest_ver=4,
            codex_long_pass=True,
            snapshot_image_ref="snap-v1-s1",
        )
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            2,
            launch_manifest_ver=4,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            2,
            1,
            "resolved",
            trajectory_path="/tmp/train-feature-v1-s2.jsonl",
            grading_manifest_ver=4,
            codex_long_pass=True,
            snapshot_image_ref="snap-v1-s2",
        )
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v2",
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=4,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v2",
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "failed",
            grading_manifest_ver=4,
            codex_long_pass=False,
        )

        # Matching SWE-Agent success on just one scenario_id.
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "swe_agent",
            1,
            launch_manifest_ver=4,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "swe_agent",
            1,
            1,
            "resolved",
            grading_manifest_ver=4,
            codex_long_pass=True,
        )

        swe_runs = manager.list_training_eligible_runs("swe_bench", "qwen3.5-27b", "codex")
        codex_runs = manager.list_training_eligible_runs("codex_long", "qwen3.5-27b", "codex")
        assert [run.scenario_id for run in swe_runs] == ["ctrl-1"]
        assert sorted(run.scenario_id for run in codex_runs) == ["train-feature/v1", "train-feature/v1"]
        assert manager.get_matched_scenario_ids("qwen3.5-27b") == ["train-feature/v1"]

        with pytest.raises(TrainingAccessViolation):
            manager.assert_training_eligible("test_long")

        summary = manager.get_family_solve_summary("train-feature", "qwen3.5-27b", "codex")
        assert summary["total_variants"] == 2
        assert summary["finished_variants"] == 2
        assert summary["solved_variants"] == 1
        assert summary["resolved_traces"] == 2
        assert summary["solved_scenario_ids"] == ["train-feature/v1"]
        assert summary["variant_solve_rate"] == 0.5

        progress = manager.get_campaign_progress("codex_long", "train_long", "qwen3.5-27b", "codex", 1)
        assert progress["resolved"] == 1
        assert progress["by_outcome"] == {"resolved": 1, "failed": 1}

        labels = manager.label_trajectory(codex_runs[0])
        assert labels["training_eligible"] is True
        assert labels["family_id"] == "train-feature"
        assert labels["variant_id"] == "v1"
    finally:
        manager.close()


def test_retryable_crashes_remain_in_listing_filters(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        assert manager.claim_run("swe_bench", "dev_bench", "dev-1", "qwen3.5-27b", "codex", 1)
        manager.finish_run("swe_bench", "dev_bench", "dev-1", "qwen3.5-27b", "codex", 1, 1, "crash")

        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="train-feature",
            scenario_type="feature_evolution",
        )
        manager.finish_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "codex",
            1,
            1,
            "crash",
            grading_manifest_ver=3,
        )

        swe_tasks = manager.list_swe_bench_tasks(
            "dev_bench",
            model_id="qwen3.5-27b",
            harness="codex",
            seed=1,
        )
        codex_envs = manager.list_codex_long_envs(
            "train_long",
            model_id="qwen3.5-27b",
            harness="codex",
            seed=1,
        )

        assert [task["instance_id"] for task in swe_tasks] == ["dev-1", "dev-2"]
        assert "train-feature/v1" in [env.scenario_id for env in codex_envs]
        assert manager.check_dispatch_eligible(
            "swe_bench", "dev_bench", "dev-1", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RETRY
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", "train-feature/v1", "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RETRY
    finally:
        manager.close()


def test_seed_assignment_defaults_follow_signed_off_policy(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        assert manager.list_assigned_seeds("swe_bench", "dev_bench", "qwen3.5-27b", "codex") == [1]
        assert manager.list_assigned_seeds("swe_bench", "bench_control", "qwen3.5-27b", "codex") == [1]
        assert manager.list_assigned_seeds("swe_bench", "final_test", "qwen3.5-27b", "codex") == [1, 2, 3]
        assert manager.list_assigned_seeds("swe_bench", "final_test", "qwen3.5-27b", "swe_agent") == [1]
        assert manager.list_assigned_seeds("codex_long", "train_long", "qwen3.5-27b", "codex") == [1, 2]
        assert manager.list_assigned_seeds("codex_long", "test_long", "qwen3.5-27b", "codex") == [1]
    finally:
        manager.close()


def test_seed_assignment_reload_supports_gate4_updates(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    seed_config_path = tmp_path / "seed_config.yaml"
    _write_yaml(
        seed_config_path,
        {
            "swe_bench": {
                "dev_bench": {"default_seeds": 1},
                "bench_control": {"default_seeds": 2, "max_seeds": 2},
                "final_test": {
                    "default_seeds": 1,
                    "overrides": [
                        {"model": "qwen3.5-27b", "harness": "codex", "seeds": 2},
                        {"model": "*", "harness": "*", "seeds": 1},
                    ],
                },
            },
            "codex_long": {
                "train_long": {"default_seeds": 3, "max_seeds": 3},
                "val_long": {"default_seeds": 1},
                "test_long": {"default_seeds": 2, "max_seeds": 2},
                "public_dev": {"default_seeds": 1},
            },
        },
    )
    try:
        assert manager.list_assigned_seeds("codex_long", "train_long", "qwen3.5-27b", "codex") == [1, 2]

        manager.reload_seed_config(seed_config_path)

        assert manager.list_assigned_seeds("swe_bench", "bench_control", "qwen3.5-27b", "codex") == [1, 2]
        assert manager.list_assigned_seeds("swe_bench", "final_test", "qwen3.5-27b", "codex") == [1, 2]
        assert manager.list_assigned_seeds("codex_long", "train_long", "qwen3.5-27b", "codex") == [1, 2, 3]
        assert manager.list_assigned_seeds("codex_long", "test_long", "qwen3.5-27b", "codex") == [1, 2]
    finally:
        manager.close()
