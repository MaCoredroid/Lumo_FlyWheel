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
TYPE_SUFFIXES = {
    "feature_evolution": "feature",
    "migration_refactor": "migration",
    "build_ci_breakage": "build",
    "investigate_then_fix": "investigate",
    "cross_layer_changes": "cross",
}


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _sha256(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()}"


def _family_specs(prefix: str, counts_by_type: list[tuple[str, int]]) -> list[tuple[str, str]]:
    families: list[tuple[str, str]] = []
    for scenario_type, count in counts_by_type:
        suffix = TYPE_SUFFIXES[scenario_type]
        for index in range(count):
            family_id = f"{prefix}-{suffix}" if index == 0 else f"{prefix}-{suffix}-{index + 1}"
            families.append((family_id, scenario_type))
    return families


def _fixture_family_specs(full_plan: bool = False) -> dict[str, list[tuple[str, str]]]:
    if full_plan:
        return {
            "train_long": _family_specs(
                "train",
                [
                    ("feature_evolution", 7),
                    ("migration_refactor", 7),
                    ("build_ci_breakage", 6),
                    ("investigate_then_fix", 5),
                    ("cross_layer_changes", 5),
                ],
            ),
            "val_long": _family_specs(
                "val",
                [
                    ("feature_evolution", 2),
                    ("migration_refactor", 2),
                    ("build_ci_breakage", 2),
                    ("investigate_then_fix", 2),
                    ("cross_layer_changes", 2),
                ],
            ),
            "test_long": _family_specs(
                "test",
                [
                    ("feature_evolution", 2),
                    ("migration_refactor", 2),
                    ("build_ci_breakage", 2),
                    ("investigate_then_fix", 2),
                    ("cross_layer_changes", 2),
                ],
            ),
            "public_dev": _family_specs(
                "public",
                [
                    ("feature_evolution", 1),
                    ("migration_refactor", 1),
                    ("build_ci_breakage", 1),
                    ("investigate_then_fix", 1),
                    ("cross_layer_changes", 1),
                ],
            ),
        }

    return {
        "train_long": _family_specs(
            "train",
            [
                ("feature_evolution", 4),
                ("migration_refactor", 4),
                ("build_ci_breakage", 4),
                ("investigate_then_fix", 4),
                ("cross_layer_changes", 4),
            ],
        ),
        "val_long": _family_specs(
            "val",
            [
                ("feature_evolution", 2),
                ("migration_refactor", 2),
                ("build_ci_breakage", 1),
                ("investigate_then_fix", 1),
                ("cross_layer_changes", 1),
            ],
        ),
        "test_long": _family_specs(
            "test",
            [
                ("feature_evolution", 2),
                ("migration_refactor", 1),
                ("build_ci_breakage", 1),
                ("investigate_then_fix", 1),
                ("cross_layer_changes", 1),
            ],
        ),
        "public_dev": _family_specs(
            "public",
            [
                ("feature_evolution", 1),
                ("migration_refactor", 1),
            ],
        ),
    }


def _fixture_files(tmp_path: Path, *, full_plan: bool = False) -> tuple[Path, Path, Path, dict[str, list[str]]]:
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

    family_specs_by_split = _fixture_family_specs(full_plan=full_plan)
    families_by_split = {
        split_name: [family_id for family_id, _scenario_type in family_specs]
        for split_name, family_specs in family_specs_by_split.items()
    }

    assignment = {
        "freeze_date": "2026-06-01",
        "seed": 42,
        "total_families": sum(len(family_ids) for family_ids in families_by_split.values()),
        "splits": {},
    }
    manifest_variants: list[dict] = []
    for split_name, family_specs in family_specs_by_split.items():
        assignment["splits"][split_name] = {"families": []}
        for family_id, scenario_type in family_specs:
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
            "freeze_date": "2026-06-01",
            "split_assignment_hash": split_hash,
            "grader_image_digest": _sha256("codex-long-grader"),
            "variants": manifest_variants,
            "change_log": [
                {
                    "manifest_version": 2,
                    "date": "2026-06-10",
                    "change": "Added train-feature/v2 after Gate 4 review",
                    "reason": "Expand low-yield family",
                    "affected_variants": ["train-feature/v2"],
                    "affected_hashes": ["image_digest", "agents_md_hash"],
                    "re_gate_required": False,
                },
                {
                    "manifest_version": 3,
                    "date": "2026-06-15",
                    "change": "Fixed train-feature verifier false negative",
                    "reason": "Trusted grading bugfix",
                    "affected_variants": ["train-feature/v1", "train-feature/v2"],
                    "affected_hashes": ["verifier_hash", "family_spec_hash"],
                    "re_gate_required": True,
                },
            ],
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


def test_manager_rejects_codex_long_freezes_below_minimum_family_floor(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["total_families"] = 34
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="minimum viable Codex-Long freeze"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "below-minimum-family-floor.db",
        )


def test_manager_rejects_full_plan_when_a_scenario_type_falls_below_six_families(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path, full_plan=True)

    split_assignment = yaml.safe_load(split_path.read_text())
    scenario_rewrites = {
        ("train_long", "train-cross-2"): "feature_evolution",
        ("train_long", "train-cross-3"): "feature_evolution",
        ("train_long", "train-cross-4"): "feature_evolution",
        ("train_long", "train-cross-5"): "feature_evolution",
        ("val_long", "val-cross"): "feature_evolution",
    }
    for split_name, family_id in scenario_rewrites:
        for family in split_assignment["splits"][split_name]["families"]:
            if family["family_id"] == family_id:
                family["scenario_type"] = scenario_rewrites[(split_name, family_id)]
                break
        else:
            raise AssertionError(f"Family {split_name}/{family_id} not found in fixture")
    _write_yaml(split_path, split_assignment)

    manifest = yaml.safe_load(manifest_path.read_text())
    for entry in manifest["variants"]:
        rewritten = scenario_rewrites.get((entry["split"], entry["family_id"]))
        if rewritten is not None:
            entry["scenario_type"] = rewritten
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="at least 6 families per scenario type"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "full-plan-type-floor.db",
        )


def test_manager_rejects_non_canonical_mid_sized_freezes(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["total_families"] = 40
    _write_yaml(split_path, split_assignment)

    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="signed-off freeze regimes"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "mid-sized-freeze.db",
        )


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

    empty_split_manifest = yaml.safe_load(manifest_path.read_text())
    empty_split_manifest["variants"][0]["split"] = ""
    with pytest.raises(IntegrityError, match="non-empty split"):
        _find_manifest_variant(empty_split_manifest, empty_split_manifest["variants"][0]["family_id"], "v1")

    bad_scenario_type_manifest = yaml.safe_load(manifest_path.read_text())
    bad_scenario_type_manifest["variants"][0]["scenario_type"] = "unknown_type"
    with pytest.raises(IntegrityError, match="unknown scenario_type"):
        _find_manifest_variant(
            bad_scenario_type_manifest,
            bad_scenario_type_manifest["variants"][0]["family_id"],
            "v1",
        )

    bad_milestone_id_manifest = yaml.safe_load(manifest_path.read_text())
    bad_milestone_id_manifest["variants"][0]["milestone_hashes"] = {"": _sha256("m1")}
    with pytest.raises(IntegrityError, match="milestone_hashes keys"):
        _find_manifest_variant(
            bad_milestone_id_manifest,
            bad_milestone_id_manifest["variants"][0]["family_id"],
            "v1",
        )

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


def test_manager_requires_change_log_entries_for_post_freeze_manifest_versions(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"] = []
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="document every post-freeze manifest_version bump"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "missing-change-log.db",
        )


def test_manager_rejects_change_log_entries_on_manifest_version_1(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["manifest_version"] = 1
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="manifest_version 1 must not include post-freeze change_log entries"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "manifest-v1-change-log.db",
        )


def test_manager_rejects_malformed_change_log_entries(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"][1]["re_gate_required"] = "yes"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="re_gate_required must be a boolean"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "bad-change-log-boolean.db",
        )

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"] = manifest["change_log"][:-1]
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="missing versions \\[3\\]"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "missing-change-log-version.db",
        )

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"][1]["affected_variants"] = [
        manifest["change_log"][1]["affected_variants"][0],
        "missing-family/v9",
    ]
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="references unknown scenario id 'missing-family/v9'"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "unknown-change-log-scenario.db",
        )

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"][1]["affected_hashes"] = ["verifier_hash", "verifier_hash"]
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="affected_hashes contains duplicate field 'verifier_hash'"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "duplicate-change-log-hash.db",
        )

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["change_log"][1]["affected_hashes"] = ["not_a_locked_hash"]
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="references unknown locked field 'not_a_locked_hash'"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "unknown-change-log-hash.db",
        )


def test_manager_rejects_duplicate_yaml_keys_in_frozen_files(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8")
        + """
freeze_date: 2026-06-02
""",
        encoding="utf-8",
    )

    with pytest.raises(IntegrityError, match="benchmark_manifest.lock must not contain duplicate YAML keys"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "duplicate-manifest-key.db",
        )

    split_dup_dir = tmp_path / "split-dup"
    split_dup_dir.mkdir()
    pools_path, split_path, manifest_path, _ = _fixture_files(split_dup_dir)
    split_path.write_text(
        split_path.read_text(encoding="utf-8")
        + """
freeze_date: 2026-06-02
""",
        encoding="utf-8",
    )
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="split_assignment.yaml must not contain duplicate YAML keys"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "duplicate-split-key.db",
        )


def test_manager_rejects_non_positive_manifest_version(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["manifest_version"] = 0
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="manifest_version must be >= 1"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "non-positive-version.db",
        )


def test_manager_requires_manifest_freeze_date(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest.pop("freeze_date")
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="freeze_date"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "missing-manifest-freeze-date.db",
        )


def test_manager_requires_iso_freeze_dates(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["freeze_date"] = "June 1, 2026"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="YYYY-MM-DD"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "bad-manifest-freeze-date.db",
        )

    split_case = tmp_path / "split-case"
    split_case.mkdir()
    pools_path, split_path, manifest_path, _ = _fixture_files(split_case)
    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["freeze_date"] = "tomorrow"
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="YYYY-MM-DD"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "bad-split-freeze-date.db",
        )


def test_manager_requires_split_assignment_freeze_metadata(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment.pop("freeze_date")
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="freeze_date"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "missing-freeze-date.db",
        )


def test_manager_rejects_freeze_date_mismatch_between_manifest_and_split_assignment(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["freeze_date"] = "2026-06-02"
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="freeze_date='2026-06-02'"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "freeze-date-mismatch.db",
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


def test_public_dev_type_carve_out_is_only_allowed_on_smaller_v1_path(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["total_families"] = 55
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="Split 'public_dev' is missing scenario types"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "public-dev-full-plan.db",
        )


def test_manager_rejects_total_family_mismatch_and_duplicate_variant_ids(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)

    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["splits"]["public_dev"]["families"].append(
        {
            "family_id": "public-build",
            "scenario_type": "build_ci_breakage",
            "variant_ids": ["v1"],
            "variant_count": 1,
        }
    )
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["variants"].append(
        {
            "family_id": "public-build",
            "variant_id": "v1",
            "split": "public_dev",
            "scenario_type": "build_ci_breakage",
            "family_spec_hash": _sha256("family-public-build"),
            "image_digest": _sha256("public-build-v1"),
            "verifier_hash": _sha256("verifier-public-build-v1"),
            "milestone_hashes": {"m1": _sha256("m1-public-build-v1")},
            "agents_md_hash": _sha256("agents-public-build"),
            "verifier_data_hash": _sha256("data-public-build"),
        }
    )
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="total_families mismatch"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "family-count-mismatch.db",
        )

    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    split_assignment = yaml.safe_load(split_path.read_text())
    split_assignment["splits"]["train_long"]["families"][0]["variant_ids"] = ["v1", "v1"]
    split_assignment["splits"]["train_long"]["families"][0]["variant_count"] = 2
    _write_yaml(split_path, split_assignment)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["split_assignment_hash"] = f"sha256:{hashlib.sha256(split_path.read_bytes()).hexdigest()}"
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="duplicate variant_id 'v1'"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "duplicate-variant-ids.db",
        )


def test_manager_rejects_manifest_variants_outside_frozen_split_assignment(tmp_path: Path) -> None:
    pools_path, split_path, manifest_path, _ = _fixture_files(tmp_path)
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["variants"].append(
        {
            "family_id": "orphan-family",
            "variant_id": "v1",
            "split": "train_long",
            "scenario_type": "feature_evolution",
            "family_spec_hash": _sha256("family-orphan-family"),
            "image_digest": _sha256("orphan-family-v1"),
            "verifier_hash": _sha256("verifier-orphan-family-v1"),
            "milestone_hashes": {"m1": _sha256("m1-orphan-family-v1")},
            "agents_md_hash": _sha256("agents-orphan-family"),
            "verifier_data_hash": _sha256("data-orphan-family"),
        }
    )
    _write_yaml(manifest_path, manifest)

    with pytest.raises(IntegrityError, match="not present in split_assignment.yaml"):
        DataPoolManager(
            swe_bench_pools_path=pools_path,
            split_assignment_path=split_path,
            manifest_path=manifest_path,
            db_path=tmp_path / "orphan-manifest-variant.db",
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
            launch_manifest_ver=3,
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
            grading_manifest_ver=3,
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
        with pytest.raises(IntegrityError, match="manifest_version mismatch"):
            manager.claim_run(
                "codex_long",
                "train_long",
                "train-feature/v1",
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=2,
            )

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


def test_finish_run_rejects_missing_or_stale_grading_manifest_version(tmp_path: Path) -> None:
    missing_case = tmp_path / "missing"
    missing_case.mkdir()
    manager, _ = _manager(missing_case)
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

        with pytest.raises(IntegrityError, match="requires grading_manifest_ver"):
            manager.finish_run(
                "codex_long",
                "train_long",
                scenario_id,
                "qwen3.5-27b",
                "codex",
                1,
                1,
                "failed",
            )
    finally:
        manager.close()

    stale_case = tmp_path / "stale"
    stale_case.mkdir()
    manager, _ = _manager(stale_case)
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

        with pytest.raises(IntegrityError, match="manifest_version mismatch"):
            manager.finish_run(
                "codex_long",
                "train_long",
                scenario_id,
                "qwen3.5-27b",
                "codex",
                1,
                1,
                "failed",
                grading_manifest_ver=2,
            )
    finally:
        manager.close()


def test_finish_run_requires_snapshot_for_non_crash_codex_long_outcomes(tmp_path: Path) -> None:
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

        with pytest.raises(IntegrityError, match="requires snapshot_image_ref"):
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
            )
    finally:
        manager.close()


def test_invalidation_distinguishes_regrade_and_rerun(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        for scenario_id, snapshot in [("train-feature/v1", "snap-1"), ("train-feature/v2", "snap-2")]:
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


def test_regrade_downgrades_to_rerun_for_legacy_rows_missing_snapshot(tmp_path: Path) -> None:
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
        manager.db.execute(
            """
            UPDATE runs
            SET snapshot_image_ref = NULL
            WHERE track = 'codex_long' AND pool_or_split = 'train_long' AND scenario_id = ?
              AND model_id = 'qwen3.5-27b' AND harness = 'codex' AND seed = 1 AND attempt = 1
            """,
            (scenario_id,),
        )
        manager.db.connection.commit()

        count = manager.invalidate_stale_runs(
            family_id="train-feature",
            new_manifest_version=4,
            affected_artifact="verifier",
            reason="legacy run missing retained snapshot",
            affected_variant_ids=["v1"],
        )
        assert count == 1
        assert manager.check_dispatch_eligible(
            "codex_long", "train_long", scenario_id, "qwen3.5-27b", "codex", 1
        ) is DispatchDecision.RERUN_NEEDED
    finally:
        manager.close()


def test_invalidation_rejects_unknown_or_malformed_scope(tmp_path: Path) -> None:
    manager, _ = _manager(tmp_path)
    try:
        with pytest.raises(KeyError, match="Unknown family_id 'missing-family'"):
            manager.invalidate_stale_runs(
                family_id="missing-family",
                new_manifest_version=4,
                affected_artifact="verifier",
                reason="bugfix",
            )

        with pytest.raises(IntegrityError, match="does not define affected_variant_ids"):
            manager.invalidate_stale_runs(
                family_id="train-feature",
                new_manifest_version=4,
                affected_artifact="verifier",
                reason="bugfix",
                affected_variant_ids=["v3"],
            )

        with pytest.raises(IntegrityError, match="duplicate variant_id 'v1'"):
            manager.invalidate_stale_runs(
                family_id="train-feature",
                new_manifest_version=4,
                affected_artifact="verifier",
                reason="bugfix",
                affected_variant_ids=["v1", "v1"],
            )

        with pytest.raises(IntegrityError, match="empty affected_variant_ids scope"):
            manager.invalidate_stale_runs(
                family_id="train-feature",
                new_manifest_version=4,
                affected_artifact="verifier",
                reason="bugfix",
                affected_variant_ids=[],
            )

        with pytest.raises(IntegrityError, match="non-empty variant ids"):
            manager.invalidate_stale_runs(
                family_id="train-feature",
                new_manifest_version=4,
                affected_artifact="verifier",
                reason="bugfix",
                affected_variant_ids=[""],
            )
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
        with pytest.raises(IntegrityError, match="sealed pool/split 'final_test'"):
            manager.claim_run("swe_bench", "final_test", "final-1", "qwen3.5-27b", "codex", 1)
        with pytest.raises(IntegrityError, match="sealed pool/split 'test_long'"):
            manager.claim_run(
                "codex_long",
                "test_long",
                "test-cross/v1",
                "qwen3.5-27b",
                "codex",
                1,
                launch_manifest_ver=3,
                family_id="test-cross",
                scenario_type="cross_layer_changes",
            )

        manager.unseal("final_test", operator="benchmark_runner", reason="Sprint 3 B2 eval start")
        manager.unseal("test_long", operator="benchmark_runner", reason="Sprint 3 B1 eval start")
        assert [task["instance_id"] for task in manager.list_swe_bench_tasks("final_test")] == ["final-1", "final-2"]
        assert len(manager.list_codex_long_envs("test_long")) == 6
        assert manager.claim_run("swe_bench", "final_test", "final-1", "qwen3.5-27b", "codex", 1)
        assert manager.claim_run(
            "codex_long",
            "test_long",
            "test-cross/v1",
            "qwen3.5-27b",
            "codex",
            1,
            launch_manifest_ver=3,
            family_id="test-cross",
            scenario_type="cross_layer_changes",
        )
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
        assert len(reloaded.list_codex_long_envs("test_long")) == 6
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
            2,
            "resolved",
            trajectory_path="/tmp/train-feature-v1-s1.jsonl",
            grading_manifest_ver=3,
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
            2,
            1,
            "resolved",
            trajectory_path="/tmp/train-feature-v1-s2.jsonl",
            grading_manifest_ver=3,
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
            launch_manifest_ver=3,
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
            grading_manifest_ver=3,
            codex_long_pass=False,
            snapshot_image_ref="snap-v2-s1",
        )

        # Matching SWE-Agent success on just one scenario_id.
        assert manager.claim_run(
            "codex_long",
            "train_long",
            "train-feature/v1",
            "qwen3.5-27b",
            "swe_agent",
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
            "swe_agent",
            1,
            1,
            "resolved",
            grading_manifest_ver=3,
            codex_long_pass=True,
            snapshot_image_ref="snap-swe-agent-v1-s1",
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
