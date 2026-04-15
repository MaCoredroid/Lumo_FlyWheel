from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

from .yaml_utils import DuplicateKeyError, load_yaml_file

logger = logging.getLogger(__name__)

DEFAULT_SEED_CONFIG: dict[str, Any] = {
    "swe_bench": {
        "dev_bench": {"default_seeds": 1},
        "bench_control": {"default_seeds": 1, "max_seeds": 2},
        "final_test": {
            "default_seeds": 1,
            "overrides": [
                {"model": "qwen3.5-27b", "harness": "codex", "seeds": 3},
                {"model": "codex-sft-all", "harness": "codex", "seeds": 3},
                {"model": "*", "harness": "*", "seeds": 1},
            ],
        },
    },
    "codex_long": {
        "train_long": {"default_seeds": 2, "max_seeds": 3},
        "val_long": {"default_seeds": 1},
        "test_long": {"default_seeds": 1, "max_seeds": 2},
        "public_dev": {"default_seeds": 1},
    },
}

SCENARIO_TYPES = {
    "feature_evolution",
    "migration_refactor",
    "build_ci_breakage",
    "investigate_then_fix",
    "cross_layer_changes",
}
MIN_CODEX_LONG_FAMILIES = 35
FULL_PLAN_CODEX_LONG_FAMILIES = 55
FULL_PLAN_MIN_FAMILIES_PER_TYPE = 6
SMALLER_V1_SPLIT_FAMILY_COUNTS = {
    "train_long": 20,
    "val_long": 7,
    "test_long": 6,
    "public_dev": 2,
}
TRAINING_ELIGIBLE = {"bench_control", "train_long"}
SEALABLE_POOLS = {"final_test", "test_long"}
_ARTIFACT_RECOVERY = {
    "verifier": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "milestone": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "verifier_data": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "family_spec": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "grader_image": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "image": {"ver_column": "launch_manifest_ver", "recovery": "rerun_full"},
    "agents_md": {"ver_column": "launch_manifest_ver", "recovery": "rerun_full"},
}
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FAMILY_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_VARIANT_ID_RE = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_KNOWN_CHANGE_LOG_HASH_FIELDS = {
    "split_assignment_hash",
    "grader_image_digest",
    "family_spec_hash",
    "image_digest",
    "verifier_hash",
    "milestone_hashes",
    "agents_md_hash",
    "verifier_data_hash",
}


class IntegrityError(RuntimeError):
    """Raised when frozen benchmark artifacts disagree or are incomplete."""


class TrainingAccessViolation(RuntimeError):
    """Raised when a caller attempts to use a non-training split for training."""


class DispatchDecision(Enum):
    PROCEED = "proceed"
    SKIP = "skip"
    RETRY = "retry"
    REGRADE_NEEDED = "regrade_needed"
    RERUN_NEEDED = "rerun_needed"
    BLOCKED = "blocked"
    DUPLICATE = "duplicate"


@dataclass(frozen=True)
class CodexLongFamily:
    family_id: str
    scenario_type: str
    split: str
    variant_ids: tuple[str, ...]
    variant_count: int
    manifest_version: int


@dataclass(frozen=True)
class CodexLongEnv:
    family_id: str
    variant_id: str
    scenario_id: str
    split: str
    scenario_type: str
    image_digest: str


@dataclass(frozen=True)
class CodexLongLaunchArtifacts:
    scenario_id: str
    family_id: str
    variant_id: str
    split: str
    scenario_type: str
    manifest_version: int
    image_digest: str
    agents_md_hash: str
    family_spec_hash: str


@dataclass(frozen=True)
class CodexLongGradingArtifacts:
    scenario_id: str
    family_id: str
    variant_id: str
    split: str
    scenario_type: str
    manifest_version: int
    grader_image_digest: str
    verifier_hash: str
    milestone_hashes: dict[str, str]
    verifier_data_hash: str


@dataclass
class RunRecord:
    track: str
    pool_or_split: str
    scenario_id: str
    model_id: str
    harness: str
    seed: int
    attempt: int
    exec_state: str
    outcome: str | None
    started_at: str | None
    completed_at: str | None
    wall_time_seconds: float | None
    trajectory_path: str | None
    family_id: str | None
    scenario_type: str | None
    launch_manifest_ver: int | None
    grading_manifest_ver: int | None
    is_current: bool
    superseded_by: int | None
    recovery_action: str | None
    snapshot_image_ref: str | None
    re_gate_required: bool
    codex_long_pass: bool | None
    milestone_results: dict[str, Any] | None


@dataclass(frozen=True)
class Gate4Outcome:
    total_families: int
    b1_viable: bool
    projected_codex_traces: int
    projected_wall_clock_days: float
    projected_matched_ids: int
    projected_matched_families: int
    b2_viable: bool
    gate4_decision: str
    recorded_at: str


def _utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_scenario_id(family_id: str, variant_id: str) -> str:
    return f"{family_id}/{variant_id}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256_value(value: Any, *, field_name: str, allow_bare: bool = False) -> str:
    if not isinstance(value, str) or not value:
        raise IntegrityError(f"{field_name} must be a non-empty string")
    if value.startswith("sha256:"):
        digest = value.removeprefix("sha256:")
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise IntegrityError(f"{field_name} must include a 64-character sha256 hex digest")
        return digest
    if allow_bare:
        if not _SHA256_HEX_RE.fullmatch(value):
            raise IntegrityError(f"{field_name} must be recorded as a 64-character sha256 hex digest")
        return value
    raise IntegrityError(f"{field_name} must be recorded as a sha256 digest")


def _require_iso_date(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{field_name} must record a non-empty freeze date")
    if not _ISO_DATE_RE.fullmatch(value):
        raise IntegrityError(f"{field_name} must use ISO date format YYYY-MM-DD")
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise IntegrityError(f"{field_name} must use a real calendar date in YYYY-MM-DD format") from exc
    return value


def _require_non_empty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise IntegrityError(f"{field_name} must be a non-empty string")
    return value


def _require_family_id(value: Any, *, field_name: str) -> str:
    family_id = _require_non_empty_string(value, field_name=field_name)
    if not _FAMILY_ID_RE.fullmatch(family_id):
        raise IntegrityError(f"{field_name} must be kebab-case and must not contain '/'")
    return family_id


def _require_variant_id(value: Any, *, field_name: str) -> str:
    variant_id = _require_non_empty_string(value, field_name=field_name)
    if not _VARIANT_ID_RE.fullmatch(variant_id):
        raise IntegrityError(
            f"{field_name} must be a slash-free lowercase slug using letters, digits, '.', '_' or '-'"
        )
    return variant_id


def load_codex_long_manifest(path: str | Path) -> dict[str, Any]:
    try:
        manifest = load_yaml_file(path) or {}
    except DuplicateKeyError as exc:
        raise IntegrityError(f"benchmark_manifest.lock must not contain duplicate YAML keys: {exc}") from exc
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc
    if not isinstance(manifest, dict):
        raise IntegrityError("benchmark_manifest.lock must be a YAML mapping")
    try:
        manifest["manifest_version"] = int(manifest["manifest_version"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("benchmark_manifest.lock must record an integer manifest_version") from exc
    if manifest["manifest_version"] < 1:
        raise IntegrityError("benchmark_manifest.lock manifest_version must be >= 1")
    _require_iso_date(manifest.get("freeze_date"), field_name="benchmark_manifest.lock freeze_date")
    _require_non_empty_string(manifest.get("generator"), field_name="benchmark_manifest.lock generator")

    _require_sha256_value(manifest.get("split_assignment_hash"), field_name="split_assignment_hash")
    _require_sha256_value(manifest.get("grader_image_digest"), field_name="grader_image_digest")

    change_log = manifest.get("change_log", [])
    if not isinstance(change_log, list):
        raise IntegrityError("benchmark_manifest.lock change_log must be a list")
    seen_change_versions: set[int] = set()
    for index, entry in enumerate(change_log):
        if not isinstance(entry, dict):
            raise IntegrityError(f"benchmark_manifest.lock change_log[{index}] must be a mapping")
        try:
            change_version = int(entry["manifest_version"])
        except (KeyError, TypeError, ValueError) as exc:
            raise IntegrityError(
                f"benchmark_manifest.lock change_log[{index}] must record an integer manifest_version"
            ) from exc
        if change_version < 2:
            raise IntegrityError(
                f"benchmark_manifest.lock change_log[{index}] manifest_version must be >= 2"
            )
        if change_version in seen_change_versions:
            raise IntegrityError(
                f"benchmark_manifest.lock change_log contains duplicate manifest_version {change_version}"
            )
        seen_change_versions.add(change_version)
        _require_iso_date(entry.get("date"), field_name=f"benchmark_manifest.lock change_log[{index}] date")
        _require_non_empty_string(
            entry.get("change"),
            field_name=f"benchmark_manifest.lock change_log[{index}] change",
        )
        _require_non_empty_string(
            entry.get("reason"),
            field_name=f"benchmark_manifest.lock change_log[{index}] reason",
        )
        affected_variants = entry.get("affected_variants")
        if not isinstance(affected_variants, list) or not affected_variants:
            raise IntegrityError(
                f"benchmark_manifest.lock change_log[{index}] affected_variants must be a non-empty list"
            )
        for variant_index, scenario_id in enumerate(affected_variants):
            if not isinstance(scenario_id, str) or not scenario_id.strip():
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_variants[{variant_index}] must be a non-empty scenario id"
                )
        affected_hashes = entry.get("affected_hashes")
        if not isinstance(affected_hashes, list) or not affected_hashes:
            raise IntegrityError(
                f"benchmark_manifest.lock change_log[{index}] affected_hashes must be a non-empty list"
            )
        for hash_index, field_name in enumerate(affected_hashes):
            if not isinstance(field_name, str) or not field_name.strip():
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_hashes[{hash_index}] must be a non-empty string"
                )
        re_gate_required = entry.get("re_gate_required")
        if not isinstance(re_gate_required, bool):
            raise IntegrityError(
                f"benchmark_manifest.lock change_log[{index}] re_gate_required must be a boolean"
            )
    if manifest["manifest_version"] == 1:
        if seen_change_versions:
            raise IntegrityError(
                "benchmark_manifest.lock manifest_version 1 must not include post-freeze change_log entries"
            )
    else:
        expected_versions = set(range(2, manifest["manifest_version"] + 1))
        if seen_change_versions != expected_versions:
            missing_versions = sorted(expected_versions - seen_change_versions)
            extra_versions = sorted(seen_change_versions - expected_versions)
            details: list[str] = []
            if missing_versions:
                details.append(f"missing versions {missing_versions}")
            if extra_versions:
                details.append(f"unexpected versions {extra_versions}")
            rendered = ", ".join(details) if details else "inconsistent version coverage"
            raise IntegrityError(
                "benchmark_manifest.lock change_log must document every post-freeze manifest_version bump; "
                f"{rendered}"
            )
    manifest["change_log"] = change_log

    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise IntegrityError("benchmark_manifest.lock must contain a 'variants' list")
    seen_scenario_ids: set[str] = set()
    for index, entry in enumerate(variants):
        if not isinstance(entry, dict):
            raise IntegrityError(f"Manifest variants[{index}] must be a mapping")
        family_id = entry.get("family_id")
        variant_id = entry.get("variant_id")
        if family_id is None or variant_id is None:
            raise IntegrityError(
                f"Manifest variants[{index}] must include string family_id and variant_id fields"
            )
        family_id = _require_family_id(family_id, field_name=f"Manifest variants[{index}] family_id")
        variant_id = _require_variant_id(variant_id, field_name=f"Manifest variants[{index}] variant_id")
        scenario_id = make_scenario_id(family_id, variant_id)
        if scenario_id in seen_scenario_ids:
            raise IntegrityError(f"Variant '{scenario_id}' has multiple entries in benchmark_manifest.lock")
        seen_scenario_ids.add(scenario_id)

    for index, entry in enumerate(change_log):
        affected_variants = entry["affected_variants"]
        seen_change_scenarios: set[str] = set()
        for variant_index, scenario_id in enumerate(affected_variants):
            if scenario_id in seen_change_scenarios:
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_variants contains duplicate scenario id '{scenario_id}'"
                )
            seen_change_scenarios.add(scenario_id)
            if scenario_id not in seen_scenario_ids:
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_variants[{variant_index}] references unknown scenario id '{scenario_id}'"
                )

        affected_hashes = entry["affected_hashes"]
        seen_hash_fields: set[str] = set()
        for hash_index, field_name in enumerate(affected_hashes):
            if field_name in seen_hash_fields:
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_hashes contains duplicate field '{field_name}'"
                )
            seen_hash_fields.add(field_name)
            if field_name not in _KNOWN_CHANGE_LOG_HASH_FIELDS:
                raise IntegrityError(
                    "benchmark_manifest.lock change_log"
                    f"[{index}] affected_hashes[{hash_index}] references unknown locked field '{field_name}'"
                )
    return manifest


def _find_manifest_variant(manifest: dict[str, Any], family_id: str, variant_id: str) -> dict[str, Any]:
    variants = manifest.get("variants")
    if not isinstance(variants, list):
        raise IntegrityError("benchmark_manifest.lock must contain a 'variants' list")

    matches: list[dict[str, Any]] = []
    for index, entry in enumerate(variants):
        if not isinstance(entry, dict):
            raise IntegrityError(f"Manifest variants[{index}] must be a mapping")

        entry_family_id = entry.get("family_id")
        entry_variant_id = entry.get("variant_id")
        if entry_family_id is None or entry_variant_id is None:
            raise IntegrityError(
                f"Manifest variants[{index}] must include string family_id and variant_id fields"
            )
        entry_family_id = _require_family_id(
            entry_family_id,
            field_name=f"Manifest variants[{index}] family_id",
        )
        entry_variant_id = _require_variant_id(
            entry_variant_id,
            field_name=f"Manifest variants[{index}] variant_id",
        )

        if entry_family_id == family_id and entry_variant_id == variant_id:
            matches.append(entry)

    if len(matches) > 1:
        raise IntegrityError(
            f"Variant '{family_id}/{variant_id}' has multiple entries in benchmark_manifest.lock"
        )
    if matches:
        entry = matches[0]
        required_fields = [
            "split",
            "scenario_type",
            "image_digest",
            "verifier_hash",
            "family_spec_hash",
            "agents_md_hash",
            "verifier_data_hash",
            "milestone_hashes",
        ]
        missing = [field for field in required_fields if field not in entry]
        if missing:
            raise IntegrityError(
                f"Manifest entry for '{family_id}/{variant_id}' is missing required fields: {missing}"
            )
        split = entry["split"]
        if not isinstance(split, str) or not split:
            raise IntegrityError(f"Manifest entry for '{family_id}/{variant_id}' must include a non-empty split")
        if split not in {"train_long", "val_long", "test_long", "public_dev"}:
            raise IntegrityError(
                f"Manifest entry for '{family_id}/{variant_id}' has unknown split '{split}'"
            )
        scenario_type = entry["scenario_type"]
        if not isinstance(scenario_type, str) or not scenario_type:
            raise IntegrityError(
                f"Manifest entry for '{family_id}/{variant_id}' must include a non-empty scenario_type"
            )
        if scenario_type not in SCENARIO_TYPES:
            raise IntegrityError(
                f"Manifest entry for '{family_id}/{variant_id}' has unknown scenario_type '{scenario_type}'"
            )
        _require_sha256_value(entry["image_digest"], field_name=f"{family_id}/{variant_id} image_digest")
        _require_sha256_value(entry["verifier_hash"], field_name=f"{family_id}/{variant_id} verifier_hash")
        _require_sha256_value(entry["family_spec_hash"], field_name=f"{family_id}/{variant_id} family_spec_hash")
        _require_sha256_value(entry["agents_md_hash"], field_name=f"{family_id}/{variant_id} agents_md_hash")
        _require_sha256_value(
            entry["verifier_data_hash"],
            field_name=f"{family_id}/{variant_id} verifier_data_hash",
        )
        milestone_hashes = entry["milestone_hashes"]
        if not isinstance(milestone_hashes, dict) or not milestone_hashes:
            raise IntegrityError(
                f"Manifest entry for '{family_id}/{variant_id}' must include non-empty milestone_hashes"
            )
        for milestone_id, milestone_hash in milestone_hashes.items():
            if not isinstance(milestone_id, str) or not milestone_id:
                raise IntegrityError(
                    f"Manifest entry for '{family_id}/{variant_id}' milestone_hashes keys must be non-empty strings"
                )
            _require_sha256_value(
                milestone_hash,
                field_name=f"{family_id}/{variant_id} milestone_hashes[{milestone_id}]",
            )
        return entry

    raise IntegrityError(
        f"Variant '{family_id}/{variant_id}' appears in split_assignment.yaml but has no entry in "
        "benchmark_manifest.lock. Cannot verify image digest or verifier hash."
    )


def load_swe_bench_pools(path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    try:
        raw = load_yaml_file(path) or {}
    except DuplicateKeyError as exc:
        raise IntegrityError(f"swe_bench_pools.yaml must not contain duplicate YAML keys: {exc}") from exc
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc
    if not isinstance(raw, dict):
        raise IntegrityError("swe_bench_pools.yaml must be a YAML mapping")
    upstream_commit = raw.get("upstream_commit")
    if not upstream_commit:
        raise IntegrityError("swe_bench_pools.yaml must record upstream_commit")
    pools_raw = raw.get("pools")
    if not isinstance(pools_raw, dict):
        raise IntegrityError("swe_bench_pools.yaml is missing a 'pools' mapping")

    pools: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[str] = set()
    for pool_name in ("dev_bench", "bench_control", "final_test"):
        pool = pools_raw.get(pool_name)
        if not isinstance(pool, dict):
            raise IntegrityError(f"Missing pool '{pool_name}' in swe_bench_pools.yaml")
        tasks = list(pool.get("tasks", []))
        declared_total = int(pool.get("total", len(tasks)))
        if declared_total != len(tasks):
            raise IntegrityError(
                f"Pool '{pool_name}' declares total={declared_total} but contains {len(tasks)} task entries"
            )
        pool_ids = {task["instance_id"] for task in tasks}
        overlap = seen_ids & pool_ids
        if overlap:
            raise IntegrityError(f"Pool '{pool_name}' overlaps prior pools: {sorted(overlap)}")
        seen_ids.update(pool_ids)
        pools[pool_name] = tasks

    metadata = {
        "upstream_commit": upstream_commit,
        "generation_seed": raw.get("generation_seed"),
        "generation_date": raw.get("generation_date"),
    }
    return pools, metadata


def load_codex_long_splits(
    split_assignment_path: str | Path,
    manifest_path: str | Path,
) -> tuple[dict[str, list[CodexLongFamily]], dict[str, CodexLongEnv]]:
    try:
        assignment = load_yaml_file(split_assignment_path) or {}
    except DuplicateKeyError as exc:
        raise IntegrityError(f"split_assignment.yaml must not contain duplicate YAML keys: {exc}") from exc
    except ValueError as exc:
        raise IntegrityError(str(exc)) from exc
    if not isinstance(assignment, dict):
        raise IntegrityError("split_assignment.yaml must be a YAML mapping")
    manifest = load_codex_long_manifest(manifest_path)

    freeze_date = _require_iso_date(assignment.get("freeze_date"), field_name="split_assignment.yaml freeze_date")
    manifest_freeze_date = manifest["freeze_date"]
    if freeze_date != manifest_freeze_date:
        raise IntegrityError(
            "Frozen benchmark metadata disagreement: "
            f"split_assignment.yaml freeze_date='{freeze_date}', "
            f"benchmark_manifest.lock freeze_date='{manifest_freeze_date}'"
        )
    try:
        assignment["seed"] = int(assignment["seed"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("split_assignment.yaml must record an integer seed") from exc
    try:
        declared_total_families = int(assignment["total_families"])
    except (KeyError, TypeError, ValueError) as exc:
        raise IntegrityError("split_assignment.yaml must record an integer total_families") from exc
    if declared_total_families < MIN_CODEX_LONG_FAMILIES:
        raise IntegrityError(
            "split_assignment.yaml total_families must be >= "
            f"{MIN_CODEX_LONG_FAMILIES} for the minimum viable Codex-Long freeze; "
            f"got {declared_total_families}"
        )
    if MIN_CODEX_LONG_FAMILIES < declared_total_families < FULL_PLAN_CODEX_LONG_FAMILIES:
        raise IntegrityError(
            "split_assignment.yaml total_families must use one of the signed-off freeze regimes: "
            f"exactly {MIN_CODEX_LONG_FAMILIES} for the smaller-v1 path, or >= {FULL_PLAN_CODEX_LONG_FAMILIES} "
            f"for the full plan; got {declared_total_families}"
        )

    actual_hash = sha256_file(split_assignment_path)
    expected_hash = _require_sha256_value(
        manifest.get("split_assignment_hash"),
        field_name="split_assignment_hash",
    )
    if actual_hash != expected_hash:
        raise IntegrityError(
            f"split_assignment.yaml hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    all_family_ids: set[str] = set()
    all_scenario_ids: set[str] = set()
    splits: dict[str, list[CodexLongFamily]] = {}
    env_index: dict[str, CodexLongEnv] = {}
    manifest_version = manifest["manifest_version"]
    total_families_loaded = 0
    smaller_v1_public_dev_carve_out = declared_total_families == MIN_CODEX_LONG_FAMILIES
    family_type_counts = {scenario_type: 0 for scenario_type in SCENARIO_TYPES}

    split_mapping = assignment.get("splits")
    if not isinstance(split_mapping, dict):
        raise IntegrityError("split_assignment.yaml is missing a 'splits' mapping")

    for split_name in ("train_long", "val_long", "test_long", "public_dev"):
        split_data = split_mapping.get(split_name)
        if not isinstance(split_data, dict):
            raise IntegrityError(f"split_assignment.yaml is missing split '{split_name}'")
        families_raw = split_data.get("families", [])
        if not isinstance(families_raw, list):
            raise IntegrityError(f"split_assignment.yaml split '{split_name}' must include a 'families' list")
        families: list[CodexLongFamily] = []
        for family_index, family in enumerate(families_raw):
            if not isinstance(family, dict):
                raise IntegrityError(
                    f"split_assignment.yaml split '{split_name}' families[{family_index}] must be a mapping"
                )
            family_id = _require_family_id(
                family.get("family_id"),
                field_name=f"split_assignment.yaml split '{split_name}' families[{family_index}] family_id",
            )
            if family_id in all_family_ids:
                raise IntegrityError(f"Family '{family_id}' appears in multiple splits")
            all_family_ids.add(family_id)

            scenario_type = family.get("scenario_type")
            if not isinstance(scenario_type, str) or not scenario_type:
                raise IntegrityError(f"Family '{family_id}' must include a non-empty scenario_type")
            if scenario_type not in SCENARIO_TYPES:
                raise IntegrityError(f"Family '{family_id}' has unknown scenario_type '{scenario_type}'")

            variant_ids_raw = family.get("variant_ids", [])
            if not isinstance(variant_ids_raw, list):
                raise IntegrityError(f"Family '{family_id}' variant_ids must be a list")
            if not variant_ids_raw:
                raise IntegrityError(f"Family '{family_id}' must define at least one variant_id")
            variant_ids_list: list[str] = []
            seen_variant_ids: set[str] = set()
            for variant_index, variant_id in enumerate(variant_ids_raw):
                variant_id = _require_variant_id(
                    variant_id,
                    field_name=f"Family '{family_id}' variant_ids[{variant_index}]",
                )
                if variant_id in seen_variant_ids:
                    raise IntegrityError(f"Family '{family_id}' contains duplicate variant_id '{variant_id}'")
                seen_variant_ids.add(variant_id)
                variant_ids_list.append(variant_id)
            variant_ids = tuple(variant_ids_list)
            try:
                variant_count = int(family["variant_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise IntegrityError(f"Family '{family_id}' must declare an integer variant_count") from exc
            if variant_count != len(variant_ids):
                raise IntegrityError(
                    f"Family '{family_id}' declares variant_count={variant_count} but lists {len(variant_ids)} variants"
                )

            families.append(
                CodexLongFamily(
                    family_id=family_id,
                    scenario_type=scenario_type,
                    split=split_name,
                    variant_ids=variant_ids,
                    variant_count=variant_count,
                    manifest_version=manifest_version,
                )
            )
            total_families_loaded += 1
            family_type_counts[scenario_type] += 1

            for variant_id in variant_ids:
                scenario_id = make_scenario_id(family_id, variant_id)
                if scenario_id in all_scenario_ids:
                    raise IntegrityError(f"Duplicate scenario_id '{scenario_id}'")
                all_scenario_ids.add(scenario_id)

                manifest_entry = _find_manifest_variant(manifest, family_id, variant_id)
                if manifest_entry.get("split") and manifest_entry["split"] != split_name:
                    raise IntegrityError(
                        f"Metadata disagreement for '{scenario_id}': split_assignment.yaml says split='{split_name}', "
                        f"benchmark_manifest.lock says split='{manifest_entry['split']}'"
                    )
                if (
                    manifest_entry.get("scenario_type")
                    and manifest_entry["scenario_type"] != scenario_type
                ):
                    raise IntegrityError(
                        f"Metadata disagreement for '{scenario_id}': split_assignment.yaml says "
                        f"scenario_type='{scenario_type}', benchmark_manifest.lock says "
                        f"scenario_type='{manifest_entry['scenario_type']}'"
                    )
                env_index[scenario_id] = CodexLongEnv(
                    family_id=family_id,
                    variant_id=variant_id,
                    scenario_id=scenario_id,
                    split=split_name,
                    scenario_type=scenario_type,
                    image_digest=manifest_entry["image_digest"],
                )

        types_present = {family.scenario_type for family in families}
        missing_types = SCENARIO_TYPES - types_present
        if missing_types:
            if split_name == "public_dev" and smaller_v1_public_dev_carve_out and len(families) < len(SCENARIO_TYPES):
                logger.warning(
                    "Public-Dev has only %s families; missing scenario types: %s",
                    len(families),
                    sorted(missing_types),
                )
            else:
                raise IntegrityError(f"Split '{split_name}' is missing scenario types: {sorted(missing_types)}")
        splits[split_name] = families

    if total_families_loaded != declared_total_families:
        raise IntegrityError(
            "split_assignment.yaml total_families mismatch: "
            f"declared {declared_total_families}, loaded {total_families_loaded}"
        )
    if declared_total_families == MIN_CODEX_LONG_FAMILIES:
        smaller_v1_counts = {split_name: len(families) for split_name, families in splits.items()}
        if smaller_v1_counts != SMALLER_V1_SPLIT_FAMILY_COUNTS:
            rendered_actual = ", ".join(
                f"{split_name}={smaller_v1_counts.get(split_name, 0)}"
                for split_name in ("train_long", "val_long", "test_long", "public_dev")
            )
            rendered_expected = ", ".join(
                f"{split_name}={count}"
                for split_name, count in SMALLER_V1_SPLIT_FAMILY_COUNTS.items()
            )
            raise IntegrityError(
                "The 35-family Codex-Long freeze must use the signed-off smaller-v1 split geometry: "
                f"{rendered_expected}; got {rendered_actual}"
            )
    if declared_total_families >= FULL_PLAN_CODEX_LONG_FAMILIES:
        missing_floor = {
            scenario_type: count
            for scenario_type, count in family_type_counts.items()
            if count < FULL_PLAN_MIN_FAMILIES_PER_TYPE
        }
        if missing_floor:
            rendered = ", ".join(
                f"{scenario_type}={count}" for scenario_type, count in sorted(missing_floor.items())
            )
            raise IntegrityError(
                "Frozen full-plan benchmark must include at least "
                f"{FULL_PLAN_MIN_FAMILIES_PER_TYPE} families per scenario type; "
                f"below floor: {rendered}"
            )
    manifest_scenario_ids = {
        make_scenario_id(entry["family_id"], entry["variant_id"]) for entry in manifest["variants"]
    }
    extra_manifest_scenarios = sorted(manifest_scenario_ids - all_scenario_ids)
    if extra_manifest_scenarios:
        raise IntegrityError(
            "benchmark_manifest.lock contains variants that are not present in split_assignment.yaml: "
            f"{extra_manifest_scenarios}"
        )

    return splits, env_index


class _Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")

    @contextmanager
    def begin(self) -> Iterator[sqlite3.Cursor]:
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN IMMEDIATE")
            yield cursor
        except Exception:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def close(self) -> None:
        self.connection.close()


class SealState:
    def __init__(self, persist_path: str | Path | None = None) -> None:
        self._sealed = {name: True for name in SEALABLE_POOLS}
        self._unseal_log: list[dict[str, str]] = []
        self._persist_path = Path(persist_path) if persist_path else None
        self._load_persisted_unseal_events()

    def is_sealed(self, pool_or_split: str) -> bool:
        return self._sealed.get(pool_or_split, False)

    def unseal(self, pool_or_split: str, operator: str, reason: str) -> None:
        if pool_or_split not in self._sealed:
            raise ValueError(f"'{pool_or_split}' is not a sealable pool")
        if not self._sealed[pool_or_split]:
            logger.info("'%s' is already unsealed", pool_or_split)
            return
        event = {
            "pool_or_split": pool_or_split,
            "operator": operator,
            "reason": reason,
            "timestamp": _utcnow(),
        }
        self._sealed[pool_or_split] = False
        self._unseal_log.append(event)
        logger.info("UNSEAL: %s", event)
        self._persist_unseal_event(event)

    @property
    def unseal_log(self) -> list[dict[str, str]]:
        return list(self._unseal_log)

    def _persist_unseal_event(self, event: dict[str, str]) -> None:
        if self._persist_path is None:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        with self._persist_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")

    def _load_persisted_unseal_events(self) -> None:
        if self._persist_path is None or not self._persist_path.exists():
            return
        for line_number, line in enumerate(self._persist_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring malformed unseal log entry at %s line %s",
                    self._persist_path,
                    line_number,
                )
                continue
            pool_or_split = event.get("pool_or_split")
            if pool_or_split not in self._sealed:
                logger.warning(
                    "Ignoring unseal log entry with unknown pool_or_split=%r at %s line %s",
                    pool_or_split,
                    self._persist_path,
                    line_number,
                )
                continue
            self._sealed[pool_or_split] = False
            self._unseal_log.append(event)


class DataPoolManager:
    def __init__(
        self,
        swe_bench_pools_path: str | Path,
        split_assignment_path: str | Path,
        manifest_path: str | Path,
        db_path: str | Path,
        seed_config_path: str | Path | None = None,
        recover_running: bool = True,
    ) -> None:
        self.swe_bench_pools, self.swe_bench_metadata = load_swe_bench_pools(swe_bench_pools_path)
        self.manifest = load_codex_long_manifest(manifest_path)
        self.codex_long_splits, self.codex_long_env_index = load_codex_long_splits(
            split_assignment_path=split_assignment_path,
            manifest_path=manifest_path,
        )
        self.manifest_variant_index = {
            make_scenario_id(entry["family_id"], entry["variant_id"]): entry for entry in self.manifest["variants"]
        }
        self.seed_config_path = Path(seed_config_path) if seed_config_path else None
        self.db = _Database(db_path)
        self._init_schema()
        self.seed_config = self._load_seed_config(self.seed_config_path)
        self.seal_state = SealState(Path(db_path).with_suffix(".unseal_log.jsonl"))
        self.b1_viable = self.check_rule_1(self.codex_long_splits)
        self.gate4_outcome: Gate4Outcome | None = None
        if recover_running:
            self.recovered_runs = self.recover_from_crash()
        else:
            self.recovered_runs = 0

    def close(self) -> None:
        self.db.close()

    @staticmethod
    def _load_seed_config(path: Path | None) -> dict[str, Any]:
        if path is None or not path.exists():
            return json.loads(json.dumps(DEFAULT_SEED_CONFIG))
        try:
            raw = load_yaml_file(path) or {}
        except DuplicateKeyError as exc:
            raise IntegrityError(f"seed_config.yaml must not contain duplicate YAML keys: {exc}") from exc
        except ValueError as exc:
            raise IntegrityError(str(exc)) from exc
        if not isinstance(raw, dict):
            raise IntegrityError("seed_config.yaml must be a mapping")
        return raw

    def reload_seed_config(self, seed_config_path: str | Path | None = None) -> dict[str, Any]:
        if seed_config_path is not None:
            self.seed_config_path = Path(seed_config_path)
        self.seed_config = self._load_seed_config(self.seed_config_path)
        return self.seed_config

    def _seed_policy(self, track: str, pool_or_split: str) -> dict[str, Any]:
        try:
            policy = self.seed_config[track][pool_or_split]
        except KeyError as exc:
            raise KeyError(f"No seed policy configured for {track}/{pool_or_split}") from exc
        if not isinstance(policy, dict):
            raise IntegrityError(f"Seed policy for {track}/{pool_or_split} must be a mapping")
        return policy

    @staticmethod
    def _override_matches(override: dict[str, Any], model_id: str, harness: str) -> bool:
        return override.get("model", "*") in {"*", model_id} and override.get("harness", "*") in {"*", harness}

    def assigned_seed_count(self, track: str, pool_or_split: str, model_id: str, harness: str) -> int:
        policy = self._seed_policy(track, pool_or_split)
        count = int(policy.get("default_seeds", 1))
        for override in policy.get("overrides", []):
            if not isinstance(override, dict):
                raise IntegrityError(f"Seed override for {track}/{pool_or_split} must be a mapping")
            if self._override_matches(override, model_id=model_id, harness=harness):
                count = int(override["seeds"])
                break
        max_seeds = policy.get("max_seeds")
        if max_seeds is not None:
            count = min(count, int(max_seeds))
        if count < 1:
            raise IntegrityError(f"Seed policy for {track}/{pool_or_split} resolved to {count}; expected >= 1")
        return count

    def list_assigned_seeds(self, track: str, pool_or_split: str, model_id: str, harness: str) -> list[int]:
        return list(range(1, self.assigned_seed_count(track, pool_or_split, model_id, harness) + 1))

    def _init_schema(self) -> None:
        self.db.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                track TEXT NOT NULL,
                pool_or_split TEXT NOT NULL,
                scenario_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                harness TEXT NOT NULL,
                seed INTEGER NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 1,
                exec_state TEXT NOT NULL DEFAULT 'pending',
                outcome TEXT,
                started_at TEXT,
                completed_at TEXT,
                wall_time_s REAL,
                trajectory_path TEXT,
                family_id TEXT,
                scenario_type TEXT,
                launch_manifest_ver INTEGER,
                grading_manifest_ver INTEGER,
                is_current INTEGER NOT NULL DEFAULT 1,
                superseded_by INTEGER,
                recovery_action TEXT,
                re_gate_required INTEGER DEFAULT 0,
                snapshot_image_ref TEXT,
                cl_pass INTEGER,
                milestone_json TEXT,
                PRIMARY KEY (track, pool_or_split, scenario_id, model_id, harness, seed, attempt),
                CHECK (exec_state IN ('pending', 'running', 'finished')),
                CHECK (outcome IN ('resolved', 'failed', 'no_patch', 'timeout', 'crash') OR outcome IS NULL),
                CHECK (track IN ('swe_bench', 'codex_long')),
                CHECK (harness IN ('codex', 'swe_agent'))
            );
            CREATE INDEX IF NOT EXISTS idx_pool_exec ON runs(pool_or_split, exec_state);
            CREATE INDEX IF NOT EXISTS idx_pool_outcome ON runs(pool_or_split, outcome) WHERE outcome IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_model ON runs(model_id, exec_state);
            CREATE INDEX IF NOT EXISTS idx_family ON runs(family_id) WHERE family_id IS NOT NULL;
            CREATE INDEX IF NOT EXISTS idx_current ON runs(is_current) WHERE is_current = 1;
            CREATE VIEW IF NOT EXISTS latest_runs AS
            SELECT r.*
            FROM runs r
            INNER JOIN (
                SELECT track, pool_or_split, scenario_id, model_id, harness, seed, MAX(attempt) AS max_attempt
                FROM runs
                GROUP BY track, pool_or_split, scenario_id, model_id, harness, seed
            ) latest
            ON r.track = latest.track
            AND r.pool_or_split = latest.pool_or_split
            AND r.scenario_id = latest.scenario_id
            AND r.model_id = latest.model_id
            AND r.harness = latest.harness
            AND r.seed = latest.seed
            AND r.attempt = latest.max_attempt;
            """
        )
        self.db.connection.commit()

    def _row_to_run_record(self, row: sqlite3.Row) -> RunRecord:
        milestone_json = row["milestone_json"]
        return RunRecord(
            track=row["track"],
            pool_or_split=row["pool_or_split"],
            scenario_id=row["scenario_id"],
            model_id=row["model_id"],
            harness=row["harness"],
            seed=int(row["seed"]),
            attempt=int(row["attempt"]),
            exec_state=row["exec_state"],
            outcome=row["outcome"],
            started_at=row["started_at"],
            completed_at=row["completed_at"],
            wall_time_seconds=row["wall_time_s"],
            trajectory_path=row["trajectory_path"],
            family_id=row["family_id"],
            scenario_type=row["scenario_type"],
            launch_manifest_ver=row["launch_manifest_ver"],
            grading_manifest_ver=row["grading_manifest_ver"],
            is_current=bool(row["is_current"]),
            superseded_by=row["superseded_by"],
            recovery_action=row["recovery_action"],
            snapshot_image_ref=row["snapshot_image_ref"],
            re_gate_required=bool(row["re_gate_required"]),
            codex_long_pass=None if row["cl_pass"] is None else bool(row["cl_pass"]),
            milestone_results=json.loads(milestone_json) if milestone_json else None,
        )

    def _query_runs(
        self,
        track: str,
        pool_or_split: str,
        scenario_id: str,
        model_id: str,
        harness: str,
        seed: int,
    ) -> list[RunRecord]:
        rows = self.db.execute(
            """
            SELECT *
            FROM runs
            WHERE track = ? AND pool_or_split = ? AND scenario_id = ? AND model_id = ? AND harness = ? AND seed = ?
            ORDER BY attempt ASC
            """,
            (track, pool_or_split, scenario_id, model_id, harness, seed),
        ).fetchall()
        return [self._row_to_run_record(row) for row in rows]

    def _query_latest_current_runs(
        self,
        track: str,
        pool_or_split: str | None = None,
        model_id: str | None = None,
        harness: str | None = None,
        outcome: str | None = None,
    ) -> list[RunRecord]:
        clauses = ["track = ?", "is_current = 1"]
        params: list[Any] = [track]
        if pool_or_split is not None:
            clauses.append("pool_or_split = ?")
            params.append(pool_or_split)
        if model_id is not None:
            clauses.append("model_id = ?")
            params.append(model_id)
        if harness is not None:
            clauses.append("harness = ?")
            params.append(harness)
        if outcome is not None:
            clauses.append("outcome = ?")
            params.append(outcome)
        rows = self.db.execute(
            f"SELECT * FROM latest_runs WHERE {' AND '.join(clauses)} ORDER BY scenario_id, seed",
            tuple(params),
        ).fetchall()
        return [self._row_to_run_record(row) for row in rows]

    def _query_latest_current_runs_by_family(self, family_id: str, model_id: str, harness: str) -> list[RunRecord]:
        rows = self.db.execute(
            """
            SELECT *
            FROM latest_runs
            WHERE track = 'codex_long' AND family_id = ? AND model_id = ? AND harness = ? AND is_current = 1
            ORDER BY scenario_id, seed
            """,
            (family_id, model_id, harness),
        ).fetchall()
        return [self._row_to_run_record(row) for row in rows]

    def _is_sealed(self, pool_or_split: str) -> bool:
        return self.seal_state.is_sealed(pool_or_split)

    @staticmethod
    def _counts_as_finished_for_listing(run: RunRecord) -> bool:
        if run.exec_state != "finished":
            return False
        if run.outcome == "crash" and run.attempt < 2:
            return False
        return True

    def _get_envs_for_split(self, split: str) -> list[CodexLongEnv]:
        scenario_ids = [make_scenario_id(f.family_id, variant_id) for f in self.codex_long_splits.get(split, []) for variant_id in f.variant_ids]
        return [self.codex_long_env_index[scenario_id] for scenario_id in scenario_ids]

    def _get_family(self, family_id: str) -> CodexLongFamily:
        for families in self.codex_long_splits.values():
            for family in families:
                if family.family_id == family_id:
                    return family
        raise KeyError(f"Unknown family_id '{family_id}'")

    def _get_manifest_entry_for_scenario(self, scenario_id: str) -> dict[str, Any]:
        try:
            return self.manifest_variant_index[scenario_id]
        except KeyError as exc:
            raise KeyError(f"Unknown scenario_id '{scenario_id}'") from exc

    def get_codex_long_launch_artifacts(self, scenario_id: str) -> CodexLongLaunchArtifacts:
        entry = self._get_manifest_entry_for_scenario(scenario_id)
        return CodexLongLaunchArtifacts(
            scenario_id=scenario_id,
            family_id=entry["family_id"],
            variant_id=entry["variant_id"],
            split=entry["split"],
            scenario_type=entry["scenario_type"],
            manifest_version=self.manifest["manifest_version"],
            image_digest=entry["image_digest"],
            agents_md_hash=entry["agents_md_hash"],
            family_spec_hash=entry["family_spec_hash"],
        )

    def get_codex_long_grading_artifacts(self, scenario_id: str) -> CodexLongGradingArtifacts:
        entry = self._get_manifest_entry_for_scenario(scenario_id)
        return CodexLongGradingArtifacts(
            scenario_id=scenario_id,
            family_id=entry["family_id"],
            variant_id=entry["variant_id"],
            split=entry["split"],
            scenario_type=entry["scenario_type"],
            manifest_version=self.manifest["manifest_version"],
            grader_image_digest=self.manifest["grader_image_digest"],
            verifier_hash=entry["verifier_hash"],
            milestone_hashes=dict(entry["milestone_hashes"]),
            verifier_data_hash=entry["verifier_data_hash"],
        )

    def check_rule_1(self, splits: dict[str, list[CodexLongFamily]]) -> bool:
        test_long_family_count = len(splits.get("test_long", []))
        b1_viable = test_long_family_count >= 8
        if not b1_viable:
            logger.warning(
                "RULE 1 FIRED: Test-Long has %s families (floor is 8). B1 is dropped on this path.",
                test_long_family_count,
            )
        return b1_viable

    def can_retry(self, logical_key: tuple[str, str, str, str, str, int]) -> bool:
        latest = self._get_latest_attempt(logical_key)
        return bool(
            latest
            and latest.exec_state == "finished"
            and latest.outcome == "crash"
            and latest.attempt < 2
        )

    def _get_latest_attempt(self, logical_key: tuple[str, str, str, str, str, int]) -> RunRecord | None:
        attempts = self._query_runs(*logical_key)
        if not attempts:
            return None
        return max(attempts, key=lambda run: run.attempt)

    def check_dispatch_eligible(
        self,
        track: str,
        pool_or_split: str,
        scenario_id: str,
        model_id: str,
        harness: str,
        seed: int,
    ) -> DispatchDecision:
        if self._is_sealed(pool_or_split):
            return DispatchDecision.BLOCKED

        attempts = self._query_runs(track, pool_or_split, scenario_id, model_id, harness, seed)
        if not attempts:
            return DispatchDecision.PROCEED

        latest = max(attempts, key=lambda run: run.attempt)
        if latest.exec_state == "running":
            return DispatchDecision.DUPLICATE
        if latest.exec_state == "finished":
            if latest.is_current:
                if latest.outcome == "crash" and latest.attempt < 2:
                    return DispatchDecision.RETRY
                return DispatchDecision.SKIP
            if latest.recovery_action == "regrade_only":
                if latest.snapshot_image_ref:
                    return DispatchDecision.REGRADE_NEEDED
                logger.warning(
                    "Regrade requested for %s but snapshot_image_ref is missing; downgrading to RERUN_NEEDED",
                    scenario_id,
                )
                return DispatchDecision.RERUN_NEEDED
            return DispatchDecision.RERUN_NEEDED
        return DispatchDecision.PROCEED

    def claim_run(
        self,
        track: str,
        pool_or_split: str,
        scenario_id: str,
        model_id: str,
        harness: str,
        seed: int,
        attempt: int = 1,
        launch_manifest_ver: int | None = None,
        family_id: str | None = None,
        scenario_type: str | None = None,
    ) -> bool:
        if self._is_sealed(pool_or_split):
            raise IntegrityError(f"Cannot claim run for sealed pool/split '{pool_or_split}'")
        if track == "codex_long":
            try:
                env = self.codex_long_env_index[scenario_id]
            except KeyError as exc:
                raise IntegrityError(f"Unknown Codex-Long scenario_id '{scenario_id}'") from exc
            if env.split != pool_or_split:
                raise IntegrityError(
                    f"Scenario '{scenario_id}' belongs to split '{env.split}', not '{pool_or_split}'"
                )
            if family_id is not None and family_id != env.family_id:
                raise IntegrityError(
                    f"Scenario '{scenario_id}' belongs to family '{env.family_id}', not '{family_id}'"
                )
            if scenario_type is not None and scenario_type != env.scenario_type:
                raise IntegrityError(
                    f"Scenario '{scenario_id}' has scenario_type '{env.scenario_type}', not '{scenario_type}'"
                )
            if launch_manifest_ver is None:
                raise IntegrityError("Codex-Long claim_run() requires launch_manifest_ver from benchmark_manifest.lock")
            if launch_manifest_ver != self.manifest["manifest_version"]:
                raise IntegrityError(
                    f"Codex-Long claim_run() manifest_version mismatch for '{scenario_id}': "
                    f"got {launch_manifest_ver}, current is {self.manifest['manifest_version']}"
                )
            family_id = env.family_id
            scenario_type = env.scenario_type
        with self.db.begin() as txn:
            result = txn.execute(
                """
                INSERT OR IGNORE INTO runs
                    (track, pool_or_split, scenario_id, model_id, harness, seed, attempt,
                     exec_state, started_at, launch_manifest_ver, family_id, scenario_type)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)
                """,
                (
                    track,
                    pool_or_split,
                    scenario_id,
                    model_id,
                    harness,
                    seed,
                    attempt,
                    _utcnow(),
                    launch_manifest_ver,
                    family_id,
                    scenario_type,
                ),
            )
            claimed = result.rowcount == 1
            if claimed and attempt > 1:
                txn.execute(
                    """
                    UPDATE runs
                    SET superseded_by = ?
                    WHERE track = ? AND pool_or_split = ? AND scenario_id = ? AND model_id = ? AND harness = ? AND seed = ?
                      AND attempt = ?
                    """,
                    (
                        attempt,
                        track,
                        pool_or_split,
                        scenario_id,
                        model_id,
                        harness,
                        seed,
                        attempt - 1,
                    ),
                )
            return claimed

    def finish_run(
        self,
        track: str,
        pool_or_split: str,
        scenario_id: str,
        model_id: str,
        harness: str,
        seed: int,
        attempt: int,
        outcome: str,
        trajectory_path: str | None = None,
        wall_time_seconds: float | None = None,
        grading_manifest_ver: int | None = None,
        codex_long_pass: bool | None = None,
        milestone_results: dict[str, Any] | None = None,
        snapshot_image_ref: str | None = None,
    ) -> None:
        if outcome not in {"resolved", "failed", "no_patch", "timeout", "crash"}:
            raise ValueError(f"Invalid outcome '{outcome}'")
        if track == "codex_long":
            manifest_entry = self._get_manifest_entry_for_scenario(scenario_id)
            expected_milestone_ids = set(manifest_entry["milestone_hashes"])
            if grading_manifest_ver is None:
                raise IntegrityError("Codex-Long finish_run() requires grading_manifest_ver from benchmark_manifest.lock")
            if grading_manifest_ver != self.manifest["manifest_version"]:
                raise IntegrityError(
                    f"Codex-Long finish_run() manifest_version mismatch for '{scenario_id}': "
                    f"got {grading_manifest_ver}, current is {self.manifest['manifest_version']}"
                )
            if outcome != "crash" and (not isinstance(snapshot_image_ref, str) or not snapshot_image_ref.strip()):
                raise IntegrityError(
                    "Codex-Long finish_run() requires snapshot_image_ref for non-crash outcomes so "
                    "Phase 2/3 grading can be re-run from the committed snapshot image"
                )
            if outcome != "crash" and not isinstance(codex_long_pass, bool):
                raise IntegrityError(
                    "Codex-Long finish_run() requires codex_long_pass as a boolean copied from "
                    "Phase 3 verify_result.json for non-crash outcomes"
                )
            if outcome == "resolved" and codex_long_pass is not True:
                raise IntegrityError(
                    "Codex-Long finish_run() outcome='resolved' requires codex_long_pass=True from "
                    "Phase 3 verify_result.json"
                )
            if outcome != "crash" and outcome != "resolved" and codex_long_pass is not False:
                raise IntegrityError(
                    "Codex-Long finish_run() non-resolved outcomes must record codex_long_pass=False "
                    "from Phase 3 verify_result.json"
                )
            if outcome != "crash" and not isinstance(milestone_results, dict):
                raise IntegrityError(
                    "Codex-Long finish_run() requires milestone_results copied from Phase 3 "
                    "verify_result.json for non-crash outcomes"
                )
            if outcome != "crash" and not milestone_results:
                raise IntegrityError(
                    "Codex-Long finish_run() requires non-empty milestone_results from Phase 3 "
                    "verify_result.json for non-crash outcomes"
                )
            if isinstance(milestone_results, dict):
                actual_milestone_ids = set(milestone_results)
                if actual_milestone_ids != expected_milestone_ids:
                    details: list[str] = []
                    missing = sorted(expected_milestone_ids - actual_milestone_ids)
                    unexpected = sorted(actual_milestone_ids - expected_milestone_ids)
                    if missing:
                        details.append(f"missing {missing}")
                    if unexpected:
                        details.append(f"unexpected {unexpected}")
                    rendered = ", ".join(details) if details else "mismatch"
                    raise IntegrityError(
                        "Codex-Long finish_run() milestone_results must match manifest-locked milestone ids "
                        f"for '{scenario_id}': {rendered}"
                    )
                for milestone_id, achieved in milestone_results.items():
                    if not isinstance(milestone_id, str) or not milestone_id.strip():
                        raise IntegrityError(
                            "Codex-Long finish_run() milestone_results keys must be non-empty strings"
                        )
                    if not isinstance(achieved, bool):
                        raise IntegrityError(
                            "Codex-Long finish_run() milestone_results values must be booleans from "
                            "Phase 3 verify_result.json"
                        )
        with self.db.begin() as txn:
            result = txn.execute(
                """
                UPDATE runs
                SET exec_state = 'finished',
                    outcome = ?,
                    completed_at = ?,
                    wall_time_s = ?,
                    trajectory_path = ?,
                    grading_manifest_ver = ?,
                    cl_pass = ?,
                    milestone_json = ?,
                    snapshot_image_ref = ?
                WHERE track = ? AND pool_or_split = ? AND scenario_id = ? AND model_id = ? AND harness = ? AND seed = ?
                  AND attempt = ? AND exec_state = 'running'
                """,
                (
                    outcome,
                    _utcnow(),
                    wall_time_seconds,
                    trajectory_path,
                    grading_manifest_ver,
                    None if codex_long_pass is None else int(codex_long_pass),
                    json.dumps(milestone_results) if milestone_results is not None else None,
                    snapshot_image_ref,
                    track,
                    pool_or_split,
                    scenario_id,
                    model_id,
                    harness,
                    seed,
                    attempt,
                ),
            )
            if result.rowcount != 1:
                raise IntegrityError(
                    f"No running record found for {track}/{pool_or_split}/{scenario_id}/{model_id}/{harness}/seed{seed}/attempt{attempt}"
                )

    def invalidate_stale_runs(
        self,
        family_id: str,
        new_manifest_version: int,
        affected_artifact: str,
        reason: str,
        affected_variant_ids: list[str] | None = None,
        re_gate_required: bool = False,
    ) -> int:
        if affected_artifact not in _ARTIFACT_RECOVERY:
            raise ValueError(
                f"Unknown affected_artifact: '{affected_artifact}'. Valid values: {sorted(_ARTIFACT_RECOVERY)}"
            )
        spec = _ARTIFACT_RECOVERY[affected_artifact]
        ver_column = spec["ver_column"]
        recovery = spec["recovery"]
        family = self._get_family(family_id)
        where_clauses = [
            "track = 'codex_long'",
            f"{ver_column} < ?",
            "is_current = 1",
            "exec_state IN ('running', 'finished')",
        ]
        where_params: list[Any] = [new_manifest_version]

        if affected_variant_ids is not None:
            if not affected_variant_ids:
                raise IntegrityError(
                    f"Family '{family_id}' received an empty affected_variant_ids scope; "
                    "pass None for a family-wide invalidation or list the specific variants to invalidate"
                )
            seen_variant_ids: set[str] = set()
            unknown_variant_ids: list[str] = []
            scenario_ids: list[str] = []
            for variant_id in affected_variant_ids:
                if not isinstance(variant_id, str) or not variant_id:
                    raise IntegrityError("affected_variant_ids must contain non-empty variant ids")
                if variant_id in seen_variant_ids:
                    raise IntegrityError(
                        f"affected_variant_ids contains duplicate variant_id '{variant_id}' for family '{family_id}'"
                    )
                seen_variant_ids.add(variant_id)
                if variant_id not in family.variant_ids:
                    unknown_variant_ids.append(variant_id)
                    continue
                scenario_ids.append(make_scenario_id(family_id, variant_id))
            if unknown_variant_ids:
                raise IntegrityError(
                    f"Family '{family_id}' does not define affected_variant_ids {sorted(unknown_variant_ids)}"
                )
            placeholders = ",".join("?" for _ in scenario_ids)
            where_clauses.append(f"scenario_id IN ({placeholders})")
            where_params.extend(scenario_ids)
        else:
            where_clauses.append("family_id = ?")
            where_params.append(family_id)

        set_params = [recovery, int(re_gate_required)]
        with self.db.begin() as txn:
            result = txn.execute(
                f"""
                UPDATE runs
                SET is_current = 0,
                    recovery_action = ?,
                    re_gate_required = ?
                WHERE {' AND '.join(where_clauses)}
                """,
                (*set_params, *where_params),
            )
            count = result.rowcount
            if count > 0:
                scope = f"variants {affected_variant_ids}" if affected_variant_ids else f"family '{family_id}' (all variants)"
                logger.warning(
                    "Invalidated %s runs for %s: %s changed at manifest v%s. Recovery: %s. Re-gate required: %s. Reason: %s",
                    count,
                    scope,
                    affected_artifact,
                    new_manifest_version,
                    recovery,
                    re_gate_required,
                    reason,
                )
            return count

    def recover_from_crash(self) -> int:
        with self.db.begin() as txn:
            result = txn.execute(
                """
                UPDATE runs
                SET exec_state = 'finished', outcome = 'crash', completed_at = ?
                WHERE exec_state = 'running'
                """,
                (_utcnow(),),
            )
            if result.rowcount > 0:
                logger.warning("Recovered %s runs stuck in 'running' -> 'crash'", result.rowcount)
            return result.rowcount

    def unseal(self, pool_or_split: str, operator: str, reason: str) -> None:
        self.seal_state.unseal(pool_or_split, operator, reason)

    def list_swe_bench_tasks(
        self,
        pool: str,
        model_id: str | None = None,
        harness: str | None = None,
        seed: int | None = None,
        exclude_finished: bool = True,
    ) -> list[dict[str, Any]]:
        if self._is_sealed(pool):
            logger.warning("Attempted to list tasks from sealed pool '%s'", pool)
            return []
        tasks = list(self.swe_bench_pools[pool])
        if exclude_finished and model_id and harness and seed is not None:
            finished_ids = {
                run.scenario_id
                for run in self._query_latest_current_runs("swe_bench", pool, model_id=model_id, harness=harness)
                if self._counts_as_finished_for_listing(run) and run.seed == seed
            }
            tasks = [task for task in tasks if task["instance_id"] not in finished_ids]
        return tasks

    def list_codex_long_envs(
        self,
        split: str,
        model_id: str | None = None,
        harness: str | None = None,
        seed: int | None = None,
        scenario_type: str | None = None,
        family_id: str | None = None,
        exclude_finished: bool = True,
    ) -> list[CodexLongEnv]:
        if self._is_sealed(split):
            logger.warning("Attempted to list envs from sealed split '%s'", split)
            return []
        envs = self._get_envs_for_split(split)
        if scenario_type is not None:
            envs = [env for env in envs if env.scenario_type == scenario_type]
        if family_id is not None:
            envs = [env for env in envs if env.family_id == family_id]
        if exclude_finished and model_id and harness and seed is not None:
            finished_ids = {
                run.scenario_id
                for run in self._query_latest_current_runs("codex_long", split, model_id=model_id, harness=harness)
                if self._counts_as_finished_for_listing(run) and run.seed == seed
            }
            envs = [env for env in envs if env.scenario_id not in finished_ids]
        return envs

    def list_families(self, split: str | None = None, scenario_type: str | None = None) -> list[CodexLongFamily]:
        families: list[CodexLongFamily] = []
        target_splits = [split] if split else ["train_long", "val_long", "test_long", "public_dev"]
        for split_name in target_splits:
            if self._is_sealed(split_name):
                logger.warning("Attempted to list families from sealed split '%s'", split_name)
                continue
            for family in self.codex_long_splits.get(split_name, []):
                if scenario_type and family.scenario_type != scenario_type:
                    continue
                families.append(family)
        return families

    def get_family_solve_summary(self, family_id: str, model_id: str, harness: str) -> dict[str, Any]:
        family = self._get_family(family_id)
        latest_runs = [
            run
            for run in self._query_latest_current_runs_by_family(family_id, model_id, harness)
            if run.exec_state == "finished"
        ]

        variant_outcomes: dict[str, dict[str, int]] = {}
        total_resolved_traces = 0
        for run in latest_runs:
            outcome = variant_outcomes.setdefault(run.scenario_id, {"seeds_finished": 0, "seeds_resolved": 0})
            outcome["seeds_finished"] += 1
            if run.outcome == "resolved" and run.codex_long_pass:
                outcome["seeds_resolved"] += 1
                total_resolved_traces += 1

        finished_variants = len(variant_outcomes)
        solved_variants = sum(1 for outcome in variant_outcomes.values() if outcome["seeds_resolved"] > 0)
        solved_scenario_ids = sorted(
            scenario_id
            for scenario_id, outcome in variant_outcomes.items()
            if outcome["seeds_resolved"] > 0
        )
        return {
            "family_id": family_id,
            "scenario_type": family.scenario_type,
            "split": family.split,
            "total_variants": family.variant_count,
            "finished_variants": finished_variants,
            "solved_variants": solved_variants,
            "variant_solve_rate": solved_variants / finished_variants if finished_variants else 0.0,
            "solved_scenario_ids": solved_scenario_ids,
            "resolved_traces": total_resolved_traces,
            "total_finished_runs": len(latest_runs),
        }

    def list_training_eligible_runs(self, track: str, model_id: str, harness: str) -> list[RunRecord]:
        eligible_pools = {"swe_bench": ["bench_control"], "codex_long": ["train_long"]}
        if track not in eligible_pools:
            raise ValueError(f"Unknown track: {track}")
        results: list[RunRecord] = []
        for pool_or_split in eligible_pools[track]:
            runs = self._query_latest_current_runs(
                track=track,
                pool_or_split=pool_or_split,
                model_id=model_id,
                harness=harness,
                outcome="resolved",
            )
            if track == "codex_long":
                runs = [run for run in runs if run.codex_long_pass]
            results.extend(runs)
        return results

    def assert_training_eligible(self, pool_or_split: str) -> None:
        if pool_or_split not in TRAINING_ELIGIBLE:
            raise TrainingAccessViolation(
                f"Pool/split '{pool_or_split}' is NOT training-eligible. Only {TRAINING_ELIGIBLE} may be used for gradient updates."
            )

    def get_matched_scenario_ids(self, model_id: str) -> list[str]:
        codex_successes = {
            run.scenario_id
            for run in self.list_training_eligible_runs("codex_long", model_id, "codex")
        }
        swe_agent_successes = {
            run.scenario_id
            for run in self._query_latest_current_runs(
                track="codex_long",
                pool_or_split="train_long",
                model_id=model_id,
                harness="swe_agent",
                outcome="resolved",
            )
            if run.codex_long_pass
        }
        return sorted(codex_successes & swe_agent_successes)

    def get_campaign_progress(
        self,
        track: str,
        pool_or_split: str,
        model_id: str,
        harness: str,
        seed: int,
    ) -> dict[str, Any]:
        if track == "swe_bench":
            total = len(self.swe_bench_pools[pool_or_split])
        else:
            total = sum(family.variant_count for family in self.codex_long_splits.get(pool_or_split, []))

        latest = [
            run
            for run in self._query_latest_current_runs(track, pool_or_split, model_id=model_id, harness=harness)
            if run.seed == seed
        ]
        by_outcome: dict[str, int] = {}
        pending_or_running = 0
        for run in latest:
            if run.exec_state == "finished" and run.outcome:
                by_outcome[run.outcome] = by_outcome.get(run.outcome, 0) + 1
            else:
                pending_or_running += 1
        total_finished = sum(by_outcome.values())
        return {
            "track": track,
            "pool_or_split": pool_or_split,
            "model_id": model_id,
            "harness": harness,
            "seed": seed,
            "total_tasks": total,
            "finished": total_finished,
            "by_outcome": by_outcome,
            "in_progress": pending_or_running,
            "not_started": total - total_finished - pending_or_running,
            "resolved": by_outcome.get("resolved", 0),
        }

    def label_trajectory(self, run: RunRecord) -> dict[str, Any]:
        labels: dict[str, Any] = {
            "track": run.track,
            "pool_or_split": run.pool_or_split,
            "scenario_id": run.scenario_id,
            "model_id": run.model_id,
            "harness": run.harness,
            "seed": run.seed,
            "outcome": run.outcome,
            "trajectory_path": run.trajectory_path,
            "training_eligible": run.pool_or_split in TRAINING_ELIGIBLE,
            "is_current": run.is_current,
        }
        if run.track == "codex_long":
            labels.update(
                {
                    "family_id": run.family_id,
                    "scenario_type": run.scenario_type,
                    "variant_id": run.scenario_id.split("/", 1)[1],
                    "codex_long_pass": run.codex_long_pass,
                    "launch_manifest_ver": run.launch_manifest_ver,
                    "grading_manifest_ver": run.grading_manifest_ver,
                }
            )
        return labels


__all__ = [
    "CodexLongEnv",
    "CodexLongFamily",
    "CodexLongGradingArtifacts",
    "CodexLongLaunchArtifacts",
    "DataPoolManager",
    "DispatchDecision",
    "Gate4Outcome",
    "IntegrityError",
    "RunRecord",
    "SealState",
    "TrainingAccessViolation",
    "load_codex_long_manifest",
    "load_codex_long_splits",
    "load_swe_bench_pools",
    "make_scenario_id",
    "sha256_file",
    "_find_manifest_variant",
]
