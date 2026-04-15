from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import yaml

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


def _find_manifest_variant(manifest: dict[str, Any], family_id: str, variant_id: str) -> dict[str, Any]:
    for entry in manifest.get("variants", []):
        if entry["family_id"] == family_id and entry["variant_id"] == variant_id:
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
            return entry

    raise IntegrityError(
        f"Variant '{family_id}/{variant_id}' appears in split_assignment.yaml but has no entry in "
        "benchmark_manifest.lock. Cannot verify image digest or verifier hash."
    )


def load_swe_bench_pools(path: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
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
    assignment = yaml.safe_load(Path(split_assignment_path).read_text()) or {}
    manifest = yaml.safe_load(Path(manifest_path).read_text()) or {}

    actual_hash = sha256_file(split_assignment_path)
    expected_hash = manifest.get("split_assignment_hash")
    if actual_hash != expected_hash:
        raise IntegrityError(
            f"split_assignment.yaml hash mismatch: expected {expected_hash}, got {actual_hash}"
        )

    all_family_ids: set[str] = set()
    all_scenario_ids: set[str] = set()
    splits: dict[str, list[CodexLongFamily]] = {}
    env_index: dict[str, CodexLongEnv] = {}
    manifest_version = int(manifest["manifest_version"])

    split_mapping = assignment.get("splits")
    if not isinstance(split_mapping, dict):
        raise IntegrityError("split_assignment.yaml is missing a 'splits' mapping")

    for split_name in ("train_long", "val_long", "test_long", "public_dev"):
        split_data = split_mapping.get(split_name)
        if not isinstance(split_data, dict):
            raise IntegrityError(f"split_assignment.yaml is missing split '{split_name}'")
        families: list[CodexLongFamily] = []
        for family in split_data.get("families", []):
            family_id = family["family_id"]
            if family_id in all_family_ids:
                raise IntegrityError(f"Family '{family_id}' appears in multiple splits")
            all_family_ids.add(family_id)

            scenario_type = family["scenario_type"]
            if scenario_type not in SCENARIO_TYPES:
                raise ValueError(f"Family '{family_id}' has unknown scenario_type '{scenario_type}'")

            variant_ids = tuple(family.get("variant_ids", []))
            variant_count = int(family["variant_count"])
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
            if split_name == "public_dev" and len(families) < 5:
                logger.warning(
                    "Public-Dev has only %s families; missing scenario types: %s",
                    len(families),
                    sorted(missing_types),
                )
            else:
                raise IntegrityError(f"Split '{split_name}' is missing scenario types: {sorted(missing_types)}")
        splits[split_name] = families

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
        self.codex_long_splits, self.codex_long_env_index = load_codex_long_splits(
            split_assignment_path=split_assignment_path,
            manifest_path=manifest_path,
        )
        self.manifest = yaml.safe_load(Path(manifest_path).read_text()) or {}
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
        raw = yaml.safe_load(path.read_text()) or {}
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

    def _get_envs_for_split(self, split: str) -> list[CodexLongEnv]:
        scenario_ids = [make_scenario_id(f.family_id, variant_id) for f in self.codex_long_splits.get(split, []) for variant_id in f.variant_ids]
        return [self.codex_long_env_index[scenario_id] for scenario_id in scenario_ids]

    def _get_family(self, family_id: str) -> CodexLongFamily:
        for families in self.codex_long_splits.values():
            for family in families:
                if family.family_id == family_id:
                    return family
        raise KeyError(f"Unknown family_id '{family_id}'")

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
        where_clauses = [
            "track = 'codex_long'",
            f"{ver_column} < ?",
            "is_current = 1",
            "exec_state = 'finished'",
        ]
        where_params: list[Any] = [new_manifest_version]

        if affected_variant_ids:
            scenario_ids = [make_scenario_id(family_id, variant_id) for variant_id in affected_variant_ids]
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
                if run.exec_state == "finished" and run.seed == seed
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
                if run.exec_state == "finished" and run.seed == seed
            }
            envs = [env for env in envs if env.scenario_id not in finished_ids]
        return envs

    def list_families(self, split: str | None = None, scenario_type: str | None = None) -> list[CodexLongFamily]:
        families: list[CodexLongFamily] = []
        target_splits = [split] if split else ["train_long", "val_long", "test_long", "public_dev"]
        for split_name in target_splits:
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
    "DataPoolManager",
    "DispatchDecision",
    "Gate4Outcome",
    "IntegrityError",
    "RunRecord",
    "SealState",
    "TrainingAccessViolation",
    "load_codex_long_splits",
    "load_swe_bench_pools",
    "make_scenario_id",
    "sha256_file",
    "_find_manifest_variant",
]
