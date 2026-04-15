# LLD-02 · Data Pool Manager

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S1 → Implement S1 early  
> Status: DRAFT v0.1

---

## Changelog

| Version | Change |
|---|---|
| v0.1 | Initial draft. Full rewrite from pre-v2.3 three-pool SWE-bench design. Two-track architecture (SWE-bench Verified + Codex-Long). Family-level metadata tracking for Codex-Long. Dual seal enforcement (Final-Test + Test-Long). Training access control for LLD-10. Run-state tracking per pool × model × seed. Reduced-split geometry for the 35-family path. |

---

## 1. Purpose & Scope

This document specifies the Data Pool Manager — the component that owns pool and split membership, run-state tracking, seed assignment, seal enforcement, and training-set access control for the entire Codex-Bench project. Every run (collection, evaluation, or training) passes through this layer to determine which tasks or environments are eligible and to record what has been completed.

**Responsibilities:**

- Maintain two strictly separated track structures: Track 1 (SWE-bench Verified, three disjoint pools) and Track 2 (Codex-Long, four family-disjoint splits)
- Consume the frozen `split_assignment.yaml` and `benchmark_manifest.lock` from LLD-13 as the authoritative source for all Codex-Long split membership and family metadata
- Assign and track seeds per pool × model × harness combination
- Track run state (pending / running / completed / failed / timeout / crash) per task-or-env × model × seed
- Enforce deduplication: prevent duplicate runs of the same task × model × seed combination
- Enforce seal protocols: Final-Test (SWE-bench) and Test-Long (Codex-Long) are inaccessible until Sprint 3
- Enforce training-set-only access for LLD-10: only Bench-Control and Train-Long tasks/envs may be used for gradient updates — Final-Test and Test-Long are never training data
- Provide task/env listing APIs consumed by LLD-07 (Benchmark Runner) and LLD-06 (Trajectory Parser)
- Support the reduced-split geometry for the 35-family path without code changes

**Out of scope:** Task execution (LLD-03), scenario family authoring and verifier design (LLD-13), model serving (LLD-01), trajectory parsing and SFT formatting (LLD-06), benchmark campaign orchestration (LLD-07), and the Gate 4 pilot campaign itself (LLD-07 manages pilot family selection). This LLD provides the data layer those components query and mutate.

**Must exist before any run starts.** LLD-07 cannot dispatch a single task until the pool manager is initialized with valid pool definitions and the Codex-Long split assignment is loaded.

---

## 2. Pool Architecture Overview

Two independent tracks with strictly separated roles. No task or environment appears in both tracks. Within each track, pools/splits are disjoint.

```
┌──────────────────────────────────────────────────────────────────┐
│                    DATA POOL MANAGER                             │
│                                                                  │
│  TRACK 1 — SWE-bench Verified                                   │
│  ┌──────────┐  ┌───────────────┐  ┌────────────────────────┐    │
│  │ Dev-Bench │  │ Bench-Control │  │ Final-Test (SEALED)    │    │
│  │ 50 tasks  │  │ ~50 tasks     │  │ 100 tasks              │    │
│  │ All 6     │  │ 27B only      │  │ Sprint 3 only          │    │
│  │ models    │  │ 1–2 seeds     │  │ Contribution B2 eval   │    │
│  └──────────┘  └───────────────┘  └────────────────────────┘    │
│                                                                  │
│  TRACK 2 — Codex-Long (from LLD-13 frozen split table)          │
│  ┌────────────┐ ┌──────────┐ ┌──────────────────┐ ┌──────────┐ │
│  │ Train-Long │ │ Val-Long │ │ Test-Long        │ │Public-Dev│ │
│  │ ~30 fam    │ │ ~10 fam  │ │ (SEALED)         │ │ ~5 fam   │ │
│  │ ~150–240   │ │ ~30–50   │ │ ~10 fam / ~40–60 │ │ ~15–20   │ │
│  │ envs       │ │ envs     │ │ envs             │ │ envs     │ │
│  └────────────┘ └──────────┘ └──────────────────┘ └──────────┘ │
│                                                                  │
│  CROSS-TRACK ENFORCEMENT                                        │
│  Training-eligible: Bench-Control + Train-Long ONLY              │
│  Sealed until Sprint 3: Final-Test + Test-Long                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Track 1 — SWE-bench Verified Pools

### 3.1 Pool Definitions

All three pools are disjoint subsets of SWE-bench Verified. Pool membership is assigned once at initialization and is immutable.

| Pool | Size | Models | Seeds | HLD Role |
|---|---|---|---|---|
| **Dev-Bench** | 50 tasks | All 6 (Gates 1b/1c conditional) | 1 | Contribution A leaderboard |
| **Bench-Control** | ~50 tasks | 27B only | 1–2 | In-domain SFT control arm (appendix diagnostic) |
| **Final-Test** | 100 tasks | Trained models + base + SWE-Agent baselines | 1–3 | Contribution B2 evaluation (sealed until Sprint 3) |

**Dev-Bench** (50 tasks): Published Contribution A results. Run count (~300) is baseline contingent on the full six-model lineup passing Gates 1b and 1c. Models excluded by gate failures reduce the count proportionally.

**Bench-Control** (~50 tasks): Small in-domain SFT control arm. At ~15% solve rate, yields ~7–8 successful traces per seed (~15–20 at 2 seeds). This is a design control for interpretability, not a primary training source. The resulting SWE-Bench-Control-SFT model is reported in an appendix only — not in mainline B1 or B2 results.

**Final-Test** (100 tasks): Sealed evaluation set. Unsealed only when Sprint 3 evaluation begins. Used for B2 headline results and all B2 comparison arms. See §8 for seal enforcement.

### 3.2 SWE-bench Task Record Schema

```yaml
# Pool definition: swe_bench_pools.yaml
# Generated once from SWE-bench Verified task list; immutable after initialization.

pools:
  dev_bench:
    tasks:
      - instance_id: "django__django-11099"
        repo: "django/django"
        base_commit: "abc1234..."
        pool: dev_bench
      # ... 49 more tasks
    total: 50

  bench_control:
    tasks:
      - instance_id: "sympy__sympy-20049"
        repo: "sympy/sympy"
        base_commit: "def5678..."
        pool: bench_control
      # ... ~49 more tasks
    total: 50

  final_test:
    sealed: true
    unseal_sprint: 3
    tasks:
      - instance_id: "scikit-learn__scikit-learn-13779"
        repo: "scikit-learn/scikit-learn"
        base_commit: "ghi9012..."
        pool: final_test
      # ... 99 more tasks
    total: 100
```

### 3.3 Pool Assignment Protocol

Pool assignment draws from the full SWE-bench Verified task set (500 tasks). The 200 tasks allocated to the three pools are selected using a fixed random seed and are disjoint by construction.

```python
import random

def assign_swe_bench_pools(
    all_tasks: list[str],
    seed: int = 42,
    dev_bench_size: int = 50,
    bench_control_size: int = 50,
    final_test_size: int = 100,
) -> dict[str, list[str]]:
    """
    Assign SWE-bench Verified tasks to three disjoint pools.
    
    The remaining ~300 tasks are unassigned and unused.
    Assignment is deterministic given the same task list and seed.
    """
    rng = random.Random(seed)
    shuffled = list(all_tasks)
    rng.shuffle(shuffled)

    total_needed = dev_bench_size + bench_control_size + final_test_size
    assert len(shuffled) >= total_needed, (
        f"Need {total_needed} tasks but only {len(shuffled)} available"
    )

    cursor = 0
    pools = {}
    for pool_name, size in [
        ("dev_bench", dev_bench_size),
        ("bench_control", bench_control_size),
        ("final_test", final_test_size),
    ]:
        pools[pool_name] = shuffled[cursor : cursor + size]
        cursor += size

    # Verify disjointness
    all_assigned = set()
    for name, tasks in pools.items():
        overlap = all_assigned & set(tasks)
        assert not overlap, f"Pool {name} overlaps: {overlap}"
        all_assigned.update(tasks)

    return pools
```

**The assignment seed and the resulting pool membership are published alongside results.** Any reproduction attempt must use the same seed and task list to obtain identical pools.

---

## 4. Track 2 — Codex-Long Splits

### 4.1 Relationship to LLD-13

The pool manager does **not** own Codex-Long split membership. Split membership is defined by LLD-13's frozen `split_assignment.yaml` and is consumed read-only by this component. The pool manager's responsibilities for Track 2 are:

- Load and validate the frozen split assignment at initialization
- Index family-level metadata for fast lookup (scenario type, variant list, split assignment)
- Provide filtered task/env lists to LLD-07 by split
- Track run state per env × model × harness × seed
- Enforce Test-Long seal (§8)
- Enforce Train-Long-only access for LLD-10 training (§9)

### 4.2 Split Summary

| Split | Families | Envs/Family | Total Envs | Role | Training-eligible? | Sealed? |
|---|---|---|---|---|---|---|
| **Train-Long** | ~30 | ~5–8 | ~150–240 | Primary SFT/RL trajectory collection | **Yes** | No |
| **Val-Long** | ~10 | ~3–5 | ~30–50 | RL early stopping / HP selection only | No | No |
| **Test-Long** | ~10 | ~4–6 | ~40–60 | Sealed secondary benchmark | No | **Yes** — Sprint 3 only |
| **Public-Dev** | ~5 | ~3–4 | ~15–20 | Published dev set | No | No |

**Family-disjoint splits.** The split is family-disjoint, not merely environment-disjoint. Families in Val-Long and Test-Long are not seen in Train-Long — not merely unseen variants. The pool manager validates this invariant at load time.

### 4.3 Codex-Long Family Metadata Record

For each family in the loaded split assignment, the pool manager maintains an indexed metadata record:

```python
@dataclass(frozen=True)
class CodexLongFamily:
    family_id: str                    # e.g. "dependency-migration-npm"
    scenario_type: str                # one of the 5 named types
    split: str                        # train_long | val_long | test_long | public_dev
    variant_ids: tuple[str, ...]      # e.g. ("lodash-3-to-4", "express-4-to-5", ...)
    variant_count: int
    # From benchmark_manifest.lock — per-variant hashes for drift detection
    manifest_version: int


@dataclass(frozen=True)
class CodexLongEnv:
    family_id: str
    variant_id: str
    split: str
    scenario_type: str
    image_tag: str                    # e.g. "codex-long/dependency-migration-npm/lodash-3-to-4:<hash>"
    image_digest: str                 # from benchmark_manifest.lock
```

### 4.4 Split Assignment Loading and Validation

At initialization, the pool manager loads `split_assignment.yaml` and performs structural validation.

```python
def load_codex_long_splits(
    split_assignment_path: str,
    manifest_path: str,
) -> dict[str, list[CodexLongFamily]]:
    """
    Load and validate the frozen Codex-Long split assignment.
    
    Validations (all must pass — abort initialization on failure):
    1. Every family appears in exactly one split (family-disjointness)
    2. Every split contains ≥ 1 family from each of the 5 scenario types
       (except Public-Dev on the 35-family path — see §10.1 carve-out)
    3. Split family counts are within expected ranges
    4. The split_assignment.yaml hash matches the manifest's split_assignment_hash
    5. Every variant referenced in the split assignment has a corresponding
       entry in benchmark_manifest.lock
    """
    assignment = yaml.safe_load(open(split_assignment_path))
    manifest = yaml.safe_load(open(manifest_path))

    # Verify split assignment integrity against manifest
    actual_hash = sha256_file(split_assignment_path)
    if actual_hash != manifest["split_assignment_hash"]:
        raise IntegrityError(
            f"split_assignment.yaml hash mismatch: "
            f"expected {manifest['split_assignment_hash']}, got {actual_hash}. "
            f"The split assignment may have been modified after freeze."
        )

    # Build family index
    all_family_ids: set[str] = set()
    splits: dict[str, list[CodexLongFamily]] = {}

    SCENARIO_TYPES = {
        "feature_evolution",
        "migration_refactor",
        "build_ci_breakage",
        "investigate_then_fix",
        "cross_layer_changes",
    }

    for split_name in ("train_long", "val_long", "test_long", "public_dev"):
        split_data = assignment["splits"][split_name]
        families = []
        for fam in split_data["families"]:
            fid = fam["family_id"]

            # Check family-disjointness
            if fid in all_family_ids:
                raise IntegrityError(
                    f"Family '{fid}' appears in multiple splits. "
                    f"Splits must be family-disjoint."
                )
            all_family_ids.add(fid)

            # Validate scenario type
            if fam["scenario_type"] not in SCENARIO_TYPES:
                raise ValueError(
                    f"Family '{fid}' has unknown scenario_type '{fam['scenario_type']}'"
                )

            families.append(CodexLongFamily(
                family_id=fid,
                scenario_type=fam["scenario_type"],
                split=split_name,
                variant_ids=tuple(fam.get("variant_ids", [])),
                variant_count=fam["variant_count"],
                manifest_version=manifest["manifest_version"],
            ))
        splits[split_name] = families

        # Type coverage check (per-split)
        types_present = {f.scenario_type for f in families}
        missing_types = SCENARIO_TYPES - types_present
        if missing_types:
            if split_name == "public_dev" and len(families) < 5:
                # Public-Dev carve-out on 35-family path (LLD-13 §12.1)
                logger.warning(
                    f"Public-Dev has only {len(families)} families — "
                    f"cannot cover all 5 scenario types. Missing: {missing_types}. "
                    f"This is expected on the 35-family path."
                )
            else:
                raise IntegrityError(
                    f"Split '{split_name}' is missing scenario types: {missing_types}. "
                    f"All 5 types must be represented in Train-Long, Val-Long, and Test-Long."
                )

    return splits
```

---

## 5. Run-State Tracking

### 5.1 Run-State Model

Every run is identified by a composite key and tracked through a defined lifecycle. The key structure differs by track.

**Track 1 (SWE-bench):** `(pool, instance_id, model_id, harness, seed)`

**Track 2 (Codex-Long):** `(split, family_id, variant_id, model_id, harness, seed)`

The `harness` field distinguishes Codex from SWE-Agent runs on the same task/env — both produce run records that must be tracked independently.

### 5.2 Run States

```
pending → running → completed
                  → failed
                  → timeout
                  → crash
                  → no_patch    (SWE-bench only — agent produced no patch file)
```

All terminal states (completed, failed, timeout, crash, no_patch) are final. A run that reaches a terminal state is never re-entered to the `running` state. Retries (§5.4) create a new run record with a distinct `attempt` counter.

### 5.3 Run Record Schema

```python
@dataclass
class RunRecord:
    # Composite key
    track: str                     # "swe_bench" | "codex_long"
    pool_or_split: str             # "dev_bench" | "bench_control" | "final_test" |
                                   # "train_long" | "val_long" | "test_long" | "public_dev"
    task_id: str                   # instance_id (SWE-bench) or variant_id (Codex-Long)
    family_id: Optional[str]       # None for SWE-bench; family_id for Codex-Long
    scenario_type: Optional[str]   # None for SWE-bench; scenario type for Codex-Long
    model_id: str                  # e.g. "qwen3.5-27b"
    harness: str                   # "codex" | "swe_agent"
    seed: int                      # run seed

    # Run state
    state: str                     # pending | running | completed | failed | timeout | crash | no_patch
    attempt: int                   # 1-indexed; incremented on retry (max 2 per HLD §11 health monitoring)

    # Metadata
    started_at: Optional[str]      # ISO 8601 timestamp
    completed_at: Optional[str]    # ISO 8601 timestamp
    wall_time_seconds: Optional[float]
    manifest_version: Optional[int]  # Codex-Long only: benchmark_manifest.lock version at run time
    trajectory_path: Optional[str]   # path to JSONL output file
    result: Optional[str]            # "resolved" | "failed" | ... (HLD failure contract)

    # Grading outcome (Codex-Long only — from LLD-13 verify_result.json)
    codex_long_pass: Optional[bool]
    milestone_results: Optional[dict]  # milestone_id → pass/fail
```

### 5.4 Retry Policy

Per HLD §11 (LLD-01 mid-run health monitoring): if a task crashes due to a model server failure, LLD-03 may retry once (max 2 total attempts). The pool manager tracks attempts via the `attempt` field. A retry creates a new run record with `attempt = 2` and the original record's state remains at `crash`.

```python
def can_retry(self, run: RunRecord) -> bool:
    """A run can be retried if it crashed and has not exceeded max attempts."""
    return run.state == "crash" and run.attempt < 2
```

### 5.5 Run-State Storage

Run state is persisted to a SQLite database (`run_state.db`) in the project working directory. SQLite provides ACID transactions for concurrent access (LLD-03 writes run state; LLD-07 reads it for campaign progress), is zero-dependency, and supports the query patterns needed (filter by pool, model, seed, state).

```sql
CREATE TABLE runs (
    -- Composite key
    track          TEXT NOT NULL,
    pool_or_split  TEXT NOT NULL,
    task_id        TEXT NOT NULL,
    family_id      TEXT,
    scenario_type  TEXT,
    model_id       TEXT NOT NULL,
    harness        TEXT NOT NULL,
    seed           INTEGER NOT NULL,
    attempt        INTEGER NOT NULL DEFAULT 1,

    -- Run state
    state          TEXT NOT NULL DEFAULT 'pending',
    started_at     TEXT,
    completed_at   TEXT,
    wall_time_s    REAL,
    manifest_ver   INTEGER,
    trajectory_path TEXT,
    result         TEXT,

    -- Codex-Long grading
    cl_pass        INTEGER,          -- 0/1 boolean
    milestone_json TEXT,             -- JSON string of milestone results

    PRIMARY KEY (track, pool_or_split, task_id, model_id, harness, seed, attempt),
    CHECK (state IN ('pending','running','completed','failed','timeout','crash','no_patch')),
    CHECK (track IN ('swe_bench','codex_long')),
    CHECK (harness IN ('codex','swe_agent'))
);

-- Indexes for common query patterns
CREATE INDEX idx_pool_state ON runs(pool_or_split, state);
CREATE INDEX idx_model_state ON runs(model_id, state);
CREATE INDEX idx_family ON runs(family_id) WHERE family_id IS NOT NULL;
```

---

## 6. Seed Assignment

### 6.1 Seed Semantics

A seed controls the non-determinism in the model's sampling during a task attempt. Different seeds on the same task produce different trajectories — this is required for multi-seed variance estimation on B2 (3 seeds on Final-Test) and optional second-seed runs on Train-Long and Bench-Control.

**Seeds are sequential integers starting at 1.** The pool manager assigns seeds; it does not control how they are used in the inference stack. LLD-03 is responsible for translating a seed assignment into a Codex invocation parameter (e.g., temperature seed or system-prompt variation).

### 6.2 Seed Assignments by Pool/Split

| Pool/Split | Max Seeds | Seed Set | Notes |
|---|---|---|---|
| Dev-Bench | 1 | {1} | One seed per model |
| Bench-Control | 2 | {1, 2} | Seed 2 conditional on budget |
| Final-Test (B2 headline) | 3 | {1, 2, 3} | 3 seeds for Base/Codex and Codex-SFT-all/Codex |
| Final-Test (B2 comparison arms) | 1 | {1} | 1 seed for matched arms and SWE-Agent runs |
| Train-Long | 2 | {1, 2} | Seed 2 conditional on budget/Gate 4 |
| Val-Long | 1 | {1} | — |
| Test-Long (B1) | 1–2 | {1} or {1, 2} | Seed 2 only if Gate 4 wall-clock allows |
| Public-Dev | 1 | {1} | — |

The `max_seeds` per pool/split is configured at initialization. LLD-07 queries the pool manager for the seed set applicable to each (pool/split, model, harness) combination.

### 6.3 Seed Configuration

```yaml
# seed_config.yaml
# Controls how many seeds are run per pool/split × model × harness.
# Updated after Gate 4 produces wall-clock estimates.

swe_bench:
  dev_bench:
    default_seeds: 1
  bench_control:
    default_seeds: 1
    max_seeds: 2          # Seed 2 enabled post-Gate-4 if budget allows
  final_test:
    # Per-model × harness seed counts — see HLD §1.9
    overrides:
      - model: "qwen3.5-27b"          # base model
        harness: codex
        seeds: 3
      - model: "codex-sft-all"        # B2 headline
        harness: codex
        seeds: 3
      - model: "*"                     # all other arms
        harness: "*"
        seeds: 1

codex_long:
  train_long:
    default_seeds: 2       # 2 seeds to maximize trace yield
    max_seeds: 3           # Seed 3 as fallback if trace count critically low (< 30)
  val_long:
    default_seeds: 1
  test_long:
    default_seeds: 1
    max_seeds: 2           # Seed 2 only if Gate 4 wall-clock allows
  public_dev:
    default_seeds: 1
```

---

## 7. Deduplication Guards

### 7.1 Purpose

Prevent duplicate runs of the same (task/env, model, harness, seed) tuple. Duplicates waste Spark compute and can distort results if both copies are accidentally included in downstream analysis.

### 7.2 Pre-Dispatch Check

Before LLD-07 dispatches a task to LLD-03 for execution, it must call the pool manager's deduplication check:

```python
def check_dispatch_eligible(
    self,
    track: str,
    pool_or_split: str,
    task_id: str,
    model_id: str,
    harness: str,
    seed: int,
) -> DispatchDecision:
    """
    Returns a dispatch decision for the requested run.
    
    DispatchDecision:
      PROCEED     — no prior run exists; safe to dispatch
      SKIP        — a completed run already exists for this key
      RETRY       — a crashed run exists and retry is allowed (attempt < 2)
      BLOCKED     — pool/split is sealed (§8) or not training-eligible (§9)
      DUPLICATE   — a running instance exists; do not dispatch a second
    """
    # Check seal enforcement first (§8)
    if self._is_sealed(pool_or_split):
        return DispatchDecision.BLOCKED

    # Query existing runs for this key
    existing = self._query_runs(track, pool_or_split, task_id, model_id, harness, seed)

    if not existing:
        return DispatchDecision.PROCEED

    latest = max(existing, key=lambda r: r.attempt)

    if latest.state == "running":
        return DispatchDecision.DUPLICATE
    if latest.state == "completed":
        return DispatchDecision.SKIP
    if latest.state == "crash" and self.can_retry(latest):
        return DispatchDecision.RETRY

    # Other terminal states (failed, timeout, no_patch) — already done, don't retry
    return DispatchDecision.SKIP
```

### 7.3 Atomicity

The transition from `PROCEED` → creating a `pending` record → setting state to `running` must be atomic. This prevents a race condition where LLD-07 dispatches the same task twice if two campaign threads query simultaneously.

```python
def claim_run(
    self,
    track: str,
    pool_or_split: str,
    task_id: str,
    model_id: str,
    harness: str,
    seed: int,
    attempt: int = 1,
) -> bool:
    """
    Atomically create a run record in 'running' state.
    Returns True if the claim succeeded, False if a concurrent claim won.
    Uses INSERT OR IGNORE + rowcount check for atomicity.
    """
    with self.db.begin() as txn:
        result = txn.execute(
            """INSERT OR IGNORE INTO runs
               (track, pool_or_split, task_id, model_id, harness, seed, attempt, state, started_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)""",
            (track, pool_or_split, task_id, model_id, harness, seed, attempt,
             datetime.utcnow().isoformat()),
        )
        return result.rowcount == 1
```

---

## 8. Seal Enforcement

### 8.1 Sealed Pools/Splits

Two pools/splits are sealed — inaccessible for any purpose until their seal condition is met:

| Pool/Split | Seal Condition | Who Unseals |
|---|---|---|
| **Final-Test** (SWE-bench) | Sprint 3 evaluation begins | LLD-07 via `unseal()` call |
| **Test-Long** (Codex-Long) | Sprint 3 evaluation begins | LLD-07 via `unseal()` call |

**Seal enforcement is a hard gate.** Any `check_dispatch_eligible()` call against a sealed pool returns `BLOCKED`. Any `list_tasks()` or `list_envs()` call against a sealed pool returns an empty list and logs a warning. This prevents accidental access during collection (Sprint 2) or orchestrator development (Sprint 1).

### 8.2 Seal State

```python
class SealState:
    """
    Manages seal status for Final-Test and Test-Long.
    
    Unsealing is a one-way operation — once unsealed, a pool cannot be re-sealed.
    Unseal events are logged with timestamp and operator identity.
    """
    def __init__(self):
        self._sealed: dict[str, bool] = {
            "final_test": True,
            "test_long": True,
        }
        self._unseal_log: list[dict] = []

    def is_sealed(self, pool_or_split: str) -> bool:
        return self._sealed.get(pool_or_split, False)

    def unseal(self, pool_or_split: str, operator: str, reason: str) -> None:
        if pool_or_split not in self._sealed:
            raise ValueError(f"'{pool_or_split}' is not a sealable pool")
        if not self._sealed[pool_or_split]:
            logger.info(f"'{pool_or_split}' is already unsealed")
            return

        self._sealed[pool_or_split] = False
        event = {
            "pool_or_split": pool_or_split,
            "operator": operator,
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat(),
        }
        self._unseal_log.append(event)
        logger.info(f"UNSEAL: {event}")

        # Persist to run_state.db
        self._persist_unseal_event(event)
```

### 8.3 Unseal Protocol

LLD-07 unseals Final-Test and Test-Long at the start of Sprint 3 evaluation. The unseal call requires an explicit operator name and reason string. The pool manager logs the event but does not enforce any additional authorization — the expectation is that the Sprint 3 campaign script is the only caller and the seal is a development-time guard, not a security boundary.

```python
# Called by LLD-07 at the start of Sprint 3 B1 evaluation campaign
pool_manager.unseal("test_long", operator="benchmark_runner", reason="Sprint 3 B1 eval start")

# Called by LLD-07 at the start of Sprint 3 B2 evaluation campaign
pool_manager.unseal("final_test", operator="benchmark_runner", reason="Sprint 3 B2 eval start")
```

---

## 9. Training Access Control

### 9.1 Purpose

LLD-10 (SFT Training Pipeline) must only train on data from training-eligible pools/splits. This prevents test-set contamination — a hard scientific validity requirement.

**Training-eligible pools/splits:**
- **Bench-Control** (SWE-bench) — yields SWE-Bench-Control-SFT (appendix diagnostic only)
- **Train-Long** (Codex-Long) — yields Codex-SFT-all and Codex-SFT-matched (mainline B1/B2)

**Never training-eligible:**
- Dev-Bench, Final-Test (SWE-bench)
- Val-Long (no gradient updates — RL early stopping and HP selection only), Test-Long, Public-Dev (Codex-Long)

### 9.2 Access Control API

```python
def list_training_eligible_tasks(
    self,
    track: str,
    model_id: str,
    harness: str,
) -> list[RunRecord]:
    """
    Return completed runs from training-eligible pools/splits only.
    
    For LLD-10: these are the runs whose trajectories may be used for SFT training.
    Filters to 'completed' state only — failed/timeout/crash runs are never training data.
    For Codex-Long, further filters to runs where codex_long_pass == True (successful traces only).
    """
    eligible_pools = {
        "swe_bench": ["bench_control"],
        "codex_long": ["train_long"],
    }

    if track not in eligible_pools:
        raise ValueError(f"Unknown track: {track}")

    results = []
    for pool in eligible_pools[track]:
        runs = self._query_runs_by_pool(
            track=track,
            pool_or_split=pool,
            model_id=model_id,
            harness=harness,
            state="completed",
        )
        if track == "codex_long":
            # Only successful traces are training data
            runs = [r for r in runs if r.codex_long_pass]
        results.extend(runs)

    return results


def assert_training_eligible(self, pool_or_split: str) -> None:
    """
    Hard check — raises if the pool/split is not training-eligible.
    Called by LLD-10 before loading any trajectory data.
    """
    TRAINING_ELIGIBLE = {"bench_control", "train_long"}
    if pool_or_split not in TRAINING_ELIGIBLE:
        raise TrainingAccessViolation(
            f"Pool/split '{pool_or_split}' is NOT training-eligible. "
            f"Only {TRAINING_ELIGIBLE} may be used for gradient updates. "
            f"This is a hard scientific validity constraint."
        )
```

### 9.3 Matched-ID Splitting for B1

LLD-10 trains two matched SFT variants for the B1 2×2 comparison: Codex-SFT-matched (Codex traces from matched scenario IDs) and SWE-Agent-SFT-matched (SWE-Agent traces from matched scenario IDs). A "matched" scenario ID is one that was solved by both harnesses.

The pool manager provides a query to identify matched scenario IDs within Train-Long:

```python
def get_matched_scenario_ids(
    self,
    model_id: str,
) -> list[str]:
    """
    Return Train-Long variant_ids (scenario IDs) that were successfully solved
    by BOTH Codex and SWE-Agent for the given model.
    
    Used by LLD-10 to construct the matched training sets for B1.
    """
    codex_successes = {
        r.task_id for r in self.list_training_eligible_tasks("codex_long", model_id, "codex")
    }
    swe_agent_successes = {
        r.task_id for r in self._query_runs_by_pool(
            track="codex_long",
            pool_or_split="train_long",
            model_id=model_id,
            harness="swe_agent",
            state="completed",
        )
        if r.codex_long_pass
    }
    return sorted(codex_successes & swe_agent_successes)
```

---

## 10. Reduced-Split Geometry (35-Family Path)

### 10.1 Pre-Declared Geometry

If Sprint 0b freezes at ~35 families, the split geometry is pre-declared by the HLD and implemented by LLD-13. The pool manager requires no code changes — it loads whatever `split_assignment.yaml` provides. The validation logic in §4.4 accommodates the 35-family path via the Public-Dev carve-out (5-type coverage cannot be achieved with ~2 families).

| Split | 55-Family Path | 35-Family Path |
|---|---|---|
| Train-Long | ~30 families / ~150–240 envs | ~20 families / ~100–160 envs |
| Val-Long | ~10 families / ~30–50 envs | ~7 families / ~21–35 envs |
| Test-Long | ~10 families / ~40–60 envs | ~6 families / ~24–36 envs |
| Public-Dev | ~5 families / ~15–20 envs | ~2 families / ~6–8 envs |

### 10.2 Rule 1 Enforcement

At load time, the pool manager checks whether the Test-Long family count meets the hard floor of 8 families (HLD Rule 1). If not, B1 is flagged as dropped. This is an informational flag — the pool manager does not block Test-Long runs, but LLD-07 and LLD-12 must check this flag before planning or reporting B1 results.

```python
def check_rule_1(self, splits: dict[str, list[CodexLongFamily]]) -> bool:
    """
    HLD Rule 1: Hard Test-Long family floor.
    If Test-Long has fewer than 8 families, B1 is dropped.
    Returns True if B1 is viable, False if B1 is dropped.
    """
    test_long_family_count = len(splits.get("test_long", []))
    b1_viable = test_long_family_count >= 8

    if not b1_viable:
        logger.warning(
            f"RULE 1 FIRED: Test-Long has {test_long_family_count} families "
            f"(floor is 8). B1 is dropped on this path."
        )

    return b1_viable
```

### 10.3 B2-Only Proceed Rule

On the 35-family path, B2 survives only if Gate 4 confirms:

- Projected Codex traces ≥ 50
- Projected Train-Long collection wall-clock ≤ 25 Spark days

These thresholds are evaluated by LLD-07 after Gate 4 completes. The pool manager stores the Gate 4 outcome as project-level metadata:

```python
@dataclass
class Gate4Outcome:
    total_families: int
    b1_viable: bool            # Rule 1 result
    projected_codex_traces: int
    projected_wall_clock_days: float
    projected_matched_ids: int
    projected_matched_families: int
    b2_viable: bool            # True if traces ≥ 50 AND wall-clock ≤ 25 (35-family) or Gate 4 PROCEED (55-family)
    gate4_decision: str        # "PROCEED" | "ADJUST" | "KILL"
    recorded_at: str           # ISO 8601
```

---

## 11. Task/Env Listing API

The primary query interface for LLD-07 (Benchmark Runner) to obtain the set of tasks or environments to run.

### 11.1 SWE-bench Task Listing

```python
def list_swe_bench_tasks(
    self,
    pool: str,
    model_id: Optional[str] = None,
    seed: Optional[int] = None,
    exclude_completed: bool = True,
) -> list[dict]:
    """
    Return SWE-bench tasks for the given pool.
    
    If exclude_completed is True, returns only tasks without a completed run
    for the given model × seed (i.e., remaining work).
    
    Returns BLOCKED (empty list + warning) if pool is sealed.
    """
    if self._is_sealed(pool):
        logger.warning(f"Attempted to list tasks from sealed pool '{pool}'")
        return []

    tasks = self.swe_bench_pools[pool]

    if exclude_completed and model_id and seed:
        completed_ids = {
            r.task_id for r in self._query_runs_by_pool(
                "swe_bench", pool, model_id=model_id, state="completed"
            )
            if r.seed == seed
        }
        tasks = [t for t in tasks if t["instance_id"] not in completed_ids]

    return tasks
```

### 11.2 Codex-Long Environment Listing

```python
def list_codex_long_envs(
    self,
    split: str,
    model_id: Optional[str] = None,
    harness: Optional[str] = None,
    seed: Optional[int] = None,
    scenario_type: Optional[str] = None,
    family_id: Optional[str] = None,
    exclude_completed: bool = True,
) -> list[CodexLongEnv]:
    """
    Return Codex-Long environments for the given split.
    
    Supports filtering by scenario_type and family_id for Gate 4 pilot
    (LLD-07 selects specific pilot families) and for per-type wall-clock
    measurement.
    
    Returns BLOCKED (empty list + warning) if split is sealed.
    """
    if self._is_sealed(split):
        logger.warning(f"Attempted to list envs from sealed split '{split}'")
        return []

    envs = self._get_envs_for_split(split)

    if scenario_type:
        envs = [e for e in envs if e.scenario_type == scenario_type]
    if family_id:
        envs = [e for e in envs if e.family_id == family_id]

    if exclude_completed and model_id and harness and seed is not None:
        completed_ids = {
            r.task_id for r in self._query_runs_by_pool(
                "codex_long", split, model_id=model_id, harness=harness, state="completed"
            )
            if r.seed == seed
        }
        envs = [e for e in envs if e.variant_id not in completed_ids]

    return envs
```

### 11.3 Family-Level Queries

```python
def list_families(
    self,
    split: Optional[str] = None,
    scenario_type: Optional[str] = None,
) -> list[CodexLongFamily]:
    """Return family metadata, optionally filtered by split and/or scenario type."""
    families = []
    target_splits = [split] if split else ["train_long", "val_long", "test_long", "public_dev"]
    for s in target_splits:
        for fam in self.codex_long_splits.get(s, []):
            if scenario_type and fam.scenario_type != scenario_type:
                continue
            families.append(fam)
    return families


def get_family_solve_summary(
    self,
    family_id: str,
    model_id: str,
    harness: str,
) -> dict:
    """
    Return solve statistics for a single family across all its variants.
    Used by LLD-07 for Gate 4 extrapolation and by LLD-12 for family-clustered bootstrap.
    """
    family = self._get_family(family_id)
    completed = [
        r for r in self._query_runs_by_family(family_id, model_id, harness)
        if r.state == "completed"
    ]
    solved = [r for r in completed if r.codex_long_pass]

    return {
        "family_id": family_id,
        "scenario_type": family.scenario_type,
        "split": family.split,
        "total_variants": family.variant_count,
        "completed_runs": len(completed),
        "solved_runs": len(solved),
        "solve_rate": len(solved) / len(completed) if completed else 0.0,
        "solved_variant_ids": [r.task_id for r in solved],
    }
```

---

## 12. Campaign Progress API

LLD-07 needs to monitor overall campaign progress and identify remaining work.

```python
def get_campaign_progress(
    self,
    track: str,
    pool_or_split: str,
    model_id: str,
    harness: str,
    seed: int,
) -> dict:
    """
    Return progress summary for a specific campaign segment.
    """
    if track == "swe_bench":
        total = len(self.swe_bench_pools[pool_or_split])
    else:
        total = sum(
            f.variant_count for f in self.codex_long_splits.get(pool_or_split, [])
        )

    runs = self._query_runs_by_pool(
        track, pool_or_split, model_id=model_id, harness=harness
    )
    seed_runs = [r for r in runs if r.seed == seed]

    by_state = {}
    for r in seed_runs:
        by_state[r.state] = by_state.get(r.state, 0) + 1

    return {
        "track": track,
        "pool_or_split": pool_or_split,
        "model_id": model_id,
        "harness": harness,
        "seed": seed,
        "total_tasks": total,
        "by_state": by_state,
        "completed": by_state.get("completed", 0),
        "remaining": total - sum(
            by_state.get(s, 0) for s in ("completed", "failed", "timeout", "crash", "no_patch")
        ),
        "solve_rate": (
            by_state.get("completed", 0) / total if total > 0 else 0.0
        ) if track == "swe_bench" else None,  # Codex-Long solve rate requires grading check
    }
```

---

## 13. Output Labeling for LLD-06

LLD-06 (Trajectory Parser) needs to label each trajectory with its pool/split and family metadata for downstream use (matched-ID splitting, family-clustered bootstrap, training-set filtering).

The pool manager exposes a labeling function that annotates a trajectory file path with its metadata:

```python
def label_trajectory(self, run: RunRecord) -> dict:
    """
    Return metadata labels for a completed run's trajectory.
    Consumed by LLD-06 for trajectory parsing and SFT data formatting.
    """
    labels = {
        "track": run.track,
        "pool_or_split": run.pool_or_split,
        "task_id": run.task_id,
        "model_id": run.model_id,
        "harness": run.harness,
        "seed": run.seed,
        "result": run.result,
        "trajectory_path": run.trajectory_path,
        "training_eligible": run.pool_or_split in {"bench_control", "train_long"},
    }

    if run.track == "codex_long":
        family = self._get_family(run.family_id)
        labels.update({
            "family_id": run.family_id,
            "scenario_type": run.scenario_type,
            "variant_id": run.task_id,
            "codex_long_pass": run.codex_long_pass,
            "manifest_version": run.manifest_version,
        })

    return labels
```

---

## 14. Initialization and Lifecycle

### 14.1 Initialization Sequence

```
1. Load SWE-bench Verified task list → assign to three pools (§3.3)
2. Load split_assignment.yaml from LLD-13 → validate (§4.4)
3. Load benchmark_manifest.lock from LLD-13 → verify split_assignment_hash
4. Build Codex-Long family and environment indices
5. Check Rule 1: Test-Long family count ≥ 8? → set b1_viable flag
6. Initialize run_state.db (create tables if not exists)
7. Load seed_config.yaml
8. Initialize seal state (Final-Test and Test-Long sealed)
9. Log initialization summary: pool sizes, family counts, type coverage, Rule 1 result
```

### 14.2 Re-Initialization After Crash

If the orchestrator or runner crashes mid-campaign, the pool manager can be re-initialized from the persisted `run_state.db`. Runs left in `running` state after a crash are detected and transitioned to `crash`:

```python
def recover_from_crash(self) -> int:
    """
    Find runs stuck in 'running' state (from a prior crash) and mark them as 'crash'.
    Returns the number of recovered runs.
    """
    with self.db.begin() as txn:
        result = txn.execute(
            """UPDATE runs SET state = 'crash', completed_at = ?
               WHERE state = 'running'""",
            (datetime.utcnow().isoformat(),),
        )
        if result.rowcount > 0:
            logger.warning(
                f"Recovered {result.rowcount} runs stuck in 'running' state → 'crash'"
            )
        return result.rowcount
```

---

## 15. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-01** vLLM Serving Layer | No direct interface. LLD-01 serves models; this LLD tracks which tasks those models are run on. LLD-01 §11 references marking tasks as `crash` in LLD-02 run state on server failure. |
| **LLD-03** Task Orchestrator | Primary writer. LLD-03 calls `claim_run()` before dispatch, updates run state on completion/failure, and writes grading results (Codex-Long pass/fail, milestones) back to the run record. |
| **LLD-05** Evaluator | Indirect — LLD-05 produces solve/fail results that LLD-03 writes to the run record here. LLD-05 does not call this LLD directly. |
| **LLD-06** Trajectory Parser | Consumer. LLD-06 calls `label_trajectory()` to annotate trajectory files with pool/split and family metadata. Uses `get_matched_scenario_ids()` for matched-ID splitting. |
| **LLD-07** Benchmark Runner | Primary reader and campaign orchestrator. LLD-07 calls `list_swe_bench_tasks()` and `list_codex_long_envs()` to get dispatch lists. Calls `get_campaign_progress()` to track completion. Calls `unseal()` at Sprint 3 start. Calls `check_dispatch_eligible()` before every dispatch. |
| **LLD-09** mini-SWE-Agent | Writer — same interface as LLD-03 but for SWE-Agent harness runs. Records run state with `harness = "swe_agent"`. |
| **LLD-10** SFT Training Pipeline | Consumer with access control. LLD-10 calls `list_training_eligible_tasks()` to get training trajectories. Must call `assert_training_eligible()` before loading any data. The pool manager enforces that only Bench-Control and Train-Long data are returned. |
| **LLD-12** Results & Artifact Generator | Consumer. LLD-12 queries run state for campaign completeness checks, solve rate computation, and family-clustered bootstrap data. Uses `get_family_solve_summary()` for B1 statistics. |
| **LLD-13** Codex-Long Scenario Framework | Upstream dependency. This LLD consumes `split_assignment.yaml` and `benchmark_manifest.lock` from LLD-13. The pool manager does not modify these files — they are read-only inputs. Family metadata flows from LLD-13 → this LLD → LLD-06 and LLD-12. |

---

## 16. Sprint 1 Validation Checklist

### Pool Initialization

- [ ] SWE-bench pool assignment generates three disjoint pools of correct sizes (50 / ~50 / 100) from SWE-bench Verified
- [ ] Pool assignment is deterministic: same seed + task list → identical pools
- [ ] No task appears in more than one pool

### Codex-Long Loading

- [ ] `split_assignment.yaml` loads successfully and passes all validation checks (§4.4)
- [ ] `benchmark_manifest.lock` hash matches `split_assignment.yaml`
- [ ] Family-disjointness validated: no family appears in multiple splits
- [ ] Type coverage validated: all 5 types in Train-Long, Val-Long, and Test-Long
- [ ] Public-Dev carve-out works correctly on both 55-family and 35-family geometry
- [ ] Rule 1 check fires correctly when Test-Long has < 8 families

### Run-State Tracking

- [ ] Run records created with correct composite keys
- [ ] State transitions follow the defined lifecycle (§5.2)
- [ ] SQLite persistence survives process restart
- [ ] Crash recovery (§14.2) correctly transitions orphaned `running` → `crash`

### Deduplication

- [ ] `check_dispatch_eligible()` returns `SKIP` for completed runs
- [ ] `check_dispatch_eligible()` returns `DUPLICATE` for currently-running runs
- [ ] `claim_run()` atomicity: concurrent claims for the same key result in exactly one success
- [ ] Retry logic: crashed runs with attempt < 2 return `RETRY`

### Seal Enforcement

- [ ] `list_swe_bench_tasks("final_test")` returns empty list while sealed
- [ ] `list_codex_long_envs("test_long")` returns empty list while sealed
- [ ] `check_dispatch_eligible()` returns `BLOCKED` for sealed pools/splits
- [ ] `unseal()` transitions seal state and logs the event
- [ ] Unseal is one-way: re-sealing is not possible

### Training Access Control

- [ ] `list_training_eligible_tasks("codex_long", ...)` returns only Train-Long data
- [ ] `list_training_eligible_tasks("swe_bench", ...)` returns only Bench-Control data
- [ ] `assert_training_eligible()` raises for Dev-Bench, Final-Test, Val-Long, Test-Long, Public-Dev
- [ ] Only successful traces (codex_long_pass == True) are returned for Codex-Long training data

### Integration

- [ ] End-to-end: LLD-07 mock dispatches 10 tasks from Dev-Bench → LLD-03 mock writes run state → pool manager reflects progress
- [ ] End-to-end: LLD-07 mock dispatches 10 Codex-Long envs from Train-Long → run state tracked with family metadata
- [ ] `label_trajectory()` returns correct metadata for both SWE-bench and Codex-Long runs
- [ ] `get_matched_scenario_ids()` correctly identifies the intersection of Codex and SWE-Agent successes

---

## 17. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **SWE-bench Verified task list may change upstream** | MEDIUM | Pin the SWE-bench Verified commit hash at initialization. Pool assignment is a function of (task list, seed) — changing the input list changes the pools. Record the commit hash in pool metadata. |
| **Codex-Long split_assignment.yaml drift after freeze** | LOW (structurally prevented) | The manifest hash check in §4.4 catches any modification. LLD-13 post-freeze rules (§12.5) govern what changes are allowed. |
| **SQLite write contention under parallel dispatches** | MEDIUM | SQLite WAL mode supports concurrent reads with serialized writes. At the expected dispatch rate (one task per ~40–110 min), contention is negligible. If parallel overnight runs (HLD mitigation option 5) create contention, consider upgrading to PostgreSQL or adding a write queue. |
| **Seed semantics differ between Codex and SWE-Agent** | MEDIUM | Seeds control sampling non-determinism. The mapping from integer seed to actual model behavior is harness-specific (Codex may use it as a temperature seed; SWE-Agent may use it differently). This LLD assigns seeds; LLD-03 and LLD-09 implement the mapping. Document the mapping in those LLDs. |
| **35-family path leaves B1 unviable (Rule 1)** | LOW (pre-registered) | This is a planned degradation path. The pool manager flags B1 as dropped but does not block Test-Long runs — Test-Long data may still be used for B2 diagnostics if needed. |
| **Gate 4 outcome changes seed configuration** | LOW | Seed config (§6.3) is loaded at initialization but may be updated after Gate 4 produces wall-clock estimates. The pool manager supports reloading `seed_config.yaml` without reinitializing the full system. |
| **Matched-ID count depends on run ordering** | LOW | The set of matched scenario IDs grows as runs complete. LLD-10 should compute matched IDs after all Train-Long collection is finished, not mid-campaign. The pool manager returns the current snapshot; LLD-10 is responsible for timing. |

---

## 18. Open Questions — Status

| Question | Status |
|---|---|
| SWE-bench Verified task list version and commit hash | **OPEN — Pin at Sprint 1 initialization. Use the latest SWE-bench Verified release at that time.** |
| Exact Bench-Control task count (50 vs slightly more/fewer) | **OPEN — Depends on SWE-bench Verified total size after excluding Dev-Bench and Final-Test. The ~50 target is a planning number; actual count determined at pool assignment.** |
| Seed-to-behavior mapping for Codex and SWE-Agent | **OPEN — LLD-03 and LLD-09 define how integer seeds translate to sampling parameters. This LLD assigns seeds only.** |
| Parallel dispatch concurrency model | **OPEN — Single-threaded dispatch assumed in Sprint 1. If parallel overnight runs (HLD mitigation option 5) are adopted post Gate 4, SQLite WAL mode may be insufficient. Decision deferred until Gate 4 wall-clock estimates are known.** |
| Val-Long access control for RL early stopping | **OPEN — Val-Long is not training-eligible (no gradient updates), but LLD-11 (DAPO, stretch) needs to read Val-Long solve rates for early stopping. Since Phase 2b is pre-killed, this access path is not designed here. If Gate 5 passes, add a `list_early_stopping_eligible()` API that returns Val-Long data read-only.** |
| Run-state database location and backup strategy | **OPEN — `run_state.db` location TBD (project root vs dedicated data directory). Backup strategy (periodic copies vs continuous replication) TBD. Loss of run state mid-campaign requires re-scanning trajectory files to reconstruct — expensive but recoverable.** |

---

*LLD-02 · Data Pool Manager · Draft v0.1 · April 2026*
