# LLD-02 · Data Pool Manager

> Codex-Bench · Low-Level Design  
> Derived from HLD Spec v2.3 · April 2026  
> Sprint: Design S1 → Implement S1 early  
> Status: SIGNED OFF v0.8

---

## Changelog

| Version | Change |
|---|---|
| v0.8 | **Two post-sign-off cleanups (non-blocking).** Cleanup 1: Cross-artifact metadata assertion added after `_find_manifest_variant()` — validates that the manifest entry's `split` and `scenario_type` match `split_assignment.yaml`. Catches stale metadata disagreement between the two frozen artifacts if one was regenerated without the other. Cleanup 2: `superseded_by` wired in `claim_run()` — when `attempt > 1` (retry or regrade), the prior attempt's `superseded_by` is set to the new attempt number within the same transaction. Eliminates dead state in the schema and provides a forward audit link from old → new attempt. |
| v0.7 | **One P0, two P1 fixes — targeting sign-off.** P0-1: Image launch contract fixed. Removed the fabricated `image_ref = "codex-long/<family>/<variant>@sha256:<digest>"` field from `CodexLongEnv` — this repo-digest format is not guaranteed to be runnable for locally-built images (LLD-13's env factory builds locally, not via a registry). `CodexLongEnv` now stores `image_digest` only, which is the local image ID (`sha256:<hex>`) produced by the env factory build. Docker accepts bare image IDs as runnable references (`docker run sha256:<hex>`), so `image_digest` is both the verification artifact (manifest hash check) and the launch reference — no fabrication needed. P1-1: Stale "preserved trajectory" language in LLD-07 and LLD-03 connection entries normalized to "retained snapshot image" / `snapshot_image_ref`, matching the v0.5 snapshot-based regrade model. P1-2: LLD-02 sign-off is explicitly conditioned on one of two outcomes being resolved during LLD-03 design: either (a) LLD-13/LLD-03 are amended to retain snapshot images (making `REGRADE_NEEDED` implementable), or (b) `REGRADE_NEEDED` is dropped and all invalidations become `RERUN_NEEDED` (simpler but wastes Spark compute on verifier-only fixes). §6.5 cross-LLD note updated to make this a concrete design decision, not an open dependency. |
| v0.6 | **One P0, two P1 fixes.** P0-1: SQL parameter binding bug fixed in `invalidate_stale_runs()` — `new_manifest_version` was bound after `family_id`/`scenario_id` params but the WHERE clause consumed it first (`{ver_column} < ?`), causing `grading_manifest_ver < family_id` comparisons. Fix: `where_params` list now starts with `new_manifest_version` before family/scenario filters, matching placeholder declaration order. P1-1: `check_dispatch_eligible()` now fails closed when `recovery_action == "regrade_only"` but `snapshot_image_ref` is null — downgrades to `RERUN_NEEDED` with a warning rather than returning an impossible work item. P1-2: Cross-LLD contract change for snapshot retention explicitly flagged in §6.5. The signed-off LLD-13 specifies post-grading snapshot cleanup; this LLD requires retention. §6.5 now includes required LLD-13/LLD-03 errata and a fallback path (drop `REGRADE_NEEDED` entirely) if the retention policy is rejected during LLD-03 design review. |
| v0.5 | **One P0, two P1 fixes.** P0-1: Regrade artifact model fixed to match the signed-off LLD-13 grading contract. The regrade-only recovery path now operates on a retained Docker snapshot image (the `docker commit` output from LLD-13 Phase 1), not on trajectory JSONL. Run records store `snapshot_image_ref` — the committed image reference that LLD-03 can re-launch Phase 2 (functional checks) and Phase 3 (integrity verification) from without re-executing the agent session. `invalidated_trajectory_path` replaced with `snapshot_image_ref`. LLD-03 is required to retain snapshot images for all finished Codex-Long runs until the run is either superseded by a new current attempt or explicitly purged; cleanup policy specified in §6.5. P1-1: Invalidation API expanded to match the full signed-off LLD-13 manifest contract. `invalidate_stale_runs()` now accepts an optional `affected_variant_ids` list for variant-scoped invalidation (prevents over-invalidating an entire family on variant-local changes). `affected_artifact` extended to include `family_spec` and `grader_image` alongside `verifier`, `milestone`, `verifier_data`, and `image`. Recovery action mapping updated: `family_spec` → `regrade_only` (functional check commands/timeouts changed); `grader_image` → `regrade_only` (trusted grading tools changed). `re_gate_required` flag from LLD-13 manifest changelog propagated to invalidated run records. P1-2: Image reference contract closed. `CodexLongEnv` stores `image_ref` — a fully qualified digest reference (`codex-long/<family_id>/<variant_id>@sha256:<digest>`) that Docker can run directly. Consumers do not reconstruct references from bare digests. `image_digest` retained as a separate field for manifest hash verification. |
| v0.4 | **Three P0 patches, one P1 cleanup.** P0-1: `get_family_solve_summary()` now exposes two first-class output tiers bound to distinct consumers. `solved_variants` (unique variants where any seed resolved) is the primary metric for Gate 4 matched-family coverage, B1 family-clustered bootstrap, and matched-ID counts. `resolved_traces` (total resolved variant × seed pairs) is the primary metric for Gate 4 trace-budget projection and the B2-only proceed rule's ≥ 50-trace threshold. Both are top-level fields; `_attempts` is removed. P0-2: Image-tag fabrication removed from `_find_manifest_variant()`. `CodexLongEnv` now stores `image_digest` only — no `image_tag` field. LLD-03 launches containers by digest (`<repo>@sha256:...`), which is what LLD-13's locked manifest actually records. The prior synthesized tag (`codex-long/.../:sha256:abc123`) did not match LLD-13's `<family_id>/<variant_id>:<build_hash>` naming convention and would have caused launch failures. P0-3: Manifest invalidation now persists a `recovery_action` per invalidated run: `regrade_only` (verifier/milestone/verifier_data change — trajectory is valid, only grading re-runs) or `rerun_full` (image change — entire agent session must be re-executed). `check_dispatch_eligible()` returns `REGRADE_NEEDED` or `RERUN_NEEDED` instead of collapsing both into `RERUN_NEEDED`. Invalidated run records also store `invalidated_trajectory_path` so the regrade path can locate the preserved agent output without re-executing. P1-1: Doc/code alignment fixes. `_find_manifest_variant()` now includes `milestone_hashes` in `required_fields`. `load_codex_long_splits()` docstring validation list item 3 ("split family counts within expected ranges") removed — the implementation does not perform this check, and the remaining validations (disjointness, type coverage, manifest hash, variant-level join) are sufficient. |
| v0.3 | **One P0, one P1 — completing the structural fixes from v0.2.** P0-1: `get_family_solve_summary()` rewritten to aggregate at the variant level, not raw attempt/row level. Multi-seed and retry rows are collapsed: each variant × seed pair uses the latest current attempt only, and each variant is counted as solved if *any* seed resolved. `total_variants`, `finished_variants`, `solved_variants` are now the primary fields; raw run counts are demoted to an `_attempts` sub-dict for auditability. Prevents Gate 4 extrapolation and B1 family-clustered bootstrap from being distorted by retry duplication or multi-seed inflation. P1-1: Variant-level manifest validation fully specified. `_find_manifest_variant()` helper implemented with explicit lookup, structured error on missing entry, and `image_tag` extraction. The split loader now performs a verified join against `benchmark_manifest.lock` for every variant — the validation claim in the docstring is now backed by executable code. |
| v0.2 | **Three P0 patches, two P1 cleanups.** P0-1: Codex-Long canonical identity fixed — introduced `scenario_id = "<family_id>/<variant_id>"` as the canonical identifier throughout. All composite keys, SQL primary keys, deduplication queries, env-listing filters, matched-ID logic, and label/export paths now use `scenario_id` instead of bare `variant_id`. Prevents collision between families that reuse the same variant name. P0-2: Manifest-version provenance split into `launch_manifest_ver` and `grading_manifest_ver`. Added `is_current` flag and `superseded_by` column to support post-freeze re-evaluation after verifier bug-fixes (LLD-13 §12.5/§12.6). Dispatch logic now returns `RERUN_NEEDED` for runs completed under stale grading artifacts; all result queries filter to `is_current = true`. Added `invalidate_stale_runs()` API. P0-3: Run lifecycle normalized to `exec_state` (pending / running / finished) + `outcome` (resolved / failed / no_patch / timeout / crash), matching the HLD failure contract exactly. All training-eligibility queries now filter on `outcome = 'resolved'` (not bare `exec_state = 'finished'`), preventing failed SWE-bench runs from leaking into training selection. Progress accounting is outcome-aware. P1-1: SWE-bench pool assignment is now a one-time generation step producing a frozen `swe_bench_pools.yaml` with a pinned upstream commit hash, matching the frozen-manifest discipline used for Codex-Long. The pool manager loads this file as immutable input — it never recomputes pool membership at runtime. P1-2: Campaign progress API now collapses retries by grouping on the logical run key and using only the latest attempt per key. A crash followed by a successful retry counts as one finished task, not two terminal events. |
| v0.1 | Initial draft. Full rewrite from pre-v2.3 three-pool SWE-bench design. Two-track architecture (SWE-bench Verified + Codex-Long). Family-level metadata tracking for Codex-Long. Dual seal enforcement (Final-Test + Test-Long). Training access control for LLD-10. Run-state tracking per pool × model × seed. Reduced-split geometry for the 35-family path. |

---

## 1. Purpose & Scope

This document specifies the Data Pool Manager — the component that owns pool and split membership, run-state tracking, seed assignment, seal enforcement, and training-set access control for the entire Codex-Bench project. Every run (collection, evaluation, or training) passes through this layer to determine which tasks or environments are eligible and to record what has been completed.

**Responsibilities:**

- Maintain two strictly separated track structures: Track 1 (SWE-bench Verified, three disjoint pools) and Track 2 (Codex-Long, four family-disjoint splits)
- Consume frozen pool/split definitions: `swe_bench_pools.yaml` (materialized once, pinned to a SWE-bench Verified commit hash) and LLD-13's `split_assignment.yaml` / `benchmark_manifest.lock`
- Use the canonical Codex-Long identifier `scenario_id = "<family_id>/<variant_id>"` in all keys, queries, and exports — preventing collision between families that share variant names
- Assign and track seeds per pool × model × harness combination
- Track run state using a two-axis lifecycle: `exec_state` (pending / running / finished) and `outcome` (resolved / failed / no_patch / timeout / crash), aligned with the HLD failure contract
- Track grading provenance (`launch_manifest_ver`, `grading_manifest_ver`) so runs graded under stale artifacts can be identified and re-evaluated after manifest bumps
- Enforce deduplication: prevent duplicate runs of the same logical key
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
│  TRACK 1 — SWE-bench Verified (frozen swe_bench_pools.yaml)     │
│  ┌──────────┐  ┌───────────────┐  ┌────────────────────────┐    │
│  │ Dev-Bench │  │ Bench-Control │  │ Final-Test (SEALED)    │    │
│  │ 50 tasks  │  │ ~50 tasks     │  │ 100 tasks              │    │
│  │ All 6     │  │ 27B only      │  │ Sprint 3 only          │    │
│  │ models    │  │ 1–2 seeds     │  │ Contribution B2 eval   │    │
│  └──────────┘  └───────────────┘  └────────────────────────┘    │
│                                                                  │
│  TRACK 2 — Codex-Long (frozen split_assignment.yaml from LLD-13)│
│  ┌────────────┐ ┌──────────┐ ┌──────────────────┐ ┌──────────┐ │
│  │ Train-Long │ │ Val-Long │ │ Test-Long        │ │Public-Dev│ │
│  │ ~30 fam    │ │ ~10 fam  │ │ (SEALED)         │ │ ~5 fam   │ │
│  │ ~150–240   │ │ ~30–50   │ │ ~10 fam / ~40–60 │ │ ~15–20   │ │
│  │ envs       │ │ envs     │ │ envs             │ │ envs     │ │
│  └────────────┘ └──────────┘ └──────────────────┘ └──────────┘ │
│                                                                  │
│  CANONICAL IDENTIFIERS                                           │
│  SWE-bench:  instance_id (globally unique)                       │
│  Codex-Long: scenario_id = "<family_id>/<variant_id>"            │
│                                                                  │
│  CROSS-TRACK ENFORCEMENT                                        │
│  Training-eligible: Bench-Control + Train-Long ONLY              │
│  Sealed until Sprint 3: Final-Test + Test-Long                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Track 1 — SWE-bench Verified Pools

### 3.1 Pool Definitions

All three pools are disjoint subsets of SWE-bench Verified. Pool membership is materialized once into a frozen `swe_bench_pools.yaml` file and loaded as immutable input — the pool manager never recomputes membership at runtime.

| Pool | Size | Models | Seeds | HLD Role |
|---|---|---|---|---|
| **Dev-Bench** | 50 tasks | All 6 (Gates 1b/1c conditional) | 1 | Contribution A leaderboard |
| **Bench-Control** | ~50 tasks | 27B only | 1–2 | In-domain SFT control arm (appendix diagnostic) |
| **Final-Test** | 100 tasks | Trained models + base + SWE-Agent baselines | 1–3 | Contribution B2 evaluation (sealed until Sprint 3) |

**Dev-Bench** (50 tasks): Published Contribution A results. Run count (~300) is baseline contingent on the full six-model lineup passing Gates 1b and 1c. Models excluded by gate failures reduce the count proportionally.

**Bench-Control** (~50 tasks): Small in-domain SFT control arm. At ~15% solve rate, yields ~7–8 successful traces per seed (~15–20 at 2 seeds). This is a design control for interpretability, not a primary training source. The resulting SWE-Bench-Control-SFT model is reported in an appendix only — not in mainline B1 or B2 results.

**Final-Test** (100 tasks): Sealed evaluation set. Unsealed only when Sprint 3 evaluation begins. Used for B2 headline results and all B2 comparison arms. See §8 for seal enforcement.

### 3.2 Frozen Pool File

Pool assignment is a **one-time generation step**, not a runtime operation. The generator script draws from the SWE-bench Verified task set using a fixed random seed, produces `swe_bench_pools.yaml`, and is never run again. This matches the frozen-manifest discipline used for Codex-Long and eliminates drift risk from upstream SWE-bench changes.

```yaml
# swe_bench_pools.yaml (frozen — do not regenerate after sign-off)
#
# Generated by: scripts/generate_swe_bench_pools.py v1.0
# Generation seed: 42
# Source: SWE-bench Verified
# Upstream commit: princeton-nlp/SWE-bench@<pinned_commit_hash>
# Generated on: 2026-05-01

upstream_commit: "princeton-nlp/SWE-bench@abc1234def5678"
generation_seed: 42
generation_date: "2026-05-01"

pools:
  dev_bench:
    tasks:
      - instance_id: "django__django-11099"
        repo: "django/django"
        base_commit: "abc1234..."
      # ... 49 more tasks
    total: 50

  bench_control:
    tasks:
      - instance_id: "sympy__sympy-20049"
        repo: "sympy/sympy"
        base_commit: "def5678..."
      # ... ~49 more tasks
    total: 50

  final_test:
    tasks:
      - instance_id: "scikit-learn__scikit-learn-13779"
        repo: "scikit-learn/scikit-learn"
        base_commit: "ghi9012..."
      # ... 99 more tasks
    total: 100
```

### 3.3 Pool Generation Script (One-Time)

The generation script is committed alongside the frozen output. It exists for reproducibility auditing — not for re-execution.

```python
import random, yaml

def generate_swe_bench_pools(
    all_tasks: list[dict],
    seed: int = 42,
    dev_bench_size: int = 50,
    bench_control_size: int = 50,
    final_test_size: int = 100,
) -> dict:
    """
    One-time pool assignment. Output is committed as swe_bench_pools.yaml.
    
    After generation, this script is NOT re-run. The frozen YAML file is the
    single source of truth for pool membership — identical to how
    split_assignment.yaml governs Codex-Long splits.
    """
    rng = random.Random(seed)
    shuffled = list(all_tasks)
    rng.shuffle(shuffled)

    total_needed = dev_bench_size + bench_control_size + final_test_size
    assert len(shuffled) >= total_needed

    cursor = 0
    pools = {}
    for pool_name, size in [
        ("dev_bench", dev_bench_size),
        ("bench_control", bench_control_size),
        ("final_test", final_test_size),
    ]:
        pools[pool_name] = {"tasks": shuffled[cursor : cursor + size], "total": size}
        cursor += size

    # Verify disjointness
    all_ids = set()
    for name, pool in pools.items():
        ids = {t["instance_id"] for t in pool["tasks"]}
        assert not (all_ids & ids), f"Pool {name} overlaps with prior pools"
        all_ids.update(ids)

    return pools
```

---

## 4. Track 2 — Codex-Long Splits

### 4.1 Relationship to LLD-13

The pool manager does **not** own Codex-Long split membership. Split membership is defined by LLD-13's frozen `split_assignment.yaml` and is consumed read-only by this component. The pool manager's responsibilities for Track 2 are:

- Load and validate the frozen split assignment at initialization
- Index family-level metadata for fast lookup (scenario type, variant list, split assignment)
- Construct canonical `scenario_id` values for each environment
- Provide filtered task/env lists to LLD-07 by split
- Track run state per scenario_id × model × harness × seed
- Enforce Test-Long seal (§8)
- Enforce Train-Long-only access for LLD-10 training (§9)

### 4.2 Canonical Codex-Long Identity

**`scenario_id = "<family_id>/<variant_id>"`** is the canonical identifier for every Codex-Long environment. LLD-13 defines `variant_id` as unique within a family, but two different families may reuse the same variant name (e.g., two families might both have a variant named `v1`). Using bare `variant_id` as a key would create collisions in run-state tracking, deduplication, and matched-ID logic.

The `scenario_id` is constructed at load time and used in every composite key, SQL primary key, dedup query, listing filter, matched-ID computation, and label/export path throughout this LLD.

```python
def make_scenario_id(family_id: str, variant_id: str) -> str:
    """Canonical Codex-Long environment identifier. Globally unique."""
    return f"{family_id}/{variant_id}"
```

### 4.3 Split Summary

| Split | Families | Envs/Family | Total Envs | Role | Training-eligible? | Sealed? |
|---|---|---|---|---|---|---|
| **Train-Long** | ~30 | ~5–8 | ~150–240 | Primary SFT/RL trajectory collection | **Yes** | No |
| **Val-Long** | ~10 | ~3–5 | ~30–50 | RL early stopping / HP selection only | No | No |
| **Test-Long** | ~10 | ~4–6 | ~40–60 | Sealed secondary benchmark | No | **Yes** — Sprint 3 only |
| **Public-Dev** | ~5 | ~3–4 | ~15–20 | Published dev set | No | No |

**Family-disjoint splits.** The split is family-disjoint, not merely environment-disjoint. Families in Val-Long and Test-Long are not seen in Train-Long — not merely unseen variants. The pool manager validates this invariant at load time.

### 4.4 Codex-Long Family and Environment Records

```python
@dataclass(frozen=True)
class CodexLongFamily:
    family_id: str                    # e.g. "dependency-migration-npm"
    scenario_type: str                # one of the 5 named types
    split: str                        # train_long | val_long | test_long | public_dev
    variant_ids: tuple[str, ...]      # e.g. ("lodash-3-to-4", "express-4-to-5", ...)
    variant_count: int
    manifest_version: int             # from benchmark_manifest.lock at load time


@dataclass(frozen=True)
class CodexLongEnv:
    family_id: str
    variant_id: str
    scenario_id: str                  # canonical: "<family_id>/<variant_id>"
    split: str
    scenario_type: str
    image_digest: str                 # from benchmark_manifest.lock; the local image ID
                                      # (sha256:<hex>) produced by the LLD-13 env factory build.
                                      # Directly runnable: `docker run <image_digest>`.
                                      # Used both as the launch reference and for manifest
                                      # hash verification (LLD-13 §12.6 Phase 1).
                                      # Note: this is a LOCAL image ID, not a registry repo
                                      # digest. LLD-13's env factory builds images locally
                                      # and tags them as codex-long/<family>/<variant>:<build_hash>,
                                      # but the tag is a human-readable alias — the digest
                                      # is the canonical locked identifier.
```

### 4.5 Variant-Level Manifest Validation

The split loader performs a verified join against `benchmark_manifest.lock` for every variant in `split_assignment.yaml`. Any variant without a manifest entry aborts initialization — this ensures no environment can be dispatched without a known image digest and verifier hash.

```python
def _find_manifest_variant(
    manifest: dict,
    family_id: str,
    variant_id: str,
) -> dict:
    """
    Look up a specific variant's entry in benchmark_manifest.lock.
    
    The manifest's `variants` list contains one entry per variant across all
    splits, keyed by (family_id, variant_id). This function performs an
    exact match and raises on miss — a missing entry means the split
    assignment references an artifact that was never locked, which would
    allow silent drift on image digests and verifier hashes.
    
    Returns the manifest entry dict with fields: family_id, variant_id,
    split, scenario_type, family_spec_hash, image_digest, verifier_hash,
    milestone_hashes, agents_md_hash, verifier_data_hash.
    
    Note: LLD-13's manifest records image_digest (sha256:...), not
    image_tag. Image tags follow the LLD-13 §5 naming convention
    (codex-long/<family_id>/<variant_id>:<build_hash>) and are a
    build-time artifact — they are NOT stored in the manifest lock.
    LLD-03 launches containers by digest, not by tag. This LLD does
    not fabricate or store image tags.
    """
    for entry in manifest.get("variants", []):
        if entry["family_id"] == family_id and entry["variant_id"] == variant_id:
            # Verify all required fields are present
            required_fields = [
                "image_digest", "verifier_hash", "family_spec_hash",
                "agents_md_hash", "verifier_data_hash", "milestone_hashes",
            ]
            missing = [f for f in required_fields if f not in entry]
            if missing:
                raise IntegrityError(
                    f"Manifest entry for '{family_id}/{variant_id}' is missing "
                    f"required fields: {missing}"
                )
            return entry

    raise IntegrityError(
        f"Variant '{family_id}/{variant_id}' appears in split_assignment.yaml "
        f"but has no entry in benchmark_manifest.lock. Cannot verify image "
        f"digest or verifier hash — this environment must not be dispatched."
    )
```

### 4.6 Split Assignment Loading and Validation

At initialization, the pool manager loads `split_assignment.yaml` and performs structural validation.

```python
def load_codex_long_splits(
    split_assignment_path: str,
    manifest_path: str,
) -> tuple[dict[str, list[CodexLongFamily]], dict[str, CodexLongEnv]]:
    """
    Load and validate the frozen Codex-Long split assignment.
    Returns (splits_by_name, env_index_by_scenario_id).
    
    Validations (all must pass — abort initialization on failure):
    1. Every family appears in exactly one split (family-disjointness)
    2. Every split contains ≥ 1 family from each of the 5 scenario types
       (except Public-Dev on the 35-family path — see §11.1 carve-out)
    3. The split_assignment.yaml hash matches the manifest's split_assignment_hash
    4. Every variant referenced in the split assignment has a corresponding
       entry in benchmark_manifest.lock with all required fields (§4.5)
    5. No two environments share the same scenario_id (global uniqueness)
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

    all_family_ids: set[str] = set()
    all_scenario_ids: set[str] = set()
    splits: dict[str, list[CodexLongFamily]] = {}
    env_index: dict[str, CodexLongEnv] = {}

    SCENARIO_TYPES = {
        "feature_evolution", "migration_refactor", "build_ci_breakage",
        "investigate_then_fix", "cross_layer_changes",
    }

    for split_name in ("train_long", "val_long", "test_long", "public_dev"):
        split_data = assignment["splits"][split_name]
        families = []
        for fam in split_data["families"]:
            fid = fam["family_id"]

            if fid in all_family_ids:
                raise IntegrityError(f"Family '{fid}' appears in multiple splits.")
            all_family_ids.add(fid)

            if fam["scenario_type"] not in SCENARIO_TYPES:
                raise ValueError(f"Family '{fid}' has unknown scenario_type '{fam['scenario_type']}'")

            variant_ids = tuple(fam.get("variant_ids", []))
            families.append(CodexLongFamily(
                family_id=fid, scenario_type=fam["scenario_type"], split=split_name,
                variant_ids=variant_ids, variant_count=fam["variant_count"],
                manifest_version=manifest["manifest_version"],
            ))

            # Build per-env index keyed by scenario_id.
            # Verified join: every variant must have a corresponding manifest entry.
            for vid in variant_ids:
                sid = make_scenario_id(fid, vid)
                if sid in all_scenario_ids:
                    raise IntegrityError(f"Duplicate scenario_id '{sid}'")
                all_scenario_ids.add(sid)

                # Look up manifest entry for this variant — hard error on miss
                manifest_entry = _find_manifest_variant(manifest, fid, vid)

                # Cross-artifact metadata consistency: the manifest entry's split
                # and scenario_type must agree with split_assignment.yaml. A mismatch
                # means one artifact was regenerated without the other.
                if manifest_entry.get("split") and manifest_entry["split"] != split_name:
                    raise IntegrityError(
                        f"Metadata disagreement for '{fid}/{vid}': "
                        f"split_assignment.yaml says split='{split_name}', "
                        f"but benchmark_manifest.lock says split='{manifest_entry['split']}'. "
                        f"One artifact may have been regenerated without the other."
                    )
                if manifest_entry.get("scenario_type") and manifest_entry["scenario_type"] != fam["scenario_type"]:
                    raise IntegrityError(
                        f"Metadata disagreement for '{fid}/{vid}': "
                        f"split_assignment.yaml says scenario_type='{fam['scenario_type']}', "
                        f"but benchmark_manifest.lock says scenario_type='{manifest_entry['scenario_type']}'."
                    )

                env_index[sid] = CodexLongEnv(
                    family_id=fid, variant_id=vid, scenario_id=sid,
                    split=split_name, scenario_type=fam["scenario_type"],
                    image_digest=manifest_entry["image_digest"],
                )

        splits[split_name] = families

        # Type coverage check (per-split)
        types_present = {f.scenario_type for f in families}
        missing_types = SCENARIO_TYPES - types_present
        if missing_types:
            if split_name == "public_dev" and len(families) < 5:
                logger.warning(
                    f"Public-Dev has only {len(families)} families — "
                    f"cannot cover all 5 scenario types. Missing: {missing_types}. "
                    f"This is expected on the 35-family path."
                )
            else:
                raise IntegrityError(
                    f"Split '{split_name}' is missing scenario types: {missing_types}."
                )

    return splits, env_index
```

---

## 5. Run-State Tracking

### 5.1 Two-Axis Lifecycle Model

Every run is tracked through a two-axis lifecycle that separates execution progress from task outcome, aligned with the HLD failure contract.

**Axis 1 — Execution state (`exec_state`):**

```
pending  →  running  →  finished
```

`finished` means the execution attempt is complete — it does not imply success. Every terminal execution reaches `finished`.

**Axis 2 — Outcome (`outcome`):** Set when `exec_state` transitions to `finished`.

```
resolved     — task solved (SWE-bench: patch applies and tests pass; Codex-Long: verifier pass)
failed       — task attempted but not solved
no_patch     — SWE-bench only: agent produced no patch file
timeout      — execution exceeded time limit
crash        — infrastructure failure (model server, container, etc.)
```

These five values match the HLD failure contract exactly. `outcome` is NULL while `exec_state` is `pending` or `running`.

**Why two axes instead of a flat state enum:** The v0.1 design used `state = completed | failed | ...` but then added a separate `result` field, creating ambiguity about which field governed training eligibility and progress accounting. Splitting into `exec_state` + `outcome` makes the semantics unambiguous: training queries filter on `outcome = 'resolved'`; progress queries count distinct `outcome` values among `finished` runs; the `exec_state` axis drives dispatch logic.

### 5.2 Composite Keys

The composite key structure differs by track.

**Track 1 (SWE-bench):** `(pool, instance_id, model_id, harness, seed)`

**Track 2 (Codex-Long):** `(split, scenario_id, model_id, harness, seed)`

where `scenario_id = "<family_id>/<variant_id>"` (§4.2).

The `harness` field distinguishes Codex from SWE-Agent runs on the same task/env — both produce run records that must be tracked independently.

### 5.3 Logical Run vs Physical Attempts

A **logical run** is a unique combination of the composite key fields. A **physical attempt** is one execution of that logical run. Retries (§5.5) create additional attempts under the same logical run.

All progress, training eligibility, and reporting queries operate on **logical runs** using the latest attempt. A crash on attempt 1 followed by a resolved attempt 2 means the logical run has `outcome = 'resolved'`. The `attempt` column is for auditability, not for counting work.

### 5.4 Run Record Schema

```python
@dataclass
class RunRecord:
    # ── Logical key ──
    track: str                     # "swe_bench" | "codex_long"
    pool_or_split: str             # "dev_bench" | "bench_control" | "final_test" |
                                   # "train_long" | "val_long" | "test_long" | "public_dev"
    scenario_id: str               # instance_id (SWE-bench) or "<family_id>/<variant_id>" (Codex-Long)
    model_id: str                  # e.g. "qwen3.5-27b"
    harness: str                   # "codex" | "swe_agent"
    seed: int                      # run seed

    # ── Physical attempt ──
    attempt: int                   # 1-indexed; incremented on retry (max 2)

    # ── Lifecycle ──
    exec_state: str                # pending | running | finished
    outcome: Optional[str]         # NULL until finished; then: resolved | failed | no_patch | timeout | crash

    # ── Metadata ──
    started_at: Optional[str]      # ISO 8601 timestamp
    completed_at: Optional[str]    # ISO 8601 timestamp
    wall_time_seconds: Optional[float]
    trajectory_path: Optional[str]   # path to JSONL output file

    # ── Provenance (Codex-Long only) ──
    family_id: Optional[str]         # extracted from scenario_id for indexed queries
    scenario_type: Optional[str]
    launch_manifest_ver: Optional[int]   # benchmark_manifest.lock version at agent launch
    grading_manifest_ver: Optional[int]  # benchmark_manifest.lock version at grading time

    # ── Validity ──
    is_current: bool               # True if this attempt's grading is valid under the current manifest.
                                   # Set to False by invalidate_stale_runs() after a manifest bump.
    superseded_by: Optional[int]   # attempt number that supersedes this one (if invalidated and re-run)
    recovery_action: Optional[str] # NULL unless invalidated; then: "regrade_only" | "rerun_full"
                                   # regrade_only: agent output is valid, only grading (Phase 2+3)
                                   #   needs re-execution from the retained snapshot image
                                   # rerun_full: image changed, entire agent session must be re-executed
    snapshot_image_ref: Optional[str]  # Docker image ref for the committed agent snapshot
                                       # (the docker commit output from LLD-13 Phase 1).
                                       # Retained for all finished Codex-Long runs so that
                                       # regrade_only recovery can re-execute Phase 2+3
                                       # without re-running the agent session. See §6.5.
    re_gate_required: bool             # From LLD-13 manifest change_log. If True, results
                                       # graded under the prior manifest must be re-evaluated
                                       # before they are reportable by LLD-12. Default False.

    # ── Grading (Codex-Long only — from LLD-13 verify_result.json) ──
    codex_long_pass: Optional[bool]
    milestone_results: Optional[dict]  # milestone_id → pass/fail
```

### 5.5 Retry Policy

Per HLD §11 (LLD-01 mid-run health monitoring): if a task crashes due to a model server failure, LLD-03 may retry once (max 2 total attempts). A retry creates a new run record with `attempt = 2`; the original record remains with its `outcome = 'crash'`.

```python
def can_retry(self, logical_key: tuple) -> bool:
    """A logical run can be retried if its latest attempt crashed and attempt < 2."""
    latest = self._get_latest_attempt(logical_key)
    return (
        latest is not None
        and latest.exec_state == "finished"
        and latest.outcome == "crash"
        and latest.attempt < 2
    )
```

### 5.6 Run-State Storage

Run state is persisted to a SQLite database (`run_state.db`) in the project data directory. SQLite provides ACID transactions for concurrent access (LLD-03 writes run state; LLD-07 reads it for campaign progress), is zero-dependency, and supports the query patterns needed.

```sql
CREATE TABLE runs (
    -- Logical key
    track          TEXT NOT NULL,
    pool_or_split  TEXT NOT NULL,
    scenario_id    TEXT NOT NULL,      -- instance_id (SWE-bench) or "family_id/variant_id" (Codex-Long)
    model_id       TEXT NOT NULL,
    harness        TEXT NOT NULL,
    seed           INTEGER NOT NULL,

    -- Physical attempt
    attempt        INTEGER NOT NULL DEFAULT 1,

    -- Lifecycle (two-axis)
    exec_state     TEXT NOT NULL DEFAULT 'pending',
    outcome        TEXT,               -- NULL until finished

    -- Metadata
    started_at     TEXT,
    completed_at   TEXT,
    wall_time_s    REAL,
    trajectory_path TEXT,

    -- Codex-Long provenance
    family_id      TEXT,               -- extracted from scenario_id; indexed for family-level queries
    scenario_type  TEXT,
    launch_manifest_ver  INTEGER,      -- manifest version verified at agent container launch
    grading_manifest_ver INTEGER,      -- manifest version verified at grading time

    -- Validity
    is_current     INTEGER NOT NULL DEFAULT 1,   -- 1 = valid; 0 = superseded by re-eval
    superseded_by  INTEGER,                       -- attempt number that replaces this one
    recovery_action TEXT,                         -- NULL unless invalidated; "regrade_only" | "rerun_full"
    re_gate_required INTEGER DEFAULT 0,           -- from LLD-13 manifest change_log; 1 = re-eval before reporting
    snapshot_image_ref TEXT,                      -- committed agent snapshot (docker commit output);
                                                  -- retained for regrade_only recovery (§6.5)

    -- Codex-Long grading
    cl_pass        INTEGER,            -- 0/1 boolean
    milestone_json TEXT,               -- JSON string of milestone results

    PRIMARY KEY (track, pool_or_split, scenario_id, model_id, harness, seed, attempt),
    CHECK (exec_state IN ('pending', 'running', 'finished')),
    CHECK (outcome IN ('resolved', 'failed', 'no_patch', 'timeout', 'crash') OR outcome IS NULL),
    CHECK (track IN ('swe_bench', 'codex_long')),
    CHECK (harness IN ('codex', 'swe_agent'))
);

-- ── Indexes ──
CREATE INDEX idx_pool_exec    ON runs(pool_or_split, exec_state);
CREATE INDEX idx_pool_outcome ON runs(pool_or_split, outcome) WHERE outcome IS NOT NULL;
CREATE INDEX idx_model         ON runs(model_id, exec_state);
CREATE INDEX idx_family        ON runs(family_id) WHERE family_id IS NOT NULL;
CREATE INDEX idx_current       ON runs(is_current) WHERE is_current = 1;

-- ── View: latest attempt per logical key (collapses retries) ──
CREATE VIEW latest_runs AS
SELECT r.*
FROM runs r
INNER JOIN (
    SELECT track, pool_or_split, scenario_id, model_id, harness, seed,
           MAX(attempt) AS max_attempt
    FROM runs
    GROUP BY track, pool_or_split, scenario_id, model_id, harness, seed
) latest
ON  r.track = latest.track
AND r.pool_or_split = latest.pool_or_split
AND r.scenario_id = latest.scenario_id
AND r.model_id = latest.model_id
AND r.harness = latest.harness
AND r.seed = latest.seed
AND r.attempt = latest.max_attempt;
```

**The `latest_runs` view** is the primary query surface for all progress, training-eligibility, and reporting queries. Raw `runs` table queries are used only for auditability (e.g., listing all attempts for a given logical run).

---

## 6. Manifest Provenance and Re-Evaluation

### 6.1 Problem

LLD-13 §12.5/§12.6 explicitly allows post-freeze manifest bumps (verifier bug-fixes, Train-Long variant additions) and requires that affected runs be re-evaluated or noted with the manifest version they ran against. Without provenance tracking, a stale grading result is indistinguishable from a valid one, and the dispatch logic wrongly skips re-runs.

### 6.2 Two Provenance Fields

Each Codex-Long run records two manifest versions:

- **`launch_manifest_ver`**: The `manifest_version` verified by LLD-03 at Phase 1 (pre-run hash checks per LLD-13 §12.6). Records the image digest, AGENTS.md hash, and family spec hash that were validated before launching the agent container.
- **`grading_manifest_ver`**: The `manifest_version` verified by LLD-03 at Phase 2 (pre-grading hash checks per LLD-13 §12.6). Records the verifier hash, milestone hashes, verifier_data hash, and grader image digest that were validated before grading.

If the manifest is bumped between launch and grading (unlikely but possible during long runs), the two versions may differ. Both are recorded for full auditability.

### 6.3 Invalidation After Manifest Bump

When LLD-13 issues a manifest version bump (e.g., verifier bug-fix for a specific family), the pool manager must invalidate affected runs so they can be re-graded or re-run.

```python
# ── Artifact → recovery action mapping ──
# Matches the full set of locked artifacts in LLD-13 benchmark_manifest.lock.
_ARTIFACT_RECOVERY = {
    # Grading artifacts — agent session output is valid; only Phase 2+3 re-run needed.
    # LLD-03 re-launches Phase 2 (functional checks FROM the retained snapshot image)
    # and Phase 3 (integrity verification in trusted grader with agent fs mounted r/o).
    "verifier":      {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "milestone":     {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "verifier_data": {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
    "family_spec":   {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
        # family_spec covers functional_checks commands/timeouts, grading_invariant
        # structure, and shortcut resistance notes — all affect Phase 2 behavior.
    "grader_image":  {"ver_column": "grading_manifest_ver", "recovery": "regrade_only"},
        # grader_image is the trusted codex-long-grader; changing it affects Phase 3
        # integrity checks. Agent session is unaffected.

    # Agent-environment artifacts — the container the agent ran in has changed.
    # Full re-execution required.
    "image":         {"ver_column": "launch_manifest_ver",  "recovery": "rerun_full"},
    "agents_md":     {"ver_column": "launch_manifest_ver",  "recovery": "rerun_full"},
        # agents_md is baked into the image and changes the task description the agent sees.
}


def invalidate_stale_runs(
    self,
    family_id: str,
    new_manifest_version: int,
    affected_artifact: str,
    reason: str,
    affected_variant_ids: Optional[list[str]] = None,
    re_gate_required: bool = False,
) -> int:
    """
    Mark runs graded/launched under a stale manifest as non-current,
    and record the recovery action needed.
    
    Parameters:
      family_id:             The family affected by the manifest bump.
      new_manifest_version:  The new manifest_version after the bump.
      affected_artifact:     One of the keys in _ARTIFACT_RECOVERY.
      reason:                Human-readable change description (from
                             manifest change_log entry).
      affected_variant_ids:  If provided, only invalidate runs for these
                             specific variants within the family. If None,
                             invalidate ALL variants in the family.
                             Prevents over-invalidation on variant-local
                             changes (e.g., a verifier bug-fix that only
                             affects one variant).
      re_gate_required:      From LLD-13 manifest change_log. If True,
                             runs graded under the prior manifest must be
                             re-evaluated before results are reportable.
                             Stored per invalidated run for LLD-12 to check.
    
    Recovery actions (persisted per invalidated run):
    
    - "regrade_only": The affected artifact is a grading artifact.
      The agent's final container state is captured in the retained
      snapshot image (docker commit output from LLD-13 Phase 1).
      LLD-03 can re-launch Phase 2 (functional checks FROM the
      snapshot) and Phase 3 (integrity verification in the trusted
      grader with agent filesystem mounted read-only) without
      re-running the agent session. Cost: seconds.
    
    - "rerun_full": The affected artifact is the Docker image or
      AGENTS.md — the agent session was executed in a different
      environment than what the manifest now specifies. The entire
      run must be re-done. Cost: 40–110 minutes of Spark compute.
    
    Returns the number of invalidated runs.
    """
    if affected_artifact not in _ARTIFACT_RECOVERY:
        raise ValueError(
            f"Unknown affected_artifact: '{affected_artifact}'. "
            f"Valid values: {sorted(_ARTIFACT_RECOVERY.keys())}"
        )

    spec = _ARTIFACT_RECOVERY[affected_artifact]
    ver_column = spec["ver_column"]
    recovery = spec["recovery"]

    # Build WHERE clause — optionally scope to specific variants
    where_clauses = [
        f"{ver_column} < ?",
        "is_current = 1",
        "exec_state = 'finished'",
    ]
    # where_params are bound in declaration order: ver_column < ? first,
    # then family/scenario filter(s). new_manifest_version MUST be first.
    where_params: list = [new_manifest_version]

    if affected_variant_ids:
        # Variant-scoped invalidation: build scenario_id list
        scenario_ids = [
            make_scenario_id(family_id, vid) for vid in affected_variant_ids
        ]
        placeholders = ",".join("?" * len(scenario_ids))
        where_clauses.append(f"scenario_id IN ({placeholders})")
        where_params.extend(scenario_ids)
    else:
        # Family-wide invalidation
        where_clauses.append("family_id = ?")
        where_params.append(family_id)

    # Full parameter tuple: SET params first, then WHERE params in clause order
    set_params = [recovery, int(re_gate_required)]

    with self.db.begin() as txn:
        result = txn.execute(
            f"""UPDATE runs
                SET is_current = 0,
                    recovery_action = ?,
                    re_gate_required = ?
                WHERE {' AND '.join(where_clauses)}""",
            (*set_params, *where_params),
        )
        count = result.rowcount
        if count > 0:
            scope = (
                f"variants {affected_variant_ids}"
                if affected_variant_ids
                else f"family '{family_id}' (all variants)"
            )
            logger.warning(
                f"Invalidated {count} runs for {scope}: "
                f"{affected_artifact} changed at manifest v{new_manifest_version}. "
                f"Recovery: {recovery}. Re-gate required: {re_gate_required}. "
                f"Reason: {reason}"
            )
        return count
```

### 6.4 Impact on Dispatch Logic

After invalidation, the dispatch logic (§7) must recognize that a logical run with only stale (non-current) attempts needs remediation. The `recovery_action` stored on the invalidated run determines the dispatch decision:

- `recovery_action = "regrade_only"` → `REGRADE_NEEDED`: LLD-07 instructs LLD-03 to re-execute grading only. LLD-03 locates the retained snapshot image via `snapshot_image_ref` on the invalidated run record, then re-runs LLD-13 Phase 2 (functional checks FROM the snapshot with `--network none`) and Phase 3 (integrity verification in the trusted `codex-long-grader` with the agent filesystem mounted read-only at `/agent/`). The agent session is not re-executed. Cost: seconds.
- `recovery_action = "rerun_full"` → `RERUN_NEEDED`: LLD-07 must re-execute the full agent session against the updated image. Cost: 40–110 minutes of Spark compute.

This distinction is why `recovery_action` is persisted per run rather than inferred at query time — the `invalidate_stale_runs()` call has the artifact-type context needed to set it correctly, and that context is not available later when `check_dispatch_eligible()` runs.

### 6.5 Snapshot Retention Policy

The regrade-only recovery path depends on the committed agent snapshot image being available after the original run finishes. LLD-03 must therefore **retain snapshot images** rather than cleaning them up immediately after grading.

**Retention rules:**

- LLD-03 retains the `docker commit` snapshot image for every finished Codex-Long run and records its reference in `snapshot_image_ref` via `finish_run()`.
- A snapshot may be removed only when one of the following is true:
  - The run has been superseded by a new current attempt (the new attempt has its own snapshot).
  - The run's pool/split campaign is fully complete and all results are finalized in LLD-12 (no further manifest bumps expected).
  - An explicit purge command is issued by the operator.
- Snapshots for SWE-bench runs are not retained (SWE-bench grading is patch-based, not state-based).

**Storage cost:** Committed images are typically small (the delta layer from the base image). At ~50–200 MB per snapshot and ~400 total Codex-Long runs (200 envs × 2 seeds), total snapshot storage is ~20–80 GB — well within the DGX Spark's local disk capacity.

**Naming convention:** `codex-long-snapshot/<scenario_id>/<model_id>/<harness>/seed<N>/attempt<M>` — deterministic from run metadata so LLD-03 can locate snapshots without querying the database.

**Cross-LLD contract change — requires LLD-13 and LLD-03 amendment:**

The signed-off LLD-13 (v0.6) specifies that LLD-03 cleans up `codex-long-snapshot/<run_id>` and `/grading/<run_id>/agent_root` after grading completes. This LLD-02 retention policy supersedes that cleanup behavior for the snapshot image specifically: LLD-03 must **not** remove the committed snapshot image after grading. The grading workspace (`/grading/<run_id>/agent_root`) may still be cleaned up — it is a temporary extraction, not the retained artifact.

This is a cross-doc contract change. Before LLD-03 implementation begins, the following amendments are required:

- **LLD-13 errata:** Add a note to §6 (or the relevant cleanup section) stating that snapshot image retention is governed by LLD-02 §6.5, and that LLD-03 must not `docker rmi` the committed snapshot after grading. The snapshot is the handle that makes `REGRADE_NEEDED` implementable.
- **LLD-03 design:** LLD-03's post-grading cleanup step must skip `docker rmi` for the committed snapshot. It should still clean up the Phase 2 and Phase 3 grading containers and any extracted filesystem trees. The snapshot image tag follows the naming convention above.

If this retention policy is rejected during LLD-03/LLD-13 design review, the alternative is to simplify the invalidation model: remove `REGRADE_NEEDED` entirely and treat all manifest-bump invalidations as `RERUN_NEEDED`. This wastes Spark compute on verifier-only bug-fixes but eliminates the retained-artifact dependency.

**Sign-off condition:** LLD-02 is signed off with the `REGRADE_NEEDED` path as specified. If LLD-03 design review concludes that snapshot retention is not feasible, the following amendments are applied to LLD-02 at that time: (1) `REGRADE_NEEDED` is removed from `DispatchDecision`; (2) all `_ARTIFACT_RECOVERY` entries map to `rerun_full`; (3) `snapshot_image_ref` becomes optional metadata rather than a recovery handle; (4) §6.5 is simplified to a recommendation, not a requirement. This is a planned simplification path, not a design failure — it trades Spark efficiency for cross-LLD simplicity.

---

## 7. Deduplication Guards and Dispatch Eligibility

### 7.1 Purpose

Prevent duplicate runs of the same logical key while ensuring that runs invalidated by manifest bumps can be re-dispatched.

### 7.2 Pre-Dispatch Check

Before LLD-07 dispatches a task to LLD-03 for execution, it must call the pool manager's dispatch check:

```python
class DispatchDecision(Enum):
    PROCEED        = "proceed"         # No prior run; safe to dispatch
    SKIP           = "skip"            # A current, finished run exists (any outcome)
    RETRY          = "retry"           # Latest attempt crashed; retry allowed (attempt < 2)
    REGRADE_NEEDED = "regrade_needed"  # Prior run invalidated — grading artifact changed;
                                       # agent output is valid, only regrade needed.
                                       # LLD-07 instructs LLD-03 to re-run Phase 2+3
                                       # from the retained snapshot image (§6.5).
    RERUN_NEEDED   = "rerun_needed"    # Prior run invalidated — image changed;
                                       # full agent re-execution required
    BLOCKED        = "blocked"         # Pool/split is sealed
    DUPLICATE      = "duplicate"       # A running instance exists; do not double-dispatch


def check_dispatch_eligible(
    self,
    track: str,
    pool_or_split: str,
    scenario_id: str,
    model_id: str,
    harness: str,
    seed: int,
) -> DispatchDecision:
    # Check seal enforcement first (§8)
    if self._is_sealed(pool_or_split):
        return DispatchDecision.BLOCKED

    # Query all attempts for this logical key
    all_attempts = self._query_runs(
        track, pool_or_split, scenario_id, model_id, harness, seed
    )

    if not all_attempts:
        return DispatchDecision.PROCEED

    latest = max(all_attempts, key=lambda r: r.attempt)

    # Currently running?
    if latest.exec_state == "running":
        return DispatchDecision.DUPLICATE

    # Finished — check currency
    if latest.exec_state == "finished":
        if latest.is_current:
            # Valid finished run exists — check if retryable crash
            if latest.outcome == "crash" and latest.attempt < 2:
                return DispatchDecision.RETRY
            return DispatchDecision.SKIP
        else:
            # Invalidated — return the specific recovery action so LLD-07
            # knows whether to regrade (seconds) or rerun (40–110 min).
            if latest.recovery_action == "regrade_only":
                # Fail closed: regrade requires the retained snapshot image.
                # If snapshot_image_ref is missing (e.g., run finished before
                # snapshot retention was implemented, or snapshot was pruned),
                # downgrade to full rerun rather than returning an impossible
                # work item.
                if latest.snapshot_image_ref:
                    return DispatchDecision.REGRADE_NEEDED
                else:
                    logger.warning(
                        f"Regrade requested for {scenario_id} but "
                        f"snapshot_image_ref is missing — downgrading to RERUN_NEEDED"
                    )
                    return DispatchDecision.RERUN_NEEDED
            else:  # "rerun_full" or missing (defensive default)
                return DispatchDecision.RERUN_NEEDED

    # Pending (shouldn't happen in normal flow, but handle gracefully)
    return DispatchDecision.PROCEED
```

### 7.3 Atomic Run Claiming

```python
def claim_run(
    self,
    track: str,
    pool_or_split: str,
    scenario_id: str,
    model_id: str,
    harness: str,
    seed: int,
    attempt: int = 1,
    launch_manifest_ver: Optional[int] = None,
    family_id: Optional[str] = None,
    scenario_type: Optional[str] = None,
) -> bool:
    """
    Atomically create a run record in 'running' state.
    Returns True if the claim succeeded, False if a concurrent claim won.
    Uses INSERT OR IGNORE + rowcount check for atomicity.
    
    When attempt > 1 (retry or regrade), also sets superseded_by on the
    prior attempt so the audit trail links old → new.
    """
    with self.db.begin() as txn:
        result = txn.execute(
            """INSERT OR IGNORE INTO runs
               (track, pool_or_split, scenario_id, model_id, harness, seed, attempt,
                exec_state, started_at, launch_manifest_ver, family_id, scenario_type)
               VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?)""",
            (track, pool_or_split, scenario_id, model_id, harness, seed, attempt,
             datetime.utcnow().isoformat(), launch_manifest_ver, family_id, scenario_type),
        )
        claimed = result.rowcount == 1

        # Link prior attempt → this one via superseded_by
        if claimed and attempt > 1:
            txn.execute(
                """UPDATE runs SET superseded_by = ?
                   WHERE track = ? AND pool_or_split = ? AND scenario_id = ?
                     AND model_id = ? AND harness = ? AND seed = ?
                     AND attempt = ?""",
                (attempt,
                 track, pool_or_split, scenario_id, model_id, harness, seed,
                 attempt - 1),
            )

        return claimed
```

### 7.4 Finishing a Run

```python
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
    trajectory_path: Optional[str] = None,
    wall_time_seconds: Optional[float] = None,
    grading_manifest_ver: Optional[int] = None,
    codex_long_pass: Optional[bool] = None,
    milestone_results: Optional[dict] = None,
    snapshot_image_ref: Optional[str] = None,
) -> None:
    """
    Transition a running attempt to finished with the given outcome.
    
    For Codex-Long runs, snapshot_image_ref should be the docker commit
    output from LLD-13 Phase 1 — the committed agent container image.
    This ref is retained so that regrade_only recovery can re-execute
    Phase 2+3 without re-running the agent session (see §6.5).
    """
    assert outcome in ("resolved", "failed", "no_patch", "timeout", "crash")

    with self.db.begin() as txn:
        txn.execute(
            """UPDATE runs
               SET exec_state = 'finished', outcome = ?, completed_at = ?,
                   wall_time_s = ?, trajectory_path = ?,
                   grading_manifest_ver = ?,
                   cl_pass = ?, milestone_json = ?,
                   snapshot_image_ref = ?
               WHERE track = ? AND pool_or_split = ? AND scenario_id = ?
                 AND model_id = ? AND harness = ? AND seed = ? AND attempt = ?
                 AND exec_state = 'running'""",
            (outcome, datetime.utcnow().isoformat(), wall_time_seconds,
             trajectory_path, grading_manifest_ver,
             int(codex_long_pass) if codex_long_pass is not None else None,
             json.dumps(milestone_results) if milestone_results else None,
             snapshot_image_ref,
             track, pool_or_split, scenario_id, model_id, harness, seed, attempt),
        )
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
def list_training_eligible_runs(
    self,
    track: str,
    model_id: str,
    harness: str,
) -> list[RunRecord]:
    """
    Return RESOLVED runs from training-eligible pools/splits only, using
    the latest current attempt per logical key.
    
    For LLD-10: these are the runs whose trajectories may be used for SFT.
    
    Filters:
    1. Pool/split must be training-eligible (Bench-Control or Train-Long)
    2. exec_state = 'finished' AND outcome = 'resolved'
    3. is_current = true (not invalidated by manifest bump)
    4. Latest attempt only (collapses retries)
    5. For Codex-Long: codex_long_pass = true (verifier passed)
    
    Note: SWE-bench outcome = 'resolved' already implies the patch applies
    and tests pass. Codex-Long additionally requires the verifier pass check
    because 'resolved' is set by LLD-03 based on verifier output, but the
    explicit cl_pass filter is a defense-in-depth guard.
    """
    eligible_pools = {
        "swe_bench": ["bench_control"],
        "codex_long": ["train_long"],
    }

    if track not in eligible_pools:
        raise ValueError(f"Unknown track: {track}")

    results = []
    for pool in eligible_pools[track]:
        runs = self._query_latest_current_runs(
            track=track,
            pool_or_split=pool,
            model_id=model_id,
            harness=harness,
            outcome="resolved",
        )
        if track == "codex_long":
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

```python
def get_matched_scenario_ids(
    self,
    model_id: str,
) -> list[str]:
    """
    Return Train-Long scenario_ids that were successfully resolved
    by BOTH Codex and SWE-Agent for the given model.
    
    Uses canonical scenario_id (family_id/variant_id) — globally unique,
    so no collision between families that reuse variant names.
    
    Used by LLD-10 to construct the matched training sets for B1.
    Should only be called after all Train-Long collection is complete.
    """
    codex_successes = {
        r.scenario_id
        for r in self.list_training_eligible_runs("codex_long", model_id, "codex")
    }
    swe_agent_resolved = {
        r.scenario_id
        for r in self._query_latest_current_runs(
            track="codex_long",
            pool_or_split="train_long",
            model_id=model_id,
            harness="swe_agent",
            outcome="resolved",
        )
        if r.codex_long_pass
    }
    return sorted(codex_successes & swe_agent_resolved)
```

---

## 10. Seed Assignment

### 10.1 Seed Semantics

A seed controls the non-determinism in the model's sampling during a task attempt. Different seeds on the same task produce different trajectories — this is required for multi-seed variance estimation on B2 (3 seeds on Final-Test) and optional second-seed runs on Train-Long and Bench-Control.

**Seeds are sequential integers starting at 1.** The pool manager assigns seeds; it does not control how they are used in the inference stack. LLD-03 is responsible for translating a seed assignment into a Codex invocation parameter (e.g., temperature seed or system-prompt variation).

### 10.2 Seed Assignments by Pool/Split

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

### 10.3 Seed Configuration

```yaml
# seed_config.yaml
# Controls how many seeds are run per pool/split × model × harness.
# Updated after Gate 4 produces wall-clock estimates.

swe_bench:
  dev_bench:
    default_seeds: 1
  bench_control:
    default_seeds: 1
    max_seeds: 2
  final_test:
    overrides:
      - model: "qwen3.5-27b"
        harness: codex
        seeds: 3
      - model: "codex-sft-all"
        harness: codex
        seeds: 3
      - model: "*"
        harness: "*"
        seeds: 1

codex_long:
  train_long:
    default_seeds: 2
    max_seeds: 3
  val_long:
    default_seeds: 1
  test_long:
    default_seeds: 1
    max_seeds: 2
  public_dev:
    default_seeds: 1
```

---

## 11. Reduced-Split Geometry (35-Family Path)

### 11.1 Pre-Declared Geometry

If Sprint 0b freezes at ~35 families, the split geometry is pre-declared by the HLD and implemented by LLD-13. The pool manager requires no code changes — it loads whatever `split_assignment.yaml` provides. The validation logic in §4.6 accommodates the 35-family path via the Public-Dev carve-out.

| Split | 55-Family Path | 35-Family Path |
|---|---|---|
| Train-Long | ~30 families / ~150–240 envs | ~20 families / ~100–160 envs |
| Val-Long | ~10 families / ~30–50 envs | ~7 families / ~21–35 envs |
| Test-Long | ~10 families / ~40–60 envs | ~6 families / ~24–36 envs |
| Public-Dev | ~5 families / ~15–20 envs | ~2 families / ~6–8 envs |

### 11.2 Rule 1 Enforcement

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

### 11.3 B2-Only Proceed Rule

On the 35-family path, B2 survives only if Gate 4 confirms projected Codex traces ≥ 50 AND projected Train-Long collection wall-clock ≤ 25 Spark days. These thresholds are evaluated by LLD-07 after Gate 4 completes. The pool manager stores the Gate 4 outcome as project-level metadata:

```python
@dataclass
class Gate4Outcome:
    total_families: int
    b1_viable: bool
    projected_codex_traces: int
    projected_wall_clock_days: float
    projected_matched_ids: int
    projected_matched_families: int
    b2_viable: bool
    gate4_decision: str        # "PROCEED" | "ADJUST" | "KILL"
    recorded_at: str           # ISO 8601
```

---

## 12. Task/Env Listing API

The primary query interface for LLD-07 (Benchmark Runner) to obtain the set of tasks or environments to run.

### 12.1 SWE-bench Task Listing

```python
def list_swe_bench_tasks(
    self,
    pool: str,
    model_id: Optional[str] = None,
    harness: Optional[str] = None,
    seed: Optional[int] = None,
    exclude_finished: bool = True,
) -> list[dict]:
    """
    Return SWE-bench tasks for the given pool.
    
    If exclude_finished is True, returns only tasks without a current finished
    run (any outcome) for the given model × harness × seed.
    
    Returns BLOCKED (empty list + warning) if pool is sealed.
    """
    if self._is_sealed(pool):
        logger.warning(f"Attempted to list tasks from sealed pool '{pool}'")
        return []

    tasks = self.swe_bench_pools[pool]

    if exclude_finished and model_id and harness and seed is not None:
        finished_ids = {
            r.scenario_id
            for r in self._query_latest_current_runs(
                "swe_bench", pool, model_id=model_id, harness=harness
            )
            if r.exec_state == "finished" and r.is_current
            and r.seed == seed
        }
        tasks = [t for t in tasks if t["instance_id"] not in finished_ids]

    return tasks
```

### 12.2 Codex-Long Environment Listing

```python
def list_codex_long_envs(
    self,
    split: str,
    model_id: Optional[str] = None,
    harness: Optional[str] = None,
    seed: Optional[int] = None,
    scenario_type: Optional[str] = None,
    family_id: Optional[str] = None,
    exclude_finished: bool = True,
) -> list[CodexLongEnv]:
    """
    Return Codex-Long environments for the given split.
    
    Supports filtering by scenario_type and family_id for Gate 4 pilot
    (LLD-07 selects specific pilot families) and for per-type wall-clock
    measurement.
    
    Uses scenario_id (not bare variant_id) for completion filtering.
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

    if exclude_finished and model_id and harness and seed is not None:
        finished_ids = {
            r.scenario_id
            for r in self._query_latest_current_runs(
                "codex_long", split, model_id=model_id, harness=harness
            )
            if r.exec_state == "finished" and r.is_current
            and r.seed == seed
        }
        envs = [e for e in envs if e.scenario_id not in finished_ids]

    return envs
```

### 12.3 Family-Level Queries

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
    Return solve statistics for a single family with two first-class
    output tiers, each bound to distinct downstream consumers.
    
    Tier 1 — Variant-level coverage (unique variants solved):
      A variant counts as solved if ANY seed produced outcome = resolved.
      A variant run at 2 seeds where both resolve → 1 solved variant.
      Consumer: Gate 4 matched-family coverage, B1 family-clustered
      bootstrap, matched-ID counts, Gate 4 diversity checks.
    
    Tier 2 — Trace-level yield (total resolved variant × seed pairs):
      Each resolved (variant, seed) pair is one usable training trace.
      A variant run at 2 seeds where both resolve → 2 resolved traces.
      Consumer: Gate 4 projected Codex trace yield (HLD threshold ≥ 80
      for PROCEED, ≥ 50 for B2-only), LLD-10 training set sizing.
    
    Both tiers collapse retries: each (scenario_id, seed) pair uses
    the latest current attempt only. Multi-attempt inflation is
    impossible regardless of tier.
    """
    family = self._get_family(family_id)

    # All latest current attempts for this family × model × harness
    latest_runs = [
        r for r in self._query_latest_current_runs_by_family(family_id, model_id, harness)
        if r.exec_state == "finished"
    ]

    # ── Group by (scenario_id, seed) — one entry per logical run ──
    # Then aggregate up to variant level for Tier 1
    variant_outcomes: dict[str, dict] = {}   # scenario_id → {seeds_finished, seeds_resolved}
    total_resolved_traces = 0

    for r in latest_runs:
        sid = r.scenario_id
        if sid not in variant_outcomes:
            variant_outcomes[sid] = {"seeds_finished": 0, "seeds_resolved": 0}
        variant_outcomes[sid]["seeds_finished"] += 1
        if r.outcome == "resolved" and r.codex_long_pass:
            variant_outcomes[sid]["seeds_resolved"] += 1
            total_resolved_traces += 1

    finished_variants = len(variant_outcomes)
    solved_variants = sum(
        1 for v in variant_outcomes.values() if v["seeds_resolved"] > 0
    )
    solved_scenario_ids = sorted(
        sid for sid, v in variant_outcomes.items() if v["seeds_resolved"] > 0
    )

    return {
        "family_id": family_id,
        "scenario_type": family.scenario_type,
        "split": family.split,
        "total_variants": family.variant_count,

        # ── Tier 1: variant-level coverage ──
        # Use for: Gate 4 matched-family coverage, B1 bootstrap, matched-ID counts
        "finished_variants": finished_variants,
        "solved_variants": solved_variants,
        "variant_solve_rate": solved_variants / finished_variants if finished_variants > 0 else 0.0,
        "solved_scenario_ids": solved_scenario_ids,

        # ── Tier 2: trace-level yield ──
        # Use for: Gate 4 projected trace count, B2-only proceed rule (≥ 50),
        #          LLD-10 training set sizing
        "resolved_traces": total_resolved_traces,
        "total_finished_runs": len(latest_runs),
    }
```

---

## 13. Campaign Progress API

LLD-07 needs to monitor overall campaign progress and identify remaining work. All progress queries collapse retries by operating on the `latest_runs` view — a crash followed by a successful retry counts as one finished logical run, not two terminal events.

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
    
    Uses the latest attempt per logical key (collapses retries).
    Reports outcome breakdown among finished runs.
    """
    if track == "swe_bench":
        total = len(self.swe_bench_pools[pool_or_split])
    else:
        total = sum(
            f.variant_count for f in self.codex_long_splits.get(pool_or_split, [])
        )

    # Query latest attempt per logical key for this seed
    latest = [
        r for r in self._query_latest_current_runs(
            track, pool_or_split, model_id=model_id, harness=harness
        )
        if r.seed == seed
    ]

    # Count by outcome (only among finished runs)
    by_outcome = {}
    pending_or_running = 0
    for r in latest:
        if r.exec_state == "finished" and r.outcome:
            by_outcome[r.outcome] = by_outcome.get(r.outcome, 0) + 1
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
```

---

## 14. Output Labeling for LLD-06

LLD-06 (Trajectory Parser) needs to label each trajectory with its pool/split and family metadata for downstream use (matched-ID splitting, family-clustered bootstrap, training-set filtering).

```python
def label_trajectory(self, run: RunRecord) -> dict:
    """
    Return metadata labels for a finished run's trajectory.
    Consumed by LLD-06 for trajectory parsing and SFT data formatting.
    """
    labels = {
        "track": run.track,
        "pool_or_split": run.pool_or_split,
        "scenario_id": run.scenario_id,
        "model_id": run.model_id,
        "harness": run.harness,
        "seed": run.seed,
        "outcome": run.outcome,
        "trajectory_path": run.trajectory_path,
        "training_eligible": run.pool_or_split in {"bench_control", "train_long"},
        "is_current": run.is_current,
    }

    if run.track == "codex_long":
        labels.update({
            "family_id": run.family_id,
            "scenario_type": run.scenario_type,
            "variant_id": run.scenario_id.split("/", 1)[1],  # extract from canonical form
            "codex_long_pass": run.codex_long_pass,
            "launch_manifest_ver": run.launch_manifest_ver,
            "grading_manifest_ver": run.grading_manifest_ver,
        })

    return labels
```

---

## 15. Initialization and Lifecycle

### 15.1 Initialization Sequence

```
1. Load swe_bench_pools.yaml (frozen input) → validate pool sizes and disjointness
2. Load split_assignment.yaml from LLD-13 → validate (§4.6)
3. Load benchmark_manifest.lock from LLD-13 → verify split_assignment_hash
4. Build Codex-Long family and environment indices (keyed by scenario_id)
5. Check Rule 1: Test-Long family count ≥ 8? → set b1_viable flag
6. Initialize run_state.db (create tables + latest_runs view if not exists)
7. If resuming: call recover_from_crash() to handle orphaned 'running' records
8. Load seed_config.yaml
9. Initialize seal state (Final-Test and Test-Long sealed)
10. Log initialization summary: pool sizes, family counts, type coverage, Rule 1 result
```

### 15.2 Re-Initialization After Crash

If the orchestrator or runner crashes mid-campaign, the pool manager can be re-initialized from the persisted `run_state.db`. Runs left in `running` state after a crash are detected and transitioned to `finished` with `outcome = 'crash'`:

```python
def recover_from_crash(self) -> int:
    """
    Find runs stuck in 'running' state (from a prior crash) and finish them
    with outcome = 'crash'. Returns the number of recovered runs.
    """
    with self.db.begin() as txn:
        result = txn.execute(
            """UPDATE runs
               SET exec_state = 'finished', outcome = 'crash', completed_at = ?
               WHERE exec_state = 'running'""",
            (datetime.utcnow().isoformat(),),
        )
        if result.rowcount > 0:
            logger.warning(
                f"Recovered {result.rowcount} runs stuck in 'running' → 'crash'"
            )
        return result.rowcount
```

---

## 16. Connections to Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-01** vLLM Serving Layer | No direct interface. LLD-01 serves models; this LLD tracks which tasks those models are run on. LLD-01 §11 references marking tasks with `outcome = 'crash'` in LLD-02 run state on server failure. |
| **LLD-03** Task Orchestrator | Primary writer. LLD-03 calls `claim_run()` before dispatch (with `launch_manifest_ver` for Codex-Long), calls `finish_run()` on completion/failure (with `grading_manifest_ver`, `codex_long_pass`/`milestone_results`, and `snapshot_image_ref` for Codex-Long). Launches Codex-Long agent containers using `image_digest` from `CodexLongEnv` (the local image ID, directly runnable via `docker run <digest>`). Retains committed snapshot images per §6.5. For `REGRADE_NEEDED` runs, LLD-03 locates the retained snapshot via `snapshot_image_ref` and re-executes only Phase 2+3 grading (not the agent session). |
| **LLD-05** Evaluator | Indirect — LLD-05 produces solve/fail outcomes that LLD-03 writes to the run record here. LLD-05 does not call this LLD directly. |
| **LLD-06** Trajectory Parser | Consumer. LLD-06 calls `label_trajectory()` to annotate trajectory files with pool/split, family metadata, and `scenario_id`. Uses `get_matched_scenario_ids()` for matched-ID splitting. |
| **LLD-07** Benchmark Runner | Primary reader and campaign orchestrator. Calls `list_swe_bench_tasks()` and `list_codex_long_envs()` for dispatch lists. Calls `get_campaign_progress()` for completion tracking. Calls `unseal()` at Sprint 3 start. Calls `check_dispatch_eligible()` before every dispatch — must handle `REGRADE_NEEDED` (re-invoke grading from retained snapshot image) and `RERUN_NEEDED` (full agent re-execution) as distinct recovery paths. Calls `invalidate_stale_runs()` after manifest bumps. Uses `get_family_solve_summary()` Tier 1 fields for Gate 4 family coverage and Tier 2 fields for trace-budget projection. |
| **LLD-09** mini-SWE-Agent | Writer — same interface as LLD-03 but for SWE-Agent harness runs. Records run state with `harness = "swe_agent"`. |
| **LLD-10** SFT Training Pipeline | Consumer with access control. Calls `list_training_eligible_runs()` to get training trajectories. Must call `assert_training_eligible()` before loading any data. The pool manager enforces that only `outcome = 'resolved'` runs from Bench-Control and Train-Long are returned. |
| **LLD-12** Results & Artifact Generator | Consumer. Queries run state for campaign completeness checks, outcome breakdown, and family-clustered bootstrap data. Uses `get_family_solve_summary()` for B1 statistics. All queries filter to `is_current = true`. |
| **LLD-13** Codex-Long Scenario Framework | Upstream dependency. This LLD consumes `split_assignment.yaml` and `benchmark_manifest.lock` from LLD-13. The pool manager does not modify these files — they are read-only inputs. Family metadata flows from LLD-13 → this LLD → LLD-06 and LLD-12. Manifest version bumps from LLD-13 trigger `invalidate_stale_runs()` in this LLD. |

---

## 17. Sprint 1 Validation Checklist

### Pool Initialization

- [ ] `swe_bench_pools.yaml` loads successfully with correct pool sizes (50 / ~50 / 100)
- [ ] All three SWE-bench pools are disjoint (no task in multiple pools)
- [ ] Upstream commit hash is recorded in the frozen file
- [ ] Pool manager rejects any attempt to regenerate pools at runtime

### Codex-Long Loading

- [ ] `split_assignment.yaml` loads and passes all validation checks (§4.6)
- [ ] `benchmark_manifest.lock` hash matches `split_assignment.yaml`
- [ ] Family-disjointness validated: no family appears in multiple splits
- [ ] All `scenario_id` values are globally unique (no collision between families)
- [ ] Type coverage validated: all 5 types in Train-Long, Val-Long, and Test-Long
- [ ] Public-Dev carve-out works correctly on both 55-family and 35-family geometry
- [ ] Rule 1 check fires correctly when Test-Long has < 8 families
- [ ] Every variant in `split_assignment.yaml` has a matching `benchmark_manifest.lock` entry (§4.5)
- [ ] Cross-artifact metadata assertion: manifest entry `split` and `scenario_type` match `split_assignment.yaml` for every variant
- [ ] `_find_manifest_variant()` raises `IntegrityError` for variants missing from the manifest
- [ ] `_find_manifest_variant()` raises `IntegrityError` for entries missing required fields (image_digest, verifier_hash, etc.)

### Run-State Tracking

- [ ] Run records use `scenario_id` (not bare `variant_id`) in the primary key
- [ ] `exec_state` / `outcome` two-axis lifecycle transitions correctly
- [ ] `outcome` is NULL while `exec_state` is `pending` or `running`
- [ ] `outcome` is set to a valid value when `exec_state` transitions to `finished`
- [ ] SQLite persistence survives process restart
- [ ] Crash recovery (§15.2) correctly transitions orphaned `running` → `finished` / `crash`
- [ ] `latest_runs` view returns only the highest-attempt row per logical key
- [ ] `claim_run()` with `attempt > 1` sets `superseded_by` on the prior attempt within the same transaction
- [ ] `superseded_by` links old → new attempt (e.g., attempt 1 `superseded_by = 2` after retry)

### Manifest Provenance

- [ ] `launch_manifest_ver` and `grading_manifest_ver` are recorded separately
- [ ] `finish_run()` records `snapshot_image_ref` for Codex-Long runs
- [ ] `invalidate_stale_runs()` handles all 7 artifact types: verifier, milestone, verifier_data, family_spec, grader_image, image, agents_md
- [ ] `invalidate_stale_runs()` sets `recovery_action = 'regrade_only'` for grading artifacts (verifier, milestone, verifier_data, family_spec, grader_image)
- [ ] `invalidate_stale_runs()` sets `recovery_action = 'rerun_full'` for agent-environment artifacts (image, agents_md)
- [ ] `invalidate_stale_runs()` with `affected_variant_ids` invalidates only those variants (not the entire family)
- [ ] `invalidate_stale_runs()` without `affected_variant_ids` invalidates all variants in the family
- [ ] `invalidate_stale_runs()` propagates `re_gate_required` from manifest change_log
- [ ] Snapshot images retained per §6.5 — not cleaned up after grading
- [ ] After grading-artifact invalidation, `check_dispatch_eligible()` returns `REGRADE_NEEDED`
- [ ] After image/agents_md invalidation, `check_dispatch_eligible()` returns `RERUN_NEEDED`
- [ ] `REGRADE_NEEDED` fails closed: if `snapshot_image_ref` is null, downgrades to `RERUN_NEEDED`
- [ ] SQL parameter binding in `invalidate_stale_runs()`: `new_manifest_version` binds to `{ver_column} < ?` (first WHERE placeholder), not to family/scenario filters
- [ ] `list_training_eligible_runs()` and `get_family_solve_summary()` filter to `is_current = 1`
- [ ] Re-run/regrade after invalidation creates a new attempt with the updated manifest version

### Deduplication

- [ ] `check_dispatch_eligible()` returns `SKIP` for current finished runs
- [ ] `check_dispatch_eligible()` returns `DUPLICATE` for currently-running runs
- [ ] `check_dispatch_eligible()` returns `REGRADE_NEEDED` for grading-artifact-invalidated runs
- [ ] `check_dispatch_eligible()` returns `RERUN_NEEDED` for image/agents_md-invalidated runs
- [ ] `claim_run()` atomicity: concurrent claims for the same key result in exactly one success
- [ ] Retry logic: latest attempt crashed with `attempt < 2` returns `RETRY`

### Seal Enforcement

- [ ] `list_swe_bench_tasks("final_test")` returns empty list while sealed
- [ ] `list_codex_long_envs("test_long")` returns empty list while sealed
- [ ] `check_dispatch_eligible()` returns `BLOCKED` for sealed pools/splits
- [ ] `unseal()` transitions seal state and logs the event
- [ ] Unseal is one-way: re-sealing is not possible

### Training Access Control

- [ ] `list_training_eligible_runs("codex_long", ...)` returns only Train-Long data
- [ ] `list_training_eligible_runs("swe_bench", ...)` returns only Bench-Control data
- [ ] `assert_training_eligible()` raises for Dev-Bench, Final-Test, Val-Long, Test-Long, Public-Dev
- [ ] Only `outcome = 'resolved'` runs are returned (not `failed`, `no_patch`, etc.)
- [ ] Only `is_current = true` runs are returned (not invalidated runs)
- [ ] For Codex-Long: `codex_long_pass = true` filter applied as defense-in-depth

### Campaign Progress

- [ ] `get_campaign_progress()` collapses retries (crash + retry → one logical run)
- [ ] Progress counts by `outcome`, not by flat state
- [ ] A crash on attempt 1 followed by a resolved attempt 2 shows `resolved: 1` (not `crash: 1, resolved: 1`)

### Family Solve Summary

- [ ] `get_family_solve_summary()` exposes both Tier 1 (variant coverage) and Tier 2 (trace yield) as first-class fields
- [ ] Tier 1: a variant run at 2 seeds where both resolve counts as `solved_variants: 1`
- [ ] Tier 2: a variant run at 2 seeds where both resolve counts as `resolved_traces: 2`
- [ ] Both tiers collapse retries: a crashed attempt 1 and resolved attempt 2 → one resolved trace (not two)
- [ ] `variant_solve_rate` = `solved_variants / finished_variants`
- [ ] Gate 4 trace projection consumers use `resolved_traces`, not `solved_variants`
- [ ] Gate 4 family coverage consumers use `solved_variants`, not `resolved_traces`

### Integration

- [ ] End-to-end: LLD-07 mock dispatches 10 tasks from Dev-Bench → LLD-03 mock writes run state → pool manager reflects progress
- [ ] End-to-end: LLD-07 mock dispatches 10 Codex-Long envs from Train-Long → run state tracked with `scenario_id` and family metadata
- [ ] `label_trajectory()` returns correct metadata including `scenario_id` for both tracks
- [ ] `get_matched_scenario_ids()` uses `scenario_id` and correctly identifies the intersection

---

## 18. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| **Codex-Long variant_id reuse across families** | MEDIUM (structurally prevented) | Canonical `scenario_id = "<family_id>/<variant_id>"` used everywhere (§4.2). Validated at load time — duplicate `scenario_id` values abort initialization. |
| **Post-freeze manifest bumps invalidate completed runs** | MEDIUM | `invalidate_stale_runs()` (§6.3) marks affected runs as non-current with a specific `recovery_action` and optional variant-scoping. `regrade_only` re-executes Phase 2+3 from the retained snapshot image (seconds); `rerun_full` re-executes the agent session (40–110 min). Snapshot images retained per §6.5. Variant-scoped invalidation prevents over-invalidation on variant-local changes. |
| **Snapshot image retention increases local disk usage** | LOW | At ~50–200 MB per snapshot and ~400 total Codex-Long runs, total snapshot storage is ~20–80 GB. Well within DGX Spark local disk. Snapshots are pruned after campaign finalization (§6.5). |
| **Codex-Long split_assignment.yaml drift after freeze** | LOW (structurally prevented) | The manifest hash check in §4.6 catches any modification. LLD-13 post-freeze rules (§12.5) govern what changes are allowed. |
| **SQLite write contention under parallel dispatches** | MEDIUM | SQLite WAL mode supports concurrent reads with serialized writes. At the expected dispatch rate (one task per ~40–110 min), contention is negligible. If parallel overnight runs (HLD mitigation option 5) create contention, consider upgrading to PostgreSQL or adding a write queue. |
| **Seed semantics differ between Codex and SWE-Agent** | MEDIUM | Seeds control sampling non-determinism. The mapping from integer seed to actual model behavior is harness-specific. This LLD assigns seeds; LLD-03 and LLD-09 implement the mapping. |
| **35-family path leaves B1 unviable (Rule 1)** | LOW (pre-registered) | Planned degradation path. The pool manager flags B1 as dropped but does not block Test-Long runs. |
| **Gate 4 outcome changes seed configuration** | LOW | Seed config (§10.3) is loaded at initialization but may be updated after Gate 4. The pool manager supports reloading `seed_config.yaml` without reinitializing the full system. |
| **Matched-ID count depends on run ordering** | LOW | LLD-10 should compute matched IDs after all Train-Long collection is finished, not mid-campaign. The pool manager returns the current snapshot; LLD-10 is responsible for timing. |
| **Family solve summary consumers must use the correct tier** | LOW | `get_family_solve_summary()` exposes Tier 1 (variant coverage: `solved_variants`, `variant_solve_rate`) and Tier 2 (trace yield: `resolved_traces`). Gate 4 trace-budget projection and the B2-only ≥ 50-trace threshold must use `resolved_traces`, not `solved_variants`. Gate 4 matched-family coverage and B1 bootstrap must use `solved_variants`, not `resolved_traces`. Misuse would undercount trace yield (Tier 1 for budget) or inflate family coverage (Tier 2 for coverage). |
| **SWE-bench upstream task list changes** | LOW (structurally prevented) | Pool membership is materialized in a frozen `swe_bench_pools.yaml` with the upstream commit hash pinned. The pool manager never re-reads from upstream. |

---

## 19. Open Questions — Status

| Question | Status |
|---|---|
| SWE-bench Verified upstream commit hash for pool generation | **Resolved (§3.2):** Pinned in `swe_bench_pools.yaml` at generation time. The frozen file is the single source of truth. |
| Exact Bench-Control task count (50 vs slightly more/fewer) | **OPEN — Depends on SWE-bench Verified total size after excluding Dev-Bench and Final-Test. The ~50 target is a planning number; actual count determined at pool generation and frozen.** |
| Seed-to-behavior mapping for Codex and SWE-Agent | **OPEN — LLD-03 and LLD-09 define how integer seeds translate to sampling parameters. This LLD assigns seeds only.** |
| Parallel dispatch concurrency model | **OPEN — Single-threaded dispatch assumed in Sprint 1. If parallel overnight runs (HLD mitigation option 5) are adopted post Gate 4, SQLite WAL mode may be insufficient. Decision deferred until Gate 4 wall-clock estimates are known.** |
| Val-Long access control for RL early stopping | **OPEN — Val-Long is not training-eligible (no gradient updates), but LLD-11 (DAPO, stretch) needs to read Val-Long solve rates for early stopping. Since Phase 2b is pre-killed, this access path is not designed here. If Gate 5 passes, add a `list_early_stopping_eligible()` API that returns Val-Long data read-only.** |
| Run-state database location and backup strategy | **OPEN — `run_state.db` location TBD (project root vs dedicated data directory). Backup strategy (periodic copies vs continuous replication) TBD. Loss of run state mid-campaign requires re-scanning trajectory files to reconstruct — expensive but recoverable.** |

---

*LLD-02 · Data Pool Manager · Signed Off v0.8 · April 2026*
