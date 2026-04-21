# LLD-05 · Evaluator (Dual-Track)

> Codex-Bench · Low-Level Design
> Derived from HLD Spec v2.3 · April 2026
> Sprint: Design S1 → Implement S1
> Status: DRAFT v0.3 — surface-contract tightening after second red-team pass

---

## Changelog

| Version | Change |
|---|---|
| v0.3 | Surface-contract tightening. **P0-1:** SWE-bench `predictions.jsonl.model_name_or_path` now identifies the producing model / harness (`<model_id>::<harness>`), not the evaluator CLI. **P0-2:** Codex-Long contract-mismatch detection is now symmetric: both `resolved + cl_pass=false` and `failed + cl_pass=true` are flagged, with `cl_pass` explicitly treated as the deeper grading source for `passed`. **P0-3:** Codex-Long audit-path convention is now pinned to LLD-03's underscore layout (`grading/<family_id>_<variant_id>/results/verify_result.json`) and the local normalized-artifact layout matches. **P1-1:** SWE-bench §4.3 broadened from `no_patch` to all non-CLI terminal outcomes, distinguishing pre-CLI crashes / timeouts from evaluator exit-code-2 infra failures. **P1-2:** aggregation now explicitly reads through LLD-02's `latest_runs` view. **P1-3:** the stale LLD-03 prose about a run-record `verify_result.json` path is now a sign-off gate and is patched in LLD-03 §5B.4. **P1-4:** `failure_mode` enum aligned across sections; `milestone_credit_total` marked expected-null until weight snapshots exist; post-drain invalidations regenerate versioned summaries rather than overwrite. |
| v0.2 | Red-team alignment pass. **P0-1:** Codex-Long source of truth changed from a supposed run-record `verify_result.json` path to the fields LLD-02 actually stores: `outcome`, `cl_pass`, `milestone_json`, `grading_manifest_ver`, plus run identity metadata. `verify_result.json` is now an audit artifact, not the grading source. **P0-2:** Codex-Long validation rules now distinguish normal `crash` / `timeout` runs from true integrity gaps; missing verifier output is only fatal when a current run claims a graded terminal state but lacks `cl_pass` / milestone data. **P0-3:** SWE-bench `no_patch` is now explicitly a non-CLI path synthesized from the run record; campaign denominators no longer assume every finished SWE-bench run has evaluator artifacts. **P0-4:** Attempt-scoped artifact layout added for SWE-bench to avoid retry / regrade clobbering. **P1-1:** `milestone_credit_total` downgraded from required lazy recomputation to optional normalized metadata; raw milestone booleans are authoritative until a versioned weight-snapshot scheme is pinned. **P1-2:** milestone summaries keyed by `(family_id, milestone_id)` rather than bare milestone ID. **P1-3:** ARM64 SWE-bench fallback and concurrency / Docker-daemon limits are now explicit sign-off gates. **P1-4:** bootstrap ownership closed: LLD-05 emits bootstrap-ready records and point estimates; LLD-12 computes publication confidence intervals. |
| v0.1 | Initial draft for the v2.3 dual-track evaluator. Aligns to signed-off LLD-02, LLD-04, and LLD-13, and adopts the same-sprint SWE-bench invocation contract defined in LLD-03 §5B.4. Clarifies that LLD-03 owns mechanical execution and terminal run-state writes, while LLD-05 owns verdict normalization, SWE-bench harness integration, Codex-Long `verify_result.json` ingestion, and campaign-level solve / milestone aggregation. |

## 1. Purpose & Scope

This document specifies the Evaluator — the component that normalizes per-task execution outputs into benchmark verdicts and campaign-level solve summaries across both Codex-Bench tracks.

**Responsibilities:**

- Own the SWE-bench evaluation entry point `codex-bench-eval-swe`
- Convert LLD-03 patch artifacts into upstream-compatible SWE-bench `predictions.jsonl`
- Invoke the official SWE-bench harness, parse the verdict, and return the exit-code contract LLD-03 depends on
- Normalize Codex-Long grading results from the **current LLD-02 run record** (`cl_pass`, `milestone_json`, `grading_manifest_ver`) without re-executing verifier logic
- Treat `verify_result.json` as an audit artifact for Codex-Long, not as the primary grading source
- Produce per-task normalized records, campaign-level point-estimate summaries, family-level rollups, and milestone summaries for LLD-12
- Fail closed on missing or malformed evaluation metadata where the upstream LLDs guarantee those fields should exist

**Out of scope:** task execution and container lifecycle (LLD-03), pool / split ownership and dispatch state transitions (LLD-02 / LLD-07), verifier authoring and injection semantics (LLD-13), latency telemetry capture (LLD-04), and final publication artifact assembly (LLD-12).

**Ownership boundary:** LLD-03 remains the primary writer of terminal run state into LLD-02. LLD-05 determines SWE-bench verdicts via the CLI contract and performs read-only normalization / aggregation on top of the run records that LLD-03 finalized. LLD-05 does not own the dispatch state machine.

---

## 2. Architecture Overview

```text
                    LLD-03 TASK OUTPUTS
          ┌──────────────────────┬──────────────────────────┐
          │ SWE-bench            │ Codex-Long               │
          │ patch file           │ finish_run(...)          │
          │ instance_id          │ -> outcome               │
          │ no_patch possible    │ -> cl_pass               │
          │                      │ -> milestone_json        │
          │                      │ -> grading_manifest_ver  │
          └────────────┬─────────┴──────────────┬───────────┘
                       │                        │
                       ▼                        ▼
          ┌────────────────────┐   ┌────────────────────────┐
          │ codex-bench-       │   │ CodexLongNormalizer    │
          │ eval-swe           │   │ - read latest run row  │
          │ - patch→pred       │   │ - validate cl_pass     │
          │ - harness run      │   │ - preserve milestones  │
          │ - exit code        │   │ - optional audit read  │
          └────────────┬───────┘   └────────────┬───────────┘
                       │                        │
                       └────────────┬───────────┘
                                    ▼
                             Normalized EvalRecord
                                    ▼
                      Campaign / family / milestone summaries
                                    ▼
                                  LLD-12
```

The design split is deliberate:

- **SWE-bench path:** LLD-05 performs benchmark grading mechanics.
- **Codex-Long path:** LLD-05 does **not** perform grading mechanics; it consumes the authoritative grading result already written into LLD-02 by LLD-03 from LLD-13's `verify.sh`.

---

## 3. Source Of Truth & Shared Record

### 3.1 Canonical Source Of Truth

For aggregation, the source of truth is the **current LLD-02 run record**.

- For SWE-bench: `outcome`, `trajectory_path`, task identity, attempt identity, harness, model, pool.
- For Codex-Long: `outcome`, `cl_pass`, `milestone_json`, `grading_manifest_ver`, `snapshot_image_ref`, and identity metadata.

`verify_result.json` remains useful for auditability and debugging, but it is not the authoritative grading surface once LLD-03 has already denormalized its contents into LLD-02.

### 3.2 EvalRecord

Both tracks normalize into one record shape so LLD-12 can consume a single interface.

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class EvalRecord:
    # Identity from LLD-02 latest/current run row
    track: str                 # "swe_bench" | "codex_long"
    pool_or_split: str         # dev_bench / bench_control / final_test / train_long / ...
    scenario_id: str           # SWE instance_id or "family_id/variant_id"
    model_id: str
    harness: str               # "codex" | "swe_agent"
    seed: int
    attempt: int

    # Headline benchmark result
    outcome: str               # resolved / failed / no_patch / timeout / crash
    passed: bool               # True iff the headline benchmark verdict is success

    # Optional Codex-Long metadata
    family_id: Optional[str] = None
    variant_id: Optional[str] = None
    scenario_type: Optional[str] = None
    grading_manifest_ver: Optional[int] = None

    # Secondary diagnostics
    failure_mode: Optional[str] = None   # one of: tests_passed / tests_failed / patch_apply_failed / no_patch / infra_error / agent_timeout / agent_crash
    milestone_results: Optional[dict[str, bool]] = None
    milestone_credit_total: Optional[float] = None   # expected NULL in v0.3 until milestone weights are version-snapshotted
    errors: Optional[list[str]] = None   # free-form diagnostic strings, not a stable taxonomy

    # Artifact references
    source_artifact_path: str = ""       # per-track semantics documented in §3.3
    normalized_artifact_path: str = ""   # path to this record serialized as JSON
    eval_wall_clock_seconds: Optional[float] = None
```

### 3.3 Path Semantics

- **SWE-bench:** `source_artifact_path` is the patch path consumed by `codex-bench-eval-swe`.
- **Codex-Long:** `source_artifact_path` is the reconstructed audit path to `verify_result.json` if present; normalization does **not** require the file to exist when the run record already carries the authoritative fields.
- `normalized_artifact_path` is always the on-disk JSON serialization of the `EvalRecord`.

### 3.4 Normalization Rules

- `passed = True` iff the task's benchmark-specific success condition is satisfied.
- `outcome = "resolved"` implies `passed = True`.
- `no_patch` is a valid SWE-bench terminal outcome that bypasses the CLI path.
- Aggregation consumes only **current** attempts (`is_current = true`), so retries and regrades do not double-count.

---

## 4. SWE-bench Path

### 4.1 Authoritative Invocation Contract

LLD-05 adopts the same-sprint bilateral contract defined in LLD-03 §5B.4 without amendment.

```bash
codex-bench-eval-swe \
  --instance-id <instance_id> \
  --patch-path <path_to_patch_file> \
  --output-dir <path_for_eval_artifacts> \
  --dataset-name princeton-nlp/SWE-bench_Verified
```

| Argument | Type | Description |
|---|---|---|
| `--instance-id` | string | SWE-bench instance ID such as `django__django-11099` |
| `--patch-path` | path | Patch extracted by LLD-03 from the final workspace state |
| `--output-dir` | path | Attempt-scoped directory where evaluator artifacts are written |
| `--dataset-name` | string | Always `princeton-nlp/SWE-bench_Verified` |

### 4.2 Exit-Code Contract

| Exit Code | Meaning | LLD-03 maps to |
|---|---|---|
| `0` | Evaluation completed and the patch resolves the task | `outcome = "resolved"` |
| `1` | Evaluation completed and the patch does not resolve the task | `outcome = "failed"` |
| `2` | Evaluation infrastructure failure | `outcome = "crash"` |

No additional exit codes are introduced. Missing / unreadable patch files, unexpected exceptions, or harness-level infrastructure crashes must all collapse to exit code `2` after writing a diagnostic log.

### 4.3 Non-CLI Terminal Outcomes

Some SWE-bench terminal outcomes are decided **before** `codex-bench-eval-swe` is invoked and therefore produce no evaluator artifacts.

Implications:

- `no_patch` is decided by LLD-03 when the agent produced no patch at all.
- Pre-CLI `timeout` / `crash` paths are also possible if LLD-03 fails before invoking the evaluator subprocess.
- For these pre-CLI paths, no `predictions.jsonl`, `eval_report.json`, or `eval.log` exists.
- LLD-05 synthesizes the `EvalRecord` directly from the current run record for denominator accounting.
- `failure_mode` is `no_patch`, `agent_timeout`, or `agent_crash` respectively.

This is distinct from an evaluator subprocess that **does** run and returns exit code `2`; that path writes `eval.log` / `eval_report.json` and normalizes to `failure_mode = "infra_error"`.

### 4.4 CLI Responsibilities

The `codex-bench-eval-swe` entry point must:

1. Validate that the patch file exists and is readable. Missing / unreadable patch -> `eval.log` + exit `2`.
2. Convert the patch into the upstream SWE-bench `predictions.jsonl` schema.
3. Invoke `python -m swebench.harness.run_evaluation` using the pinned `swebench` package version selected during Sprint 1 setup.
4. Distinguish the authoritative SWE-bench evaluator failure-mode enum in `eval_report.json`: `tests_passed`, `tests_failed`, `patch_apply_failed`, `infra_error`.
5. Write stable attempt-scoped artifacts.
6. Return exit code `0`, `1`, or `2` exactly.

### 4.5 SWE-bench Artifacts

Attempt-scoped layout is required to prevent retries or regrades from clobbering prior artifacts.

| Artifact | Format | Purpose |
|---|---|---|
| `{output_dir}/predictions.jsonl` | JSONL | Upstream-compatible input / reproducibility artifact |
| `{output_dir}/eval_report.json` | JSON | Structured per-attempt evaluator summary |
| `{output_dir}/eval.log` | text | Raw harness stdout / stderr |
| `{output_dir}/normalized_eval.json` | JSON | LLD-05 common-schema record |

Pinned `predictions.jsonl` target schema:

```json
{
  "instance_id": "django__django-11099",
  "model_patch": "<unified diff text>",
  "model_name_or_path": "qwen3.5-27b::codex"
}
```

Suggested `eval_report.json` shape:

```json
{
  "track": "swe_bench",
  "pool_or_split": "dev_bench",
  "instance_id": "django__django-11099",
  "model_id": "qwen3.5-27b",
  "harness": "codex",
  "seed": 1,
  "attempt": 1,
  "patch_path": "/runs/.../final.patch",
  "prediction_path": "/runs/.../predictions.jsonl",
  "dataset_name": "princeton-nlp/SWE-bench_Verified",
  "verdict": "failed",
  "passed": false,
  "failure_mode": "patch_apply_failed",
  "harness_exit_code": 1,
  "eval_wall_clock_seconds": 187.4,
  "error": null
}
```

### 4.6 Version Pinning, ARM64 Gate, And Fallback

The exact `swebench` version must be pinned in the evaluator implementation's package lockfile during Sprint 1 setup. A version bump after that point is a tracked project change, not an implicit dependency refresh.

Sprint 1 sign-off must validate that the chosen SWE-bench evaluation path runs correctly on DGX Spark ARM64. If native ARM64 execution fails, the project must explicitly choose and document one fallback before sign-off:

1. Rebuild required evaluation images natively for ARM64.
2. Run x86 images under emulation.
3. Offload SWE-bench evaluation to a separate compatible host.

The fallback is not cosmetic. It changes wall-clock planning for Contribution A and B2 and must be reflected in the sprint budget before sign-off.

### 4.7 Concurrency Contract

`codex-bench-eval-swe` must be safe for concurrent invocation **only when each call has a distinct attempt-scoped output directory**.

Because each invocation spawns additional Docker work, LLD-07 must apply a host-local semaphore for evaluator subprocesses. Default policy before validation: **one concurrent SWE-bench evaluator per host**. Any higher concurrency limit requires Sprint 1 validation against Docker-daemon contention, cache reuse, and cleanup behavior.

This concurrency cap must feed back into LLD-07 / HLD wall-clock planning for Sprint 3. Serialized or near-serialized evaluation is acceptable, but it is not free and must be budgeted honestly.

---

## 5. Codex-Long Path

### 5.1 Authority Boundary

LLD-05 does **not** execute verifiers or milestone scripts for Codex-Long. Per signed-off LLD-13, `verify.sh` is the sole execution authority and writes one authoritative `verify_result.json`. LLD-03 already parses that file and writes the grading result into LLD-02 via `finish_run()`.

### 5.2 Authoritative Input Contract

For each current Codex-Long run, LLD-05 consumes the current LLD-02 run record:

- `track`, `pool_or_split`, `scenario_id`, `family_id`, `variant_id`, `scenario_type`
- `model_id`, `harness`, `seed`, `attempt`
- `outcome`
- `cl_pass`
- `milestone_json`
- `grading_manifest_ver`
- `trajectory_path`
- `snapshot_image_ref`

This is the authoritative grading source. `verify_result.json` is audit-only.

### 5.3 Audit Artifact Derivation

LLD-03's grading workspace layout defines where `verify_result.json` lives on disk. The pinned audit-path convention is:

`/grading/<family_id>_<variant_id>/results/verify_result.json`

LLD-05 may reconstruct that path from the same directory convention for debugging or artifact packaging, but verdict normalization must not depend on the file's presence once the run record already stores `cl_pass` and `milestone_json`.

### 5.4 Validation Rules

LLD-05 must fail closed on the Codex-Long path, but only when the upstream contracts imply grading actually completed.

- Current run with `outcome in {"resolved", "failed"}` and missing `cl_pass` -> integrity error.
- Current run with malformed `milestone_json` when `outcome in {"resolved", "failed"}` -> integrity error.
- Current run with `outcome = "resolved"` and `cl_pass = false` -> contract mismatch.
- Current run with `outcome = "failed"` and `cl_pass = true` -> contract mismatch.
- Current run with `outcome in {"crash", "timeout"}` and no milestone data -> expected; emit `passed = false` with no milestone fields.
- Superseded attempts are ignored completely.
- Regraded runs are consumed only through the latest current attempt after manifest invalidation.

When a contract mismatch is surfaced, `passed` is derived from `cl_pass`, not from `outcome`. This matches the upstream causality in LLD-03, where `outcome` is computed from the parsed verifier result.

### 5.5 Milestone Policy

Binary solve rate is always driven by `cl_pass`.

Milestone analysis is secondary and diagnostic:

- Preserve raw milestone booleans exactly as emitted by `verify.sh` and stored in `milestone_json`.
- Treat `milestone_credit_total` as **optional normalized metadata**, not as an authoritative recomputation requirement.
- Until the project pins a versioned storage scheme for milestone weights, LLD-05 must not lazily recompute historical credit totals from mutable family YAML.

This keeps B1 / B2 headline claims binary while avoiding a false promise that historic credit totals can be reconstructed from data the signed-off LLDs do not currently preserve.

### 5.6 Codex-Long Normalized Artifact

Example:

```json
{
  "track": "codex_long",
  "pool_or_split": "train_long",
  "scenario_id": "dependency-migration-npm/lodash-3-to-4",
  "family_id": "dependency-migration-npm",
  "variant_id": "lodash-3-to-4",
  "scenario_type": "migration_refactor",
  "model_id": "qwen3.5-27b",
  "harness": "codex",
  "seed": 1,
  "attempt": 1,
  "outcome": "resolved",
  "passed": true,
  "milestone_results": {
    "m1_dep_updated": true,
    "m2_imports_fixed": true,
    "m3_tests_passing": true
  },
  "milestone_credit_total": null,
  "errors": [],
  "grading_manifest_ver": 4,
  "source_artifact_path": "/grading/.../results/verify_result.json"
}
```

---

## 6. Campaign Aggregation

LLD-05 owns evaluation-layer normalization and point-estimate summaries. LLD-12 owns final publication packaging and confidence-interval computation.

### 6.1 Aggregation Inputs

- Current run records only (`is_current = true`)
- Queried through LLD-02's `latest_runs` view, not the raw `runs` table
- Terminal attempts only (`exec_state = "finished"`)
- Explicit harness filter (`codex` vs `swe_agent`)
- Per-attempt normalized artifacts when they exist
- Run-record-only synthesized records for `no_patch`, `timeout`, and `crash` paths that never produced evaluator artifacts

### 6.2 Summary Outputs

| Output | Grain | Primary use |
|---|---|---|
| `campaign_eval_records.jsonl` | per current task attempt | Audit trail and LLD-12 input |
| `campaign_solve_summary.json` | model × pool/split × harness | Headline point-estimate solve summary |
| `family_eval_summary.json` | family × model × harness | Gate 4 coverage and B1 family-level analysis |
| `milestone_summary.json` | family × milestone × model × split | Diagnostic / RL-signal analysis |

Suggested `campaign_solve_summary.json` fields:

- `track`
- `pool_or_split`
- `model_id`
- `harness`
- `n_current_runs`
- `n_passed`
- `solve_rate`
- `n_families` for Codex-Long campaigns

### 6.3 Bootstrap Responsibility

LLD-05 emits bootstrap-ready per-task records, including family identity for Codex-Long. Authoritative publication confidence intervals are computed by LLD-12, not by LLD-05.

This avoids duplicate CI logic and lets one downstream layer own the reporting method while LLD-05 remains the normalization layer.

### 6.4 Denominator Policy

The denominator is the set of **current finished runs** selected for the campaign view. `resolved` is the only success numerator. `failed`, `timeout`, `crash`, and `no_patch` remain non-success terminal outcomes unless later invalidated and superseded.

### 6.5 Campaign-Close Trigger

LLD-07 invokes the campaign aggregator after the campaign's dispatch queue drains and all in-flight attempts finish. Interim summaries are allowed for operator visibility, but only the post-drain aggregation is reportable.

If a later `invalidate_stale_runs()` call supersedes previously reportable runs, LLD-05 emits a **new** summary artifact under a versioned suffix rather than overwriting the old summary in place.

---

## 7. Artifact Layout

```text
artifacts/
  swe_eval/
    <instance_id>/
      seed<seed>-attempt<attempt>/
        predictions.jsonl
        eval_report.json
        eval.log
        normalized_eval.json

  codex_long_eval/
    <family_id>_<variant_id>/
      seed<seed>-attempt<attempt>/
        normalized_eval.json

  campaign_eval/
    <campaign_id>/
      campaign_eval_records.jsonl
      campaign_solve_summary.json
      family_eval_summary.json
      milestone_summary.json
```

The campaign ID naming convention must be deterministic from `(track, pool_or_split, model_id, harness)` so LLD-12 can locate outputs without bespoke lookup logic.

---

## 8. Sprint 1 Validation Checklist

### SWE-bench Path

- [ ] `codex-bench-eval-swe` implements the exact LLD-03 §5B.4 argument contract
- [ ] Exit codes `0 / 1 / 2` map correctly to `resolved / failed / crash`
- [ ] `no_patch` runs are synthesized from run records without requiring evaluator artifacts
- [ ] Generated `predictions.jsonl` matches the pinned upstream SWE-bench schema
- [ ] `swebench.harness.run_evaluation` executes correctly on DGX Spark ARM64, or the fallback path is explicitly chosen and budgeted
- [ ] `eval_report.json` distinguishes at least `tests_failed` vs `patch_apply_failed`
- [ ] Retries write to distinct attempt-scoped output dirs and do not clobber prior artifacts
- [ ] `model_name_or_path` identifies the producing model / harness, not the evaluator CLI

### Codex-Long Path

- [ ] Current run selection excludes superseded retries and regrades
- [ ] `resolved` / `failed` runs without `cl_pass` fail closed
- [ ] `crash` / `timeout` runs normalize without false integrity violations
- [ ] both `resolved + cl_pass = false` and `failed + cl_pass = true` are detected and surfaced
- [ ] Milestone summaries key on `(family_id, milestone_id)`, not bare milestone ID

### Cross-track

- [ ] Both tracks normalize into the shared `EvalRecord` schema
- [ ] Aggregation reads current attempts through LLD-02's `latest_runs` view
- [ ] Campaign summaries are reproducible from current run rows plus normalized artifacts
- [ ] LLD-12 can consume summary outputs without parsing raw SWE-bench logs or raw verifier files
- [ ] Structured evaluator logs are sufficient for operators to identify failing tasks and dominant failure modes during long campaigns
- [ ] Concurrency cap for SWE-bench evaluator subprocesses is validated and documented

---

## 9. Connections To Other LLDs

| LLD | Relationship |
|---|---|
| **LLD-02** Data Pool Manager | Primary read source for aggregation. LLD-05 consumes current run rows, especially `outcome`, `cl_pass`, `milestone_json`, and identity metadata. |
| **LLD-03** Task Orchestrator | Primary upstream dependency. Drives `codex-bench-eval-swe` on SWE-bench runs and writes Codex-Long grading results into LLD-02 after reading `verify_result.json`. |
| **LLD-04** Latency Telemetry Capture | Parallel downstream input to LLD-12. Campaign keys must align, but there is no direct control path. |
| **LLD-07** Benchmark Runner | Owns campaign boundaries and invokes aggregation after the queue drains. Also owns any concurrency cap applied to evaluator subprocesses. |
| **LLD-12** Results & Artifact Generator | Primary downstream consumer of normalized evaluator artifacts and campaign summaries. Owns publication confidence intervals and final report assembly. |
| **LLD-13** Codex-Long Scenario Framework | Authoritative source for verifier semantics and milestone meaning. LLD-05 consumes results derived from `verify.sh` but does not execute verifier logic. |

---

## 10. Known Issues & Risks

| Issue | Severity | Mitigation |
|---|---|---|
| SWE-bench ARM64 evaluation may fail on DGX Spark | HIGH | Native ARM64 validation is a Sprint 1 gate. If it fails, choose and re-budget a fallback explicitly before sign-off. |
| Docker-daemon contention from concurrent SWE-bench evaluators | MEDIUM | Default to one concurrent evaluator per host until higher concurrency is validated. |
| Historic milestone credit totals cannot be recomputed from current signed-off artifacts | MEDIUM | Treat raw milestone booleans as authoritative. Do not require lazy historical credit recomputation until weights are version-snapshotted. |
| Free-form `errors` strings are not a stable taxonomy | LOW | Use `failure_mode` for the coarse reportable taxonomy; keep `errors` diagnostic-only. |

---

## 11. Sign-off Conditions

This draft should not be considered signed off until all of the following are explicit in implementation:

- LLD-05 adopts the LLD-03 §5B.4 SWE-bench contract without divergence.
- Codex-Long normalization uses the fields LLD-02 actually stores, not a nonexistent `verify_result` path column.
- `no_patch`, `timeout`, and `crash` are handled as first-class denominator paths.
- ARM64 SWE-bench evaluation is validated end-to-end on DGX Spark or an explicit fallback path is chosen and re-budgeted.
- Attempt-scoped artifact layout is implemented for both tracks.
- LLD-03 §5B.4 prose is amended so it no longer claims LLD-02 stores a `verify_result.json` path.
- LLD-12 consumption boundaries are clean: LLD-05 owns normalization and point-estimate summaries; LLD-12 owns publication CI computation and final report assembly.

---

## 12. Immediate Follow-up

- Pin the exact `swebench` package version and lockfile location when the evaluator implementation skeleton is created.
- Decide whether milestone weights will be version-snapshotted in a future manifest amendment or whether `milestone_credit_total` remains optional indefinitely.
