# `candidate-worktree-shootout` Task Spec

**Track:** 11 — Subagents Orchestration
**Family id:** `candidate-worktree-shootout`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1` through `v5`)

## Canonical Task Prompt

Repair the report-filter normalization bug by evaluating two plausible fixes in
isolated worktree directories and then landing one coherent final patch in the
main workspace.

- **Candidate A** keeps normalization local to `src/report_filters/cli.py`.
- **Candidate B** moves normalization ownership to
  `src/report_filters/service.py`.

The visible regression sits on the CLI path, but the repo evidence shows at
least two non-CLI callers also depend on the shared service contract. A good
solution proves both candidates were evaluated in isolation, rejects the weaker
one with concrete evidence, and lands only one final ownership model.

Distinct isolated directories are sufficient for the benchmark's "worktree"
surface. The grader checks for concrete isolated paths plus per-candidate
commands and outcomes; it does not require the literal `git worktree` subcommand.

## Required Deliverables

The solver must create all four files under `artifacts/comparison/`:

- `candidate_a_eval.md`
- `candidate_b_eval.md`
- `comparison_summary.md`
- `verification_note.md`

Each candidate evaluation file must include:

- a concrete `Worktree Path`
- exact commands run
- touched files
- observed results
- evidence explaining why the candidate is strong or weak

`comparison_summary.md` must name the chosen strategy and the rejected strategy.
`verification_note.md` must record both isolated worktree paths and the
validation command used in each worktree and in the final workspace.

## Expected Final Edit Surface

The final integrated patch is intentionally narrow:

- `src/report_filters/service.py`
- `src/report_filters/cli.py`
- `tests/test_service.py`
- `docs/filter_contract.md`
- `artifacts/comparison/*`

Everything else is immutable for scoring purposes, especially:

- `src/report_filters/normalization.py`
- `tests/test_cli.py`
- `artifacts/candidates/*`
- `repo_evidence/*`
- `release_context/*`
- `incident_context/*`

## Required Verification Command

```bash
python -m pytest -q tests/test_cli.py tests/test_service.py
```

The visible contract is only `tests/test_cli.py`, but a high-scoring solution
must add or preserve a service-level regression in `tests/test_service.py` and
must make the direct service caller behavior correct.

## Workspace Bundle

Every variant ships the same repo-shaped workspace:

```text
.scenario_variant
AGENTS.md
Dockerfile
src/report_filters/__init__.py
src/report_filters/normalization.py
src/report_filters/service.py
src/report_filters/cli.py
tests/test_cli.py
tests/test_service.py
docs/filter_contract.md
artifacts/candidates/strategy_a_notes.md
artifacts/candidates/strategy_b_notes.md
artifacts/comparison/README.md
repo_evidence/caller_matrix.md
repo_evidence/contract_history.md
```

Variant-gated context is additive:

- **V2** adds `repo_evidence/stale/cli_hotfix_memo_2026_01.md`
- **V3** adds `repo_evidence/dirty_state/unfinished_cli_patch.md` and
  `artifacts/partial_work/cli_local_patch.diff`
- **V4** adds `release_context/importer_callers.md` and
  `release_context/release_gate.md`
- **V5** adds `incident_context/rollback_2026_07.md` and
  `incident_context/prior_selection.md`

## Variant Progression

### v1 — clean baseline

The caller matrix alone makes Candidate B correct: the scheduled importer and
saved-view repair job already call `service.compile_filters(...)` directly.

### v2 — noisy distractor

Adds a stale archived memo that still argues for the CLI-only hotfix. The memo
is true historically, but false for the current repo state.

### v3 — dirty state

Adds abandoned partial work for the CLI-local strategy. The right move is to
recognize sunk cost, not "finish what's already started."

### v4 — multi-corpus objective

Adds release context that makes the batch importer the active release blocker.
The solver must re-weight toward the direct-caller objective, not just the
visible CLI failure.

### v5 — recovery in thread

Adds incident context proving the last CLI-only hotfix was rolled back. A good
solution must not re-select the same strategy without incident-aware reasoning.

## Quality Targets

Deterministic baselines implemented in this repo:

- oracle overlay: `100 / 100` on every variant
- empty workspace: `0 / 100` on every variant
- CLI-local shortcut: `25 / 100` on every variant

The family has Layer B artifacts implemented locally:

- dual-band scorer at
  `verifiers/candidate-worktree-shootout/score_shootout.py`
- milestone scripts under
  `verifier_data/candidate-worktree-shootout/{variant}/milestones/`
- verification matrices at
  [`verification_matrix.md`](./verification_matrix.md) and
  [`verification_matrix_v5.md`](./verification_matrix_v5.md)

Layer A remains intentionally open: the next step is a live `codex exec` probe
loop against the family. That loop was not launched in this handoff.

## Saturation and Renewal Plan

This family saturates when mean `P_benchmark > 80` for two consecutive probe
rounds. Renewal queue:

1. `v6_midrun_owner_flip` — a new direct caller appears after the first
   candidate evaluation.
2. `v7_conflicting_hotfix_audit` — rollout notes conflict on whether a local
   hotfix actually shipped.
3. `retire_v1` — retire the clean baseline once it becomes a pure floor check.
