# Task Spec: `parallel-explorer-root-cause-map`

## Task Prompt

Investigate why the `release_readiness` CLI started over-reporting blocked owners after a scheduler refactor. The workspace contains a plausible renderer symptom, stale operator guidance, and at least one tempting in-progress hotfix that is not the real fix. You may use explorer-style subagents for read-heavy analysis, but the final deliverable is one unified root-cause brief plus one bounded remediation plan. Do not land speculative code.

Your finished brief must identify the true causal path from owner normalization into aggregation, explain why the renderer is only a downstream symptom, quote the exact failing assertion, and name the smallest safe patch target. Strong answers show at least two non-overlapping investigation threads and explicitly disprove at least one attractive false lead.

## Structured Output Contract

The family uses a CLI-shaped deliverable, even though the task is read-only.

- Author `brief_input.json` at the workspace root.
- Run `./bin/cnb55-brief validate brief_input.json`.
- Run `./bin/cnb55-brief submit brief_input.json`.
- The CLI writes canonical outputs to `brief/manager_brief.json` and `brief/manager_brief.md`.

Schema version: `cnb55.root_cause_brief.v1`

Required top-level fields:

- `schema_version`
- `variant_id`
- `accepted`
- `ranking`
- `investigation_threads`
- `evidence_table`
- `remediation_plan`
- `verification_note`

## Suspect Taxonomy

The solver must rank these seeded suspects:

- `S1-fs-alias-normalization`
- `S2-aggregation-grouping`
- `S3-renderer-duplicate-headings`
- `S4-env-watchlist-parser`

The point of the task is not naming an arbitrary component. The point is proving which suspect is primary, which one only amplifies the defect, and which one is merely downstream-visible.

## Workspace Bundle

Each variant ships the same top-level structure:

- `AGENTS.md`
- `Dockerfile`
- `bin/cnb55-brief`
- `brief/`
- `src/release_readiness/`
- `tests/test_root_cause_map.py`
- `docs/triage-policy.md`
- `artifacts/logs/`
- `artifacts/review/incident_thread.md`

Variant-specific optional corpora:

- `release_context/` for V4+
- `incident_context/` for V5
- `artifacts/wip/` for V3

## Deliverables

- Ranked suspect list with all four suspects, contiguous ranks, and exact file + symbol references.
- At least two non-overlapping investigation threads.
- Evidence table with at least four rows.
- One bounded remediation plan naming the intended patch target and explicit non-goals.
- One verification note quoting the exact failing assertion and one contradictory artifact that was ruled out.

## Variant Progression

### V1 Clean Baseline

- Minimal false leads.
- True cause is discoverable by reading adapter normalization plus aggregation.

### V2 Noisy Distractor

- Adds stale operator notes and doc language that over-emphasize renderer output.
- Strong answer must disprove stale renderer-centric guidance with workspace evidence.

### V3 Dirty State

- Adds an abandoned renderer hotfix patch under `artifacts/wip/`.
- Strong answer must avoid finishing sunk-cost work or treating that patch as the fix.

### V4 Multi-Corpus Objective

- Adds `release_context/` clarifying the hotfix window and downstream contract risk.
- Strong answer still identifies the same root cause, but the remediation plan must stay adapter-local rather than proposing a broad aggregation rewrite.

### V5 Recovery In Thread

- Adds `incident_context/` documenting a prior renderer-only hotfix that was rolled back.
- Strong answer must acknowledge the rollback and explicitly avoid re-selecting the renderer path.

## Required Surfaces

- `subagents`
- `repo_search`
- `shell`

Preferred:

- `review`
- `worktree`

## Quality Gate

Naive solver target:

- leaderboard probe should struggle to stay above `30 / 100` without exact symbol tracing, contradictory-artifact disproof, and a bounded remediation target.

Current hardening levers:

- symbol-level scoring instead of prose-only scoring
- explicit penalty for renderer-first fixes
- explicit penalty for missing contradictory-artifact disproof
- explicit penalty for single-path investigations
- variant-gated penalties for sunk-cost patch completion, aggregation-overreach, and incident-blind rerender proposals

## Saturation And Renewal Plan

Trigger:

- if the moving mean `P_benchmark` exceeds `80` for two consecutive probe rounds, mark the family `saturation_renewal_due`.

Planned renewal queue:

- add a V6 variant where scheduler aliases change mid-thread after the first investigation pass
- add a V7 variant where two contradictory operator snapshots must be reconciled before the root cause can be trusted
- retire the easiest floor variant if V1 becomes pure formality
