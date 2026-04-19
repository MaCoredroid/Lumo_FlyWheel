# Task Spec: `parallel-explorer-root-cause-map`

## Task Prompt

Investigate why the `release_readiness` CLI started over-reporting blocked owners after a scheduler refactor. You are allowed to use explorer-style subagents for read-heavy analysis, but you must return one unified root-cause brief and one bounded remediation plan. Do not land speculative code. The benchmark bundle includes conflicting runtime logs, stale docs, and several plausible suspects across aggregation, adapter normalization, and report rendering.

Your final answer in the task should identify the true root cause, explain why the other top suspects are not primary causes, cite exact files and functions, and propose the smallest safe patch plan.

## Workspace Bundle

- Medium Python repo with:
  - `src/release_readiness/core/aggregation.py`
  - `src/release_readiness/adapters/fs_source.py`
  - `src/release_readiness/adapters/env_source.py`
  - `src/release_readiness/renderers/markdown_renderer.py`
  - `tests/` with mostly passing suites and one failing investigative assertion
  - `docs/triage-policy.md`
  - `artifacts/logs/` with contradictory operator notes and runtime snapshots
  - `artifacts/review/incident_thread.md` summarizing a noisy handoff
- One visible failing command:
  - `pytest tests/test_root_cause_map.py -q`
- No patch is required for success; the output artifact is a report plus bounded plan.

## Seeded Orchestration Problem

The parent agent should split the read-heavy investigation into at least two non-overlapping explorer questions, for example:

- trace how owner totals are computed and where duplicates can enter
- trace how environment aliases and watchlist owners are normalized before aggregation
- inspect whether Markdown rendering is only exposing an upstream defect or introducing one

The seeded trap is that renderer-facing evidence is more salient than the true defect. Only a cross-check of normalization, aggregation, and rendered output can distinguish the primary cause from downstream symptoms. A shallow single-path read can produce a plausible but wrong answer.

## Required Surfaces

- `subagents`
- `repo_search`
- `shell`

Preferred:

- `review`
- `worktree`

## Deliverables

- Root-cause brief with exact file and symbol references
- Ranked suspect list covering at least three candidate causes
- Evidence table with at least four rows: claim, exact file plus symbol, exact test or artifact evidence, and why competing evidence is misleading
- One bounded remediation plan with explicit non-goals
- Verification note quoting the exact failing assertion and at least one contradictory artifact entry that was ruled out with explanation

## Grader Contract

The task scores well only if the submission:

- identifies the true causal path from normalization into aggregation
- explicitly rules out the renderer as a primary cause
- cites at least one adapter symbol, one aggregation symbol, one renderer symbol ruled out, and one test or artifact location, all with correct causal roles
- proposes a minimal patch plan that fixes source normalization rather than output formatting
- explains why at least one conflicting log, doc, or review artifact is stale, misleading, or downstream-only
- presents findings from at least two non-overlapping explorer questions and integrates both into the final brief
- avoids speculative code changes outside the named bounded remediation

Low-score behaviors include generic "investigate adapters" advice, grep-only blame assignment, repeating task-spec language without workspace-derived symbol evidence, or a plan that edits the renderer first.

## Red-Team Traps

- Blame `markdown_renderer.py` because visible logs mention doubled headings and blocked totals
- Treat stale `docs/triage-policy.md` language as source-of-truth over code paths
- Read only one suspect path and ignore conflicting evidence in adapter normalization
- Propose a broad refactor instead of one bounded fix
- Omit disproof for top false-positive suspects

## Quality Gate

- Probe target: naive GPT-5.4/high explorer fanout should not exceed 30/100 by summarizing logs and naming the most visible suspect.
- Hardening applied after adversarial probe:
  - removed the answer leak naming the true root cause directly
  - added an evidence-table requirement and contradictory-artifact disproof requirement
  - made orchestration credit depend on synthesizing at least two non-overlapping explorer questions
  - made symbol-level causal tracing mandatory before substantial credit
- Probe record:
  - probe model: child GPT-5.4/high adversarial review
  - initial verdict: over 30 too easy because the first draft leaked the answer and allowed spec-completable writeups
- Current difficulty judgment: under 30 likely for a naive GPT-5.4/high solver after hardening
