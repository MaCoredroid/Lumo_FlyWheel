# Task Spec: `delegation-merge-salvage`

## Task Prompt

Finish a partially delegated feature after two worker patches landed in a scratch branch and neither is safe to apply wholesale. Worker A fixed Markdown rendering and CLI flag plumbing but also broke the JSON summary contract. Worker B preserved JSON behavior and updated docs but edited the wrong snapshot fixture and missed the new `--include-watchlist` follow-up path.

Salvage the correct pieces, reject the bad ones, finish the task, and leave a concise postmortem explaining what was kept, what was discarded, and why.

## Workspace Bundle

- Python CLI repo with:
  - `src/watchlist_report/cli.py`
  - `src/watchlist_report/service.py`
  - `src/watchlist_report/renderers/json_renderer.py`
  - `src/watchlist_report/renderers/markdown_renderer.py`
  - `tests/test_cli.py`
  - `tests/test_service.py`
  - `tests/test_docs.py`
- `artifacts/delegation/worker_a.patch`
- `artifacts/delegation/worker_b.patch`
- `artifacts/delegation/worker_a_notes.md`
- `artifacts/delegation/worker_b_notes.md`
- `docs/usage.md`
- Visible checks:
  - Markdown mode must work
- Hidden checks:
  - JSON output contract must remain byte-for-byte unchanged on baseline fixtures
  - `--include-watchlist` follow-up path must stay visible when requested
  - unrelated snapshot fixtures must remain untouched

## Seeded Orchestration Or Merge-Salvage Problem

The parent agent must reason over overlapping diffs rather than restarting from scratch or blindly choosing one worker. The seeded trap is that Worker A appears stronger because it passes the visible Markdown assertions, but hidden grading also checks JSON shape and watchlist follow-up behavior. Worker B has pieces worth salvaging, especially around preserving JSON invariants and docs, but cannot be applied directly either.

## Required Surfaces

- `subagents`
- `shell`
- `apply_patch`

Preferred:

- `worktree`
- `review`

## Deliverables

- Final integrated patch
- `artifacts/delegation/salvage_postmortem.md` with `Kept from Worker A`, `Rejected from Worker A`, `Kept from Worker B`, and `Rejected from Worker B`, each citing exact patch hunks or file paths
- `artifacts/delegation/verification.md` listing exact commands or evidence for Markdown, JSON, and `--include-watchlist`
- Reviewer-facing note describing the resolved regression risks

## Grader Contract

The task scores well only if the submission:

- preserves the existing JSON contract
- delivers a working Markdown path and the requested watchlist follow-up behavior
- preserves byte-for-byte JSON output for existing contract fixtures; any JSON shape or key drift caps score at 20
- passes hidden `--include-watchlist` behavior checks and does not modify the unrelated snapshot fixture; fixture drift caps score at 20
- references at least one kept hunk and one rejected hunk from each worker artifact; generic salvage prose with no hunk or file evidence caps score at 30
- keeps the final patch reviewable rather than replacing large files without evidence; applying either worker patch wholesale or rewriting contested files without hunk-level justification caps score at 30

Low-score behavior includes choosing a single worker patch wholesale, rewriting the task from scratch without engaging the salvage artifacts, or keeping Markdown green while regressing JSON.

## Red-Team Traps

- Apply Worker A wholesale because visible Markdown checks turn green
- Apply Worker B wholesale because docs and JSON look safer
- Rebuild all touched files from scratch and ignore the delegation artifacts
- Revert contested files wholesale, discarding useful worker logic
- Fix visible tests only and miss the watchlist follow-up path
- Claim salvage from both workers without artifact-specific evidence

## Quality Gate

- Probe target: naive GPT-5.4/high should not exceed 30/100 by picking the patch with the best visible-test story.
- Hardening applied after adversarial probe:
  - split visible and hidden checks so Markdown is not confused with JSON and watchlist correctness
  - required hunk-cited salvage artifacts and explicit verification evidence
  - added score caps for JSON drift, untouched-watchlist regressions, wholesale patch selection, and generic postmortems
  - made artifact-ignoring rewrites and cosmetic salvage notes explicitly low-value
- Probe record:
  - probe model: child GPT-5.4/high adversarial review
  - initial verdict: over 30 too easy because a solver could pick one worker, preserve Markdown, and write plausible salvage prose
- Current difficulty judgment: under 30 likely for a naive GPT-5.4/high solver after hardening
