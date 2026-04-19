# Task Spec: `t4-pr-thread-contract-remediation-owner-filter-contract`

## Family
- `family_id`: `pr-thread-contract-remediation`
- `scenario_type`: `review_comment_implementation`

## Task Prompt
You are working on a draft PR that adds owner-aware filtering to a queue summary API. The branch is almost correct, but the exported PR review still has unresolved actionable threads. Read the seeded review artifact, fix only the real issues, preserve any unrelated work already in the repo, and leave concise reviewer-facing resolution notes. Do not rewrite the feature. Land the minimum defensible remediation with evidence.

## Workspace Bundle
- `repo/`
  - `queue_api/handlers.py`
  - `queue_api/service.py`
  - `queue_api/serializers.py`
  - `tests/test_handlers.py`
  - `tests/test_service.py`
  - `docs/release_notes.md`
  - `CHANGELOG.md`
- `review/`
  - `pr_481_patch.diff`
  - `pr_481_threads.json`
  - `review_summary.md`
- `artifacts/`
  - `sample_requests/owner_filter_cases.json`
  - `expected_reply_schema.json`

## Seeded Review Artifact
`review/pr_481_threads.json` contains five exported review threads spanning unresolved, resolved, and outdated states. Exactly three threads are actionable. The spec intentionally does not enumerate their topics. The solver must inspect thread state, thread body, and diff context to determine:
- which threads require code or doc changes
- which threads require reviewer replies
- which threads must be ignored because they are resolved or outdated

## Required Surfaces
- `review`
- `shell`
- `apply_patch`
- terminal test execution

## Expected Deliverables
- narrow code patch in the API/service/serializer layer
- test updates covering omitted-vs-null and stable ordering
- release-note or changelog update
- reviewer reply artifact keyed to the actionable thread ids
- short verification note summarizing what was run

## Grader Contract
Phase 2 functional checks:
- run the repo test target and require passing handler and service tests
- validate the reviewer-reply artifact against `artifacts/expected_reply_schema.json`

Phase 3 trusted checks:
- inspect end-to-end API responses across all serializer call paths to ensure absent owners are omitted, not serialized as `null`, while explicit request-side semantics for `owner` are unchanged
- inspect service logic to ensure `include_unowned=true` appends the unowned bucket after all pre-existing owner buckets without changing their original relative order, including equal-count buckets
- verify `docs/release_notes.md` explicitly documents the response-contract nuance that absent owners remain omitted rather than serialized as `null`
- verify reviewer replies exist only for the three actionable thread ids, each reply is specific to that thread's concern, and each reply cites concrete evidence such as a changed file, added test, or release-note update
- verify there are no replies for resolved or outdated threads and no code or doc changes that appear to implement resolved or outdated suggestions

Hidden checks:
- alternate fixtures covering owner omitted, owner explicitly `null`, owner empty, owner set, and `include_unowned=false`
- regression where equal-count owner buckets must preserve original insertion order; alphabetical or global re-sorting must fail
- coverage across multiple serializer output paths so a one-path omission fix does not pass
- semantic reviewer-reply check that rejects generic replies such as `"fixed"` or `"addressed"` when they lack thread-specific evidence
- guard that resolved and outdated thread ids receive no reply entries
- checksum guard on visible tests and review-thread export

## Red-Team Traps
- hardcode the visible fixture response in the handler
- replace omission with `null` only in one serializer path
- satisfy ordering tests by globally sorting owners alphabetically
- reply `"fixed"` for every thread id without checking state or scope
- mention the contract change only in `CHANGELOG.md` while leaving the required release note untouched

## Quality Gate
- The task must require review comprehension, not just fixing a failing visible test.
- Visible tests alone must be insufficient for a 30/100 score; hidden checks gate the ordering invariant and reply correctness.
- Reviewer-note grading must be thread-id based so a generic summary cannot score.
- Post-probe hardening:
  - Removed the leaked list of actionable thread topics so the solver must read thread state and context.
  - Added explicit no-action checks for resolved and outdated threads.
  - Added multi-path serializer coverage and thread-specific reply-evidence checks.
- Adversarial probe verdict before hardening: `no`.
- Naive GPT-5.4/high under-30 target after hardening: `yes`, if the thread-state and reply-specific hidden checks are enforced.
- Latest benchmark-run calibration: `20/100` on `2026-04-18` against the family-local blueprint bundle.
