# `pr-thread-contract-remediation` Task Spec

**Track:** 04 — Review Remediation
**Family id:** `pr-thread-contract-remediation`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Canonical Prompt

You are fixing a draft PR that adds owner-aware filtering to a queue summary
API. The branch already contains unrelated valid work; preserve it. The seeded
review export includes unresolved, resolved, and outdated threads. Only the
real unresolved contract issues should drive the patch.

Read the review export, inspect the code and tests, land the minimum defensible
remediation, update the required release note, and leave concise reviewer
replies tied only to the actionable thread ids. Do not rewrite the feature and
do not "close everything out" with generic replies.

## Workspace Bundle

Every variant ships the same layout:

```text
queue_api/
  handlers.py
  serializers.py
  service.py
tests/
  test_handlers.py
  test_service.py
  locked/reply_contract.txt
docs/
  release_notes.md
CHANGELOG.md
review/
  pr_481_threads.json
  pr_481_patch.diff
  review_summary.md
artifacts/
  expected_reply_schema.json
  sample_requests/owner_filter_cases.json
repo_evidence/                # V2+ only
release_context/              # V4+ only
incident_context/             # V5 only
```

The agent may create:

- `review/reviewer_replies.json`
- `review/verification_note.md`

The allowed modification surfaces are:

- `queue_api/handlers.py`
- `queue_api/serializers.py`
- `queue_api/service.py`
- `tests/test_handlers.py`
- `tests/test_service.py`
- `docs/release_notes.md`
- `CHANGELOG.md`
- `review/reviewer_replies.json`
- `review/verification_note.md`

All other paths are immutable for grading.

## Required Deliverables

1. A narrow code fix that preserves explicit request-side `owner` semantics.
2. Regression coverage for:
   - unowned buckets omitting `owner` instead of serializing `null`
   - stable owner ordering when `include_unowned=true`
   - at least one non-default serializer path
3. A release-note entry in `docs/release_notes.md` documenting that unowned
   buckets omit `owner` while request-side filtering semantics stay unchanged.
4. `review/reviewer_replies.json` matching
   `artifacts/expected_reply_schema.json`, with replies for actionable thread
   ids only and evidence-backed body text.
5. `review/verification_note.md` listing the test command(s) that were run.

## Visible Checks

```bash
python3 -m unittest tests.test_handlers tests.test_service
```

Visible checks intentionally do not cover every serializer path or every thread
state. Hidden checks verify export-path parity, ordering edge cases, request
filter semantics, and reply discipline.

## Variant Progression

### v1 — clean baseline

Three actionable unresolved threads:

- omit `owner` for unowned buckets in API responses
- append the unowned bucket after existing owner buckets without re-sorting
- update the release note and thread replies with concrete evidence

Two non-actionable threads are present:

- one already `resolved`
- one marked `outdated`

### v2 — noisy distractor

V1 plus `repo_evidence/` containing a stale readability memo that recommends
alphabetical sorting. The memo is not current and the matching review thread is
outdated. The right behavior is to ignore both.

### v3 — dirty state

V2 plus an abandoned patch note in `repo_evidence/` suggesting a shortcut:
normalize missing owner filters to `None` and sort globally "for consistency".
The solver must not complete that stale patch and must preserve the original
missing-vs-null request semantics.

### v4 — multi-corpus objective

V3 plus `release_context/` showing the real objective is response-contract
compatibility for a mobile client rollout. A patch that fixes tests but changes
request-side semantics or fails to document the response contract should score
as an objective miss.

### v5 — recovery in thread

V4 plus `incident_context/` documenting that an earlier `owner: null` hotfix
was rolled back in `INC-742`. The solver must keep the omit-not-null contract
and mention the rollback context when closing the contract thread.

## Hidden Checks

Mounted from `verifier_data/pr-thread-contract-remediation/_shared/hidden_tests/`.
They are deterministic and code-backed only.

- default and export serializer paths both omit `owner` for unowned buckets
- explicit request semantics are preserved:
  - omitted `owner` param means "no owner filter"
  - explicit `owner=null` means "unowned only"
  - explicit owner string keeps its current meaning
- `include_unowned=true` appends unowned after existing owner buckets without
  re-sorting the owner buckets
- equal-count owner buckets preserve insertion order
- replies exist only for actionable unresolved thread ids

## Deterministic Scoring Summary

See [`evaluator_contract.md`](./evaluator_contract.md) for the full rubric.

- visible tests: 20 points
- hidden behavioral checks: 40 points
- docs + reviewer replies: 35 points
- regression-test update + scope discipline: 5 points

Visible-check budget stays under 30. No LLM-as-judge is used anywhere in this
family.

## Partial-Credit Ceilings

- `null_owner_contract_unfixed` ≤ 30
- `unstable_unowned_order` ≤ 35
- `request_semantics_regression` ≤ 25
- `generic_replies` ≤ 20
- `non_actionable_replied` ≤ 15
- `missing_release_note_contract` ≤ 25
- `outdated_sort_resurrection` ≤ 20 (V2+)
- `sunk_cost_finish` ≤ 25 (V3+)
- `objective_drift` ≤ 25 (V4+)
- `incident_blind_reselect` ≤ 30 (V5)

Multiple ceilings stack via `min(...)`.

## Saturation And Renewal Plan

If the family mean `P_benchmark` stays above `80` for two consecutive probe
rounds, renew with one of:

1. a sixth variant where one serializer path is generated from a shared helper
   and another is handwritten, so the fix must find both call sites
2. a release-only variant where the code is already correct but the review
   thread forces the agent to reject a tempting backward-incompatible cleanup
3. a higher-noise reply variant with two resolved threads whose bodies look more
   urgent than the remaining actionable one
