# `request-path-evidence-brief` Task Spec

**Track:** 02 — Codebase Understanding  
**Family id:** `request-path-evidence-brief`  
**Spec version:** CNB-55 v1.0  
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt (canonical)

You are dropped into a small Python service repo plus a stale support note.

The support note claims:

1. `owner_source` in the exported payload is coming from storage.
2. `routing_key` is computed before the CLI applies `--owner`.

Do not start by editing code. Read the repo first. Trace the live path for `--owner`, `owner_source`, and `routing_key` from CLI ingress through service logic, storage, serialization, and the tests that prove the behavior.

Produce exactly three files under `artifacts/`:

- `request_path_brief.md`
- `path_map.json`
- `docs_correction.md`

No behavior patch is required. If the only valid change is a docs or review-note correction, state that precisely.

## Required Output Contract

### Markdown outputs

In `request_path_brief.md` and `docs_correction.md`, cite evidence using backticked `path::symbol` tokens, for example:

- `sync_app/cli.py::main`
- `sync_app/service.py::_resolve_owner`
- `sync_app/serializer.py::build_routing_key`

The brief must:

- state whether the support note is correct,
- name the live hop sequence in order,
- distinguish derivation from emission,
- cite at least one proving test, and
- reject at least one plausible-but-nonlive decoy.

The correction note must stay narrow. It should correct the false claim and avoid speculative refactors.

### `artifacts/path_map.json`

Schema version: `cnb55.request_path_map.v1`

Required top-level fields:

- `schema_version`
- `variant_id`
- `live_path[]`
- `field_derivations.owner_source`
- `field_derivations.routing_key`
- `field_derivations.emission`
- `test_observations[]`
- `rejected_decoys[]`

Each `live_path[]` row must include:

- `step`
- `file`
- `symbol`
- `role`
- `caller_symbol`
- `callee_symbol`

`caller_symbol` / `callee_symbol` are used to prove contiguous hop adjacency, not just module ordering.

## Required Surfaces

- read-only shell inspection
- repo search and file reads
- optional `pytest -q`
- optional `python -m sync_app.cli ...` to observe payload behavior
- writing the three `artifacts/` files only

No browser, network, MCP, or subagents.

## Workspace Bundle

Every variant ships a repo-local bundle under `workspace_bundle/<variant>/` with:

```text
AGENTS.md
Dockerfile
.scenario_variant
sync_app/{cli.py,service.py,store.py,serializer.py,__init__.py}
config/defaults.json
docs/{cli.md,data_flow.md}
ops/support_note.md
tests/{conftest.py,test_sync.py,test_docs.py,test_trace_outputs.py}
artifacts/README.md
release_context/...        # v4+
incident_context/...       # v5
ops/previous_session_path.md  # v3+
```

The only writable outputs are:

- `artifacts/request_path_brief.md`
- `artifacts/path_map.json`
- `artifacts/docs_correction.md`

Evidence outside the provided bundle is invalid.

## Variant Ladder

### v1 — clean baseline

- Live code path is fully present and test-backed.
- One stale support note is wrong.
- One plausible dead helper exists in storage.

### v2 — noisy distractor

- V1 plus a second storage-shaped helper that looks grep-live.
- Still a single live code path, but the wrong helper is easier to anchor on.

### v3 — dirty state

- V2 plus `ops/previous_session_path.md`, an abandoned draft from a prior investigation.
- Tests whether the solver can reject stale prose instead of treating it as code evidence.

### v4 — multi-corpus / objective drift

- V3 plus `release_context/` design notes describing a future serializer cleanup.
- Tests whether the solver can separate planned architecture from implemented behavior.

### v5 — recovery in thread

- V4 plus `incident_context/` showing that a previous docs correction was rolled back.
- Tests whether the solver can acknowledge the rollback and avoid repeating the stale correction.

## Grader Contract

### Visible checks (≤ 30 points)

`pytest -q tests/test_sync.py tests/test_docs.py tests/test_trace_outputs.py`

Visible checks cover:

- the three required artifacts exist,
- `path_map.json` parses with the expected schema version,
- `variant_id` matches `.scenario_variant`,
- the repo’s code and docs surfaces still match the promised bundle.

### Hidden checks (deterministic)

Hidden checks score:

- exact live-path step sequence,
- correct field derivations for `owner_source` / `routing_key` / emission,
- correct test evidence,
- correct decoy rejection,
- markdown grounding via `path::symbol` tokens,
- narrow docs correction that rejects the support note’s false claim.

### Trusted final-state checks

Any mutation to readonly inputs or writes outside the three allowed artifact files triggers integrity failure.

## Partial-Credit Ceilings

- `no_artifacts` → `0`
- `malformed_path_map` → `10`
- `missing_symbol_adjacency` → `25`
- `store_claimed_as_decision_layer` → `25`
- `pre_owner_routing_claim` → `30`
- `no_support_note_verdict` → `35`
- `no_decoy_rejection` → `35`
- `weak_markdown_grounding` → `45`
- `dirty_state_trust` → `40` (V3)
- `future_architecture_anchor` → `45` (V4)
- `incident_blind_reselect` → `30` (V5)
- `external_evidence` → `20`

## Quality Gate

The family is only meaningful if a careful solver can reconstruct the live chain from code plus tests, while a shallow solver gets clipped for:

- trusting support prose over code,
- mistaking dead helpers for live hops,
- confusing derivation with serialization,
- skipping a concrete decoy rejection,
- or writing polished but ungrounded markdown.

## Saturation / Renewal

If probe means rise above the saturation threshold for two consecutive rounds, renew with:

- a second CLI ingress that shares serializer code but changes the live path, or
- cross-test shadow helpers that deepen the dead-helper ambiguity without inventing rubric traps.
