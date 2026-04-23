# `codex-provider-rollover` Task Spec

**Track:** 03 — Refactor Modernization
**Family id:** `codex-provider-rollover`
**Spec version:** CNB-55 v1.0
**Variants:** 5 (`v1-clean-baseline` through `v5-recovery-in-thread`)

## Task Prompt (canonical)

Restore the local maintenance Codex profile after a provider rollover. The repo
was moved from a direct vLLM endpoint to a proxy-backed Responses route, but
the maintenance profile drifted:

- `serving_maintenance/.codex/config.toml` still selects the legacy provider.
- `serving_maintenance/scripts/smoke_responses_profile.py` only proves first
  turn success and does not assert exact `previous_response_id` continuity.
- `serving_maintenance/docs/provider_rollover.md` and
  `serving_maintenance/docs/smoke.md` still describe the wrong provider id or
  direct endpoint workflow.

Repair the concrete maintenance workspace in place. Preserve the unrelated
local tuning block in the TOML file exactly as written. Do not rewrite tests,
fixtures, or read-only rollout evidence.

## Required Surfaces

- `shell`
- `apply_patch`
- terminal test execution
- TOML editing with preservation discipline
- JSON fixture inspection
- docs update

No network, browser, or subagents are required for the benchmark itself.

## Workspace Bundle (per variant)

Every variant ships the following under `workspace_bundle/<variant_id>/`:

```text
AGENTS.md
Dockerfile
.scenario_variant
bin/run-visible-tests
serving_maintenance/.codex/config.toml
serving_maintenance/scripts/smoke_responses_profile.py
serving_maintenance/fixtures/http/turn1_ok.json
serving_maintenance/fixtures/http/turn2_ok.json
serving_maintenance/fixtures/http/turn2_missing_store.json
serving_maintenance/docs/provider_rollover.md
serving_maintenance/docs/smoke.md
tests/test_config_profile.py
tests/test_docs_sync.py
tests/test_smoke_profile.py
release_context/*            # V4+ only
incident_context/*           # V5 only
```

## Seeded Modernization Breakage

The starting workspace is intentionally broken in the same family-wide ways:

- top-level `provider` still points at the legacy direct endpoint profile
- smoke script reports success after turn one and does not reject bad turn-two
  chaining
- docs are out of sync with the actual proxy-backed Responses path
- the local tuning block is present in the config file and must survive
  byte-for-byte across the repair

## Variant Progression

### `v1-clean-baseline`

Single-profile rollover. The solver must move the selected provider to the
Responses proxy path, require `store = true`, and validate exact
`previous_response_id` continuity on the second turn.

- Primary stress: basic modernization correctness.
- Expected capable-model mean after live calibration: low `20s`.

### `v2-noisy-distractor`

Adds a stale canary note and an extra provider stanza that looks attractive but
is not the maintained profile. The correct fix is still the canonical
`responses_proxy` profile, not the canary or the legacy direct route.

- Primary stress: ignoring plausible-but-wrong provider distractors.
- New ceiling: `canary_anchoring`.
- Expected capable-model mean: about `20`.

### `v3-dirty-state`

Adds a read-only template-rewrite helper and template TOML. The tempting but
wrong move is to regenerate the entire config from the template, which drops
the local tuning block and fails preservation checks.

- Primary stress: precise patching instead of destructive regeneration.
- New ceiling: `template_overwrite`.
- Expected capable-model mean: about `17`.

### `v4-multi-corpus-objective`

Adds `release_context/` showing the operational objective is the specific
maintenance profile used during proxy-backed Responses cutovers, not a generic
"make the smoke green" cleanup. Docs must align to the maintenance profile
label and proxy-backed route.

- Primary stress: objective alignment across added evidence.
- New ceiling: `maintenance_profile_drift`.
- Expected capable-model mean: about `14`.

### `v5-recovery-in-thread`

Adds `incident_context/` documenting a rollback caused by a prior hotfix that
checked only first-turn success while leaving `store = false` and follow-up
retrieval broken. The updated docs must acknowledge that rollback and the smoke
must reject the same failure shape deterministically.

- Primary stress: recovery-aware modernization under incident evidence.
- New ceiling: `rollback_blind`.
- Expected capable-model mean: about `10`.

### Ladder Monotonicity Target

Calibration target for GPT-5.4/high after the first counted live rerun:

- `v1-clean-baseline`: `22`
- `v2-noisy-distractor`: `20`
- `v3-dirty-state`: `17`
- `v4-multi-corpus-objective`: `14`
- `v5-recovery-in-thread`: `10`

Family mean target: `16.6`, within the Layer A freeze window `[15, 25]`.

## Expected Deliverables

- targeted update to `serving_maintenance/.codex/config.toml`
- strengthened `serving_maintenance/scripts/smoke_responses_profile.py`
- aligned `serving_maintenance/docs/provider_rollover.md`
- aligned `serving_maintenance/docs/smoke.md`

No other workspace writes are necessary or allowed.

## Visible Checks

```bash
bin/run-visible-tests
```

Visible checks cover:

- selected provider id is the proxy-backed Responses profile
- the selected provider stanza has the expected base URL and `store = true`
- good turn-two continuity passes and bad turn-two continuity fails
- docs mention the selected provider and exact `previous_response_id` follow-up

Visible success is intentionally insufficient for a passing score.

## Hidden Checks

Hidden checks are deterministic and code-backed only. No LLM judge is used.

- alternative bad follow-up fixtures must fail for the same continuity contract,
  not because the script hardcoded one fixture filename
- provider id, base URL, and `store` must be validated as one coupled contract
- the local tuning block must remain byte-for-byte identical
- docs must stay aligned with the selected provider id and variant-specific
  release / incident expectations
- read-only surfaces must remain untouched

## Trusted Final-State Checks

- no edits outside:
  - `serving_maintenance/.codex/config.toml`
  - `serving_maintenance/scripts/smoke_responses_profile.py`
  - `serving_maintenance/docs/provider_rollover.md`
  - `serving_maintenance/docs/smoke.md`
- no edits to `tests/`, `serving_maintenance/fixtures/`, `release_context/`, or
  `incident_context/`
- no `sitecustomize.py`, `usercustomize.py`, or `pytest.py` shim
- no `.network_egress_detected` marker

## Partial-Credit Ceilings

- `no_visible_green` ≤ `25`
  Trigger: visible test slice still fails.
- `legacy_default_selected` ≤ `20`
  Trigger: selected provider remains `legacy_vllm` or another non-canonical id.
- `proxy_route_incorrect` ≤ `30`
  Trigger: selected provider stanza is not the proxy-backed Responses route with
  `store = true`.
- `continuity_not_exact` ≤ `20`
  Trigger: smoke does not require exact `previous_response_id` chaining and bad
  hidden fixtures are not rejected.
- `config_rewritten` ≤ `30`
  Trigger: the preserved local tuning block is not byte-identical.
- `docs_unaligned` ≤ `40`
  Trigger: provider id, proxy route, or `previous_response_id` semantics are
  inconsistent across docs.
- `canary_anchoring` ≤ `35`
  Trigger: V2 docs or config anchor on the distractor canary profile.
- `template_overwrite` ≤ `30`
  Trigger: V3 drops the tuning block or reintroduces template markers.
- `maintenance_profile_drift` ≤ `45`
  Trigger: V4 ignores release-context maintenance-profile requirements.
- `rollback_blind` ≤ `35`
  Trigger: V5 ignores the rollback context around `store = true` and follow-up
  continuity.

## Red-Team Traps

1. Rewrite the whole TOML file and lose the local tuning block.
2. Make the smoke check only that turn one produced an id.
3. Patch docs alone while leaving the selected provider stale.
4. Hardcode fixture filenames or response ids instead of validating structure.
5. Select the canary or legacy provider because it looks superficially valid.

## Saturation And Renewal Plan

Per Layer B readiness, this family is renewed when `mean P_benchmark > 80` for
two consecutive probe rounds.

Renewal queue:

1. Add a V6 with proxy auth header rollover layered onto the same profile.
2. Add a V7 with per-profile store-policy drift across two maintenance scripts.
3. Retire V1 once provider-id rollover becomes purely mechanical.
