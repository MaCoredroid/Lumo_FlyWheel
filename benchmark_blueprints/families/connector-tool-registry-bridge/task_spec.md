# Connector Tool Registry Bridge

- `task_id`: `t9_connector_tool_registry_bridge_pr_triage`
- `target_family_id`: `connector-tool-registry-bridge`

## Task Prompt
Repair a Codex integration gateway after the GitHub and Gmail tool registry drifted. Discovery fixtures now advertise renamed tool ids, but the local registry manifest, `.codex/config.toml`, and router fallback logic still assume the old names. Fix the gateway so discovery, routing, and fallback all agree again. Do not solve this by hardcoding one connector or disabling fallback tests.

## Workspace Bundle
- Small Python integration mini-bundle with a visible registry, config, and routing layer.
- Key paths:
  - `registry/tool_catalog.yaml`
  - `.codex/config.toml`
  - `gateway/router.py`
  - `gateway/discovery.py`
  - `gateway/tool_registry.py`
  - `docs/tool_routing.md`
- Primary local commands:
  - `python -m gateway.discovery <jsonl replay>`
  - `python -m py_compile gateway/*.py`

## Seeded Integration Or Plugin Drift
- Discovery fixtures renamed `github.list_prs` to `github.list_pull_requests` and `gmail.find_threads` to `gmail.search_threads`.
- `registry/tool_catalog.yaml` still maps the old ids and one old handler name.
- `.codex/config.toml` prefers the GitHub connector for `github.list_prs`, which no longer exists.
- Router fallback logic treats unknown `search_threads` calls as generic search and can incorrectly fall back into the GitHub lane when Gmail is unavailable.
- Hidden checks expect exact connector ownership plus deterministic fallback order when the preferred connector is missing one tool.

## Required Surfaces
- `plugins_mcp`
- `shell`
- `apply_patch`
- `tool_registry`

## Deliverables
- Code and config patch aligning discovery, registry, and fallback routing.
- Updated tool-routing doc.
- Coherent visible routing behavior without narrow hardcoded one-off fixes.
- Short report on what drift was fixed and which connector owns each renamed tool.

## Grader Contract
- Phase 1: replay the provided discovery session and regenerate the in-memory tool map, then assert that the regenerated map contains only canonical tool ids and canonical handler names.
- Phase 2: inspect the visible registry, config, router, discovery helper, and docs for consistent canonical ownership; this mini bundle ships no meaningful visible test suite.
- Phase 3 trusted checks:
  - Renamed tool ids are present in the registry, config, and regenerated discovery map exactly once, and retired ids are absent from live surfaces.
  - Old handler names are absent from the registry, generated tool map, and docs.
  - Router behavior preserves connector ownership when the preferred connector is present and when it is missing a tool; Gmail tools cannot silently route through GitHub search helpers.
  - Hidden replay cases use reversed connector discovery order, Gmail-unavailable fallback, GitHub-missing-one-tool fallback, and a third generic `search_*` tool to catch string-prefix hardcoding.
  - Generated registry outputs, `.codex/config.toml`, and `docs/tool_routing.md` remain consistent after replay and name the canonical owners for the renamed tools.
  - Per-tool routing must derive from registry or discovery metadata rather than hardcoded `if tool_id == ...` branches.
  - Patching tests or fixtures, broad catch-all routing, narrow hardcoded special cases, or alias-only hacks fails grading.
  - Visible discovery plus fallback tests are worth at most 25/100 without passing the hidden routing and metadata checks.

## Red-Team Traps
- Add both old and new ids everywhere without removing ambiguous ownership.
- Hardcode Gmail tools to whichever connector happens to be loaded first.
- Patch tests or fixtures instead of fixing registry and router logic.
- Accept any `*.search_*` tool as a generic search route.
- Update config only, leaving runtime router behavior stale.

## Quality Gate
- Intended naive-solver failure mode: rename the obvious tool ids, pass visible discovery tests, and miss the hidden connector-ownership and fallback-order checks.
- Hardening built into the task:
  - Hidden grading verifies preferred-present routing and preferred-missing fallback semantics, not just static registry content.
  - Cross-surface consistency is required across registry YAML, TOML, discovery replay, generated maps, docs, and router code.
  - Alias-preserving fixes are penalized when they leave ambiguous ownership.
  - Hidden replay cases catch narrow tool-id hardcoding and connector-order assumptions.
  - Visible tests and fixtures are integrity-guarded against weakening.
- GPT-5.4/high probe result: pre-hardening estimate `35-45/100`; easy path was rename-plus-special-case routing that could bank visible points without canonical handler cleanup or robust fallback semantics.
- Hardening applied after probe:
  - Added exact absence checks for retired ids and handler names across live surfaces.
  - Added hidden preferred-present and preferred-missing routing cases, including reversed discovery order and a third generic search tool.
  - Required docs and regenerated discovery output to match canonical ownership.
  - Added anti-hardcode language and capped visible-path credit below 30.
- Final under-30 judgment for a naive GPT-5.4/high solver: `Yes`, now looks under 30 because a two-id rename or narrow router special case no longer earns meaningful credit.
- Observed GPT-5.4/high benchmark run: `20/100` on the visible bundle, which lands in the intended `15-25` target band.
