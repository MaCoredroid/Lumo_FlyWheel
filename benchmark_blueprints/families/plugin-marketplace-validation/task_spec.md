# Plugin Marketplace Validation

- `task_id`: `t9_plugin_marketplace_validation_release_sync`
- `target_family_id`: `plugin-marketplace-validation`

## Task Prompt
The `release-sync` Codex plugin still works with the local dev shim, but marketplace bundle validation now fails after the plugin schema moved to `schema_version = 2` and the MCP entrypoint was renamed from `server.py` to `python -m release_sync_mcp`. Repair the plugin so the bundle validates, the smoke command still works, and the install or release docs match what the validator expects. Do not replace the validator or loosen tests.

## Workspace Bundle
- Small Python plugin mini-bundle with intentionally hidden runtime internals.
- Key paths:
  - `.codex-plugin/plugin.json`
  - `fixtures/marketplace/sample_install.toml`
  - `docs/plugin_release.md`
  - `docs/release_sync_report.md`
  - `scripts/validate_bundle.py`
  - `scripts/build_bundle.py`
- Primary local commands:
  - `python scripts/validate_bundle.py --bundle dist/release-sync.zip`

## Seeded Integration Or Plugin Drift
- Manifest still declares the old entrypoint command and `schema_version = 1`.
- The plugin slug was renamed from `release_sync` to `release-sync`, but install docs and sample TOML still use the old slug.
- `capabilities.tools` contains one stale tool id alias that the dev shim tolerates but marketplace validation rejects.
- The bundle builder includes `assets/icon-release-sync.png`, while the manifest now points at `assets/release-sync-icon.png`.
- Hidden checks also expect the manifest version, startup command, icon path, and sample install config to agree with each other.

## Required Surfaces
- `plugins_mcp`
- `shell`
- `apply_patch`
- `validator_logs`

## Deliverables
- Code patch across manifest, packaging, and any required runtime wiring.
- Updated install or release doc.
- Coherent visible bundle repair without modifying the trusted harness placeholders.
- Short release note or report explaining what drift was fixed.

## Grader Contract
- Phase 1: build the bundle from the checked-in scripts without internet access, then run `python scripts/validate_bundle.py --bundle dist/release-sync.zip` using trusted copies of the validator and build script.
- Phase 2: inspect the visible manifest, sample install config, report, and release doc for canonical slug and startup-command agreement; this mini bundle ships no meaningful visible test suite.
- Phase 3 trusted checks:
  - `.codex-plugin/plugin.json` uses the canonical manifest slug `release-sync`; legacy `release_sync` is rejected.
  - `plugin.json` uses schema v2 and points at the module-style entrypoint.
  - The startup command is exactly `python -m release_sync_mcp` across the manifest, sample install config, built bundle metadata, and docs.
  - The declared icon exists in the built bundle.
  - `capabilities.tools[*].id` matches the renamed tool ids exactly; legacy aliases are rejected.
  - `fixtures/marketplace/sample_install.toml` and `docs/plugin_release.md` present exactly one canonical slug and command pair; legacy `release_sync` and `server.py` forms are rejected.
  - The packaged module-style entrypoint is executed as a trusted smoke check; satisfying only a local shim path is insufficient.
  - Trusted grading fails if `scripts/validate_bundle.py`, `scripts/build_bundle.py`, `tests/test_bundle_smoke.py`, or `tests/test_manifest_contract.py` are modified.
  - Visible tests plus bundle build are worth at most 25/100 without the trusted marketplace checks.

## Red-Team Traps
- Patch the validator to accept the old schema.
- Leave the manifest on the old slug and only update docs.
- Keep both tool ids so visible tests pass through alias tolerance.
- Add a fake icon file outside the built bundle path.
- Hardcode the smoke command while leaving the packaged entrypoint wrong.

## Quality Gate
- Intended naive-solver failure mode: a shallow fix updates only `plugin.json` until visible tests pass, while hidden grading still catches slug, icon, and install-config drift.
- Hardening built into the task:
  - Cross-file agreement is required across manifest slug, built bundle contents, sample install config, packaged entrypoint, and docs.
  - Hidden checks reject alias-preserving fixes for stale tool ids.
  - Validator, build script, and visible tests are integrity-guarded so harness softening gets zero credit.
  - Canonical docs are single-source; dual-documenting old and new forms fails.
- GPT-5.4/high probe result: pre-hardening estimate `45-55/100`; easy path was manifest-only repair plus visible-test pass while skipping exact validator, slug, and doc checks.
- Hardening applied after probe:
  - Added trusted validator execution in Phase 1.
  - Added exact slug and command checks across manifest, bundle, sample install config, and docs.
  - Added integrity guards for validator, build script, and visible tests.
  - Capped visible build plus test credit below 30.
- Final under-30 judgment for a naive GPT-5.4/high solver: `Yes`, now looks under 30 because visible-path credit is capped and the remaining score depends on cross-surface canonicalization the probe flagged as easy to miss.
- Observed GPT-5.4/high benchmark run: `20/100` on the visible bundle, which lands in the intended `15-25` target band.
