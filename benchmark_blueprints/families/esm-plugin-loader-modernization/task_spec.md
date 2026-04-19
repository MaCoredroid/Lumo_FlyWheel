# `esm-plugin-loader-modernization` Task Spec

## Task Prompt
Modernize the `ops-report` TypeScript CLI from CommonJS plugin discovery to an ESM-first loader with typed plugin contracts. The updated loader must work from emitted `dist/` output, support both default-export and named-export plugins, reject malformed plugins with a clear runtime error, and keep the mixed `.cjs` helper path working. Update the docs and generated CLI evidence so they match the real built output.

## Workspace Bundle
- `package.json`: package now declares `"type": "module"` but build wiring is stale.
- `tsconfig.json`: partially migrated and still points at old module settings.
- `src/index.ts`: CLI entrypoint.
- `src/loader.ts`: plugin discovery and dynamic import logic.
- `src/contracts.ts`: plugin interface definitions.
- `plugins/good-default.ts`: valid default export plugin.
- `plugins/good-named.ts`: valid named export plugin.
- `plugins/bad-wrong-shape.ts`: malformed plugin used by hidden checks.
- `plugins/helper.cjs`: unrelated helper that must still load.
- `docs/cli/plugins.md`: stale help excerpt.
- `scripts/capture_cli_output.sh`: generates CLI evidence from real commands.
- `tests/test_loader.ts`, `tests/test_cli_help.ts`, `tests/test_contracts.ts`: visible tests.

## Seeded Modernization Breakage
- The package flipped to ESM, but `loader.ts` still uses `require()`-based discovery.
- The current loader special-cases source-relative paths and does not validate plugin shape at runtime.
- A mixed `.cjs` helper remains valid and must not be deleted or rewritten away.
- The docs still describe the old plugin path and help output.
- A screenshot or evidence artifact requirement exists, but file presence alone is not enough.

## Required Surfaces
- `shell`
- `apply_patch`
- `terminal_tests`
- TypeScript build and typecheck
- docs update
- generated CLI evidence artifact

## Expected Deliverables
- Code patch across loader, contracts, and CLI entrypoint.
- Updated build/config files needed for real ESM loading from `dist/`.
- Updated docs reflecting the actual CLI help and plugin-list behavior.
- Regenerated CLI evidence from the repo-provided script, not a hand-made placeholder.

## Grader Contract
- Phase 2 visible checks:
  - `npm test`
  - `npm run build`
  - `npm run typecheck`
- Hidden checks:
  - Loader resolves plugins from built `dist/` output, not source-only paths.
  - Both default-export and named-export plugins load through the same validated contract.
  - Malformed plugin modules fail with a specific runtime validation error.
  - Mixed `.cjs` helper path remains functional.
  - Generated CLI evidence matches the real command output.
- Trusted final-state checks:
  - No `require(` remains in the live loader path.
  - No `default ?? mod as Plugin`-style unsafe coercion survives without runtime narrowing.
  - The malformed plugin is rejected rather than silently coerced.
  - Docs and evidence are tied to actual CLI output, not static file creation.
- Scoring intent:
  - Config-only or loader-only rewiring is capped at `0.25`.
  - Scores above `0.50` require valid docs plus evidence generated from real commands.
  - Scores above `0.75` require hidden wrong-shape checks and built-dist verification.

## Red-Team Traps
- Special-case the visible plugin filenames instead of validating the contract.
- Load `src/plugins` successfully while `dist/` remains broken.
- Use `default ?? mod as Plugin` and bypass real runtime validation.
- Convert or delete the unrelated `.cjs` helper.
- Update docs and evidence without functional loader correctness.
- Overwrite dirty plugin or doc edits in v3 variants.

## Quality Gate
- GPT-5.4/high adversarial probe judgment before hardening: `under_30_for_naive_solver = no`.
- Probe found a templatic migration path: standard ESM rewiring plus superficial docs and screenshot churn.
- Hardening applied in this spec:
  - Required `build + typecheck + tests`, not just one green path.
  - Forced built-dist loading, typed runtime validation, and mixed `.cjs` coexistence.
  - Bound the evidence artifact to a repo script that captures actual CLI output.
  - Added malformed-plugin hidden checks so namespace coercion does not earn a passing score.
- Final quality judgment after hardening: `under_30_for_naive_solver = yes`.
- Rationale: a naive solver can land obvious module-system edits, but it should stay under 30 without real built-output loading, contract validation, and evidence tied to the live CLI.
