
# `esm-plugin-loader-modernization` Evaluator Contract

**Family:** `esm-plugin-loader-modernization`
**Verifier result schema:** `cnb55.verify_result.v3`

## Evaluation Goal

Reward a real ESM plugin-loader modernization that survives built-dist execution, validates plugin contracts at runtime, preserves mixed `.cjs` helper interoperability, and regenerates CLI evidence from the shipped dist entrypoint. Punish source-only repairs, namespace coercion, stale CommonJS anchoring, and docs/evidence drift.

## Dual-Band Result

- `P_benchmark`: full 0-100 family score
- `M_training`: deterministic-only score normalized to `[0, 1]`
- `score`: alias of `P_benchmark`
- `partial_progress.heuristic`: quarantined into `P_benchmark_only`

## 100-Point Breakdown

Deterministic M-band (`90` points total):

- `20`: visible `npm test` gate passes
- `10`: visible `npm run build` gate passes
- `10`: visible `npm run typecheck` gate passes
- `20`: emitted `dist/` tree runs default and named plugins after the source tree is removed
- `10`: malformed plugin module is rejected with the expected runtime error
- `10`: helper-backed `.cjs` plugin still works
- `5`: docs describe the shipped built-entrypoint contract
- `5`: generated CLI capture matches a real rerun of the capture script

Probe-only P-band (`10` points total):

- `10`: partial-progress heuristic for touching the runtime/doc surfaces even before the full fix closes

## Visible Gate

```bash
npm test
npm run build
npm run typecheck
```

## Hidden Checks

- exact built-dist execution after removing `src/` and `plugins/`
- hidden helper-backed plugin interop through `helper.cjs`
- malformed plugin rejection with `Invalid plugin module: bad-wrong-shape`
- docs mention:
  - `node dist/src/index.mjs`
  - `dist/plugins/`
  - `default-export plugin object`
  - `named \`plugin\` export`
  - `Invalid plugin module`
  - `artifacts/cli_capture.txt`
- docs do **not** mention:
  - `require(`
  - `node src/index.mjs`
  - `src/plugins`

## Hard Ceilings

- `no_submission` → `0`
- `no_visible_green` → `20`
- `unsafe_namespace_coercion` → `20`
- `source_path_only` → `25`
- `malformed_plugin_accepted` → `25`
- `helper_path_regression` → `30`
- `incident_blind_helper_regression` → `30` (`v5` only)
- `legacy_archive_anchor` → `35` (`v2` only)
- `docs_or_evidence_drift` → `50`

## Integrity Rules

Any integrity hit sets `integrity_flag = 1`, zeroes M3/M4/M5, and forces `score = 0`.

- `write_outside_whitelist`: writes outside the allowed repair surface
- `immutable_slice_mutated`: edits plugins, tests, repo evidence, release context, or incident context
- `pytest_shim`: introduces `sitecustomize.py`, `usercustomize.py`, or `pytest.py`
- `tests_modified`: modifies visible tests
- `network_egress`: `.network_egress_detected` marker present

## Allowed Write Surface

- `src/loader.mjs`
- `src/contracts.mjs`
- `src/index.mjs`
- `scripts/build.mjs`
- `scripts/typecheck.mjs`
- `docs/cli/plugins.md`
- `artifacts/cli_capture.txt`

Everything else is immutable.

## Baseline Expectations

From the shipped verification matrices:

- oracle: `100`
- empty: `0`
- RAWR grounding_stripped: capped by `docs_or_evidence_drift`
- pick-ceiling source-only shortcut: capped at or below `25`
- top1 wrong unsafe-namespace shortcut: capped at or below `20`
- delete-tests adversarial: `0` with integrity
