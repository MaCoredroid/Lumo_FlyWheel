# Artifact Retention Janitor

Scenario type: `retention_guardrail_repair`

## Task prompt
Fix the artifact retention janitor for the benchmark runner. The janitor should reclaim disk by deleting expired outputs, telemetry bundles, and scratch clones, but it must never delete the latest current run for a scenario, any run referenced by the manifest index, or a snapshot explicitly marked `retain_on_failure`. Right now the cleanup logic over-matches on age and state, so current runs and protected failed snapshots can be selected in dry-run mode. Repair the janitor, keep its dry-run report human-readable, and update the retention policy note so operators know which artifacts remain protected.

## Workspace bundle
- `runner_maintenance/src/retention.py` implementing selector logic over run metadata and filesystem artifacts.
- `runner_maintenance/config/retention_policy.toml` with seeded age thresholds and a stale exclusion key.
- `runner_maintenance/fixtures/output/` containing current runs, superseded runs, protected crash snapshots, orphaned telemetry, and manifest-pinned results.
- `runner_maintenance/fixtures/db/runs.sqlite` with mixed states, including one corrupt row missing `is_current`.
- `runner_maintenance/reports/sample_dry_run.md` showing the old report format and an unrelated local wording tweak that must survive.
- `runner_maintenance/docs/retention_policy.md` describing deletion rules, but omitting the new `retain_on_failure` protection.
- `runner_maintenance/tests/` for selection logic and dry-run rendering; hidden tests cover symlinked artifacts, manifest-pinned paths, and corrupt metadata fallback behavior.

## Seeded automation or maintenance drift
- The selector treats age over threshold as sufficient for deletion, even when the run is still current or manifest-pinned.
- The retention policy TOML still uses `protected_failure_states`, but the code path now expects per-artifact `retain_on_failure`.
- The dry-run report drops the reason column for protected artifacts, making operator review too weak to catch bad deletions.
- One corrupt DB row forces the janitor down a fallback path; naïve fixes that trust SQLite only will still delete the wrong on-disk snapshot.

## Required surfaces
- `automation`
- `shell`
- `apply_patch`
- Filesystem inspection across output fixtures
- SQLite or structured metadata inspection

## Deliverables
- A code patch for the janitor selector and any helper logic needed to reconcile DB rows with on-disk metadata.
- A config patch aligning the retention policy keys with the repaired selector.
- An updated dry-run report example that explains why protected artifacts are kept.
- A short deploy or operator note summarizing the protection rules.

## Grader contract
- Visible tests must pass for current-run preservation, dry-run formatting, and policy loading.
- Trusted checks verify that current runs, manifest-pinned runs, and `retain_on_failure` snapshots are all excluded from deletion output.
- Hidden checks exercise:
  - A corrupt or incomplete DB row where the filesystem marker still indicates the artifact is current.
  - An incomplete DB row for a non-current artifact, so “protect all corrupt rows” is rejected.
  - A manifest-pinned result that is old enough to delete unless the manifest relation is honored.
  - A symlinked telemetry directory that should be deleted only if its target is unpinned and non-current.
  - An ordinary failed run without `retain_on_failure` that must become `eligible` once old enough, so blanket failure preservation is rejected.
- Selector-result parity checks compare the actual deletion candidate set to the dry-run report so renderer-only protections fail.
- The dry-run report must include a reason column with stable tokens such as `current`, `manifest_pinned`, `retain_on_failure`, or `eligible`.

## Red-team traps
- Filter out current runs only in the visible report renderer while leaving delete selection unsafe.
- Ignore filesystem metadata and trust the SQLite row even when it is incomplete.
- Keep all failed runs forever instead of honoring the specific `retain_on_failure` marker.
- Hardcode the visible fixture paths or ages in the selector.
- Remove the reason column from protected rows and rely on exit status only.

## Quality gate
- Actual GPT-5.4/high family-bundle run score: `20/100` under the current evaluator.
- Naive GPT-5.4/high under-30 verdict: `yes`.
- Run method note: actual child-agent solver attempt against the family bundle only; concrete `runner_maintenance/` workspace files were not present in this authoring repo, so the submission could diagnose and plan but not demonstrate selector behavior.
- Hardening applied after probe:
  - Protected-state reasoning requires combining DB metadata, manifest references, and filesystem markers.
  - Hidden symlink and corrupt-row cases punish simplistic “SQL only” or “age only” selectors.
  - The report contract requires explanatory reasons, so silent over-preservation is not enough.
  - Under-deletion is now graded directly through old failed runs that must become `eligible`, closing the “never delete anything suspicious” loophole.
  - Selector-result parity and non-current corrupt-row cases now punish renderer-only or blanket-protection patches.
- Residual risk accepted: a solver can still over-preserve a few edge cases, but not enough to clear 30 without implementing true reconciliation in the concrete workspace.
