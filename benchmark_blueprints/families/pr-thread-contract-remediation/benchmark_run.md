# Benchmark Run — `pr-thread-contract-remediation`

## Model under test

```bash
codex exec --model gpt-5.4 --reasoning-effort high
```

## attempt_00 — baseline skeleton (2026-04-18)

- family bundle contained contract docs only
- no `workspace_bundle/`, scorer, verifier data, manifest, or verification matrix
- a blueprint-only reasoning pass landed at `20/100` because the solver could
  describe the fix shape but could not ground thread ids, replies, or code
  behavior in a real workspace

Verdict: not probe-ready.

## attempt_01 — family implementation and deterministic sanity (2026-04-23)

Implemented:

- full five-variant workspace bundle
- deterministic scorer at `verifiers/pr-thread-contract-remediation/score_pr_thread_contract.py`
- shared hidden tests
- per-variant oracle overlays, workspace manifests, milestone scripts, and
  `manifest.lock.json`
- verification matrices for V1 and V5

Deterministic validation was run with:

```bash
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/regen_family.py
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v1-clean-baseline --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix.md
python3 benchmark_blueprints/families/pr-thread-contract-remediation/tools/run_verification_matrix.py --variant v5-recovery-in-thread --out benchmark_blueprints/families/pr-thread-contract-remediation/verification_matrix_v5.md
```

Observed deterministic baselines:

- oracle: `95/100` on all five variants
- empty: `0/100` on all five variants
- shortcut: `12/100` on all five variants

Layer A status after this attempt: **live probe still pending**

Why pending:

- this turn focused on turning the family from a skeleton into a runnable
  family-local benchmark with deterministic scoring and verification assets
- a whole-family `codex exec` probe has not yet been run against the freshly
  generated workspace bundle

Next live step:

```bash
codex exec --cd benchmark_blueprints/families/pr-thread-contract-remediation/workspace_bundle/v1-clean-baseline --skip-git-repo-check --sandbox workspace-write --color never --ephemeral "Read AGENTS.md and fix only the actionable review-thread issues."
```
