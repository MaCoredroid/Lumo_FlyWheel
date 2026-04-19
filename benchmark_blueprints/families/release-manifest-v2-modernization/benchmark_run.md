# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-6ac1-7aa0-adcc-8e732e985489`
  - result: over target under the original evaluator because the visible release dry-run and visible smoke path were over-credited.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-cf98-7a03-8d2f-44e216a50103`
  - workspace: `probe_runs/attempt_02/workspace`

## Hardening Applied After Attempt 01
- Shifted most score weight behind withheld staging, reusable-workflow, and alias-retirement packs.
- Added a cap at `20` when only the visible dry-run pack is proven.
- Reduced credit for syntax-only workflow/manifest modernization.

## Attempt 02 Evidence
- Visible command: `pytest -q tests/test_manifest_contract.py tests/test_release_driver.py`
  - result: `PASS` (`2 passed`)
- Visible command: `python scripts/run_ci.py --mode release-dry-run`
  - result: `PASS`
- Extra command: `python deploy/check_release.py --env staging`
  - result: `PASS`
- Solver changed workflow, manifest, config, and rollout-doc files while preserving the unrelated env override comment.

## Scoring Against Hardened Evaluator
- Visible tests pass: `10/10`
- Release dry-run succeeds: `10/10`
- Hidden staging pack: `0/20`
- Hidden reusable-workflow pack: `0/20`
- Hidden alias-retirement and dirty-workspace pack: `0/20`
- Trusted anti-bypass checks: `0/10`
- Config alignment: `5/5`
- Rollout doc accuracy: `5/5`
- Raw subtotal before caps: `30/100`
- Cap triggered: only the visible dry-run pack was proven under the benchmark-owned artifacts
- Final scored result: `20/100`

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive GPT-5.4/high solver can modernize the visible release pack, but without the withheld staging and reusable-workflow packs it stays in the intended `~20/100` band.
