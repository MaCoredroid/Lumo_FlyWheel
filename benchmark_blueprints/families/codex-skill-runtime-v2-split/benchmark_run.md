# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-6a35-7211-aa6f-fc326aaa75a1`
  - result: over target under the original evaluator because the visible bundle plus smoke path were over-credited.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-cfd3-78a3-a44a-a23daabaa9f7`
  - workspace: `probe_runs/attempt_02/workspace`

## Hardening Applied After Attempt 01
- Shifted most score weight behind withheld follow-up reuse, stale-reference retirement, and duplicate-automation packs.
- Added a cap at `20` when only the visible bundle is proven.
- Reduced credit for plain file creation and visible-only alignment.

## Attempt 02 Evidence
- Visible command: `pytest -q tests/test_skill_bundle.py tests/test_config_refs.py tests/test_automation_smoke.py`
  - result: `PASS` (`3 passed`)
- Visible command: `python scripts/run_handoff.py --input fixtures/handoff_input.json --output /tmp/handoff.md`
  - result: `PASS`
- Extra audit: `rg` over live authored files
  - result: stale live references removed from the visible bundle
- Solver changed skill/config/automation/runbook files and left the dirty sentinel untouched.

## Scoring Against Hardened Evaluator
- Visible tests pass: `10/10`
- Smoke command succeeds: `10/10`
- Hidden follow-up reuse bundle: `0/20`
- Hidden stale-reference retirement pack: `0/20`
- Hidden duplicate-automation pack: `0/20`
- Trusted canonical-alignment checks: `0/10`
- Hidden dirty-workspace pack: `0/10`
- Raw subtotal before caps: `20/100`
- Final scored result: `20/100`

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive GPT-5.4/high solver can align the visible bundle, but without the withheld follow-up and duplicate-retirement packs it remains at the intended `~20/100` level.
