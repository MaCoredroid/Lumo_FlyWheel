# Benchmark Run

## Attempt History
- `attempt_01`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da338-68da-7523-9acc-d07e32d173df`
  - result: over target under the original evaluator because the visible pack plus config/docs were over-credited.
- `attempt_02`
  - model: `gpt-5.4`
  - reasoning: `high`
  - agent: `019da33f-ce6c-7f10-b165-ceff906b1220`
  - workspace: `probe_runs/attempt_02/workspace`

## Hardening Applied After Attempt 01
- Moved most score weight behind withheld replay, interleaving, and future-event packs.
- Added a cap at `20` when only the visible fixture pack is proven.
- Added a cap at `20` if benchmark-owned tests are edited.

## Attempt 02 Evidence
- Visible command: `pytest -q tests/test_adapter.py tests/test_replay.py tests/test_render.py`
  - result: `PASS` (`3 passed`)
- Extra solver check: ad hoc normalize/serialize/replay sanity check
  - result: `PASS`
- Solver changed code, config, and docs inside the attempt workspace.

## Scoring Against Hardened Evaluator
- Visible tests pass: `10/10`
- Visible adapter message-block handling: `5/5`
- Visible renderer tool-result correlation: `5/5`
- Hidden replay pack: `0/20`
- Hidden interleaved-order pack: `0/20`
- Hidden future-event pack: `0/20`
- Trusted anti-legacy-shim checks: `0/10`
- Config alignment: `5/5`
- Migration doc accuracy: `5/5`
- Raw subtotal before caps: `30/100`
- Cap triggered: only the visible fixture pack was exercised
- Final scored result: `20/100`

## Final Judgment
- Target achieved: `yes`
- Current hardness judgment: a naive GPT-5.4/high solver can clear the visible pack, but without the withheld replay/interleaving suites it stays in the intended `~20/100` band.
