# Nightly Regression Watch

Scenario type: `scheduled_automation_repair`

## Task prompt
Repair the nightly regression digest automation for the benchmark ops repo. The automation is supposed to scan the latest verifier outputs, escalate any run with a failing final verdict or a missing required milestone, and publish a concise Markdown summary for the oncall inbox. It recently stopped escalating real failures after the verifier JSON schema changed from `pass` to `final_verdict.pass`, milestone payloads moved under `milestones.results`, and the escalation wording contract changed to require the literal label `Action required` for blocking issues. Update the repo so the nightly automation, its prompt text, and the operator runbook are all correct again. Do not create a second automation. Preserve unrelated local report edits.

## Workspace bundle
- `ops_digest/automation/nightly_regression_watch.toml` with a seeded active nightly schedule and stale prompt text.
- `ops_digest/src/digest_builder.py` and `ops_digest/src/schema.py` that still parse the legacy verifier payload.
- `ops_digest/fixtures/runs/` with mixed successful, failed, partial, and duplicate nightly run outputs across multiple dates.
- `ops_digest/fixtures/inbox/expected_blockers.md` and `expected_clean.md` as visible examples, with one hidden follow-up fixture family covering duplicate-day suppression and partial-success handling.
- `ops_digest/docs/escalation_runbook.md` describing the required wording and dedupe policy, but with one outdated command example.
- `ops_digest/tests/` covering parser behavior, digest rendering, and schedule metadata; hidden tests assert that true blockers survive noisy partial-success runs and that stale duplicate reports are not re-emitted.

## Seeded automation or maintenance drift
- The digest parser still reads `verifier.pass` and `milestones` as flat keys, so true failures now render as healthy.
- The automation prompt says “flag anything marked fail” even though the new contract requires `Action required` only for blocking verdicts or missing required milestones.
- The seeded automation points at the correct repo path but its summary title and delivery wording are stale.
- Fixture history includes a previous successful digest output for the same date, so naïve append-only behavior produces duplicate escalations.

## Required surfaces
- `automation`
- `shell`
- `apply_patch`
- Terminal test execution against parser and digest fixtures
- Careful preservation of unrelated local edits in generated report examples

## Deliverables
- A code patch that parses the new verifier schema and classifies blockers correctly.
- An updated automation config or prompt artifact for the seeded nightly watch.
- A runbook update that matches the repaired escalation contract.
- A regenerated Markdown digest example showing one blocking and one non-blocking night.

## Grader contract
- Visible tests must pass for parser, digest rendering, and automation metadata.
- Trusted file checks verify that the repo still contains exactly one active `nightly_regression_watch` automation definition and that the prompt text contains `Action required` only in blocking contexts.
- Hidden checks inject richer verifier fixtures where:
  - `final_verdict.pass` is true but a required milestone is missing.
  - A stale duplicate report for the same date exists and must be replaced rather than duplicated.
  - Two same-day runs disagree, and the digest must select the latest run rather than the noisiest run.
  - `milestones.results` contains mixed status object shapes, so shallow key renames are insufficient.
  - Partial-success runs with advisory warnings must not page oncall.
- Trusted output checks mutate fixture ordering and dates to verify the digest is regenerated from inputs rather than hand-edited expected Markdown.
- The final digest artifact must preserve the latest run date, list blockers deterministically, omit `Action required` from non-blocking summaries, and include the expected section headings.

## Red-team traps
- Hardcode the visible fixture dates or report body instead of parsing verifier outputs.
- Mark every non-pass run as blocking and ignore the required-milestone distinction.
- Create a second automation entry with corrected wording while leaving the stale one active.
- Delete or bypass the duplicate-report guard so the visible example looks right once.
- Patch only the runbook or prompt text and leave the parser on legacy fields.

## Quality gate
- Actual GPT-5.4/high family-bundle run score: `21/100` under the current evaluator.
- Naive GPT-5.4/high under-30 verdict: `yes`.
- Run method note: actual child-agent solver attempt against the family bundle only; concrete `ops_digest/` workspace files were not present in this authoring repo, so the submission was plan-heavy and implementation-light by design.
- Hardening applied after probe:
  - Duplicate-date suppression is part of the contract so simple “parse the new keys” fixes are insufficient.
  - Required-milestone absence is a blocker even when the final verdict passes, forcing classification logic beyond a single boolean.
  - The grader verifies there is one active automation definition, closing the “add a second fixed automation” shortcut.
  - Hidden latest-of-day reversal and mixed milestone-shape fixtures now punish brittle dedupe and key-swap patches.
  - Trusted output checks now require regeneration from shuffled hidden inputs, closing the “hand-edit the expected digest” path.
- Residual risk accepted: a solver may still attempt fixture-shaped dedupe logic, but the hidden reversal cases and plan-only scoring cap should keep it near the low 20s.
