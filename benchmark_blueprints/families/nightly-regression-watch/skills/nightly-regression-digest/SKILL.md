# Nightly Regression Digest

Use this skill when the task is to repair or validate a scheduled regression digest, escalation summary, or benchmark-nightly inbox artifact.

## Inputs
- Verifier JSON payloads or fixture directories
- Existing automation config and prompt text
- Runbook or escalation wording contract

## Workflow
1. Parse the verifier payload shape before editing wording or examples.
2. Determine what counts as blocking versus advisory.
3. Preserve the existing automation identity; do not duplicate it.
4. Regenerate the digest artifact from code, then compare it with the contract.
5. Update operator docs only after parser and classification behavior are correct.

## Avoid
- Hardcoding visible fixture dates or report rows.
- Marking every non-pass signal as blocking.
- Adding a replacement automation alongside the stale one.
- Hand-editing output examples without proving regeneration.

## Expected output
- One repaired automation definition
- One regenerated digest example
- One concise note explaining blocker classification and duplicate-day handling
