# Transcript Merge Regression

## Canonical Prompt

Saved transcript replays are dropping or duplicating tool-output blocks
after transcript merge. Repair the reducer invariant in `replay/merge.py`,
preserve legitimate deferred tool output, and update the incident note in
`reports/incidents/transcript-merge.md`. Do not hide the symptom with a
render-layer duplicate filter.

## Deliverables

- reducer fix in `replay/merge.py`
- summary-preserving fix in `replay/incident_summary.py` if needed
- note update in `reports/incidents/transcript-merge.md`
- optional regression additions in `tests/*.py`

## Visible checks

- `python3 -m unittest tests.test_merge tests.test_render tests.test_incident_summary`

## Variant progression

- `v1-clean-baseline`: same-name tool outputs collide under the buggy reducer
- `v2-noisy-distractor`: benign duplicate debug lines tempt render filtering
- `v3-dirty-state`: abandoned patch suggests dropping all post-completion fragments
- `v4-multi-corpus-objective`: release context makes summary stability the primary objective
- `v5-recovery-in-thread`: rollback evidence proves symptom-only filtering already failed

## Saturation and renewal plan

If the family mean probe score stays above `80` for two rounds, renew with:

1. a variant where assistant fragments and tool fragments share sequence ranges
2. a variant where summary consumers ingest both rendered lines and merged-event counts

## Current Calibration Attestation

The latest whole-family live probe is already recorded in
`benchmark_run.md` attempt_02. That run showed this family is currently too
easy for the target §10.1 window on the underlying reducer/runtime repair:

- `family_mean = 72.67`
- `max_variant_mean = 86.67`
- `min_variant_mean = 65.00`
- monotonic failed at `V3 -> V4` and `V4 -> V5`

For this family, the current differentiator is still too note-heavy: several
V2-V5 failures came from the incident-note ceiling rather than a stronger
runtime miss. This documentation follow-up does not claim a new calibration
result and does not run a fresh probe. Until a later hardening pass proves
otherwise, treat `transcript-merge-regression` as an honest
frontier-easy / widening-candidate family, and prefer future changes that add
runtime or evidence-triage difficulty over more note-layer pressure.
