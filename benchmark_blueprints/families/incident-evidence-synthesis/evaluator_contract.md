# `incident-evidence-synthesis` Evaluator Contract

## Evaluation Goal
Score whether a solver can synthesize a reliable incident packet from a frozen evidence pack.

## Visible Checks
- `pytest -q tests/test_incident_packet.py`

## Trusted Checks
- Packet identifies trigger and failed guardrail correctly.
- Follow-up action is bounded and evidence-backed.

## Hidden Checks
- Distinguishes symptom evidence from causal evidence.
- Preserves unresolved ambiguity where the corpus is genuinely incomplete.
- Rejects stale remediation notes as definitive proof.

## 100-Point Breakdown
- `10`: visible packet-shape checks pass.
- `25`: triggering condition correctness.
- `25`: guardrail and follow-up correctness.
- `20`: evidence synthesis quality.
- `20`: ambiguity handling and stale-note rejection.
