# LLM-judge rubric — partial-progress sub-metric

**Model:** `gpt-5.4`  
**Temperature:** `0.0`  
**Reasoning effort:** `high`  
**Seeds:** 2 (average). Set via `CNB55_JUDGE_SEED_A=1` and `CNB55_JUDGE_SEED_B=2`.

You are grading a submitted manager brief against a gold brief. Score each dimension 0–2, then sum (max 10).

## Dimensions

### D1 — Quality of rejection reasoning for the hardest close call (0-2)

The hardest close call in each variant:

- V1: P4 vs P2.
- V2: P4 vs P2, and P5 vs P1.
- V3: P4 vs P2, and P3 being demoted despite the in-progress patch.
- V4: P2 vs P5, and P4 being demoted under the new objective.
- V5: P5 vs P4, and P2 being rejected despite having been the prior pick.

Score:

- 0 — The close call is not named, or the reasoning is tautological ("P4 is better because it wins").
- 1 — The close call is named and one dimension of trade-off is articulated.
- 2 — The close call is named, ≥ 2 trade-off dimensions are articulated, and the decision dimension is explicit.

### D2 — Coherence of risk + mitigation pairing for the accepted proposal (0-2)

Score:

- 0 — No primary risk given, or the mitigation does not address the risk.
- 1 — Primary risk is stated, mitigation is gestured at but not concrete.
- 2 — Primary risk is stated and the mitigation concretely addresses it (e.g., naming a specific gating / rollout / warmup mechanism).

### D3 — Honest handling of contradictory evidence (V2+; V1 = free 2) (0-2)

Score (V2+):

- 0 — Contradictory evidence (stale bench, ambiguous staffing) is not acknowledged.
- 1 — Acknowledged but used anyway without reflagging.
- 2 — Acknowledged, reflagged, and treated as a confidence reduction in the relevant rationale.

### D4 — Honest reframing under objective drift (V4+; V1–V3 = free 2) (0-2)

Score (V4+):

- 0 — Ranks against the old objective; does not notice the flip.
- 1 — Notices the flip but does not re-rank.
- 2 — Notices the flip, re-ranks, and explains why the former leader is now secondary.

### D5 — Incident anchoring (V5; V1–V4 = free 2) (0-2)

Score (V5):

- 0 — Does not reference INC-2481 or the rollback.
- 1 — References the incident but does not use it to re-rank.
- 2 — References INC-2481, uses it to reject the rolled-back proposal, and anchors the new accepted in incident-grounded reasoning.

## Output format

```
{
  "D1": <0|1|2>,
  "D2": <0|1|2>,
  "D3": <0|1|2>,
  "D4": <0|1|2>,
  "D5": <0|1|2>,
  "total": <sum>,
  "notes": "<one paragraph of rationale>"
}
```

## Stability

- Temperature 0, two-seed average, JSON-only output.
- Judge does not see the gold brief's rationales — only its accepted id, ranking, and list of permitted close-call names. This prevents the judge from rubber-stamping any brief that quotes the gold back.
