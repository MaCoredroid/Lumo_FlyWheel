# Upgrade Evidence Pack

Use this skill when a task requires an evidence-backed dependency upgrade recommendation.

## Workflow
1. Gather external release and migration evidence.
2. Cross-check internal call sites and past incidents.
3. Build a risk matrix before choosing upgrade now, defer, or stage.
4. Reject stale risks that were reverted or mitigated already.

## Anti-Patterns
- Treating CVE presence as automatic upgrade-now.
- Ignoring internal call-site exposure.
- Copying changelog bullets without a decision.

## Done Signal
- The final pack contains a correct recommendation, evidence matrix, and bounded execution note.
