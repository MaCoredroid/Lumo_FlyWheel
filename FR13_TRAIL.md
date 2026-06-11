# FR-13 TRAIL — the critical path to the goal (canonical; monitor checks every tick)

**GOAL:** B=4 remains the final CUDA-captured SWE-Verified endpoint: e2e lossless (within native self-floor + per-depth argmax + temp-0.6 distributional) + superset accept ≥ same-shape native + wall TPS ≥ native E5. Active pivot (user 2026-06-11): stop the B=4 chase-down until B=1 is both lossless and near-native in forward speed.

**Quality/lossless invariant (user 2026-06-11):**
- Ground truth quality: pure target-model decode, no MTP.
- Deployed reference: native MTP-5, which verifies a single spine.
- Current object: a 9-node caterpillar tree verifier from the MTP head, with one branch off each spine position.
- Lossless claim: the tree verifier/committer preserves the underlying model's output distribution, not merely that it serves or accepts more tokens.
- Superset claim: because the tree contains the native MTP spine plus branches, a correct verifier should match native-spine quality and accept at least native MTP-5, with branches adding opportunity.

**Monitor contract (user 2026-06-11):** keep this trail on track autonomously; escalate to the user ONLY when (a) the trail goes off-track AND the monitor cannot decide as a researcher, (b) a step's pass/fail close, (c) the marked USER-DECISION points, (d) lineage-table factual changes.

| # | step | success criterion | owner of the call | status |
|---|---|---|---|---|
| P0 | freeze B=4 chase-down | B=4 handoff crash fixed and Step 3 failure/speed evidence banked; no further B=4 diagnostics until B=1 clears | monitor | **pivot bound** in `FR13_B1_SPEED_LOSSLESS_PIVOT.md`; speed evidence in `FR13_STEP3_SPEED_FORENSICS.md` |
| B1-1 | clean current B=1 speed+lossless gate | replay-on tree vs native MTP-5 at `MAX_NUM_SEQS=1`, BI=0, METRICS=0, FULL capture proven, paired prompts/seeds; lossless bar = latest B=1 chasedown S1/S2 classification plus temp-0.6 floor after greedy is clean; report accept/event, warm TPS, and `/metrics` s/fwd | monitor/subagent | **bound fail** in `FR13_B1_CURRENT_GATE_BIND.md`: tree/native s/fwd `1.41x`, accept/event `2.1515` vs `3.1613`, S1 clean, served forks not S2-classified |
| B1-2 | fix B=1 lossless blockers | S1 stays healed (`bonus_violations=0`, true-spine diagnostics); no S2/gross verify-forward corruption or served-stream fork outside accepted native/cross-boot floor; S3 tracked separately as accept/superset blocker | monitor/subagent | pending B1-1 |
| B1-3 | fix B=1 speed blocker | tree/native per-forward ratio near `1.0x`; a `1.1x`-class result is not enough without explicit break-even proof | monitor/subagent | pending B1-1/B1-2 |
| B4-1 | return to B=4 lossless | after B=1 lossless+speed clears, rerun B=4 lossless/superset gate against native E5 K=5 | **USER pass/fail** | frozen |

Step 3 speed-forensics note: `FR13_STEP3_SPEED_FORENSICS.md` separates current replay-on per-forward tax from the old replay-off HBM state-materialization story.

**Known risks to the trail (named, with the planned response):**
- R-A (the real research risk): temp-0.6 accept margin vs native is plausible-but-thin (+5-12% conditional). If step 4 lands BELOW native and the root-sibling topology (+0.08-0.15) + forward-drift recovery don't close it → ESCALATE with the measured margin (decision: accept parity-class vs drafter work).
- R-B: step-3 gate fails post-fixes → localize with the deterministic-substrate instruments (playbook); only escalate if a NEW bug class with no discriminator emerges.
- R-C: replay re-gate fails again → one more boundary-trace iteration; if the actor is again outside our code (native machinery) → escalate (may need an upstream-vLLM decision).
- R-D: environment (disk/mem/reboot) → recover per playbook rule 6, never silently degrade measurements.

**Off-trail definition:** any work not advancing steps 1-6 (except banked-lesson maintenance), any gate run on a contaminated regime presented as a verdict, any reward-hack-adjacent shortcut, or >1 day stuck on one step without a named discriminator.
