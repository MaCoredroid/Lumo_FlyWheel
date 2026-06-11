# FR-13 TRAIL — the critical path to the goal (canonical; monitor checks every tick)

**GOAL:** B=4, CUDA-captured, SWE-Verified: e2e lossless (within native self-floor + per-depth argmax + temp-0.6 distributional) + superset accept ≥ same-shape native + wall TPS ≥ native E5. Prototype purpose: drafter-agnostic fast+lossless tree verifier for MTP⊕suffix fusion.

**Quality/lossless invariant (user 2026-06-11):**
- Ground truth quality: pure target-model decode, no MTP.
- Deployed reference: native MTP-5, which verifies a single spine.
- Current object: a 9-node caterpillar tree verifier from the MTP head, with one branch off each spine position.
- Lossless claim: the tree verifier/committer preserves the underlying model's output distribution, not merely that it serves or accepts more tokens.
- Superset claim: because the tree contains the native MTP spine plus branches, a correct verifier should match native-spine quality and accept at least native MTP-5, with branches adding opportunity.

**Monitor contract (user 2026-06-11):** keep this trail on track autonomously; escalate to the user ONLY when (a) the trail goes off-track AND the monitor cannot decide as a researcher, (b) a step's pass/fail close, (c) the marked USER-DECISION points, (d) lineage-table factual changes.

| # | step | success criterion | owner of the call | status |
|---|---|---|---|---|
| 1 | replay wiring fix re-gate (page-stomp stride fix) | B=1 same-seed 4/4 bit-identical (eager AND captured) + accept ≥ legacy 2.02 + durable diff=0 on spine-no-flip events | monitor (gate is mechanical) | 🔶 in flight (wq10ibu3t fix phase) |
| 2 | merge `fr13-replay-route` to main | step-1 green + flow verify holds | **USER DECISION** | pending |
| 3 | B=4 3-arm corruption gate on the FIXED build | bag-TV ≤ 0.113 floor + real-loss ≤ 0.05 + accept within slack of same-shape **E5 (K=5) native — NOT K=9** (native+noise arms MUST boot `qwen3_5_mtp num_speculative_tokens=5`; assert draft_tokens/drafts==5; the K=9 arm already measured is context only) | pass/fail = **USER**; running it = monitor | **FAIL bound** in `FR13_STEP3_POST_HANDOFF_BIND.md` (post-fix replay-on tree completed; reducer `valid=true`, `verdict=FAIL`) |
| 4 | healthy-p2 paired capture (temp 0.6, post-fix) | measures the branch bonus; accept/event vs same-shape native | monitor measures; interpretation to user if thin | the superset-accept verdict |
| 5 | speed: topology-cache quick win + deployment-regime measurement (B=4, BI=0, METRICS=0, FULL capture incl. replay, SWE workload, per FR13_SPEED_TAX_BASELINE spec + overhead protocol) | measured per-forward ratio + wall TPS vs native E5 | monitor measures | the TPS verdict |
| 6 | FINAL: B=4 CUDA SWE-Verified e2e vs E5 | within-floor + accept ≥ native + TPS ≥ native | **USER pass/fail** | the goal |

**Known risks to the trail (named, with the planned response):**
- R-A (the real research risk): temp-0.6 accept margin vs native is plausible-but-thin (+5-12% conditional). If step 4 lands BELOW native and the root-sibling topology (+0.08-0.15) + forward-drift recovery don't close it → ESCALATE with the measured margin (decision: accept parity-class vs drafter work).
- R-B: step-3 gate fails post-fixes → localize with the deterministic-substrate instruments (playbook); only escalate if a NEW bug class with no discriminator emerges.
- R-C: replay re-gate fails again → one more boundary-trace iteration; if the actor is again outside our code (native machinery) → escalate (may need an upstream-vLLM decision).
- R-D: environment (disk/mem/reboot) → recover per playbook rule 6, never silently degrade measurements.

**Off-trail definition:** any work not advancing steps 1-6 (except banked-lesson maintenance), any gate run on a contaminated regime presented as a verdict, any reward-hack-adjacent shortcut, or >1 day stuck on one step without a named discriminator.
