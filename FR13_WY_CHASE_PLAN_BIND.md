# FR13 WY / replay accept-only chase — PLAN (the kernel is already shipped)

Workflow `wf_1ddbb3a5-8b8` (CPU plan, 5 agents). Raw:
`research/fr13_workflows/wy_chase_plan_wf_1ddbb3a5.raw.json`. Adversarial verify
**holds=TRUE** (plan endorsed unchanged). HEAD fdf5ffa7.

## The big reframe
The "WY / accept-only / replay" kernel the user asked to chase is **NOT a new build** — it is
**already on HEAD, merged, and DEFAULT-ON** (`FR13_REPLAY_ROUTE=1` at every read site, d2a0ff51):
a **sequential rank-1 accept-only/no-copy tree-scan** (`_tree_gdn_kernel` with
`STORE_NODE_STATES=False`, `fr10_gdn_tree_kernel.py:387-543`) whose accepted-chain state is
published by a sibling **replay kernel** (`_tree_gdn_replay_kernel` :546-846) re-executing only
the accepted path from h0 via the shared `_gdn_node_step` body. **NOT chunked-WY** (WY is
prefill-only, ~6e-5 chunk-vs-recurrent gap, never bit-exact for verify; parked on
`fr13-wy-archive` c0448bd7 + 8a975837 as a last-resort PORT, not a cherry-pick).

The three names are ONE idea in stages, not three kernels:
**accept-only (fr13-accept-only-wip, parked, stage 1)** → **replay route (merged, default-ON,
stage 2, fixes accept-only's seams)** → **WY one-pass (different math, last-resort fallback)**.

## Gate-4 live-fail = RESOLVED (STOP+REPORT-class correction to the prior bind)
The gate-4 accept collapse (2.024→1.521, FR13_ACCEPT_ONLY_GATE4_FAIL_BIND f8ad5f92) was **NOT**
the deferred-publish `_FR10_PENDING_TREE_STATE_PUBLISH` ordering nor stale-rejected-row slots
(both were real design hazards the rebuilt route neutralizes). The boundary-trace (byte-level:
producer wrote row-3, consumer read all-zeros 4KB) pinned it to a **conv-remap PAGE-STOMP**:
vLLM builds conv (kv[0]) and ssm (kv[1]) banks as `as_strided` VIEWS over the SAME mamba page;
the frozen Triton `_remap_state_rows` used `state.stride(0)` as BOTH row-offset multiplier AND
per-row copy extent, so a conv-only remap dragged never-written node-column ssm bytes (boot
zeros / stale block leftovers) over the replay's just-published linear-column ssm state. FIX =
pure wiring (`fr13_replay_conv_remap.py:41-122` index_select gather + index_copy_ scatter
touching only the conv view), wired at patcher :1838-1846, commit **02b1627a**, page-safe both
regimes f4d971c1. Post-fix: determinism 4/4, token-identical to legacy, accept 2.08-2.15.
Lineage table FR13_GDN_KERNEL_LINEAGE.md:27 carries this.

## THE UNIFYING HYPOTHESIS IS MOSTLY REFUTED — replay is SPEED-ONLY (~80-85%)
Honest probability the WY/replay chase fixes lossless/accept: **~15-20%**; fixes speed only:
**~80-85%**. The cat10 −28 and the 22-flips are in channels the verify-path replay does not touch:
- **cat10 −28-accept = DRAFTER-side (S3).** The root sibling is added to the **drafter packing
  order ONLY** (`:9499-9508` "only the drafter packing order is touched"); the contamination is
  in the drafter GDN/conv state shared across the 2-row depth batches of the rollout, computed
  BEFORE any state commit (FR13_ACCEPTANCE_LADDER_BIND S3 :105-115). A verify replay can't reach
  it. (Root still UNBOUND between drafter-co-residency / BI-asymmetry / state-rebuild — the
  decisive discriminator = spine-only-drafter A/B, UNRUN.)
- **22-flips = committer serve-path / verify-wiring** (FR13_GOLD_MARGIN_BIND), NOT GDN state.
  CAVEAT/CONFLICT to resolve at the ladder: commit 0b5de164 said "CHANNEL 2 verify-forward gap,
  committer EXONERATED 0/944 ch1" while reader-3 calls it a committer row-mapping defect — the
  node-7 ladder + Gate-0 settle which.
- **Decisive on-disk proof:** post-page-stomp-fix the replay route gave accept/event 2.08-2.15
  == legacy all-rows-publish, **token-identical** → the state-publish change moves accept by ZERO.
- Replay's real value: **0.86× native HBM** (36→6 row-touches/layer), **spill-free at any tree
  width** (single register tile, no h_cache) = the scaling unlock for wide/suffix trees.

## THE PLAN (greenlit; 5-7 GPU boots, 2-workflow cap, within-floor bar, TRUE E5 not naive_mtp)
**GATE 0 = DECISIVE EARLY TEST (1 boot, run FIRST as go/no-go):** re-run cat10, 3 arms +
spine-only-drafter A/B, vs TRUE native E5 (`fr10_launch_speed_server.sh num_spec=5 FLASH`, NOT
naive_mtp):
- ARM1 `FR13_REPLAY_ROUTE=1 FR13_CAT10_ROOT_SIBLING=1` (accept-only + sibling)
- ARM2 `FR13_REPLAY_ROUTE=0 FR13_CAT10_ROOT_SIBLING=1` (legacy + sibling = the original −28 run)
- ARM3 `FR13_REPLAY_ROUTE=1 FR13_CAT10_ROOT_SIBLING=0` (= cat9 spine, replay on)
- **If ARM1 heals the −28 (depth≥1 counts climb toward cat9 371) + deep flips drop → unifying
  HOLDS (reopen justified) → full ladder Gates A-F. If ARM1==ARM2 (−28 persists, accept ~2.08-2.15)
  → unifying KILLED → demote replay to SPEED-ONLY, keep Gates E/F for speed.** (On-disk evidence
  predicts the KILL.)
- NOTE the original cat10 gate already produced ARM1 (cat10, replay default-ON) + ARM3 (cat9), so
  Gate-0 mainly adds ARM2 (replay-OFF+sibling) + TRUE E5 + the spine-only-drafter A/B.

Gates A-F (only if Gate-0 holds, within-floor not abs-0.0): A offline scan-equality + accepted-rows
torch_equal + regular-decode==pristine (already discharged 9d4d22e3, re-confirm); B LIVE B=1
publish-before-next-h0-read boundary probe (FR13_REPLAY_BOUNDARY two-tap, the instrument that
cracked the page-stomp — reuse, byte-equal at every (req_id,read_row)); C B=1 gold-margin argmax
vs the non-MTP oracle; D B=1 accept vs TRUE E5; E s/fwd + spill vs E5 (target sub-native 0.86×);
F B=4 captured corruption gate vs floor 0.113. STOP on first fail.

BANNED (route around, never propose): copy-recurrent multi-spine (NOT lossless CLOSED_NON_SHIP),
dense GDN substitute, reward-hack splice (FR13_HIDDEN_SUBSTITUTE = oracle-only, gate splice-OFF
with OUR kernel computing), chunked-WY for verify (prefill-only), naive_mtp as baseline,
aggregate decode_seconds for speed, per-stage tolerances.

## Separate tracks (the chase is NOT one kernel)
1. **SPEED/SCALING** = re-gate the already-shipped replay route as speed-only (Gates E/F) +
   confirm the 0.86× + spill-free width-scaling.
2. **ACCEPT (cat10 −28 / S3)** = the spine-only-drafter A/B discriminator → fix the DRAFTER
   co-residency (NOT the verify replay).
3. **22-FLIPS (lossless)** = the committer-row / verify-forward localization — the queued node-7
   per-sub-op ladder + the committer-row argmax gate (resolve the 0b5de164-vs-gold-margin conflict).
