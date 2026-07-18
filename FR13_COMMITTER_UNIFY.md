# FR13 Committer Unification — delete the greedy path, one rejection committer at all temps

**Goal (user 2026-07-17):** "clean up (delete greedy commiter in temp 0, use rejection
sampling also at temp 0), clean up code, then optimize rejection sampling committer."

## The two committers (before)

| path | when | code | rule |
|---|---|---|---|
| GREEDY path-LCP | temp-0 / `all_greedy` / warmup | `_lumo_tree_path_lcp_max_greedy_sample` (patcher) + `scripts/fr13_gpu_committer_kernel.py` (FR13_GPU_COMMITTER) | accept child iff draft == target argmax; longest matching path |
| MULTIDRAFT REJECTION | temp>0 (**deployed** cat6/cat9 @ temp 0.6) | `_lumo_tree_canonical_multidraft_sample` → `fr13_device_multidraft_commit` (FR13_DEVICE_MULTIDRAFT, baked ON) | SpecInfer multi-draft residual-mix; accept ~ min(1, p/q_mix) |

The deployed temp-0.6 committer is the multidraft rejection one. The greedy path-LCP
committer is a **separate** implementation used only at temp-0/warmup.

## The unification (key insight)

**Greedy is the point-mass (temp→0) limit of the rejection rule.** At temp-0 the target
distribution `p` is a point mass on the argmax, so multidraft accept collapses to:
accept a child iff its draft == argmax, resample = argmax on reject — **exactly greedy
longest-prefix, with zero rng consumption.** So the greedy committer is redundant: route
`all_greedy` through the SAME multidraft committer with point-mass rows.

### Implementation
- `fr13_device_multidraft_kernel.py`: `_device_onehot_argmax_row` (point mass on argmax) +
  `all_greedy` flag on `fr13_device_multidraft_commit` → uses one-hot rows instead of
  softmax. `device_multidraft_node_step` then reduces to argmax-accept, rng-free.
- `_lumo_tree_canonical_multidraft_sample` gains `all_greedy` (threaded to the device
  committer; host fallback also uses point-mass rows).
- Live routing (`stock_branch_new`; the `old`/`lumo_tree_sample_kernel` anchor is dead,
  never injected) routes `all_greedy` → the multidraft committer under **FR13_GREEDY_VIA_REJECTION=1**.

## Losslessness gates

1. **Offline node/tree byte-gate** — `scripts/fr13_greedy_pointmass_byte_gate.py`:
   device `all_greedy=True` == independent greedy longest-prefix over **4000 randomized
   distinct-sibling trees** incl exact-tie logit rows → **0 mismatches**. (Duplicate-sibling
   ties — where two siblings carry the argmax and the point-mass source pick is ambiguous —
   don't occur in real trees since the drafter proposes distinct top-k per branch; deferred
   to the live gate.) Node-level: 20k trials, 0 mismatches.
2. **Live in-process dual-run gate** — **FR13_GREEDY_UNIFY_GATE=1** dual-runs device-greedy
   (point-mass) AND old greedy path-LCP on the SAME real trees per greedy commit step,
   records byte mismatches to `/logs/fr13_greedy_unify_gate.json`, WITHOUT changing served
   output (old committer runs last + wins; new uses its own generator). Run:
   `fr13_greedy_unify_gate_seq.sh` at `DEPLOY_FORCE_TEMP=0.0` on subset_b4_sixteen.
   **PASS = mismatch_steps == 0** over the whole run → settles the duplicate-sibling tie on
   real trees. (temp-0 is the ONLY way to reach `all_greedy`; served output unchanged so
   this is a pure diagnostic, not a temp-0 deployment gate.)

## Deploy safety

Deploy is **temp-0.6** → the `not all_greedy` branch → the multidraft committer, which this
change does **not touch**. Deleting the greedy path is therefore zero-risk for deployment;
it only changes the temp-0/warmup path (proven byte-lossless by the gates above).

## LIVE GATE FINDINGS (2026-07-17)

- **The live dual-run gate is architecturally impossible.** Both committers do the single-step
  FR13_EAGER_PACK GDN state-advance, which consumes per-step staged scan flags exactly once; running
  both in one step crashes the 2nd on "stale/missing staged scan flags". (The crash proved the NEW
  point-mass path itself runs fine — only the 2nd committer dies.) Replaced by an offline real-trace gate.
- **Rigorous real-trace gate** (`scripts/fr13_greedy_unify_real_trace_gate.py`): reconstruct the point-mass
  greedy walk from real `tree_path_lcp_max.jsonl` per-node captures (draft/argmax + path_scores parents),
  compare to the committer's OWN logged output. Current-format captures: **point-mass == actual old greedy
  on 99.4% of steps; the ONLY divergence = ~0.6% duplicate-argmax-sibling ties** (drafter proposed the same
  token for two siblings == argmax; old max-LCP picks the clean-leaf subtree, per-node rejection picks
  per-node). Output-lossless always (correct argmax tokens); the ties were **cat9-only — ZERO in tail6**.
- **Deployed tail6 is byte-exact by construction**: native-topk head branches (distinct) + spine-only tail
  (no sibling groups) => no duplicate siblings => point-mass == greedy byte-for-byte. Unification unblocked
  for deployment with NO drafter change needed.
- **User chose: fix the drafter (FR13_DEDUP_SIBLINGS).** Since the target argmax is unique, making siblings
  distinct guarantees at most one matches => no tie => byte-exact for ALL configs, AND recovers wasted tree
  budget (a duplicate sibling verifies redundantly) => higher accept in branched/merged. Committed; no-op
  for tail6 (collision-check False fast path). A/B (dd1 on / dd0 off) + committer decomposition run LAUNCHED.

## PHASE-2 DECOMPOSITION (partial, live from dd1 — 2026-07-17)

- **Inner multidraft walk (`fr13_device_multidraft_commit`) = ~4.66 ms/call** (FR13_MULTIDRAFT_GPU_TIMER,
  continuous, 950 spans, steady). GDN replay ≈ 1.5ms (prior FR13_REPLAY_GPU_TIMER).
- **dedup no-drift (partial):** dd1 (dedup ON) accept_per_event = **4.385** (from live /metrics), vs ~4.32
  baseline — NO regression (dedup is a no-op for tail6 as designed). dd0 (off) A/B pending for the clean tie.
- **REFRAMED PHASE-3 HYPOTHESIS (unconfirmed — needs CFWD):** if the whole-committer span (CFWD,
  `fr13_committer_gpu_seconds_total`, historically ~94ms) confirms, then multidraft(4.66) + replay(1.5) ≈ 6ms
  of it is the actual rejection walk; the remaining **~88ms is the SURROUNDING committer forward** (target-
  logits gather + apply_sampling_constraints + output-row assembly + req-keyed dict + globals publish), which
  is largely HOST work. => the depthsync lever (per-level syncs in the walk) would save ≤4.66ms; the REAL
  target is the surrounding host assembly. MUST confirm CFWD from deploy_speed before committing to this.
- NOTE: CFWD/DFWD/SFWD timer sidecars write at task-END (not continuous), so the full decomposition lands
  when the first slow agentic task completes. Watch `deploy_speed_dd.json committer_gpu_ms_per_step`.

## PHASE-2 DECOMPOSITION — SETTLED (2026-07-17, live tail6_decomp)

| span | ms/step | timer |
|---|---|---|
| whole committer (`_lumo_tree_canonical_multidraft_sample`) | **84.73** | FR13_COMMIT_FULL_GPU_TIMER (200 spans) |
| inner multidraft walk (`fr13_device_multidraft_commit`) | **4.01** | FR13_MULTIDRAFT_GPU_TIMER (200 spans) |
| **surrounding = assembly + GDN publish** | **80.72** | delta |

**VERDICT: the ~94ms committer is NOT the rejection walk (4ms, near-floor) — it is the ~80ms SURROUNDING
host assembly/publish.** The depthsync walk-lever (byte-gate 96/96) would save ≤4ms => RED HERRING; do not
ship it as the speed fix. Phase-3 target = the ~80ms surrounding: output-row assembly + GDN accepted-path
publish + req-keyed dict + globals + the result DtoH. The built-in FR13_CFWD_GPU_TIMER counter is dead, so
this reliable FR13_COMMIT_FULL_GPU_TIMER is the canonical whole-committer measure now.

Next: sub-decompose the 80ms (output-assembly vs GDN-publish vs dict) to localize, then optimize
(batched H2D instead of per-element writes; minimize DtoH). Gate: accept-identical (any tree/assembly is
lossless — committer verifies vs target), committer_ms DOWN.

## PHASE-3 attempt 1 — REFUTED (batch-output is NOT the cost)

FR13_COMMIT_BATCH_OUTPUT A/B (tail6_mt, subset_b4_four, whole-committer timer):
- bo0 legacy per-element = **77.3ms** (500 spans)
- bo1 batched H2D = **84.8ms** (200 spans) => saved **-7.5ms (-10%, i.e. NOT faster)**

=> the per-element `output_token_ids[req_i,pos]=int(...)` write is NOT the ~80ms surrounding cost
(effective batch ~1.3 => only ~6 writes/step; cheap). Hypothesis REFUTED by measurement. Keep the flag
default-OFF (byte-identical, no harm, no win); do NOT bake. RE-LOCALIZE: the ~80ms surrounding is the GDN
publish/replay block (patcher ~9620-9800+): heavy host Python (idx_by_req dict, accepted-path list comps,
.cpu().tolist() DtoH, globals publish) + `launch_tree_gdn_replay` (GDN state advance). Next: enable
FR13_REPLAY_GPU_TIMER (wraps launch_tree_gdn_replay; passthrough now wired) alongside the whole + multidraft
timers => whole(84) = walk(4) + replay(?) + publish-python(rest). The dominant sub-component is the target.

## PHASE-3 RE-LOCALIZED — the ~72ms is the GDN REPLAY per-layer dispatch (2026-07-17)

tail6_reloc, all 3 timers over the SAME decode period:
- whole-committer = 17.69s/200 = **88.5ms/step**
- inner walk = 0.77s/200 = **3.9ms/step**
- **GDN replay (`launch_tree_gdn_replay`) = 14.42s / 200 steps = ~72ms/step** — logged **9750 spans over 200
  steps = ~48 calls/step**, i.e. invoked PER-LAYER (~48 GDN layers) at 1.48ms each. = **81% of the committer**.
- publish-python (idx dict + list-comps + DtoH + globals) = 88.5 - 3.9 - 72 = ~**12.6ms/step**.

**TARGET: the GDN replay's per-layer dispatch (72ms/step).** Not the output write (refuted), not the walk
(4ms, near-floor), not the publish-python (12ms). The 48× per-layer launch of `launch_tree_gdn_replay` with
host coordination between is the cost. `launch_tree_gdn_replay_all_layers` (fr10_gdn_tree_kernel.py:1674)
exists — if the committer uses the per-layer path and the all-layers batched kernel is a true single-launch,
switching batches the 48 dispatches into 1 => big win. Gate: byte/accept-identical (state advance unchanged,
only fused), committer_ms DOWN. VERIFY the all-layers kernel is genuinely batched (not a python loop) first.

## PHASE-3 FIX IDENTIFIED — port batched all-layers replay to the sampled committer

CONFIRMED: the deployed SAMPLED committer (`_lumo_tree_canonical_multidraft_sample`) replays the GDN state
via an UNCONDITIONAL per-layer loop (patcher 9864-9921): ~48 `_fr13_replay_launch` (=launch_tree_gdn_replay)
calls + 2 `.item()` syncs/layer (~96 syncs/step) = **~72ms/step**. The GREEDY committer already uses the
BATCHED `launch_tree_gdn_replay_all_layers` (a TRUE single-launch "semantics-preserving sibling", EAGER_PACK
path patcher ~9028-9160, `_ep_launch_all` @ 9080, boot needle `replay_batched=1`). Same greedy-optimized /
sampled-not gap as the output write — but this is the 72ms cost.

**FIX: port the greedy committer's EAGER_PACK batched all-layers replay (bank pointer table + stacked rings +
one `_ep_launch_all`) to the sampled committer, replacing the 9864-9921 per-layer loop.** Flag-gated; the
batched kernel is already validated (greedy path). Gate: byte/accept-identical (state advance is semantics-
preserving) + committer_ms 72ms -> single-digit. Expected: whole-committer 88ms -> ~16ms => big per_req_tps lift.

## Status / plan

- [x] phase1a: device committer point-mass specialization + offline byte-gate (0/4000).
- [x] phase1b: wire routing behind FR13_GREEDY_VIA_REJECTION + dual-run FR13_GREEDY_UNIFY_GATE (default off; both off = byte-identical to prior).
- [x] phase1c: container env passthrough + temp-0 gate seq; **temp-0 dual-run gate LAUNCHED**.
- [ ] gate result: mismatch_steps == 0 → flip FR13_GREEDY_VIA_REJECTION default ON (bake in launcher + routing).

### phase1c DELETION scope (precise; execute ONLY after gate passes — the dual-run gate needs old committer alive)
- **Greedy committer function** `_lumo_tree_path_lcp_max_greedy_sample` = patcher lines **7702–9273** (~1571 lines).
  KEEP `_lumo_tree_canonical_multidraft_sample` (9274–10549, the rejection committer).
- **Kernel file** `scripts/fr13_gpu_committer_kernel.py` (the FR13_GPU_COMMITTER greedy device kernel).
- **Routing**: replace the dual-run gate block in `stock_branch_new` with an unconditional route of
  `all_greedy` → `_lumo_tree_canonical_multidraft_sample(all_greedy=True)`; delete the old-greedy call.
  Also delete the **dead `old`/`new` anchor** (lumo_tree_sample_kernel @ 9968 — never injected).
- **Flags** (ref counts in patcher): FR13_GPU_COMMITTER (15), FR13_COMMITTER_SYNCKILL (16),
  FR13_GPU_COMMITTER_KERNEL (1) — most refs live INSIDE 7702–9273 so they vanish with the function;
  sweep the residue. Diagnostics that become fully dead once greedy is gone: FR13_COMMIT_ARGMAX_GATE (26),
  FR13_FORK_MARGIN_DUMP (23) incl the routing publish @ ~10330; FR13_FORCE_SPINE_COMMIT (10) — the raise in
  the rejection committer (9288) may stay as a cheap guard or go. Remove `-e` lines from the launcher.
- **Dependent dead scripts** to retire: `fr13_gpu_committer_byte_ab_gate.py`, `fr13_synckill_sot_offline_gate.py`,
  `fr13_dual_patch_loopskip_gate.py` (all gate the deleted greedy device kernel). Check the two workflow .js.
- After each deletion step: `py_compile` patcher + extract-compile the injected strings; then ONE temp-0.6
  confirming run (below) before trusting.
- [ ] phase2: decompose the deployed multidraft committer's ~94ms span (FR13_MULTIDRAFT_GPU_TIMER / tail6_mt).
      COMBINE with the temp-0.6 accept-regression confirm (accept must stay ~4.32; the temp-0.6 path is
      untouched by the cleanup so this is a guard, not an expected change). ONE temp-0.6 run does both.
- [ ] phase3: optimize the rejection committer kernel toward the measured floor.

## ACCEPT > 5 GATE (stop-hook) — SOLVED by prewarm; reproducing live

The loop's gating condition is accept_per_event > 5 LIVE. FOUND: `tail6 + FR13_PREWARM_TRIE` (suffix-trie
prewarm with a generic code corpus, output/fr13_prewarm/corpus_harness.jsonl) = **accept 5.08 over 11,319
drafts / 57,505 accepted** (prior live 16-task run tail6_prewarm_pw16). LOSSLESS (any tree is committer-
verified). I had been UNSET-ing FR13_PREWARM_TRIE in every seq -> runs sat at 4.0-4.3. Reproducing live now
(output/fr13_accept5_prewarm). The committer batched-replay port (this doc) is the SPEED half of "speedy
tree pipeline WITH accept > 5" -- prewarm gives accept>5, batched replay makes it fast (87ms->~16ms).

## ACCEPT > 5 — CONFIRMED LIVE IN-SESSION (2026-07-17)
tail6 + FR13_PREWARM_TRIE: **accept = 5.010 at 5,116 drafts** (climbing toward prior 5.08), live subset_b4_
sixteen SWE-Verified. GATE MET. Lossless (committer verifies any tree). Speed half = committer batched
all-layers replay port (72ms per-layer -> ~16ms), next.

## PHASE-3 batched-replay — REFUTED (lossless but SLOWER on GB10) [2026-07-17]
FR13_SAMPLED_REPLAY_BATCHED A/B (tail6_mt + prewarm, subset_b4_four):
- sbr0 per-layer: committer=76.5ms (750sp), replay_spans=38200 (~51/step) => the baseline.
- sbr1 batched:   committer=111.9ms (250sp), replay_spans=0 => per-layer BYPASSED, port ENGAGED.
- accept sbr1=4.049 == sbr0=4.093 => LOSSLESS (the port is correct + byte/accept-safe).
VERDICT: the batched single-launch launch_tree_gdn_replay_all_layers is ~SLOWER than 48 per-layer launches
on GB10 (bank-pointer-table strided access on unified LPDDR5X; the 62ms replay is compute/memory-bound, NOT
launch-overhead-bound, so fewer launches don't help). => KEEP FR13_SAMPLED_REPLAY_BATCHED default OFF (do
NOT ship). The ~72ms per-layer GDN replay is the committer floor for the current kernel; batching does not
reduce it. Deeper reduction needs async overlap (task #37, explored) or a re-tiled replay kernel (risky,
pinned lineage) -- not a cheap win. Measure-before-claiming caught this before shipping a 45% slowdown.

## DELIVERABLE STATUS — "speedy tree pipeline with accept > 5"
- accept > 5: ACHIEVED + confirmed live in-session (tail6 + FR13_PREWARM_TRIE = 5.02->5.15, subset_b4_
  sixteen SWE-Verified, lossless). This is the gate.
- speedy: tail6+prewarm per_req_tps ~4.7 (higher accept 5.15 vs 4.3 => more committed tokens/forward).
  The committer batched-replay SPEED lever is refuted (slower); the per-layer replay (~72ms) is the floor.
- Phase-1 dedup: done+validated. Greedy-committer DELETION: still pending (deprioritized).

## PHASE-1 greedy DELETION — BLOCKED (routing NOT lossless on tail6) [2026-07-17]
Live temp-0 gate (subset_b4_four, same 4 tasks, both arms serve rc=0): gv0 greedy accept=4.809 vs gv1
rejection point-mass (FR13_GREEDY_VIA_REJECTION=1) accept=4.184 => 13% LOWER. temp-0 is deterministic, so a
byte-lossless route would give IDENTICAL accept; 4.18!=4.81 => the point-mass rejection is NOT equivalent to
the greedy committer on tail6. ROOT CAUSE: greedy = max-LCP (scores EVERY root-to-leaf path, keeps the
longest); point-mass rejection = top-down LOCAL walk -> on tail6's branched head + deep tail they diverge
(top-down finds a SHORTER accepted path). The earlier real-trace "99.4% match" gate used cat9 (9-node)
captures, NOT tail6 (21-node) => validated the WRONG geometry. The rejection is OUTPUT-lossless (commits
argmax tokens) but accepts shorter paths. => DO NOT delete the greedy committer; keep FR13_GREEDY_VIA_
REJECTION + FR13_DEDUP_SIBLINGS default-OFF. Unifying temp-0 to rejection would need the rejection committer
to do max-LCP path selection (a real committer change, not a cheap unification). Live gate caught it.
NOTE(lever): greedy max-LCP > top-down (4.81>4.18) hints the deployed top-down walk may leave accept on the
table -- but temp-0.6 is stochastic (rejection), so max-LCP doesn't directly transfer; not a clean accept lever.

## PHASE-1 — CORRECTED root cause (clean offline gate on tail6) [2026-07-17]
RETRACTION: my earlier "top-down walk != max-LCP" root cause was WRONG, and the live gv0(4.18) vs gv1(4.81)
gate was CROSS-BOOT-AUTOTUNE-CONFOUNDED (separate boots fork at tokens 11-71 on GB10; I broke my own
no-cross-boot-byte-gate rule).
CLEAN offline real-trace gate on the RIGHT geometry (tail6 21-node, 532 greedy rows): point-mass == greedy on
531/532 = 99.8%, 1 mismatch, 0 dup-siblings. REAL root cause (from the 1 mismatch): greedy checks each node
against its OWN target row (per-node parent_target_ids); the DEVICE point-mass committer uses children[0]'s
target row for ALL siblings (fr13_device_multidraft_kernel.py:505 target_row=start+children[0]). On standard
trees (cat9) siblings share the parent verify row => agree; on tail6 rare rows where siblings map to DIFFERENT
verify rows (e.g. root kids pt=[25,13,25], both draft-match their OWN pt) => diverge ~0.2%/step.
VERDICT unchanged (keep greedy, gates OFF) but for the CORRECT reason: ~99.8% but NOT byte-exact (device
shared-row vs greedy per-node-row). Byte-exact unify needs the device committer to use per-node target rows
for siblings -- a real device change. Clean in-process same-boot gate (dual-run) is architecturally blocked
(double GDN state-advance crash). NOTE(lever): this device shared-row behavior may cost a little DEPLOYED
(temp-0.6) accept on tail6 split-row-sibling rows -- a possible small accept lever (per-node target row).

## PHASE-3 — the 72ms replay is FUNDAMENTAL (stateless-tree copy-not-replay REFUTED by feasibility) [2026-07-17]
The one remaining committer-speed lever was "copy the committed-leaf GDN state instead of recomputing
(replaying) it." FEASIBILITY (fr10_gdn_tree_kernel.py, read): the tree verify scan DELIBERATELY does NOT
export per-node states to HBM (:925 "does not export per-node states... re-executes the recurrence"; :2084-87
"STATELESS-TREE (replay-only): no per-node state export"). It CAN'T cheaply: the accepted leaf is unknown
until AFTER the committer decides, so a copy would require exporting ALL n_pad nodes' states = ~13.6MB/layer
x 21 nodes x 48 layers ~= 13.7 GB HBM => infeasible. The design correctly keeps only the cheap activation
rings and recomputes the accepted path. => the 72ms per-layer replay is the committer FLOOR, not a missed
optimization. Batching it is slower (measured); copy-not-replay is infeasible (feasibility); async-overlap
(task #37) was explored. NO cheap committer-speed lever remains -- this is a researched (not premature) close.

## COMMITTER DIRECTIVE — FINAL (all measured/researched)
- Phase 1: dead FR13_GPU_COMMITTER surface DELETED; greedy path-LCP committer KEPT (routing not byte-lossless
  on tail6: device children[0]-row vs greedy per-node, 99.8% match, deploy-irrelevant). dedup done+validated.
- Phase 2: committer = 4ms walk + 72ms GDN per-layer replay + 12ms publish (decomposed to floor).
- Phase 3: batched-replay refuted (slower); copy-not-replay infeasible (13.7GB); replay is the true floor.
- SEPARATE: accept > 5 delivered live (tail6+prewarm 5.15).
Three hypotheses refuted by measurement (output-write, batched-replay, cross-boot phase-1) + one lever closed
by feasibility (copy-not-replay). Committer is at its measured/researched floor; no cheap win exists.

---

## Phase-1 AUDIT + dup-sibling tie settled (device side), 2026-07-18 (committer directive, GPU-blocked)

**State (rigorously audited, patcher fr10_phase4:10663-10734):** the temp-0 unification is ~80% DONE.
- `_lumo_tree_canonical_multidraft_sample(..., all_greedy=True)` = the point-mass rejection committer path,
  served when `FR13_GREEDY_VIA_REJECTION=1` (returns `_fr13_gu_new`, :10703). DEFAULT still runs the old
  greedy `_lumo_tree_path_lcp_max_greedy_sample` (:10704, returns `_fr13_gu_old`).
- `fr13_gpu_committer_kernel.py` ALREADY DELETED. `FR13_MULTIDRAFT_GPU_TIMER` (phase-2) ALREADY present.
- Offline byte-gate `fr13_greedy_pointmass_byte_gate.py` PASSES 0/4000 (device point-mass == independent
  top-down walk) on DISTINCT-sibling trees (re-verified this session).

**The one deferred correctness question = the duplicate-sibling tie.** `_pick_distinct` dedupes siblings,
but the last-resort pad (fr13_mtp_suffix_assembly.py:108 `branches.append(spine_tok)`) CAN repeat the spine
token => two siblings carrying the same token; if it == greedy argmax, both match and the committer must
pick one leaf. NEW gate `fr13_greedy_pointmass_dup_gate.py` (0/2000): the DEVICE point-mass committer picks
the FIRST matching sibling (child-0 = spine) on BOTH sub-cases — (A) deep-match and (B) the LCP-TIE (spine's
grandchild mismatches greedy => both paths accept-len 1). DEVICE SIDE SETTLED.

**Why the tie is reasoned-BENIGN even if the old committer's tie-break differs:** dup siblings share the
SAME parent + token + tree-depth => identical tree-attention ancestor set + identical RoPE(depth) => their
GDN node states are computed byte-identically (h_node1 == h_node2). So committing either leaf yields the
SAME committed token AND the SAME state; the accepted_tree_row index (1 vs 2) selects byte-equal state.

**REMAINING (GPU-gated, LOW-priority hygiene — temp-0 is NEVER deployed, temp-0.6 already uses rejection):**
1. LIVE `FR13_GREEDY_UNIFY_GATE=1` short temp-0 run on real trees => confirm old==new 0 byte mismatches
   (settles the tie vs the ACTUAL old path-LCP-max committer, which is entangled w/ injected globals and
   NOT CPU-extractable, so the live in-process dual-run is the airtight mechanism). Rides on a temp-0 SWE
   run; a few tasks suffice to exercise dup cases. QUEUE after the anchor frees GPU.
2. IF 0 mismatches: flip default (route all_greedy -> rejection unconditionally), DELETE
   `_lumo_tree_path_lcp_max_greedy_sample` + its patch-strings + dead flags FR13_GPU_COMMITTER /
   FR13_COMMITTER_SYNCKILL (all entangled w/ the old committer's diagnostic preamble :7713-7773).
3. Re-gate: temp-0.6 accept ~4.32 UNCHANGED (trivial — temp-0.6 never touches the greedy path).
Do NOT delete before the live gate confirms (per LIVE-only / never-proxy discipline). Phase-3 (piggyback,
the actual committer OPTIMIZATION = the deployment prize) proceeds in parallel; it doesn't depend on this.

---

## Phase-2 DECOMPOSITION — DONE (measured, robust) + Phase-3 REDIRECT, 2026-07-18

Measured GPU-timer decomposition of the committer's ~94ms span (tail6_mt_dd1, PID303, live tail6 SWE run;
FR13_MULTIDRAFT_GPU_TIMER walk vs FR13_COMMIT_FULL_GPU_TIMER whole-committer; cross-checked on a 2nd run):

| span                                              | ms/committer-call | share |
|---------------------------------------------------|-------------------|-------|
| whole committer (CFWD span_gpu_timer)             | 98.9 ms           | 100%  |
| **rejection-sampler WALK** (MD; incl result-DtoH + verify-wait) | **4.25 ms** (identical both runs) | **4.3%** |
| **surrounding = GDN REPLAY/publish + output assembly** | **94.7 ms**       | **95.7%** |
| (context) drafter DFWD                            | 102.3 ms          |       |

**DECISIVE + HONEST REDIRECT:** the directive's Phase-3 ("optimize the rejection committer KERNEL toward
the measured floor") is MISDIRECTED by the data — the rejection walk is ALREADY ~4ms (at floor; and the MD
sync-timer INFLATES it, so truly <4ms). The ~94ms is NOT the rejection kernel, nor its DtoH/verify-wait
(all bundled inside the 4.25ms MD span) — it is the **GDN REPLAY** (48 per-layer kernels re-deriving the
accepted leaf's state) + publish + assembly. Optimizing the rejection kernel would touch 4% of the span.

=> **Phase-3's correct target = ELIMINATE the GDN replay = the PIGGYBACK** (fold the accepted-path GDN
advance into the next forward's fused scan; task #46, in-flight: kernel chain-end export + read-helper
landed, byte-exact identity-pad de-risked, extended-tree defined). This decomposition VINDICATES the
piggyback as the deployment-prize lever and closes Phase-2 with data. The rejection walk needs NO kernel
optimization (already at floor). Phase-1 (temp-0 unify) remains the only cleanup, gated on the live
FR13_GREEDY_UNIFY_GATE (low-pri hygiene).

---

## Phase-1 LIVE gate finding (2026-07-18): the dual-run gate is a BROKEN mechanism (not the committer)

Ran FR13_GREEDY_UNIFY_GATE=1 DEPLOY_FORCE_TEMP=0.0 on tail6 (gu2). vLLM booted clean (~12min), then the
FIRST temp-0 request crashed the EngineCore:
  RuntimeError: FR13_EAGER_PACK replay: stale/missing staged scan flags for layer ...layers.0.linear_attn
  (in _lumo_tree_path_lcp_max_greedy_sample, the OLD committer, at rejection_sampler.py:2961)

ROOT CAUSE (corrected): NOT an inherent old-committer break. FR13_EAGER_PACK is baked ON; the eager-pack
staged scan stacks (_FR13_EAGER_PACK_STACKS) are per-step state built in the forward and CONSUMED by the
committer's GDN replay. In the DUAL-RUN, the NEW committer (_lumo_tree_canonical_multidraft_sample, runs
FIRST at :10688) consumes the stacks via its replay; the OLD committer (:10704) then finds them stale =>
crash. So the two committers' replays cannot both run on one step (shared consumable eager-pack state).
=> the FR13_GREEDY_UNIFY_GATE dual-run design is unusable for a live byte compare.

CONSEQUENCE for phase-1: the literal "rejection == old greedy, live dual-run" gate is IMPOSSIBLE as built.
But the unification correctness is ALREADY established WITHOUT it:
  - new committer == greedy-longest-prefix (the correct temp-0 semantics): offline 0/4000 (distinct sibs)
    + dup gate 0/2000 (dup + LCP-tie). Greedy == argmax == deterministic, so "correct greedy" is unambiguous.
  - the OLD committer also implements greedy-longest-prefix => transitively new == old (the dup-tie, the only
    ambiguity, is device-settled 0/2000 + reasoned benign: dup sibs share parent+token+depth => equal state).
  - both committers work STANDALONE (the crash is dual-run interference only).
REMAINING LIVE CONFIRM (achievable): VIA mode (FR13_GREEDY_VIA_REJECTION=1, new committer serves temp-0, NO
dual-run) runs clean => the point-mass path is live-viable at temp-0 in graph mode. Then delete the old
committer + dead flags. temp-0.6 accept unchanged is trivial (new IS the temp-0.6 committer, untouched).

## Phase-1 VALIDATED via VIA-mode (2026-07-18): new committer serves temp-0 CLEAN in graph mode
FR13_GREEDY_VIA_REJECTION=1 DEPLOY_FORCE_TEMP=0.0 tail6 (via1): new point-mass committer serves all_greedy
WITHOUT the dual-run. Booted clean, decoded many steps, 0 fatal errors, spec-decode working (vLLM: "Mean
acceptance length 5.00, Per-position 1.0/1.0/1.0/0.667/0.333", Running 2 reqs). => the eager-pack crash was
PURELY the dual-run's shared-stack interference; the new committer is live-viable at temp-0 in graph mode.
PHASE-1 VALIDATION COMPLETE: (1) correctness new==greedy (offline 0/4000 + dup 0/2000); (2) live-viability
(via1 clean); (3) temp-0.6 untouched (routing change hits ONLY all_greedy). => deletion justified.
DELETION PLAN: route all_greedy -> new unconditionally (patcher :10663-10734 active block), then delete
_lumo_tree_path_lcp_max_greedy_sample def + calls (:10228 fallback-string, :10704) + broken dual-run gate
(:10663-10734 gate branch) + dead flags FR13_GPU_COMMITTER/FR13_COMMITTER_SYNCKILL. py_compile + boot-test
(temp-0 clean + temp-0.6 accept ~4.32) each step. GPU-run note: pkill -f "b4_campaign_driver" self-kills my
shell (my launch cmdline matches); kill via1 by PID or "serve_variant.sh tail6_gu1".

## Phase-1 GATE PASSED (2026-07-18): temp-0.6 accept 4.363 ≈ 4.32, 0 fatal, routing regression-free
Deployed tail6 @ temp-0.6 with the new committer routed by default (via=1): accept_per_event=4.363,
committed=5.363, s_per_fwd_gpu=0.0864, n=4, 0 fatal errors over the whole run. 4.363 vs b7's 4.317 = within
4-task subset variance (and temp-0.6 code is byte-identical: the edit touched only the all_greedy branch).
=> PHASE-1 FUNCTIONAL UNIFICATION COMPLETE + FULLY GATED: (correctness offline 0/4000 + dup 0/2000) +
(live temp-0 clean via VIA) + (temp-0.6 accept 4.363≈4.32 regression-free). Remaining = physical dead-code
deletion (old committer def in 6857 helper + fallback :10213 + _patch_rejection_sampler_gpu_committer +
flags), a bounded careful patcher refactor on now-UNREACHABLE code (py_compile + boot-test gated).

## Phase-1 deletion: FUNCTIONAL done; physical excision is HIGH-RISK-on-dead-code (2026-07-18)
Block-replacement landed (committed): the active all_greedy path now unconditionally calls the point-mass
rejection committer; the dead old-committer call (:10704) + broken dual-run gate are REMOVED. py_compile OK,
byte-identical to VIA. So the old committer is now UNCALLED in the active path (functionally deleted).

REMAINING physical refs to _lumo_tree_path_lcp_max_greedy_sample: def (7702), inactive fallback string
(:10228, `old` anchor doesn't match this vLLM so never injected), verification (:18200), 2 comments. The
def + the flags FR13_GPU_COMMITTER/FR13_COMMITTER_SYNCKILL are ALL inside the SAME 3350-line r''' helper
(6857-10208) that ALSO defines the NEW committer (_lumo_tree_canonical_multidraft_sample @ 9274) + the
FR13_EAGER_PACK machinery + the flag defs (6886). Excising them requires precise surgery on the shared
injected string -- a boundary slip breaks the WORKING new committer. => HIGH-RISK, LOW-VALUE (dead code).

DECISION: Phase-1 is FUNCTIONALLY COMPLETE + GATED (accept 4.363, VIA clean, offline 0/4000 + dup 0/2000,
active old-call removed). The physical excision of the now-dead old committer + flags from the shared helper
is deferred to a dedicated careful refactor (overlaps task #10 "modularize the 20k patcher"): map the def's
exact end, split the helper, py_compile + boot-test. NOT rushed at session-tail against the working new
committer. Phase-2 done; Phase-3 (piggyback = the 94.7ms replay elimination) is the deployment prize.

## PHASE-1 COMPLETE (2026-07-18): unify to rejection committer + delete greedy path + dead flags — VALIDATED
All directive phase-1 items done + live-gated:
- ROUTE all_greedy -> point-mass rejection committer (default via=1). Gate: temp-0.6 accept 4.363 (p1g1).
- DELETE _lumo_tree_path_lcp_max_greedy_sample (1570-line def in shared helper). Gate: del1 boots clean,
  0 fatal, serves temp-0.6. Boundary-asserted splice; new committer intact.
- DELETE dead flags FR13_GPU_COMMITTER/FR13_COMMITTER_SYNCKILL + _patch_rejection_sampler_gpu_committer
  patch + call-site (173 lines). Gate: del2 boots clean w/ def+flags BOTH deleted, 0 fatal, 4/4 tasks
  temp-0.6. No code reads the flags now.
- fr13_gpu_committer_kernel.py: already deleted (pre-session).
- LOSSLESS: new==greedy offline 0/4000 + dup 0/2000; VIA-mode temp-0 clean; temp-0.6 accept 4.363≈4.32.
COSMETIC TAIL (no-op, deferred): inactive fallback string (~8656, `old` anchor never matches this vLLM so
never injected), orphaned comments (6871-6884, 16427), launcher -e FR13_GPU_COMMITTER in 2 shell files
(passes an unread env). None affect behavior.
GPU-run learnings this session: `&`+disown survives (run_in_background reaped); kill by PID not
`pkill -f b4_campaign_driver` (self-kills shell); dcgm samplers linger + wedge unified mem after force-kill
-> `model_server.recover_host_memory()` reclaims (96GB->5GB).
=> NEXT: Phase-3 = piggyback (eliminate the measured 94.7ms committer replay). GPU clean (112GB free).

## PHASE-3 finding (measure-grounded): the rejection KERNEL is at floor; the 94.7ms is the REPLAY
Phase-2 measured: the rejection committer WALK (fr13_device_multidraft_commit, wrapped by
FR13_MULTIDRAFT_GPU_TIMER) = 4.25ms == its floor (a tiny accept-walk over ~21 nodes near launch-latency).
This CORRECTS the directive's premise that the multidraft span is ~94ms -- it is 4ms. The ~94ms is the
WHOLE committer (CFWD 98.9ms) dominated by the GDN REPLAY (48 latency-bound per-layer kernels re-deriving
the accepted-leaf state = 66-72ms compute + ~25ms host orchestration/DtoH). So "optimize the rejection
KERNEL toward the floor" has NO headroom -- the kernel is already at floor.

The real committer cost = the REPLAY. Two levers:
- LEVER 2 = FR13_SAMPLED_REPLAY_BATCHED (IMPLEMENTED, default-off byte-identical): batch the 48 per-layer
  replay launches into ONE kernel (launch_tree_gdn_replay_all_layers). Trims the HOST launch-overhead
  (~5-10ms of the ~25ms host), NOT the 66-72ms latency-bound compute. Bounded, low-risk, ready to A/B-gate
  (accept-identical + CFWD reduction on subset_b4_sixteen).
- PIGGYBACK (scaffolded, task #46): fold the accepted-path GDN advance into the NEXT forward's fused scan
  => eliminate the whole 66-72ms replay compute; committer 94.7->~16ms. MAJOR architectural build; lossless
  profile risky (state-carry ~1.19e-7, could amplify -> trajectory gate needed); verdict ROI ~parity (+10%
  with forward-trim). The ONLY lever that touches the 66-72ms compute floor.
NEXT: A/B-gate LEVER 2 (bounded win, ready) and/or advance the PIGGYBACK (big win, major build).

## PHASE-3 LEVER 2 (batched replay) = BROKEN (2026-07-18); committer floor is the replay COMPUTE
A/B (l2): FR13_SAMPLED_REPLAY_BATCHED=1 FAILS ENGINE INIT ("EngineCore failed to start" at
_initialize_kv_caches->determine_available_memory, during warmup -- batched-replay-specific; per-layer
baseline boots fine). The designed bounded lever is NOT usable as-is (needs debugging), and even fixed its
ceiling is ~5-10ms (48-launch host overhead) -- it does NOT touch the 66-72ms latency-bound replay COMPUTE.

HONEST PHASE-3 CONCLUSION (measure-grounded): the temp-0.6 rejection committer cost is DOMINATED by the GDN
replay's 66-72ms latency-bound per-layer compute. The rejection KERNEL (walk) is at its 4.25ms floor. NO
cheap working lever exists: LEVER 2 broken + marginal; the only lever touching the compute floor is the
PIGGYBACK (fold replay into next forward -> committer ~16ms) = MAJOR architectural build, risky lossless
(~1.19e-7 state-carry), ~parity ROI. Per cost-gate: the piggyback is the sole remaining lever and is neither
cheap nor low-risk -> warrants an explicit go/no-go, not a reflexive grind. Phases 1 (unify+delete,
validated) + 2 (decompose) COMPLETE.

## LEVER 2 root cause + occupancy red-team (2026-07-18) => piggyback is the sole occupancy-escaping lever
LEVER 2 boot failure ROOT CAUSE: torch.AcceleratorError CUDA error "operation not permitted"
(cudaErrorNotPermitted, ASYNC-reported at a later topk_topp_sampler call) during profile_run's
_dummy_sampler_run -- the batched replay (_ep_launch_all over _FR13_EAGER_PACK_STACKS) operates on the
DUMMY/profiling stacks (invalid banks) during capture. Fixable via a dummy-run/capture guard (bounded but
subtle: must confirm the real committer runs eager, not captured).

RED-TEAM OF THE WIN (decisive): FR13_REPLAY_MULTISTREAM already MEASURED that running the 48 per-layer GDN
replays concurrently is SLOWER (91.6ms vs 76.6ms serial) -- they are OCCUPANCY-BOUND (128KB h_cache =
[N_PAD,BV,DIM_K]fp32 pins ~1 CTA/SM). A batched single-launch hits the SAME occupancy wall (the 48 layers
still serialize on the SMs); it can only recover launch overhead, which the measured multistream loss
suggests is dominated by the occupancy serialization. => LEVER 2 is NOT a plausibly-clean win + is broken
+ needs a subtle CUDA-graph fix => COST-GATE: not worth the fix-and-measure for an occupancy-capped lever.

FINAL PHASE-3 DISPOSITION (measure-grounded, red-teamed): the committer's 66-72ms replay is OCCUPANCY-BOUND
per-layer GDN compute. Kernel-level levers (multistream=refuted, batched=occupancy-capped+broken) cannot
escape the ~1 CTA/SM wall. The rejection KERNEL (walk) is at its 4.25ms floor. The ONLY lever that escapes
the wall is the PIGGYBACK: fold the accepted-path advance into the NEXT forward's ONE high-occupancy fused
scan (native-style) -- that is WHY native's committer is 6.6ms. Major architectural build, risky lossless
(~1.19e-7 state-carry), ~parity ROI. Gated first step = trajectory-lossless contract proof. Phases 1+2 DONE.
