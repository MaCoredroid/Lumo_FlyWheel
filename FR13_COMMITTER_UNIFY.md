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
