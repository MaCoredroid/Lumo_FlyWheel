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
