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

## Status / plan

- [x] phase1a: device committer point-mass specialization + offline byte-gate (0/4000).
- [x] phase1b: wire routing behind FR13_GREEDY_VIA_REJECTION + dual-run FR13_GREEDY_UNIFY_GATE (default off; both off = byte-identical to prior).
- [x] phase1c: container env passthrough + temp-0 gate seq; **temp-0 dual-run gate LAUNCHED**.
- [ ] gate result: mismatch_steps == 0 → flip FR13_GREEDY_VIA_REJECTION default ON.
- [ ] DELETE `_lumo_tree_path_lcp_max_greedy_sample`, `scripts/fr13_gpu_committer_kernel.py`,
      dead flags FR13_GPU_COMMITTER / FR13_COMMITTER_SYNCKILL / dead `old`/`new` routing anchor.
- [ ] phase2: decompose the deployed multidraft committer's ~94ms span (FR13_MULTIDRAFT_GPU_TIMER / tail6_mt).
- [ ] phase3: optimize the rejection committer kernel toward the measured floor.
