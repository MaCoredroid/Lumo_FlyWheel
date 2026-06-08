# FR-13 → codex_fr17 handoff: grind the WY scan STATE write bit-exact (the located fix)

codex_fr16 stood down (1h15m marathon turn, 64M-token context) after the root cause was located. Fresh session, clean context, ONE focused task.

## THE LOCATED ROOT CAUSE (grounded, decisive — see FR13_COMMITTER_TOPOLOGY_REDTEAM.md + FR13_WY_B4_REDTEAM.md)
The WY tree-verify e2e gives accept/event **1.199** (vs E5 3.076, 56% step-0 reject) because the **verify FORWARD is LOSSY** — specifically the **WY GDN scan STATE write drifts ~1.66e-3** and compounds recurrently over 64 layers to **final-logit max_abs 3.32**, flipping argmaxes over the live run.
- **NOT committer/topology** (both RULED OUT, grounded: topology = correct caterpillar `[-1,0,0,1,1,3,3,5,5]`, spine byte-identical to native; committer = native-identical rules — it only rejects because the drifted verify gives the draft p=0).
- **NOT batch-dependence** (the WY scan is batch-invariant: no `pid_batch`, no autotune, static-range, per-element slice — `FR13_WY_B4_REDTEAM.md`).
- **The blind spot we fixed:** the bf16-tap work drove the scan **OUTPUT** to 1 bf16 ULP (`1f0c7237`/`43434f01`/`1f11b9d7`), but the scan **STATE** (the recurrent carry) was never confirmed — it is **1.66e-3 (~13× the output)** and is what compounds. `Gate A NEVER passed` (always ~3.32 final logits); the "argmax-lossless 6/6" was a single greedy-event coincidence.
- Trace discriminator: conditioned on root-match, depth-1 spine match is only **0.635** (would be ~1.0 if the verify were native-exact) → forward drift, confirmed.

## THE TASK (the only one): drive the WY scan STATE bit-exact to native FLA
In `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` `_tree_gdn_wy_kernel`, make the **state-update / state-write path** byte-exact (within bf16 self-noise) to native FLA's materialized state surface (`fused_sigmoid_gating_delta_rule_update` / `chunk_delta_h.py` state store, `/tmp/vllm_live_019/.../fla/ops/`). Focus: the **`#4/#5` taps + the per-node outer-product accumulation ORDER** in the state write (native accumulates the outer products in fp32 and rounds the state ONCE at the chunk store; check WY's order/rounding of the state, distinct from the output readout which is already at floor). Gate behind `FLA_BF16_BOUNDARIES` so the fp32-oracle path stays.

## METHOD (fast, per the monitor's standing rules)
1. **Offline first:** capture the WY scan STATE vs native FLA state on a fixed payload (reuse `output/fr13_wy_l1_payload_*` + the replay gate), drive the **state** max_abs from 1.66e-3 → bf16 floor (~6e-5). Seconds/iter, no server boots.
2. **Then ONE live ladder** (B=1 then B=4, eager, FR12 capture hooks gated OFF per `ced25bd3`): strict top-down input→every layer→**final logits = within floor**, **spine AND branch** (4 branch-path oracles on leaf paths `[0,1,3]`/`[0,1,2,5]`/`[0,1,2,4,7]`/`[0,1,2,4,6,9]`, per-node argmax). Bind to FR13_LADDER_LOG.md.
3. **Then re-e2e** (clean: FR10_METRICS=0, diagnostics off, B=4 CUDA-graph): vs E5 `output/fr10_native_mtp5_same8_20260604T210257Z` — bag-TV ≤ floor + accept/event ≥ 3.21 (should recover once the state is lossless) + TPS ≥ native. Report the E5-vs-TREE table; do NOT self-declare PASS.

## DISCIPLINE (standing, user)
ONE GPU (no concurrent docker --gpus; relaunch WITHOUT --rm; `recover_host_memory` between arms — forked-FA2 exit wedges ~90GB). NO copy / state-copy / reroute / splice / dense — OUR kernel computes, verify vs native-on-path oracle. Gate-2 (regular decode == pristine) must stay 0.0. `FR10_ALLOW_LINEAR_FALLBACK` runs are diagnostic-only, never bound. Commit+push+bind EVERY step (in HEAD AND pushed). Report at the deliverable (lossless + accept/event ≥ E5 + TPS ≥ native) or a genuine wall. The monitor runs parallel CPU workflows to read code + cheap-verify ahead of you.
