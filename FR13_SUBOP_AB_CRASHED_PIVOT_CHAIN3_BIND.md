# FR13 — L0-GDN sub-op A/B INCONCLUSIVE (engine crashed flag-ON); pivot to the chain3 floor-probe (empirical depth-model test)

Date 2026-06-14. GPU workflow `wf_e3fc5285-6f7` (task w68z6gxgy), **verdict holds=False, carrierSubOp
UNRESOLVED**. Raw: `research/fr13_workflows/gdn_subop_ab_crash_wf_e3fc5285.raw.json`.

## What happened
The FR13_GDN_SUBOP_MAB hook (decoherence-free in-process A/B: conv_state snapshot + h0 clone per arm,
captures conv1d_out + scan_out at M10/M5/M1 on the deep-spine carrier row) was built sound and committed
default-OFF (15ecad72, 7/7 CPU wiring tests, purely additive 0-deletion, flag-OFF no-op verified). BUT the
**flag-ON path CRASHES the engine with a CUDA device-side assert on the first cat9 tree-verify forward**,
reproduced on BOTH boots (run2 CUDA_LAUNCH_BLOCKING=1 confirmed the hook is the first-faulter). Root: the
M5/M1 reduced-row arm geometry in `_scan_arm` (L1465-1485) / `_conv_arm` passes the deep node's prior state
with an invalid bank / wrong reduced-row layout. **ZERO JSONL records → co-residency-vs-depth UNRESOLVED.**
Hook is FIXABLE (verdict nextAction: pass the deep node's prior state as a 1-row initial_state at bos=0 with
a valid-bank state) but **PARKED as fragile** — it has burned ~3 boots + 40 min with no numbers.

## Why pivot to chain3 instead of re-fixing the A/B
The A/B's only remaining unique value was the mechanistic co-residency-vs-depth split (does M10-spine ≠
M5-spine on the deep row?). That question is answerable EMPIRICALLY in the DEPLOYABLE context with PROVEN
instruments, no fragile in-process re-run:
- **chain3** `[(0,),(0,0),(0,0,0)]` (pure depth-3, no width) → flip count. If ~3, the depth model holds AND
  there is no constant-position floor; if ~5 (= chain5's level), flips crystallize at fixed positions
  (L60/L61) regardless of depth → the depth lever is dead (FR13_RESHAPE_DEPTH_MODEL_BIND issue #1).
- **cat3w** `[(0,),(1,),(0,0),(0,1),(0,0,0)]` (depth-3 + root/d1 width) → flip + accept. If cat3w flips ≈
  chain3 flips, width adds NO co-residency (depth-dominated → the deployable answer to the A/B's question);
  if cat3w > chain3, width re-introduces co-residency flips → prefer sparse confidence-gated width.
This is more decisive for the GOAL (cat9 22 → native 3 via reshape) than the A/B proxy, and uses
fr13_oracle_stream_teacher_force.py (the binding each-vs-own-oracle flip count) + the shape-gate driver.

## Gating notes carried into the chain3 boot (from FR13_RESHAPE_DEPTH_MODEL_BIND)
- chain3/cat3w need a ~15-30 line drafter packing branch each (exact-match dispatch, NOT TREE-env-only),
  flag-gated/default-cat9-preserving — copy the cat9/chain5/cat10 packing pattern. NEVER enable
  FR10_ALLOW_LINEAR_FALLBACK (banned). Downstream (masks/committer/replay/conv-fusion) auto-adapts off
  tree_choices; only the drafter packing is hand-rolled.
- Class-8 within-boot determinism [T,T,T,T] + class-9 engagement (tok/draft==3 for chain3, ==5 for cat3w)
  are FIRST gates — fail loud, no vacuous number.
- FLIP count (each-vs-own no-spec decode oracle, thr 1.0 nat) is the COMPARABLE lossless metric vs native 3
  / chain5 5 / cat9 22. accept/event is the SPEED metric and is class-12 cross-boot confounded (the "≥3.16"
  target is a shifting baseline) → report with caveat; the lossless flip bar is the primary GOAL.

## NEXT
chain3 floor-probe (build packing + boot + flip/accept) → conditional cat3w. Verdict: depth model holds
(reshape reaches native) or constant-position floor (depth dead → only scan-align/WY, parked, would reach
native). Pairs with [[project_fr13_22flip_carrier_l0gdn]], [[project_fr13_tree_reshape_unifying_lever]],
[[reference_scalar_metric_per_token_blindspot]], [[feedback_kill_wrong_gpu_task_immediately]],
[[feedback_check_artifact_before_concluding]].
