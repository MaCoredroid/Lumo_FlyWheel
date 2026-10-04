# Drafter-pass sweep (2026-10-02)

## Question

Does shrinking serial drafting improve end-to-end decode rate on the deployed LumoTree stack? The stack is Qwen3.8-27B NVFP4 with the hydra27_fixed32 tree, at batch 1.

## Why the proposed tree shapes were not run

Two shapes were proposed: 28 nodes at depth 7, and 16 nodes at depth 6.

- **Depth does not drive drafting cost.** The drafter runs a fixed 5 MTP passes, one per depth 1–5, at about 10 ms each. Depths 6–11 come from host-side Arctic suffix lookups that cost almost nothing.
- **A shallower tree saves no time but loses acceptance.** Depths 6–11 are worth about 1.08 accepted tokens per step, and depths 8–11 about 0.59.
- **Wider branching at depths 1–3 cannot be built today.** Fan-out 3 and 32 verify rows are pinned across the drafter, sampler and kernels. Children of interior runner-ups have no drafter source.
- **A 16-node tree still computes 32 rows.**

The lever actually tested was the number of MTP passes: 5 versus 3, using the existing FR14 suffix pass gate.

## Setup

- **Requests:** the 43-request tuning corpus, sampled at temperature 0.6, max 1024 tokens.
- **Vehicle:** the same image and deployed vehicle for every arm (split-K FA2 tier-B, fused top-k).
- **Timing:** per-step GPU timers.
- **Metric:** pooled decode rate (N − R) / (Σe2e − Σttft).

| Arm | Rate (tok/s) | vs T avg | Accepted per step | Step wall (ms) | Target (ms) | Drafter (ms) | Commit (ms) |
|---|---|---|---|---|---|---|---|
| T1 deployed hydra27, 5 passes | 28.78 | — | 4.469 | 189.1 | 112.0 | 49.7 | 20.8 |
| T2 deployed hydra27, 5 passes | 28.59 | — | 4.441 | 189.6 | 112.2 | 50.0 | 20.8 |
| G1 gate, pre-registered predicate (as shipped) | 26.51 | −7.6% | 3.950 | 185.8 | 111.8 | 46.8 | 20.8 |
| A1 gate, permissive (≈always 3 passes) (as shipped) | 24.52 | −14.5% | 3.197 | 170.8 | 111.9 | 31.7 | 20.6 |
| GF1 gate, pre-registered, **fixed** | 27.05 | −5.7% | 4.089 | 187.6 | 112.6 | 47.7 | 20.7 |
| AF1 always-3, **fixed** | 28.94 | +0.9% | 3.973 | 171.3 | 112.1 | 32.0 | 20.7 |
| M1 native MTP-5 (drift control) | 26.01 | — | 3.155 | — | — | — | — |

Survival by depth, depths 1–11:

| Arm | Survival |
|---|---|
| T1 | .96 .81 .64 .52 .43 .26 .22 .19 .17 .15 .14 |
| AF1 | .96 .80 .64 .35 .27 .21 .18 .16 .14 .13 .13 |
| GF1 | .95 .80 .62 .48 .38 .22 .17 .14 .13 .11 .10 |

## Defect found in the shipped gate

On a gated step, the depth-4/5 runner-up slots are filled with copies of their spine token. Under sampled tree acceptance, the walk picks among identical children roughly uniformly. The copies are leaves, so most paths that reach depth 4 dead-end there.

- **Evidence (passive trace of G1's predicate, 733 steps):**
  - the gate fired on 122 steps (16.6%);
  - the depth-4 node on those steps was spine 19, copy 30, copy 27;
  - 60 of the 122 gated steps stopped at exactly 4 accepted tokens.
- **Fix:** fill those slots with a never-sampled token (`<|vision_pad|>`), applied at container start. Ungated steps are unchanged.
- **Check after the fix:** every gated depth-4 path takes the spine (96 of 97; the other goes through the rank-1 rescue chain). Gated-step acceptance rose from 3.17 to 4.19 tokens per step.

## Verdict

- **No pass-count variant beats the deployed tree.**
- **AF1 (+0.9%):** within the run-to-run spread of the deployed arm (T1 vs T2 differ by 0.7%; earlier same-code runs span 28.4–29.4). Its 18 ms per-step saving (−9.5% step time) is cancelled by 11% lower acceptance.
- **GF1 (−5.7%):** the gate fires on easy steps, where the 5-pass tree already accepts long runs.
- **No configuration change is warranted;** hydra27 with 5 passes stays.
- **Not run:** the confirmation-set replicates, since no candidate won.

Every result is retained under `replay/tree-sw*`, `gatediag/` and `gatefix/`.
