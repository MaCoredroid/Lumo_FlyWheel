# FR13 Method-A BI campaign — PARTIAL bind (stopped early by user decision, 2026-06-10)

**Campaign stopped after T1/T2/N1** (user: "stop measure and chase down cause we remeasure anyways") —
the discriminator was already answered; N2/Nn floor refinements get redone post-fix regardless.
Artifacts: `output/fr13_method_a_bi_campaign/{T1,T2,N1}/` (engagement asserts + cuda_graph_proof + probes per arm).
Prep commit (flag-gated enablement, inert by default): `0a0d4433`. Engagement VERIFIED per arm:
`VLLM_BATCH_INVARIANT=1` + `LUMO_BATCH_INVARIANT_VLLM=1` + `FR13_BI_TREE_ATTN=1` in container env,
patched batch_invariant.py, guard passed through boot + FULL CUDA capture, backend/0.82/eager=False asserted.

## Results (B=4, FULL-capture, seed 1313, temp 0.6)
| comparison | result |
|---|---|
| **native BI-on within-boot same-seed (N1a vs N1b)** | **3/4 prompts EXACT; 38/256 mismatch** (one prompt diverges at pos 25) |
| native BI-off same-seed floor (reference 234548Z) | 1/4 exact, bag-TV 0.113 |
| **tree BI-on within-boot same-seed (T1a vs T1b)** | **0/4 exact; lcp 1/11/11/57**; one finish-reason flip |
| tree BI-on cross-boot same-seed (T1 vs T2) | 0/4 exact; lcp 17/11/11/... |
| tree BI-on accept/event | **2.096 (P64) / 1.867 (P128)** — unmoved from the contaminated ~2.0 band |
| native BI-on accept/event | 3.161/3.926 (P64 a/b) / 3.682 (P128) |

## CASE CALL (monitor, on the partial data)
**CASE-2-TREE-SPECIFIC: the carrier is a tree-path channel that BI does not cover.**
- BI demonstrably works on native (1/4 → 3/4 exact): the BI-coverable numerics (bf16 lm_head, softmax,
  RMSNorm, cuBLAS split-k) were a real chunk of NATIVE's noise — but NOT the tree's.
- The tree stays grossly non-deterministic under full BI, and **diverges at lcp=1** (first generated
  token) within one boot: a tree-path op produces different logits for IDENTICAL input — no
  accept-history feedback can explain token-1 divergence. accept/event unmoved ⇒ same channel drives the loss.
- Native's small residual (38/256, one prompt @pos25) = the non-BI-covered shared remainder
  (un-swapped fp8 cutlass GEMM / scheduler timing) — secondary.

## What this unblocks
Native nearly deterministic under BI ⇒ background noise is now TINY ⇒ the tree's divergence is
isolated + the fixed-row localization that failed at e263a45b becomes feasible once the tree channel
is found. Next: CHASE-DOWN (read code → classify non-det ops → B=1 same-seed bisection: B=1 non-det
= per-op nondeterminism (duplicate-index index_copy_/scatter/atomic); B=1 det but B=4 not =
batch-composition/slot wiring).
