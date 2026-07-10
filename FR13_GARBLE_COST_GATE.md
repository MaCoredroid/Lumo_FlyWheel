# FR13 garble — compute-only (no-HBM) path EXHAUSTED → cost-gate

**Date:** 2026-07-10. Follows FR13_GARBLE_COMMITTER_CLEARED.md (committer proven correct →
garble = tree-verify forward drift = PATH A). This doc closes the compute-only search with a
rigorous negative result and states the decision.

## What was searched (workflow wf_7768bcbf-42e, 8 agents, adversarial verify)

Exhaustive 21-op map of the tree-verify SPINE forward (L0→logits), code-grounded against the
pinned vLLM image + qwen3.6-27b-fp8, each op classified for M-dependence + compute-only
fixability, then each compute-only candidate adversarially refuted vs {no-HBM, differential-vs-E5,
correct-to-M=1, real}.

## Result: NO compute-only lever survives

| candidate (compute-only-fixable) | verdict | why it dies |
|---|---|---|
| `in_proj_ba` GEMM pad-block | already baked | LUMO_FB_KERNEL_ROWS ON; **garble persisted with it on** |
| full-attn FA2 **QPAD** | **measured null** | already built: drove L31 full-attn 3.9e-3→0.0 **yet e2e garble stayed 24** — downstream amplifier, shared w/ E5 |
| MoE **router gate** pad-block | refuted 3× | native E5 runs same M-dep router at M>1 & doesn't garble (within-floor/shared); GDN mixer runs *before* router so input already drifted; in_proj_ba precedent |
| MoE shared_expert_gate | no-op | N=1, never measured M-dependent, shared code both arms |
| lm_head GEMM | non-differential | tail projection, shared, doesn't feed the recurrent state |

**Structural finding (the map):** the GDN **scan** and **conv** *ops* are ALREADY
M-invariant (per-sequence grid, one program per (seq,head); NPAD_INVARIANT + SCAN_ALIGN; conv
bit-exact, 0 int-view mismatches). So the L0 birth 1-ULP is **not** an M-dependent op at L0 —
it is the **conv/ssm-state BANK CONTENT**, a recurrent 1-ULP carry from an earlier forward.
Correcting resident state content = re-read/re-write = **HBM tax** (forbidden). The only no-HBM
route is fixing the upstream *writer*, and every writer is either already-padded (in_proj_ba) or
a shared/downstream amplifier (router, FA2 — the latter a **measured** null).

⇒ **Compute-only PATH A is exhausted** — established by a 21-op map + adversarial verify +
two direct prior measurements (QPAD e2e-null; in_proj_ba baked-yet-persists), not by assumption.

## The remaining fix is tree-reshape — a cost-gate

The garble is caused by co-resident **branch** rows perturbing the spine (M>1). Removing them
removes the carrier:

- **chain5 / spine-only (0 co-resident branch rows, M=5):** the SPINE_PERTURBATION carrier is
  **structurally absent** → existence proof (de-cascades raw-5→2 clear-margin flips, ≤ native's 3;
  live N=2 shows no garble signature). **Kills the garble** (caveat: mechanistic + N=2, not a
  direct garble-reproduction A/B).
  - **Cost:** per-forward speed is EQUAL to cat8 (~248 ms, weight-floor-dominated); the cost is
    lost accept/event — **~-7% (B=1) to ~-9% (B=4) committed-token throughput** (chain5 forgoes
    the tree's entire accept edge over native, reverting to ~native-linear speed while keeping the
    forked kernels). Sources: FR13_PAPER_RESULTS_AND_VERDICTS.md §2-4, FR13_B4_CACHE_MATRIX_RESULTS.md §2,
    FR13_PLUS2_DECASCADE.md, FR13_E5_VS_CAT9_SPINE_DRIFT.md §1d.
- **cat6root (1 root-sibling branch, M=6):** narrows branching but drift-effect UNMEASURED and
  accept confounded (cat6⊂cat8). Not a clean lever.

## Decision (user's call — speed vs correctness)

The garble is **already within the accepted lossless floor** (canonical bar: lossless accepted at
big-denom 13% honest vs E5; committer proven correct; intermittent ~13%-of-a-rare-novel-identifier,
low-margin-only). Fixing it further has **no cheap (no-HBM) path**. So:

- **Option A — accept within-floor (keep cat8 speed):** ship the tree's +10% edge; the garble is a
  documented within-floor forward-drift artifact. Recommended per "speed is the goal."
- **Option B — reshape to chain5 (kill garble):** pay ~7-9% throughput (revert to ~native speed,
  losing the tree's justification) for zero branch-drift garble.

Reshape is a runtime tree-shape switch (already supported) = an on-demand escape hatch either way.
