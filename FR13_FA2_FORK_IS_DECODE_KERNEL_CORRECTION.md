# FR13 — CORRECTION: the FA2-fork IS the deployed decode kernel (0.00195 is moot)

Date 2026-06-14. From the FA2-tree-decode feasibility workflow (wkaexrv30, FR13_FA2_TREE_DECODE_FEASIBILITY.md)
+ my direct code verification. **This CORRECTS a wrong "code-confirmed" claim** made in
FR13_TOTAL_DRIFT_REANALYSIS_LEADS_BIND.md, FR13_PIVOT_REPLAY_FA2_FRONTS_BIND.md, the reanalysis v1/v2 docs
(473cbda0/e69f3444), and cron fe803474: that "decode backend = TREE_ATTN, FA2-fork = PREFILL-only." That was a
NAMING-SLIP TRAP (the boot echoes the backend NAME "TREE_ATTN", but the decode IMPL under it is the fork) +ans
an incomplete grep (the prior check grep'd FA2_PREFILL/TREE_ATTN but NOT `FR13_FA2_TREE_BIAS`).

## The fact (code-verified directly)
- `scripts/fr13_launch_locked.sh:24` exports **`FR13_FA2_TREE_BIAS=1`** (I missed it last tick).
- `scripts/fr13_patch_fa2_tree_bias.py` (_patch_tree_attn decode branch) rewrites TreeAttentionImpl's DECODE
  path to:
  ```
  if FR13_FA2_TREE_BIAS==1 and tree_bias.numel()>0 and decode_meta.max_query_len>1:
      flash_attn_varlen_func(..., tree_bias=tree_bias, fa_version=2)   # THE FORK (native FA2 + additive -inf tree bias)
  else:
      unified_attention(..., qq_bias=decode_meta.tree_attn_bias)        # the EXP2-Triton TREE_ATTN fallback
  ```
- cat9 decode tree-verify has `max_query_len = tree_len = 9 > 1` with a non-empty `-inf` ancestry bias →
  **the decode rows go through the FORK**, not the Triton fallback.

## Consequences
1. **The 0.00195 is MOOT.** It is the `unified_attention` EXP2-Triton FALLBACK residual (reached only when
   `FR13_FA2_TREE_BIAS=0` or `max_query_len==1`). The locked build shadows that path off. The user's
   hypothesis ("route decode through the fork to close 0.00195") is **already implemented** — nothing to swap.
2. **The live full-attn decode is at the fork's own floor: 0.0039** (14/16 whole-tree byte-exact 0.0, 15/16
   spine byte-exact, exactly 2 single-bf16-ULP in ~983k; root = irreducible MMA fp32 fragment-grouping over
   scattered no-copy KV, `project_fr13_fa2_fork_nocopy_floor`). **~15× below the E5 ~0.059 floor =
   argmax/distributionally lossless on the 16 full-attn layers.**
3. **Full-attn is NOT the carrier.** node7 first-nonzero is **L0 GDN linear_attn (0.0078), UPSTREAM of L3
   full-attn (0.00409)** — a full-attn kernel swap is structurally incapable of removing an L0-GDN-born
   divergence. **This REINFORCES the replay-durable-state pivot** (w2vaqcsmx): the residual lives on the
   cross-event GDN recurrent-state path, not the attention backend.
4. CUDA-graph: the fork-as-decode-kernel **FULL-captures + serves at B=4** (FR13_LADDER_LOG: "Capturing CUDA
   graphs (decode, FULL)", MAX_NUM_SEQS=4) — so the "TREE_ATTN chosen for capture" rationale does not
   distinguish it; the fork captures too. The standing ruling ("0.00195 within E5 floor → TREE_ATTN deploy
   wins") is satisfied for the full-attn sub-question, and the deployed kernel is the fork (floor 0.0039).
5. **Reward-hack check: LEGITIMATE.** The fork is FLASH + an additive `-inf` tree-mask computed in the real
   serving path, byte-verified splice-OFF (the fork genuinely computes the tree attention). It IS the deployed
   verifier — the opposite of a reroute (nothing is copied/re-streamed from a native call).

## Red-team caveat (do NOT report as current state)
The workflow cited "bag-TV 0.42-0.50 / accept 1.1134" (FR13_LADDER_LOG) as "the deliverable miss carried by L0
GDN." That is a **STALE B=4 run predating FR13_FA2_PREFILL_NATIVE** (prefill drift, a structural break), NOT
current locked cat9 (accept 3.1513, bag-TV ≤0.0593, 21 flips). The structural claim (full-attn swap can't fix
L0-GDN-upstream divergence) HOLDS; the specific stale numbers are off-stream (`feedback_check_artifact_before_concluding`).

## Net
The FA2-tree question is ANSWERED + closed: already deployed, lossless on full-attn, 0.00195 moot. No GPU test
needed (the fork floor is already measured; the optional confirmatory full-attn-layer A/B in the feasibility
doc is not on the critical path). The single live carrier hypothesis remains the **L0-GDN cross-event replay
durable-state** (w2vaqcsmx). Bug-class: #11 (naming-slip / measurement trap — a backend NAME masked the decode
IMPL), #12 (the real carrier is the cross-event GDN path, not the attention co-residency).
