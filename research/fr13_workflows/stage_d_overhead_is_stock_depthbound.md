# FR13 Stage D RESULT: the +28ms overhead is STOCK vLLM depth-bound tree-drafting, NOT our committer (2026-06-17)

## The profile (4-way per-step split, B=1 t0.6, cat6 vs native E5)
- verify forward (s/fwd_gpu): cat6 0.138 == native 0.137  -> ~0 tax (MEASURED, our forked-FA2 + GDN-tree-scan)
- committer (fr13_device_multidraft_commit, OURS): 0.34-0.35 ms/call, shape-invariant (MEASURED microbench, GPU-isolated)
- propose (tree-drafting): the +28 ms/step over E5 lives HERE
- stock scheduler/engine-loop/gap: CANCELS in the cat6-vs-E5 delta (both run identical stock)

## Why propose, and why it's depth-bound (read from the pinned container - eagle.py propose_tree)
propose_tree runs `for level in range(tree_depth - 1)`: each level does (1) a metadata REBUILD
(`replace(common_attn_metadata,...)` + `tree_attn_metadata_builder.build_for_drafting`), (2) a slot-mapping
gather, (3) a FULL draft-model forward over GROWING rows (`self.model(...)`, query_len = total_num_drafts so
far), (4) compute_logits + topk. So depth-5 = 4 sequential draft forwards + 4 metadata rebuilds. Native E5
uses the STOCK CHAIN propose (1-row forwards, no tree-metadata) -> much cheaper. The delta is the tree's
extra per-level work.

DEPTH-bound, not node-bound: cat6 (6 nodes) and cat9 (9 nodes) have ~IDENTICAL wall/step (0.260 vs 0.252)
despite cat9 having 50% more nodes -- because both are depth-5 -> both do 4 sequential draft forwards. Node
count only changes rows/forward (HBM-bound, ~free). CORROBORATES the depth-bound model.

## CORRECTION to the "tree-propose is OURS / 100% ours to fix" premise (overhead_profiling_future_work.raw.json)
propose_tree is STOCK vLLM (read from the container image). We override only the tree SHAPE
(cu_drafts_per_level / child_drafts_per_level), NOT the propose loop. So the +28 ms is stock vLLM's inherent
tree-drafting cost for our chosen shape -- NOT our code to cheaply tune. What we OWN: the committer (0.35 ms,
already minimal) + the SHAPE choice. The earlier "the entire slowdown vs E5 is our committer + tree-propose,
100% ours to fix" was half-right: the committer is ours (and minimal); the propose loop is stock.

## Stage D tuning VERDICT (per the plan: "else don't ship, report the negative")
NEGATIVE on a committer/propose CODE-tune:
- committer (ours) is 0.35 ms -> already optimal, nothing to recover.
- propose_tree is stock vLLM + the growing-row candidate forwards are IRREDUCIBLE (they generate the extra
  candidates that buy cat6's +15.7% accept; remove them and you lose the accept).
- the only tunable slice = caching the static tree-attn metadata (shape is fixed) inside stock propose_tree
  = a complex stock-vLLM patch with PARTIAL payoff (seq_lens/slot_mapping still vary per step). Not a cheap,
  clearly-lossless win. Gate FAILS -> do NOT ship a code-tune.

## REDIRECT: the lever is SHAPE = DEPTH (we own the shape)
Step overhead is set by DEPTH (# levels = # sequential draft forwards). To cut it, go SHALLOWER:
- DEPTH-3 shapes (cat555 / cat3w) = 2 draft forwards instead of 4 -> ~half the propose overhead, AND (per
  [[project_fr13_tree_reshape_unifying_lever]]) fewer lossless flips + fewer verify rows. Trade: lower accept
  (depth-3 < depth-5). OPEN QUESTION = does cheap-step depth-3 beat cat6's absolute 18.51 TPS on net
  committed/wall? Compares to native E3 (depth-matched). Needs wiring (cat555/cat3w TREE override not yet in
  fr13_bigdenom_swe_serve_variant.sh).
- cat55222 ([5,5,2,2,2]) is STILL DEPTH-5 -> does NOT cut propose overhead (same 4 forwards); it only
  redistributes width (accept-distribution experiment), not a step-cost lever.

DELIVERABLE STANDS: cat6/cat9 beat native E5 + lossless, merged to main (2b29e599). Stage D adds: the win is
already near-optimal for depth-5; more TPS needs a shallower shape (open) or accepting the depth-5 result.
