# FR13 General Drafter Interface — MTP head / suffix decoding / merged

Design 2026-07-14. Owner: correctness-critical (committer contract). Grounded in the
measured reality of this session, not aspiration.

## Measured constraints that shape the design
1. **Drafter is HOST-orchestration-bound, not compute-bound** (split10: 23ms GPU compute —
   8.3ms model fwd + 15.1ms lm_head — of a 140ms/step drafter; 117ms is host). => a new source
   MUST NOT add host Python overhead; and avoiding the draft *forward* (suffix) saves only the
   23ms GPU slice unless it also cuts host orchestration.
2. **Tree width is HARD-CAPPED at n_pad=16 by a register wall** (Front 1, CORRECTED): the FR10
   tree kernel holds h_cache=[N_PAD,BV=16,DIM_K=128] fp32 register-resident (128 KB/CTA at 16,
   spills HBM at 32; fr10_gdn_tree_kernel.py:408). Both cat8(9) and t33333(16) run at n_pad=16;
   n_pad=32 fails to boot. => a cheap draft source CANNOT add nodes past 16 — its value is
   **filling the fixed <=16 branch slots with BETTER tokens** (higher accept per slot), NOT
   widening. Widening needs a BLOCK_V kernel re-tile (S-D, not cheap).
3. **The committer already verifies arbitrary per-node candidates — PROVEN committer-transparent**
   (Gate 1: scripts/fr13_suffix_committer_contract_gate.py, 32/32 vs the REAL rule
   host_multidraft_accept_probs). The deployed multidraft rejection rule (draft_probs=None path)
   computes overlaps=p[cands], weights=overlaps/overlap_mass, accept_prob(s)=min(1,p[t]/q_mix[t]),
   residual=max(p-q_mix,0). The exact math (analytic + Monte-Carlo confirmed):
   - **OUTPUT is EXACTLY p, for ANY candidate set** (P(out=t)=min(q_mix,p)+max(p-q_mix,0)=p[t]).
     Adding suffix/garbage candidates NEVER changes the output distribution => LOSSLESS,
     UNCONDITIONAL. (NB: the earlier draft's "accept *distribution* unchanged" was WRONG — the
     per-source weights and accept RATE do change; what's invariant is the OUTPUT.)
   - **ACCEPT RATE = p(S)** = target mass on the DISTINCT candidate set; adding a candidate with
     p>0 RAISES it (monotone). This is the exact speed-win: suffix candidates are a pure monotone
     accept lever. (Requires MergedSource to DEDUP candidates for the clean identity; lossless
     holds even without dedup.)
   - **ZERO garble**: a garbage token g (p[g]~0) is committed at rate EXACTLY p[g]~0 — the safety
     is the SOURCE-SELECTION weight (=p[g]/overlap_mass ~0), NOT accept_prob (which = overlap_mass,
     not small). So a bad suffix candidate is almost never even selected; if selected it still
     commits only at rate p[g]. Suffix decoding carries ZERO garble risk BY THE COMMITTER MATH.

## The interface

```
class DraftSource(Protocol):
    def propose(self, ctx: DraftContext) -> NodeCandidates: ...
    # ctx: running token ids (committed prefix per req), last hidden states,
    #      the tree_spec (per-node parent + which nodes need candidates),
    #      batch_size, generators.
    # returns: per (req, tree_node) -> candidate token id(s). One-per-node for a
    #      fixed tree; the committer picks/verifies. draft_probs stays None
    #      (deployment path) so no [nodes x vocab] q is materialised.
```

Backends:
- **MtpHeadSource** (the current path): the FR13_DRAFTER_SINGLE_LOGITS eager MTP forward +
  compute_logits; argmax spine + topk(root/level) branches. Unchanged; it IS the baseline.
- **SuffixSource** (new): a per-request suffix automaton / hash-of-last-k-gram over the request's
  OWN committed token stream (code editing repeats identifiers/paths/boilerplate). Match the
  current suffix (last k tokens) -> propose the historical continuation as the draft chain.
  ZERO forward, ZERO lm_head. Host cost = an O(1)-amortised dict lookup per req (NOT a Python
  per-node loop — that's the host-overhead trap; keep it a vectorised gather off a prebuilt
  index). Feeds candidates for as many tree nodes as the match length covers.
- **MergedSource** (the goal): compose. Two compose modes, both committer-transparent:
  (M1) **spine-quality + suffix-width**: MTP provides the spine chain (depth, the accept
       backbone); suffix provides EXTRA branch candidates at each depth at ~zero GPU cost ->
       widens the tree past what MTP topk cheaply gives, lifting accept per Front-2 constraint.
  (M2) **confidence pick**: per node, prefer the source with higher agreement; falls back to
       MTP when suffix has no match (cold context). Deterministic, committer-verified either way.

## Correctness gates (mandatory, in order)
1. **Committer contract proof (CPU) — DONE, PASSED 32/32** (scripts/fr13_suffix_committer_contract
   _gate.py): proved vs the real rule that the OUTPUT distribution = p EXACTLY for any candidate set
   (lossless, unconditional), ACCEPT = p(distinct S) monotone (the speed lever), and a garbage token
   commits only at rate p[g]~0 (garble-safe via source-selection weight). See constraint #3.
2. **Garble gate** (fr13_garble_gate.py, temp-0.6 matrix): suffix must not raise undefined-name
   rate above native — it structurally cannot (committer-verified), but PROVE it live.
3. **Engagement / non-vacuous**: SuffixSource ENGAGED log + match-rate needle (how often suffix
   supplies a candidate; 0% match = vacuous = the source is dead weight).
4. **Live B=4 A/B vs cat8+SLOT_REORDER baseline**: accept, resolve, give-ups, garble, deploy-speed.
   DELIVERY = merged >= baseline on quality AND (accept-per-forward up OR wall/tok down).

## Why this is the RIGHT next lever (given the campaign's cost-gate)
The speed campaign concluded the tree's remaining gap is structural drafter host overhead.
Suffix decoding attacks a DIFFERENT axis: it doesn't make the existing drafter faster — it adds
ACCEPT (better tokens in the fixed <=16 branch slots + spine-extension on repetitive context)
which is the HBM-bound lever (accept-per-forward), the ONE thing that isn't dead. On agentic code
editing (high repeat: re-reading files, repeated identifiers, boilerplate), suffix match rates are
plausibly high -> free accepted tokens. Bounded by the n_pad=16 register wall (constraint #2), so
the win is per-slot token QUALITY within 16, not wider trees. Net: a plausible accept win with
ZERO garble risk (Gate 1 PROVEN) and ZERO GPU compute add — the profile the measurements point to.

## Impl order (each independently gated; delegate only mechanical/isolated pieces)
S-A. SuffixSource module (isolated new file): per-req rolling k-gram index + match -> candidates.
     CPU-testable in isolation (feed a token stream, assert matches). DELEGABLE (isolated file).
S-B. Wire the DraftSource seam at the propose() call (I own — drafter edit): flag
     FR13_DRAFT_SOURCE=mtp|suffix|merged, default mtp (byte-identical). MtpHeadSource = current.
S-C. MergedSource compose (I own): suffix candidates -> extra tree children; committer-transparent.
S-D. (optional, NOT cheap) widen past n_pad=16 for wide merged trees. NOTE: the 16 cap is a REAL
     register wall, not buffer sizing — the FR10 tree kernel (fr10_gdn_tree_kernel.py:408)
     holds h_cache=[N_PAD,BLOCK_V=16,DIM_K=128] fp32 register-resident (128 KB/CTA at 16 = half
     the SM reg file; 32 spills to HBM). Widening REQUIRES re-tiling BLOCK_V 16->8 (+2x V-grid,
     redundant q/k reloads) — a kernel rewrite. So prefer keeping merged trees <=16 nodes and
     spending the free suffix candidates on BRANCH DENSITY within 16, not on going wider.
Gates 1->4 per step. Suffix is losslessness-safe by the committer, so the risk is SPEED/ACCEPT
(does it help), not CORRECTNESS (it can't garble).

## S-B/S-C SEAM LOCATED (2026-07-14) — small + localized, NOT a drafter rewrite
The tree branch leaf tokens are built at patcher :13361 (FR13_DRAFTER_SINGLE_LOGITS):
  _fr10_logits = self.model.compute_logits(sample_hidden_states)
  draft_token_ids = _fr10_logits.argmax(-1)                    # spine token (per depth)
  _fr10_root_topk = torch.topk(_fr10_logits, k, -1).indices
  _fr10_root_leaf_token  = _fr10_root_topk[:, 1]               # branch rank-2  <-- SEAM
  _fr10_root_leaf2_token = _fr10_root_topk[:, 2]               # branch rank-3  <-- SEAM
The tree GEOMETRY is fixed (tree_choices, n_pad<=16 register wall); the branch LEAF TOKENS are the
free variable. Suffix improves WHICH token fills each branch slot (Gate 1: replacing a low-p
drafter rank-3 with a target-plausible suffix token raises p(S) -> accept, losslessly). NOT adding
nodes (register wall forbids) -- filling fixed slots better ("branch density within 16").

GRAPH CONSTRAINT (must respect): the drafter runs in a PIECEWISE cudagraph; the suffix dict lookup
is a HOST op that CANNOT run in-graph. => S-B wires a PERSISTENT device buffer
(suffix_branch_tokens + suffix_valid mask) at captured addresses + a per-step host->device copy
(host: SuffixSource.propose fills the buffer, like slot_mapping staging). S-C does the in-graph
blend: `_fr10_root_leaf_token = torch.where(suffix_valid, suffix_branch_tokens, drafter_topk_rank2)`
(device torch.where, graph-safe). Flag FR13_DRAFT_SOURCE=mtp|merged, default mtp = byte-identical
(don't touch the tokens); MUST dedup vs the spine argmax (Gate 1: duplicate adds nothing).

COST-GATE (honest): (+) host cost negligible -- SuffixSource.propose ~7us (50k-tok test 0.35s) vs
140ms/step host-bound drafter (<0.01%). (+) DOWNSIDE ZERO -- lossless (Gate 1), can't regress
correctness; a bad suffix guess is just rejected. (?) UPSIDE is workload-dependent -- accept gain
only materializes if suffix guesses carry real target-p on agentic code-editing (repetitive:
re-read files, repeated identifiers). That's the LIVE B=4 A/B (Gate 4) -- unknowable until measured,
but it's a PURE-UPSIDE bet (cheap + can't regress + plausible). Cost-gate PASSES.

NEXT (mine, correctness-critical, next cycle): S-B buffer+staging, then S-C in-graph blend; gate
each with the CPU committer contract (Gate 1 extended to the blend), garble gate, then live B=4 A/B
vs cat8+SLOT_REORDER.

## PIVOT (2026-07-14, user): use the REAL Snowflake code, not my hand-rolled k-gram
The suffix source = **arctic_inference.suffix_decoding.SuffixDecodingCache** (pip arctic-inference
==0.1.2), surfaced by vLLM's built-in `vllm.v1.spec_decode.suffix_decoding.SuffixDecodingProposer`.
Prior git history integrated it (Round-2/Track-B: T1 `_SessionRoutedSuffixDecodingCache` router
+ prelaunch install hook + T3 schema-aware composite drafting; commits 29116417/8d4c4a0b/af3676fb).
REAL API (from the vLLM proposer, authoritative):
  cache = SuffixDecodingCache(max_tree_depth, max_cached_requests)
  cache.start_request(req_id, prompt_token_ids)          # builds suffix tree for the prompt
  cache.add_active_response(req_id, sampled_ids)          # ingest generated tokens
  draft = cache.speculate(req_id, pattern, max_spec_tokens=, max_spec_factor=, min_token_prob=)
  draft.token_ids                                          # dynamic-length speculation
  cache.stop_request(req_id) / evict_cached_response(req_id); .active_requests / .cached_requests
Config knobs (vLLM speculative_config): suffix_decoding_max_tree_depth (1-64), max_spec_factor
(0-4), min_token_prob (0-1), max_cached_requests (0-10000), num_speculative_tokens.
INSTALL: container prelaunch (arctic builds a C++ ext + pulls torch at build; host CPU-venv build
fails -- goes in the vLLM image like all GPU/container deps).
GATE 1 PRESERVED: it proved ANY candidate set fed to our committer is lossless + monotone-accept
(source-agnostic) -- so it carries over unchanged; only the SOURCE swaps (arctic <- my k-gram).
My scripts/fr13_suffix_source.py is SUPERSEDED (kept for now; committer-contract gate stays -- it's
source-agnostic). OPEN FORK (user owns): (A) vLLM-native suffix proposer standalone = replaces our
GDN tree + committer with vLLM's native path (Round-2 shape); (B) arctic as the SUFFIX BACKEND of
our merged drafter, feeding OUR committer/tree (preserves the FR13 lossless deliverable). Awaiting call.

## CHOSEN ARCHITECTURE (user, 2026-07-14): MTP-k spine + Arctic-suffix GROW to cat33333
"use native mtp1 or 2 then use suffix decoding to grow into a cat33333 tree." Path B (feeds OUR
committer -> preserves the FR13 lossless deliverable). MOTIVATION: the FR13 drafter is PARALLEL
(one forward), so its DEEP spine tokens (depth 2-4) are weak parallel-drafts; Arctic suffix decoding
RETRIEVES the actual historical continuation (high accept on repetitive agentic context) and is FREE
(no forward -- the host-bound-drafter insight). So swap MTP's weak deep-parallel-drafts for suffix's
strong retrieved continuations.

TARGET GEOMETRY = t33333 (16 nodes, n_pad=16, within the register wall):
  spine: root -> (0,) -> (0,0) -> (0,0,0) -> (0,0,0,0) -> (0,0,0,0,0)   [depth-5 argmax chain]
  branches: 2 per spine level -- (1,),(2,) at d1; (0,1),(0,2) at d2; ... (0,0,0,0,1/2) at d5.
NODE MAPPING (mtp_k in {1,2}, flag FR13_MTP_SPINE_DEPTH):
  - spine[0..k-1] = MTP head tokens (argmax; existing eagle forward). Confident near tokens.
  - spine[k..4]   = Arctic suffix continuation tokens (the deep spine). FREE.
  - branches @ every level = Arctic tree alternatives at that level (ranked by min_token_prob),
    dedup vs the spine token; FALL BACK to MTP topk rank-2/3 when suffix has no alternative.
DRIVE (per req, per step, at the :13361 eager seam):
  1. MTP forward -> t0 [,t1].
  2. pattern = recent committed suffix ++ [t0[,t1]]  (so suffix continues FROM the MTP prediction,
     chaining MTP->suffix coherently).
  3. draft = suffix_cache.speculate(req_id, pattern, max_spec_tokens>=5, max_spec_factor>0) -> tree.
  4. assemble the 16 cat33333 node tokens from {MTP near, suffix deep+branches}; dedup per node.
  5. OUR committer/tree verifies (lossless, Gate 1).
LIFECYCLE (adapt vLLM SuffixDecodingProposer + Round-2 T1 router): start_request(prompt) on new req,
add_active_response(committed ids) each step, stop_request on finish; per-session routing optional.
INSTALL: container prelaunch `pip install arctic-inference==0.1.2` (Round-2 run_track_b_loop.py:1165).
FALLBACK (never regress): suffix no-match / cold -> pure MTP cat33333 (current drafter) = baseline.
GATES: (1) committer contract DONE (source-agnostic, 32/32). (2) ASSEMBLY unit test (CPU, mock
arctic tree -> correct cat33333 node tokens, dedup, fallback) -- correctness-critical core, mine.
(3) byte-identical when FR13_DRAFT_SOURCE=mtp (default). (4) garble temp-0.6. (5) live B=4 A/B:
MTP+suffix cat33333 vs MTP-only t33333 baseline -- accept, resolve, garble, deploy-speed.

## KEY FINDING (2026-07-14): MTP-k+suffix is ALSO a host-SPEED lever, not just accept
Reading the drafter spine loop (patcher :13610 `for token_index in range(_fr10_spine_steps)`):
the drafter runs ONE sequential `self.model(**model_kwargs)` forward PER spine depth (+ per-iter
EAGER orchestration: eagle_step_update_slot_mapping_and_metadata, set_forward_context,
compute_logits, argmax). `_fr10_spine_steps`=4 for the wide/cat33333 path (13191). So the depth-5
spine costs ~4 sequential forwards + 4 host-orchestration iterations -- NOT pure-parallel (corrects
my banked "single parallel pass" shorthand; that referenced eagle's num_spec==1 early-exit, not the
caterpillar spine).
=> MTP-k + suffix-grow SKIPS (spine_steps - mtp_k) spine iterations: with mtp_k=1 we run 1 forward
instead of 4, and the deep spine (depths 2-4) + branches come from Arctic (host lookup). This
attacks the DRAFTER HOST OVERHEAD -- the "structural/dead" gap the speed campaign closed -- via a
DIFFERENT lever (reduce the NUMBER of iterations, not per-iteration overhead), so it is NOT a
re-opened concluded lever. Potential DOUBLE win: fewer forwards+orchestration (SPEED) + better deep
tokens on repetitive context (ACCEPT). Verify cost UNCHANGED (still 16-node tree).
RED-TEAM (honest, UNMEASURED): (a) the per-iteration host savings is INFERRED from loop structure,
not measured -- the forward is cudagraph-captured (cheap compute); the saving is the EAGER
orchestration per iteration, magnitude unknown. MUST measure (sfwd/dfwd timers, iteration count).
(b) accept: Arctic deep-spine must accept >= MTP deep-spine (repetitive: plausibly; else fallback).
(c) net host = saved iterations - Arctic speculate() lookup; sign unknown until measured.
IMPL IMPACT: not just a token-value substitution at :13361 -- also REDUCE _fr10_spine_steps to mtp_k
and fill the skipped spine depths + their wide_topk branches with Arctic tokens before packing (the
packer consumes _fr10_spine_tokens list + _fr10_wide_topk dict; append/fill to depth-5). Careful
edit: the loop mutates seq_lens/slot_mapping/KV per step; stopping early is fine (skipped spine
tokens are Arctic draft candidates the target verifies -- Gate 1 lossless -- never run through the
draft model). Gate the speed claim with the sfwd/dfwd GPU timers + iteration-count log.
