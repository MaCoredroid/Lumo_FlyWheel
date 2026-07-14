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

### Reconciliation with prior "single parallel pass / depth lever dead" (#26/#27), honest
Code is DEFINITIVE: the cat33333 spine loop runs range(_fr10_spine_steps)=range(4) with a
self.model() forward per iter and NO parallel_drafting guard (that guard is in STOCK eagle, which
the FR10 caterpillar patch REPLACES). So the SPINE is 4 sequential forwards. The prior "parallel
drafting" was correct ONLY for the BRANCHES (each depth's runner-up branches are read via topk from
that spine step's logits -- free, no per-branch forward); it was IMPRECISE as "the whole drafter is
one forward." And "depth/shape lever dead" (#27 S2 DP) was about ACCEPT-vs-tree-SHAPE within depth-5
-- a different question than "skip spine FORWARDS." So the spine-forward-count lever (mtp_k) is NEW,
confirmed by code, and NOT contradicted by the earlier conclusions. Still UNMEASURED: the net host
delta (saved spine-iteration orchestration - Arctic lookup); the live A/B + sfwd/dfwd timers decide.

## SEAM FULLY SCOPED (2026-07-14): wide packer format -> exact injection
The wide (t33333) packer (:13905) builds the tree from a (parent_pos, rank) plan:
  for (pp, rk) in _fr10_wide_plan:
    cols.append(_fr10_spine_tokens[pp] if rk==0 else _fr10_wide_topk[pp][:, rk])
  _fr10_packed = torch.stack(cols, dim=1)
So the injection (FR13_DRAFT_SOURCE=merged) is: fill _fr10_spine_tokens[d] (the spine, [batch] tensor)
and _fr10_wide_topk[d][:, 1:3] (the 2 branches, [batch,>=3] tensor) for the DEEP positions d>=mtp_k
with Arctic tokens; near positions d<mtp_k stay MTP. The assembly's 15-node output (CAT33333_ORDER =
[spine_d, branch_a_d, branch_b_d] x5) maps directly: spine_tokens[d]=node[3d],
wide_topk[d][:,1]=node[3d+1], [:,2]=node[3d+2]. SPEED path: run only mtp_k spine-loop iterations
(range(min(_fr10_spine_steps, mtp_k))), skipping the deep MTP forwards (the -53% drafter win); the
deep _fr10_spine_tokens/_fr10_wide_topk entries are then Arctic-filled before the packer. Existing
t33333 packer + our committer/verify UNCHANGED (Gate 1 lossless).
FULL SEAM SPEC (next cycle, mine): (a) container prelaunch pip install arctic-inference==0.1.2
(Round-2 run_track_b_loop.py:1165); (b) instantiate SuffixDecodingCache on the drafter (lazy);
(c) lifecycle: start_request(prompt) on new req, add_active_response(committed) each step,
speculate(pattern=recent_suffix++MTP-near) at draft, stop_request on finish (vLLM SuffixDecoding
Proposer pattern + Round-2 session router); (d) adapt .speculate().token_ids -> relative suffix_rel;
(e) at :13860, gated merged: shorten spine loop to mtp_k + Arctic-fill deep spine/wide_topk via the
assembly, dedup; (f) flag FR13_DRAFT_SOURCE=mtp(default,byte-id)|merged, FR13_MTP_SPINE_DEPTH=1|2.
GATES: patcher self-test byte-id-off (CPU) -> garble temp06 -> live B4 A/B (accept-hold = decisive).

## RED-TEAM VERDICT (2026-07-14, 3-lens workflow wf_5ab28060): plan_sound=False -> CORRECTED
Adversarial review found the initial seam spec had BLOCKERS. Graph-safety = GO (seam IS eager,
outside cudagraph; host Arctic call + spine-loop reduction are safe). Corrected plan:

TENSOR DISCIPLINE (was 4 blockers): (1) build every Arctic column on draft_token_ids.device (int64),
one H2D per depth -- NOT torch.tensor(list) (CPU) => stack crash. (2) fill ROW-MAJOR keyed by req_id
via the SPEC-row req-id list aligned to drafter batch rows (same list committer uses :8744/:8796) --
positional against Arctic's own order = wrong-row drafts = accept collapse (garble-free). Assert
len==batch_size + per-row req-id equality (fail-loud). (3) _fr10_wide_topk[d] MUST be [batch,>=3]
(packer fail-loud checks rk < shape[1] :13928); cols 1,2 = Arctic branches, col0 = placeholder
(spine tok, free dedup). (4) densify ragged/empty Arctic output to a full [batch] column with a
concrete PAD (Gate-1: p[g]~0 commits ~0). Append spine strictly depth-ordered; wide_topk keys must
be exactly the filled depths; re-run the packer length assert (:13914).

LIFECYCLE SPLIT BY FRAME (was 3 blockers -- the initial "all at :13361" was WRONG): drive
start_request / stop_request / evict_cached_response at the MODEL-RUNNER level via the batch req-set
diff (port vllm suffix_decoding.py:63-70,92-95 verbatim; prompt from input_batch.token_ids_cpu).
Drive add_active_response at a NEW COMMIT-SITE hook (~:8730-8767) feeding the FULL accepted run (not
the 1 bonus token), where accepted_token_rows + per-row req-ids + _LUMO_FA_TREE_ACCEPT_BY_REQ already
coexist. Only speculate() runs at the :13361 seam. pattern = last max_tree_depth committed tokens per
req (stash a rolling per-req buffer at the committer; :8734 already publishes last-accepted).

SPEED LEVER = ADAPTIVE (reconciles user arch mtp1/2+grow with never-regress): the forward-skip is
FRAGILE (no fallback if Arctic empty -> deep-node garbage -> accept collapse; spine_steps is a SCALAR
so can't be per-req at B=4; per-step D2H t0 + H2D pattern syncs may ERASE the win). FIX: gate the
spine_steps reduction on Arctic having a batch-wide confident deep match for ALL active reqs BEFORE
the loop; else run the FULL MTP loop (never-regress). Fires often since agentic decode is effectively
B~1 (Running==1 ~80%). DRAFT-KV HOLE = NON-ISSUE: if we skip ALL forwards after mtp_k, no downstream
drafter forward needs the skipped KV, and verify uses the TARGET cache not draft-KV (VERIFY still to
confirm eagle reseeds draft from target hidden each step). MUST MEASURE per-step D2H/H2D sync cost
(sfwd/dfwd timers) -- the skip win can be partially/fully erased; that measurement is a gate.
TWO SHIPPABLE MODES: (A) ACCEPT-ONLY (always full MTP loop + ADD Arctic branch candidates, no skip)
= genuinely never-regress, speed via accept only; (B) ADAPTIVE-SKIP (A + forward-skip when batch-wide
match) = the speed lever, gated. Build A first (simpler, safe), then layer B behind a flag + the sync
measurement. Both are Gate-1 lossless.

## MODE DECISION (2026-07-14): build ADAPTIVE Mode B first (honors user arch + never-regress)
The flat-chain adapter (arctic .token_ids -> deep SPINE) naturally supports Mode B (forward-skip),
NOT Mode A (arctic->branches, which needs arctic's TREE not the flat chain). So build:
**ADAPTIVE Mode B** = the user's mtp1/2 + suffix-grow, made safe: run mtp_k MTP forwards; predicate
on arctic having a full-depth (>= N_DEPTH-mtp_k) match for ALL active reqs; if YES -> skip remaining
spine forwards, arctic fills deep spine (the -53% drafter win); if NO -> run the FULL MTP loop
(pure-MTP baseline, never-regress). Resolves the red-team "no fallback" blocker (adaptive gate = the
fallback). Fits the flat-chain adapter. Gate-1 lossless either way. The D2H(t0)/H2D(pattern) per-step
sync cost is the OPEN risk -> the live A/B's dfwd timer is the gate (could erase the skip win).
Mode A (accept-only, no skip, arctic branches) DEFERRED -- needs the arctic suffix TREE + a branch-
feeding assembly variant; layer it later if adaptive-B's sync cost erases the win.
CRITICAL IMPL NOTE (prelaunch agent): the runtime seam MUST read FR13_DRAFT_SOURCE via the SIDECAR-
FILE pattern (like FR13_COMMITTER_NATIVE/FR13_COMMIT_ARGMAX_GATE, launcher :402-424), NOT os.environ
-- the EngineCore WORKER drops FR13_* env vars. Prelaunch install draft ready at
fr13_launch_forked_fa2_tree_server.sh:728 (gated FR13_DRAFT_SOURCE=merged, byte-id when off).

## BUILD PROGRESS (2026-07-14): all supporting pieces DONE; patcher seam = remaining unit
DONE + committed (CPU-proven / gated):
  - Gate 1 committer contract 32/32 (fr13_suffix_committer_contract_gate.py)
  - assembly core 36/36 (fr13_mtp_suffix_assembly.py)
  - arctic adapter 30/30 (fr13_arctic_suffix_adapter.py: .token_ids->suffix_rel)
  - fill helper 35/35 (fr13_merged_fill.py: assembled nodes->spine_tokens/wide_topk, red-team discipline)
  - prelaunch install (launcher :728, gated FR13_DRAFT_SOURCE=merged, byte-id off)
  - FR13_DRAFT_SOURCE=merged SIDECAR (launcher, worker-env-drop-proof; worker reads
    /logs/fr13_draft_source_merged.arm)
REMAINING = the patcher seam edit (fr10_phase4_patch_vllm_tree_gdn.py, ALL gated on the sidecar,
default mtp = byte-identical). Precise plan:
  (S1) module-scope: `_fr13_merged_on()` reads /logs/fr13_draft_source_merged.arm (mirror
       _fr13_committer_native_on); lazy `_fr13_suffix_cache` holder (import arctic in-worker,
       SuffixDecodingCache(max_tree_depth, max_cached_requests)); import assemble_cat33333 +
       arctic_draft_to_suffix_rel + build_cat33333_columns from /workspace/scripts.
  (S2) RUNNER-level lifecycle: start_request(req_id, prompt_token_ids) for new reqs +
       stop_request/evict_cached_response for gone reqs (batch req-set diff; port vllm
       suffix_decoding.py:63-70,92-95). Find the runner frame where input_batch.req_ids +
       token_ids_cpu live (near the drafter call site).
  (S3) COMMIT-SITE hook (~:8730-8767): add_active_response(req_id, accepted_run) fed the FULL
       accepted token rows (accepted_token_rows + per-row req-ids already there); maintain a rolling
       per-req recent-committed buffer for the pattern.
  (S4) SEAM (:13361 speculate + :13860 fill): per batch ROW b keyed by spec-row req-id ->
       speculate(req_id, pattern=recent_suffix++mtp_near) -> arctic_draft_to_suffix_rel ->
       assemble_cat33333 -> collect assembled_rows -> build_cat33333_columns -> overwrite deep
       _fr10_spine_tokens[d]/_fr10_wide_topk[d]. ADAPTIVE gate: only shorten the spine loop
       (range(mtp_k)) when ALL active rows have a full-depth (>=N_DEPTH-mtp_k) arctic match; else run
       the full MTP loop (never-regress). Engagement counters (speculate_fired, assembler_engaged),
       fail-loud if merged-on but never engaged.
  (S5) patcher self-test fr13_merged_drafter_s0_test.py: apply to pristine copy -> compile -> assert
       byte-identical when sidecar absent + markers present when the injected block exists.
GATES after build: self-test (byte-id) -> detached boot merged (ENGAGED assert) -> garble temp06 ->
live B=4 16-task A/B (correctness parity + dfwd speed same-or-better; accept-hold + D2H/H2D sync are
the decisive measured gates).

## SEAM WIRING PROGRESS (2026-07-14): orchestration DONE; token-source gotcha found
DONE + committed: orchestration module fr13_merged_drafter.py (23/23) -- lifecycle + rolling buffer
+ ADAPTIVE gate + engagement counters, mock-cache tested. Now 5 CPU components proven (Gate1/assembly
/adapter/fill/orchestration) + flag plumbing (prelaunch + sidecar). Patcher hooks are THIN (call the
module).
INJECTION SITES LOCATED: commit-site ~:8734-8767 has accepted_token_rows + _fr13_row_req_ids
(_LUMO_FA_SAMPLER_ROW_REQ_IDS) in scope; seam :13361 (root token :13364, spine loop :13610, packer
:13905).
CRITICAL GOTCHA (must fix before the live gate): the Arctic add_active_response stream MUST be the
FULL committed tokens per step (accepted DRAFTS + BONUS token), NOT accepted_token_rows alone
(=_LUMO_FA_LAST_ACCEPTED_TREE_TOKEN_IDS :8734, which OMITS the bonus). A gappy (bonus-less) history
is NOT vacuous -- Arctic still matches the consistently-gappy stream, so the ENGAGEMENT gate would
FALSELY PASS -- but its predicted drafts are wrong-by-one-per-step => LOW accept (only the live A/B
catches it). Authoritative full stream = output_token_ids (accepted+bonus, :8546) or the runner
output-append; bonus per-req = native_bonus_token / bonus_targets_cpu[req_i] (:8318). => feed
ingest_committed the full committed row (accepted_token_rows[i] ++ [bonus_i]) so MY rolling buffer AND
Arctic's history are the REAL sequence, and drafts predict reality.
REMAINING THIN HOOKS: S1 module-scope (import fr13_merged_drafter, merged_on() sidecar, get_cache);
S2 RUNNER frame (note_new_requests with prompts + retire_requests batch diff -- co-locate with the
add_active_response ingestion so the stream is non-gappy); S3 ingest full committed row at the
commit-site; S4 seam (decide_and_fill after mtp_k MTP tokens -> if do_skip: apply spine_tokens/
wide_topk + shorten loop; else full MTP); S5 patcher self-test (byte-id off + markers on). Then boot
ENGAGED-assert (require match_full>0 not just speculate_fired>0, to catch the gappy-history trap) ->
garble -> live 16-task A/B.

## SIMPLIFIED INJECTION MAP (2026-07-14): runner + seam (2 sites), no commit-site needed
The runner frame (:11245, _LUMO_FA_SAMPLER_ROW_REQ_IDS set from input_batch.req_ids) has
input_batch.token_ids_cpu = the AUTHORITATIVE full committed sequence per req (prompt + generated,
incl bonus). => feed Arctic the non-gappy DELTA there (ingest_from_sequence), sidestepping the
bonus-token archaeology. No commit-site (:8734) hook needed. All logic now DONE + tested (25/25).
FINAL 3 PATCHER HOOKS (thin, all gated fr13_merged_drafter.merged_on()):
  S1 IMPORT: add /workspace/scripts to sys.path + `import fr13_merged_drafter` (in the runner patch
     _patch_gpu_model_runner_* near :11245 AND/OR gdn patch -- same process, cached module = shared
     state). get_cache() lazily (arctic max_tree_depth from a baked const, default 24).
  S2 RUNNER (~:11245, input_batch in scope): if merged_on(): cache=get_cache(); note_new_requests(
     {req_id: token_ids_cpu[idx,:num_prompt] for new reqs}); for each row ingest_from_sequence(
     cache, req_id, token_ids_cpu[i], num_tokens); retire_requests(active - batch). Row order ==
     input_batch.req_ids[:num_reqs] (== the seam's spec-row order).
  S4 SEAM (:13361 after root, :13610 loop, :13905 packer): if merged_on() and _fr10_is_wide (t33333):
     gather mtp_near (root [+ mtp_k-1 loop tokens]) + mtp_topk per row; decide_and_fill(spec_row_req_ids,
     near, topk, mtp_k, draft_token_ids.device, pad); if do_skip -> overwrite _fr10_spine_tokens[d] +
     _fr10_wide_topk[d] for deep d + set _fr10_spine_steps so the remaining loop is skipped; else run
     full MTP loop. pad = a benign in-vocab token (e.g. int(draft_token_ids[0])).
  S5 SELF-TEST fr13_merged_drafter_s0_test.py: apply patcher to pristine vLLM copies (/tmp/fr13inv or
     a fresh checkout) -> compile gdn_linear_attn + eagle -> assert markers present + (sidecar absent)
     the merged branches are dead. Host-runnable (no GPU).
