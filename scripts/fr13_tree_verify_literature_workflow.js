export const meta = {
  name: 'fr13-tree-verify-literature',
  description: 'USER (2026-06-15): online-research how OTHER tree-verification / spec-decode methods deal with the near-tie leaf-fork = the verify-vs-decode realization gap that makes our top-down LCP committer fork onto leaf paths (committing drafts[leaf] tokens that diverge from greedy decode). The user proposes a SPINE-BONUS / prefer-spine-at-near-ties tie-break (= margin-damp). KEY QUESTIONS: (1) how do SpecInfer/Sequoia/EAGLE-2/SpecTr/Medusa/DeFT/STree/Traversal-Verification handle a verify forward whose per-node argmax differs slightly from the true target greedy (realization gap) + near-tie path selection? (2) Is a deterministic SPINE/CANONICAL-PATH bonus or margin-aware tie-break a KNOWN PROVABLY-LOSSLESS pattern for GREEDY (temp-0) tree spec decoding, and under what margin condition (vs MARS-style runner-up acceptance which is LOSSY)? (3) Deep-read Traversal Verification arXiv 2505.12398 = the bottom-up lossless commit lead. (4) Map to our case: our LCP already tie-breaks spine-on-EQUAL-LCP; the fork is leaf-STRICTLY-GREATER-LCP — does the literature support a margin-aware extension (spine keeps it unless the leaf beats by a clear verify-margin) as LOSSLESS? CONSTRAINTS: the fix must be COMMITTER-ONLY (no copy, no HBM tax, GB10 273 GB/s-bound), keep leaves (mandatory), lossless within the native-floor. CPU read-only, ONLINE-FIRST, adversarial verify. Output FR13_TREE_VERIFY_LITERATURE.md.',
  phases: [
    { title: 'Research' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn + 16 full-attn layers).',
'GB10 = 273 GB/s LPDDR5X, B=1 decode is HBM-BANDWIDTH-BOUND (so any fix must add ~0 memory traffic). Repo',
'/home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU committer-margin probe runs concurrently - do NOT',
'edit code/boot). SEARCH ONLINE FIRST (arXiv/papers/blogs/vLLM+SGLang issues) for the tree-verify-losslessness',
'state of the art, THEN map to our case + read our committer. Write ONLY FR13_TREE_VERIFY_LITERATURE.md. Pathspec.',
'',
'OUR SITUATION (established, all verify HOLDS this session):',
'- cat9 = 9-node caterpillar (depth-5 spine 0-1-3-5-7 + 4 leaves). Lossless bar = per-served-token argmax ==',
'  the no-spec RECURRENT decode oracle, at native-E5 LEVEL (native MTP-5 = 3 clear-margin flips = the WITHIN-',
'  FLOOR bar, NOT abs-0.0). cat9 = 23 raw / 18 de-cascaded flips.',
'- The committer = top-down LCP (_lumo_tree_path_lcp_max_greedy_sample, scripts/fr10_phase4_patch_vllm_tree_gdn.',
'  py ~L6818-6976): for each root-to-leaf path lcp=longest prefix where drafts[node]==parent_targets[node]',
'  (parent_targets=argmax of the VERIFY forward at node); best_path=max-lcp, TIE-BREAK earliest-leaf (=spine for',
'  sorted trees) so spine wins on EQUAL lcp; commits drafts[node] for the lcp prefix + a bonus token.',
'- THE CARRIER (FR13_LEAF_CORESIDENCY_PATH, FR13_NOCOPY_LOSSLESS_LEAVES, both verify HOLDS): the flips are an',
'  LCP-committer TRAJECTORY FORK - when a leaf path lcp >= spine lcp the committer commits drafts[leaf], a token',
'  the spine never serves. ROOT CAUSE = the verify forward (tree-batched, leaves co-resident, GDN tree-scan) is',
'  a slightly DIFFERENT numerical realization than the single-sequence DECODE forward, so parent_targets (verify',
'  argmax) != decode argmax at a few near-tie nodes -> the exact-match LCP picks a leaf -> flip. NOT an algorithm',
'  bug; LCP is lossless w.r.t. the verify forward; the gap is verify-vs-decode realization. Kernel fixes are',
'  closed: scan recompute made state bit-exact yet flips ROSE 23->32 (non-causal); K1 store-boundary = ~1/3;',
'  reshape-away-leaves = lossless-but-slow (REJECTED, leaves mandatory).',
'- The user-proposed fix = SPINE-BONUS / margin-damp: do NOT let a leaf win the lcp boundary on a SUB-FLOOR',
'  near-tie parent_target (commit the spine instead). Lossless ONLY for sub-floor ties (where verify was nearly',
'  indifferent so leaf-vs-spine is within realization noise); LOSSY if it suppresses a CONFIDENT genuine leaf',
'  win (>1 nat) = rejecting a real accept. The concurrent probe is measuring the per-fork deciding margin to',
'  classify near-tie(fixable) vs confident(fundamental).',
'',
'YOUR JOB - online-first literature survey + mapping:',
'1. SURVEY how production/research tree-verify methods commit + handle the verify-vs-target realization gap and',
'   near-tie path selection. For EACH (SpecInfer 2305.09781, Sequoia, EAGLE/EAGLE-2/EAGLE-3, SpecTr, Medusa,',
'   DeFT, STree 2505.14969, Traversal Verification 2505.12398, vLLM/SGLang rejection_sampler): what is the',
'   COMMIT rule (top-down longest-accepted-path? bottom-up? rejection sampling?), how do they TIE-BREAK, and do',
'   they assume verify==target EXACTLY (so they have no realization-gap notion) or handle drift? Cite specifics.',
'2. IS SPINE-BONUS A KNOWN LOSSLESS PATTERN: for GREEDY (temp-0) tree spec decoding, is a deterministic prefer-',
'   canonical-path / spine-bonus / margin-aware tie-break PROVABLY LOSSLESS, and under what condition? Contrast',
'   with LOSSY relaxations (MARS 2601.15498 runner-up acceptance; any top-k acceptance). State the exact margin',
'   rule under which preferring the spine is lossless (the verify top-2 at the deciding node within the float',
'   realization floor => the two tokens are indistinguishable to the target => committing either is lossless;',
'   beyond the floor => committing the spine is LOSSY). Is this in the literature or novel?',
'3. TRAVERSAL VERIFICATION (2505.12398) DEEP-READ: the bottom-up commit mechanism; does it PROVABLY avoid the',
'   leaf-fork (vs top-down LCP); is it COMMITTER-ONLY (no copy, no HBM tax); greedy + temp>0; does it fit our',
'   constraints + is it strictly better than spine-bonus-margin-damp? When would we prefer it.',
'4. MAP TO OUR LCP: our committer already tie-breaks spine-on-EQUAL-lcp; the fork is leaf-STRICTLY-GREATER. Does',
'   the literature support a margin-aware extension (spine keeps it unless the leaf beats by a clear verify-',
'   margin at the deciding node) as lossless-within-floor? Is the user-proposed spine-bonus exactly a known',
'   method, a sound novel tie-break, or does it need the probe to confirm all forks are sub-floor first?',
'',
'DELIVERABLE: FR13_TREE_VERIFY_LITERATURE.md = the per-method commit/tie-break table, the spine-bonus-lossless',
'verdict + the exact margin condition, the Traversal-Verification assessment, and the mapping to our LCP (is the',
'spine-bonus principled+lossless, and is it confined to sub-floor or does it need the probe). Distinguish',
'LITERATURE/cited from INFERRED. Note that WY is PARKED by user (do not recommend reviving it). Reward-hacks',
'banned (no copy/dense/multi-spine/HBM-tax; runner-up/top-k acceptance = LOSSY). Quote FR13_BUG_CLASS_PLAYBOOK',
'(#12 trajectory). Commit pathspec.',
].join('\n');

phase('Research');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['methodCommitTable','spineBonusLosslessVerdict','traversalVerificationAssessment','mappingToOurLCP','committed','notes'],
  properties: {
    methodCommitTable: { type: 'string', description: 'per-method (SpecInfer/Sequoia/EAGLE/SpecTr/Medusa/DeFT/STree/TraversalVerif/vLLM-rejection-sampler) commit rule + tie-break + whether they handle verify-vs-target drift; cited' },
    spineBonusLosslessVerdict: { type: 'string', description: 'is a spine-bonus / margin-aware tie-break PROVABLY lossless for greedy temp-0, under what exact margin condition; vs lossy relaxations (MARS/top-k); literature or novel?' },
    traversalVerificationAssessment: { type: 'string', description: 'Traversal Verification 2505.12398 bottom-up mechanism; provably avoids the fork? committer-only no-copy/no-HBM? fits constraints? vs spine-bonus' },
    mappingToOurLCP: { type: 'string', description: 'our LCP already spine-on-equal; is the margin-aware spine-bonus extension lossless-within-floor; does it need the probe to confirm sub-floor first or is it sound regardless' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  CTX + '\n\nTASK (Research, no GPU, read-only). SEARCH ONLINE FIRST, then read our committer L6818-6976. Do '
  + 'steps 1-4. Write FR13_TREE_VERIFY_LITERATURE.md, commit pathspec. Return the schema.',
  { label: 'tree-verify-literature', phase: 'Research', schema: R_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','methodsCited','spineBonusSound','traversalSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    methodsCited: { type: 'string', description: 'is the per-method survey from ACTUAL papers (cited, not invented), esp the commit/tie-break rules?' },
    spineBonusSound: { type: 'string', description: 'is the spine-bonus-lossless verdict + margin condition logically sound (lossless ONLY at sub-floor, lossy beyond), not hand-waved or smuggling a lossy top-k accept?' },
    traversalSound: { type: 'string', description: 'is the Traversal Verification assessment grounded in the actual paper (not invented), and the no-copy/committer-only claim correct?' },
    recommendation: { type: 'string', description: 'single: is the spine-bonus a principled lossless fix (+ does it need the probe sub-floor confirmation), or is Traversal Verification the better committer? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(r) + '. Default holds=false if methods/commit-rules are '
  + 'invented not cited from real papers, the spine-bonus verdict hand-waves losslessness (must state lossless '
  + 'ONLY within the realization floor, lossy beyond - no top-k/runner-up smuggling), the Traversal Verification '
  + 'mechanism is fabricated, or it recommends reviving WY (parked by user). No close/pass-fail; no reward-hack.',
  { label: 'verify-tree-verify-literature', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, v };
