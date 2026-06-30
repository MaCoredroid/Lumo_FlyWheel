export const meta = {
  name: 'fr13-leaf-coresidency-path',
  description: 'The reshape A/B (recurrent frame, verify HOLDS) isolated WIDTH / leaf co-residency as the +16 flip carrier: at FIXED depth-3, chain3 (NO width)=1 clear flip ≤ native 3 = LOSSLESS vs cat3w (+width)=~17 ≈ OFF cat9 18; depth adds only +1 (cat3w d3=17 vs cat9 d5=18). The M-keyed in_proj_ba co-residency GEMM was already FIXED (LUMO_FB pad, "batch-invariance exhausted") yet +16 width-flips remain. So WHERE does the residual leaf co-residency perturb the spine per-token realization? Trace the EXACT path from the FRESH cat3w-vs-chain3 captures + op-by-op kernel read: (a) shared GDN tree-scan h_cache/state (the path K1=waao62oj0 touches via the bf16 store-boundary), (b) tree-attention FA2-fork (leaves in the attention), (c) a residual M-dim GEMM not covered by in_proj_ba, or (d) the LCP committer walking a leaf-forked path. GOAL: confirm whether K1 is ON the carrier path (so K1 can work) + RANK the next lever if K1 is weak. CPU read-only, adversarial verify. Output FR13_LEAF_CORESIDENCY_PATH.md.',
  phases: [
    { title: 'Trace' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn + 16 full-attn layers).',
'Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU K1 boot runs concurrently - do NOT edit code',
'or boot; read FRESH captures + our code + vLLM source via scripts/vllm_src.sh; write ONLY',
'FR13_LEAF_CORESIDENCY_PATH.md). Pathspec commits.',
'',
'GROUNDING RULE (user, EMPHATIC): read vLLM source DIRECTLY from the pinned running image via',
'`scripts/vllm_src.sh <relpath>` (image vllm/vllm-openai@sha256:3dbe092e = 0.19.2rc1.dev134), NEVER a /tmp',
'cache. CHECK ARTIFACT TIMESTAMPS before reading any output.',
'',
'COMPARE TARGET (user): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle; the deployment-correct',
'oracle is the RECURRENT single-step decode (fr13_recurrent_decode_oracle), NOT prefill/chunked. native-E5 = 3',
'clear-margin flips = the bar. int-view NEVER atol. PROVE non-vacuity before trusting any number (4 vacuous',
'instruments burned this session, playbook #9).',
'',
'THE FRESH RESULT THAT FRAMES THIS (FR13_RESHAPE_AB_RECURRENT_BIND.md, verify HOLDS + monitor red-team',
'correction): clean FIXED-depth-3 A/B in the recurrent-oracle frame -',
'  chain3 (d3, NO width)   = 1 clear flip  [0,0,1,0]  accept/event 2.266  = LOSSLESS (<= native 3)',
'  cat3w  (d3, +width)     = 17 de-cascaded (raw 27)  accept/event 2.282  ~= OFF cat9 18',
'  cat9   (d5, +width=4lf) = 23 (de-cascade 18)        accept/event 3.198',
'  native (d5, no width)   = 3                          accept/event 3.08',
'=> WIDTH adds +16 at FIXED depth-3 (chain3 1 -> cat3w 17); depth adds only +1 (cat3w 17 -> cat9 18). The',
'carrier is the LEAVES / co-residency, NOT depth. Raw: output/fr13_reshape_boot/rescore/{cat3w,chain3}_recur_',
'flips.json + output/fr13_reshape_boot/{cat3w,chain3}_capture.json (FRESH 2026-06-15; do NOT read the stale',
'06-14 output/fr13_shape_sweep/ files). The leaves are ALSO the accept edge (cat9 3.198 > native), so they',
'cannot simply be removed (chain3 is slow). The lossless+FAST path = KEEP the leaves, CUT their co-residency.',
'',
'WHAT IS ALREADY KNOWN / RULED OUT (do not re-derive; build on it):',
'- The M-keyed in_proj_ba co-residency GEMM was FIXED (LUMO_FB pad, FR13_RESIDUAL13_RESOLVED); fp8 in_proj/',
'  o_proj are M-invariant (BLOCK_SIZE_M=64 constexpr); gate M-invariant. So the residual +16 width carrier is',
'  NOT a remaining M-keyed GEMM batch-variance - it is elsewhere.',
'- GDN scan STATE-feed alignment to native packed-decode (recompute) made the scan bit-exact (int-view 0.0)',
'  yet e2e flips ROSE 23->32 (trajectory change) - FR13_SCAN_NOT_E2E_CARRIER_BIND. K1 (waao62oj0) is the',
'  WEAKER in-place store-boundary version (per-node bf16 round-trip on the carried state, keeps geometry).',
'- FA2-fork = 2-ULP MMA-grouping floor (0.0039), "no depth growth" amplifier (FR13_REALIZATION_AGREEMENT,',
'  project_fr13_fa2_fork_nocopy_floor); diffusion deep-dive: largest per-LAYER ratios at deep FULL-ATTN',
'  L35/47/51/62 (where the FA2-fork amplifies), NOT GDN gate layers.',
'- Multi-spine copy-recurrent = NOT-lossless CLOSED; dense/splice/forced-spine BANNED; native = A/B oracle',
'  only. The leaves\' co-residency cut must be OUR kernel (no copy/dense).',
'',
'YOUR JOB - trace the EXACT path of the residual +16 leaf co-residency, from the fresh captures + op-by-op:',
'1. ISOLATE the carrier from the FRESH cat3w-vs-chain3 data: same depth-3, only difference = the 2 leaves',
'   (root sibling (1,) + d1 runner-up (0,1)). Read both _recur_flips.json: WHERE do cat3w\'s 17 flips occur',
'   (positions, per-prompt) that chain3\'s do NOT? Are they at the leaf-accept events (positions where a leaf',
'   token was committed / the tree branched) or spread? This localizes co-residency to leaf-touching steps vs',
'   a global batch effect.',
'2. TRACE THE MECHANISM op-by-op (read fr10_gdn_tree_kernel.py for OUR tree-scan + scripts/fr10_phase4_patch_',
'   vllm_tree_gdn.py for the committer/forward wiring, + vllm_src.sh for the native kernels): when a leaf is',
'   CO-RESIDENT with the spine in the same tree-verify forward, HOW does it perturb the spine\'s per-token',
'   logits? Candidate paths - attribute with evidence:',
'   (a) SHARED GDN tree-scan h_cache/state: do the leaf nodes write/read the same h_cache bank rows / SRAM as',
'       the spine, so the spine\'s carried state is realized differently when leaves co-reside? THIS is the',
'       path K1 (per-node bf16 store-boundary on state_i) touches - confirm whether K1 is ON it.',
'   (b) TREE-ATTENTION FA2-fork: do the leaves enter the attention (tree-bias mask) and change the spine\'s',
'       attention output realization (the deep-full-attn amplification)?',
'   (c) A residual M-dim / grid / num_warps effect: does adding leaf rows change the kernel launch geometry',
'       (BLOCK_M, grid, h_cache N_PAD spill) for the spine rows, even with in_proj_ba fixed?',
'   (d) LCP COMMITTER trajectory: do the leaves change which token the committer accepts (a different',
'       LCP-max path), forking the trajectory (the recompute-rose mechanism) rather than a per-forward seam?',
'3. K1 ON-PATH VERDICT: from (2), is K1 (scan-state store-boundary) ON the dominant co-residency path (so it',
'   CAN drop the +16) or OFF it (the co-residency is in tree-attn / committer-trajectory, so K1 will be ~0',
'   regardless)? This PREDICTS the K1 result + interprets it: if K1 fails AND it was off-path, the next lever',
'   is the on-path one; if K1 fails AND it was on-path, the co-residency is the leaves\' fundamental shared-',
'   forward perturbation (isolated-forward = FR9 block-pool surgery, no cheap vLLM-0.19 path) -> relax.',
'4. RANK THE NEXT LEVER (if K1 is weak): the candidate kernel fix for the on-path co-residency that KEEPS the',
'   leaves (the accept edge) - e.g. per-leaf isolated h_cache bank rows, a tree-attn realization alignment,',
'   etc. - each tagged OUR-kernel-authorized (numerics-align) vs reward-hack (copy/dense/splice = BANNED) vs',
'   block-pool-surgery (FR9, expensive) vs genuinely-fundamental (-> relax). With a CHEAP non-vacuous test.',
'',
'DELIVERABLE: FR13_LEAF_CORESIDENCY_PATH.md = the carrier-position localization (leaf-touching vs global), the',
'op-by-op mechanism attribution (a/b/c/d with evidence), the K1-on-path verdict (predicts + interprets the K1',
'result), and the ranked next-lever with a cheap test. Be SKEPTICAL + quantitative; distinguish MEASURED (from',
'the fresh captures + cited kernel lines) from INFERRED. Quote FR13_BUG_CLASS_PLAYBOOK rows (#12 co-residency/',
'trajectory, #10 codegen, #9 vacuous). Commit pathspec.',
].join('\n');

phase('Trace');
const T_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['carrierPositionLocalization','mechanismAttribution','k1OnPathVerdict','nextLeverRanked','cheapTest','committed','notes'],
  properties: {
    carrierPositionLocalization: { type: 'string', description: 'from fresh cat3w vs chain3 _recur_flips.json: WHERE cat3w 17 flips occur that chain3 1 does not - at leaf-accept/branch events or spread? leaf-touching vs global' },
    mechanismAttribution: { type: 'string', description: 'op-by-op: which path (a) shared GDN h_cache/state (K1 path), (b) tree-attn FA2-fork, (c) residual M/grid, (d) LCP committer trajectory - carries the +16, with cited kernel lines + evidence' },
    k1OnPathVerdict: { type: 'string', description: 'is K1 (scan-state store-boundary) ON the dominant co-residency path (can drop +16) or OFF it (~0 regardless)? predicts + interprets the K1 result' },
    nextLeverRanked: { type: 'string', description: 'if K1 weak: ranked next kernel lever for the on-path co-residency that KEEPS leaves - each tagged authorized-numerics-align vs reward-hack(banned) vs block-pool-surgery(FR9) vs fundamental(relax)' },
    cheapTest: { type: 'string', description: 'a CHEAP non-vacuous CPU or single-GPU-A/B to confirm the top next-lever' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const t = await agent(
  CTX + '\n\nTASK (Trace, no GPU, read-only). Do steps 1-4 from the FRESH cat3w/chain3 captures + op-by-op '
  + 'kernel read. Write FR13_LEAF_CORESIDENCY_PATH.md, commit pathspec. Return the schema.',
  { label: 'leaf-coresidency-trace', phase: 'Trace', schema: T_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','localizationGrounded','attributionOpLevel','k1VerdictSound','nextLeverClean','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    localizationGrounded: { type: 'string', description: 'is the carrier-position localization from the ACTUAL fresh cat3w/chain3 flip positions (re-derive), not asserted? + fresh not stale 06-14?' },
    attributionOpLevel: { type: 'string', description: 'is the mechanism attribution op-by-op with cited kernel lines (vllm_src.sh + our file), consistent with in_proj_ba-fixed + scan-recompute-rose + FA2-fork-floor?' },
    k1VerdictSound: { type: 'string', description: 'is the K1-on-path verdict logically sound given the mechanism + the recompute-rose prior?' },
    nextLeverClean: { type: 'string', description: 'is the next lever OUR-kernel-authorized (not a reward-hack copy/dense/splice/multi-spine; isolated-forward correctly flagged as FR9 block-pool surgery)?' },
    recommendation: { type: 'string', description: 'single: is K1 on-path (worth its boot) + what the next lever is. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(t) + '. Default holds=false if the carrier localization '
  + 'is not re-derived from the ACTUAL fresh cat3w/chain3 flip positions (or reads stale 06-14 files), the '
  + 'mechanism attribution is narrative not op-by-op with cited lines, the K1-on-path verdict is unsound, or any '
  + 'next lever is a banned reward-hack (copy/dense/splice/multi-spine) or mislabels isolated-forward\'s cost. '
  + 'No close/pass-fail; no reward-hack.',
  { label: 'verify-leaf-coresidency', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { t, v };
