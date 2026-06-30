export const meta = {
  name: 'fr13-fa2-tree-decode-feasibility',
  description: 'USER QUESTION: why are we using TREE_ATTN (separate Triton kernel, 0.00195 vs FLASH) for decode instead of OUR FA2-fork (native FLASH + additive -inf tree bias, byte-exact floor 14/16 whole-tree 0.0)? The FA2-fork is currently PREFILL-only (FR13_FA2_PREFILL_NATIVE). CPU read-only: root-cause WHY it is prefill-only, whether it can serve DECODE tree-verify + CUDA-graph-capture + B=4, and whether routing decode full-attn through it would CLOSE the 0.00195 deliverable-vs-E5 residual. Reward-hack check (FLASH+tree-mask is the user-sanctioned fallback, NOT a reroute). Adversarial verify.',
  phases: [
    { title: 'Investigate' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU replay-durable-state A/B workflow runs',
'concurrently from the same patcher - do NOT edit any code; write ONLY FR13_FA2_TREE_DECODE_FEASIBILITY.md).',
'Pathspec commits only.',
'',
'THE QUESTION (user): the locked build uses TREE_ATTN for the 16 full-attention layers in the cat9 decode',
'tree-verify (launcher FR13_TREE_ATTN_EXP2_SOFTMAX=1: a SEPARATE Triton unified-attention kernel with tl.exp',
'->tl.exp2 + reversed KV-block iteration, an ATTEMPT to bit-match native CUDA FLASH_ATTN). Memory records the',
'residual TREE_ATTN-vs-FLASH_ATTN = 0.00195. We ALSO have a FORKED FA2 kernel (FR13_FA2_PREFILL_NATIVE=1,',
'scripts/fr13_patch_fa2_tree_bias.py / the FR13_FA2_FORK work) = native FLASH_ATTN + an additive -inf TREE',
'BIAS applied after QK; its byte-exactness floor is 14/16 calls whole-tree 0.0, 2 single-ULP in ~1M',
'(irreducible MMA grouping, project_fr13_fa2_fork_nocopy_floor). The FA2-fork is currently used for PREFILL',
'ONLY. User hypothesis: route the DECODE full-attn through the FA2-fork instead of TREE_ATTN -> since it IS',
'native FLASH + a mask bias, it should be byte-exact to native FLASH (2-ULP floor) -> CLOSE the 0.00195.',
'',
'IMPORTANT AXIS CLARIFICATION (bind it, do not conflate): the 0.00195 is the DELIVERABLE losslessness axis',
'(cat9-TREE_ATTN-decode vs native-E5-FLASH MTP-5). It is NOT the same as the 21-flip active goal (cat9 tree-',
'verify vs cat9 OWN no-spec oracle - both arms use the same decode backend there, so the backend residual',
'CANCELS in the flip count). So this swap targets the DELIVERABLE (lossless within E5 floor), not the 21',
'flips. Both matter. Keep the two axes distinct in the writeup.',
'',
'YOUR JOB (read-only investigation):',
'1. WHY PREFILL-ONLY: read fr13_patch_fa2_tree_bias.py + the patcher FA2 wiring + how FR13_FA2_PREFILL_NATIVE',
'   is gated. What specifically restricts the FA2-fork to prefill? Candidates: the vLLM DECODE attention path',
'   is a different code path / kernel (paged KV-cache decode vs prefill varlen), the decode metadata',
'   (block_tables, seq_lens, the tree-depth positions) is not threaded to the fork, the additive bias is built',
'   for the prefill varlen layout not the decode paged layout, or the decode path was simply never wired.',
'2. CAN IT SERVE DECODE TREE-VERIFY: the cat9 decode tree-verify forward runs M=6-10 query rows (the tree)',
'   against the paged KV cache with a tree-ancestry mask. Can the FA2-fork (a) take the tree-depth positions +',
'   tree-ancestry additive bias in the DECODE path, (b) handle the paged KV-cache block layout, (c)',
'   CUDA-GRAPH-CAPTURE (TREE_ATTN was chosen partly for capture; does the FA2-fork capture?), (d) serve at',
'   B=4 (the real deliverable gate)? Read how TREE_ATTN does each of these in tree_attn.py and whether the',
'   FA2-fork has or lacks the equivalent.',
'3. WOULD IT CLOSE 0.00195: is the FA2-fork mathematically native-FLASH + bias (=> byte-exact floor) for the',
'   DECODE tree case, or does the decode paged-attention kernel differ from the prefill varlen kernel such',
'   that the fork would have a NEW residual? Find the exact decode FLASH kernel vLLM E5 uses and whether the',
'   fork wraps THAT or only the prefill kernel.',
'4. PRIOR HISTORY (git): the FA2-QPAD branch (9ad6793f/030a1c22, fr13-fa2-qpad) and the FR13_FA2_FORK lineage',
'   - what was decided about FA2-for-decode, why TREE_ATTN was picked, and the standing ruling (memory: "do',
'   NOT patch FLASH_ATTN until TREE_ATTN confirmed dead; first check TREE_ATTN cuda-captures+serves at B=4;',
'   0.00195 within E5 floor -> TREE_ATTN deploy wins, beyond -> FLASH_ATTN+tree-mask"). Is TREE_ATTN confirmed',
'   capturing+serving at B=4 yet? Is the 0.00195 known to be within or beyond the E5 self-noise floor?',
'5. REWARD-HACK CHECK: routing decode through the FA2-fork (FLASH + tree-mask) is the user-sanctioned FALLBACK',
'   and is "our fork" (a legitimate tree-attention deliverable), NOT a reroute/splice (the reroute ban is about',
'   routing OUR computation through native to PASS a metric while our kernel stays unused; here the FA2-fork',
'   IS the deployed verifier). Confirm this framing or flag if the fork would actually bypass our own work.',
'',
'DELIVERABLE: FR13_FA2_TREE_DECODE_FEASIBILITY.md = the why-prefill-only root cause + the decode/B4/graph-',
'capture feasibility + would-it-close-0.00195 verdict + the GPU test plan IF feasible (route decode full-attn',
'through the FA2-fork, measure cat9-vs-E5 on the 16 full-attn layers + e2e). Be SKEPTICAL (this session',
'overturned FA2-tile/BV/width single-carrier overstates); if the decode kernel differs from prefill such that',
'the fork would NOT be byte-exact, say so. Quote FR13_BUG_CLASS_PLAYBOOK.md rows. Commit pathspec.'
].join('\n');

phase('Investigate');
const I_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['whyPrefillOnly','canServeDecode','graphCaptureB4','wouldCloseResidual','treeAttnAliveAtB4','priorRuling','rewardHackCheck','gpuTestPlan','committed','notes'],
  properties: {
    whyPrefillOnly: { type: 'string', description: 'the SPECIFIC restriction keeping the FA2-fork prefill-only (code path / metadata / layout / never-wired), with file:line' },
    canServeDecode: { type: 'string', description: 'can the FA2-fork take tree-depth positions + tree-ancestry bias + paged KV decode layout? what is missing' },
    graphCaptureB4: { type: ['string','null'], description: 'can the FA2-fork CUDA-graph-capture + serve B=4 (vs TREE_ATTN which was chosen partly for this)?' },
    wouldCloseResidual: { type: 'string', description: 'would routing decode through the fork close the 0.00195 (is it native-FLASH+bias for the DECODE case, or does the decode paged kernel differ -> new residual)?' },
    treeAttnAliveAtB4: { type: ['string','null'], description: 'is TREE_ATTN confirmed capturing+serving at B=4 yet? is 0.00195 within or beyond the E5 self-noise floor?' },
    priorRuling: { type: 'string', description: 'the git history of FA2-for-decode + why TREE_ATTN was picked + the standing ruling' },
    rewardHackCheck: { type: 'string', description: 'is FA2-fork-for-decode a legitimate deliverable (FLASH+tree-mask, the sanctioned fallback) or would it bypass our own work?' },
    gpuTestPlan: { type: 'string', description: 'the GPU test IF feasible (route decode full-attn through the fork, measure full-attn cat9-vs-E5 + e2e 0.00195 closure)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const inv = await agent(
  CTX + '\n\nTASK (Investigate, no GPU, read-only). Do steps 1-5. Write FR13_FA2_TREE_DECODE_FEASIBILITY.md, '
  + 'commit pathspec. Return the schema.',
  { label: 'investigate-fa2-decode', phase: 'Investigate', schema: I_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','feasible','wouldCloseResidualVerdict','axisKeptDistinct','rewardHackOk','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    feasible: { type: ['boolean','null'], description: 'is FA2-fork-for-decode actually feasible (serves decode + captures + B=4), grounded in code?' },
    wouldCloseResidualVerdict: { type: 'string', description: 'grounded verdict: would it close 0.00195, or does the decode kernel differ from prefill?' },
    axisKeptDistinct: { type: 'string', description: 'did the writeup keep the 0.00195-deliverable axis distinct from the 21-flip axis (not conflate)?' },
    rewardHackOk: { type: 'string', description: 'confirm FA2-fork-for-decode is a legitimate deliverable not a reroute' },
    recommendation: { type: 'string', description: 'single recommendation: worth a GPU test (and gated how) or not (and why). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(inv) + '. Default holds=false if the feasibility claim is '
  + 'not grounded in the actual decode-vs-prefill kernel code, or if it conflates the 0.00195 deliverable axis '
  + 'with the 21-flip axis. The valuable output is a grounded yes/no on whether the FA2-fork would close 0.00195 '
  + 'in the DECODE tree case + whether it can capture/serve B=4. No close/pass-fail; no reward-hack.',
  { label: 'verify-fa2-decode', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { inv, v };
