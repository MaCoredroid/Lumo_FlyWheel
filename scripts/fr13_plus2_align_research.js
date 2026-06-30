export const meta = {
  name: 'fr13-plus2-spine-align-research',
  description: 'Locate the REAL +2 spine gap carrier (our-spine chain5 = 5 flips vs native 3, confirmed real not frame by the recurrent re-score) and draft the bit-exact alignment of our rank-1 tree-scan to native fused_sigmoid_gating recurrence (class-10 codegen, non-WY) to close 5 -> 3 = the last mile to native. Assess achievability; adversarial verify + reward-hack check.',
  phases: [
    { title: 'Locate' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. READ-ONLY (read code/binds/git +',
'online research, write ONLY FR13_PLUS2_SPINE_ALIGN_RESEARCH.md). A GPU workflow (the ba-proj +17 fix test)',
'runs concurrently and EDITS gdn_linear_attn.py + the forked launcher - do NOT modify any kernel/patcher;',
'pathspec commits only.',
'',
'CONTEXT (settled this session): cat9 22 = native-floor 3 + REAL +2 spine + REAL +17 leaf co-residency.',
'- The +17 leaf co-residency is LOCALIZED to the bf16 in_proj_ba GEMM (M-keyed) and is being FIXED in parallel',
'  (LUMO_FB_BATCH_INVARIANT_BA_PROJ ba-proj pad-to-fixed-M). Once it lands, cat9 -> ~5 = the pure-spine level.',
'- The +2 spine gap = our pure-spine chain5 = 5 flips vs native E5 (FLASH MTP-5) = 3, on the SAME prompts,',
'  each-vs-own no-spec oracle. CONFIRMED REAL, not a measurement frame: the recurrent re-score',
'  (FR13_ORACLE_FRAME_CLOSED_BIND) gave native 3/3 and spine 5/5 BYTE-IDENTICAL vs both chunked and recurrent',
'  oracles - the chunk-vs-recurrent reframe does NOT shrink the spine (q1 9.14x L0-magnitude reduction does not',
'  translate to e2e flips; the diffuse accumulation crosses the same high-entropy argmax boundaries).',
'',
'THE +2 IS THE LAST MILE TO NATIVE. Its carrier = the our-kernel-vs-native REALIZATION on a pure spine:',
'native uses its MTP-5 recurrence (fla fused_sigmoid_gating_delta_rule_update, the kernel vLLM ships+tunes);',
'ours uses the fr10 rank-1 TREE-scan (scripts/fr10_gdn_tree_kernel.py) - a DIFFERENT Triton codegen of the',
'SAME recurrence (class-10: shared-source != shared-SASS; num_warps / static_range-unroll / tl.load order /',
'cast boundaries / raw-g / l2norm). conv is already bit-exact (FR13_CONV_NOT_CARRIER, the ex2-silu/bf16-tap',
'grind SUCCEEDED). fp8 GEMMs M-invariant. So the +2 = the SCAN realization seam (and possibly the deep',
'full-attn L60/L61 amplification of it). WY (chunked-WY kernel) is the WRONG tool + PARKED: the verify path',
'is RECURRENT not chunked (reference_gdn_verify_sequential_dispatch) - the lever is recurrent-vs-recurrent',
'codegen alignment of our tree-scan to native fused_sigmoid_gating, the SAME family that cracked the conv',
'bf16-tap and the prior scan static_range->tl.range.',
'',
'YOUR JOB: (1) LOCATE which of chain5 the 5 flips are the +2 our-kernel excess vs native 3 (read',
'output/fr13_shape_sweep/chain5_flips.json + output/fr13_verify_decisive/q3_native_classify.json + the q1',
'deep-row finding output/fr13_verify_decisive/q1_recur_vs_chunked.json; are the +2 at specific positions/',
'sub-ops, or diffuse over all 5?); (2) DRAFT the bit-exact alignment of our fr10 tree-scan to native',
'fused_sigmoid_gating recurrence (the exact class-10 codegen knobs to match: reduction/op order, num_warps,',
'static_range vs range, tl.load order, cast/dtype boundaries, raw-g, l2norm-in-kernel) - read both kernels',
'(scripts/fr10_gdn_tree_kernel.py AND the live /tmp/vllm_live_019 or the DEPLOYED image fla/ops/',
'fused_sigmoid_gating.py - note the deployed image may differ from live_019, check which the locked build',
'uses); (3) ASSESS achievability: is 5 -> 3 reachable by codegen alignment, or is ~5 an irreducible',
'our-kernel floor (research-before-deadend: native same model/fp8/64 layers drifts to 3 = existence proof a',
'recurrent chain CAN sit at 3, FR13_DIFFUSE_GDN_EXPLAINED). Give a cheap GPU test for the top alignment knob.',
'',
'CONSTRAINTS: reward-hacks BANNED - the alignment must make OUR tree-scan kernel bit-exact to native by real',
'numerics matching, NOT route the spine through native fused_sigmoid_gating to pass a metric while our kernel',
'stays unchanged (feedback_no_reroute_reward_hacking, the FR12 lesson). batch-invariance #42960 is authorized.',
'Quote FR13_BUG_CLASS_PLAYBOOK.md rows. Online research: GatedDeltaNet/Qwen3-Next recurrence numerics, fla',
'fused_recurrent vs chunked, Triton codegen determinism, fp8/bf16 op-order. Name the GB10 context.'
].join('\n');

phase('Locate');
const LOCATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['plus2Localization','scanCodegenSeam','alignmentKnobs','achievability','cheapGPUTest','onlineFindings','notes'],
  properties: {
    plus2Localization: { type: 'string', description: 'which of chain5 5 flips are the +2 excess vs native 3 (positions/sub-ops, or diffuse); evidence from the flip JSONs + q1' },
    scanCodegenSeam: { type: 'string', description: 'the exact our-tree-scan vs native-fused_sigmoid_gating codegen difference (op order / num_warps / unroll / load order / cast / raw-g / l2norm), cited to both kernel files' },
    alignmentKnobs: { type: 'string', description: 'the ranked class-10 codegen knobs to drive our scan bit-exact to native recurrence, each non-WY non-reroute' },
    achievability: { type: 'string', description: 'is 5 -> 3 reachable by codegen alignment, or ~5 irreducible? with the existence-proof reasoning' },
    cheapGPUTest: { type: 'string', description: 'the single cheapest GPU experiment to validate the top alignment knob (e.g. an in-process scan A/B our-vs-native-recurrence on identical spine input, or a flag flip)' },
    onlineFindings: { type: 'string', description: 'relevant online research' },
    notes: { type: 'string' },
  },
};
const locate = await agent(
  CTX + '\n\nTASK (Locate, no GPU). Do (1)(2)(3). Write FR13_PLUS2_SPINE_ALIGN_RESEARCH.md, commit pathspec. Return the schema.',
  { label: 'locate-plus2-align', phase: 'Locate', schema: LOCATE_SCHEMA, model: 'opus' }
);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','topKnob','rewardHackCheck','reachable','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    topKnob: { type: 'string', description: 'the single most-likely alignment knob with decisive evidence (or "the +2 is irreducible / ~5 is the floor")' },
    rewardHackCheck: { type: 'string', description: 'ADVERSARIAL: is the alignment a REAL kernel-numerics match (our kernel made bit-exact) or a banned reroute (route the spine through native to pass)? be strict' },
    reachable: { type: ['boolean','null'], description: 'is 5 -> 3 reachable by aligning our kernel, or is ~5 the irreducible floor (=> accept ~5 as near-native, a user call)?' },
    recommendation: { type: 'string', description: 'single recommendation: pursue the alignment (cheap test first) or accept ~5. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verify = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(locate) + '. Default holds=false if the seam is not '
  + 'grounded in both kernel files. Pick the top alignment knob (or declare ~5 irreducible). Hard reward-hack '
  + 'check (real numerics match vs reroute). Is 5 -> 3 reachable? No close/pass-fail; no reward-hack.',
  { label: 'verify-plus2-align', phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' }
);

return { locate, verify };
