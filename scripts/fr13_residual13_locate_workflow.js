export const meta = {
  name: 'fr13-residual13-coresidency-locate',
  description: 'Code-read + online research to localize the +13 RESIDUAL leaf co-residency channels (in_proj_ba+out_proj padding fixed only ~4 of +17; cat9 22->18). RE-EXAMINE the channels the width-locate REFUTED (now suspect: fp8 in_proj_qkvz, gate, conv, full-attn KV) with the empirical +13 as anchor; expect MULTIPLE contributors (do NOT overstate a single carrier); rank + predict A/B signature + draft per-channel batch-invariance fixes; adversarial verify.',
  phases: [
    { title: 'Locate' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. READ-ONLY (read code/binds/git + online research, write ONLY',
'FR13_RESIDUAL13_HYPOTHESES.md). A GPU workflow (ba-proj GateOFF control) runs concurrently - do NOT modify',
'any kernel/patcher; pathspec commits only.',
'',
'EMPIRICAL ANCHOR (this is the ground truth that REFRAMES the prior analysis): the +17 leaf width',
'co-residency (cat9 22 vs pure-spine 5) is MULTI-CHANNEL, NOT one GEMM. Padding in_proj_ba AND out_proj to a',
'fixed M (LUMO_FB_KERNEL_ROWS=1, M-invariant a/b + out) dropped cat9 22 -> 18 (lossless, det [T,T,T,T],',
'accept/event 3.017 ~ native, leaf edge preserved) = the bf16 in_proj_ba/out_proj Split-K shift accounts for',
'ONLY ~4 of the +17 (~18%). The RESIDUAL +13 (18 vs pure-spine 5) is OTHER co-residency channel(s) on the',
'spine when leaves co-reside. (FR13_WIDTH_CARRIER_INPROJ_BA_BIND.md RESULT section.)',
'',
'CRITICAL - the prior width-locate (w6rbv6qot) was WRONG TWICE, do not trust it: (1) it rated in_proj_ba',
'★★★★★ as THE single carrier - empirically it is only ~4/17, a CONTRIBUTOR. (2) it REFUTED the other',
'channels (fp8 in_proj_qkvz "M-invariant", gate "row-local M-invariant", conv "row-occupancy M-invariant",',
'cross-branch mask "0.0", full-attn "inert") - but the empirical +13 PROVES at least one of those refutations',
'is WRONG. RE-EXAMINE each with fresh eyes + the +13 anchor. Expect MULTIPLE contributors summing to ~13; do',
'NOT overstate a single new carrier.',
'',
'CANDIDATE CHANNELS for the +13 (when leaves M=tree_n=10 co-reside with the spine M=5):',
'1. fp8 in_proj_qkvz M-keying: width-locate said M-invariant (single M-tile <=64, K=128 pinned, no Spark',
'   cfg). RE-VERIFY: is the fp8 w8a8_triton_block_scaled_mm truly M-invariant at M=10 vs M=5 on GB10? Read',
'   the actual triton kernel + the block-scale path; a per-M autotune or a tail-tile could still shift q/k/v.',
'2. FULL-ATTENTION KV co-residency: cat9 has 4 leaf rows in the 16 full-attn layers KV. The FA2-fork',
'   tree-bias masks cross-branch in SCORES, but the ATTENTION KERNEL processes M=10 rows - is the spine',
'   row attn_out M-keyed (more KV rows -> different FA2 kBlockM tiling / softmax accum / Split sizing)? The',
'   QPAD experiment (FR13_FA2_CARRIER_OVERTURNED) padded the QUERY tile but that was judged vs a different',
'   question; re-examine whether the leaf KV rows perturb the spine attn_out (scripts/fr13_patch_fa2_tree_bias.py',
'   + the FA2 kernel). This is a STRONG candidate (16 full-attn layers, each a chance to perturb).',
'3. gate (RMSNormGated norm(core_attn_out, z)): bf16 elementwise, claimed row-local. RE-VERIFY it has NO',
'   cross-row reduction that M-keys (the rms over the hidden dim is per-row, but check the kernel launch).',
'4. conv (causal_conv1d / our fused tree conv): claimed row-occupancy M-invariant. RE-VERIFY the per-row',
'   loop has no M-keyed batched GEMM/reduction.',
'5. Any OTHER bf16 GEMM/op on the spine data path that is M-keyed (the in_proj_ba lesson: bf16 cuBLASLt',
'   Split-K is M-keyed; find every other bf16 matmul - the lm-head is post-forward, but in-block?).',
'',
'OUTPUT: rank the +13 channels by likely contribution (summing to ~13), each with code evidence + the GPU',
'A/B SIGNATURE that would confirm it (so the next GPU experiment discriminates) + the targeted fix',
'(per-channel batch-invariance / padding, like LUMO_FB_KERNEL_ROWS for the GEMMs; a full-attn KV-pad or',
'fixed-M attention for channel 2). Note the LUMO_FB padding ALREADY covers in_proj_ba+out_proj - check if it',
'can be extended to in_proj_qkvz (fp8 - does padding even help an fp8 GEMM?) and whether a full-attn fix is a',
'different mechanism. Reward-hacks BANNED (real M-invariance / numerics match, not copy/dense/splice/reroute).',
'Batch-invariance #42960 is directive-authorized. Quote FR13_BUG_CLASS_PLAYBOOK.md rows. Online research:',
'fp8 block-scaled GEMM batch-invariance, FlashAttention-2 batch/seqlen-invariance, Thinking Machines #42960,',
'tree-attention KV co-residency numerics.'
].join('\n');

phase('Locate');
const LOCATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['rankedChannels','fp8Qkvz','fullAttnKV','gate','conv','otherBf16','abSignatures','targetedFixes','onlineFindings','notes'],
  properties: {
    rankedChannels: { type: 'string', description: 'the +13 channels ranked by likely contribution (summing ~13), each with code evidence; expect MULTIPLE, do not overstate one' },
    fp8Qkvz: { type: 'string', description: 're-verify: is fp8 in_proj_qkvz M-invariant at M=10 vs M=5 on GB10? read the actual triton w8a8 kernel + autotune/tail-tile' },
    fullAttnKV: { type: 'string', description: 'does the leaf KV co-residency perturb the spine attn_out in the 16 full-attn layers (FA2 kBlockM tiling / softmax accum M-keyed despite the score mask)?' },
    gate: { type: 'string', description: 're-verify gate RMSNormGated is truly row-local M-invariant (no cross-row reduction)' },
    conv: { type: 'string', description: 're-verify conv row-occupancy M-invariance' },
    otherBf16: { type: 'string', description: 'any other M-keyed bf16 op on the spine data path' },
    abSignatures: { type: 'string', description: 'per-channel: the GPU A/B signature (or targeted-flag test) that confirms it' },
    targetedFixes: { type: 'string', description: 'per-channel targeted M-invariance fix (LUMO_FB extension to qkvz? full-attn KV-pad? gate/conv?), each real not reward-hack' },
    onlineFindings: { type: 'string' },
    notes: { type: 'string' },
  },
};
const locate = await agent(
  CTX + '\n\nTASK (Locate, no GPU). Re-examine all 5 candidate channels with the +13 empirical anchor + fresh '
  + 'eyes (the width-locate refutations are suspect). Read the fp8 w8a8 triton kernel, the FA2-fork + FA2 kernel, '
  + 'the gate/conv, and find any other M-keyed bf16 op. Rank the +13 channels, give per-channel A/B signatures '
  + '+ targeted fixes. Write FR13_RESIDUAL13_HYPOTHESES.md, commit pathspec. Return the schema.',
  { label: 'locate-residual13', phase: 'Locate', schema: LOCATE_SCHEMA, model: 'opus' }
);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','topChannels','overstatementCheck','rewardHackCheck','cheapestGPUTest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    topChannels: { type: 'string', description: 'the top 1-3 +13 channels with decisive code evidence (or "genuinely needs the A/B to discriminate")' },
    overstatementCheck: { type: 'string', description: 'ADVERSARIAL: is any channel rated as a single dominant carrier without evidence (the width-locate H1 mistake)? force per-channel quantification' },
    rewardHackCheck: { type: 'string', description: 'are the targeted fixes real M-invariance/numerics matches, not reroute/copy/dense?' },
    cheapestGPUTest: { type: 'string', description: 'the single cheapest GPU experiment to discriminate the top channels (e.g. extend LUMO_FB to qkvz + measure; or a full-attn KV-pad flag; or the extended sub-op A/B)' },
    recommendation: { type: 'string', description: 'single recommendation for the next GPU experiment. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verify = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(locate) + '. Default holds=false if channels are not '
  + 'grounded in actual kernel code. HARD overstatement check (do not repeat the width-locate single-carrier '
  + 'mistake - the +13 is multi-channel, quantify each). Reward-hack check on the fixes. Give the cheapest '
  + 'discriminating GPU test. No close/pass-fail; no reward-hack.',
  { label: 'verify-residual13', phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' }
);

return { locate, verify };
