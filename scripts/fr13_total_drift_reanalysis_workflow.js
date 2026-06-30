export const meta = {
  name: 'fr13-total-drift-reanalysis',
  description: 'Fresh independent CPU re-analysis (code scan + git history + online research) of the TOTAL cat9 drift: the BAKED-verified build is STILL 21 flips vs native 3. Re-derive the full accounting of all 21 from scratch (do NOT trust prior conclusions), RE-CHECK every ruled-out channel, hunt for any NEVER-examined source, and name the biggest remaining lever toward native. Adversarial verify.',
  phases: [
    { title: 'Reanalyze' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. READ-ONLY (read code + git log + binds + online research; write ONLY',
'FR13_TOTAL_DRIFT_REANALYSIS.md). A workflow edits the SUBOP_MAB hook region concurrently - do NOT modify any',
'kernel/patcher; pathspec commits only.',
'',
'THE FACT THAT MOTIVATES THIS: the in_proj_ba M-invariance fix is now BAKED into locked cat9 (a666f9ec, B=1',
'HOLD, lossless+speed-neutral), but the baked-VERIFIED build is STILL 21 clear-margin flips [3,6,6,6] vs',
'native 3 (per-token argmax vs its own no-spec decode oracle, thr 1.0 nat, prompts_swe4). So the in_proj_ba',
'channel was NOT the dominant driver - 21 of the original ~22 remain. The user wants a FRESH, INDEPENDENT',
'accounting of ALL 21 - re-derived from code + git history + research, NOT taken on faith from prior binds.',
'',
'WHAT PRIOR WORK CLAIMED (RE-CHECK each against code/git - some were overturned mid-session, treat ALL as',
'SUSPECT until you re-verify):',
'- 22 = native 3 + (+2 spine) + (+17 leaf co-residency). The +2 spine: claimed a class-12 CASCADE',
'  (FR13_PLUS2_DECASCADE: chain5 5 raw = 2 independent events) + a small FA2 2-ULP floor + diffuse GDN.',
'- +17 leaf co-residency: in_proj_ba bf16 GEMM M-keying (the baked fix, ~4-8 same-boot); residual claimed',
'  depth-intrinsic chunk-vs-recurrent + FA2-downstream.',
'- RULED OUT (re-verify EACH from the kernel code, not the bind): conv (row-M-invariant, our fused tree',
'  conv), GDN scan (bit-exact to native at BOTH BV geometries, FR13_BV_GEOMETRY RAW 0.0 D16=D32 N_PAD1/16),',
'  fp8 in_proj_qkvz + o_proj (M-invariant, w8a8_triton_block_scaled_mm BLOCK_SIZE_M=64 constexpr), gate',
'  (RMSNormGated M-invariant), BV/warps scan codegen (refuted), the chunk-vs-recurrent oracle frame (the',
'  recurrent re-score found it REAL not a frame: native 3/3, spine 5/5 byte-id, cat9 22/20 ours-only), reshape',
'  (depth dead chain3=chain5=5, width adds co-residency).',
'- The empirical L0-GDN sub-op A/B (conv1d/scan M10-vs-M5) is BLOCKED (5 infra failures, now the reduced-row',
'  arm device-asserts in FLA fused_post_conv_prep:215) - a concurrent fix is attempting it; do NOT depend on it.',
'',
'YOUR JOB (fresh + skeptical):',
'1. FULL ACCOUNTING of the 21 baked flips: native 3 + a quantified breakdown of the other 18. Read the baked',
'   flip records (re-run is blocked, use output/fr13_shape_sweep/*_flips.json + output/fr13_verify_decisive/',
'   q3_*_classify.json + the bake fingerprint [3,6,6,6] in FR13_BAKE_B1_HOLD_BIND). De-cascade (class-12) so',
'   the count is independent events. Which prompts/positions/boundaries; is it the same diffuse high-entropy',
'   boundary set as native (just more of them) or distinct positions?',
'2. RE-CHECK EVERY RULED-OUT CHANNEL against the actual kernel code (not the bind): is the scan REALLY',
'   bit-exact (re-read FR13_BV_GEOMETRY + the kernel)? Is fp8 REALLY M-invariant at the cat9 verify geometry?',
'   etc. Flag any ruling that does not hold up to a fresh read.',
'3. NEVER-EXAMINED SOURCES (the most valuable - hunt for what was missed): scan the FULL baked cat9 forward',
'   path end-to-end for drift sources NOT yet examined - e.g. the 16 full-attn layers beyond the FA2 tree-bias',
'   (RoPE/MRoPE position wiring, the q/k norm, the attn output gate), the lm-head / final norm, the sampler /',
'   committer / accepted-path selection, the eager-pack + conv-fused REPLAY (baked, always-on - is the replay',
'   itself a drift source?), the per-event h0/conv state handoff across verify events, cross-event accumulation.',
'   Use git log (git log --oneline, the FR13 commit lineage, the 27 remote branches) to see what was tried,',
'   overturned, or never finished.',
'4. THE BIGGEST REMAINING LEVER toward native (if any), and whether the residual is genuinely an irreducible',
'   diffuse/cascade floor (accept/event already ~native = sub-deployment-impact) or has a missed paddable/',
'   alignable channel. Online research: tree speculative decoding lossless gaps on GDN/Mamba/GatedDeltaNet',
'   hybrids, fp8 + tree co-residency numerics, diffuse bf16 accumulation across deep layers.',
'',
'Be SKEPTICAL of the prior accounting (this session overturned several premature conclusions: FA2-tile carrier,',
'depth model, BV seam, oracle frame). Quote FR13_BUG_CLASS_PLAYBOOK.md rows. Name the GB10 bandwidth context.',
'Reward-hacks BANNED. Write FR13_TOTAL_DRIFT_REANALYSIS.md, commit pathspec.'
].join('\n');

phase('Reanalyze');
const RE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['fullAccounting','rulingsRechecked','neverExaminedSources','gitHistoryFindings','biggestLever','irreducibleOrMissed','onlineFindings','notes'],
  properties: {
    fullAccounting: { type: 'string', description: 'the 21 baked flips = native 3 + a QUANTIFIED breakdown of the other 18 (de-cascaded to independent events), with positions/boundaries' },
    rulingsRechecked: { type: 'string', description: 'each ruled-out channel (conv/scan/fp8/gate/FA2/oracle/reshape) re-verified from kernel code: HOLDS or FLAGGED' },
    neverExaminedSources: { type: 'string', description: 'the most valuable: drift sources NOT yet examined (full-attn RoPE/qk-norm/out-gate, lm-head, sampler/committer, replay, cross-event handoff) + which look live' },
    gitHistoryFindings: { type: 'string', description: 'from git log + branches: what was tried/overturned/never-finished relevant to the residual' },
    biggestLever: { type: 'string', description: 'the single biggest remaining lever toward native (or none)' },
    irreducibleOrMissed: { type: 'string', description: 'is the 21 residual an irreducible diffuse/cascade floor (accept/event ~native = sub-impact) or a missed paddable/alignable channel?' },
    onlineFindings: { type: 'string' },
    notes: { type: 'string' },
  },
};
const re = await agent(
  CTX + '\n\nTASK (Reanalyze, no GPU). Do steps 1-4 fresh + skeptical. Write FR13_TOTAL_DRIFT_REANALYSIS.md, commit pathspec. Return the schema.',
  { label: 'reanalyze-total-drift', phase: 'Reanalyze', schema: RE_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','accountingComplete','anyFlaggedRuling','topNeverExamined','rewardHackCheck','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    accountingComplete: { type: 'string', description: 'does the breakdown sum to ~21 without double-counting or hand-waving? grounded in flip records + code?' },
    anyFlaggedRuling: { type: 'string', description: 'did the re-check overturn any prior ruled-out channel (a real new lead) or do they all hold?' },
    topNeverExamined: { type: 'string', description: 'the single most-likely never-examined source worth a cheap test, with code evidence (or none - residual is the diffuse floor)' },
    rewardHackCheck: { type: 'string', description: 'any proposed lever a real fix vs reward-hack/reroute?' },
    recommendation: { type: 'string', description: 'single recommendation: a new channel to test, or accept the diffuse floor (accept/event ~native) and proceed to speed/B=4. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(re) + '. Default holds=false if the accounting hand-waves '
  + 'or a re-check is not grounded in actual kernel code. The MOST valuable output is a never-examined source '
  + 'with code evidence OR a confident statement that the 21 is the irreducible diffuse/cascade floor (with '
  + 'accept/event ~native = sub-deployment-impact). No close/pass-fail; no reward-hack.',
  { label: 'verify-reanalysis', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { re, v };
