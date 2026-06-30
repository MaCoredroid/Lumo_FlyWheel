export const meta = {
  name: 'fr13-journey-reground',
  description: 'OWN the complete re-grounding of EVERY load-bearing code-read from the 22→3 carrier-hunt journey against the REAL running image (0.19.2 via scripts/vllm_src.sh), because they were originally read against the STALE /tmp/vllm_live_019 (0.19.0, 15/40/123-line drift). The staleness audit (w3hax7wlb) already re-verified conv/causal_conv1d + fused_recurrent packed-decode + fused_sigmoid_gating + commit 7441fc43 (all HELD, line#-drift only) — this workflow COVERS THE REST + consolidates ONE authoritative re-ground ledger, flagging any conclusion whose BASIS SHIFTED on the real source (= re-open). Read-only, adversarial verify.',
  phases: [
    { title: 'Reground' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU verify-boot',
'runs concurrently — do NOT edit code; write ONLY FR13_JOURNEY_REGROUND.md). Pathspec commits only.',
'',
'GROUNDING RULE (user, MANDATORY): read vLLM source DIRECTLY from the pinned running image via',
'`scripts/vllm_src.sh <relpath>` (vllm/vllm-openai@sha256:3dbe092e = 0.19.2rc1.dev134; --sha prints the digest;',
'no-arg re-extracts the tree to /tmp/vllm_cu130_src). NEVER read /tmp/vllm_live_019 (DELETED, was STALE 0.19.0)',
'or any /tmp cache. THE WHOLE POINT: the carrier-hunt binds cite vLLM line-numbers that are 0.19.0-keyed and',
'OFF by tens-to-hundreds of lines (causal_conv1d alone drifted 123 lines). Most CONCLUSIONS held when re-read',
'(the audit found line#-drift-only) but this workflow OWNS the COMPLETE check so nothing hides.',
'',
'ALREADY RE-GROUNDED (by the staleness audit w3hax7wlb, FR13_VERSION_STALENESS_AUDIT.md — cite as HELD, do NOT',
'redo): (1) conv / causal_conv1d.py — conv FIXED+CLOSED, 3-tap FIFO, spec offset=num_accepted-1 (real L852-866,',
'state_len width-1 L1183-1186) HOLDS; (2) fused_recurrent packed-decode — recurrent rank-1, 5 ops, num_warps=1',
'num_stages=3 (real :437-439) HOLDS; (3) fused_sigmoid_gating — sequential rank-1 dispatch (real def L24, ops',
'L144-153, warps=4 stages=3 L211-212) HOLDS; (4) commit 7441fc43 native-ref — correct (imports LIVE kernel).',
'',
'YOUR JOB — re-ground the REMAINING load-bearing code-reads of the journey, each vs the REAL 0.19.2 image, and',
'give a per-analysis verdict {HOLDS (conclusion valid, line# may drift) / SHIFTED (basis changed → which',
'conclusion re-opens)}. Enumerate from the binds; the major ones to cover (find their cited vLLM file:func):',
'- FA2 / TREE_ATTN decode path (FR13_FA2_FORK_IS_DECODE_KERNEL_CORRECTION): the decode dispatch where',
'  FR13_FA2_TREE_BIAS routes to flash_attn_varlen_func vs unified_attention — re-confirm the TreeAttentionImpl',
'  decode branch + the flash_attn_varlen_func(tree_bias=) signature exist as cited in the REAL image (the',
'  patcher fr13_patch_fa2_tree_bias.py edits a vLLM file — re-confirm the anchor/needle vs 0.19.2). Is "fork is',
'  the deployed decode kernel at 0.0039 floor; full-attn NOT the carrier" still grounded?',
'- mamba_utils cross-step state contract (the FR10 keystone, feedback_read_vllm_source_first): the',
'  prepare/postprocess_mamba accept_token_bias / curr_state_idx / in-place-no-copy logic (~L224-254 in the old',
'  read). Re-locate on 0.19.2 — does the cross-step read-base contract our committer/replay depends on still',
'  hold?',
'- gdn_linear_attn.py forward dispatch: the packed-decode gating (VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE,',
'  _forward_core num_prefills==0 + num_decodes>0, _forward_core_decode_non_spec → packed_decode) that',
'  fr13_native_packed_decode_ref + the scan-align fix rely on — re-confirm on 0.19.2.',
'- the GDN scan / SUBOP path: causal_conv1d_update spec-path state_len=width-1+(seqlen-1) (the 5x-assert root)',
'  + fused_post_conv_prep (the audit copy named a line that does not exist live — re-confirm the REAL FLA op',
'  the verify path calls).',
'- fp8 GEMM M-invariance (in_proj_qkvz / o_proj): w8a8_triton_block_scaled_mm BLOCK_SIZE_M=64 constexpr (the',
'  "M<=64 one-tile → M-invariant" claim) — re-confirm on 0.19.2.',
'- gate (RMSNormGated) ROWS_PER_BLOCK=1 per-row; rejection_sampler.py forward (the target-constraint line the',
'  committer composes with); any other bind that cites a vLLM file:line.',
'For EACH: vllm_src.sh the real file, find the cited construct, verdict HOLDS/SHIFTED + the REAL 0.19.2 file:line',
'(so binds can be annotated). Build the ONE consolidated ledger. If ANY conclusion SHIFTED (esp a carrier-search',
'closure), that is an ESCALATION — call it out at the top.',
'',
'DELIVERABLE: FR13_JOURNEY_REGROUND.md = the authoritative re-ground ledger (per-analysis: cited-construct →',
'real-0.19.2 file:line → HOLDS/SHIFTED), the audit-covered items cited as HELD, a SHIFTED list (or "none —',
'all line#-drift, conclusions intact"), and a corrected-citation table binds can adopt. Be SKEPTICAL +',
'thorough; this exists because a stale source already burned us once. Quote FR13_BUG_CLASS_PLAYBOOK rows (#10',
'codegen-identity, #11 naming-slip/version-skew). Commit pathspec.'
].join('\n');

phase('Reground');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['anyConclusionShifted','fa2TreeAttn','mambaUtilsContract','gdnForwardDispatch','scanSubopPath','fp8Minvariance','gateRejectionSampler','otherReads','correctedCitationTable','committed','notes'],
  properties: {
    anyConclusionShifted: { type: 'string', description: 'TOP-LINE: did re-reading the real 0.19.2 source SHIFT any carrier-hunt conclusion? list each SHIFTED (or "NONE - all line#-drift, conclusions intact")' },
    fa2TreeAttn: { type: 'string', description: 'FA2/TREE_ATTN decode dispatch re-grounded: HOLDS/SHIFTED + real file:line' },
    mambaUtilsContract: { type: 'string', description: 'mamba_utils cross-step state contract re-grounded on 0.19.2: HOLDS/SHIFTED + real file:line' },
    gdnForwardDispatch: { type: 'string', description: 'gdn_linear_attn packed-decode gating re-grounded: HOLDS/SHIFTED + real file:line' },
    scanSubopPath: { type: 'string', description: 'causal_conv1d_update spec-path / the real FLA verify op (fused_post_conv_prep audit-copy mismatch): HOLDS/SHIFTED + real file:line' },
    fp8Minvariance: { type: 'string', description: 'w8a8_triton_block_scaled_mm BLOCK_SIZE_M=64 M-invariance re-grounded: HOLDS/SHIFTED + real file:line' },
    gateRejectionSampler: { type: 'string', description: 'RMSNormGated + rejection_sampler target-constraint re-grounded: HOLDS/SHIFTED + real file:line' },
    otherReads: { type: 'string', description: 'any other journey code-read re-grounded' },
    correctedCitationTable: { type: 'string', description: 'cited-construct → real-0.19.2 file:line corrections binds can adopt' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  CTX + '\n\nTASK (Reground, no GPU, read-only). Re-ground EVERY remaining load-bearing journey code-read vs '
  + 'the REAL 0.19.2 image via vllm_src.sh. Write FR13_JOURNEY_REGROUND.md, commit pathspec. Return the schema.',
  { label: 'journey-reground', phase: 'Reground', schema: R_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','allGroundedInRealImage','anyMissedRead','anyShiftConfirmed','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    allGroundedInRealImage: { type: 'string', description: 'is each re-ground actually read from the real 0.19.2 image (vllm_src.sh), not re-asserted from a bind?' },
    anyMissedRead: { type: 'string', description: 'any load-bearing journey code-read NOT covered?' },
    anyShiftConfirmed: { type: 'string', description: 'did any conclusion genuinely SHIFT on the real source (re-open), or are all line#-drift-only?' },
    recommendation: { type: 'string', description: 'if a shift: which conclusion to re-open. If none: the journey is grounded, binds need only citation annotation. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(r) + '. Default holds=false if any re-ground is asserted '
  + 'from a bind rather than read from the real 0.19.2 image (spot-check 2-3 via vllm_src.sh yourself), or a '
  + 'load-bearing read was skipped. The valuable output is whether ANY conclusion SHIFTED (re-open) vs all '
  + 'line#-drift-only (journey intact, annotate citations). No close/pass-fail; no reward-hack.',
  { label: 'verify-journey-reground', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, v };
