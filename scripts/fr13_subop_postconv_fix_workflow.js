export const meta = {
  name: 'fr13-subop-postconv-arm-fix',
  description: 'ONE-MORE-FIX (user): the SUBOP_MAB reduced-row (M5/M1) arm trips a device-side assert INSIDE FLA fused_post_conv_prep.py:215 (beyond Front B host-guard 8cdda4c4). Build a kernel-VALID reduced-row geometry so the conv-post-prep does not assert, then boot EAGER + capture deep-spine conv1d_out/scan_out M10-vs-M5 = the +13 discriminator (predicts ~0 depth-intrinsic). Rebuilt gate (6ed4bb4c) stage markers prove engagement.',
  phases: [
    { title: 'Fix' },
    { title: 'BootCapture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'CONTEXT: the SUBOP_MAB L0-GDN sub-op A/B is the empirical +13 residual discriminator (deep-spine',
'conv1d_out/scan_out M10-vs-M5; predicts ~0 = depth-intrinsic per FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC).',
'The rebuilt gate (HEAD 6ed4bb4c, FR13_SUBOP_MAB_REBUILD) FINALLY ENGAGES: stage markers proved env-in-worker',
'(via sidecar /logs/fr13_gdn_subop_mab.flag - bare -e is curated out of the mp/spawn EngineCore worker) +',
'call-site + engagement (layer=language_model.model.layers.0.linear_attn tree_n=10 deep_row=8 num_spec=1).',
'MUST boot ENFORCE_EAGER=1 (the hook is eager-only; CUDA-graph replay bypasses it - boot1 non-eager showed',
'worker-env marker only, no capture).',
'',
'THE REMAINING BLOCKER (task wo9l6gi98, the 5th, deepest): the reduced-row M5/M1 arm (the spine-slice + decode',
'arms that re-run the GDN sub-ops at reduced M to compare vs the M10 full-tree) hits a DEVICE-SIDE ASSERT',
'INSIDE the FLA Triton kernel fused_post_conv_prep (e.g. /tmp/vllm_live_019/.../fla/.../fused_gdn_prefill_post_conv.py:215),',
'BEYOND Front B host-side guard 8cdda4c4 (_guard_rows + prior conv/ssm bank-validity raises cover host indices,',
'NOT the kernel internal index/shape preconditions). The arm-fail stage marker catches the Python RuntimeError',
'cleanly, BUT a device-side assert POISONS the whole CUDA context -> the next legitimate fused_post_conv_prep',
'crashes the engine. So recordsWritten=0, no M10-vs-M5 number.',
'',
'YOUR JOB:',
'PHASE 1 (Fix, no GPU). Read the FLA fused_post_conv_prep kernel (fused_gdn_prefill_post_conv.py around :215 -',
'the assert / index expression) AND the SUBOP_MAB reduced-row arm in scripts/fr10_phase4_patch_vllm_tree_gdn.py',
'(_scan_arm / _conv_arm / the M5/M1 slice that calls the GDN sub-ops incl fused_post_conv_prep). ROOT-CAUSE',
'the exact precondition the reduced (M5/M1, deep_row=8, num_spec=1) geometry violates (likely: cu_seqlens /',
'seqlen / chunk-boundary / num_tokens / a chunk_indices or padded-row count the kernel computes from the full',
'tree but the sliced arm passes inconsistently -> an OOB program_id / tl.load). FIX = build a kernel-VALID',
'reduced-row geometry for the arm (pad the reduced rows to the kernel-required shape and slice the real rows',
'back, OR recompute cu_seqlens/chunk metadata consistently for the reduced arm) so fused_post_conv_prep does',
'NOT assert - PREVENT the assert (do NOT just catch it; a device-side assert is unrecoverable). This is the',
'SAME pad-to-valid-shape idea that worked for in_proj_ba, applied to the conv-post-prep arm. Observe-only',
'(per-arm cloned state); the M10 full-tree arm is the faithful reference and is untouched. Keep',
'FR13_GDN_SUBOP_MAB default-OFF + the locked cat9 path byte-identical. Add a CPU wiring test for the new',
'geometry. Commit pathspec (only the patcher + the test). FAIL with the blocker if the kernel precondition',
'cannot be satisfied from the reduced arm (then the arm may need to run at full M with a post-hoc row select).',
'PHASE 2 (BootCapture, GPU - the ONLY boot). Hygiene + boot cat9 ENFORCE_EAGER=1 with FR13_GDN_SUBOP_MAB=1',
'FR13_GDN_SUBOP_MAB_LAYER=language_model.model.layers.0.linear_attn FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=10. Watch',
'the FR13_SUBOP_STAGE markers; assert env-in-worker + engaged + RECORD-WRITTEN (the discriminator the prior',
'boot never reached). Read output/fr13_gdn_subop_mab/*.jsonl: deep-spine conv1d_out + scan_out M10-vs-M5 RAW',
'max_abs + first-nonzero. Teardown + recover; never leak.',
'DISCRIMINATOR: conv1d_out OR scan_out M10-vs-M5 != 0 => a paddable M-keyed L0-GDN op remains (align it); both',
'~0 => residual depth-intrinsic + FA2-downstream (BI exhausted at in_proj_ba, the empirical close). Reward-hacks',
'BANNED (observe-only; pad-to-valid is geometry not a value change). Quote FR13_BUG_CLASS_PLAYBOOK.md rows.'
].join('\n');

phase('Fix');
const FIX_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['assertRootCause','geometryFix','preventsNotCatches','defaultOffProof','cpuTest','committed','ready','notes'],
  properties: {
    assertRootCause: { type: 'string', description: 'the exact fused_post_conv_prep:215 precondition the reduced M5/M1 geometry violates (cu_seqlens/chunk/num_tokens/index), with file:line' },
    geometryFix: { type: 'string', description: 'the kernel-valid reduced-row geometry built (pad-to-shape / recomputed cu_seqlens-chunk metadata) so the kernel does not assert' },
    preventsNotCatches: { type: ['boolean','null'], description: 'does the fix PREVENT the device-side assert (not just catch it - a device assert is unrecoverable)?' },
    defaultOffProof: { type: 'string', description: 'FR13_GDN_SUBOP_MAB-OFF byte-identical (locked cat9 unaffected)' },
    cpuTest: { type: 'string', description: 'CPU wiring test for the new geometry + pass count' },
    committed: { type: 'string' },
    ready: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const fix = await agent(
  CTX + '\n\nTASK (Fix, no GPU). Do PHASE 1. Commit pathspec. Return the schema. If the kernel precondition '
  + 'cannot be satisfied from the reduced arm, ready=false with the blocker + the full-M-with-row-select fallback.',
  { label: 'fix-postconv-arm', phase: 'Fix', schema: FIX_SCHEMA, model: 'opus' }
);

phase('BootCapture');
const BC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['stageMarkersSeen','recordWritten','booted','eager','engaged','tok_per_draft','recordsWritten','conv1d_out_m10_vs_m5','scan_out_m10_vs_m5','firstNonzeroSubOp','noAssert','ok','notes'],
  properties: {
    stageMarkersSeen: { type: 'string', description: 'which FR13_SUBOP_STAGE markers fired (env/engaged/record-written/arm-fail)' },
    recordWritten: { type: ['boolean','null'], description: 'did the record-written marker FINALLY fire (recordsWritten>0)?' },
    booted: { type: 'boolean' }, eager: { type: ['boolean','null'] }, engaged: { type: 'boolean' },
    tok_per_draft: { type: ['number','null'] }, recordsWritten: { type: ['integer','null'] },
    conv1d_out_m10_vs_m5: { type: ['number','string','null'] },
    scan_out_m10_vs_m5: { type: ['number','string','null'] },
    firstNonzeroSubOp: { type: ['string','null'] },
    noAssert: { type: ['boolean','null'], description: 'did the fused_post_conv_prep device-side assert NOT recur (fix worked)?' },
    ok: { type: 'boolean' }, notes: { type: 'string' },
  },
};
let bc = null;
if (fix && fix.ready) {
  bc = await agent(
    CTX + '\n\nTASK (BootCapture, GPU). Fix: ' + JSON.stringify(fix) + '. Boot cat9 ENFORCE_EAGER=1 + the SUBOP '
    + 'env. Watch the stage markers; the goal is the RECORD-WRITTEN marker + recordsWritten>0 (the prior boot '
    + 'crashed at arm-fail before this). Capture deep-spine conv1d_out/scan_out M10-vs-M5 + first-nonzero. '
    + 'Confirm noAssert (fused_post_conv_prep did not device-assert). Teardown + recover. Return the schema.',
    { label: 'bootcapture-postconv', phase: 'BootCapture', schema: BC_SCHEMA, model: 'opus' }
  );
} else {
  log('Fix not ready (kernel precondition unsatisfiable from the reduced arm) -> SKIP boot; Verdict reports the blocker + the full-M fallback.');
}

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','captured','paddableCarrier','residualNature','empiricalClose','nextAction','issues'],
  properties: {
    holds: { type: 'boolean' },
    captured: { type: ['boolean','null'], description: 'did the A/B FINALLY capture (records>0, no assert)?' },
    paddableCarrier: { type: ['boolean','null'], description: 'conv1d/scan M10-vs-M5 != 0?' },
    residualNature: { type: 'string', description: 'nonzero => align that op; both ~0 => depth-intrinsic, BI exhausted at in_proj_ba (the EMPIRICAL close confirming the kernel evidence)' },
    empiricalClose: { type: ['boolean','null'], description: 'is the +13 residual now EMPIRICALLY closed (matches the kernel-evidence prediction)?' },
    nextAction: { type: 'string', description: 'single next action (proceed to OPT-1/OPT-A speed + B=4 / align a found op / accept kernel evidence if still blocked). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Fix ' + JSON.stringify(fix) + ' BootCapture ' + JSON.stringify(bc) + '. '
  + 'Read output/fr13_gdn_subop_mab/*.jsonl + the stage markers. Default holds=false if not captured (records==0 '
  + '/ assert recurred / blocked). If captured, conclude paddable vs depth-intrinsic + whether it confirms the '
  + 'kernel-evidence depth-intrinsic prediction. If STILL blocked after this one-more-fix, say so plainly (the '
  + 'kernel evidence stands; proceed to OPT-1/OPT-A speed + B=4). No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { fix, bc, v };
