export const meta = {
  name: 'fr13-l0gdn-subop-ab-rerun',
  description: 'Run the FIXED FR13_GDN_SUBOP_MAB L0-GDN sub-op A/B (Front B fix 8cdda4c4, valid-bank init-state + bounds-guards) on cat9: capture deep-spine conv1d_out + scan_out FIRST-NONZERO at M=10 vs M=5 (decoherence-free, same captured input) = the decisive discriminator for the +13 residual. nonzero => paddable M-keyed L0-GDN carrier to align; ~0 => residual is depth/positional/FA2-downstream (batch-invariance exhausted at in_proj_ba).',
  phases: [
    { title: 'Capture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'GOAL: discriminate the +13 RESIDUAL leaf co-residency (cat9 22/26-cross-boot vs pure-spine 5; in_proj_ba',
'LUMO_FB pad already removed ~8 same-boot 26->18). The residual is NOT code-rankable (FR13_RESIDUAL13_DECISIVE_TEST_BIND.md,',
'residual-13 verify holds=FALSE). The decisive question: is there ANOTHER paddable M-keyed L0-GDN op (like',
'in_proj_ba) in the residual, or is the residual depth/positional/FA2-downstream?',
'',
'THE INSTRUMENT: the FIXED FR13_GDN_SUBOP_MAB in-process A/B (decoherence-free). Front B re-fixed the prior',
'CUDA-assert crash (commit 8cdda4c4 / d30755c8): the reduced-row M5/M1 arm now uses the valid committed prior',
'bank ssi0[0,0] as a 1-row initial_state at col 0 (nacc=None), and EVERY index/bank/seqlen is bounds-guarded',
'with a clean Python raise BEFORE the native kernel (so a CUDA assert cannot recur - it fails loud as a',
'Python error). 11/11 CPU wiring tests pass; default-OFF byte-identical (the locked cat9 path unaffected).',
'The hook is in HEAD. NOTE: this is the FIRST GPU validation of the fix (Front B was CPU-only) - if a',
'bounds-guard fires or it disengages, that is a clean fail-loud, report it (do NOT mask).',
'',
'WHAT IT MEASURES: on the deep-spine carrier verify event, re-run the GDN sub-ops at M=10 (full cat9 tree,',
'leaves co-resident) vs M=5 (spine-slice) vs M=1 (decode) reusing the SAME captured input + h0 + conv-state,',
'and record the deep-spine-row RAW max_abs of conv1d_out and scan_out for M10-vs-M5 (the CLEAN co-residency',
'verdict). The harness is BLIND to in_proj_ba (it slices the M=10 ba output) - that is intended; in_proj_ba',
'is already the banked fix. So this tests conv1d_out + scan_out = the right residual question.',
'',
'DISCRIMINATOR (the verdict): ',
'- conv1d_out OR scan_out M10-vs-M5 deep-spine != 0 => a paddable/alignable M-keyed L0-GDN carrier in the',
'  residual => align/pad it (more of the +13 fixable). [Note: residual-13 predicts conv1d_out~0 (conv',
'  row-M-invariant) and scan_out~0 (depth-intrinsic), so a NONZERO here is a surprise that re-opens a channel.]',
'- BOTH ~0 => the residual +13 has NO paddable L0-GDN op => it is the DEPTH-intrinsic chunk-vs-recurrent (R3,',
'  the +2 spine + depth-scaling, NOT batch-invariance-fixable) + small FA2-downstream correlate. Batch-',
'  invariance is then EXHAUSTED at in_proj_ba; remaining levers = served-len-pinned FA2 (R1) or accept the floor.',
'Also verify M5-vs-M1 ~0 (spine-slice == decode = no co-residency in the spine-only case, a sanity check).',
'',
'GATES: engagement tok/draft==9 (class-9, FAIL LOUD), within_boot_det [T,T,T,T] (class-8). Reward-hacks BANNED',
'(observe-only, per-arm cloned state). Quote FR13_BUG_CLASS_PLAYBOOK.md rows. gpuRunPlan (Front B): boot the',
'locked cat9 launcher + env FR13_GDN_SUBOP_MAB=1 FR13_GDN_SUBOP_MAB_LAYER=language_model.model.layers.0.linear_attn',
'FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=10 FR13_GDN_SUBOP_MAB_LIMIT=1 FR13_GDN_SUBOP_MAB_DUMP=output/fr13_gdn_subop_mab/fr13_gdn_subop_mab.jsonl,',
'run the cat9 capture (a few SWE prompts), read the JSONL.'
].join('\n');

phase('Capture');
const CAP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['booted','engaged','tok_per_draft','within_boot_det','recordsWritten','conv1d_out_m10_vs_m5','scan_out_m10_vs_m5','m5_vs_m1','firstNonzeroSubOp','ok','notes'],
  properties: {
    booted: { type: 'boolean' },
    engaged: { type: 'boolean' },
    tok_per_draft: { type: ['number','null'] },
    within_boot_det: { type: 'string' },
    recordsWritten: { type: ['integer','null'], description: 'number of JSONL A/B records (0 => crashed/disengaged = fail loud)' },
    conv1d_out_m10_vs_m5: { type: ['number','string','null'], description: 'deep-spine conv1d_out RAW max_abs M10-vs-M5 (the co-residency verdict for conv)' },
    scan_out_m10_vs_m5: { type: ['number','string','null'], description: 'deep-spine scan_out RAW max_abs M10-vs-M5 (the co-residency verdict for scan)' },
    m5_vs_m1: { type: 'string', description: 'M5-vs-M1 sanity (should be ~0)' },
    firstNonzeroSubOp: { type: ['string','null'], description: 'pre_conv | conv1d_out | scan_out | none' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const cap = await agent(
  CTX + '\n\nTASK (Capture, GPU - the ONLY boot). Hygiene + boot the locked cat9 with the FR13_GDN_SUBOP_MAB env '
  + '(per the gpuRunPlan above). Detached+poll if slow. Engagement tok/draft==9 + within_boot_det [T,T,T,T] '
  + '(FAIL LOUD else). Read output/fr13_gdn_subop_mab/*.jsonl; report deep-spine conv1d_out + scan_out '
  + 'M10-vs-M5 RAW max_abs + the first-nonzero sub-op. If recordsWritten==0 (crash/disengage/bounds-guard), '
  + 'ok=false with the exact failure. Teardown + recover; never leak. Return the schema.',
  { label: 'capture-l0gdn-ab', phase: 'Capture', schema: CAP_SCHEMA, model: 'opus' }
);

phase('Verdict');
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','paddableCarrier','residualNature','nextLever','issues'],
  properties: {
    holds: { type: 'boolean', description: 'false if the capture gate (engaged/det/records) failed' },
    paddableCarrier: { type: ['boolean','null'], description: 'is there a paddable M-keyed L0-GDN op in the residual (conv1d/scan M10-vs-M5 != 0)?' },
    residualNature: { type: 'string', description: 'conv1d/scan nonzero => align that op; both ~0 => residual is depth-intrinsic (R3) + FA2-downstream, batch-invariance exhausted at in_proj_ba' },
    nextLever: { type: 'string', description: 'the single next lever given the result (align a paddable op / served-len-pinned FA2 / accept the floor). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verdict = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(cap) + '. Read output/fr13_gdn_subop_mab/*.jsonl. Default '
  + 'holds=false if engaged/det/records failed. Conclude: paddable L0-GDN carrier in the residual (conv1d/scan '
  + 'M10-vs-M5 != 0) or both ~0 (depth-intrinsic + FA2-downstream, BI exhausted at in_proj_ba)? Give the single '
  + 'next lever. No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }
);

return { cap, verdict };
