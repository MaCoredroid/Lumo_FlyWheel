export const meta = {
  name: 'fr13-chasedown-rebuilt-gate',
  description: 'Apply the 5 SUBOP_MAB rebuild EDITs (FR13_SUBOP_MAB_REBUILD.md c8061eda) + delete the stale BLOCKER comment, then run the EMPIRICAL L0-GDN sub-op A/B with the working gate: deep-spine conv1d_out/scan_out M10-vs-M5 (the +13 residual discriminator the 4 infra failures blocked). FR13_SUBOP_STAGE markers + worker-env gate confirm engagement before trusting any number. Predicts ~0 (depth-intrinsic).',
  phases: [
    { title: 'ApplyRebuild' },
    { title: 'BootCapture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'CONTEXT: the in_proj_ba pad is now BAKED into locked cat9 (a666f9ec, B=1 HOLD, lossless+speed-neutral). The',
'remaining question (user re-sequence: chase-down BEFORE final B=1) is the EMPIRICAL +13 residual',
'discriminator: deep-spine conv1d_out/scan_out M10-vs-M5 on the cat9 verify carrier. nonzero => a paddable',
'M-keyed L0-GDN op remains (align it); ~0 => depth-intrinsic (FR13_RESIDUAL13_RESOLVED_DEPTH_INTRINSIC, the',
'kernel evidence predicts ~0). The L0-GDN A/B (FR13_GDN_SUBOP_MAB) failed 4x on INFRASTRUCTURE; the rebuild',
'(FR13_SUBOP_MAB_REBUILD.md, c8061eda, verify holds=True) fixes it: 5 ready-to-apply EDITs = plain -e env +',
'PATCHER-IMPORT-TIME SIDECAR (/logs flag, the RELIABLE channel - the bake PROVED the mp/spawn worker env is',
'CURATED: LUMO_FB_PROJ_PAD_ROWS was -e-passed to pid-1 but DROPPED from the worker, so bare -e is NOT reliable;',
'the sidecar is) + a worker-env STAGE GATE + 4 ERROR-level FR13_SUBOP_STAGE markers (env/call-site/engagement/',
'capture) + HOIST the engagement asserts out of the swallowing try. Root cause of the 4 failures: (1) CUDA',
'assert reduced-row geometry (Front B 8cdda4c4 fixed, never exercised); (2,3) env via inert ray allowlist not',
'the working channel; (4) silent early-return at not-enabled (env absent in worker).',
'',
'YOUR JOB:',
'PHASE 1 (ApplyRebuild, no GPU): apply the 5 EDITs from FR13_SUBOP_MAB_REBUILD.md to',
'scripts/fr10_phase4_patch_vllm_tree_gdn.py VERBATIM (the doc has exact needles + replacements, all verified',
'unique + AST-valid against HEAD). Also DELETE the stale BLOCKER comment in',
'scripts/fr13_launch_forked_fa2_tree_server.sh (~L118-123, falsely claims the LUMO_FB pad block is not inserted',
'/ INERT - it is LIVE, the bake proved it). Add the rebuild §4 CPU wiring tests. PROVE FR13_GDN_SUBOP_MAB-OFF',
'is byte-identical (locked cat9 path unaffected - the bake just proved the OFF path is the live cat9). Patcher',
'AST-parses. Commit pathspec (only the patcher + the launcher + the tests). FAIL with the blocker if any EDIT',
'needle is not found verbatim (do NOT force-fit).',
'PHASE 2 (BootCapture, GPU - the ONLY boot): hygiene + boot cat9 with FR13_GDN_SUBOP_MAB=1',
'FR13_GDN_SUBOP_MAB_LAYER=language_model.model.layers.0.linear_attn FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=10. Watch',
'the FR13_SUBOP_STAGE=<tag> ERROR markers in the container log at EACH stage (env-in-worker / call-site-reached',
'/ engagement-pass / capture-written) - they make the instrument never-vacuous. ASSERT the env reached the',
'WORKER (/proc/<EngineCore pid>/environ OR the FR13_SUBOP_STAGE=ENV marker) BEFORE trusting any number; FAIL',
'LOUD if the stage markers show disengagement. engagement tok/draft==9 + within_boot_det [T,T,T,T]. Read',
'output/fr13_gdn_subop_mab/*.jsonl: deep-spine conv1d_out + scan_out M10-vs-M5 RAW max_abs + first-nonzero. This',
'is also the FIRST real exercise of the Front B bounds-guard (8cdda4c4) - watch for a clean Python raise vs the',
'prior CUDA assert. Teardown + recover; never leak.',
'DISCRIMINATOR: conv1d_out OR scan_out M10-vs-M5 != 0 => paddable M-keyed L0-GDN carrier in the residual (align',
'it); both ~0 => residual depth-intrinsic + FA2-downstream (BI exhausted at in_proj_ba, the empirical close).',
'Reward-hacks BANNED (observe-only). Quote FR13_BUG_CLASS_PLAYBOOK.md row 9 (silent/vacuous - the rebuild is the fix).'
].join('\n');

phase('ApplyRebuild');
const AR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['editsApplied','staleCommentDeleted','astParses','cpuTests','defaultOffProof','committed','ready','notes'],
  properties: {
    editsApplied: { type: 'string', description: 'the 5 EDITs applied verbatim (which needles); any not-found' },
    staleCommentDeleted: { type: 'string', description: 'the stale BLOCKER comment removed from the forked launcher' },
    astParses: { type: ['boolean','null'] },
    cpuTests: { type: 'string', description: 'the rebuild wiring tests added + pass count' },
    defaultOffProof: { type: 'string', description: 'FR13_GDN_SUBOP_MAB-OFF byte-identical (locked cat9 unaffected)' },
    committed: { type: 'string' },
    ready: { type: 'boolean', description: 'true only if all 5 EDITs applied + AST-valid + default-OFF byte-identical' },
    notes: { type: 'string' },
  },
};
const ar = await agent(
  CTX + '\n\nTASK (ApplyRebuild, no GPU). Apply the 5 EDITs + delete the stale comment + add the §4 tests. '
  + 'Commit pathspec. Return the schema. If an EDIT needle is not found verbatim, ready=false with the blocker.',
  { label: 'apply-rebuild', phase: 'ApplyRebuild', schema: AR_SCHEMA, model: 'opus' }
);

phase('BootCapture');
const BC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['stageMarkersSeen','workerEnvConfirmed','booted','engaged','tok_per_draft','within_boot_det','recordsWritten','conv1d_out_m10_vs_m5','scan_out_m10_vs_m5','boundsGuardOrCrash','firstNonzeroSubOp','ok','notes'],
  properties: {
    stageMarkersSeen: { type: 'string', description: 'which FR13_SUBOP_STAGE markers appeared (env/call-site/engagement/capture) - proves where it engaged or stopped' },
    workerEnvConfirmed: { type: ['boolean','null'] },
    booted: { type: 'boolean' },
    engaged: { type: 'boolean' },
    tok_per_draft: { type: ['number','null'] },
    within_boot_det: { type: 'string' },
    recordsWritten: { type: ['integer','null'] },
    conv1d_out_m10_vs_m5: { type: ['number','string','null'] },
    scan_out_m10_vs_m5: { type: ['number','string','null'] },
    boundsGuardOrCrash: { type: 'string', description: 'first real exercise of Front B bounds-guard 8cdda4c4: clean raise / clean capture / CUDA assert recurrence?' },
    firstNonzeroSubOp: { type: ['string','null'] },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
let bc = null;
if (ar && ar.ready) {
  bc = await agent(
    CTX + '\n\nTASK (BootCapture, GPU). ApplyRebuild: ' + JSON.stringify(ar) + '. Boot cat9 + the SUBOP env. '
    + 'Watch the FR13_SUBOP_STAGE markers; assert env-in-WORKER before capture (FAIL LOUD via the markers if '
    + 'disengaged). engagement tok/draft==9 + det [T,T,T,T]. Capture deep-spine conv1d_out/scan_out M10-vs-M5 '
    + '+ first-nonzero. Report boundsGuardOrCrash. Teardown + recover. Return the schema.',
    { label: 'bootcapture-rebuilt', phase: 'BootCapture', schema: BC_SCHEMA, model: 'opus' }
  );
} else {
  log('ApplyRebuild not ready (an EDIT needle not found) -> SKIP boot. Verdict reports the blocker.');
}

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','engagedCleanly','paddableCarrier','residualNature','nextLever','issues'],
  properties: {
    holds: { type: 'boolean', description: 'false if ApplyRebuild blocked, or stage markers show disengagement, or 0 records' },
    engagedCleanly: { type: ['boolean','null'], description: 'did the rebuilt gate FINALLY engage + capture (stage markers all green, records>0)?' },
    paddableCarrier: { type: ['boolean','null'], description: 'conv1d/scan M10-vs-M5 != 0?' },
    residualNature: { type: 'string', description: 'nonzero => align that op; both ~0 => depth-intrinsic, BI exhausted at in_proj_ba (empirical close)' },
    nextLever: { type: 'string', description: 'single next lever (proceed to final B=1 / align a found op). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ApplyRebuild ' + JSON.stringify(ar) + ' BootCapture ' + JSON.stringify(bc)
  + '. Read output/fr13_gdn_subop_mab/*.jsonl + the FR13_SUBOP_STAGE markers. Default holds=false if blocked / '
  + 'disengaged / 0 records (do NOT report depth-intrinsic from a vacuous boot - the whole point of the rebuild '
  + 'is to make engagement provable). Conclude paddable vs depth-intrinsic. No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { ar, bc, v };
