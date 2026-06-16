export const meta = {
  name: 'fr13-fix-measure-opt1',
  description: 'USER (2026-06-15): fix + measure OPT-1 (the GPU-committer sync-kill, FR13_COMMITTER_SYNCKILL) - the ONE speed lever that attacks the overhead ABOVE the HBM floor (cat9 sits ~2.5x the 98.6ms weight-read floor; the main-thread committer DtoH+sync BLOCKS the launch thread 91.9% vs native 0.8% = lost run-ahead). It CRASHED un-measured in the campaign: EngineCoreDead at rejection_sampler.py - the G2.a host-input nulling (sets parents_cpu=None under synckill) does NOT compose with the legacy per-node loops that subscript parents_cpu (fr10_phase4_patch_vllm_tree_gdn.py:6975 + :8232). FIX: under FR13_COMMITTER_SYNCKILL, SKIP/guard those legacy loops + source the committer outputs from the device arm (fr13_gpu_committer_device_full); fix the byte-A/B gate to test the LIVE DISPATCH (the blind spot that missed the crash - it tested the device kernel in isolation). MEASURE on the deployment regime (bounded codex): the DIRECT proof = the run-ahead census (does the main-thread block drop 91.9% -> toward 0.8%?), independent of the s/fwd basis ambiguity; PLUS deploy-speed OFF vs ON AND wall-throughput OFF vs ON (OPT-1 axis may be WALL not decode_seconds - if decode_seconds is GPU-compute-only the sync tax is invisible to s/fwd, so measure wall tokens/sec too); LOSSLESS = served stream byte-identical OFF==ON (pure-integer location-only). On branch fr13-speedfix. Phase1 fix (CPU) -> Phase2 GPU measure -> Phase3 verify. Output FR13_OPT1_FIX_MEASURE.md.',
  phases: [
    { title: 'Fix' },
    { title: 'GPUMeasure' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, B=1 decode HBM-bound, 98.6ms weight-read floor). Repo',
'/home/mark/shared/lumoFlyWheel, BRANCH fr13-speedfix (the OPT-1 build lives here, NOT main). Read code + git',
'history + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec',
'commit on fr13-speedfix (git commit -m ".." -- <files>, NEVER git add -A). GPU SERIALIZED + FREE; only GPU user.',
'',
'OPT-1 = GPU-resident committer sync-kill (FR13_COMMITTER_SYNCKILL, default-OFF, under FR13_GPU_COMMITTER). It',
'removes the committer DtoH + .synchronize() that BLOCKS the main launch thread (census: cat9 91.9% main-thread',
'block in memcpyAsync vs native 0.8% = lost run-ahead; FR13_BEAT_NATIVE_SPEED_DESIGN_BIND). It is the ONLY lever',
'that attacks the ~2.5x-above-HBM-floor overhead (the reshape/OPT-A levers only touch per-forward compute, which',
'is HBM-bound = barely moves; campaign verdict NO candidate beat cat9). G2.a feeds device tensors (skip the sync),',
'G2.b side-streams the output readback + CUDA event + device->device writeback (mirrors native AsyncGPUModelRunner',
'Output run-ahead). Lossless = pure-integer location-only (the committer DECISION bytes are identical, only WHERE',
'inputs live + WHEN outputs cross to host changes).',
'',
'THE DEFECT (located, crashed the campaign): under FR13_COMMITTER_SYNCKILL the G2.a fork nulls the host committer-',
'input lists (parents_cpu etc.) but the LEGACY per-node loops at fr10_phase4_patch_vllm_tree_gdn.py:6975 + :8232',
'(`parents = parents_cpu[start:start+node_count]`) still run -> TypeError NoneType subscript -> EngineCoreDead.',
'FIX: under _FR13_COMMITTER_SYNCKILL, SKIP/guard those legacy per-node loops (they are replaced by the device arm)',
'+ source the committer outputs from fr13_gpu_committer_device_full (scripts/fr13_gpu_committer_kernel.py). The',
'CPU byte-A/B gate (fr13_gpu_committer_byte_ab_gate.py) PASSED but tested the device kernel IN ISOLATION, missing',
'the live dispatch composition - fix it to exercise the LIVE synckill dispatch path (or add a dispatch-composition',
'check) so it would have caught this. Keep default-OFF byte-identical (the :6975/:8232 legacy loops run verbatim',
'when synckill OFF).',
'',
'MEASUREMENT (the basis matters - OPT-1 axis may be WALL not decode_seconds): (1) DIRECT PROOF = the run-ahead',
'census - capture the main-thread block fraction (memcpyAsync / sync wait on the launch thread) OFF vs ON; OPT-1',
'works iff it drops from ~91.9% toward native 0.8% (this is independent of the s/fwd basis ambiguity). (2) deploy-',
'speed s/fwd OFF vs ON (d(request_decode_time_seconds_sum)/d(spec_drafts)) - IF decode_seconds includes the main-',
'thread block (wall-of-decode-step) the sync tax IS in s/fwd and OPT-1 lowers it; IF decode_seconds is GPU-compute-',
'only the tax is invisible here - DETERMINE which (read the vLLM instrumentation: what does request_decode_time_',
'seconds measure - wall of the decode step or GPU compute?). (3) WALL throughput OFF vs ON (returned_tokens / wall-',
'decode-seconds) - the deployment-faithful end number that DOES include the block; this is where OPT-1 must show',
'if (2) does not. Deployment regime (real codex+SWE bounded, instrument OFF for speed). LOSSLESS = the served',
'stream byte-identical OFF==ON (greedy, same boot, pure-integer location-only) - the lossless gate for a lever.',
'Compare like-for-like trajectory (cross-check drafts/gen_tok/verdict, exclude degenerate forks = the cat6root r1',
'#12 lesson).',
].join('\n');

phase('Fix');
const F_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['fix','gateFix','losslessByConstruction','offlineComposed','committed','notes'],
  properties: {
    fix: { type: 'string', description: 'the exact fix: how the :6975/:8232 legacy per-node loops are skipped/guarded under synckill + outputs sourced from the device arm; the engine no longer crashes' },
    gateFix: { type: 'string', description: 'how the byte-A/B gate now tests the LIVE synckill DISPATCH (not the kernel in isolation) so it would catch a composition defect' },
    losslessByConstruction: { type: 'string', description: 'default-OFF byte-identical (legacy loops verbatim when OFF) + pure-integer location-only when ON' },
    offlineComposed: { type: 'string', description: 'the offline composition gate result (the synckill dispatch produces the same committer outputs as the legacy/OFF path; default-OFF byte-identical), CPU-validated' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const f = await agent(
  BASE + '\n\nTASK (Fix, CPU, no GPU). Fix the located defect (guard the :6975/:8232 legacy loops under synckill + '
  + 'source outputs from the device arm) + fix the byte-A/B gate to test the live dispatch. Offline-gate the '
  + 'composition + default-OFF byte-identity. Commit pathspec on fr13-speedfix. Return the schema.',
  { label: 'fix-opt1', phase: 'Fix', schema: F_SCHEMA, model: 'opus' }
);

phase('GPUMeasure');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['engagesNoCrash','runAheadCensus','speedBasis','wallThroughput','losslessOffEqOn','verdict','committed','notes'],
  properties: {
    engagesNoCrash: { type: 'string', description: 'GPU: cat9 + FR13_COMMITTER_SYNCKILL=1 boots + serves the deployment codex run WITHOUT EngineCoreDead (the crash is fixed); engagement needle fired' },
    runAheadCensus: { type: 'string', description: 'the DIRECT proof: main-thread block fraction OFF vs ON - does it drop from ~91.9% toward native 0.8%? how captured' },
    speedBasis: { type: 'string', description: 'what request_decode_time_seconds MEASURES (wall-of-decode-step incl block, or GPU-compute-only) + the deploy-speed s/fwd OFF vs ON' },
    wallThroughput: { type: 'string', description: 'wall tokens/sec OFF vs ON on the deployment regime - the end number that includes the block; does OPT-1 improve it?' },
    losslessOffEqOn: { type: 'string', description: 'served stream byte-identical OFF==ON (greedy, same boot)?' },
    verdict: { type: 'string', description: 'is OPT-1 a REAL speed lever (run-ahead restored + wall/s-fwd improves) or not, with the numbers + the like-for-like caveat' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  BASE + '\n\nTASK (GPUMeasure - USE GPU, serialized, prelaunch recover_host_memory+assert>=95GiB+docker-empty per '
  + 'boot + teardown). The fix: ' + JSON.stringify(f) + '\nBoot cat9 + FR13_COMMITTER_SYNCKILL=1 on the deployment '
  + 'regime; confirm no crash; capture the run-ahead census (block % OFF vs ON), determine what decode_seconds '
  + 'measures, deploy-speed s/fwd OFF vs ON, WALL throughput OFF vs ON, and byte-identical OFF==ON. Verdict: is '
  + 'OPT-1 a real lever? Commit results. Return the schema.',
  { label: 'gpu-measure-opt1', phase: 'GPUMeasure', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','crashFixed','censusReal','benefitOnRightBasis','losslessHeld','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    crashFixed: { type: 'string', description: 'did it actually boot+serve without EngineCoreDead (the crash genuinely fixed, not just disabled)?' },
    censusReal: { type: 'string', description: 'is the run-ahead census a real measurement of the main-thread block (OFF vs ON), proving OPT-1 engages/works?' },
    benefitOnRightBasis: { type: 'string', description: 'is the speed benefit (if any) measured on the RIGHT basis - decode_seconds-s/fwd if it includes the block, ELSE wall throughput - not a banned/hand-rolled basis or a degenerate-fork confound?' },
    losslessHeld: { type: 'string', description: 'byte-identical OFF==ON confirmed (pure-integer location-only)?' },
    recommendation: { type: 'string', description: 'single: is OPT-1 a real deployable speed lever (ship it on top of cat9) or not, and the multi-task confirm needed. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if the crash was merely disabled '
  + 'not fixed (synckill must actually compose + serve), if the run-ahead census is asserted not measured, if a '
  + 'speed benefit is claimed on the WRONG basis (decode_seconds s/fwd when it excludes the block, or a banned/'
  + 'wall-jitter basis, or a degenerate-fork #12 confound), or if OFF==ON byte-identity is not confirmed. research-'
  + 'before-deadend. No close/pass-fail; no reward-hack.',
  { label: 'verify-opt1', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { f, m, v };
