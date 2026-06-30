export const meta = {
  name: 'fr13-replay-durable-state-ab',
  description: 'PIVOT (user confirmed YES): the PRIME un-examined channel. Our replay route _tree_gdn_replay_kernel writes the durable next-event GDN state; native MTP writes it with fused_sigmoid_gating_delta_rule_update. The byte-A/B that "passed" compared replay-vs-OUR-OWN-scan, NEVER vs native. Build an OBSERVE-ONLY cross-event A/B: replay cat9 accepted chain through BOTH kernels from the SAME cloned h0, record per-layer per-event max_abs(H_ours - H_native_seq) + first-nonzero + back-loading. nonzero+accumulating => the carrier (align OUR replay kernel bit-exact, NOT reroute); ~0 => faithful, pivot to TREE_ATTN. Reuse the PROVEN SUBOP_MAB sidecar-env + stage markers; operate on the LINEAR accepted chain to dodge the reduced-row assert that killed conv/scan 5x.',
  phases: [
    { title: 'Design' },
    { title: 'BootCapture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene: source .venv; recover_host_memory();',
'assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker rm -f the container + recover_host_memory.',
'',
'WHY THIS A/B (the PRIME lead from FR13_TOTAL_DRIFT_REANALYSIS_LEADS_BIND.md, user confirmed): the baked',
'cat9 shows 21 per-token-argmax flips vs its no-spec oracle (native 3). The reanalysis v2 found the flips',
'BACK-LOAD (flip-position norm mean 0.696 = concentrate LATE in the stream) = the signature of CROSS-EVENT',
'ACCUMULATION, corroborated by sglang #25587 (conv-state corruption after partial accept diverges after ~100',
'tokens) + STree 2505.14969 (recurrent-state replay is the bf16-sensitive path for GDN/Mamba hybrids, NOT the',
'per-forward scan). The locked build runs FR13_REPLAY_ROUTE=1 (ALWAYS ON): at commit it RE-EXECUTES the',
'accepted chain from h0 via _tree_gdn_replay_kernel (a rank-1 Triton kernel; fr10_gdn_tree_kernel.py ~:546,',
'patcher _fr13_replay_launch ~L7348) and writes the durable next-event GDN recurrent state into the bank.',
'Native MTP produces ITS durable state with the SEQUENTIAL recurrent kernel fused_sigmoid_gating_delta_rule_',
'update (reference_gdn_verify_sequential_dispatch). The byte-A/B that "PASSED" (lineage L27/L30,',
'FR13_REPLAY_GPU_GATES_BIND) compared replay-vs-OUR-OWN-scan-chain, NEVER vs native. A ~1-ULP/event durable-',
'state difference would ACCUMULATE across verify events = the ONLY mechanism that turns per-forward-bit-exact',
'kernels (scan/conv/fp8/gate all re-confirmed M-invariant on fresh read) into ~18 e2e back-loaded flips.',
'',
'THE A/B (OBSERVE-ONLY, the lossless reference for the durable state = the SEQUENTIAL recurrent state after',
'the accepted tokens; bit-exactness to sequential IS the spec, so aligning OUR replay to it is build-our-',
'kernel-bit-exact NOT a reroute - feedback_no_reroute_reward_hacking; the reroute would be replacing the',
'VERIFY scan, which we do NOT touch): on each commit event, after our replay writes H_ours, ALSO run native',
'fused_sigmoid_gating_delta_rule_update over the SAME accepted token chain from the SAME cloned h0/conv-state',
'(a CLONE, never the served bank), and record per-GDN-layer max_abs(H_ours - H_native_seq) + first-nonzero',
'layer + the per-EVENT progression (does the divergence GROW across events = back-loading). Served stream',
'UNTOUCHED (observe-only clone).',
'',
'CRITICAL - do NOT repeat the conv/scan A/B 5-failure history (all infra): (a) ENV-TO-WORKER is SOLVED - REUSE',
'the SUBOP_MAB patcher-import-time SIDECAR bridge (/logs/fr13_gdn_subop_mab.flag pattern; the mp/spawn',
'EngineCore worker env is CURATED so bare -e is unreliable) + the loud FR13_SUBOP_STAGE-style ERROR markers',
'(env-in-worker / engaged / record-written / arm-fail) so this can NEVER be silently vacuous (class 9). (b)',
'BOOT ENFORCE_EAGER=1 (the hook is eager-only; CUDA-graph replay bypasses it). (c) The accepted chain is a',
'LINEAR sequence (M = num_accepted), a STANDARD fused_sigmoid_gating call - this DODGES the reduced-row M5/M1',
'tree-slice geometry that device-asserted in causal_conv1d_update (the conv/scan A/B killer). If native',
'fused_sigmoid_gating asserts on ANY cloned geometry, PREVENT it kernel-valid (do NOT catch - a device assert',
'poisons the CUDA context, unrecoverable; same lesson as fe0af022).',
'',
'YOUR JOB:',
'PHASE 1 (Design, no GPU): read the replay site (fr10_gdn_tree_kernel.py _tree_gdn_replay_kernel ~:546 +',
'patcher _fr13_replay_launch ~L7348 + the commit/bank-write site) AND native fused_sigmoid_gating_delta_rule_',
'update (signature, the h0/conv-state/g/beta inputs, the durable-state output). Design the OBSERVE-ONLY A/B',
'hook (default-OFF env FR13_REPLAY_DURABLE_AB, gate behind a top-guard, sidecar-bridged like SUBOP_MAB, loud',
'stage markers). Clone h0+conv-state per event; run native seq kernel on the accepted chain; record',
'max_abs(H_ours - H_native_seq) per layer + first-nonzero + per-event index (for back-loading). PROVE default-',
'OFF byte-identical (locked cat9 path unaffected). Patcher AST-parses. CPU wiring tests. Commit pathspec (only',
'the patcher + tests). FAIL ready=false with the blocker if the durable-state interception point or the native',
'kernel inputs cannot be cleanly captured observe-only.',
'PHASE 2 (BootCapture, GPU - the ONLY boot): hygiene + boot cat9 ENFORCE_EAGER=1 with FR13_REPLAY_DURABLE_AB=1.',
'Watch the stage markers; assert env-in-worker + engaged + RECORD-WRITTEN (records>0) before trusting any',
'number; FAIL LOUD if disengaged. within_boot_det [T,T,T,T]. Read the jsonl: per-layer per-event',
'max_abs(H_ours - H_native_seq), first-nonzero layer/event, and whether it GROWS across events. Teardown +',
'recover; never leak.',
'DISCRIMINATOR: nonzero AND growing across events => the replay durable-state IS the back-loaded carrier ->',
'next step align OUR replay kernel bit-exact to the sequential reference (build-our-kernel, NOT reroute); ~0 =>',
'replay is faithful, the carrier is elsewhere (pivot to the TREE_ATTN full-attn query/KV M-invariance lead).',
'Reward-hacks BANNED (observe-only; the served stream is never altered). Quote FR13_BUG_CLASS_PLAYBOOK.md rows',
'(class 10 codegen-identity-not-spec-guaranteed, class 12 cross-event/co-residency, class 9 silent/vacuous).'
].join('\n');

phase('Design');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['interceptionPoint','nativeRefKernel','observeOnlyProof','assertPrevention','defaultOffProof','cpuTest','committed','ready','notes'],
  properties: {
    interceptionPoint: { type: 'string', description: 'where H_ours (our replay durable state) is intercepted + where the accepted chain + cloned h0/conv-state come from, with file:line' },
    nativeRefKernel: { type: 'string', description: 'how native fused_sigmoid_gating_delta_rule_update is invoked observe-only over the accepted chain (inputs, output durable state)' },
    observeOnlyProof: { type: 'string', description: 'proof the native arm uses a CLONE and never touches the served bank (served stream byte-unaffected)' },
    assertPrevention: { type: ['string','null'], description: 'is the native seq call kernel-valid for all cloned geometries (linear M=num_accepted, standard call)? any geometry that could assert + how prevented (not caught)' },
    defaultOffProof: { type: 'string', description: 'FR13_REPLAY_DURABLE_AB-OFF byte-identical (locked cat9 unaffected)' },
    cpuTest: { type: 'string', description: 'CPU wiring tests + pass count' },
    committed: { type: 'string' },
    ready: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (Design, no GPU). Do PHASE 1. Commit pathspec. Return the schema. ready=false with the blocker '
  + 'if the durable-state cannot be intercepted observe-only or the native kernel inputs are not cleanly available.',
  { label: 'design-replay-durable-ab', phase: 'Design', schema: D_SCHEMA, model: 'opus' }
);

phase('BootCapture');
const BC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['stageMarkersSeen','recordWritten','booted','eager','engaged','within_boot_det','recordsWritten','durable_max_abs_per_layer','first_nonzero_layer','grows_across_events','backLoading','noAssert','ok','notes'],
  properties: {
    stageMarkersSeen: { type: 'string' },
    recordWritten: { type: ['boolean','null'] },
    booted: { type: 'boolean' }, eager: { type: ['boolean','null'] }, engaged: { type: 'boolean' },
    within_boot_det: { type: 'string' },
    recordsWritten: { type: ['integer','null'] },
    durable_max_abs_per_layer: { type: ['string','null'], description: 'per-GDN-layer max_abs(H_ours - H_native_seq), RAW' },
    first_nonzero_layer: { type: ['string','null'] },
    grows_across_events: { type: ['boolean','null'], description: 'does max_abs GROW across successive verify events (back-loading)?' },
    backLoading: { type: ['string','null'], description: 'the per-event progression of the divergence' },
    noAssert: { type: ['boolean','null'], description: 'did NO device-side assert occur (kernel-valid held)?' },
    ok: { type: 'boolean' }, notes: { type: 'string' },
  },
};
let bc = null;
if (d && d.ready) {
  bc = await agent(
    CTX + '\n\nTASK (BootCapture, GPU). Design: ' + JSON.stringify(d) + '. Boot cat9 ENFORCE_EAGER=1 with '
    + 'FR13_REPLAY_DURABLE_AB=1 + the sidecar. Watch the stage markers; the goal is RECORD-WRITTEN + '
    + 'recordsWritten>0. Capture per-layer per-event max_abs(H_ours - H_native_seq), first-nonzero, and whether '
    + 'it GROWS across events (back-loading). Confirm noAssert. Teardown + recover. Return the schema.',
    { label: 'bootcapture-replay-durable', phase: 'BootCapture', schema: BC_SCHEMA, model: 'opus' }
  );
} else {
  log('Design not ready (durable-state not cleanly interceptable observe-only) -> SKIP boot; Verdict reports the blocker + alternative.');
}

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','captured','replayIsCarrier','backLoadingConfirmed','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    captured: { type: ['boolean','null'], description: 'did the A/B capture (records>0, no assert, markers green)?' },
    replayIsCarrier: { type: ['boolean','null'], description: 'is the replay durable-state nonzero AND growing across events (the back-loaded carrier)?' },
    backLoadingConfirmed: { type: ['boolean','null'], description: 'does the per-event growth match the reanalysis back-loading signature?' },
    nextAction: { type: 'string', description: 'if carrier: align OUR replay kernel bit-exact to the sequential reference (build-our-kernel, bring the alignment plan); if ~0: pivot to TREE_ATTN query/KV M-invariance lead. No close/pass-fail.' },
    rewardHackCheck: { type: 'string', description: 'confirm observe-only, served stream untouched, no reroute/splice' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Design ' + JSON.stringify(d) + ' BootCapture ' + JSON.stringify(bc) + '. '
  + 'Read the jsonl + stage markers. Default holds=false if not captured (records==0 / assert / disengaged) - do '
  + 'NOT report a verdict from a vacuous boot. If captured, conclude replay-is-carrier (nonzero+growing) vs '
  + 'faithful (~0), and whether back-loading is confirmed. Confirm observe-only (no reward-hack). No close/pass-fail.',
  { label: 'verdict-replay-durable', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, bc, v };
