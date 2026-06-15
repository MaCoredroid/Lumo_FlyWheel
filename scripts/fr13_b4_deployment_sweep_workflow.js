export const meta = {
  name: 'fr13-b4-deployment-sweep',
  description: 'USER (2026-06-15): consolidate everything into ONE B=4 deployment sweep (the OPT-1 fix is already committed 34a9e1ce; do NOT re-fix, just use + GPU-validate it). Build the remaining pieces then run the B=4 sweep sequenced after the build. BUILD (CPU): (a) the 3-3-3 wide shape [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2)] (depth-3, 3 candidates/depth, needs DRAFTER RANK-2 = top-3 logit read), (b) the deploy-temp06-drift infra = the TEMP-0.6 distributional lossless drift on the DEPLOYMENT regime (capture the spec verify-forward top-K logits q id-keyed on the deployment served stream + the no-spec RECURRENT decode p forced-decoded on the SAME stream + per-position TV(softmax(q/0.6),softmax(p/0.6)) each-arm-vs-own-oracle vs the DEPTH-MATCHED native floor - the trust-ledger #1 gap, ZERO Tier-A temp-0.6 lossless ever measured; extend fr13_measure deploy-temp06-drift). MEASURE at B=4 (MAX_NUM_SEQS=4 co-residency, the deployable regime, ~30min codex wall): capture native E3/E4/E5 at B=4 as the DEPTH BARS (E3/E4 UNMEASURED) + cat9 + OPT-1-ON (FR13_COMMITTER_SYNCKILL, validate no-crash + run-ahead census at B=4) + cat6root + cat10 + 3-3-3; per arm deploy-speed (s/fwd+accept+TPS) + deploy-lossless (temp-0 flip-rate) + deploy-temp06-drift (temp-0.6 TV); DEPTH-MATCH depth-3 {3-3-3,cat3w,chain3}->E3, depth-5 {cat6root,cat9,cat10}->E5. Scope-aware (prioritize the bars + key candidates, bounded codex, exclude degenerate forks #12, report incrementally). On fr13-speedfix. Phase1 build (CPU) -> Phase2 B=4 measure (GPU serialized) -> Phase3 verify. Output FR13_B4_DEPLOYMENT_SWEEP.md.',
  phases: [
    { title: 'Build' },
    { title: 'B4Measure' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, decode HBM-bound: 98.6ms weight-read floor at every deployment batch;',
'per-forward speed opts are dead ends, accept-per-forward is the only lever - reference_decode_hbm_bound_accept_',
'is_the_lever). Repo /home/mark/shared/lumoFlyWheel, BRANCH fr13-speedfix. Read code + git history + vLLM source',
'via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec commit on fr13-speedfix',
'(NEVER git add -A). GPU SERIALIZED + FREE; this is the only GPU user. MAX 2 concurrent workflows. EXTEND the',
'canonical fr13_measure infra (feedback_canonical_speed_lossless_infra); measure ONLY on the real SWE+codex',
'deployment regime (deploy-* subcommands), NEVER the deprecated raw /v1/completions (off-distribution degenerate',
'loop).',
'',
'STATE: the OPT-1 G2 sync-kill FIX is ALREADY COMMITTED (34a9e1ce, "fix synckill committer crash + live-dispatch',
'composition gate") - do NOT re-fix; USE it (FR13_COMMITTER_SYNCKILL=1 under FR13_GPU_COMMITTER=1) and GPU-VALIDATE',
'it (boots without EngineCoreDead + the run-ahead census). The 3-3-3 shape + the deploy-temp06-drift infra are NOT',
'yet built. cat9/cat3w/cat6root/cat10/chain3/chain5 shapes ARE built (exact-match, default-OFF). deploy-speed +',
'deploy-lossless EXIST (fr13_measure); deploy-temp06-drift does NOT (build it).',
'',
'B=4 (user "starting now we should do B=4, it is 30min wall anyway"): MAX_NUM_SEQS=4 co-residency = the deployable',
'regime + where the FR13 co-residency effects live (the B=1 screen could not see them). s/fwd is ~B-invariant',
'(HBM-bound) but ACCEPT is B-dependent (co-residency may DEGRADE it) - that is the open question. DEPTH-MATCH',
'(feedback_depth_matched_accept_compare): each tree vs native MTP-of-its-depth - depth-3 -> native E3 (MTP-3),',
'depth-5 -> native E5; native E3/E4 accept are UNMEASURED (trust ledger) - capture at B=4 as the bars FIRST.',
'',
'TEMP-0.6 DISTRIBUTIONAL DRIFT (the trust-ledger #1 gap, user re-added): the deployment-binding lossless gate is',
'per-position TV(softmax(q/0.6), softmax(p/0.6)), q = the spec verify-forward top-K logits (NOT argmax - argmax is',
'temp-0, necessary-not-sufficient), p = the no-spec RECURRENT decode top-K (forced-decode the SAME served stream,',
'FR12_NO_SPECULATIVE_CONFIG=1, RECURRENT_PATH_ENGAGED, NOT chunked/streamed/serial). q is NOT captured on the',
'deployment regime today - BUILD it: forced-decode the deployment served stream through the SPEC serve to log the',
'verify top-K (q, id-keyed), + through the recurrent oracle (p); reduce per-position TV at temp 0.6, each-arm-vs-',
'its-OWN-oracle, vs the DEPTH-MATCHED native floor (cat9 vs native-E5 TV-floor; 3-3-3 vs native-E3 TV-floor). Pair',
'the scalar with the per-position vector (scalar-metric blindspot). PASS = cat9/tree TV not separated above its',
'depth-matched native floor.',
'',
'TRUTHFUL accounting + INSTRUMENT ON/OFF carried: s/fwd = d(request_decode_time_seconds_sum)/d(spec_drafts) OFF-',
'mode only; accept B-dependent + trajectory-bound (served_stream_fingerprint, like-for-like, exclude degenerate',
'forks = cat6root-r1 #12 lesson); TPS derived; deploy-lossless/temp06-drift = ON-mode. Prelaunch recover_host_',
'memory + assert>=95GiB + docker-empty per boot + teardown.',
].join('\n');

phase('Build');
const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['target','built','losslessByConstruction','offlineGate','citations'],
  properties: {
    target: { type: 'string' },
    built: { type: 'string', description: 'what was built + how to engage/invoke' },
    losslessByConstruction: { type: 'string', description: 'the lossless/correctness argument (default-OFF byte-identical for the shape; for temp06-drift, q/p forced on the same served stream, recurrent oracle not chunked, id-keyed not string)' },
    offlineGate: { type: 'string', description: 'offline CPU validation (shape engages/fail-loud + default byte-identical; temp06-drift reducer non-vacuous on synthetic/banked q,p, neg-control)' },
    citations: { type: 'string' },
  },
};
const builds = await parallel([
  () => agent(
    BASE + '\n\nTASK (Build branch = 3-3-3 SHAPE). Build the 3-3-3 shape [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),'
    + '(0,0,1),(0,0,2)] (depth-3, 3 candidates/depth) in fr10_phase4_patch_vllm_tree_gdn.py as a new exact-match '
    + 'shape like cat3w, EXTENDING the drafter to read the rank-2 (top-3) token from the same spine logits (no extra '
    + 'lm-head; child-rank>=2 was a packing limit). Default-OFF, fail-loud, offline-gate. If rank-2 is infeasibly '
    + 'invasive, FLAG + propose the nearest rank<=1 alternative. Commit pathspec. Return the schema (target="3-3-3").',
    { label: 'build:333', phase: 'Build', schema: BUILD_SCHEMA, model: 'opus' }
  ).catch(() => null),
  () => agent(
    BASE + '\n\nTASK (Build branch = deploy-temp06-drift INFRA). Extend fr13_measure with deploy-temp06-drift: '
    + 'forced-decode the deployment served stream through the SPEC serve to capture the verify top-K logits q (id-'
    + 'keyed) + through the recurrent oracle for p, reduce per-position TV(softmax(q/0.6),softmax(p/0.6)) each-arm-'
    + 'vs-own-oracle vs the depth-matched native floor + the per-position vector. Reuse the existing capture-q / '
    + 'recurrent_decode_oracle / deploy-rescore machinery (do NOT re-invent). Offline-gate the reducer non-vacuous '
    + '(id-aligned, n_scored>0, neg-control). Commit pathspec. Return the schema (target="deploy-temp06-drift infra").',
    { label: 'build:temp06drift', phase: 'Build', schema: BUILD_SCHEMA, model: 'opus' }
  ).catch(() => null),
]);
const goodBuilds = builds.filter(Boolean);
log('Build: ' + goodBuilds.length + '/2 branches');

phase('B4Measure');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['depthBars','speedScreen','losslessAndDrift','opt1','winner','committed','notes'],
  properties: {
    depthBars: { type: 'string', description: 'native E3/E4/E5 at B=4: s/fwd + accept + the temp-0 flip-rate floor + the temp-0.6 TV-floor (the depth-matched bars; E3/E4 were UNMEASURED)' },
    speedScreen: { type: 'string', description: 'deploy-speed at B=4 per arm (cat9, cat6root, cat10, 3-3-3) s/fwd+accept+TPS vs the depth-matched native; does any WIDER tree net-beat at B=4 (accept gain > co-residency cost)?' },
    losslessAndDrift: { type: 'string', description: 'deploy-lossless (temp-0 flip-rate) + deploy-temp06-drift (temp-0.6 TV) per arm vs the depth-matched native floor - which hold lossless at B=4 + temp-0.6' },
    opt1: { type: 'string', description: 'OPT-1 (FR13_COMMITTER_SYNCKILL ON) at B=4: boots no-crash (fix validated)? run-ahead census (block % OFF vs ON)? wall/s-fwd benefit? byte-identical OFF==ON?' },
    winner: { type: 'string', description: 'any candidate that is faster (TPS) AND lossless (flip-rate + temp-0.6 within floor) at B=4 vs its depth-matched native - or NONE, with cat9 as the standing deployable' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  BASE + '\n\nTASK (B4Measure - USE GPU, serialized, prelaunch per boot, bounded codex ~B=4 wall). Builds: '
  + JSON.stringify(goodBuilds) + '\nRun the B=4 deployment sweep: capture native E3/E4/E5 bars FIRST, then cat9 + '
  + 'OPT-1-ON + cat6root + cat10 + 3-3-3; per arm deploy-speed + deploy-lossless + deploy-temp06-drift, depth-'
  + 'matched. Validate the OPT-1 fix (no crash) + its B=4 run-ahead census. Prioritize the bars + key candidates, '
  + 'report incrementally, exclude degenerate forks (#12). Commit results. Return the schema.',
  { label: 'b4-measure', phase: 'B4Measure', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','onDeploymentB4','depthMatched','temp06Real','winnerSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    onDeploymentB4: { type: 'string', description: 'were the numbers on the real codex+SWE deployment regime at B=4 (MAX_NUM_SEQS=4), not raw /v1/completions or B=1?' },
    depthMatched: { type: 'string', description: 'is each tree compared to its DEPTH-MATCHED native (3-3-3->E3, cat6root/cat9/cat10->E5), with native E3/E4/E5 actually captured at B=4?' },
    temp06Real: { type: 'string', description: 'is the temp-0.6 distributional drift a REAL per-position TV (id-aligned q,p forced on the same stream, recurrent oracle not chunked, n_scored>0) - not the string/id artifact or a temp-0 argmax stand-in?' },
    winnerSound: { type: 'string', description: 'is any claimed TPS win real (truthful basis, like-for-like trajectory, not a degenerate-fork #12 confound) AND lossless (flip-rate + temp-0.6 within depth-matched floor)?' },
    recommendation: { type: 'string', description: 'single: which candidate ships as faster+lossless at B=4 (or cat9 stands); is OPT-1 a real lever; the temp-0.6 lossless verdict. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if any number is on raw /v1/'
  + 'completions or B=1 (must be deployment B=4), if a tree is not depth-matched (3-3-3 must be vs E3), if the '
  + 'temp-0.6 drift is the string/id artifact or a temp-0 argmax stand-in (must be real id-aligned TV n_scored>0), '
  + 'if a TPS win is a degenerate-fork #12 confound or not lossless, or if the OPT-1 crash was disabled not fixed. '
  + 'research-before-deadend. No close/pass-fail; no reward-hack (WY parked).',
  { label: 'verify-b4-sweep', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { builds: goodBuilds, m, v };
