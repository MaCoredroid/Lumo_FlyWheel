export const meta = {
  name: 'fr13-speed-campaign-deployment',
  description: 'USER (2026-06-15): lossless is ACCEPTED at the big-denom 13% (cat9 13.5% ~= native 14%, our honest lossless-vs-E5) - do NOT build the temp-0.6 gate. PIVOT to SPEED on the DEPLOYMENT regime (real SWE-Verified + codex, fr13_measure deploy-speed/deploy-lossless): do (A) the CAT-SHAPE SWAP lossless-gated + (B) the TWO speed levers (OPT-1 + OPT-A). Candidates: R4 cat6root (full spine + root sibling, ALL deep leaves removed = 6 nodes -> pad8, sheds cat9 pad16 h_cache step ~40ms = the lead reshape) + cat10 (cat9 + root sibling, revive). Levers: OPT-A (GB10 fp8 GEMV, BUILT e90de7ef, flag toggle) + OPT-1 (GPU committer sync-kill, build the unbuilt G2 = device-input + side-stream readback so the main thread stops blocking 91.9%, FR13_COMMITTER_SYNCKILL default-OFF). ALL default-OFF + lossless-by-construction (reshape = drafter packing only, levers = byte-identical OFF==ON). MEASURE on the deployment regime: deploy-speed (s/fwd + accept + derived TPS on a bounded codex run, truthful accounting, INSTRUMENT OFF) for the screen; LOSSLESS GATE = deploy-lossless flip-rate vs the recurrent oracle WITHIN native floor (the 13%-style) on the speed-promising candidates + OFF==ON byte-identical for the levers. Phase1 build (CPU) -> Phase2 deployment measure (GPU, serialized) -> Phase3 verify. Output FR13_SPEED_CAMPAIGN_DEPLOYMENT.md.',
  phases: [
    { title: 'Build' },
    { title: 'DeploymentMeasure' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, B=1 decode HBM-bound). Repo /home/mark/shared/lumoFlyWheel. Read code',
'+ git history + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec',
'commit (git commit -m ".." -- <files>, NEVER git add -A). GPU SERIALIZED + FREE; this wf is the only GPU user.',
'MAX 2 concurrent workflows. Convention: EXTEND the canonical infra; measure ONLY on the deployment regime.',
'',
'CONTEXT: lossless ACCEPTED at the big-denom 13% (cat9 ~= native, our honest lossless-vs-E5; the temp-0.6 gate is',
'NOT being built per the user). On the DEPLOYMENT regime (real SWE-Verified + codex), the first numbers show cat9',
's/fwd 0.248 / accept 3.685 / TPS 18.88 vs native 0.233 / 3.267 / 18.28 = cat9 TPS-competitive-to-faster (accept',
'edge real on real content). s/fwd is regime-robust (cat9 ~1.03-1.06x native per-forward). GOAL: push cat9 clearly',
'> native B=1 decode-TPS via the cat-shape swap + the two levers, LOSSLESS held (deployment flip-rate within',
'native floor). The raw-/v1/completions handrolled regime is DEPRECATED (off-distribution degenerate loop) - DO',
'NOT measure there; use fr13_measure deploy-speed/deploy-lossless on the codex+SWE deployment harness.',
'',
'THE DEPLOYMENT MEASUREMENT INFRA (use it, do NOT hand-roll - feedback_canonical_speed_lossless_infra): scripts/',
'fr13_measure.py deploy-speed (reduces the per-task /metrics brackets run_swe_bench_q36_a captures DURING the',
'codex loop -> s/fwd = d(decode_seconds)/d(drafts), accept/event, committed=accept+1, derived TPS; INSTRUMENT OFF',
'= clean; banned bases blocked) + deploy-lossless (recurrent-oracle flip-rate within-floor verdict) via scripts/',
'fr13_measure_orchestrate.sh (deploy-serve / deploy-speed / deploy-rescore / deploy-full). Boot via fr13_bigdenom',
'_swe_serve.sh (cat9 = fr13_launch_locked.sh, shapes/levers = TREE/flag overrides on the forked launcher). DEV-',
'iter = bounded codex (AGENT_WALL_S=360-600 on subset_astropy12907.json); prelaunch recover_host_memory + assert',
'>=95GiB + docker-empty per boot + teardown. TRUTHFUL accounting + INSTRUMENT ON/OFF carried (s/fwd OFF only).',
'',
'CANDIDATES + how to build each (all default-OFF, lossless-by-construction):',
'- OPT-A = GB10 fp8 GEMV (BUILT, e90de7ef, FR13_GB10_FP8_GEMV_CFG flag): NO build, just toggle the flag on the',
'  boot. Lossless = bit-identical (BLOCK_SIZE_K=128 pinned). Measure deploy-speed OFF vs ON on cat9 AND native.',
'- OPT-1 = GPU committer sync-kill (build the UNBUILT G2): the first-draft 10ebccac is GPU-resident for the',
'  DECISION but still does the FR13_EAGER_PACK :6760-6761 .synchronize() + kernel.py:471-474 .cpu().tolist() =',
'  ADDS a sync. G2 (FR13_SPEED_TUNING_PLAN_BRANCH_A.md): G2.a feed device tensors (skip the :6761 sync), G2.b',
'  side-stream the output readback + CUDA event (mirror native AsyncGPUModelRunnerOutput) + device->device',
'  writeback. New flag FR13_COMMITTER_SYNCKILL (under FR13_GPU_COMMITTER), default-OFF byte-identical. Capture-',
'  safe (committer eager, side-stream outside any captured region). Lossless = pure-integer location-only.',
'- R4 cat6root = [(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)] (6 nodes, depth-5, pad8): build the drafter',
'  packing as a NEW exact-match shape like cat3w (_fr10_is_cat6root + _fr10_cat6root_choices + torch.stack',
'  packing, ~15-30 lines, reuse the root-sibling slot from cat3w; spine_steps=4, root-sibling at slot 1). FAIL-',
'  LOUD on disengagement. Lossless gate = deploy-lossless flip-rate within native floor.',
'- cat10 = cat9 + (1,) root sibling [10 nodes, depth-5, pad16]: NEW exact-match shape (_fr10_is_cat10 + choices +',
'  packing). The revive (the prior 2.932 accept was a class-12 trajectory/sibling-stop DENOM artifact; measure on',
'  the deployment regime, depth-matched d5->native E5, per-event NOT cross-trajectory).',
'',
'LOSSLESS GATE (held per candidate, the user 13% bar): RESHAPE (R4/cat10) -> deploy-lossless flip-rate vs its OWN',
'recurrent oracle WITHIN native floor (not Wilson-separated above native, the 13%-style); LEVERS (OPT-1/OPT-A) ->',
'byte-identical OFF==ON (structural default-OFF) + deploy flip-rate unchanged. Each-arm-vs-own-oracle, NEVER cross',
'-boot. A candidate that improves TPS but FAILS lossless does NOT ship.',
].join('\n');

phase('Build');
const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['target','built','losslessByConstruction','offlineGate','citations'],
  properties: {
    target: { type: 'string' },
    built: { type: 'string', description: 'what was built (the exact code change, default-OFF flag/shape) + how to engage it on a boot' },
    losslessByConstruction: { type: 'string', description: 'the lossless argument (default-OFF byte-identical; reshape = drafter-packing-only; lever = pure-integer/bit-identical)' },
    offlineGate: { type: 'string', description: 'the offline byte-A/B gate result (default-OFF byte-identical; the shape engages / fails-loud; the sync-kill pure-integer) - CPU-validated' },
    citations: { type: 'string' },
  },
};
const builds = await parallel([
  () => agent(
    BASE + '\n\nTASK (Build branch = RESHAPE). Build R4 cat6root + cat10 as NEW exact-match drafter shapes in '
    + 'scripts/fr10_phase4_patch_vllm_tree_gdn.py (like cat3w :11005/:11021-11029/:11515-11538), default cat9 '
    + 'untouched, FAIL-LOUD unbuilt. Offline-gate (default path byte-identical, the shapes engage at their exact '
    + 'tree). Commit pathspec. Return the schema (target="reshape R4+cat10").',
    { label: 'build:reshape', phase: 'Build', schema: BUILD_SCHEMA, model: 'opus' }
  ).catch(() => null),
  () => agent(
    BASE + '\n\nTASK (Build branch = OPT-1 G2 sync-kill). Build the unbuilt G2 (FR13_COMMITTER_SYNCKILL, default-'
    + 'OFF, under FR13_GPU_COMMITTER): G2.a device-input (skip the :6760-6761 sync), G2.b side-stream readback + '
    + 'CUDA event + device->device writeback (per FR13_SPEED_TUNING_PLAN_BRANCH_A.md). Capture-safe, pure-integer '
    + 'lossless. Extend the existing fr13_gpu_committer byte-A/B gate with the device arm. Commit pathspec. Return '
    + 'the schema (target="OPT-1 G2 sync-kill").',
    { label: 'build:opt1', phase: 'Build', schema: BUILD_SCHEMA, model: 'opus' }
  ).catch(() => null),
]);
const goodBuilds = builds.filter(Boolean);
log('Build: ' + goodBuilds.length + '/2 branches');

phase('DeploymentMeasure');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['speedScreen','losslessGate','winner','committed','notes'],
  properties: {
    speedScreen: { type: 'string', description: 'deploy-speed (deployment regime, bounded codex) per arm: cat9 baseline, OPT-A on, OPT-1 on, R4 cat6root, cat10 - s/fwd + accept + derived TPS, which improve TPS vs cat9/native' },
    losslessGate: { type: 'string', description: 'the lossless gate on the speed-promising candidates: deploy-lossless flip-rate within native floor (reshape) + OFF==ON byte-identical (levers); which PASS lossless' },
    winner: { type: 'string', description: 'the candidate(s) that improve TPS AND hold lossless - the deployable speed win (vs native), with the deployment numbers' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  BASE + '\n\nTASK (DeploymentMeasure - USE GPU, serialized, prelaunch per boot). Builds: ' + JSON.stringify(goodBuilds)
  + '\nMeasure on the DEPLOYMENT regime (deploy-speed bounded codex): cat9 baseline + OPT-A-on + OPT-1-on + R4 '
  + 'cat6root + cat10. Screen TPS, then LOSSLESS-GATE the speed-promising candidates (deploy-lossless flip-rate '
  + 'within native floor + OFF==ON for levers). Report the winner(s). Commit results. Return the schema.',
  { label: 'deployment-measure', phase: 'DeploymentMeasure', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','measuredOnDeployment','losslessHeld','winnerSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    measuredOnDeployment: { type: 'string', description: 'were the numbers measured on the real codex+SWE deployment regime (deploy-speed/deploy-lossless), NOT the deprecated raw /v1/completions?' },
    losslessHeld: { type: 'string', description: 'is the lossless gate genuinely held per candidate (deployment flip-rate within native floor for reshapes; byte-identical OFF==ON for levers), not relaxed for speed?' },
    winnerSound: { type: 'string', description: 'is the claimed TPS win real (truthful s/fwd basis, deployment regime, accounting for the one-task/bounded caveats) - not a contaminated or hand-rolled number?' },
    recommendation: { type: 'string', description: 'single: which candidate(s) ship as lossless-AND-faster on deployment, and the final B=4/4-task confirm needed. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if any number is on the '
  + 'deprecated raw /v1/completions regime (must be deployment codex+SWE), if the lossless gate was relaxed/skipped '
  + 'for a candidate, if a TPS win uses a banned basis or ignores the one-task/bounded-codex caveat, or if a build '
  + 'is not actually default-OFF byte-identical. research-before-deadend; no reward-hack (WY parked). No close/pass-fail.',
  { label: 'verify-speed-campaign', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { builds: goodBuilds, m, v };
