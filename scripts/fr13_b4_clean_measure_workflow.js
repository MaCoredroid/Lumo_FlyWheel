export const meta = {
  name: 'fr13-b4-clean-measure',
  description: 'The CLEAN B=4 deployment sweep on the now-validated OFFLOADED harness (codex+proxy on alienware, GB10 = vLLM ONLY; OFFLOAD_CODEX=1 default, proven during-run + network-resilient, wvwik5sj4). Builds are DONE + committed on fr13-speedfix (3-3-3 d020e161, deploy-temp06-drift 1ca265cd, OPT-1 fix 34a9e1ce). MEASURE-ONLY at B=4 (MAX_NUM_SEQS=4 co-residency, the deployable regime + where the 273GB/s unified-mem contamination would bite hardest = exactly why we offloaded). Capture native E3/E4/E5 at B=4 as the DEPTH BARS (UNMEASURED) + cat9 + OPT-1-ON (FR13_COMMITTER_SYNCKILL, run-ahead census) + cat6root + cat10 + 3-3-3; per arm deploy-speed (per-DRAFT s/fwd AND per-TOKEN ms/tok - the B=1 validation found contamination hides in per-draft but shows per-token, so report BOTH) + deploy-lossless (temp-0 flip-rate) + deploy-temp06-drift (temp-0.6 TV); DEPTH-MATCH depth-3 {3-3-3,cat3w,chain3}->E3, depth-5 {cat6root,cat9,cat10}->E5. PLUS one cat9 B=4 CONTAMINATED contrast (OFFLOAD_CODEX=0, codex co-located on GB10) to QUANTIFY the contamination the offload removes at B=4 (the decisive number the B=1 validation could not pin). Scope-aware (bars + key candidates first, bounded codex, exclude degenerate forks #12, report incrementally). On fr13-speedfix. Phase1 B=4 measure (GPU serialized) -> Phase2 verify. Output FR13_B4_CLEAN_SWEEP.md.',
  phases: [
    { title: 'B4Measure' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (gx10-edb9 aarch64, UNIFIED MEMORY 273 GB/s; decode HBM-bound, 98.6ms weight-read floor;',
'accept-per-forward is the only speed lever - reference_decode_hbm_bound_accept_is_the_lever). Repo',
'/home/mark/shared/lumoFlyWheel, BRANCH fr13-speedfix. Read code + git history + vLLM source via scripts/vllm_src.sh',
'(pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec commit on fr13-speedfix. GPU SERIALIZED + FREE; the',
'only GPU user. MAX 2 concurrent workflows. Measure ONLY on the deployment regime (deploy-* subcommands).',
'',
'THE HARNESS IS NOW OFFLOADED + CLEAN (validated wvwik5sj4, verify HOLDS): codex+proxy run on alienware (x86,',
'tailscale 100.83.202.36), GB10 runs vLLM ONLY (proven during-run: GB10 docker=only vLLM, nvidia-smi=only',
'VLLM::EngineCore). OFFLOAD_CODEX defaults to 1 (offload.sh sync+start the alienware proxy on :8023 -> GB10:9950,',
'codex docker on alienware, pair-dumps rsynced back). The deploy-speed measurement reads the GB10 /metrics LOCALLY',
'(network-stall-immune by construction). USE the variant vehicle (fr13_bigdenom_swe_serve_variant.sh,',
'MAX_NUM_SEQS_OVR=4) with OFFLOAD_CODEX=1 for the CLEAN arms. Network-robust (real 25s blip survived in validation;',
'a wire-stalled window is flagged-not-recorded; watchdog fail-loud if link down >300s). Prelaunch recover_host_',
'memory + assert>=95GiB + docker-empty per boot + teardown.',
'',
'BUILDS DONE (do NOT rebuild, just engage): 3-3-3 [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2)] +',
'cat6root/cat10/cat3w/chain3/chain5/cat9 shapes (exact-match, default-OFF, TREE-override boot); deploy-temp06-drift',
'(q=spec verify top-K forced on the deployment stream, p=recurrent oracle, per-position TV(softmax(q/0.6),softmax',
'(p/0.6))); OPT-1 FR13_COMMITTER_SYNCKILL (fix 34a9e1ce, the run-ahead lever). native arms via the speed launcher',
'SPEC_CONFIG num_speculative_tokens=N (E3/E4/E5).',
'',
'B=4 + DEPTH-MATCH + TRUTHFUL: B=4 = MAX_NUM_SEQS=4 co-residency (deployable + where co-residency effects + the',
'unified-mem contention live). s/fwd ~B-invariant (HBM-bound) but ACCEPT is B-dependent (co-residency may degrade)',
'+ trajectory-bound (served_stream_fingerprint, like-for-like, exclude degenerate forks #12). Report per-DRAFT',
's/fwd (=d(request_decode_time_seconds_sum)/d(spec_drafts), the canonical basis) AND per-TOKEN ms/tok (=decode_s/',
'gen_tokens, where the B=1 contamination surfaced). committed=accept+1, TPS derived-NOT-measured. DEPTH-MATCH each',
'tree to native MTP-of-its-depth (3-3-3/cat3w/chain3 -> E3; cat6root/cat9/cat10 -> E5); native E3/E4 UNMEASURED -',
'capture at B=4 as the bars FIRST. deploy-lossless/temp06-drift = ON-mode (separate GB10 vLLM-only rescore phase).',
].join('\n');

phase('B4Measure');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['depthBars','speedScreen','contaminationContrast','losslessAndDrift','opt1','winner','committed','notes'],
  properties: {
    depthBars: { type: 'string', description: 'native E3/E4/E5 at B=4 CLEAN: per-draft s/fwd + per-token ms/tok + accept + temp-0 flip-floor + temp-0.6 TV-floor (the depth-matched bars; E3/E4 were UNMEASURED)' },
    speedScreen: { type: 'string', description: 'deploy-speed at B=4 CLEAN per arm (cat9, cat6root, cat10, 3-3-3) per-draft s/fwd + per-token ms/tok + accept + TPS vs the depth-matched native; does any WIDER tree (3-3-3) net-beat at B=4 co-residency?' },
    contaminationContrast: { type: 'string', description: 'cat9 B=4 CLEAN (OFFLOAD_CODEX=1) vs CONTAMINATED (OFFLOAD_CODEX=0, codex co-located on GB10): the s/fwd + ms/tok delta = how much the unified-mem contamination costs at B=4 (the decisive number)' },
    losslessAndDrift: { type: 'string', description: 'deploy-lossless (temp-0 flip-rate) + deploy-temp06-drift (temp-0.6 TV) per arm vs the depth-matched native floor at B=4 - which hold lossless (the temp-0.6 Tier-A number we never had)' },
    opt1: { type: 'string', description: 'OPT-1 (FR13_COMMITTER_SYNCKILL ON) at B=4 CLEAN: boots no-crash? run-ahead census (block % OFF vs ON)? s/fwd or wall benefit now that the box is uncontended? byte-identical OFF==ON?' },
    winner: { type: 'string', description: 'any candidate faster (TPS) AND lossless (flip + temp-0.6 within floor) at B=4 clean vs its depth-matched native - or cat9 stands' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  BASE + '\n\nTASK (B4Measure - USE GPU, serialized, prelaunch per boot; OFFLOAD_CODEX=1 clean harness via the '
  + 'variant vehicle MAX_NUM_SEQS_OVR=4). Capture native E3/E4/E5 bars FIRST, then cat9 + OPT-1-ON + cat6root + '
  + 'cat10 + 3-3-3; per arm deploy-speed (per-draft + per-token) + deploy-lossless + deploy-temp06-drift, depth-'
  + 'matched. PLUS the cat9 OFFLOAD_CODEX=0 contaminated contrast. Prioritize bars+key candidates, bounded codex, '
  + 'exclude #12 forks, report incrementally. Commit results. Return the schema.',
  { label: 'b4-clean-measure', phase: 'B4Measure', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','cleanOffloaded','depthMatched','temp06Real','winnerSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    cleanOffloaded: { type: 'string', description: 'were the numbers taken on the CLEAN offloaded harness (OFFLOAD_CODEX=1, GB10 vLLM-only during the run) at B=4, not the contaminated/B=1 path? the contamination contrast is a real OFFLOAD_CODEX=0 vs 1 pair?' },
    depthMatched: { type: 'string', description: 'each tree vs its DEPTH-MATCHED native (3-3-3->E3, cat6root/cat9/cat10->E5), native E3/E4/E5 captured at B=4?' },
    temp06Real: { type: 'string', description: 'is the temp-0.6 drift a REAL per-position id-aligned TV (n_scored>0), not the string/id artifact or a temp-0 stand-in?' },
    winnerSound: { type: 'string', description: 'any TPS win real (truthful per-draft+per-token basis, like-for-like trajectory, no #12 fork) AND lossless (flip+temp-0.6 within floor)?' },
    recommendation: { type: 'string', description: 'single: which candidate ships faster+lossless at B=4 clean (or cat9 stands); is OPT-1 a real lever uncontended; the temp-0.6 verdict; the B=4 contamination magnitude. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if any number is on the '
  + 'contaminated/B=1/raw path (must be CLEAN offloaded B=4, GB10 vLLM-only), if a tree is not depth-matched, if '
  + 'the temp-0.6 drift is the string/id artifact or a temp-0 stand-in, if a TPS win is a #12 fork or not lossless, '
  + 'or if the contamination contrast is not a real OFFLOAD_CODEX 0-vs-1 pair. research-before-deadend. No close/'
  + 'pass-fail; no reward-hack (WY parked).',
  { label: 'verify-b4-clean', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { m, v };
