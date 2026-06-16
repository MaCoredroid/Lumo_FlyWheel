export const meta = {
  name: 'fr13-offload-harness',
  description: 'USER (2026-06-15, MAIN concern): the GB10 is UNIFIED MEMORY (Grace ARM CPU + Blackwell GPU share 273 GB/s), so ANY non-vLLM compute co-located on the GB10 (the codex agent Docker, the inference_proxy) competes with vLLMs HBM-bound weight read = CONTAMINATES the timing-sensitive speed measurement (s/fwd, run-ahead/OPT-1, accept timing). REBUILD the deployment harness to OFFLOAD everything except vLLM to the x86 box (alienware, reachable over SSH): codex agent + proxy run on alienware, vLLM runs ALONE on the GB10. Confirmed feasible: alienware = x86_64, has docker + the codex-runner:v1 image + swe_eval_offload venv (the SWE eval is ALREADY offloaded there). Architecture: vLLM serve on GB10 (port 9950, reachable from alienware - likely tailscale) <- proxy on alienware (forwards codex->GB10 + pair-dumps) <- codex-runner:v1 Docker on alienware. The deploy-* measurement reads the GB10 vLLM /metrics (clean - the model decode, no contention) + the alienware pair-dumps (synced back) for the lossless rescore (the recurrent oracle is a SEPARATE GB10 GPU phase, also clean). GOAL: during a codex run, the GB10 runs ONLY vLLM (verify via ps/docker on the GB10 = nothing but the vllm container). Build the offloaded harness (update fr13_bigdenom_swe_serve.sh + run_swe_bench_q36_a.py to launch proxy+codex on alienware over SSH; vLLM stays GB10) + validate clean + flag any alienware setup gaps. On fr13-speedfix. Phase1 build (CPU+SSH) -> Phase2 validate (GPU+SSH) -> Phase3 verify. Output FR13_OFFLOAD_HARNESS.md.',
  phases: [
    { title: 'BuildOffload' },
    { title: 'ValidateClean' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (gx10-edb9, aarch64, UNIFIED MEMORY Grace+Blackwell 273 GB/s - the contamination risk).',
'Repo /home/mark/shared/lumoFlyWheel, BRANCH fr13-speedfix. The x86 box = alienware (mark-Alienware-Aurora,',
'x86_64), reachable `ssh alienware` (BatchMode, the SWE eval offload already uses it). int-view never atol.',
'Pathspec commit on fr13-speedfix. GPU SERIALIZED + FREE. MAX 2 concurrent workflows.',
'',
'THE CONTAMINATION (user main concern): the deployment harness currently runs ON the GB10, during a codex run:',
'(1) the vLLM serve (MUST stay on the GB10 - the GPU), (2) the inference_proxy (host CPU process, lumo_flywheel_',
'serving.inference_proxy, the pair-dump), (3) the codex-runner:v1 Docker (run_swe_bench_q36_a.py, --network=host).',
'(2) and (3) compete with vLLM for the GB10 unified-memory bandwidth + CPU = CONTAMINATE the timing-sensitive',
'speed numbers (s/fwd from request_decode_time_seconds, the run-ahead / OPT-1 main-thread block, accept timing).',
'The lossless numbers (argmax/distributional) are timing-INDEPENDENT so not contaminated, but speed IS. FIX: move',
'(2) + (3) to alienware; the GB10 runs ONLY vLLM.',
'',
'ALIENWARE INVENTORY (confirmed): x86_64, docker present, codex-runner:v1 image PRESENT (0b6ac07c, the codex',
'agent), lumo-alpha-codex-session images present, ~/swe_eval_offload (swe_eval_x86_worker.py + venv) present. The',
'codex CLI binary is MISSING but the codex-runner:v1 IMAGE is what runs (docker run codex-runner:v1 codex exec).',
'The inference_proxy code + its python deps may need syncing to alienware (or run via an image) - ASSESS + set up.',
'',
'TARGET ARCHITECTURE: vLLM serve on GB10 (port 9950, bound to an interface reachable from alienware - find the',
'GB10<->alienware network, likely tailscale, get the GB10 IP/hostname alienware can reach) <- proxy on alienware',
'(forwards alienware-codex -> GB10:9950 + writes the pair-dumps locally on alienware) <- codex-runner:v1 Docker on',
'alienware (hits the alienware proxy). After the codex run: rsync the alienware pair-dumps back to the GB10 repo',
'(output/...) for the deploy-lossless recurrent-oracle rescore (a SEPARATE GB10 GPU phase, also vLLM-only = clean).',
'deploy-speed reads the GB10 vLLM /metrics (the model decode time, now uncontended). Nothing about the canonical',
'measurement basis changes - only WHERE the codex+proxy compute runs.',
'',
'BUILD: extend the harness (fr13_bigdenom_swe_serve.sh + run_swe_bench_q36_a.py + a new offload helper) so the',
'proxy + codex launch on alienware over SSH (the SWE eval already does this pattern - ssh alienware ...), vLLM',
'stays on GB10, pair-dumps rsync back. Keep it FLAG-GATED (e.g. OFFLOAD_CODEX=1) so the legacy on-GB10 path still',
'works if alienware is down. Default the deployment measurement to the offloaded path. Reuse the existing eval-',
'offload SSH/rsync plumbing (run_swe_bench_q36_a.py:392-432). Do NOT change the canonical measurement (deploy-',
'speed/deploy-lossless/deploy-temp06-drift) - only the codex+proxy LOCATION.',
'',
'NETWORK ROBUSTNESS (user MAIN reminder - the Spark<->alienware tailscale link can be UNSTABLE; a blip must NOT',
'become a silent failed/contaminated run). REQUIREMENTS, build them in: (1) the SPEED basis is ALREADY network-',
'robust BY CONSTRUCTION and must STAY so - s/fwd = d(request_decode_time_seconds_sum)/d(spec_drafts) is read from',
'the GB10 vLLM /metrics LOCALLY (on the GB10, not over the wire); decode_seconds only advances while the serve is',
'actually DECODING, so a network stall (codex request hanging) just PAUSES it = the per-event s/fwd is unaffected',
'(confirm this; never read the speed counter across the wire). (2) the codex<->vLLM request path (alienware codex',
'-> proxy -> GB10:9950 over tailscale) must SURVIVE transient drops: generous request timeouts + retry/reconnect',
'with backoff (do not fail the whole 30-min codex task on one blip); the proxy retries the upstream. (3) the pair-',
'dump rsync alienware->GB10 must be resilient: rsync --partial --append-verify + retry loop (a dropped rsync must',
'not lose the served stream the lossless rescore needs). (4) NETWORK-DROP DETECTION + non-mis-attribution: log the',
'tailscale link state (ping/curl /health from alienware) throughout; a request failure must be CLASSIFIED network',
'-drop vs real (model/measurement) so a blip is NOT mistaken for a degenerate fork (#12) or a real failure; if a',
'measurement window spans a confirmed network stall, FLAG/discard it (do not record an s/fwd or accept from a',
'wire-stalled window). (5) WATCHDOG/teardown: if the link is down past a threshold, fail LOUD + clean teardown +',
'recover, do not hang. (6) the GB10 vLLM serve binds to the tailscale interface but its health/measurement is',
'always checkable LOCALLY on the GB10 (so a Spark<->alienware drop never blinds the measurement side).',
].join('\n');

phase('BuildOffload');
const B_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['alienwareSetup','network','networkRobustness','harnessBuilt','proxyOffload','gaps','committed','notes'],
  properties: {
    alienwareSetup: { type: 'string', description: 'what was set up on alienware (codex-runner:v1 ready, the proxy code+deps synced/imaged, swe_eval present) + how the codex+proxy launch there' },
    network: { type: 'string', description: 'the GB10<->alienware network (tailscale? the GB10 IP/hostname alienware reaches the vLLM 9950 on); confirmed alienware can curl the GB10 /health' },
    networkRobustness: { type: 'string', description: 'how network INSTABILITY is handled (user main reminder): s/fwd read LOCALLY on GB10 (network-stall-immune by construction); codex<->vLLM timeouts+retry/reconnect with backoff (survive a blip, not fail the 30-min task); rsync --partial --append-verify + retry; tailscale-link logging + network-drop-vs-real classification (no #12 mis-attribution); discard a wire-stalled measurement window; watchdog fail-loud+teardown if link down past threshold' },
    harnessBuilt: { type: 'string', description: 'the harness change (flag-gated OFFLOAD_CODEX): proxy+codex on alienware over SSH, vLLM on GB10, pair-dumps rsync back; reusing the eval-offload SSH plumbing' },
    proxyOffload: { type: 'string', description: 'how the inference_proxy runs on alienware (synced code+venv, or an image) forwarding to GB10:9950 + capturing pair-dumps locally' },
    gaps: { type: 'string', description: 'anything the USER must do (e.g. build/pull an image on alienware, SSH key, firewall) - or NONE' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const b = await agent(
  BASE + '\n\nTASK (BuildOffload, CPU + SSH to alienware, no GPU boot yet). Assess alienware + the GB10<->alienware '
  + 'network; set up the proxy on alienware; build the flag-gated offloaded harness (proxy+codex on alienware, '
  + 'vLLM on GB10, pair-dumps rsync back), reusing the eval-offload SSH/rsync plumbing. Flag any user-setup gaps. '
  + 'Commit pathspec on fr13-speedfix. Return the schema.',
  { label: 'build-offload', phase: 'BuildOffload', schema: B_SCHEMA, model: 'opus' }
);

phase('ValidateClean');
const V1_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gb10VllmOnly','codexRanOnAlienware','measurementWorks','networkResilience','speedCleanVsContaminated','committed','notes'],
  properties: {
    gb10VllmOnly: { type: 'string', description: 'GPU: during a SHORT offloaded codex run, ps+docker ON THE GB10 show ONLY the vllm container (no codex-runner, no inference_proxy) - the contamination is gone? cite the GB10 process/docker snapshot taken DURING the run' },
    networkResilience: { type: 'string', description: 'test network instability: inject a brief tailscale blip (or drop a request) mid-run + confirm the run SURVIVES (retry/reconnect, task continues) OR cleanly detects+flags it (no silent contamination); confirm a wire-stalled window is NOT recorded as a bad s/fwd; the local s/fwd held steady across the blip' },
    codexRanOnAlienware: { type: 'string', description: 'the codex agent + proxy genuinely ran on alienware (the served stream is coherent codex multi-turn, pair-dumps captured on alienware + rsynced back)' },
    measurementWorks: { type: 'string', description: 'deploy-speed read the GB10 vLLM /metrics fine + the deploy-lossless pair-dump path works through the offload (the canonical measurement is intact)' },
    speedCleanVsContaminated: { type: 'string', description: 'the offloaded s/fwd vs the prior co-located s/fwd (is the clean number lower / different = contamination was real)?' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const v1 = await agent(
  BASE + '\n\nTASK (ValidateClean - USE GPU, serialized, prelaunch per boot). The offloaded harness: '
  + JSON.stringify(b) + '\nRun a SHORT offloaded codex run (vLLM on GB10, codex+proxy on alienware); take a GB10 '
  + 'ps+docker snapshot DURING the run to PROVE the GB10 is vLLM-only; confirm codex ran on alienware + the '
  + 'measurement (deploy-speed/lossless) works through the offload; compare the clean s/fwd to the contaminated '
  + 'one. Commit results. Return the schema.',
  { label: 'validate-clean', phase: 'ValidateClean', schema: V1_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','gb10IsVllmOnly','measurementIntact','networkRobust','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    gb10IsVllmOnly: { type: 'string', description: 'is the GB10 PROVEN vLLM-only during a codex run (the during-run ps/docker snapshot shows no codex/proxy on the GB10) - the contamination genuinely removed, not asserted?' },
    measurementIntact: { type: 'string', description: 'is the canonical deployment measurement intact through the offload (deploy-speed reads GB10 /metrics, lossless pair-dumps rsynced) - nothing reward-hacked or broken?' },
    networkRobust: { type: 'string', description: 'is Spark<->alienware network instability genuinely handled (s/fwd local + stall-immune, codex/rsync retry/reconnect, wire-stalled window flagged-not-recorded) AND exercised by a real blip test - not just asserted?' },
    recommendation: { type: 'string', description: 'single: is the offloaded harness ready to re-run the B=4 deployment sweep cleanly? any user-setup gap? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(v1) + '. Default holds=false if the GB10 is NOT proven '
  + 'vLLM-only during the codex run (the during-run GB10 ps/docker snapshot is the proof - codex/proxy must be '
  + 'ABSENT on the GB10), if the codex/proxy secretly still ran on the GB10, if the measurement was broken/reward-'
  + 'hacked by the offload, if NETWORK INSTABILITY is not genuinely handled (s/fwd must be read locally + network-'
  + 'stall-immune; codex<->vLLM retry/reconnect; rsync resilient; a wire-stalled window flagged not silently '
  + 'recorded - the resilience test must actually exercise a blip), or if a required alienware setup gap is hidden. '
  + 'No close/pass-fail.',
  { label: 'verify-offload', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { b, v1, v };
