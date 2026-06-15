export const meta = {
  name: 'fr13-reaim-measure-deployment',
  description: 'USER (2026-06-15): the handrolled prompts_swe4 + RAW /v1/completions regime is OFF-DISTRIBUTION - CONFIRMED in the data: native E5 served_token_ids repeat the block [271,248068,271,248069,271,40] = "\\n<think>\\n</think>\\nI" at positions 0/27/58 = a DEGENERATE empty-<think></think> repetition loop (no chat template, the model trained/deployed chat-templated goes off the rails). This (not a kernel bug) is what tanked native accept to 1.589 vs the gold 3.161 fork. So: MEASURE ON REAL SWE-VERIFIED + CODEX (the deployment regime, chat-templated /v1/responses through the codex agent loop) - NOT handrolled raw prompts. The big-denom ALREADY proved this regime is faithful (codex on astropy-12907, native ~= cat9 13.5/14%, NO degenerate loop). RE-AIM the canonical fr13_measure to the deployment harness: integrate the big-denom machinery (fr13_bigdenom_swe_serve.sh serve + run_swe_bench_q36 codex + LUMO_PROXY_PAIR_DUMP served-stream capture + fr13_swe_stream_to_oracle_src reduce + fr13_recurrent_decode_oracle rescore) as the canonical path; measure {s/fwd (decode_seconds DURING the real workload), accept/event, lossless-flip (proxy-dump + oracle rescore), temp-0.6 (q,p) drift} on the deployment regime; DEPRECATE the raw-/v1/completions handrolled path (keep only as a documented off-distribution cautionary note). Keep truthful accounting + INSTRUMENT ON/OFF separation + B=1/B=4 + cat-shape param + the q-by-token-id temp06 fix + paired-teacher-forced accept. Dev-iteration = cheapest deployment-faithful proxy (short codex run, NOT raw prompts); FINAL judgment = real SWE-Verified + codex B=4/CUDA-captured/4-task/30min. Phase1 = re-aim (CPU). Phase2 = USE GPU: run native E5 + cat9 through real codex+SWE, confirm COHERENT streams (no degenerate loop) + reconcile vs big-denom. Phase3 = adversarial verify. Output FR13_MEASURE_DEPLOYMENT_REGIME.md.',
  phases: [
    { title: 'ReAim' },
    { title: 'GPUValidate' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, B=1 decode HBM-bound). Repo /home/mark/shared/lumoFlyWheel. Read code',
'+ git history + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec',
'commit (git commit -m ".." -- <files>, NEVER git add -A). GPU SERIALIZED + FREE; this wf is the only GPU user.',
'MAX 2 concurrent workflows. Convention (feedback_canonical_speed_lossless_infra): EXTEND fr13_measure, never',
'fork a new probe.',
'',
'THE REGIME BUG (confirmed in the data, the headline): the canonical fr13_measure regime sends the HANDROLLED',
'prompts_swe4.json as a RAW string to /v1/completions with NO chat template. That is OFF-DISTRIBUTION for this',
'chat/thinking-trained model: native E5 served_token_ids for prompt 0 REPEAT the block [271,248068,271,248069,',
'271,40] = "\\n<think>\\n</think>\\nI" at positions 0/27/58 = a DEGENERATE empty-<think></think> repetition loop',
'(verified output/fr13_measure/native_e5_q_temp06_on.json). prompts_swe4[0] itself has NO <think> tags. This',
'off-distribution degeneration (NOT a kernel bug) is what tanked native accept to 1.589 and forks cross-boot',
'(the GB10 near-tie). The no-spec oracle ranks the COHERENT continuation correct by 11 nats - so the real model',
'decode is coherent; only the off-distribution raw-prompt spec boots degenerate.',
'',
'THE FIX (user): measure on REAL SWE-VERIFIED + CODEX = the deployment regime (the codex agent loop on real',
'SWE-bench-Verified tasks, chat-templated via /v1/responses, multi-turn, real tool calls). The big-denom ALREADY',
'proved this regime faithful + representative: codex on astropy-12907 -> native ~= cat9 (13.55% vs 13.99% clear-',
'margin flips, NO degenerate loop, spec-vs-non-spec CONFIRMED). So the deployment regime is the canonical one;',
'the raw-/v1/completions handrolled path is DEPRECATED (keep only as a documented off-distribution cautionary',
'note + maybe for the regime-robust s/fwd, clearly flagged).',
'',
'BIG-DENOM MACHINERY TO INTEGRATE (the deployment harness, already built + proven): scripts/fr13_bigdenom_swe_',
'serve.sh (serve cat9 via fr13_launch_locked / native via fr10_launch_speed_server num_spec=5, + LUMO_PROXY_PAIR_',
'DUMP capture via inference_proxy.py, + spec-engagement raw-/metrics asserts), scripts/run_swe_bench_q36_a.py (the',
'codex agent loop on a SWE-Verified task, the gold-gate-proven astropy-12907 path), scripts/fr13_swe_stream_to_',
'oracle_src.py (byte-exact detok pair-dump -> oracle src), scripts/fr13_recurrent_decode_oracle.py (rescore vs',
'the no-spec RECURRENT decode oracle = the lossless flip), scripts/fr13_bigdenom_phase3_rescore.sh + consolidate.',
'fr13_measure should ORCHESTRATE these as its deployment-regime measurement, not re-invent them.',
'',
'TRUTHFUL ACCOUNTING + MODES (carry forward, unchanged): s/fwd = d(request_decode_time_seconds_sum)/d(spec_drafts)',
'DURING the real codex workload (decode-only, per-event, ~B-invariant), NEVER TPS/accept/wall; accept = d(accepted)',
'/d(drafts) B-DEPENDENT + now ON THE DEPLOYMENT TRAJECTORY (no degenerate fork); committed=accept+1; TPS DERIVED.',
'INSTRUMENT ON/OFF: SPEED only from clean-OFF (FR10_METRICS=0, no q-capture/probe = the codex run as deployed);',
'lossless/q-capture only from ON; diag-residue OFF-vs-ON; per-number mode recorded + never-mixed. B=1 + B=4. Any',
'cat shape (TREE-param, fail-loud unbuilt). The temp-0.6 (q,p) drift = the q-by-TOKEN-ID capture (string/id align',
'fix) + forced-decode recurrent p over the FULL deployment stream; paired-teacher-forced accept on a COMMON',
'oracle trajectory (fork-immune). DEV-iteration = the CHEAPEST deployment-faithful proxy (a SHORT codex run / few',
'turns, NOT raw prompts); FINAL judgment = real SWE-Verified + codex, B=4 + CUDA-captured + 4 tasks + ~30min,',
'lossless re-confirmed at B=4 (the deployable gate the user set).',
].join('\n');

phase('ReAim');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['deploymentRegimeDesign','machineryIntegrated','rawDeprecated','devVsFinal','committed','notes'],
  properties: {
    deploymentRegimeDesign: { type: 'string', description: 'how fr13_measure now measures on the real SWE-Verified + codex deployment regime (chat-templated /v1/responses, codex agent loop) - the canonical path; the 4 metrics on the deployment trajectory' },
    machineryIntegrated: { type: 'string', description: 'how the big-denom machinery (serve + run_swe_bench codex + proxy pair-dump + oracle rescore) is integrated/orchestrated into fr13_measure (not re-invented)' },
    rawDeprecated: { type: 'string', description: 'the raw-/v1/completions handrolled path is deprecated (off-distribution cautionary note); what if anything it is still used for (regime-robust s/fwd only, flagged)' },
    devVsFinal: { type: 'string', description: 'the dev-iteration (cheap deployment-faithful proxy, short codex run) vs final-judgment (real SWE+codex B=4/CUDA/4-task/30min) split' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  BASE + '\n\nTASK (ReAim, CPU, no GPU). Re-aim fr13_measure to the real SWE-Verified + codex deployment regime by '
  + 'integrating the big-denom machinery; deprecate the raw-/v1/completions path; keep truthful accounting + ON/OFF '
  + '+ B=1/B=4 + cat-shapes + the q-by-id temp06 fix + paired accept. Write FR13_MEASURE_DEPLOYMENT_REGIME.md + '
  + 'the code, commit pathspec. Return the schema.',
  { label: 'reaim-deployment', phase: 'ReAim', schema: R_SCHEMA, model: 'opus' }
);

phase('GPUValidate');
const G_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['coherentStreams','reconcileBigDenom','metricsOnDeployment','committed','notes'],
  properties: {
    coherentStreams: { type: 'string', description: 'GPU: run native E5 + cat9 through real codex+SWE (a SWE-Verified task) - are the served streams COHERENT (NO degenerate <think></think> repetition loop, unlike the raw-prompt regime)?' },
    reconcileBigDenom: { type: 'string', description: 'do the deployment-regime metrics reconcile with the big-denom direction (native ~= cat9 flips; representative accept, not the off-distribution 1.589)?' },
    metricsOnDeployment: { type: 'string', description: 's/fwd + accept + flip measured on the deployment trajectory - the numbers + that they are representative (not degenerate-fork-tanked)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const g = await agent(
  BASE + '\n\nTASK (GPUValidate - USE GPU, serialized, prelaunch recover_host_memory+assert>=95GiB+docker-empty '
  + 'per boot + teardown). The infra was re-aimed: ' + JSON.stringify(r) + '\nRun native E5 + cat9 through the real '
  + 'codex+SWE deployment harness (a SWE-Verified task, e.g. the proven astropy-12907 or a short one); CONFIRM the '
  + 'served streams are COHERENT (no degenerate <think></think> loop) + measure the deployment-regime metrics + '
  + 'reconcile vs the big-denom. (If a full codex task is too long for one GPU window, run a bounded turn count + '
  + 'note it.) Commit results. Return the schema.',
  { label: 'gpu-validate-deployment', phase: 'GPUValidate', schema: G_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','deploymentFaithful','noDegenerate','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    deploymentFaithful: { type: 'string', description: 'is the regime now the REAL SWE-Verified + codex deployment harness (chat-templated, codex loop), not handrolled raw prompts, integrating the big-denom machinery (not re-invented)?' },
    noDegenerate: { type: 'string', description: 'did the GPU validation confirm COHERENT deployment streams (no degenerate <think></think> loop) + representative metrics reconciling with the big-denom?' },
    recommendation: { type: 'string', description: 'single: is the infra now measuring on the deployment regime so the campaign numbers are trustworthy (not off-distribution)? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify({ reaim: r, gpu: g }) + '. Default holds=false if the regime '
  + 'is still raw-/v1/completions handrolled prompts (must be real SWE-Verified + codex chat-templated), if the '
  + 'GPU validation did NOT confirm coherent deployment streams (a degenerate <think></think> loop persisting = '
  + 'still off-distribution), if it re-invented the big-denom machinery instead of integrating it, or if a metric '
  + 'is measured on the off-distribution raw regime. research-before-deadend. No close/pass-fail; no reward-hack.',
  { label: 'verify-deployment-regime', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, g, v };
