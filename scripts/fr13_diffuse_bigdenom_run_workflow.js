export const meta = {
  name: 'fr13-diffuse-bigdenom-run',
  description: 'RUN the big-denominator diffuse + SWE-quality test (user asked; plan FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md verify HOLDS, runnable-now). cat9 (deployed locked flags) AND native-E5 on the gold-gate SWE-Verified task astropy-12907 (~1500s agent budget/arm), capture the FULL served stream via the default-OFF proxy pair-dump = a BIG denominator (~3k tok/arm, fixes the 4-prompt/450-tok ±9-flip small-sample). Rescore both vs the deployment-correct RECURRENT oracle (fr13_recurrent_decode_oracle, FR12_NO_SPECULATIVE_CONFIG=1 = spec counters cannot advance, non-vacuity code-enforced) -> clear-margin flip RATE + Wilson 95% CI (cat9 ~[4.2,6.4]% vs native ~[0.3,1.2]% predicted non-overlapping). Then the DIFFUSE-vs-FIXABLE discriminator (CI separation + token-class distribution + leaf-fork R/C/W split w/ CORRECTED reducer + per-layer sample) per the plan decision rule, AND the BINDING SWE-QUALITY gate: does cat9 SOLVE the task (patch+tests) vs native DESPITE the flips? K1 is NOT baked (mechanism refuted, do not enable FR13_SCAN_ALIGN). GPU ~75-130 min serialized. Adversarial verify.',
  phases: [
    { title: 'Reducer' },
    { title: 'SWEServe' },
    { title: 'Rescore' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene EVERY',
'boot: source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f + recover_host_memory after each boot/arm.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol.',
'FOLLOW THE COMMITTED PLAN: FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md (ac3540eb) has the exact scripts/flags/tasks/',
'metrics/decision-rule/SWE-quality-gate. Read it FIRST and execute it; this CTX is the guardrails.',
'',
'COMPARE TARGET (user): lossless = cat9 vs native-E5 each-vs-its-own no-spec RECURRENT decode oracle',
'(fr13_recurrent_decode_oracle.py, deployment-correct: FR12_NO_SPECULATIVE_CONFIG=1, single-step recurrent',
'_forward_core_decode_non_spec, NOT prefill/chunked/streamed-logprobs). clear-margin = served!=recurrent-argmax',
'AND (out of oracle top-20 OR deviation_nat>1.0). native-E5 = the within-floor BAR.',
'',
'K1 IS NOT BAKED (resolved 2026-06-15 DO NOT BAKE: the mechanism proof showed K1 moves the scan state 22.8x AWAY',
'from native; deployed scan already near-native). So run cat9 with the DEPLOYED locked flags ONLY - do NOT enable',
'FR13_SCAN_ALIGN. cat9 LOCKED.',
'',
'YOUR JOB (execute the plan):',
'PHASE 1 (Reducer, no GPU): write the ONE new ~80-line reducer scripts/fr13_swe_stream_to_oracle_src.py per the',
'  plan: glob the proxy pair-dump JSONs (LUMO_PROXY_PAIR_DUMP_DIR, one JSON/turn = {upstream request payload =',
'  the oracle PROMPT, parsed response = served output text + usage}) -> the oracle --src schema {prompts:[str],',
'  records:[{served_token_ids:[int]}]} (one pair/turn), with class-#12 BYTE-EXACT detokenization (re-tokenize the',
'  served TEXT with the model tokenizer, or use served_token_ids if the dump carries them). Unit-test it on any',
'  existing pair-dump or a synthetic pair. Commit pathspec.',
'PHASE 2 (SWEServe, GPU): per the plan, run BOTH arms on astropy__astropy-12907 (gold-gate-proven RESOLVED both',
'  arms ~1565s) via scripts/run_swe_bench_q36_a.py with LUMO_PROXY_PAIR_DUMP_DIR set (default-OFF pair-dump ON):',
'  ARM cat9 = scripts/fr13_launch_locked.sh (deployed: num_spec=9 TREE_ATTN, FIX-1/2/3/A, REPLAY_ROUTE=1,',
'    FA2_TREE_BIAS=1, CONV_COMMITTED_PATH=1, LUMO_FB_PROJ_PAD_ROWS=16, BATCH_INVARIANT=0; NO FR13_SCAN_ALIGN).',
'  ARM native-E5 = the naive_mtp/FLASH_ATTN/num_spec=5 server (scripts/fr10_launch_speed_server.sh).',
'  Capture per-task SWE resolved/failed (report.json) + the pair-dump served streams. Budget: ~1500s agent/arm;',
'  do >=1 rep/arm for the RATE; add a 2nd same-instance rep/arm for the SWE-quality gate IF the GPU budget allows',
'  (the gate wants N>=2 to not be a single coin-flip). Teardown+recover between arms. NON-VACUITY: pair-dump',
'  non-empty (>0 turns, served tokens present); the deployed flags live in the worker; SWE eval actually ran.',
'PHASE 3 (Rescore, GPU): load the recurrent oracle ONCE (fr13_recurrent_decode_oracle.py), rescore BOTH arms\'',
'  served streams (via the Phase-1 reducer --src): per-position clear-margin flips, n_positions_scored, drop',
'  round-trip-fail turns (record n_turns_dropped/n_positions_dropped), within-process rep1==rep2 determinism.',
'  Compute the flip RATE + Wilson 95% CI per arm. NON-VACUITY (code-enforced + assert): oracle engaged (recurrent',
'  decode calls >0), spec counters did NOT advance during rescore, gold-margin NOT streamed logprobs, denominator',
'  = validated round-trip token count (NOT text length).',
'PHASE 4 (Verdict). Apply the plan DECISION RULE: DIFFUSE (relax/ship-if-quality-passes) requires ALL: cat9 CI lo',
'  > native CI hi + half-width<0.015 + det; structural-boundary token-class clustering (no non-structural class',
'  >50%); corrected-reducer W~0 with R/C spread (no single tree depth/leaf-slot >40%, R:C ~1:2..2:1); per-layer',
'  first-nonzero on ~8-12 sampled R flips = L0-GDN with no reproducible >2x later-layer spike. FIXABLE (re-open a',
'  lever) on ANY single concentration (each maps to a named lever). PLUS the BINDING SWE-QUALITY gate: cat9',
'  resolves where native resolves on the solvable instance (flips did NOT change the outcome) = deployable-',
'  lossless-enough; cat9 fails where native resolves = quality-relevant, re-open. VERDICT: Diffuse + SWE-quality',
'  PASS => recommend ship cat9 as lossless-enough (bring to user, bake = user call). Fixable OR SWE-quality FAIL',
'  => do NOT relax, name the re-opened lever. Reward-hacks BANNED (no copy/dense/multi-spine/bonus/WY/K1-bake);',
'  native = its own deployed arm (a real A/B, not a splice). Quote FR13_BUG_CLASS_PLAYBOOK (#9 vacuous, #12',
'  length/denominator/cross-trajectory). NO bake/ship decision (user call).',
].join('\n');

phase('Reducer');
const RD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['reducerWritten','unitTested','committed','notes'],
  properties: {
    reducerWritten: { type: 'string', description: 'scripts/fr13_swe_stream_to_oracle_src.py: pair-dump glob -> oracle --src, byte-exact detok approach' },
    unitTested: { type: ['boolean','string'], description: 'unit-tested on a pair-dump/synthetic (served text round-trips to token ids)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const rd = await agent(
  CTX + '\n\nTASK (Reducer, no GPU). Read the plan, write + unit-test scripts/fr13_swe_stream_to_oracle_src.py, '
  + 'commit pathspec. Return the schema.',
  { label: 'bigdenom-reducer', phase: 'Reducer', schema: RD_SCHEMA, model: 'opus' }
);

phase('SWEServe');
const SS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cat9_swe_resolved','native_swe_resolved','cat9_pairdump_turns','native_pairdump_turns','flagsLive','reps','ok','notes'],
  properties: {
    cat9_swe_resolved: { type: 'string', description: 'cat9 astropy-12907 resolved/failed per rep' },
    native_swe_resolved: { type: 'string', description: 'native-E5 resolved/failed per rep' },
    cat9_pairdump_turns: { type: ['integer','string','null'], description: 'cat9 pair-dump turns + total served tokens (the denominator)' },
    native_pairdump_turns: { type: ['integer','string','null'] },
    flagsLive: { type: ['boolean','null'], description: 'deployed cat9 flags live in worker; native server is naive_mtp num_spec=5' },
    reps: { type: 'string', description: 'reps/arm run + GPU time used' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const ss = await agent(
  CTX + '\n\nTASK (SWEServe, GPU). Run cat9 + native-E5 on astropy-12907 with pair-dump ON, capture SWE resolved + '
  + 'served streams. Prove pair-dump non-empty + flags live + SWE eval ran. Teardown+recover between arms. Return '
  + 'the schema.',
  { label: 'bigdenom-swe-serve', phase: 'SWEServe', schema: SS_SCHEMA, model: 'opus' }
);

phase('Rescore');
const RS_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['oracleEngaged','specCountersFrozen','cat9_flips','cat9_n_positions','cat9_rate_ci','native_flips','native_n_positions','native_rate_ci','det','ok','notes'],
  properties: {
    oracleEngaged: { type: ['boolean','null'], description: 'recurrent decode calls >0, spec counters did NOT advance (code-enforced)' },
    specCountersFrozen: { type: ['boolean','null'] },
    cat9_flips: { type: ['integer','null'], description: 'cat9 clear-margin flips' },
    cat9_n_positions: { type: ['integer','null'], description: 'cat9 denominator (validated round-trip tokens)' },
    cat9_rate_ci: { type: ['string','null'], description: 'cat9 flip rate + Wilson 95% CI' },
    native_flips: { type: ['integer','null'] },
    native_n_positions: { type: ['integer','null'] },
    native_rate_ci: { type: ['string','null'] },
    det: { type: 'string', description: 'within-process rep1==rep2 determinism' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const rs = await agent(
  CTX + '\n\nTASK (Rescore, GPU). Rescore both arms vs the recurrent oracle via the reducer, clear-margin flips + '
  + 'Wilson CI + denominator. PROVE oracle engaged + spec counters frozen + not streamed logprobs. Return the '
  + 'schema.',
  { label: 'bigdenom-rescore', phase: 'Rescore', schema: RS_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','rate_ci_separation','diffuseDiscriminator','sweQualityGate','diffuseOrFixable','shipRecommendation','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'pair-dump real + oracle engaged + spec frozen + denominator validated + flags live all proven?' },
    rate_ci_separation: { type: 'string', description: 'cat9 vs native flip-rate CIs (non-overlapping?) - the tight big-denominator answer' },
    diffuseDiscriminator: { type: 'string', description: 'token-class distribution + R/C/W split + per-layer sample = diffuse (structural spread) or fixable (a concentration)?' },
    sweQualityGate: { type: 'string', description: 'did cat9 SOLVE astropy-12907 (patch+tests) where native resolves, DESPITE the flips? the binding deployable answer' },
    diffuseOrFixable: { type: ['string','null'], description: 'DIFFUSE (all criteria) or FIXABLE (a named concentration -> lever)?' },
    shipRecommendation: { type: 'string', description: 'Diffuse+SWE-pass = ship cat9 lossless-enough (user bakes); Fixable/SWE-fail = re-open. For the user, NO ship decision.' },
    rewardHackCheck: { type: 'string', description: 'no copy/dense/multispine/bonus/WY/K1-bake; native = real deployed A/B arm; cat9 = locked deployed flags' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: serve=' + JSON.stringify(ss) + ' rescore=' + JSON.stringify(rs) + '. Default '
  + 'holds=false if the denominator is text-length not validated round-trip tokens, the oracle was not engaged / '
  + 'spec counters advanced (vacuous), flips from streamed logprobs, the diffuse-vs-fixable verdict ignores a '
  + 'concentration, or the SWE-quality gate conflates the both-fail control (astropy-13033) with a tree '
  + 'regression. Conclude honestly: diffuse+SWE-pass (ship-able) or fixable/SWE-fail (re-open). No ship/bake '
  + 'decision; no reward-hack.',
  { label: 'verify-bigdenom-run', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { rd, ss, rs, v };
