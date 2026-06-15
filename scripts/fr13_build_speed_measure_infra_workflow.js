export const meta = {
  name: 'fr13-build-speed-measure-infra',
  description: 'USER (2026-06-15): build the canonical SPEED/lossless MEASUREMENT INFRA as PROPER Python code (not per-agent hand-rolled probes - that just caused a regime bug: fr10_quick sent a RAW prompt string to /v1/completions -> native accept 1.70, while the gold-gate sent TOKENIZED context_ids -> 3.161 on the SAME prompts). ONE validated entry point that measures {s/fwd, accept/event, lossless-flip, temp-0.6 distributional drift} for ANY arm (native MTP-N, any caterpillar TREE shape e.g. cat9/cat10/cat6root, OPT-1/OPT-A flags) in ONE canonical regime. REQUIREMENTS (user): the WORKFLOW must USE THE GPU to test it + RECONCILE with the historic banked numbers (native E5 s/fwd 0.2182 accept 3.161, cat9 s/fwd 0.2248 accept 3.18); support B=1 AND B=4; support temp 0.6 + top_p 0.95 (deployment sampling) AND temp 0 greedy; support different cat shapes (TREE-param, fail-loud on unbuilt); and TRUTHFUL speed accounting (s/fwd = delta(request_decode_time_seconds)/delta(spec_drafts) decode-only per-event, NEVER TPS/accept/wall; accept = delta(accepted)/delta(drafts) is B-DEPENDENT via co-residency, s/fwd ~B-invariant for bandwidth-bound; B=1 must be SEQUENTIAL not concurrent samples). Phase1 = diagnose the regime bug definitively + BUILD the canonical harness on the CORRECT regime. Phase2 = USE THE GPU: boot native E5 + cat9, run the harness, RECONCILE vs the banked numbers (must reproduce 3.161/3.18), + a B=4 and a temp-0.6/top_p0.95 smoke. Phase3 = adversarial verify. Prelaunch host-mem protocol + engagement asserts + serialized boots. Output FR13_SPEED_MEASURE_INFRA.md + the code.',
  phases: [
    { title: 'Build' },
    { title: 'GPUValidate' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode HBM-bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. Read our code + git history + the banked binds + vLLM source via scripts/',
'vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec commit (git commit -m ".." -- <files>,',
'NEVER git add -A). GPU is SERIALIZED + currently FREE; the ONLY GPU user is this workflow (a CPU temp-0.6 wf',
'runs alongside, no GPU contention). MAX 2 concurrent workflows total - do NOT spawn more.',
'',
'THE BUG THIS FIXES (root-caused this session): two hand-rolled probes gave two different "native accept" on the',
'SAME prompts_swe4 / max_tokens 128 / temp 0 / seed 1313 - fr10_quick_decode_tps_probe.py sends "prompt": RAW',
'STRING to /v1/completions (line 132/166-179) -> native accept 1.70; fr13_gold_margin_probe.py sends "prompt":',
'TOKENIZED context_ids (tokenize via /tokenize first, line 356/417-426) -> the banked 3.161. Same B=1 (batch_size',
'1 is SEQUENTIAL, line 164-165, co-residency RULED OUT). The raw-vs-tokenized prompt forks the trajectory ->',
'different served stream -> different accept. DIAGNOSE this definitively (which exact regime detail causes 1.70',
'vs 3.161) and BUILD on the regime that reproduces 3.161.',
'',
'CANONICAL REGIME (bake ONE, no agent re-rolls it): tokenized-prompt path (the gold-gate regime), prompts =',
'output/fr13_acceptance_ladder/prompts_swe4.json, seed 1313, max_tokens 128, FR10_METRICS=0, BI=0 pinned both',
'arms. TRUTHFUL SPEED ACCOUNTING: s/fwd = delta(vllm:request_decode_time_seconds_sum)/delta(vllm:spec_decode_num_',
'drafts_total) (decode-only, per-event, RAW counter, ~B-invariant for bandwidth-bound) - NEVER TPS, accept, or',
'wall as the basis; accept/event = delta(vllm:spec_decode_num_accepted_tokens_total)/delta(spec_drafts) (B-',
'DEPENDENT: co-residency degrades it, so B=1 and B=4 are DIFFERENT and must be labelled); committed/event =',
'accept+bonus; TPS = committed/s_fwd reported as DERIVED not measured. Engagement asserts BEFORE any number',
'(class-9): tok/draft == N (native) / == len(TREE) (tree), has_tree_parent_indices, tree_sample_accept; FAIL LOUD',
'+ record nothing on disengagement. Within-boot determinism (class-8) rep1==rep2. Cross-boot byte gate BANNED.',
'',
'TEMP-0.6 DISTRIBUTIONAL CAPTURE (FIRST-CLASS metric, per FR13_TEMP06_DRIFT_GATE.md / wuhlwhr0l - the q-capture',
'machinery does NOT exist yet, BUILD it into the infra). The deployment-binding lossless gate is NOT the temp-0',
'argmax flip (that is necessary-not-sufficient); it is per-position TV(softmax(q/0.6), softmax(p/0.6)) where',
'q = the SPEC VERIFY-FORWARD full top-20 log_softmax per SERVED position (extend the fr13_gold_margin_probe verify',
'-top_logprobs capture to the WHOLE served stream + record per-position truncated tail mass for an error bar) and',
'p = the no-spec RECURRENT decode oracle top-20 on the IDENTICAL served prefix (forced-decode, fr13_recurrent_',
'decode_oracle, FR12_NO_SPECULATIVE_CONFIG=1, FLASH_ATTN, single-token roll, NOT chunked; assert RECURRENT_PATH_',
'ENGAGED + call-counter>0, #9 fail-loud). softmax(.../0.6) is recoverable from a recorded top-20 log_softmax (the',
'per-position additive constant cancels) - record the tail mass as the truncation error. Each arm vs its OWN',
'oracle. The infra must: (1) CAPTURE q over the FULL served stream (NEW - nothing records it today, that was the',
'gap), (2) capture the forced-decode recurrent p, (3) reduce per-position TV(softmax(q/0.6),softmax(p/0.6)) + KL +',
'the per-position over-floor vector (PAIR the scalar with the per-position view, reference_scalar_metric_per_token',
'_blindspot). PLUS the realized multi-seed BAG-TV cross-check: N=6-8 native temp-0.6/top_p0.95 seeds -> floor =',
'p95 of C(N,2) native-vs-native draws (UPGRADE the single-draw 0.1133, #12); cat9 same seeds -> cat9-vs-native',
'bag-TV PASS iff <= floor_p95; B=4 deployed shape for the verdict tier. Expose as `--metric temp06_drift` of the',
'ONE canonical infra (not a separate hand-roll) so the lossless-lever decision runs through the validated path.',
'',
'VALIDATED TOOLS TO REUSE (do NOT re-invent): scripts/fr13_shape_gate.sh (prelaunch recover_host_memory + assert',
'MemAvailable>=95GiB + docker-empty + boot forked + engagement + capture + flip + teardown - but TREE-only + no',
's/fwd); scripts/fr13_gold_margin_probe.py (the CANONICAL tokenized-prompt capture = q/verify-forward margin);',
'scripts/fr13_oracle_stream_teacher_force.py + fr13_recurrent_decode_oracle.py (the lossless flip vs the no-spec',
'RECURRENT decode oracle = p); scripts/fr10_quick_decode_tps_probe.py (s/fwd counters, but the RAW-prompt regime',
'is the bug - reuse the COUNTER read, not the request format); launchers fr10_launch_speed_server.sh (native,',
'SPEC_CONFIG num_speculative_tokens=N, CONTAINER fr10-speed-start, PORT 9950) + fr13_launch_locked.sh (cat9) +',
'fr13_launch_forked_fa2_tree_server.sh (TREE-param tree shapes, derives num_spec + tree from len(TREE)).',
'',
'HISTORIC NUMBERS TO RECONCILE (the GPU validation MUST reproduce these or explain the delta): native E5 (MTP-5,',
'FLASH) s/fwd 0.218160 accept/event 3.161290 decode_tps 18.93 (FR13_B1_CURRENT_GATE_BIND, prompts_swe4 B=1 greedy);',
'cat9 s/fwd ~0.2248 accept ~3.18 (FR13_B1_FIX3_GATE_BIND); the Phase-0 s/fwd that MATCHED (native 0.2159/cat9',
'0.2241) but accept that did NOT (native 1.70/cat9 3.02 = the raw-prompt-regime bug). Lossless-at-scale: big-denom',
'cat9 13.55% ~= native 13.99% (temp-0 argmax).',
].join('\n');

phase('Build');
const B_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['regimeBugDiagnosis','infraDesign','codeBuilt','speedAccounting','committed','notes'],
  properties: {
    regimeBugDiagnosis: { type: 'string', description: 'the DEFINITIVE cause of fr10_quick 1.70 vs gold-gate 3.161 (the exact regime detail: raw-string vs tokenized-prompt -> trajectory fork, or other), code-cited' },
    infraDesign: { type: 'string', description: 'the canonical fr13_measure API: ONE entry point measuring {s/fwd, accept, flip, temp-0.6 drift} for any arm (native MTP-N / TREE-shape / OPT flags) at B=1+B=4, temp 0 + temp 0.6/top_p 0.95, the canonical regime baked, fail-loud on unbuilt shape' },
    codeBuilt: { type: 'string', description: 'the actual Python module(s) written (paths) + the boot orchestration reusing the validated tools; how to invoke for any arm' },
    speedAccounting: { type: 'string', description: 'the truthful speed accounting baked in (s/fwd decode_seconds basis, accept B-dependent, TPS derived-not-measured, B=1 sequential) - and where the BANNED forms are blocked' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const b = await agent(
  BASE + '\n\nTASK (Build, CPU, no GPU yet). Diagnose the regime bug definitively, then BUILD the canonical '
  + 'measurement infra (write the Python module + boot orchestration, reuse the validated tools, bake the correct '
  + 'tokenized regime + truthful speed accounting + B=1/B=4 + temp 0/0.6 + TREE-param). Write FR13_SPEED_MEASURE_'
  + 'INFRA.md + the code, commit pathspec. Return the schema.',
  { label: 'build-measure-infra', phase: 'Build', schema: B_SCHEMA, model: 'opus' }
);

phase('GPUValidate');
const G_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['reconcileNativeE5','reconcileCat9','b4Smoke','temp06Smoke','reproducesHistoric','committed','notes'],
  properties: {
    reconcileNativeE5: { type: 'string', description: 'GPU: boot native E5, run the harness, the MEASURED s/fwd + accept + flip vs the banked (0.2182 / 3.161); does it REPRODUCE 3.161 (fixing the 1.70 bug)? the delta + why' },
    reconcileCat9: { type: 'string', description: 'GPU: boot cat9, run the harness, MEASURED vs banked (0.2248 / 3.18); reproduces?' },
    b4Smoke: { type: 'string', description: 'GPU: a B=4 measurement on one arm - does the harness measure B=4 s/fwd + accept (co-residency-degraded accept expected, s/fwd ~B-invariant) sanely?' },
    temp06Smoke: { type: 'string', description: 'GPU: the temp-0.6 (q,p) DISTRIBUTIONAL capture on one arm over a short served stream - does the infra capture q (spec verify full top-20) + the forced-decode recurrent p + reduce per-position TV(softmax(q/0.6),softmax(p/0.6)) sanely (non-vacuous: q present over the full stream, p RECURRENT_PATH_ENGAGED, tail-mass recorded)?' },
    reproducesHistoric: { type: 'string', description: 'VERDICT: does the canonical harness reproduce the historic B=1 numbers (native 3.161 / cat9 3.18 / s/fwd), confirming the regime is correct + the 1.70 was the raw-prompt bug?' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const g = await agent(
  BASE + '\n\nTASK (GPUValidate - USE THE GPU, serialized boots, prelaunch recover_host_memory + assert >=95GiB + '
  + 'docker-empty before EACH boot + teardown after). The harness was just built: ' + JSON.stringify(b) + '\nBoot '
  + 'native E5 + cat9, run the harness, RECONCILE vs the banked numbers (must reproduce native 3.161 / cat9 3.18 / '
  + 's/fwd - if it still gives ~1.70 the regime is still wrong, fix it). + a B=4 smoke + a temp-0.6/top_p0.95 '
  + 'smoke on one arm. Commit the reconciliation results. Return the schema.',
  { label: 'gpu-validate-infra', phase: 'GPUValidate', schema: G_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','reproducesHistoric','accountingTruthful','coverageComplete','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    reproducesHistoric: { type: 'string', description: 'did the GPU validation actually reproduce the banked native 3.161 / cat9 3.18 (not still 1.70)? spot-check the measured numbers' },
    accountingTruthful: { type: 'string', description: 'is the speed accounting truthful (s/fwd decode_seconds basis, accept B-labelled, TPS derived, no banned TPS/accept-as-measured), B=1 sequential?' },
    coverageComplete: { type: 'string', description: 'does the infra support B=1+B=4, temp 0 + temp 0.6/top_p 0.95, native + any TREE cat shape (fail-loud on unbuilt), the 4 metrics INCLUDING the temp-0.6 DISTRIBUTIONAL drift (the NEW q-capture over the full served stream + forced-decode recurrent p + per-position TV(softmax(q/0.6),softmax(p/0.6)) + multi-seed bag-TV p95 floor) - one canonical regime, not a separate hand-roll?' },
    recommendation: { type: 'string', description: 'single: is the canonical infra ready to drive the speed campaign (and the Phase-0 accept re-measured correctly)? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY build+validate: ' + JSON.stringify({ build: b, gpu: g }) + '. Default holds=false '
  + 'if the GPU validation did NOT reproduce the banked 3.161/3.18 (still the raw-prompt 1.70 bug), the speed '
  + 'accounting uses a BANNED basis (TPS/accept/wall) or mislabels B-dependence, the infra misses B=4 / temp-0.6+'
  + 'top_p0.95 / TREE-cat-shapes / a metric, or it is still hand-rolled-per-call rather than one canonical regime; '
  + 'if the temp-0.6 DISTRIBUTIONAL drift is only LISTED not IMPLEMENTED (the infra must actually CAPTURE q over the '
  + 'full served stream + the forced-decode recurrent p + reduce per-position TV(softmax(q/0.6),softmax(p/0.6)) - the '
  + 'q-capture is NEW, nothing records it today, so a "supports temp06_drift" that has no q-capture code FAILS). '
  + 'research-before-deadend. No close/pass-fail.',
  { label: 'verify-measure-infra', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { b, g, v };
