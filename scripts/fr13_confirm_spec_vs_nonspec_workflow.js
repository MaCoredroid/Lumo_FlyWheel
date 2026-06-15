export const meta = {
  name: 'fr13-confirm-spec-vs-nonspec',
  description: 'USER (2026-06-15): just CONFIRM the big-denom SWE-quality gate is genuinely comparing SPEC-DECODE (the served stream) vs NON-SPEC (recurrent single-token) DECODE, for BOTH arms (cat9 + native), source-grounded + non-vacuous. If confirmed, cat9 13.548% ~= native 13.985% (CIs overlap) is a VALID lossless-vs-native result at deployable scale. The user bar (feedback_fr13_lossless_compare_target): no-spec recurrent decode = ground truth; US lossless iff flip-vs-own-no-spec-oracle matches E5 within floor; the oracle must be the ACTUAL non-spec recurrent decode, NEVER a chunked-prefill / streamed-logprobs / serial-torch ref / fallback path / backend NAME (bug-class #10/#11), and the served stream must be the ACTUAL spec-decode output (cat9 tree-verify / native MTP-5), non-vacuous (#9). CPU read-only on banked data + harness scripts + vLLM source via vllm_src.sh. Output FR13_CONFIRM_SPEC_VS_NONSPEC.md.',
  phases: [
    { title: 'Confirm' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10. Repo /home/mark/shared/lumoFlyWheel. CPU read-only (GPU free but do NOT boot - this is a',
'pure confirmation on BANKED data + source). Read the harness scripts + the banked evidence + vLLM source via',
'scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). Write ONLY FR13_CONFIRM_SPEC_VS_NONSPEC.md. Pathspec commit.',
'',
'THE BIG-DENOM RESULT to validate (output/fr13_bigdenom_rescore/consolidated.json): cat9 = 1181/8717 = 13.548%',
'CI[12.85,14.28]; native = 1224/8752 = 13.985% CI[13.27,14.73]; CIs OVERLAP, cat9 slightly LOWER. Each arm vs',
'its OWN no-spec RECURRENT decode oracle; clear-margin = served != recurrent_argmax AND (out-of-top20 OR',
'deviation_nat>1.0). It CLAIMS non-vacuity: oracle_engaged_both, recurrent_decode_calls 830496(cat9)/834048',
'(native), within_proc_det_both, gold_margin_in_process_recurrent_not_streamed_logprobs, denominator_is_',
'validated_roundtrip_tokens. spec_frozen_evidence.json at output/fr13_bigdenom_rescore/spec_frozen_evidence.json.',
'',
'THE ONE QUESTION (user): is this GENUINELY spec-decode (served) vs non-spec (recurrent) decode, BOTH arms? If',
'yes, the result is a valid lossless-vs-native comparison and lossless is met at scale. CONFIRM each link:',
'1. SERVED = SPEC-DECODE: the cat9_src.json / native_src.json served streams are the ACTUAL spec-decode serve',
'   output - cat9 = the forked-FA2 tree-verify spec path ENGAGED (the 9-node caterpillar verified per step);',
'   native = the MTP-5 spec path ENGAGED (5-token linear spec). Confirm via the serve launch flags (fr13_launch',
'   _forked_fa2_tree_server.sh + the native launch) + the proxy pair-dump capture (inference_proxy.py LUMO_PROXY',
'   _PAIR_DUMP) - the served tokens being rescored came from a real spec serve, not a non-spec or fallback serve.',
'2. ORACLE = NON-SPEC RECURRENT DECODE: the rescore (scripts/fr13_recurrent_decode_oracle.py rescore --arm) runs',
'   the ACTUAL non-spec, single-token, RECURRENT decode - FR12_NO_SPECULATIVE_CONFIG=1 (spec disabled, counters',
'   CANNOT advance), the GDN recurrent single-step path genuinely engaged (RECURRENT_PATH_ENGAGED, ~830k calls =',
'   1 per decoded token), NOT chunked re-prefill, NOT streamed logprobs, NOT a serial-torch reference, NOT a',
'   backend NAME, NOT a silent fallback. READ the oracle source + the in-container rescore wrapper (fr13_recur_',
'   rescore_in_container.sh) + the relevant vLLM decode dispatch to PROVE it is the deployment recurrent decode.',
'3. BOTH ARMS IDENTICAL FRAMING: native_a and cat9_a both = served-from-spec-serve, rescored-by-same-recurrent-',
'   oracle, same clear-margin def, same threshold/top-k (1.0/20). No asymmetry (e.g. one arm chunked, one',
'   recurrent; or different oracle per arm).',
'4. NON-VACUOUS: spec_frozen_evidence.json actually shows the spec config FROZEN during the oracle (counters',
'   pinned); the recurrent path actually RAN (not 0-call fallback); the denominator = validated round-trip',
'   tokens (detok byte-exact, dropped turns accounted). The 830k recurrent calls are REAL.',
'',
'RELEVANT FILES (read the ACTUAL code, cite line numbers): scripts/fr13_bigdenom_autoadvance.sh, fr13_bigdenom_',
'phase3_rescore.sh, fr13_recur_rescore_in_container.sh, fr13_recurrent_decode_oracle.py, fr13_swe_stream_to_',
'oracle_src.py, the proxy pair-dump in src/.../inference_proxy.py (LUMO_PROXY_PAIR_DUMP), the serve launchers;',
'output/fr13_bigdenom_rescore/{consolidated.json,spec_frozen_evidence.json,rescore_{cat9,native}.json head};',
'vLLM recurrent decode path via scripts/vllm_src.sh. int-view never atol.',
'',
'DELIVERABLE: FR13_CONFIRM_SPEC_VS_NONSPEC.md = a per-link CONFIRMED/REFUTED table (served=spec both arms;',
'oracle=non-spec-recurrent both arms; identical framing; non-vacuous) with the source citations, and a single',
'VERDICT: is the big-denom a valid spec-vs-non-spec lossless comparison (=> cat9~=native at scale is a real',
'lossless-vs-native PASS) or is there a framing flaw (name it). Distinguish CODE-READ from INFERRED. Quote FR13_',
'BUG_CLASS_PLAYBOOK (#9 vacuous, #10/#11 the incumbent/oracle identity). Commit pathspec.',
].join('\n');

phase('Confirm');
const C_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['servedIsSpec','oracleIsNonSpecRecurrent','bothArmsIdentical','nonVacuous','verdict','committed','notes'],
  properties: {
    servedIsSpec: { type: 'string', description: 'CONFIRMED/REFUTED + source: are the cat9/native served streams the ACTUAL spec-decode output (cat9 tree-verify engaged, native MTP-5 engaged), captured via the proxy pair-dump from a real spec serve?' },
    oracleIsNonSpecRecurrent: { type: 'string', description: 'CONFIRMED/REFUTED + source: is the rescore oracle the ACTUAL non-spec recurrent single-token decode (FR12_NO_SPECULATIVE_CONFIG, recurrent path engaged ~830k calls), NOT chunked-prefill/streamed/serial-ref/backend-name/fallback?' },
    bothArmsIdentical: { type: 'string', description: 'CONFIRMED/REFUTED: native_a and cat9_a use identical framing (served-from-spec, same recurrent oracle, same clear-margin def/threshold), no asymmetry?' },
    nonVacuous: { type: 'string', description: 'CONFIRMED/REFUTED: spec_frozen_evidence shows spec frozen, recurrent path actually ran (not 0-call), denominator = validated round-trip tokens?' },
    verdict: { type: 'string', description: 'single: is the big-denom a VALID spec-vs-non-spec lossless comparison (cat9~=native = real lossless-vs-native PASS at scale) or is there a framing flaw (named)?' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const c = await agent(
  CTX + '\n\nTASK (Confirm, CPU read-only, no GPU). Read the harness scripts + the oracle source + vLLM dispatch '
  + '+ the banked evidence; confirm each link (served=spec / oracle=non-spec-recurrent / both-arms-identical / '
  + 'non-vacuous) source-grounded. Write FR13_CONFIRM_SPEC_VS_NONSPEC.md, commit pathspec. Return the schema.',
  { label: 'confirm-spec-vs-nonspec', phase: 'Confirm', schema: C_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','servedSpecGrounded','oracleNonSpecGrounded','symmetryChecked','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    servedSpecGrounded: { type: 'string', description: 'is "served=spec" CODE-READ (serve launch flags + proxy pair-dump path), not asserted from the consolidated.json claim?' },
    oracleNonSpecGrounded: { type: 'string', description: 'is "oracle=non-spec recurrent" PROVEN from the oracle source + the vLLM decode dispatch + FR12_NO_SPECULATIVE_CONFIG (counters frozen), and explicitly NOT chunked-prefill/streamed/serial/backend-name/fallback?' },
    symmetryChecked: { type: 'string', description: 'was the both-arms-identical-framing claim actually checked (not assumed) - same oracle, same def, no per-arm asymmetry?' },
    recommendation: { type: 'string', description: 'single: is the big-denom a valid spec-vs-non-spec comparison (lossless met at scale, proceed to speed) or is there a flaw to fix first? No close/pass-fail unless clean.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(c) + '. Default holds=false if "served=spec" or "oracle=non-'
  + 'spec-recurrent" is asserted from the consolidated.json non_vacuity flags rather than CODE-READ from the '
  + 'oracle source + vLLM dispatch + the serve launch + the pair-dump; if the oracle could be chunked-prefill / '
  + 'streamed-logprobs / a serial-torch ref / a backend-name / a silent fallback (the exact things the user said '
  + 'we previously got wrong); or if the two arms are not symmetric. int-view never atol. No close unless the '
  + 'spec-vs-non-spec framing is genuinely confirmed both arms.',
  { label: 'verify-spec-vs-nonspec', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { c, v };
