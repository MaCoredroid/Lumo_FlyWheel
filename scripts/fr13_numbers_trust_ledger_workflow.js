export const meta = {
  name: 'fr13-numbers-trust-ledger',
  description: 'USER (2026-06-15): audit the FR13 git history + banked docs and STAPLE every key number with its MEASUREMENT REGIME, because two contaminations were just found: (1) OUT-OF-DISTRIBUTION = handrolled prompts sent RAW to /v1/completions with NO chat template -> the model degenerates (empty <think></think> repetition loop, confirmed) -> accept/lossless numbers on that regime are unreliable; (2) TEMP=0 = greedy/argmax is a DIAGNOSTIC point-measure, NOT the deployment regime which is temp 0.6 + top_p 0.95 sampling ("we really should only consider under temp 0.6"). Produce a TRUST LEDGER classifying each banked number/verdict by regime: {prompt = real-SWE+codex chat-template vs raw /v1/completions handrolled}, {temp = 0 greedy-diagnostic vs 0.6 deployment-binding}, {B = 1 vs 4}, and a TIER: A deployment-binding (real-SWE+codex chat + temp0.6), B deployment-prompt-but-temp0 (chat but greedy = necessary-not-sufficient, e.g. the big-denom 13%), C regime-robust (s/fwd = content/temp-independent, trustworthy), D contaminated (raw-/v1/completions off-distribution), E diagnostic-only (temp-0 determinism/byte-exact). Then: which banked CONCLUSIONS still STAND on deployment-faithful evidence vs which must be RE-MEASURED on the deployment regime (chat + temp 0.6 + real SWE). CPU read-only (a GPU re-aim wf runs; no GPU here). fan-out over accept / lossless-flip / speed / verdicts. Output FR13_NUMBERS_TRUST_LEDGER.md.',
  phases: [
    { title: 'Audit' },
    { title: 'Synthesize' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10. Repo /home/mark/shared/lumoFlyWheel. CPU read-only (a GPU re-aim wf w8q5jg1k0 runs; do',
'NOT boot). Read GIT HISTORY (git log/show) + the banked FR13_*.md binds + research/fr13_workflows/INDEX.md +',
'the memories. Write ONLY FR13_NUMBERS_TRUST_LEDGER.md. Pathspec commit.',
'',
'THE TWO CONTAMINATIONS to staple every number against:',
'(1) OUT-OF-DISTRIBUTION PROMPT: handrolled prompts (prompts_swe4.json) sent as a RAW string to /v1/completions',
'    with NO chat template = off-distribution for this chat/thinking-trained model -> it DEGENERATES (native E5',
'    served_token_ids repeat [271,248068,271,248069,271,40]="\\n<think>\\n</think>\\nI" = empty-think repetition',
'    loop, output/fr13_measure/native_e5_q_temp06_on.json). accept/lossless numbers on this regime are UNRELIABLE',
'    (the banked native 3.161 / cat9 3.18 / the per-event superset gate / the 23-flips were on this raw regime).',
'    The deployment regime = real SWE-Verified + codex agent loop, chat-templated /v1/responses (the big-denom).',
'(2) TEMP=0 vs TEMP=0.6: temp 0 = greedy, served token = ARGMAX = a DIAGNOSTIC point-measure (good for determinism',
'    / byte-exact, NECESSARY-NOT-SUFFICIENT for lossless). Deployment = temp 0.6 + top_p 0.95 SAMPLING, where',
'    lossless is DISTRIBUTIONAL (FR13_TEMP06_DRIFT_GATE). The user: "we really should only consider under temp',
'    0.6" for the deployment-binding evaluation. The big-denom 13% (cat9 13.55% ~= native 13.99%) is chat-regime',
'    BUT temp-0 argmax = TIER B (deployment-prompt, but not yet the temp-0.6 distributional gate).',
'',
'TIERS to assign each number: A = deployment-binding (real-SWE+codex chat + temp 0.6 [+ B as noted]); B =',
'deployment-prompt-but-temp0 (chat real-SWE but greedy argmax = necessary-not-sufficient); C = regime-robust',
'(s/fwd = decode-time-per-event, ~content/temp/B-independent for bandwidth-bound, TRUSTWORTHY regardless of',
'prompt regime); D = CONTAMINATED (raw-/v1/completions handrolled off-distribution accept/lossless); E =',
'diagnostic-only (temp-0 determinism/byte-exact/argmax-localization, valid for its narrow purpose not deployment).',
'Be HONEST where the regime of a historical number is UNDETERMINABLE from the doc (mark UNKNOWN-REGIME).',
'',
'KEY NUMBERS/VERDICTS to staple (non-exhaustive - find them in the history): native E5 accept 3.161 + cat9 3.18',
'(FR13_B1_CURRENT/FIX3_GATE); s/fwd 0.2182/0.2248 + the FIX-1/2/3 1.40x->1.03x progression; the per-event',
'SUPERSET gate +15 (FR13_PEREVENT_SUPERSET_GATE_RESULT); the 23-flips decomposition (6 leaf + 17 spine); the',
'big-denom 13.55%/13.99% (FR13_CONFIRM_SPEC_VS_NONSPEC / consolidated.json); the confound-free 7.39x (wgb0yegin);',
'chain5/chain3/cat3w/cat10 reshape accept+flip numbers; the bag-TV floor 0.0593/0.1133; the OPT-1/OPT-A reach;',
'the wgb0yegin "M-invariance vs topology both refuted" verdict; the speed-history reconcile (cat9 1.03x native).',
'For EACH: its prompt-regime (chat/raw/unknown), temp (0/0.6/unknown), B, and TIER, + whether the CONCLUSION it',
'drove still STANDS (deployment-faithful) or must be RE-MEASURED on the deployment regime.',
].join('\n');

phase('Audit');
const SLICES = [
  { key: 'accept', label: 'accept-numbers', focus: 'all ACCEPT/event numbers + the break-even + the per-event SUPERSET gate (native 3.161, cat9 3.18, chain5/chain3/cat3w/cat10 accepts, the +15 net, the +0.0176 edge). For each: prompt-regime (raw /v1/completions handrolled? or chat real-SWE?), temp (0/0.6), B, TIER. Most are RAW + temp-0 = TIER D (contaminated by the degenerate fork) - confirm which.' },
  { key: 'lossless', label: 'lossless-flip-numbers', focus: 'all LOSSLESS/FLIP numbers: the temp-0 ARGMAX flips (23, the big-denom 13.55%/13.99%, the confound-free 7.39x, chain5=5/chain3=5/cat9=22), the bag-TV floors (0.0593, 0.1133), the per-token argmax probe. Staple each: chat vs raw prompt, temp 0 vs 0.6, B. The big-denom is chat (real codex) but temp-0 = TIER B; the prompts_swe4 flip numbers are raw = TIER D. Which lossless verdict is deployment-binding (NONE yet at temp-0.6)?' },
  { key: 'speed', label: 'speed-numbers', focus: 'all SPEED numbers: s/fwd (native 0.2182, cat9 0.2248, the Phase-0 0.2159/0.2241), the FIX-1/2/3 progression (1.40x->1.03x), the OPT-1 ~4-6ms reach, the OPT-A 1.45-1.55x, the stale 2.336x. s/fwd is ~regime-robust (TIER C, trustworthy) - CONFIRM it is content/temp-independent (per-event decode time, bandwidth-bound) so the 1.03x cat9-vs-native s/fwd STANDS even though accept was contaminated. Flag any speed number that secretly depends on accept (TPS) as not-regime-robust.' },
  { key: 'verdicts', label: 'driving-verdicts', focus: 'the major VERDICTS that drove decisions + whether they stand on deployment-faithful evidence: "lossless-vs-native met at scale" (big-denom, TIER B chat+temp0 - does it survive at temp 0.6?); "M-invariance + topology-reshape both refuted" (wgb0yegin, raw+temp0 - re-examine); the per-event superset PASS (raw+temp0); the reshape leads (R4 cat6root, on raw+temp0 accept); the OPT-1/OPT-A path. For each verdict: the regime of its evidence + STANDS / NEEDS-REMEASURE-on-deployment.' },
];
const READER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['slice','stapledNumbers','tierSummary','citations'],
  properties: {
    slice: { type: 'string' },
    stapledNumbers: { type: 'string', description: 'each key number in this slice with {value, prompt-regime chat/raw/unknown, temp 0/0.6/unknown, B, TIER A-E}' },
    tierSummary: { type: 'string', description: 'which numbers/conclusions in this slice are deployment-faithful (stand) vs contaminated (re-measure on chat+temp0.6+real-SWE)' },
    citations: { type: 'string', description: 'commits/docs/binds cited' },
  },
};
const readers = await parallel(SLICES.map((S) => () =>
  agent(
    BASE + '\n\nTASK (Audit, slice = ' + S.label + '). ' + S.focus + ' Cite the actual docs/commits; staple each '
    + 'number with its regime + TIER; honest UNKNOWN-REGIME where the doc does not say. Return the schema.',
    { label: 'audit:' + S.key, phase: 'Audit', schema: READER_SCHEMA }
  ).then((x) => ({ key: S.key, ...x })).catch(() => null)
));
const good = readers.filter(Boolean);
log('Audit: ' + good.length + '/' + SLICES.length + ' slices');

phase('Synthesize');
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['trustLedger','standsVsRemeasure','deploymentGaps','committed','notes'],
  properties: {
    trustLedger: { type: 'string', description: 'the consolidated TRUST LEDGER: each key number stapled with prompt-regime + temp + B + TIER A-E' },
    standsVsRemeasure: { type: 'string', description: 'which CONCLUSIONS still STAND on deployment-faithful evidence (s/fwd regime-robust; big-denom chat-prompt) vs which must be RE-MEASURED on the deployment regime (chat + temp 0.6 + real SWE)' },
    deploymentGaps: { type: 'string', description: 'what has NEVER been measured deployment-faithfully (e.g. lossless at temp 0.6 on real SWE; accept on the deployment trajectory) = the re-measure list for the re-aimed infra' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const synth = await agent(
  BASE + '\n\nTASK (Synthesize). Consolidate the ' + good.length + ' audit slices into ONE trust ledger + the '
  + 'stands-vs-remeasure split + the deployment-faithful gaps. Slices: ' + JSON.stringify(good) + '. Write FR13_'
  + 'NUMBERS_TRUST_LEDGER.md, commit pathspec. Return the schema.',
  { label: 'synthesize-ledger', phase: 'Synthesize', schema: SYNTH_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','regimesCorrect','tiersSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    regimesCorrect: { type: 'string', description: 'are the stapled regimes right (raw-/v1/completions vs chat real-SWE; temp 0 vs 0.6) - spot-check a few against the actual docs, not asserted?' },
    tiersSound: { type: 'string', description: 'are the tiers sound (s/fwd genuinely regime-robust TIER C; raw-prompt accept/lossless TIER D contaminated; big-denom TIER B chat-but-temp0); no number mis-tiered as trustworthy?' },
    recommendation: { type: 'string', description: 'single: the clean list of what STANDS vs what to RE-MEASURE on chat+temp0.6+real-SWE (the re-aimed infra target). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(synth) + '. Default holds=false if a regime is mis-stapled '
  + '(spot-check the actual doc - e.g. claiming a number was chat-templated when it was raw /v1/completions), if a '
  + 'contaminated raw+temp0 accept/lossless number is mis-tiered as deployment-trustworthy, if s/fwd is wrongly '
  + 'called contaminated (it is regime-robust) or a TPS number wrongly called regime-robust (it depends on the '
  + 'contaminated accept), or if "stands" is claimed without deployment-faithful evidence. No close/pass-fail.',
  { label: 'verify-ledger', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { synth, v };
