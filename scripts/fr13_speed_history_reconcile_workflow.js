export const meta = {
  name: 'fr13-speed-history-reconcile',
  description: 'USER (2026-06-15): "study the FULL git history on speed - we already got VERY CLOSE to native on cost with TWO extra levers on speed." Correct the stale ~2.3x framing (that was PRE-FIX-1, ~96% drafter double lm-head). Real current state from history: FIX-1 single-logits + FIX-2 eager-pack (chain5 1.033x / cat9 1.077x native, 7fe500b5) + FIX-3 conv-fusion + in_proj_ba bake (speed-neutral 4d0452df) => per-forward tax now ~1.05x NOT 1.63x (fdf5ffa7), accept/event ~3.18 ~= native 3.16 (FIX-A ac1d3039). The TWO LEVERS (df631112): OPT-1 = GPU-resident committer (kills the main-thread sync-DtoH readback = dominant remaining tax, restores run-ahead, a0e8cc3d) + OPT-A = GB10-tuned fp8 GEMV (no GB10 config exists, ~1.45-1.55x s/fwd reachable, lossless-by-construction, 087fbd51). Fan-out readers over 5 speed lineages (lm-head FIX-1/2/3 / per-forward-tax / accept-side / the-two-levers / measurement-methodology), each MEASURED-vs-INFERRED + corrected number, then synthesize the REAL current speed state + the two-lever path to STRICTLY sub-native + the clean B=1 measurement protocol. CONSTRAINT (6c5aeaae): tree-only overhead removal, align-to-native allowed, NEVER modify native\'s shared path. NO hand-rolled TPS/accept decompositions as MEASURED facts (decode_seconds basis only). CPU read-only (GPU free, do NOT boot - measurement runs after). Output FR13_SPEED_HISTORY_RECONCILE.md.',
  phases: [
    { title: 'SpeedHistory' },
    { title: 'Synthesize' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode HBM-bandwidth-bound; native = ~45% of peak BW per',
'087fbd51). Repo /home/mark/shared/lumoFlyWheel. CPU read-only (GPU free but do NOT boot - the measurement runs',
'AFTER, under the prelaunch host-mem protocol). Read our code + GIT HISTORY (git show <commit>/git log) + the',
'banked speed binds + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). Write ONLY FR13_SPEED_',
'HISTORY_RECONCILE.md. Pathspec commit.',
'',
'GOAL: cat9 B=1 decode-TPS STRICTLY > native E5, lossless held as a per-change gate (same-seed byte-identical',
'streams greedy + accept/event unchanged + regular-decode pristine). Lossless-vs-native is already CONFIRMED at',
'scale (big-denom cat9 13.55% ~= native 13.99%); accept/event ~3.18 ~= native 3.16. So this is purely the SPEED',
'reconcile + the path to strictly sub-native.',
'',
'THE USER CORRECTION (do NOT repeat the stale framing): the ~2.336x-slower number (FR13_WHY_SLOWER_VERDICT) is',
'STALE - it PRE-DATES FIX-1/2/3. The original gap was ~96% DRAFTER DOUBLE LM-HEAD (008631cd), not the GDN scan',
'tax. After the fixes we got VERY CLOSE to native on cost, and there are TWO extra levers to cross to sub-native.',
'CORRECT every stale speed number against the full history + current code.',
'',
'THE REAL CURRENT STATE (from history - CONFIRM + sharpen, do NOT re-derive): FIX-1 single-logits (drafter double',
'lm-head -> single, 93a4043a: s/fwd 0.321->0.246 @11k ctx); FIX-2 eager-pack (7fe500b5: chain5 1.033x native,',
'cat9 1.077x); FIX-3 conv-emulation fusion (f42aab8c); in_proj_ba baked SPEED-NEUTRAL (4d0452df: s/fwd 0.2248 vs',
'0.2249); per-forward tax now ~1.05x = +10ms/fwd (fdf5ffa7, NOT +96 pre-FIX-1; lm-head ALREADY batched, residual',
'= GDN state-row traffic + N_PAD16 spill); accept/event cat9 ~3.18 >= native ~3.16 (FIX-A ac1d3039, +1.15 from',
'2.03). So cat9 is ~1.05-1.08x native s/fwd at accept-parity = VERY close, NOT 2.3x.',
'',
'THE TWO LEVERS (df631112 "verify 2 speed fixes OPT-1 + OPT-A lossless AND fast"): (1) OPT-1 = GPU-RESIDENT',
'COMMITTER (a0e8cc3d) - the committer sync-DtoH readback is the dominant remaining main-thread tax (dd45c3c1);',
'moving the committer GPU-resident kills the sync + restores run-ahead => s/fwd toward parity + the accept-edge',
'becomes a TPS win; lossless (committer logic unchanged, just where it runs). (2) OPT-A = GB10-TUNED fp8 GEMV',
'(087fbd51) - NO GB10 fp8-GEMV config exists in the stock kernel; a GB10-tuned config is lossless-by-construction',
'(same math, better tiling) and reaches ~1.45-1.55x s/fwd (140-150ms vs 218) - a FUNDAMENTAL bandwidth win (NOT',
'the 98.6ms peak floor - GB10 lacks TMA/WGMMA). CONSTRAINT (6c5aeaae): both must be TREE-ONLY / align-to-native,',
'NEVER modify the machinery native\'s path executes (spine===native superset premise; align-to-native allowed,',
'deviate-shared banned).',
'',
'MEASUREMENT RULES (reference_fr10_speed_measurement_pitfalls + feedback_dont_handroll_speed): basis = decode_',
'seconds RAW counter, NEVER TPS/accept (banned) nor wall (tree early-stops); metrics OFF (FR10_METRICS) + FR12/13',
'diagnostics compiled out; per-request not aggregate; same pinned prompts temp 0.0 seed 1313; BI pinned identical',
'both arms (BI=1 is NOT cross-boot det at B=1 on GB10, 93a4043a); the diag-residue is <=2.5% of the gap (46e89f22,',
'gold TPS comparison FAIR). Do NOT present any per-forward TPS/accept decomposition as a MEASURED fact - label',
'INFERRED; the real number needs the clean GPU measurement.',
].join('\n');

phase('SpeedHistory');
const LINEAGES = [
  { key: 'lmhead', label: 'lmhead-FIX-1-2-3', focus:
    'The lm-head / FIX-1/2/3 lineage = the bulk of the original gap. Read commits 008631cd (~96% gap = drafter '
    + 'double lm-head), dd45c3c1 (committer sync-DtoH recon), edc39213 (joint lm-head load-bearing), 93a4043a '
    + '(FIX-1 single-logits, s/fwd 0.321->0.246), 7fe500b5 (FIX-2 eager-pack, chain5 1.033x/cat9 1.077x native), '
    + 'f42aab8c (FIX-3 conv-fusion), fdf5ffa7 (lm-head ALREADY batched, tax now ~1.05x). EXTRACT: the s/fwd '
    + 'progression with EXACT numbers, what each fix removed, what is left after FIX-1/2/3 (MEASURED vs INFERRED).' },
  { key: 'tax', label: 'per-forward-tax', focus:
    'The per-forward-tax lineage = the ~2.3x->~1.05x correction + the current residual. Read FR13_WHY_SLOWER_'
    + 'VERDICT (the STALE 2.336x = 1.432x forwards x 1.632x/fwd), fdf5ffa7 (tax now ~1.05x=+10ms, residual = GDN '
    + 'state-row traffic + N_PAD16 spill NOT lm-head), 0ecd94aa (~97% fixed overhead not bandwidth, 3x graph '
    + 'nodes), 4d0452df (in_proj_ba speed-neutral), 4b409630/07f7ce6a/3fd0717c (N_PAD16 spill + recompute-from-'
    + 'spine spill-free), b12d8a40 (num_warps geometry). EXTRACT: what is the CURRENT per-forward residual tax + '
    + 'its decomposition (GDN state traffic / spill / graph-node overhead), and which part each lever removes.' },
  { key: 'accept', label: 'accept-side', focus:
    'The accept-side lineage = tokens-committed-per-step (the TPS numerator). Read ac1d3039 (FIX-A accept 2.03->'
    + '3.1789 ABOVE native 3.1613), 4b6769ee + fdf5ffa7 (the accept-gap debate - is cat9 above or behind native?), '
    + 'the per-event SUPERSET gate (FR13_PEREVENT_SUPERSET_GATE_RESULT +15), the L3 confidence-gated root sibling '
    + '(wgb0yegin, 62% of rejects are step-0). EXTRACT: the CURRENT cat9 vs native accept/event (MEASURED, depth-'
    + 'matched), whether cat9 has a real accept EDGE (TPS numerator win), and the L3 lever\'s accept reach.' },
  { key: 'levers', label: 'the-two-levers', focus:
    'THE TWO LEVERS lineage = the path to sub-native. Read a0e8cc3d (OPT-1 GPU-resident committer, kills main-'
    + 'thread sync, restores run-ahead), dd45c3c1 (committer sync-DtoH = dominant tax), 087fbd51 (OPT-A GB10-tuned '
    + 'fp8 GEMV, ~1.45-1.55x s/fwd, lossless-by-construction, no GB10 config exists), df631112 (both as the endgame '
    + '2 fixes), 6c5aeaae (the tree-only/align-to-native constraint). EXTRACT for EACH lever: exact mechanism, '
    + 'projected s/fwd reach, IS IT IMPLEMENTED/TESTED/BANKED or just designed (check git + branches), lossless '
    + 'argument, tree-only-vs-shared-path (constraint compliance), and any blocker.' },
  { key: 'measurement', label: 'measurement-methodology', focus:
    'The measurement-methodology lineage = how to measure speed CORRECTLY. Read d7ea6ccd (exit bars: accept STRICTLY'
    + '>native before B=4, speed target at-or-under native, try sub-native s/fwd), 46e89f22 (diag-residue <=2.5%, '
    + 'gold TPS FAIR), reference_fr10_speed_measurement_pitfalls (decode_seconds basis, TPS/accept BANNED, per-'
    + 'request), 93a4043a (BI=1 not cross-boot det at B=1), the existing speed harness + the E5 native ref '
    + '(output/fr10_native_mtp5_same8_*). EXTRACT: the EXACT clean B=1 measurement protocol (basis, flags, prompts, '
    + 'prelaunch, BI) + the existing scripts to reuse; the pitfalls to avoid.' },
];

const READER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lineage','measuredVsInferred','correctedNumbers','currentState','citations'],
  properties: {
    lineage: { type: 'string' },
    measuredVsInferred: { type: 'string', description: 'per key commit: what was MEASURED (decode_seconds, captured) vs INFERRED (projected, hand-rolled)' },
    correctedNumbers: { type: 'string', description: 'the corrected/current speed numbers for this lineage (s/fwd, accept, tax, lever reach), superseding stale ones' },
    currentState: { type: 'string', description: 'what this lineage says about the CURRENT cat9-vs-native speed state + the lever(s) it bears on' },
    citations: { type: 'string', description: 'exact commits/binds/source lines read' },
  },
};

const readers = await parallel(LINEAGES.map((L) => () =>
  agent(
    BASE + '\n\nTASK (SpeedHistory, lineage = ' + L.label + '). ' + L.focus + ' Read the ACTUAL commits/binds/'
    + 'source (cite them); separate MEASURED from INFERRED; give the corrected/current numbers. Return the schema.',
    { label: 'spd:' + L.key, phase: 'SpeedHistory', schema: READER_SCHEMA }
  ).then((r) => ({ key: L.key, ...r })).catch(() => null)
));
const good = readers.filter(Boolean);
log('SpeedHistory: ' + good.length + '/' + LINEAGES.length + ' lineages read');

phase('Synthesize');
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['currentSpeedState','twoLeverPath','measurementProtocol','subNativeVerdict','openQuestions'],
  properties: {
    currentSpeedState: { type: 'string', description: 'the REAL current cat9-vs-native B=1 speed state (s/fwd ~1.05-1.08x?, accept ~3.18, the residual tax decomposed), MEASURED-vs-INFERRED, superseding the stale 2.3x' },
    twoLeverPath: { type: 'string', description: 'OPT-1 (GPU committer) + OPT-A (fp8 GEMV): each lever\'s reach, implemented/tested/banked status, lossless + tree-only compliance, and the combined projected path to STRICTLY sub-native' },
    measurementProtocol: { type: 'string', description: 'the EXACT clean B=1 measurement protocol (decode_seconds basis, metrics off, per-request, pinned prompts, prelaunch, BI pinned) + existing scripts to reuse' },
    subNativeVerdict: { type: 'string', description: 'is STRICTLY sub-native reachable via OPT-1+OPT-A (the user says yes - confirm with the corrected numbers + lever reach), and the order to apply them' },
    openQuestions: { type: 'string', description: 'what needs the GPU measurement to confirm (current gap, lever reach) vs what is code-settled' },
  },
};
const synth = await agent(
  BASE + '\n\nTASK (Synthesize). Reconcile the ' + good.length + ' speed lineage reads into the REAL current '
  + 'speed state + the two-lever path to sub-native + the clean measurement. Lineage reads: ' + JSON.stringify(good)
  + '. Confirm the user\'s "very close + two levers" with corrected numbers; do NOT repeat the stale 2.3x. Return '
  + 'the schema.',
  { label: 'synthesize-speed', phase: 'Synthesize', schema: SYNTH_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','historyComplete','numbersCorrected','leversReal','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    historyComplete: { type: 'string', description: 'did it read the FULL speed history (spot-check FIX-1/2/3 + the two levers commits say what is claimed), not a partial read?' },
    numbersCorrected: { type: 'string', description: 'are the numbers corrected to the CURRENT state (~1.05-1.08x not 2.3x, accept ~3.18) and decode_seconds-based, not hand-rolled TPS/accept presented as MEASURED?' },
    leversReal: { type: 'string', description: 'are OPT-1 + OPT-A real, lossless, tree-only (not modifying native\'s shared path), with honest implemented-vs-designed status?' },
    recommendation: { type: 'string', description: 'single: the path to strictly sub-native (apply OPT-1/OPT-A in what order) + the GPU measurement to confirm. No premature STOP (the user says we are close).' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY the synthesis: ' + JSON.stringify(synth) + '. Default holds=false if it repeats '
  + 'the STALE 2.3x instead of the current ~1.05-1.08x, presents a hand-rolled TPS/accept decomposition as MEASURED '
  + '(must be decode_seconds-based or labeled INFERRED), mis-states a lever (OPT-1/OPT-A must be lossless + tree-'
  + 'only per 6c5aeaae), or prematurely concludes sub-native is unreachable (the user + the corrected numbers say '
  + 'it is close). Spot-check the FIX-1/2/3 + lever commits. No reward-hack (WY parked, no deviate-shared, no '
  + 'copy/dense/bonus). research-before-deadend.',
  { label: 'verify-speed-reconcile', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { synth, v };
