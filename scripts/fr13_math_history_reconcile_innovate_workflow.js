export const meta = {
  name: 'fr13-math-history-reconcile-innovate',
  description: 'USER (2026-06-15, sharpened): a GENUINE math-expert workflow that (1) studies the FULL FR13 drift history, (2) CORRECTS the contradictory numbers + nuances the confounds across boots/trajectories/lengths (the reason verify-held studies CONTRADICT each other: wsvy4vn5k says co-residency/M-invariance, carrier_reopen says trajectory-fork/topology; conv-18.375 was a poisoned-ref CONFOUND closed; scan state-feed null made flips WORSE 23->32; in_proj_ba already FIXED; QPAD e2e-null), THEN (3) innovates on the MATH **and the MEASUREMENT** with CPU Python experiments on the banked data toward native-3. Phase1 = fan-out readers over 6 history lineages (conv / scan / FA2 / in_proj_ba-BI-fp8 / acceptance-deficit-topology / confound-measurement) extracting MEASURED-vs-INFERRED + the confound + the CORRECTED number, then synthesize ONE confound-corrected picture + the REAL residual. Phase2 = design THE confound-free measurement instrument (immune to cross-boot autotune +-9, trajectory-fork, length/denominator, de-cascade-by-construction, chunked-vs-recurrent oracle frame, BI asymmetry) + run innovative CPU Python experiments + propose a non-trivial route to native-3 (NOT the refuted seams; or honestly reframe the target if native-3 is the wrong/confounded bar). Phase3 = adversarial verify. CPU read-only (a big-denom serve runs - NO GPU, do NOT edit served files), write a doc + the instrument + experiments. Output FR13_MATH_HISTORY_RECONCILE.md.',
  phases: [
    { title: 'HistoryReconcile' },
    { title: 'Synthesize' },
    { title: 'InnovateMathMeasure' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN + 16 full-attn). Repo',
'/home/mark/shared/lumoFlyWheel. A big-denom GPU serve runs concurrently => CPU-ONLY, NO GPU boot, READ-ONLY on',
'served code (do NOT edit scripts/fr10_phase4_patch_vllm_tree_gdn.py etc.). Read: our committed code, GIT HISTORY',
'(git show <commit> / git log), the banked workflow raws research/fr13_workflows/*.raw.json + INDEX.md, the FR13_*',
'binds, and vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER a /tmp cache).',
'',
'THE PROBLEM (user): cat9 (9-node caterpillar tree-verify) drifts ~23 clear-margin per-token argmax flips vs its',
'own no-spec RECURRENT decode oracle; native E5 (linear MTP-5) drifts ~3 (its realization floor). The per-event',
'SUPERSET gate PASSES (+15 net lossless, leaves net-positive); the residual is the SPINE tree-verify drift. We',
'want a non-trivial route to native-3 - OR an honest, confound-corrected statement of the real bar.',
'',
'WHY THIS WORKFLOW EXISTS (the user\'s key point): prior verify-HELD studies CONTRADICT each other because the',
'MEASUREMENT is confounded across boots/trajectories/lengths. They each passed an internal adversarial verify yet',
'reached opposite conclusions. A genuine math workflow must FIRST reconcile the full record + correct the numbers',
'+ nuance the confounds, THEN innovate. Do NOT anchor on any single study\'s framing (that is how wsvy4vn5k',
'regressed - it re-proposed M-invariance via the conv-state-feed seam whose DIRECT test had already failed).',
'',
'THE CONFOUND CATALOG (watch for ALL of these - they are why the numbers disagree):',
'- CROSS-BOOT autotune +-9 flips: two boots of the SAME build fork at tokens 11-71 (autotune floor, NOT a',
'  behavior change). NEVER gate on cross-boot byte-identity. Raw flip counts across boots are +-9 noise.',
'- TRAJECTORY-FORK inflation: cat9 accepts more (leaves) => different served stream => different DOWNSTREAM',
'  flips. cat9-vs-chain5 or OFF-vs-recompute are NOT apples-to-apples unless the trajectory is HELD FIXED. The',
'  scan-recompute "23->32 worse" was length + trajectory re-roll (honest like-for-like ~25 vs 23).',
'- LENGTH / DENOMINATOR (class #12): early-EOS vs run-to-128 changes the scored-position denominator; use the',
'  per-1000-token RATE, not the raw count.',
'- DE-CASCADE BY CONSTRUCTION: the recurrent oracle teacher-forces served[i] then advances state, so a',
'  sequential cascade auto-de-cascades and re-convergence-in-1 is partly an instrument artifact, NOT proof of',
'  per-forward independence. Use the de-cascade-aware INDEPENDENT-event count + root-fork-aware basin collapse.',
'- ORACLE FRAME: the binding oracle = the no-spec RECURRENT single-step decode (fr13_recurrent_decode_oracle),',
'  NOT chunked re-prefill / streamed logprobs / a serial-torch ref / a backend NAME (chunked-vs-recurrent is a',
'  ~1-ULP realization gap that mis-measured the scan as 9x worse). int-view NEVER atol.',
'- BI ASYMMETRY: tree ran BI=1 while native BI=0 in some gates; pin BI identically on both arms or it confounds.',
'- SCALAR BLINDSPOT: accept/event, bag-TV, pass-rate, superset-count each individually MISSED a real per-token',
'  defect; the binding instrument is the per-token argmax-vs-clean-recurrent-oracle probe.',
].join('\n');

phase('HistoryReconcile');

const LINEAGES = [
  { key: 'conv', label: 'conv-priorwindow', focus:
    'The CONV prior-window lineage. Was the conv1d prior-window / state-bank read the carrier? Read commits '
    + '605c2665 (conv prior-window A/B CLOSED, 18.375 = poisoned-ref confound), b6c30b4b (conv fix design, 18.375 '
    + 'FIXED+STALE, h0 byte-exact), 4e415fb8 (conv FIXED+CLOSED cross-event), dcab4049, c0b53f5d; raws conv_'
    + 'priorwindow_ab_wf2r62ew1, conv_crossevent_investigate_ww22n39bi, conv_mechanism_wf_d27ad02e, convfix_ab_'
    + 'wtyo89tvi; the live committed-path conv at fr10_phase4...py FR13_CONV_COMMITTED_PATH. EXTRACT: was the '
    + '18.375 real or a confound? is conv the carrier now? what was MEASURED vs INFERRED, and the corrected number.' },
  { key: 'scan', label: 'gdn-scan-statefeed', focus:
    'The GDN SCAN state-feed lineage (the "co-residency / M-invariance" core). Read commits 318f8b9f (DECISIVE: '
    + 'scan state-feed null/recompute bit-exact 0.0 BUT e2e flips ROSE 23->32, NOT the carrier), ec342d86 + '
    + 'c368bc5f (+2 spine floor = chunk-vs-recurrent, measured vs WRONG oracle frame), dcab4049, 694b9813; raws '
    + 'carrier_reopen_wca1y4nll (the reconciliation: length+trajectory confound, HYBRID trajectory-fork-dominated), '
    + 'bf16_fp32_seam_scan; binds FR13_SCAN_NOT_E2E_CARRIER_BIND; the K1 (k1_mechanism_proof DO-NOT-BAKE) + N_PAD '
    + '(npad_invariant_test NULL, spine already N_PAD-invariant via one-hot select) results. EXTRACT: did nulling '
    + 'the scan state-feed move e2e flips toward native-3? the confound in "23->32 worse"; the real chunk-vs-'
    + 'recurrent floor (~1-2 ULP); is the scan the carrier (MEASURED answer).' },
  { key: 'fa2', label: 'forked-fa2-querytile', focus:
    'The forked-FA2 query-tile lineage (the "second seam"). Read commits 9ad6793f (FA2 A/B -> M_DEPENDENT, '
    + 'declared the 22-flip carrier), 030a1c22 (FR13_FA2_QPAD M-invariant query-pad fix built), 8b7684dd (QPAD '
    + 'fixed named carrier L31 3.9e-3->0.0 BUT e2e flips stayed 24 => OVERTURNED, NOT the carrier; first-nonzero '
    + 'is L0 GDN upstream of L3 full-attn), 06676346, b1176560; raw fa2_minvariance_ab_wob0t2y8v; the FA2-fork '
    + 'code scripts/fr13_patch_fa2_tree_bias.py (ancestor bias = 0.0 on spine = byte-identical to FLASH). EXTRACT: '
    + 'was the FA2 query-tile fix (QPAD) tried? did it move e2e flips? the corrected verdict (downstream amplifier '
    + 'not originator, 2-ULP/983k floor).' },
  { key: 'inproj_bi', label: 'inproj-ba-bi-fp8', focus:
    'The in_proj_ba / batch-invariance / fp8-GEMM lineage (the ONE genuinely M-dependent op). Read commits '
    + '464013ce (ba-proj fix CONFIRMED real, same-boot flag-OFF=26 vs ON=18 = -8), e89c4003 (+13 residual: conv/'
    + 'scan/fp8/gate code-proven M-invariant, only bf16 GEMM=in_proj_ba, FIXED; the rest depth-intrinsic), '
    + '4842818a (fp8 qkvz/gate/conv genuinely M-invariant), 66084461 (BI COUNTERPRODUCTIVE cat9+BI=34>22); raws '
    + 'baproj_wire_blocker, baproj_implement_test, bv_ab_lossless_verify. EXTRACT: which op was genuinely M-'
    + 'dependent, was it fixed + how many flips it bought (~-8), why BI made it worse, what is left after in_proj.' },
  { key: 'accept_topo', label: 'acceptance-deficit-topology', focus:
    'The acceptance-deficit + drafter + TOPOLOGY/reshape lineage (the carrier_reopen actionable lever). Read the '
    + 'acceptance-ladder bind FR13_ACCEPTANCE_LADDER_BIND (S1 committer bonus-row / S2 episodic verify corruption '
    + '/ S3 drafter-spine-not-token-identical), commit carrier_reopen c14393e4 (H-FORK-AMPLIFICATION: NOT a '
    + 'fixable per-forward kernel seam; dominant lever = TOPOLOGY = shallower/root-sibling tree cat3w/chain3, infra '
    + 'already committed _fr10_cat3w_choices/_fr10_chain3_choices); raws chain3_cat3w_wf_5db71b8b, cat10_*, branch_'
    + 'flip_fix_plan, branch_upside; the user reshape-lever call (project_fr13_tree_reshape_unifying_lever). '
    + 'EXTRACT: the flip STRUCTURE (hard out-of-topk forks vs fork-progeny basins vs structural-boundary crossings '
    + 'vs true near-ties - the dev distribution), and the topology lever + its predicted reach, MEASURED vs INFERRED.' },
  { key: 'confound', label: 'confound-measurement', focus:
    'The CONFOUND / MEASUREMENT lineage = catalog every confound that made flip-counts unreliable + the corrected '
    + 'measurement frame. Cross-boot +-9 autotune (feedback no_cross_boot_byte_gate); trajectory-fork inflation + '
    + 'length/denominator + de-cascade-by-construction (carrier_reopen, class #12); oracle frame chunked-vs-'
    + 'recurrent (ec342d86, must use fr13_recurrent_decode_oracle); the per-event SUPERSET gate (FR13_PEREVENT_'
    + 'SUPERSET_GATE_RESULT net +15); the scalar-metric blindspot (FR13_GATE_BLINDSPOT, the gold-margin argmax '
    + 'probe fr13_gold_margin_probe); BI-pin-both-arms; the de-cascade independent-event rule. EXTRACT: for EACH '
    + 'confound, how big it is + how to NEUTRALIZE it in a sound instrument; what the confound-free residual frame '
    + 'looks like (per-1000-token rate, trajectory-fixed in-process A/B, recurrent-oracle, de-cascade-aware count).' },
];

const READER_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lineage','measuredVsInferred','confounds','correctedNumbers','holdsConfoundFree','citations'],
  properties: {
    lineage: { type: 'string' },
    measuredVsInferred: { type: 'string', description: 'per key study/commit in this lineage: what was MEASURED (live A/B, int-view, captured) vs INFERRED (cross-boot, narrative, single-draw)' },
    confounds: { type: 'string', description: 'which confounds (cross-boot/trajectory/length/de-cascade/oracle-frame/BI) affected THIS lineage\'s numbers + how big' },
    correctedNumbers: { type: 'string', description: 'the confound-CORRECTED numbers for this lineage (e.g. QPAD e2e-null, scan-null like-for-like ~25 vs 23, in_proj -8, conv closed)' },
    holdsConfoundFree: { type: 'string', description: 'which conclusions SURVIVE confound-correction and which are refuted/confounded; is this lineage\'s seam the carrier or not (MEASURED)' },
    citations: { type: 'string', description: 'exact commits/raws/binds/source lines read' },
  },
};

const readers = await parallel(LINEAGES.map((L) => () =>
  agent(
    BASE + '\n\nTASK (HistoryReconcile, lineage = ' + L.label + '). ' + L.focus + ' Read the ACTUAL commits/raws/'
    + 'source (cite them); separate MEASURED from INFERRED; name the confounds + their size; give the corrected '
    + 'numbers; state confound-free whether this lineage\'s seam is the carrier. Return the schema.',
    { label: 'hist:' + L.key, phase: 'HistoryReconcile', schema: READER_SCHEMA }
  ).then((r) => ({ key: L.key, ...r })).catch(() => null)
));
const goodReaders = readers.filter(Boolean);
log('HistoryReconcile: ' + goodReaders.length + '/' + LINEAGES.length + ' lineages read');

phase('Synthesize');
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['correctedTimeline','realResidual','refutedLeads','contradictionsResolved','openQuestions'],
  properties: {
    correctedTimeline: { type: 'string', description: 'the confound-corrected timeline of what was tried + what each actually showed (de-confounded)' },
    realResidual: { type: 'string', description: 'the CONFOUND-FREE residual: the real flip count/rate + WHERE it lives + WHAT KIND (hard fork / basin progeny / structural-boundary crossing / true near-tie), after correcting every confound' },
    refutedLeads: { type: 'string', description: 'which leads are confounded/refuted + WHY (conv-18.375 confound, scan-null trajectory+length, FA2-QPAD e2e-null, in_proj-done, M-invariance-as-a-single-seam)' },
    contradictionsResolved: { type: 'string', description: 'how wsvy4vn5k (co-residency/M-invariance) vs carrier_reopen (trajectory-fork/topology) RECONCILE confound-free' },
    openQuestions: { type: 'string', description: 'what is genuinely still open (never cleanly tested) vs closed' },
  },
};
const synth = await agent(
  BASE + '\n\nTASK (Synthesize). Reconcile the ' + goodReaders.length + ' lineage reads into ONE confound-'
  + 'corrected picture. Lineage reads: ' + JSON.stringify(goodReaders) + '. Produce the corrected timeline, the '
  + 'REAL confound-free residual (count/rate + where + kind), the refuted leads with WHY, the wsvy4vn5k-vs-'
  + 'carrier_reopen reconciliation, and the genuinely-open vs closed questions. Do NOT anchor on one study. '
  + 'Return the schema.',
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA, model: 'opus' }
);

phase('InnovateMathMeasure');
const INNO_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['measurementInnovation','mathInnovation','pythonExperiments','pathToNative3','committed','notes'],
  properties: {
    measurementInnovation: { type: 'string', description: 'THE confound-free measurement instrument (immune to cross-boot/trajectory/length/de-cascade/oracle-frame/BI) - the design + why each confound is neutralized; the one number that resolves the contradictions' },
    mathInnovation: { type: 'string', description: 'innovative MATH levers toward native-3 that are NOT the refuted seams (e.g. margin-aware boundary handling, the structural-boundary crossing structure, a confound-free reframed target, topology math, an error model of the residual) - with the reasoning' },
    pythonExperiments: { type: 'string', description: 'the CPU Python experiments actually RUN on the banked data (the off_recur_flips.json / per_layer ladders / dump+oracle joins) with the new instrument - the numbers, a neg-control' },
    pathToNative3: { type: 'string', description: 'the non-trivial route to native-3, OR an honest confound-corrected statement (native-3 is the wrong/confounded bar; the real bar is X, and cat9 is at Y vs it) + the minimal GPU validation' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const inno = await agent(
  BASE + '\n\nTASK (InnovateMathMeasure, CPU + Python on banked data, NO GPU, do NOT edit served files). Given the '
  + 'confound-corrected picture: ' + JSON.stringify(synth) + '\n\nNow INNOVATE on MATH **and** MEASUREMENT. '
  + '(1) Design THE confound-free measurement instrument that resolves the contradictions (immune to every '
  + 'confound in the catalog) + WRITE it as a CPU Python script + RUN it on the banked data. (2) Propose '
  + 'innovative MATH levers toward native-3 that are NOT the refuted seams (conv/scan/FA2/in_proj/M-invariance-'
  + 'single-seam all tried). (3) State the non-trivial route to native-3 OR honestly reframe the bar with numbers. '
  + 'Write FR13_MATH_HISTORY_RECONCILE.md + the instrument + experiment scripts, commit pathspec '
  + '(git commit -m ".." -- <files>, NEVER git add -A). Return the schema.',
  { label: 'innovate', phase: 'InnovateMathMeasure', schema: INNO_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','historyComplete','confoundsCorrected','measurementSound','innovationReal','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    historyComplete: { type: 'string', description: 'did it read the FULL history (spot-check citations exist + say what the commits claim) - not a partial/narrative read?' },
    confoundsCorrected: { type: 'string', description: 'are the numbers genuinely confound-corrected (cross-boot/trajectory/length/de-cascade/oracle-frame), or re-asserting confounded raw counts?' },
    measurementSound: { type: 'string', description: 'is the new instrument genuinely immune to the confounds (spot-check it was RUN on banked data + the neg-control), not another vacuous metric (#9)?' },
    innovationReal: { type: 'string', description: 'are the Python experiments actually run + the math levers NOVEL (not a refuted seam re-proposed, not reward-hack)?' },
    recommendation: { type: 'string', description: 'single: the soundest non-trivial route to native-3 (or the honest confound-corrected bar + the minimal GPU validation). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY the synthesis + innovation: ' + JSON.stringify({ synth, inno }) + '. Default '
  + 'holds=false if the history read is partial/narrative (spot-check commits actually say what is claimed), the '
  + 'numbers are confounded-raw not corrected, the new instrument was not actually run on banked data or is '
  + 'another vacuous scalar, the "innovation" re-proposes a refuted seam (conv/scan/FA2/in_proj/M-invariance-'
  + 'single-seam) or is a reward-hack (copy/dense/multispine/bonus/WY), or native-3 is asserted-reachable without '
  + 'a confound-free basis. research-before-deadend. No close/pass-fail.',
  { label: 'verify-reconcile-innovate', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { synth, inno, v };
