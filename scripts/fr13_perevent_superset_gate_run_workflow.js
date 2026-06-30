export const meta = {
  name: 'fr13-perevent-superset-gate-run',
  description: 'RUN the per-event superset gate (the user deliverable gate, plan FR13_PEREVENT_SUPERSET_GATE.md verify HOLDS, RUNNABLE-NOW CPU-only). Write the ~1-file reducer fr13_perevent_superset_gate.py + run it on the BANKED cat9 FR13_FORK_MARGIN_DUMP + recurrent-oracle rescore + the existing JOIN (fr13_fork_margin_classify) to compute the ACTUAL gate: per event, spine_regressions (STRUCTURALLY 0 by committer construction winner_lcp>=spine_lcp; measure to confirm), lossless_leaf_saves (E5 cut AND leaf==recurrent-oracle greedy), lossy_leaf_saves (E5 cut AND leaf!=greedy = the flip), net = lossless - lossy - spine_regressions; PASS = net>0 AND spine_regressions==0; per-depth d=0..4 + total. PLUS the aggregate reconciliation: the +0.12 accept/event edge = GROSS leaf-saves; how much is lossless vs lossy = whether the loose superset was a LOSSLESS or a LOSSY gain. NO GPU (banked data). Adversarial verify.',
  phases: [
    { title: 'BuildRun' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10. Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY except writing the ONE new',
'reducer scripts/fr13_perevent_superset_gate.py + its output (a big-denom GPU serve runs concurrently; do NOT',
'edit serving code/boot). FOLLOW FR13_PEREVENT_SUPERSET_GATE.md (ce912070) - it has the exact spec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol. greedy',
'reference = the deployment RECURRENT decode oracle (fr13_recurrent_decode_oracle, oracle_argmax_id), NOT',
'streamed logprobs / serial-torch / a backend name (#10/#11).',
'',
'THE GATE (user, precise per-event superset; replaces the cross-trajectory aggregate 3.198 vs 3.076):',
'  - SPINE NON-REGRESSION: cat9 accepts >= the spine tokens E5 accepts at every event. STRUCTURAL: the committer',
'    (scripts/fr10_phase4_patch_vllm_tree_gdn.py L6894-6925, strict `if lcp>best_lcp`, spine-favored tie-break)',
'    forces winner_lcp >= spine_lcp => spine_regressions == 0 by construction. MEASURE to confirm (S2/S3 intact).',
'  - LEAF SAVE = where E5 CUTS the spine (depth d == spine_lcp), cat9 saves a token via a leaf (winner_lcp >',
'    spine_lcp). The aggregate edge cat9-E5 accept/event == GROSS leaf-saves/event EXACTLY.',
'  - LOSSLESS vs LOSSY: a leaf save is LOSSLESS iff the leaf token == the recurrent-oracle greedy at that served',
'    position; else it is a FLIP = a LOSSY save (the same leaf-fork the flip analysis tracks).',
'  GATE METRIC = net = lossless_leaf_saves - lossy_leaf_saves - spine_regressions (per-depth + total); PASS =',
'  net>0 AND spine_regressions==0. This UNIFIES speed (gross saves) + lossless (the flips) on the SAME leaves.',
'',
'BANKED DATA (the cat9 arm = its OWN spine_lcp IS the E5-equivalent arm, no separate E5 boot needed):',
'  output/fr13_fork_margin_probe/logs/fr13_fork_margin_dump.jsonl (per event: spine_path/spine_lcp, best_path/',
'  winner_lcp, best_leaf vs spine_leaf, split_pos/split_node, committed_row, bonus_source, margins) +',
'  output/fr13_fork_margin_probe/logs/rescore_cat9_K1_forkmargin.json (recurrent-oracle oracle_argmax_id +',
'  clear_margin flips per served position) + the JOIN in scripts/fr13_fork_margin_classify.py (served position',
'  -> dump event + depth, flat_offsets [27,111,246,382]/head_skip=1, fail-loud if a position misses a dump step).',
'  DATA CAVEAT (FR13_DIRECTIONAL_ASYMMETRY): the fork_margin OFF set is per_prompt [4,6,7,6] (DIFFERENT boot than',
'  the scan_align off_recur [5,4,5,9]); use the fork_margin set (it has the dump<->oracle join). 23 clear-margin',
'  flips, 466 positions. The flips are near-tie #1/#2 (23/23 drafted, CONFIRMED) so the lossy_leaf_saves are the',
'  near-tie leaf-forks. Banked hint (INFERRED, COMPUTE it exactly): ~35 gross saves, <=20 clear-margin flips.',
'',
'YOUR JOB:',
'PHASE 1 (BuildRun, CPU): write scripts/fr13_perevent_superset_gate.py per the plan: load the dump + the rescore',
'  + reuse the classify JOIN to map each served position -> its dump event + DEPTH d; for each event tally the',
'  leaf-saves (winner_lcp - spine_lcp positions on the fork path beyond spine_lcp) and classify each saved',
'  position LOSSLESS (committed leaf token == oracle_argmax_id at that served pos) vs LOSSY (!= = a clear-margin',
'  flip); also count spine_regressions (winner_lcp < spine_lcp = should be 0). Emit per-depth d=0..4 + total:',
'  {spine_regressions, gross_leaf_saves, lossless_leaf_saves, lossy_leaf_saves, net}. NON-VACUITY (#9): the JOIN',
'  fail-loud (every counted position lands on a real dump event); the oracle rescore is the recurrent path',
'  (RECURRENT_PATH_ENGAGED in the rescore json, not streamed logprobs); assert n_positions matches the rescore',
'  denominator; label any uncomputable quantity. Run it; commit the reducer (pathspec) + the output json.',
'PHASE 2 (Verdict). Report the tally + PASS/FAIL (net>0 & spine_regressions==0). RECONCILE with the aggregate:',
'  gross_leaf_saves/event vs the banked +0.12 accept/event edge (should match); the LOSSLESS FRACTION = lossless',
'  /(gross) = how much of the +0.12 is a genuine lossless superset gain vs a lossy flip-gain. VERDICT: net>0 =>',
'  cat9 IS a lossless superset of E5 by this gate (the +0.12 is mostly lossless); net<=0 => the loose aggregate',
'  superset hid a LOSSY gain (the leaves give speed but flip). Per-depth: is the lossy fraction d0-concentrated?',
'  CAVEAT honestly: small-sample (23 flips / ~35 saves, one boot, #12); this is a BANKED-DATA estimate - the',
'  big-denom run (if it co-armed the dump) or a fresh dump boot would tighten it. NO bake/ship decision (user',
'  call). Quote FR13_BUG_CLASS_PLAYBOOK (#9 vacuous, #12 cross-trajectory/small-sample).',
].join('\n');

phase('BuildRun');
const BR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['reducerWritten','joinNonVacuous','oracleRecurrent','perDepthTally','spine_regressions','gross_leaf_saves','lossless_leaf_saves','lossy_leaf_saves','net','committed','ok','notes'],
  properties: {
    reducerWritten: { type: 'string', description: 'scripts/fr13_perevent_superset_gate.py: load dump+rescore+JOIN, per-depth tally' },
    joinNonVacuous: { type: ['boolean','null'], description: 'the JOIN fail-loud: every counted position lands on a real dump event?' },
    oracleRecurrent: { type: ['boolean','null'], description: 'the rescore greedy is the RECURRENT decode oracle (engaged), not streamed logprobs?' },
    perDepthTally: { type: 'string', description: 'per-depth d=0..4 {spine_reg, gross, lossless, lossy, net}' },
    spine_regressions: { type: ['integer','null'], description: 'should be 0 (structural)' },
    gross_leaf_saves: { type: ['integer','null'] },
    lossless_leaf_saves: { type: ['integer','null'] },
    lossy_leaf_saves: { type: ['integer','null'], description: '= the clear-margin leaf-fork flips' },
    net: { type: ['integer','null'], description: 'lossless - lossy - spine_reg' },
    committed: { type: 'string' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const br = await agent(
  CTX + '\n\nTASK (BuildRun, CPU). Write + run scripts/fr13_perevent_superset_gate.py on the banked dump+rescore, '
  + 'per-depth tally, PROVE JOIN non-vacuous + oracle recurrent. Commit pathspec. Return the schema.',
  { label: 'superset-gate-buildrun', phase: 'BuildRun', schema: BR_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','net_verdict','aggregateReconciliation','losslessFraction','perDepth','shipReadiness','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'JOIN fail-loud + oracle recurrent + denominator matches all proven?' },
    net_verdict: { type: 'string', description: 'net = lossless - lossy - spine_reg; PASS (net>0 & spine_reg==0) or FAIL?' },
    aggregateReconciliation: { type: 'string', description: 'gross_leaf_saves/event vs the +0.12 aggregate edge (match?)' },
    losslessFraction: { type: 'string', description: 'lossless/gross = how much of the +0.12 is a genuine lossless superset gain vs lossy flip-gain' },
    perDepth: { type: 'string', description: 'is the lossy fraction d0-concentrated?' },
    shipReadiness: { type: 'string', description: 'banked-data estimate verdict (cat9 lossless-superset-of-E5 by this gate?) + the caveat it is small-sample, tighten with the big-denom/fresh-dump. For the user. No ship decision.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(br) + '. Default holds=false if the JOIN is not fail-loud '
  + '(a counted position not on a real dump event = vacuous), the greedy reference is not the recurrent oracle, '
  + 'spine_regressions was not actually measured (just asserted 0), or the lossless-vs-lossy split is not from '
  + 'the actual committed-token-vs-oracle-argmax compare. Conclude honestly: net>0 (lossless superset) or net<=0 '
  + '(lossy gain), with the small-sample caveat. No bake/ship decision; no reward-hack.',
  { label: 'verify-superset-gate', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { br, v };
