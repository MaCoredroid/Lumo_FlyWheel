export const meta = {
  name: 'fr13-perevent-superset-gate-design',
  description: 'USER (2026-06-15): replace the LOOSE aggregate "superset" (cat9 3.198 vs E5 3.076 accept/event = cross-trajectory #12) with a PRECISE PER-EVENT superset gate: where E5 accepts the spine, cat9 must also accept the spine (NO spine regression); where E5 CUTS the spine, cat9 saves a token on a LEAF (the gain). True superset = cat9 accepted-set >= E5 accepted-set at EVERY event. KEY UNIFICATION (speed+lossless in one number): a leaf save is GOOD only if (a) E5 cut the spine there AND (b) the leaf token == the greedy decode (lossless); a leaf save that is a FLIP (leaf != greedy) is a LOSSY gain. So the leaves\' saves (speed) and flips (lossless-loss) are the SAME leaves. GATE = net-lossless-leaf-saves = (saves where E5-cut AND leaf==greedy) - (leaf saves that are flips) - (spine regressions); cat9 is a LOSSLESS SUPERSET of E5 iff positive with ZERO spine regressions. = the banked "greedy branch rescue" made the deliverable gate. APPLE-TO-APPLE measurement: teacher-force the SAME greedy-decode reference through BOTH verifiers, per-position compare (E5 spine accept/cut; cat9 spine-accept/leaf-save/both-miss; leaf-save lossless-vs-flip) - NO trajectory divergence. Design the ready-to-run spec + the gate metric. CPU read-only, code + git-history, adversarial verify. Output FR13_PEREVENT_SUPERSET_GATE.md.',
  phases: [
    { title: 'Design' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a big-denominator',
'GPU serve + a directional-asymmetry CPU analysis run concurrently; do NOT edit code/boot). Read our code + GIT',
'HISTORY (how the superset/greedy-branch-rescue + the spec-decode accept trace were measured before). Write ONLY',
'FR13_PEREVENT_SUPERSET_GATE.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol. The',
'lossless/greedy reference = the deployment-correct RECURRENT decode oracle (fr13_recurrent_decode_oracle).',
'',
'THE USER GATE (this is the task): the deliverable speed claim is the DEPTH-MATCHED SUPERSET (cat9 9-node tree >',
'E5 5-spine, FR13_DIRECTION_AND_NUMBERS), but we have been quoting the AGGREGATE accept/event (3.198 vs 3.076) =',
'cross-trajectory, NOT a per-event proof. The PRECISE per-event superset:',
'  - SPINE NON-REGRESSION: at every event, cat9 accepts >= the spine tokens E5 accepts (cat9 never accepts',
'    FEWER spine tokens = no LOSS).',
'  - LEAF SAVE (the gain): where E5 CUTS the spine (the spine draft at depth d != target greedy), cat9 saves a',
'    token via a LEAF at depth d (the #2/branch).',
'  - LOSSLESS GAIN: a leaf save counts as a GOOD (lossless) gain ONLY if the leaf token == the greedy decode',
'    (the recurrent oracle argmax); a leaf save that != greedy is a FLIP = a LOSSY gain (the same leaf-fork the',
'    flip analysis tracks).',
'GATE METRIC = net_lossless_leaf_saves = (#leaf-saves where E5-cut AND leaf==greedy) - (#leaf-saves that are',
'flips) - (#spine regressions). cat9 is a LOSSLESS SUPERSET of E5 iff net > 0 AND spine_regressions == 0. This',
'UNIFIES speed (the saves) + lossless (the flips) on the SAME leaves = the banked GREEDY BRANCH RESCUE',
'(FR13_DIRECTION_AND_NUMBERS: "spine top-1 misses at depth d but the alt top-2 == target argmax -> commit the',
'alt + alt-row bonus -> one served token ahead of native reject+bonus", +35 genuine branch accepts at greedy,',
'8add39e6 acceptance ladder).',
'',
'THE APPLE-TO-APPLE MEASUREMENT (avoid #12 trajectory divergence): teacher-force the SAME greedy-decode',
'reference stream through BOTH verifiers and compare PER POSITION (NOT free-run, which diverges after the first',
'fork). For each reference position: (E5) does the MTP-5 spine draft match the target greedy (accept) or miss',
'(cut)? (cat9) does the spine match (accept), OR where the spine is cut does a leaf draft match the target',
'greedy (lossless save) / a non-greedy token (lossy flip-save), or do both miss? Same reference positions for',
'both arms = a clean per-event domination count.',
'',
'YOUR JOB - design the per-event superset gate + ready-to-run spec:',
'1. THE ACCEPT TRACE: read how to get, PER EVENT, E5\'s spine accept/cut pattern and cat9\'s spine-accept/leaf-',
'   save pattern. The committer dump (FR13_FORK_MARGIN_DUMP / FR13_COMMIT_ARGMAX_GATE, scripts/fr10_phase4_',
'   patch_vllm_tree_gdn.py) gives cat9 per-event (drafts, parent_targets, best_path, best_leaf vs spine_leaf,',
'   bonus_source). For E5 (native MTP-5) read how its spec accept trace is available (vLLM rejection_sampler /',
'   spec_decode metrics / a teacher-force capture). Specify the EXACT capture for BOTH arms on the SAME',
'   reference (greedy oracle stream or the same prompts_swe4/SWE stream).',
'2. THE GATE COMPUTATION: per reference position, classify {E5: spine-accept / spine-cut} x {cat9: spine-accept',
'   / leaf-save-lossless / leaf-save-flip / both-miss}. Tabulate: spine_regressions (E5 accept-spine but cat9',
'   not), lossless_leaf_saves (E5 cut, cat9 leaf==greedy), lossy_leaf_saves (E5 cut, cat9 leaf!=greedy = flip),',
'   net = lossless_leaf_saves - lossy_leaf_saves - spine_regressions. PASS = net>0 AND spine_regressions==0.',
'3. RECONCILE with the aggregate accept/event: show how net_lossless_leaf_saves relates to the +0.12 accept/',
'   event edge (the aggregate edge should = the gross leaf saves per event; the gate strips the lossy fraction).',
'   So a positive aggregate edge that is mostly LOSSY flips would FAIL this gate = the loose superset hid a',
'   lossy gain. Quantify from the banked fork data (the leaf-saves are the forks; how many are lossless).',
'4. RUNNABLE-NOW vs new harness: what existing scripts (the committer dump + the recurrent oracle + the SWE/',
'   prompts capture) give this, vs what needs a new E5-accept-trace capture or a new reducer. Specify the',
'   minimal GPU run (likely reuses the big-denom served streams + a per-event accept capture for both arms).',
'',
'DELIVERABLE: FR13_PEREVENT_SUPERSET_GATE.md = the per-event accept-trace capture (both arms, same reference),',
'the gate metric (spine_regressions / lossless_leaf_saves / lossy_leaf_saves / net) with the PASS condition, the',
'reconciliation with the aggregate +0.12 edge (how much of it is lossless), and the runnable-now-vs-new-harness',
'+ minimal GPU spec. Distinguish MEASURED/CODE-READ from INFERRED. Quote FR13_BUG_CLASS_PLAYBOOK (#12 cross-',
'trajectory, #9 vacuous). Commit pathspec.',
].join('\n');

phase('Design');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['acceptTraceCapture','gateComputation','aggregateReconciliation','runnableNowVsNewHarness','committed','notes'],
  properties: {
    acceptTraceCapture: { type: 'string', description: 'how to get E5 spine accept/cut + cat9 spine-accept/leaf-save per event on the SAME reference (committer dump for cat9; the E5 accept trace; the shared reference stream)' },
    gateComputation: { type: 'string', description: 'the per-position classification + the metric: spine_regressions / lossless_leaf_saves / lossy_leaf_saves / net; PASS = net>0 AND spine_regressions==0' },
    aggregateReconciliation: { type: 'string', description: 'how net_lossless_leaf_saves relates to the aggregate +0.12 accept/event edge; how much of the edge is lossless vs lossy flips, from banked fork data' },
    runnableNowVsNewHarness: { type: 'string', description: 'what existing scripts give this vs new capture/reducer needed; the minimal GPU spec (reuse big-denom streams?)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (Design, no GPU, read-only). Read the committer dump + E5 accept trace + git history. Design '
  + 'the per-event superset gate + ready-to-run spec + the aggregate reconciliation. Write FR13_PEREVENT_'
  + 'SUPERSET_GATE.md, commit pathspec. Return the schema.',
  { label: 'perevent-superset-design', phase: 'Design', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','traceGrounded','gateSound','reconciliationHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    traceGrounded: { type: 'string', description: 'is the accept-trace capture grounded in real scripts (committer dump + E5 trace), apple-to-apple on the same reference (no #12 divergence)?' },
    gateSound: { type: 'string', description: 'is the gate metric sound (net_lossless_leaf_saves, spine_regressions==0) and does it genuinely unify speed+lossless on the same leaves, not double-count?' },
    reconciliationHonest: { type: 'string', description: 'is the aggregate-edge reconciliation honest (the +0.12 could be mostly lossy = a failing superset the loose metric hid)?' },
    recommendation: { type: 'string', description: 'single: is the gate ready to run (and should it replace the aggregate superset as the deliverable gate)? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the accept trace is not '
  + 'grounded in real scripts, the measurement lets the two arms free-run and diverge (#12, not apple-to-apple), '
  + 'the gate metric double-counts or does not strip the lossy leaf-saves, or the aggregate reconciliation '
  + 'hand-waves whether the +0.12 edge is lossless. No close/pass-fail; no reward-hack.',
  { label: 'verify-perevent-superset', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
