export const meta = {
  name: 'fr13-math-expert-native3',
  description: 'USER (2026-06-15): a MATH-EXPERT workflow w/ little Python experiments on the PREVIOUS drift data + a non-trivial route to match native drift = 3. RE-AIMED after wsvy4vn5k (verify HOLDS): the 17-flip cat9-spine excess vs native-3 is NOT diffuse numerical amplification (that 1.166x/layer compounding is SHARED with E5) - it is CO-RESIDENCY M-dependence. EXISTENCE PROOF: chain5 = cat9 EXACT two kernels (forked-FA2 + GDN tree-scan) on the 5-spine ALONE (M=5) de-cascades to 2 flips AT-OR-BELOW native 3; add 4 branch rows into the SAME batched forward (M=10) and the spine drifts to ~17 (2fe2c567: 11/11 ch2 flips ON spine, 0 leaves = SPINE_PERTURBATION). So the non-trivial route to native-3 = make the SPINE ROWS M-INVARIANT (compute bit-identically regardless of co-resident branch count), reach 17->~5, residual 5->3 = chunk-vs-recurrent intrinsic that already de-cascades to 2 + superset-passes. This CPU workflow DESIGNS that fix + pre-stages the single GPU worker: (1) read the two named seams precisely (GDN deep-accept state-feed / conv prior-window / state-bank column geometry fr10_phase4_patch_vllm_tree_gdn.py:797-818 = FIRST = the L0 first-nonzero carrier; forked-FA2 query-tile row-offset M-dependence = SECOND), (2) a LITTLE NUMPY EXPERIMENT that REPRODUCES (or refutes) the M-dependence of the spine row indexing, (3) DESIGN the precise M-invariant fix keyed to the spine PATH not co-resident M, (4) PREDICT the reach from the banked drift ladders (margin race), (5) emit the exact trajectory-fixed GPU A/B (default-OFF flag, hold trajectory to dodge the scan-recompute 23->32 confound) ready to run when the serve frees. CPU read-only on served code (a big-denom serve runs) + write a numpy sim + a design doc; NO GPU; online numerical/codegen-identity lit; adversarial verify. Output FR13_MATH_EXPERT_NATIVE3.md.',
  phases: [
    { title: 'DesignFix' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN + 16 full-attn). Repo',
'/home/mark/shared/lumoFlyWheel. A big-denom GPU serve runs concurrently - so: CPU-ONLY, NO GPU boot, and do',
'NOT modify the LIVE served files (scripts/fr10_phase4_patch_vllm_tree_gdn.py etc.) - emit the fix as a precise',
'DESIGN + a ready-to-apply diff INSIDE the doc, the GPU worker applies it later behind a default-OFF flag. You',
'MAY write a NEW numpy sim script (scripts/fr13_minvariance_*.py) + FR13_MATH_EXPERT_NATIVE3.md. Pathspec commit',
'(git commit -m ".." -- <files>, -m BEFORE --, NEVER git add -A). Search online for codegen/batch-composition',
'realization identity (reduction grouping, tile occupancy, CUTLASS M-tiling, SSM state-bank indexing).',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e), NEVER a /tmp cache. int-view',
'NEVER atol. Compare target = the deployment RECURRENT decode oracle (fr13_recurrent_decode_oracle), the model\'s',
'trained/eval realization; drift from it is a QUALITY issue. native E5 = 3 = the BAR (a 3-flip realization at',
'this model/fp8 EXISTS = reachable); chain5 = 2 = OUR kernels already hit the floor at M=5.',
'',
'THE CONFIRMED FINDING (wsvy4vn5k, verify HOLDS, re-verified - build ON it, do NOT redo it): cat9-spine drifts',
'~17 clear-margin flips vs native decode; E5-spine ~3. The ~14 EXCESS is CO-RESIDENCY M-dependence, NOT cat9\'s',
'two extra kernels and NOT harder tree-math. EXISTENCE PROOF chain5 (cat9 EXACT two kernels - forked-FA2 tree-',
'bias + GDN tree-scan - on the 5-spine ALONE, M=5) = 2 de-cascaded flips <= native 3 (FR13_PLUS2_DECASCADE).',
'The two extra kernels in isolation are at the floor: forked-FA2 = SAME FA2 CUTLASS kernel, ancestor bias adds',
'0.0 on the spine = byte-identical to FLASH (2-ULP/983k, no depth growth); GDN tree-scan = scan_out 1e-6 / state',
'rel-err 2.2e-4 (K1). The drift is BORN at L0 GDN compute then rides the SHARED 1.166x/layer residual-stream',
'amplification (E5 has the SAME amplification - so amplification-reduction does NOT close the cat9-vs-E5 gap;',
'that was the prior framing, now superseded). The gap is cat9\'s BIGGER L0 birth-amplitude from co-residency.',
'2fe2c567: 11/11 channel-2 clear-margin flips land ON THE SPINE, 0 on leaves = SPINE_PERTURBATION.',
'',
'TWO NAMED SEAMS (wsvy4vn5k leverVerdict): (1) GDN deep-accept state-feed at num_accepted>1 = the FIRST-nonzero',
'L0-GDN co-residency carrier = HIGHEST VALUE, FIX FIRST (prime suspect conv1d prior-window / state-bank column',
'geometry, fr10_phase4_patch_vllm_tree_gdn.py:797-818, see project_fr13_conv_priorwindow_root = pos8/call2/L0',
'conv1d_out diverges 18.375 wrong bank-row/cols at num_accepted>1 while h0_state_in is BYTE-EXACT); (2) forked-',
'FA2 query-tile M-dependence (apply_tree_bias row offset m_block*kBlockM+(tidx/32)*16+(tidx%32)/4 is M-dependent,',
'scripts/fr13_patch_fa2_tree_bias.py L26-78) = SECOND (QPAD direction, e2e-null says fix AFTER the GDN seam).',
'',
'HONEST CAVEAT (must carry): the per-NODE scan-state recompute (FR13_SCAN_NOT_E2E_CARRIER_BIND) was BYTE-EXACT',
'yet e2e flips ROSE 23->32 via TRAJECTORY CHANGE - so any M-invariance fix MUST be validated TRAJECTORY-FIXED',
'(served stream byte-identical to fix-OFF on accepted-spine positions), else a trajectory confound masquerades.',
'',
'YOUR JOB (math-expert, design + little-numpy-experiment, pre-stage the GPU worker):',
'1. CHARACTERIZE the M-dependence PRECISELY (code-read): at the GDN seam (fr10_phase4_patch_vllm_tree_gdn.py',
'   :797-818 + the native conv1d_update / chunk-scan state-bank geometry via vllm_src.sh), WHY does the deep-',
'   spine row\'s conv1d prior-window / scan state-feed read DIFFERENT bank rows/columns at M=10 (spine+4 branches)',
'   vs M=5 (spine-only) vs M=1 (decode)? Pin the exact index expression that depends on co-resident M (the',
'   num_accepted>1 deep-accept path, the prior-window column, the state-bank row mapping). Same for the forked-',
'   FA2 row-offset.',
'2. LITTLE NUMPY EXPERIMENT (the user\'s "little python experiment"): write scripts/fr13_minvariance_indexing.py',
'   that MODELS the conv1d prior-window / state-bank indexing as a function of M and DEMONSTRATES the spine row',
'   gets a different input window at M=10 vs M=5 (reproduce the 18.375 conv1d_out divergence mechanism in index',
'   space), OR REFUTES it (the indexing is M-invariant => the carrier is elsewhere, revert to the reshape lever',
'   project_fr13_tree_reshape_unifying_lever). RUN it, report the numbers. Non-vacuity: a neg-control where',
'   M=5==M=5 gives identical indices.',
'3. DESIGN the M-INVARIANT FIX: the precise index/geometry change keyed to the spine PATH-to-root (not the co-',
'   resident M) so the deep-spine row\'s conv prior-window + state-bank columns are bit-identical at M=10/5/1.',
'   Ready-to-apply diff in the doc (default-OFF flag, byte-identical when OFF). GDN seam first, forked-FA2 second.',
'4. PREDICT THE REACH from the BANKED drift data (the "previous drift data" + math): using the node5/node7 per-',
'   layer ladders + the L60/61 margin race, if the L0 birth-amplitude drops to the chain5 level (M-invariant),',
'   how many of the 17 fall below their argmax margin -> ~5? Quantify (margin distribution of the 17 vs the L0',
'   amplitude reduction x 1.166^63). Honest if the prediction is <17->5.',
'5. EMIT THE GPU A/B (pre-staged for the worker): the exact trajectory-fixed in-process M=10-vs-M=5-vs-M=1 spine',
'   sub-op A/B (wsvy4vn5k §4): capture pre_conv->conv1d_out->scan_out->gate_out->o_proj_out for the deep-spine',
'   row at M=10/5/1 on IDENTICAL captured input (first-nonzero sub-op vs M = the carrier); apply the fix; confirm',
'   spine sub-op -> 0.0 across M; re-score cat9 vs fr13_recurrent_decode_oracle TRAJECTORY-FIXED (byte-identical',
'   served stream on accepted-spine positions) + count clear-margin spine-flips. Predict 17->~5, leaves unchanged',
'   (+15 superset holds), accept/event >=3.198. The default-OFF flag name + the launcher env passthrough.',
'',
'DELIVERABLE: FR13_MATH_EXPERT_NATIVE3.md = the M-dependence characterization (code-cited), the numpy experiment',
'(reproduced-or-refuted, with run numbers), the M-invariant fix design (ready-to-apply diff, GDN-first), the',
'reach prediction from the drift data (17->~5, math), the pre-staged trajectory-fixed GPU A/B. + the numpy sim',
'script. Commit pathspec. Distinguish CODE-READ/SIMULATED from INFERRED. CONSTRAINTS: keep cat9 leaves (superset',
'+15); no copy/HBM/dense; NOT K1/N_PAD/WY/bonus (done/parked/rejected); the fix is M-INVARIANCE (spine computes',
'the same regardless of co-resident branches), NOT a numerical correction-term. If the numpy refutes M-dependence',
'-> say so + point to the reshape lever. Quote FR13_BUG_CLASS_PLAYBOOK (#10 codegen-identity, #12 co-residency,',
'#9 non-vacuous). research-before-deadend.',
].join('\n');

phase('DesignFix');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['mDependenceCharacterized','numpyExperiment','mInvariantFixDesign','reachPrediction','gpuAbPrestaged','committed','notes'],
  properties: {
    mDependenceCharacterized: { type: 'string', description: 'code-read: the EXACT index expression (conv prior-window / state-bank row+col / deep-accept state-feed) that depends on co-resident M at the GDN seam (797-818) + the forked-FA2 row-offset; why the deep-spine row differs at M=10 vs M=5 vs M=1' },
    numpyExperiment: { type: 'string', description: 'scripts/fr13_minvariance_indexing.py: models the indexing vs M, REPRODUCES the spine-row-input-differs-at-M=10-vs-M=5 (or REFUTES it), with ACTUAL run numbers + a neg-control (M=5==M=5 identical)' },
    mInvariantFixDesign: { type: 'string', description: 'the precise index/geometry change keyed to the spine PATH not co-resident M, making the deep-spine conv-window+state-bank bit-identical across M; ready-to-apply diff (default-OFF flag); GDN-first then forked-FA2' },
    reachPrediction: { type: 'string', description: 'from the banked node5/node7 ladders + the L60/61 margin race: how many of the 17 drop below margin if the L0 birth-amplitude is M-invariant -> ~5? quantified, honest' },
    gpuAbPrestaged: { type: 'string', description: 'the exact trajectory-fixed M=10/5/1 spine sub-op A/B + re-score vs recurrent oracle (byte-identical accepted-spine), default-OFF flag name + launcher passthrough, predicted 17->~5 leaves-unchanged' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  CTX + '\n\nTASK (DesignFix, CPU + numpy on BANKED data/indexing, NO GPU, do NOT edit live served files). '
  + 'Characterize the M-dependence, RUN the numpy indexing experiment, design the M-invariant fix (diff in doc), '
  + 'predict the reach from the drift data, emit the pre-staged trajectory-fixed GPU A/B. Write FR13_MATH_EXPERT_'
  + 'NATIVE3.md + the numpy sim, commit pathspec. Return the schema.',
  { label: 'math-expert-native3', phase: 'DesignFix', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','mDependenceGrounded','numpyReal','fixConcrete','reachHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    mDependenceGrounded: { type: 'string', description: 'is the M-dependence index expression CODE-READ from the actual seam (797-818 + native conv/scan via vllm_src.sh + fr13_patch_fa2_tree_bias.py), not narrative? does it match project_fr13_conv_priorwindow_root (18.375 conv1d_out, h0 byte-exact)?' },
    numpyReal: { type: 'string', description: 'was the numpy indexing experiment actually RUN (spot-check the M=10-vs-M=5 index divergence + the M=5==M=5 neg-control), and does it genuinely reproduce-or-refute (not a toy that assumes the answer)?' },
    fixConcrete: { type: 'string', description: 'is the M-invariant fix a precise index/geometry change keyed to the spine path (a real applyable diff), default-OFF, NOT a copy/dense/correction-term/reward-hack?' },
    reachHonest: { type: 'string', description: 'is the 17->~5 reach grounded in the banked ladder margins (not asserted), and does it carry the trajectory-fixed-validation caveat (scan-recompute rose 23->32)?' },
    recommendation: { type: 'string', description: 'single: is the M-invariant fix sound + the GPU A/B ready to run when the serve frees, or did the numpy refute M-dependence (-> reshape lever). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if the M-dependence is narrative '
  + 'not code-read from the actual seam (must cite 797-818 + native conv/scan + the FA2 row-offset, match the '
  + '18.375/h0-byte-exact bind), the numpy experiment was not actually run or is a toy that assumes the answer '
  + '(spot-check the index numbers + the neg-control), the fix is a correction-term/copy/dense rather than true '
  + 'M-invariance keyed to the spine path, or the reach is asserted not margin-grounded / drops the trajectory-'
  + 'fixed caveat. research-before-deadend (chain5=2 proves reachable). No close/pass-fail; no reward-hack.',
  { label: 'verify-math-expert-native3', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { m, v };
