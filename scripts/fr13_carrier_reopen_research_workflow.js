export const meta = {
  name: 'fr13-carrier-reopen-research',
  description: 'RE-OPEN the cat9 21/23-flip carrier hunt (user: cpu read-code/research flow based on the decisive conclusion). The GDN scan state-feed is NON-VACUOUSLY ruled out as the e2e carrier (FR13_SCAN_NOT_E2E_CARRIER_BIND: recompute made scan bit-exact 0.0 to native yet e2e flips ROSE 23→32; native-E5=3 = real 7x gap, NOT irreducible). Read-code + analyze the BANKED flip data to find where the cat9-vs-no-spec-oracle 23-vs-3 gap actually lives, given the strongest per-forward kernel seam (scan) is ruled out. Test the TRAJECTORY-FORK hypothesis vs per-forward-spread, the accept/commit path, the drafter/topology, and reconcile the recompute-worse clue. Adversarial verify. CPU only.',
  phases: [
    { title: 'Research' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (no GPU boot; read',
'banked data + code + vLLM source via scripts/vllm_src.sh; write ONLY FR13_CARRIER_REOPEN.md). Pathspec commits.',
'',
'GROUNDING RULE (user): read vLLM source DIRECTLY from the pinned image via `scripts/vllm_src.sh <relpath>`',
'(0.19.2rc1.dev134), NEVER a /tmp cache. The whole 22→3 journey was re-grounded on the real image',
'(c589cd63, anyConclusionShifted=NONE) so the binds conclusions hold (line#s drifted only).',
'',
'THE DECISIVE EVIDENCE (FR13_SCAN_NOT_E2E_CARRIER_BIND.md, re-run w7wr68z06, ALL instruments proven',
'non-vacuous): (1) STATE gate - OUR deployed scan STATE diverges from native packed-decode STATE by max_abs',
'0.0289 (REAL kernel gap); recompute-from-spine makes it bit-exact int-view 0.0. (2) e2e flips vs the SAME',
'deployment-correct RECURRENT no-spec oracle (fr13_recurrent_decode_oracle, single-step _forward_core_decode_',
'non_spec): native-E5=3 [0,0,2,1] | cat9 OFF=23 [5,4,5,9] | cat9 RECOMPUTE=32 [10,9,7,6] (artifact-checked',
'common-prefix + per-position rate 0.0529→0.0625, the RISE is real). => the GDN SCAN STATE-FEED is NOT the',
'e2e carrier (kernel gap real but NON-CAUSAL); native-E5=3 is the existence proof the 23 is a REAL defect, NOT',
'irreducible; recompute is also NOT byte-lossless (369 tok diffs from OFF). DATA at output/fr13_scan_align_',
'rerun/logs/{off_recur_flips.json, recompute_recur_flips.json, native_recur_flips.json, probe_us_off.json}.',
'',
'YOUR JOB - re-open the carrier hunt with this evidence; the per-forward kernel-seam framing is WEAKENED (scan',
'was the strongest candidate and is ruled out). Find where the 23-vs-3 gap (cat9 tree-verify vs its OWN no-spec',
'recurrent oracle, vs native MTP-5 vs its own = 3) actually lives:',
'1. TRAJECTORY-FORK vs PER-FORWARD-SPREAD (the FIRST discriminator, on banked data, cheap): read off_recur_',
'   flips.json per-prompt flip POSITIONS + the served streams (probe_us_off.json). Are the 23 clear flips',
'   CLUSTERED right after an early divergence point and cascading (= the served stream forks from the no-spec',
'   oracle once, then downstream positions are scored on a diverged prefix = TRAJECTORY-FORK, a measurement/',
'   cascade phenomenon NOT a per-forward kernel carrier), OR are they SPREAD as independent isolated crossings',
'   (= a genuine diffuse per-forward divergence)? Compute the gap structure: first-flip position per prompt,',
'   inter-flip gaps, whether each flip re-converges (dev→0) before the next. This is the SAME de-cascade',
'   discipline as FR13_PLUS2_DECASCADE - apply it to the 23.',
'2. THE RECOMPUTE-WORSE CLUE: recompute REMOVES leaf co-residency (each node replayed from spine independently)',
'   + aligns to native geometry, yet flips ROSE. So the banked "+17 leaf co-residency" decomposition',
'   (FR13_WIDTH_CARRIER_INPROJ_BA, FR13_22flip_carrier_l0gdn) is challenged. Reconcile: does removing',
'   co-residency change the ACCEPTED PATH (the committer accepts different tokens) → a different trajectory →',
'   different (more) flips? Read the committer + accept logic to explain it.',
'3. ACCEPT/COMMIT vs DRAFTER vs TOPOLOGY: the 23-vs-3 gap is OUR tree spec-decode (9-node caterpillar + MTP',
'   drafter + our tree-verify: scan[ruled out] + FA2-fork[lossless 0.0039] + committer) vs native linear MTP-5.',
'   Read the LCP greedy committer (_lumo_tree_path_lcp_max_greedy_sample) + rejection_sampler (vllm_src.sh',
'   v1/sample/rejection_sampler.py) + how native MTP-5 accepts, and the cat9 vs MTP-5 tree topology. For a',
'   LOSSLESS spec-decode the tree-verify must commit the no-spec-greedy token; at the 23 flip positions, WHY',
'   does cat9 commit a different token than its no-spec greedy (verify-logit divergence? committer rule? forked',
'   prefix?). native does this at only 3. What is structurally different.',
'4. SYNTHESIZE the next-best hypothesis for the 23-vs-3 carrier (consistent with scan-ruled-out + recompute-',
'   worse) + a CHEAP test (CPU or a single targeted GPU A/B) to confirm/refute it. Do NOT conclude irreducible',
'   (native-E5=3 disproves it). research-before-deadend.',
'',
'Be SKEPTICAL + quantitative (the session repeatedly overstated single carriers; the scan was the latest',
'overturned). Quote FR13_BUG_CLASS_PLAYBOOK rows (#12 cross-event/co-residency + trajectory, #9 vacuous, #10',
'codegen). Write FR13_CARRIER_REOPEN.md, commit pathspec.'
].join('\n');

phase('Research');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['flipStructure','trajectoryForkVsSpread','recomputeWorseExplained','acceptCommitAnalysis','drafterTopologyDiff','nextHypothesis','cheapTestPlan','committed','notes'],
  properties: {
    flipStructure: { type: 'string', description: 'from off_recur_flips.json: per-prompt first-flip pos, inter-flip gaps, re-convergence; de-cascaded independent-event count of the 23' },
    trajectoryForkVsSpread: { type: 'string', description: 'verdict: are the 23 flips a TRAJECTORY-FORK cascade (early fork + downstream-on-diverged-prefix) or genuine PER-FORWARD spread? with the quantitative evidence' },
    recomputeWorseExplained: { type: 'string', description: 'why recompute (removes co-residency + aligns scan) RAISED flips - the accept-path/trajectory mechanism; what it says about the +17-co-residency decomposition' },
    acceptCommitAnalysis: { type: 'string', description: 'at the flip positions, why cat9 commits != no-spec-greedy (verify-logit / committer rule / forked prefix), vs native MTP-5 accept (grounded in rejection_sampler + the committer)' },
    drafterTopologyDiff: { type: 'string', description: 'structural cat9-tree-spec vs native-MTP-5 differences that could drive 23-vs-3 (drafter, topology, verify)' },
    nextHypothesis: { type: 'string', description: 'the single best next-carrier hypothesis consistent with scan-ruled-out + recompute-worse' },
    cheapTestPlan: { type: 'string', description: 'a CHEAP CPU or single-GPU-A/B test to confirm/refute the next hypothesis (non-vacuous instrument)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  CTX + '\n\nTASK (Research, no GPU, read-only). Do steps 1-4 quantitatively from the banked flip data + code. '
  + 'Write FR13_CARRIER_REOPEN.md, commit pathspec. Return the schema.',
  { label: 'carrier-reopen-research', phase: 'Research', schema: R_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','flipStructureGrounded','forkVsSpreadSound','hypothesisConsistent','testIsNonVacuous','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    flipStructureGrounded: { type: 'string', description: 'is the trajectory-fork-vs-spread verdict grounded in the ACTUAL flip positions (re-derive from off_recur_flips.json), not asserted?' },
    forkVsSpreadSound: { type: 'string', description: 'is the fork-vs-spread call correct + does it match the recompute-worse evidence?' },
    hypothesisConsistent: { type: 'string', description: 'is the next-carrier hypothesis consistent with BOTH scan-ruled-out AND recompute-worse (not a re-run of an already-refuted seam)?' },
    testIsNonVacuous: { type: 'string', description: 'is the proposed cheap test actually non-vacuous (a powered discriminator, not another vacuous instrument)?' },
    recommendation: { type: 'string', description: 'single recommendation for the next front. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(r) + '. Default holds=false if the fork-vs-spread verdict '
  + 'is not re-derived from the actual banked flip positions, the next hypothesis re-runs an already-refuted '
  + 'seam (scan/conv/fp8/FA2 all ruled out) or contradicts the recompute-worse evidence, or the proposed test '
  + 'would be vacuous (the session burned 3 vacuous instruments). No close/pass-fail; no reward-hack.',
  { label: 'verify-carrier-reopen', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, v };
