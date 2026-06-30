export const meta = {
  name: 'fr13-directional-asymmetry',
  description: 'USER INSIGHT (2026-06-15): if the cat9 drift were RANDOM DIFFUSE noise, the verify-argmax would land on a RANDOM (non-drafted) token => mostly "both-rejected/bonus" flips + a SYMMETRIC direction. But it appears the flips are systematically "ACCEPT LEAF OVER SPINE" (verify-argmax == the specific #2/leaf token, a near-tie with #1/spine), NOT symmetric, NOT both-rejected. That asymmetry would indicate a SYSTEMATIC directional bias (the tree-verify forward favoring the #2/leaf), NOT diffuse noise -> potentially FIXABLE -> would contradict "diffuse, relax". TEST IT from the banked fork data: categorize EVERY clear-margin flip by (a) what was committed (accept-leaf / accept-spine-but-wrong / both-rejected-bonus / non-drafted), (b) the DIRECTION (verify favored #2-leaf vs #1-spine vs third), measured SYMMETRICALLY (account for the visibility/selection effect: a leaf-fork is loud, the reverse verify=#1/decode=#2 is a quiet spine-realization flip). Verdict: SYSTEMATIC leaf-bias (re-open a lever) vs SYMMETRIC/diffuse (confirms diffuse). CPU read-only, adversarial verify. Output FR13_DIRECTIONAL_ASYMMETRY.md.',
  phases: [
    { title: 'Analyze' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10. Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a big-denominator GPU run',
'runs concurrently; do NOT edit code/boot). Read the BANKED fork data + committer code. Write ONLY',
'FR13_DIRECTIONAL_ASYMMETRY.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol. PROVE',
'non-vacuous (#9): categories re-derived from the ACTUAL banked dump, not asserted; account for cascade (de-',
'cascade FR13_PLUS2) + the selection/visibility effect.',
'',
'THE USER HYPOTHESIS (test it honestly, do not just confirm): the flips are NOT diffuse random noise but a',
'SYSTEMATIC "accept leaf over spine" bias. Reasoning: a flip = committed token (= the VERIFY-forward argmax) !=',
'the DECODE oracle argmax. If RANDOM diffuse: the verify-argmax would scatter to a RANDOM token (almost always',
'NON-DRAFTED, vocab huge) => mostly BOTH-REJECTED/bonus flips + a SYMMETRIC #1-vs-#2 direction. OBSERVED (apple-',
'to-apple): ~16-19 LEAF-FORKS (verify-argmax == the specific #2/leaf token, near-tie with #1) vs ~4 spine-',
'realization; few/no both-rejected. That = the verify-argmax keeps hitting the SPECIFIC leaf token = NEAR-TIE',
'#1-vs-#2, NOT random; AND skewed toward #2/leaf, NOT symmetric. If real + directional => a SYSTEMATIC bias',
'(tree-verify favors #2) => potentially fixable => contradicts "diffuse, relax".',
'',
'THE HONEST CAVEAT to control for: a leaf-fork is LOUD (the committer visibly branches); the REVERSE (verify',
'argmax == #1/spine but decode wanted #2/leaf) is a QUIET "spine-realization" flip (the spine just continued',
'and was wrong). So counting only the loud forks OVER-states the asymmetry. Measure the direction SYMMETRICALLY:',
'at each flip, what did the VERIFY argmax pick (#1-spine / #2-leaf / a non-drafted third) and what did DECODE',
'pick, and is verify-favors-#2 genuinely > verify-favors-#1 AFTER accounting for the selection.',
'',
'THE USER MECHANISM HYPOTHESIS (2026-06-15, test by CODE-READ): IF the asymmetry is real + directional, maybe',
'some MATH/ROUNDING in OUR kernels or committer FAVORS THE LEAF SPECIFICALLY. CODE-READ for a leaf-vs-spine',
'asymmetry: (i) the COMMITTER (_lumo_tree_path_lcp_max_greedy_sample): note the tie-break ALREADY favors the',
'SPINE (earliest-leaf on EQUAL lcp, L6839-6843) and a fork needs leaf_lcp STRICTLY > spine_lcp -- so the',
'committer LOGIC is biased AGAINST the leaf; a leaf-favoring effect must be elsewhere. Check the bonus/self_',
'target path (L6874-6896) + any >= vs > or rounding in the lcp/margin compare. (ii) the VERIFY FORWARD leaf',
'NODES: do leaf rows get a different scale/rounding than spine rows? the forked-FA2 tree-bias (fr13_patch_fa2_',
'tree_bias.py: the additive -inf for non-ancestors -- any asymmetric bias/scale on leaf vs spine rows?), the',
'GDN tree-scan leaf-branch state (fr10_gdn_tree_kernel.py: the spine state is co-residency-INVARIANT per the',
'N_PAD test, but is the LEAF branch state computed with a different rounding/op-order that systematically',
'shifts the leaf parent_target toward the #2?), the fp8 GEMM/o_proj scale per row. (iii) the DRAFTER: is the',
'MTP #2 systematically over-represented as the verify argmax (a drafter-quality not a kernel effect)? Tag each',
'as a real leaf-favoring math/rounding asymmetry (FIXABLE) vs none-found (then the asymmetry, if real, is the',
'verify-vs-decode realization at near-ties, directionally neutral). The COMMITTER-FAVORS-SPINE counter-clue is',
'important: if the committer already favors spine yet the LEAF still wins systematically, the forward bias must',
'be REAL + strong.',
'',
'DATA (banked, MEASURED): output/fr13_fork_margin_probe/logs/fr13_fork_margin_dump.jsonl (per spec-step, per',
'node: drafts, parent_targets=verify argmaxes, best_path/best_leaf/spine_leaf, committed_row, bonus_source) +',
'output/fr13_fork_margin_probe/logs/rescore_cat9_K1_forkmargin.json (DECODE oracle argmax + clear-margin flips,',
'per served position) + output/fr13_scan_align_rerun/logs/{off_recur_flips.json, native_recur_flips.json}',
'(cat9 OFF 23 + native 3 with oracle_topk per flip) + FR13_APPLE_TO_APPLE_FORK.md (the SAME_POS leaf-fork set',
'+ the corrected reducer logic + the 8R/8C/7W split). Committer = scripts/fr10_phase4_patch_vllm_tree_gdn.py',
'_lumo_tree_path_lcp_max_greedy_sample (L6818-6976: bonus_source = reject_parent_target / tree_self_target /',
'root_parent_target distinguishes accept-draft vs bonus).',
'',
'YOUR JOB:',
'1. CATEGORIZE every clear-margin flip (cat9 OFF + the fork-margin dump set) into: (A) ACCEPT-LEAF (committed a',
'   leaf draft = best_leaf!=spine_leaf, verify-argmax == the leaf token), (B) ACCEPT-SPINE-but-wrong (committed',
'   the spine token, verify==spine, decode!=spine = the quiet reverse), (C) BOTH-REJECTED/BONUS (committed a',
'   non-drafted token = bonus_source reject_parent_target/self_target where the committed != any sibling draft),',
'   (D) other. Count each. From the banked oracle_topk: is the served (verify-argmax) token a DRAFTED token',
'   (near-tie #1/#2) or a random non-drafted token?',
'2. DIRECTION symmetrically: for the flips at a branch, classify verify-pick vs decode-pick as {#1-spine,',
'   #2-leaf, third}. Compute P(verify=#2 & decode=#1) vs P(verify=#1 & decode=#2). Is there a genuine skew',
'   toward #2 AFTER controlling the selection (e.g. restrict to positions where BOTH a spine and a leaf draft',
'   exist at that depth, so both directions are observable)? Quantify the asymmetry ratio + whether it survives',
'   the visibility control.',
'3. RANDOM-vs-SYSTEMATIC: would random-diffuse produce this? Estimate the rate of BOTH-REJECTED/bonus flips a',
'   random verify-argmax perturbation would give vs the OBSERVED both-rejected count. If observed both-rejected',
'   << random-diffuse prediction AND accept-leaf dominates at near-ties => the drift is NOT uniform random; it',
'   is concentrated at #1/#2 near-ties (consistent with a small realization gap) AND possibly DIRECTIONAL.',
'4. VERDICT: (SYSTEMATIC directional leaf-bias) if verify-favors-#2 >> verify-favors-#1 survives the visibility',
'   control => a fixable directional bias -> name the candidate mechanism (the leaf co-residency tipping the',
'   parent near-tie toward #2 = the queued isolated-fork test; or a committer near-tie/tie-break asymmetry; or',
'   the drafter #2 being systematically over-confident in the verify forward) + a cheap test. (NEAR-TIE-but-',
'   SYMMETRIC) if the #2/#1 skew vanishes after the visibility control => the flips are near-tie realization',
'   (small gap at #1/#2), directionally symmetric = the diffuse-within-near-ties picture holds (relax). Be',
'   QUANTITATIVE + honest about the selection effect.',
'',
'DELIVERABLE: FR13_DIRECTIONAL_ASYMMETRY.md = the flip category counts (A/B/C/D), the symmetric direction test',
'(verify-#2 vs verify-#1 with the visibility control), the random-vs-systematic estimate, and the verdict',
'(systematic directional bias -> fixable lever, or near-tie-symmetric -> diffuse). Distinguish MEASURED from',
'INFERRED. Quote FR13_BUG_CLASS_PLAYBOOK (#12 selection/trajectory, #9 vacuous). Commit pathspec.',
].join('\n');

phase('Analyze');
const A_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['categoryCounts','directionSymmetric','randomVsSystematic','leafFavoringCodeRead','verdict','candidateMechanismIfSystematic','committed','notes'],
  properties: {
    categoryCounts: { type: 'string', description: 'A accept-leaf / B accept-spine-wrong / C both-rejected-bonus / D other counts, from the banked dump; is the verify-argmax a DRAFTED token (near-tie) or random non-drafted?' },
    directionSymmetric: { type: 'string', description: 'P(verify=#2 & decode=#1) vs P(verify=#1 & decode=#2) WITH the visibility control (positions where both spine+leaf draft exist); asymmetry ratio + does it survive' },
    randomVsSystematic: { type: 'string', description: 'observed both-rejected/bonus count vs what random-diffuse verify-perturbation predicts; is the drift uniform-random or concentrated at #1/#2 near-ties' },
    leafFavoringCodeRead: { type: 'string', description: 'CODE-READ for a leaf-vs-spine math/rounding asymmetry: committer (tie-break favors SPINE; >= vs > in lcp/margin; bonus path), verify-forward leaf rows (FA2 tree-bias scale, GDN leaf-branch state rounding, fp8 per-row scale), drafter MTP#2 over-rep — real leaf-favoring seam FOUND or NONE (cited lines)' },
    verdict: { type: 'string', description: 'SYSTEMATIC directional leaf-bias (fixable) or NEAR-TIE-SYMMETRIC (diffuse), with numbers' },
    candidateMechanismIfSystematic: { type: 'string', description: 'if systematic: the candidate mechanism (leaf-favoring math/rounding from the code-read / leaf co-residency tipping the parent near-tie / drafter #2 over-confidence) + a cheap test' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const a = await agent(
  CTX + '\n\nTASK (Analyze, CPU read-only). Categorize flips + measure the symmetric direction with the visibility '
  + 'control + the random-vs-systematic estimate, from the banked dump. Write FR13_DIRECTIONAL_ASYMMETRY.md, '
  + 'commit pathspec. Return the schema.',
  { label: 'directional-asymmetry', phase: 'Analyze', schema: A_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','countsGrounded','visibilityControlSound','verdictHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    countsGrounded: { type: 'string', description: 'are the A/B/C/D counts re-derived from the ACTUAL dump (spot-check), not asserted?' },
    visibilityControlSound: { type: 'string', description: 'does the direction test genuinely control the selection/visibility effect (the quiet reverse counted), not just count loud forks?' },
    verdictHonest: { type: 'string', description: 'is systematic-vs-diffuse backed by the controlled numbers, not the raw loud-fork skew?' },
    recommendation: { type: 'string', description: 'single: systematic directional bias (a fixable lever to pursue) or near-tie-symmetric (diffuse, relax). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(a) + '. Default holds=false if the category counts are not '
  + 're-derived from the actual dump, the direction test does NOT control the visibility/selection effect (just '
  + 'counts loud leaf-forks = invalid), or the systematic-vs-diffuse verdict rests on the uncontrolled skew. The '
  + 'user hypothesis (systematic leaf-bias) must be tested HONESTLY, not rubber-stamped. No close/pass-fail.',
  { label: 'verify-directional-asymmetry', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { a, v };
