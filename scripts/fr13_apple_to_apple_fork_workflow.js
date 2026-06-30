export const meta = {
  name: 'fr13-apple-to-apple-fork',
  description: 'USER (2026-06-15) REJECTED the spine-bonus/margin-damp ("would not agree to such basket bonus"). The real question: WHY does a leaf WIN the LCP comparison when (per the decode oracle) it should LOSE — is the win/lose comparison APPLE-TO-APPLE, or is there a wiring/row/context bug? The committer accepts a leaf node when drafts[node]==parent_targets[node] (parent_targets=argmax of the VERIFY forward). A fork = leaf-LCP > spine-LCP because the verify argmax at the branch point == the leaf token, but the DECODE oracle argmax there differs (the flip). SUSPICION: the 13 CONFIDENT forks (deciding verify top1-top2 margin 1.25..9.125 nat, heavy 7-9nat tail) disagree with decode by FAR MORE than the diffuse verify-vs-decode realization gap (~3 logits) — too large to be that gap → smells like the leaf is matched against the WRONG verify ROW/position (off-by-one / co-residency context mismatch / FR13_COMMIT_ARGMAX_GATE-class row-mapping bug) = NOT apple-to-apple. INVESTIGATE (do NOT implement the bonus): is parent_targets[deciding-node] computed from the CORRECT verify row that predicts that exact position in the leaf path, consistently for leaf AND spine; and when the leaf wins, would it STILL win against the apple-to-apple reference (the same row the decode oracle uses), or does it LOSE (= a fixable wiring/comparison bug, lossless, no bonus)? CPU-first from the banked dump + rescore + committer code; specify a minimal GPU re-derive only if needed. Adversarial verify. Output FR13_APPLE_TO_APPLE_FORK.md.',
  phases: [
    { title: 'Investigate' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY,',
'READ-ONLY (a literature workflow runs concurrently; do NOT edit code/boot). Read our committer + the BANKED',
'probe dump + rescore + vLLM source via scripts/vllm_src.sh. Write ONLY FR13_APPLE_TO_APPLE_FORK.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS.',
'',
'USER DIRECTIVE (this is the task, verbatim intent): "I would not agree to such basket bonus. What we should',
'check is WHY some leaf wins but they actually should LOSE — is it apple-to-apple compare the win and lose, or',
'why." So: do NOT propose/implement a spine-bonus or margin-damp. INVESTIGATE the comparison itself.',
'',
'WHAT IS BANKED (all verify HOLDS this session):',
'- The committer = top-down LCP _lumo_tree_path_lcp_max_greedy_sample (the IN-CONTAINER committer is the STRING',
'  helper block in scripts/fr10_phase4_patch_vllm_tree_gdn.py, ~L6818-6976 in the module copy / the served',
'  triple-quoted string helper ~L6050-8516): for each root-to-leaf path lcp=longest prefix where drafts[node]==',
'  parent_targets[node]; parent_targets[node]=argmax of the VERIFY forward predicting that node; best_path=max-',
'  lcp, tie-break earliest-leaf (=spine on EQUAL lcp); commits drafts[node] for the lcp prefix + a bonus token',
'  (parent_targets[reject node] / self_target). There is prior history of committer ROW-MAPPING bugs',
'  (FR13_COMMIT_ARGMAX_GATE: "committed_token vs argmax(verify_logits[committed_row]) — a mismatch => the',
'  committer served a non-argmax token for that row, draft/bonus/off-by-one row bug").',
'- The fork-margin probe (FR13_FORK_MARGIN_PROBE_BIND, verify HOLDS) classified 23 clear-margin forks by the',
'  deciding-node verify top1-top2 margin: 10 sub-1nat (B) + 13 confident >=1nat (A, tail 7.125/8.5/9.125). The',
'  BANKED DUMP output/fr13_fork_margin_probe/logs/fr13_fork_margin_dump.jsonl has per-spec-step, per-node:',
'  drafts, parent_targets, the per-path lcp+nodes, the winner/spine lcp-divergence nodes + their VERIFY top-2',
'  logprobs + parent_target id + draft id, the topology split node, best_path/best_leaf/spine_path/spine_leaf,',
'  committed_row. The rescore output/fr13_fork_margin_probe/logs/rescore_cat9_K1_forkmargin.json has the',
'  recurrent-DECODE-oracle argmax at each served position (the apple-to-apple "what it should be").',
'- The verify forward computes one logit row per tree node; parent_targets must come from the row that PREDICTS',
'  each node (the parent\'s row for a child), in the co-resident tree-batched forward.',
'',
'YOUR JOB - is the leaf win/lose comparison APPLE-TO-APPLE; if not, where is the bug:',
'1. WIRING / ROW-MAPPING (the prime suspect for the CONFIDENT forks): read the committer code path that builds',
'   parent_targets from the verify logits + the row indexing (which logit row feeds each node\'s parent_target;',
'   how leaf nodes vs spine nodes are indexed; the bonus-row index). Is parent_targets[node] ALWAYS the argmax',
'   of the row that predicts THAT node\'s position along ITS path — consistently for leaf AND spine? Check for',
'   off-by-one / wrong-parent / position-vs-row mismatch / a leaf node reading a row that belongs to a different',
'   tree position. Cross-check with FR13_COMMIT_ARGMAX_GATE findings. A 7-9 nat "confident match" that disagrees',
'   with decode by 7-9 nat is the signature of matching against a CONFIDENTLY-WRONG (misindexed) row.',
'2. APPLE-TO-APPLE TEST from the banked data (no GPU if possible): for EACH of the 23 forks (esp. the 13',
'   confident A), join the dump deciding-node to the recurrent-decode-oracle argmax at the SAME served position',
'   (rescore json). Compare: the committer matched the leaf draft against parent_target = VERIFY argmax; the',
'   decode oracle argmax is the apple-to-apple reference. Is the verify argmax (a) the SAME token as decode',
'   (then no wiring issue, the flip is elsewhere), (b) a DIFFERENT token by a small (~floor) margin (co-',
'   residency-perturbed near-tie), or (c) a DIFFERENT token by a LARGE margin (7-9nat) AND that token is the',
'   verify argmax for a DIFFERENT position than the one served (= a ROW/position mismatch = WIRING BUG)? For the',
'   confident forks specifically, determine whether the leaf draft == the verify argmax of the CORRECT row, or',
'   whether the matched row predicts a different position (the leaf "wins" against a row it should not be',
'   compared to). This is the literal "apple-to-apple compare the win and lose" the user asked for.',
'3. CLASSIFY each fork: (W) WIRING/row-mismatch (leaf matched a wrong-position verify row -> should LOSE -> fix',
'   the comparison, lossless, NO bonus), (C) CO-RESIDENCY-perturbed parent_target (right row but tree-context',
'   argmax differs from isolated/decode -> apple-to-apple fix = make the verify row co-residency-invariant), (R)',
'   GENUINE verify-vs-decode realization gap (right row, isolated verify also prefers the leaf, decode genuinely',
'   differs -> diffuse, hard). Quantify W vs C vs R over the 23 (esp. the 13 confident).',
'4. VERDICT + cheap test: if W-heavy (a row/comparison bug) -> name the exact mis-indexed line + the fix (the',
'   leaf correctly loses, flips drop losslessly, NO bonus). If C-heavy -> the apple-to-apple fix is verify-row',
'   isolation (relate to the no-copy options, NOT WY which is parked). If R-heavy -> the confident forks are',
'   genuine (the accept-vs-lossless tension stands). Specify the MINIMAL GPU re-derive ONLY if the banked dump',
'   cannot disambiguate W vs C vs R (e.g. need the isolated native-on-path verify logits at the deciding nodes).',
'',
'DELIVERABLE: FR13_APPLE_TO_APPLE_FORK.md = the committer parent_target row-wiring read (cited lines), the per-',
'fork W/C/R classification from the banked dump+rescore (esp. the 13 confident), the verdict (is the comparison',
'apple-to-apple; if not, the exact bug + lossless fix with NO bonus), and the minimal GPU re-derive if needed.',
'Distinguish MEASURED (from the dump/rescore/code) from INFERRED. Do NOT propose the spine-bonus/margin-damp',
'(user rejected). WY is PARKED (do not propose). Reward-hacks banned. Quote FR13_BUG_CLASS_PLAYBOOK (#10 codegen/',
'row-identity, #11 naming-slip, #12 trajectory). Commit pathspec.',
].join('\n');

phase('Investigate');
const I_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['parentTargetRowWiring','appleToAppleTest','forkClassification_W_C_R','verdict','minimalGpuRederiveIfNeeded','committed','notes'],
  properties: {
    parentTargetRowWiring: { type: 'string', description: 'CODE-READ: how parent_targets[node] is built from verify logit rows; is each node matched against the row that predicts ITS position along ITS path, consistently for leaf+spine; any off-by-one/wrong-row; cited lines' },
    appleToAppleTest: { type: 'string', description: 'per-fork join dump deciding-node vs recurrent-decode-oracle argmax at the same served position: is the verify-matched token the same as decode / small-margin-diff / large-margin-diff-against-a-different-position. esp the 13 confident' },
    forkClassification_W_C_R: { type: 'string', description: 'W (wiring/row-mismatch, should LOSE, lossless fix) vs C (co-residency-perturbed right-row) vs R (genuine realization gap) counts over 23 (and over the 13 confident), from the banked data' },
    verdict: { type: 'string', description: 'is the leaf win/lose comparison apple-to-apple? if not, the EXACT bug + the lossless fix (NO bonus). if genuine, say so honestly' },
    minimalGpuRederiveIfNeeded: { type: 'string', description: 'the EXACT minimal GPU re-derive (isolated native-on-path verify logits at the deciding nodes) ONLY if the banked dump cannot disambiguate W/C/R' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const i = await agent(
  CTX + '\n\nTASK (Investigate, CPU read-only). Read the committer parent_target row-wiring + classify the 23 '
  + 'forks W/C/R from the banked dump+rescore. Write FR13_APPLE_TO_APPLE_FORK.md, commit pathspec. Return the '
  + 'schema.',
  { label: 'apple-to-apple-fork', phase: 'Investigate', schema: I_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','wiringReadGrounded','classificationFromData','verdictSound','noBonus','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    wiringReadGrounded: { type: 'string', description: 'is the parent_target row-wiring read from ACTUAL committer code (cited lines), not narrative?' },
    classificationFromData: { type: 'string', description: 'is the W/C/R split re-derived from the ACTUAL dump+rescore (spot-check 2-3 confident forks: which row, which position, decode argmax), not asserted?' },
    verdictSound: { type: 'string', description: 'is the apple-to-apple verdict sound — if a wiring bug is claimed, is the mis-indexed line real + would the fix make the leaf correctly lose losslessly?' },
    noBonus: { type: 'string', description: 'did it AVOID proposing the rejected spine-bonus/margin-damp + WY (parked)?' },
    recommendation: { type: 'string', description: 'single: is there an apple-to-apple wiring/comparison bug to fix (lossless, no bonus) or are the confident forks genuine. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(i) + '. Default holds=false if the row-wiring is narrative '
  + 'not code-read (cite lines), the W/C/R split is asserted not re-derived from the dump+rescore (spot-check '
  + 'confident forks), a claimed wiring bug is not a real mis-indexed line, it proposes the rejected spine-bonus '
  + 'or WY, or it conflates a genuine realization gap with a wiring bug. No close/pass-fail; no reward-hack.',
  { label: 'verify-apple-to-apple', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { i, v };
