export const meta = {
  name: 'fr13-isolated-fork-test',
  description: 'ISOLATED FORK RE-DERIVE (user queued 2026-06-15, "should the leaf have lost?"). For each LCP-committer fork, capture the ISOLATED native-on-path verify argmax (run the deciding node\'s path-to-root as a degenerate 1-path/chain tree through the SAME tree-verify kernels = NO co-residency = the SpecInfer Def 4.1 / STree Eq.4-6 branch oracle) and compare THREE argmaxes per fork: (1) the TREE-BATCHED verify argmax the committer used (= the leaf, banked in fr13_fork_margin_dump + big-run); (2) the ISOLATED tree-verify argmax (NEW); (3) the DECODE oracle argmax (banked recurrent oracle). SPLIT: C (co-residency) = isolated==decode != co-resident -> the leaf SHOULD HAVE LOST, co-residency made it win (fixable by verify-row isolation, no-copy); R (genuine realization) = isolated==co-resident != decode -> the tree-verify kernels genuinely disagree with the decode kernels even isolated (diffuse realization gap, not co-residency). The DIRECT co-residency-vs-realization test (the big-denom run gives only the verify-MARGIN proxy). RUN AFTER the big-denominator run frees the GPU. K1 NOT baked. Adversarial verify.',
  phases: [
    { title: 'Prep' },
    { title: 'IsolatedCapture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED - run ONLY when free',
'(after the big-denominator run wzdmyq386). Pre-boot hygiene: source .venv; recover_host_memory(); MemAvailable',
'>=100GiB + docker ps empty. Teardown + recover after. boot ENFORCE_EAGER=1. K1 NOT baked (do NOT enable',
'FR13_SCAN_ALIGN); cat9 LOCKED deployed flags.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol. PROVE',
'non-vacuous (#9): the isolated forward is genuinely 1-path/no-co-residency; argmaxes apple-to-apple at the SAME',
'position; oracle engaged.',
'',
'THE QUESTION (user): for each fork, the committer picked the LEAF because the TREE-BATCHED VERIFY argmax at the',
'deciding node == the leaf draft, but the DECODE oracle argmax differs (the flip). WOULD an ISOLATED (non-co-',
'resident) forward at that node pick the leaf or the decode token? That splits the cause:',
'  - C (co-residency): isolated tree-verify argmax == decode argmax != the co-resident tree-batched pick. The',
'    co-residency in the FULL tree-batched forward perturbed the argmax to the leaf; isolated, it agrees with',
'    decode -> the leaf SHOULD HAVE LOST -> fixable by verify-row isolation (no-copy, NOT WY/parked).',
'  - R (genuine realization gap): isolated tree-verify argmax == the co-resident pick (leaf) != decode. Even',
'    isolated, the tree-verify KERNELS disagree with the decode KERNELS (the diffuse realization gap). The leaf',
'    genuinely matches the tree-verify realization; decode genuinely differs -> not co-residency.',
'CONTEXT (banked): the GDN SCAN spine state is already co-residency-INVARIANT (N_PAD test cat9-spine==chain5-',
'spine 0.0; K1 mechanism: deployed scan already near-native) - so any C here would be a FULL-ATTN / non-scan',
'co-residency, and R would be the diffuse multi-layer (full-attn) realization, NOT the scan. Compare target =',
'the deployment-correct RECURRENT decode oracle (fr13_recurrent_decode_oracle).',
'',
'YOUR JOB:',
'PHASE 1 (Prep, no GPU): gather the fork set + the two banked argmaxes per fork. Prefer the BIG-RUN forks (from',
'  the wzdmyq386 output, the corrected reducer R/C set) if present; else the banked output/fr13_fork_margin_',
'  probe/logs/fr13_fork_margin_dump.jsonl + FR13_APPLE_TO_APPLE_FORK (the 8 R + 8 C SAME_POS forks). For each',
'  fork extract: deciding-node position, its path-to-root (the spine prefix + the leaf branch token), the served/',
'  leaf token id, the TREE-BATCHED verify argmax id (#1, from the dump parent_targets), the DECODE oracle argmax',
'  id (#2, from the rescore). Design the ISOLATED capture: run the deciding node\'s path-to-root as a DEGENERATE',
'  single-path tree (a chain to that node, no siblings) through the tree-verify forward (reuse the chain5/_fr10_',
'  chain choices / a 1-path speculative_token_tree) so there is NO co-residency, and read the verify argmax at',
'  the deciding node (#3). Specify the EXACT minimal capture (which paths, one boot).',
'PHASE 2 (IsolatedCapture, GPU): hygiene + boot cat9 (deployed flags, EAGER), run the isolated 1-path forwards',
'  for the fork deciding nodes (same prompts_swe4 / the big-run prompts, same seed/positions), capture #3 the',
'  isolated tree-verify argmax per fork. NON-VACUITY: the served tree per isolated run has NO siblings at the',
'  deciding depth (1-path, proven via the engine speculative_token_tree / tok-per-draft); the position aligns to',
'  the fork (#1/#2/#3 same node); det. Teardown + recover.',
'PHASE 3 (Verdict). For each fork classify C vs R from the 3-way argmax compare (#1 co-resident, #2 decode, #3',
'  isolated): C = #3==#2 != #1 (leaf should have lost, co-residency); R = #3==#1 != #2 (genuine realization);',
'  also flag any #3 that equals NEITHER (a third realization, note it). Count C vs R over the forks. VERDICT:',
'  C-heavy => the forks are co-residency-caused -> there IS a no-copy verify-row-isolation lever to re-open',
'  (bring to user; the leaves should have lost). R-heavy => the forks are the genuine diffuse tree-verify-vs-',
'  decode realization gap -> no co-residency lever, the residual is fundamental (relax / the big-run SWE-quality',
'  gate is the deployable answer). NO bake/ship/close decision (user call). Reward-hacks BANNED (no copy/dense/',
'  multi-spine/bonus/WY/K1; native = oracle only; the isolated 1-path is OUR kernel, not a splice). Quote FR13_',
'  BUG_CLASS_PLAYBOOK (#9 vacuous, #12 cross-trajectory). [[reference_gdn_tree_branch_oracle_losslessness]].',
].join('\n');

phase('Prep');
const P_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['forkSet','perForkBankedArgmaxes','isolatedCaptureDesign','committed','notes'],
  properties: {
    forkSet: { type: 'string', description: 'the fork set used (big-run forks preferred, else banked 8R+8C) + count' },
    perForkBankedArgmaxes: { type: 'string', description: 'per fork: deciding-node pos, path-to-root, leaf token, #1 tree-batched verify argmax, #2 decode oracle argmax (from banked dump/rescore)' },
    isolatedCaptureDesign: { type: 'string', description: 'the degenerate 1-path tree per fork (no co-residency) to read #3 the isolated tree-verify argmax; one boot' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const p = await agent(
  CTX + '\n\nTASK (Prep, no GPU). Gather the fork set + per-fork #1/#2 + design the isolated 1-path capture. Commit '
  + 'pathspec any helper. Return the schema.',
  { label: 'isolated-fork-prep', phase: 'Prep', schema: P_SCHEMA, model: 'opus' }
);

phase('IsolatedCapture');
const IC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['isolatedIsOnePath','oracleAligned','perFork','ok','notes'],
  properties: {
    isolatedIsOnePath: { type: ['boolean','null'], description: 'the isolated runs proven NO co-residency (1-path tree, no siblings at the deciding depth)?' },
    oracleAligned: { type: ['boolean','null'], description: '#1/#2/#3 at the SAME fork position (apple-to-apple)?' },
    perFork: { type: ['array','string','null'], description: 'per fork: #1 co-resident argmax, #2 decode argmax, #3 isolated argmax + the C/R label' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const ic = await agent(
  CTX + '\n\nTASK (IsolatedCapture, GPU). Boot cat9, run the isolated 1-path forwards, capture #3 per fork. PROVE '
  + '1-path (no co-residency) + position-aligned. Teardown + recover. Return the schema.',
  { label: 'isolated-fork-capture', phase: 'IsolatedCapture', schema: IC_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','n_C_coresidency','n_R_realization','n_other','verdict','leverOrRelax','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'isolated proven 1-path/no-co-residency + position-aligned + oracle engaged?' },
    n_C_coresidency: { type: ['integer','null'], description: 'forks where isolated==decode != co-resident (leaf should have lost, co-residency)' },
    n_R_realization: { type: ['integer','null'], description: 'forks where isolated==co-resident != decode (genuine realization gap)' },
    n_other: { type: ['integer','null'], description: 'forks where isolated == neither (third realization)' },
    verdict: { type: 'string', description: 'C-heavy (co-residency lever re-opens) or R-heavy (genuine diffuse, relax)?' },
    leverOrRelax: { type: 'string', description: 'if C-heavy: the no-copy verify-row-isolation lever (bring to user). if R-heavy: relax, the SWE-quality gate is the answer. No decision here.' },
    rewardHackCheck: { type: 'string', description: 'isolated 1-path = our kernel (not a splice); native=oracle only; no copy/dense/multispine/bonus/WY/K1' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: prep=' + JSON.stringify(p) + ' capture=' + JSON.stringify(ic) + '. Default '
  + 'holds=false if the isolated run was NOT proven co-residency-free (still had siblings = vacuous), the 3 '
  + 'argmaxes are not at the same position, or the C/R split is asserted not from the actual #1/#2/#3. Conclude '
  + 'honestly: C-heavy (co-residency, the leaf should have lost, a no-copy lever) or R-heavy (genuine realization '
  + 'gap, relax). No bake/ship/close decision; no reward-hack.',
  { label: 'verify-isolated-fork', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { p, ic, v };
