export const meta = {
  name: 'fr13-nocopy-lossless-leaves-research',
  description: 'USER DIRECTIVE (2026-06-15): KEEP THE LEAVES (mandatory = the accept edge / speed), make the leafed verify forward LOSSLESS vs the no-spec recurrent decode, with NO state COPY and NO HBM-BANDWIDTH TAX (GB10 273 GB/s-bound). "do not copy stuff or make hbm tax; use cpu to research and code read; or we already have pattern?" Find the CHEAPEST no-copy/no-HBM route to make the leafed verify forward co-residency-INVARIANT on the committed path (per-node argmax == leaf-free / native-level), distinguishing what is ALREADY path-isolated (tree-attn full-attn, in_proj_ba M-inv, K1 store-boundary, margin-damp) from the OPEN leak (the diffuse GDN ~48-layer shared-h_cache tree-scan). KEY: the no-copy path-isolated GDN scan = the PARKED WY/STree direction, parked for failing ABS-0.0 — but the bar is now WITHIN-FLOOR (native 3 flips), so re-evaluate under the relaxed bar. Survey our pieces + the parked WY + the tree-verify-losslessness literature (SpecInfer/Sequoia/EAGLE/SpecTr ancestry isolation; linear-attn/GDN handling) + batch-invariance #42960. Answer: do we have a pattern, and the cheapest no-copy route. CPU read-only, adversarial verify. Output FR13_NOCOPY_LOSSLESS_LEAVES.md.',
  phases: [
    { title: 'Research' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn + 16 full-attn layers).',
'GB10 = 273 GB/s LPDDR5X unified mem, B=1 decode is HBM-BANDWIDTH-BOUND. Repo /home/mark/shared/lumoFlyWheel.',
'CPU-ONLY, READ-ONLY (a GPU committer-margin probe runs concurrently - do NOT edit code/boot; read our code +',
'vLLM source via scripts/vllm_src.sh + the binds + search online; write ONLY FR13_NOCOPY_LOSSLESS_LEAVES.md).',
'Pathspec commits. Search ONLINE first for the tree-verify-losslessness + batch-invariance state of the art.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS.',
'',
'THE USER CONSTRAINT (hard): (1) LEAVES ARE MANDATORY - the cat9 leaves are the accept edge (accept/event 3.198',
'> native 3.076); reshape-away (chain3 leaf-free = lossless but slow) is REJECTED. (2) LOSSLESS = per-served-',
'token argmax == the no-spec RECURRENT decode oracle (deployment-correct, fr13_recurrent_decode_oracle), at',
'native-E5 LEVEL (native = 3 clear-margin flips = the within-floor BAR, NOT abs-0.0). (3) NO COPY - the copy-',
'recurrent-state multi-spine route is BANNED + is itself NOT-lossless (CLOSED). (4) NO HBM TAX - the fix must be',
'COMPUTE-only (rearrange the in-SRAM scan) or COMMITTER-only (zero extra memory traffic); do NOT materialize',
'per-node isolated state buffers or add forward passes that re-read weights (that is the HBM tax on a 273 GB/s',
'part). dense/splice/forced-spine BANNED; native = A/B oracle only.',
'',
'THE MECHANISM (established, all verify HOLDS): the cat9 23-vs-native-3 flip gap = the LEAVES perturb the verify',
'forward\'s per-node argmaxes (parent_targets) via co-residency, so the LCP committer commits leaf-path /',
'boundary-shifted tokens that diverge from greedy decode (FR13_LEAF_CORESIDENCY_PATH: 16 forks vs chain3 1,',
'27-nat bursts). Decomposition of WHERE co-residency leaks:',
'  - FULL-ATTENTION (16 layers): tree-attention ANCESTRY MASK isolates each node to its path-to-root -> siblings',
'    masked out -> NO co-residency leak (confirm by reading the tree-attn mask). ALREADY path-isolated.',
'  - GDN LINEAR-ATTN (~48 layers): uses a SHARED h_cache tile (fr10_gdn_tree_kernel.py: one [N_PAD,BLOCK_V,',
'    DIM_K] tile holds ALL node states, L581/L651/L586-590), NOT the tree-attn mask. THIS is the leak site',
'    (diffusion deep-dive: geometric 1.166x/layer; FR13_DIFFUSION_DEEP_DIVE). K1 (per-node bf16 store-boundary)',
'    closed ~1/3 no-copy; in_proj_ba M-invariant (LUMO_FB pad) done; fp8/gate M-inv. The RESIDUAL diffuse leak +',
'    the confident leaf forks remain.',
'  - COMMITTER (LCP): margin-damp (do NOT fork on sub-1-nat parent_targets) = no-copy, free; the concurrent',
'    probe classifies how many forks are near-tie(fixable)-vs-confident(realization-diff). Confident forks need',
'    the GDN scan made path-isolated.',
'',
'THE PARKED LEVER (re-evaluate): a NO-COPY PATH-ISOLATED GDN tree-scan (each node\'s recurrent state = the',
'recurrence along ITS path-to-root, siblings non-interfering, computed IN the shared SRAM tile, no per-node',
'copy, no extra HBM) = the WY / STree direction. Banked: WY is PARKED-NOT-DEAD (failed ABS-0.0, not the',
'within-floor bar); STree no-copy kernel = NO-SHIP/future (reference_multispine_not_lossless_closed_nonship,',
'project_fr13_active_worker_codex_fr15, [[gdn_tree_superset_routes]]). The bar is NOW within-floor (native 3),',
'NOT abs-0.0 - so the parked WY may CLEAR the actual bar. Read FR13_WY* / the wy-archive branch notes + the STree',
'arXiv 2505.14969 root cause (path0 degraded by shared recurrent state) to judge.',
'',
'YOUR JOB - answer "do we have a pattern, and the cheapest no-copy/no-HBM route":',
'1. CONFIRM the leak decomposition: read the tree-attn mask (vllm_src.sh, the TREE_ATTN/forked-FA2 backend) -',
'   is each node TRULY ancestry-isolated (full-attn no leak)? Read the GDN tree-scan (fr10_gdn_tree_kernel.py',
'   shared h_cache) - is the per-node state ALREADY path-isolated (node reads parent, writes own) or does the',
'   shared-tile reduction/write leak across siblings? Pin the EXACT leak op (not narrative).',
'2. SURVEY the no-copy options for the GDN leak, each tagged COMPUTE-only/COMMITTER-only (allowed) vs COPY/HBM-',
'   tax (banned): (a) margin-damp (committer, free) - bounds only near-tie forks; (b) WY/STree no-copy path-',
'   isolated scan IN the shared tile - re-evaluate vs the WITHIN-FLOOR bar (does it need abs-0.0 or does within-',
'   floor suffice? what made it NO-SHIP - was it correctness or HBM/compute cost?); (c) make the existing',
'   shared-tile scan reduction batch/co-residency-invariant (extend the K1/in_proj_ba batch-invariance grind to',
'   the remaining diffuse ops - which ops, are they COMPUTE-only fixable?); (d) any literature pattern (SpecInfer',
'   Thm/Sequoia/EAGLE-2/SpecTr/Medusa tree-verify - how do THEY keep linear/SSM-attn tree-verify lossless without',
'   per-node copy? search online).',
'3. THE CHEAPEST ROUTE: rank the no-copy/no-HBM options by (lossless-within-floor reach x compute cost x',
'   implementation risk). Is there a route that keeps leaves + reaches native-3 within-floor + zero copy + zero',
'   HBM tax? Or is the honest answer "margin-damp covers the near-tie forks (no-copy, free) + the confident forks',
'   need the WY no-copy scan (compute-only, revive under within-floor) + accept the irreducible verify-vs-decode',
'   floor (~3, native has it too)"? Be concrete: name the route, the op, the expected residual flips, the',
'   compute/HBM cost.',
'4. DO WE HAVE A PATTERN: yes / partial / no, with the gap. If a fresh GPU experiment is needed to validate the',
'   top route, specify the EXACT minimal one (e.g. revive WY scan, score within-floor with leaves).',
'',
'DELIVERABLE: FR13_NOCOPY_LOSSLESS_LEAVES.md = the leak decomposition (pinned ops), the no-copy/no-HBM option',
'survey (each tagged allowed-vs-banned by the copy/HBM constraint), the cheapest route to leaves+lossless-within-',
'floor, the "do we have a pattern" verdict, and the minimal validating experiment. Distinguish MEASURED/CODE-READ',
'from INFERRED/LITERATURE. Quote FR13_BUG_CLASS_PLAYBOOK (#12 co-residency/trajectory, #10 codegen). Be SKEPTICAL',
'- the WY was parked for a reason; state HONESTLY whether within-floor revives it or not. Commit pathspec.',
].join('\n');

phase('Research');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['leakDecomposition','noCopyOptionSurvey','cheapestRoute','havePatternVerdict','minimalValidatingExperiment','committed','notes'],
  properties: {
    leakDecomposition: { type: 'string', description: 'CODE-READ: is full-attn truly ancestry-isolated (tree-attn mask, no leak)? is the GDN shared-tile scan per-node path-isolated or does it leak across siblings? the EXACT leak op cited' },
    noCopyOptionSurvey: { type: 'string', description: 'each option (margin-damp / WY-no-copy-scan / batch-invariance-extend / literature) tagged COMPUTE-only or COMMITTER-only (ALLOWED) vs COPY/HBM-tax (BANNED), with what it reaches' },
    cheapestRoute: { type: 'string', description: 'the ranked cheapest no-copy/no-HBM route to leaves+lossless-within-floor: named route, op, expected residual flips, compute/HBM cost' },
    havePatternVerdict: { type: 'string', description: 'yes/partial/no - do we have a pattern for leaves+lossless+no-copy+no-HBM, with the precise gap' },
    minimalValidatingExperiment: { type: 'string', description: 'the EXACT minimal GPU experiment to validate the top route (e.g. revive WY scan, score within-floor with leaves) if needed' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  CTX + '\n\nTASK (Research, no GPU, read-only). Search online FIRST for tree-verify-losslessness + SSM/linear-attn'
  + ' batch-invariance state of the art, then code-read our tree-attn mask + GDN shared-tile scan + the parked WY'
  + ' notes. Do steps 1-4. Write FR13_NOCOPY_LOSSLESS_LEAVES.md, commit pathspec. Return the schema.',
  { label: 'nocopy-lossless-research', phase: 'Research', schema: R_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','leakGrounded','optionsHonestlyTagged','wyReevalHonest','routeConcrete','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    leakGrounded: { type: 'string', description: 'is the leak decomposition from ACTUAL code (tree-attn mask + GDN scan lines cited), not narrative? full-attn-isolated + GDN-scan-leak claims verified?' },
    optionsHonestlyTagged: { type: 'string', description: 'are the no-copy options correctly tagged allowed (compute/committer-only) vs banned (copy/HBM), no smuggled copy/dense/multispine?' },
    wyReevalHonest: { type: 'string', description: 'is the WY within-floor re-evaluation honest (states clearly whether the relaxed bar revives it or not, with the original NO-SHIP reason)?' },
    routeConcrete: { type: 'string', description: 'is the cheapest route concrete (named op, residual flips, cost) not hand-wave; is the have-pattern verdict backed?' },
    recommendation: { type: 'string', description: 'single: the cheapest no-copy/no-HBM route + whether to validate it (the minimal experiment). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(r) + '. Default holds=false if the leak decomposition is'
  + ' narrative not code-read (must cite the tree-attn mask + GDN scan lines), any "no-copy" option actually'
  + ' smuggles a copy/dense/multi-spine/HBM-tax, the WY within-floor re-eval is hand-waved (must state the'
  + ' original NO-SHIP reason + whether within-floor changes it), or the cheapest route is not concrete. No'
  + ' close/pass-fail; no reward-hack.',
  { label: 'verify-nocopy-lossless', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, v };
