export const meta = {
  name: 'fr13-math-rounding-noflip',
  description: 'USER (2026-06-15): "K1 is a math rounding; this feels more like a math rounding issue, or some math way to make it not flip." The verify-vs-decode flip IS fundamentally a FLOATING-POINT realization difference (the tree-batched leaf-co-resident forward computes the SAME math as the single-sequence recurrent decode, with different rounding/op-order). K1 (bf16 b_h store-boundary alignment) was ONE such rounding fix = closed ~1/3. FIND the NO-COPY / NO-HBM / COMPUTE-ONLY MATH-NUMERICS way(s) to make the verify forward per-node argmax ROUNDING-INVARIANT (== the decode forward) so it does NOT flip - NOT a heuristic spine-bonus (user rejected that). Prime lead (from FR13_NOCOPY_LOSSLESS_LEAVES, verify HOLDS): the GDN "leak" = N_PAD 4->8 RECOMPILES the tl.static_range unroll + the tl.where reduction -> a different FMA/ACCUMULATION ORDER even for spine nodes = a pure rounding-order effect (bug-class #10). So the candidate math fix = make the scan reduction/accumulation order N_PAD-INVARIANT (fixed order regardless of tree size) + compensated/fp32 accumulation at the argmax-critical reductions + the lm-head GEMV. Survey batch-invariance (#42960 / deterministic-inference) + compensated-summation + fixed-reduction-order, each tagged compute-only(allowed) vs copy/HBM(banned). Distinguish this from the REFUTED recompute (which changed GEOMETRY BV32 not just order). CPU read-only, ONLINE-FIRST, adversarial verify. Output FR13_MATH_ROUNDING_NOFLIP.md.',
  phases: [
    { title: 'Research' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn + 16 full-attn layers).',
'GB10 = 273 GB/s LPDDR5X, B=1 decode HBM-BANDWIDTH-BOUND (fix must add ~0 memory traffic). Repo',
'/home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (an apple-to-apple committer workflow runs concurrently; do',
'NOT edit code/boot). SEARCH ONLINE FIRST for the numerics state of the art, then code-read our kernels via',
'scripts/vllm_src.sh + fr10_gdn_tree_kernel.py. Write ONLY FR13_MATH_ROUNDING_NOFLIP.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS.',
'',
'THE USER FRAMING (the task): the cat9 flips are a verify-vs-decode realization difference = a MATH/ROUNDING',
'issue (two float realizations of the same recurrence: tree-batched co-resident forward vs single-seq recurrent',
'decode). K1 (per-node bf16 b_h store-boundary round-trip, _gdn_node_step L503-504) was a rounding ALIGNMENT and',
'closed de-cascaded 18->12 (~1/3), accept held. FIND the MATH way(s) to close the rest = make the verify per-',
'node argmax ROUNDING-INVARIANT (matching the decode forward) so it stops flipping. NOT a spine-bonus / margin-',
'damp (user REJECTED). NOT WY (PARKED by user). The fix must be COMPUTE-ONLY (rearrange the in-SRAM math) - NO',
'copy, NO extra HBM traffic, NO extra forward pass.',
'',
'WHAT IS BANKED (all verify HOLDS this session):',
'- The flip = verify forward per-node argmax != recurrent-decode oracle argmax (FR13 compare target). Carrier =',
'  LCP-committer fork driven by this realization gap. Per-forward kernels: scan/conv/fp8/FA2 individually closed',
'  or partial; the gap is DIFFUSE (FR13_DIFFUSION_DEEP_DIVE, verify HOLDS): geometric ~1.166x/layer growth over',
'  ~48 GDN layers (NOT additive-ULP, NOT 32x); the per-layer floor enters at the GDN scan bf16-store + conv',
'  anchor-row, amplified by gate 1/rms + deep full-attn, crystallizes the argmax flip at L60/L61.',
'- THE PINNED ROUNDING-ORDER LEAK (FR13_NOCOPY_LOSSLESS_LEAVES, verify HOLDS): the GDN per-node state is',
'  ALGEBRAICALLY path-isolated (reads only ancestor rows, writes own row) - NOT a sibling bleed. The actual',
'  co-residency "leak" = adding leaves moves N_PAD = 1<<(n-1).bit_length() from 4->8 (kernel L159-163), which',
'  RECOMPILES the tl.static_range(0,N_PAD) unroll + the tl.where reduction over the offs_n range = a DIFFERENT',
'  FMA/accumulation ORDER for the SAME spine nodes (bug-class #10 codegen-identity). = a pure ROUNDING-ORDER',
'  diff. NOTE: making state bit-exact via RECOMPUTE (which ALSO changed geometry to native BV32/w1/s3) made e2e',
'  flips ROSE 23->32 - but that confounded order-invariance with a geometry change; a PURE same-geometry order-',
'  invariance fix is UNTESTED and is the prime math lead.',
'- K1 = the bf16 store-boundary alignment (1/3). K2-K5 (l2norm-div, beta-round-trip, gate-order, conv-tap)',
'  provably ~0. fp8 in_proj/o_proj M-invariant (BLOCK_SIZE_M=64). gate fp32-internal.',
'',
'YOUR JOB - the no-copy/no-HBM MATH-NUMERICS ways to make the verify argmax not flip:',
'1. CHARACTERIZE the flip as a floating-point phenomenon (build on the diffusion + no-copy binds, do not redo):',
'   WHICH ops contribute the rounding drift (the bf16-store, the N_PAD-dependent reduction order, the gate 1/rms',
'   amplification, the per-layer residual accumulation, the lm-head GEMV over the verify rows), and WHERE the',
'   argmax becomes sensitive (the small clean-margin structural-boundary tokens, 1-2 nat).',
'2. SURVEY the MATH/NUMERICS levers, each tagged COMPUTE-ONLY/no-copy/no-HBM (ALLOWED) vs copy/HBM (BANNED):',
'   (a) FIXED REDUCTION ORDER / N_PAD-INVARIANCE: make the tl.static_range + tl.where reduction order INDEPENDENT',
'       of N_PAD/tree-size (a fixed canonical order) so the spine nodes get the SAME FMA order with or without',
'       leaves - the pinned-leak fix, compute-only. Read kernel L159-163, L578-651; is it a clean reordering?',
'       Distinguish from the refuted recompute (geometry change, not just order).',
'   (b) COMPENSATED / HIGHER-PRECISION ACCUMULATION: Kahan/Neumaier or fp32 accumulation at the argmax-critical',
'       reductions (the per-layer residual add, the gate, the lm-head GEMV) so the diffuse 1.166x/layer drift',
'       does not cross the 1-2 nat clean margin. Which ops, what precision, compute-only?',
'   (c) BATCH-INVARIANCE (#42960 / deterministic-inference / thinking-machines batch-invariant kernels): make',
'       the tree-batched GDN reduction batch-composition-invariant so the spine logits == the single-seq logits.',
'       What is the state of the art + does it apply to the GDN delta-rule scan?',
'   (d) ROUNDING-MODE / OP-ORDER alignment beyond K1 (round-to-nearest-even consistency, accumulation order to',
'       match the native sequential decode).',
'3. THE CHEAPEST MATH ROUTE: rank the ALLOWED levers by (argmax-flip reduction x compute cost x risk). Is there',
'   a compute-only no-copy no-HBM math fix that makes the verify argmax rounding-invariant enough to stop the',
'   confident-fork flips (the ones a bonus cannot touch losslessly)? Name the op + expected effect. Honestly',
'   state if the drift is genuinely diffuse-over-48-layers such that no single math reorder suffices (then the',
'   compensated-accumulation route is the only math lever, quantify its reach).',
'4. ONLINE: search batch-invariant inference (PyTorch deterministic, thinking-machines "defeating',
'   nondeterminism in LLM inference", vLLM #42960), compensated summation in attention/SSM kernels, and any',
'   linear-attention/delta-rule rounding-invariance work. Cite what is real vs inferred.',
'',
'DELIVERABLE: FR13_MATH_ROUNDING_NOFLIP.md = the floating-point characterization, the no-copy/no-HBM math-lever',
'survey (each tagged allowed/banned), the cheapest compute-only math route to argmax-rounding-invariance + the',
'op + expected reach, and a minimal validating experiment. Distinguish MEASURED/CODE-READ from INFERRED/',
'LITERATURE. Do NOT propose the spine-bonus (rejected) or WY (parked). Reward-hacks banned (no copy/dense/multi-',
'spine/HBM-tax). Quote FR13_BUG_CLASS_PLAYBOOK (#10 codegen/reduction-order, #12 trajectory). Commit pathspec.',
].join('\n');

phase('Research');
const R_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['fpCharacterization','mathLeverSurvey','cheapestMathRoute','genuinelyDiffuseHonest','minimalValidatingExperiment','committed','notes'],
  properties: {
    fpCharacterization: { type: 'string', description: 'which ops contribute the rounding drift (bf16-store, N_PAD reduction order, gate 1/rms, per-layer accum, lm-head GEMV) + where the argmax becomes sensitive; built on the diffusion/no-copy binds' },
    mathLeverSurvey: { type: 'string', description: 'each math lever (a) fixed-reduction-order/N_PAD-invariance (b) compensated/fp32 accumulation (c) batch-invariance #42960 (d) rounding-mode/op-order, tagged COMPUTE-ONLY(allowed) vs copy/HBM(banned), with reach' },
    cheapestMathRoute: { type: 'string', description: 'the ranked cheapest compute-only no-copy/no-HBM math route to argmax-rounding-invariance: named op, expected effect on the confident-fork flips, cost; vs the refuted recompute' },
    genuinelyDiffuseHonest: { type: 'string', description: 'HONEST: is the drift a single fixable rounding-order op or genuinely diffuse over ~48 layers (then compensated-accumulation is the only lever, with its quantified reach)?' },
    minimalValidatingExperiment: { type: 'string', description: 'the EXACT minimal experiment to validate the top math route (e.g. N_PAD-invariant reduction order at fixed geometry, re-score vs recurrent oracle)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const r = await agent(
  CTX + '\n\nTASK (Research, no GPU, read-only). SEARCH ONLINE FIRST (batch-invariance/compensated-summation/'
  + 'deterministic-inference), then code-read the GDN scan reduction (fr10_gdn_tree_kernel.py L159-163, L578-651)'
  + '. Do steps 1-4. Write FR13_MATH_ROUNDING_NOFLIP.md, commit pathspec. Return the schema.',
  { label: 'math-rounding-noflip', phase: 'Research', schema: R_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','characterizationGrounded','leversHonestlyTagged','routeConcrete','diffuseVerdictHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    characterizationGrounded: { type: 'string', description: 'is the fp characterization from the actual binds/code (cited), not narrative?' },
    leversHonestlyTagged: { type: 'string', description: 'are the math levers correctly tagged compute-only(allowed) vs copy/HBM(banned), no smuggled copy/dense/recompute-geometry-change?' },
    routeConcrete: { type: 'string', description: 'is the cheapest math route concrete (named op, expected flip-reach, cost), and correctly distinguished from the refuted recompute (geometry vs order)?' },
    diffuseVerdictHonest: { type: 'string', description: 'is the diffuse-vs-single-op honesty backed (does it admit if no single reorder suffices and quantify the compensated-accumulation reach)?' },
    recommendation: { type: 'string', description: 'single: is there a compute-only no-copy/no-HBM math route to argmax-rounding-invariance worth testing, or is it genuinely diffuse. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(r) + '. Default holds=false if the characterization is '
  + 'narrative not cited, any "allowed" math lever smuggles a copy/HBM-tax/geometry-change (the recompute was '
  + 'refuted BECAUSE it changed geometry not just order - keep that distinction), the cheapest route is not '
  + 'concrete, the diffuse-vs-fixable verdict is hand-waved, or it proposes the rejected spine-bonus or parked '
  + 'WY. No close/pass-fail; no reward-hack.',
  { label: 'verify-math-rounding', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { r, v };
