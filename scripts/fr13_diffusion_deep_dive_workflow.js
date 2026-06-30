export const meta = {
  name: 'fr13-diffusion-deep-dive',
  description: 'DEEP DIVE on the DIFFUSION (user: stop hand-waving "diffuse ~1-ULP/48-layer GDN accumulation"). The carrier-reopen pinned the cat9 23-vs-native-3 flips on a residual TWO-kernel verify-forward floor (FA2-fork 2-ULP + diffuse-GDN ~1-ULP-over-48-layers) AMPLIFIED by tree topology - but the per-layer accumulation is ASSUMED not MEASURED. Build a QUANTITATIVE per-layer model from the BANKED captures + code: layer-by-layer cat9-verify-vs-native divergence, how it grows, the gate 1/rms amplification, which kernel (FA2-fork vs GDN-scan vs in_proj/gate/o_proj) dominates, where it crosses the argmax-flip threshold, and whether it is a FIXABLE per-layer seam or genuinely diffuse. Distinguish MEASURED from ASSUMED. native-E5=3 = NOT irreducible. CPU read-only, adversarial verify.',
  phases: [
    { title: 'DeepDive' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn layers + 16 full-attn).',
'Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a GPU reshape A/B runs concurrently - do NOT edit',
'code/boot; read banked captures + code + vLLM source via scripts/vllm_src.sh; write ONLY',
'FR13_DIFFUSION_DEEP_DIVE.md). Pathspec commits.',
'',
'GROUNDING RULE (user): read vLLM source DIRECTLY from the pinned image via scripts/vllm_src.sh (0.19.2rc1.dev134),',
'NEVER a /tmp cache. The whole journey was re-grounded (c589cd63, no shift).',
'',
'THE HAND-WAVE TO REPLACE (user: "dive deep on diffusion instead of hand-waving"): every bind invokes "diffuse',
'per-layer ~1-bf16-ULP GDN accumulation over ~48 layers, amplified ~32x by gate 1/rms, crystallizing at',
'L60/L61" (FR13_DIFFUSE_GDN_EXPLAINED.md, reference_diffuse_gdn_accumulation_explained) as the carrier of the',
'cat9 23-vs-native-3 flips - but it is largely ASSERTED. The decisive re-run RULED OUT the GDN scan STATE-feed',
'as the e2e carrier (recompute bit-exact 0.0 yet flips ROSE), and the carrier-reopen (FR13_CARRIER_REOPEN.md,',
'verify HOLDS) pinned ~16 confident STRUCTURAL-BOUNDARY forks (dev up to 10, NOT sub-ULP near-ties) on a TWO-',
'kernel verify-forward floor (forked-FA2 tree-bias + GDN tree-scan) AMPLIFIED by tree topology. So the',
'"diffusion" needs to be MEASURED + mechanistically pinned, not invoked.',
'',
'YOUR JOB - replace the hand-wave with a quantitative per-layer account:',
'1. GATHER THE MEASURED PER-LAYER EVIDENCE: find every banked per-layer / per-node capture of cat9-verify-vs-',
'   native divergence (output/fr13_node7_ladder, output/fr13_node5_ladder, the L0-L58 first-nonzero ladder, any',
'   gdn_substate / scan_capture / per-layer max_abs dumps, FR13_GATEA_DEEP_DIVERGENCE, FR13_NODE5_LADDER_DIFFUSE,',
'   FR13_22flip_carrier_l0gdn). Tabulate: per GDN/full-attn layer, the cat9-vs-native max_abs (or argmax-match)',
'   - which layers are 0.0, which first diverge, the magnitude per layer. SEPARATE what is MEASURED (a real GPU',
'   capture) from ASSUMED (the ~1-ULP/32x narrative).',
'2. THE GROWTH MODEL (quantify, do not assert): is the per-layer divergence (a) CONSTANT per layer (each layer',
'   adds ~1 ULP, linear in depth), (b) AMPLIFIED (the gate 1/rms multiplies - read RMSNormGated forward via',
'   vllm_src.sh: g=1/sqrt(mean(x^2)+eps), so small x -> large 1/rms; quantify the actual amplification factor',
'   from the banked norms, NOT "~32x"), or (c) crosses fp8 GEMM buckets (in_proj_qkvz/o_proj block-scaled).',
'   Trace ONE layer end-to-end (in_proj -> conv -> scan -> gate -> o_proj) and where the ULP enters + grows.',
'3. WHICH KERNEL DOMINATES: the scan is ruled out e2e (state bit-exact via recompute didnt help) - so is the',
'   per-layer floor from the FA2-fork (2-ULP MMA-grouping, project_fr13_fa2_fork_nocopy_floor - read whether it',
'   is DETERMINISTIC or probabilistic, and whether it compounds), the in_proj/o_proj fp8 GEMM realization, the',
'   gate, or genuinely the GDN recurrent op-order? Attribute the per-layer divergence to its source kernel.',
'4. FIXABLE-OR-DIFFUSE verdict: native-E5=3 at the SAME model/fp8/frame is the existence proof a 3-flip',
'   realization exists -> NOT irreducible. So WHERE is the fixable lever: is it one dominant layer/kernel that',
'   could be aligned (a real seam after all), a small number of seams, or genuinely ~48 independent ~1-ULP',
'   contributions (truly diffuse -> only topology/accept reduces it, not a kernel fix)? Be quantitative: if',
'   "diffuse", show the per-layer contributions are comparable + numerous; if not, name the dominant 1-3 layers.',
'5. CONNECT TO E2E: how does the per-layer floor become the ~16 structural-boundary argmax flips - at which',
'   layers/positions does the accumulated divergence first flip the final-logit argmax (the lm-head GEMV over',
'   the verify rows)? Reconcile with the carrier-reopen "structural-boundary, dev up to 10" (the flips are NOT',
'   sub-ULP - so the per-layer floor must AMPLIFY to dev~10 at boundaries: show the amplification, or revise the',
'   floor claim).',
'',
'DELIVERABLE: FR13_DIFFUSION_DEEP_DIVE.md = the per-layer divergence TABLE (measured vs assumed), the growth',
'model (quantified, not "~32x"), the dominant-kernel attribution, the fixable-or-genuinely-diffuse verdict with',
'numbers, and the e2e-flip mechanism. If a real GPU per-layer re-capture is needed to close a gap (the banked',
'ladders may be stale/0.19.0-line-keyed), specify the EXACT minimal capture. Be SKEPTICAL - this exists because',
'the "diffuse" claim has been invoked without measurement. Quote FR13_BUG_CLASS_PLAYBOOK rows (#12 depth-',
'accumulation, #10 codegen-identity). Commit pathspec.'
].join('\n');

phase('DeepDive');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['perLayerTable','measuredVsAssumed','growthModel','dominantKernel','fixableOrDiffuse','e2eFlipMechanism','minimalRecaptureIfNeeded','committed','notes'],
  properties: {
    perLayerTable: { type: 'string', description: 'per GDN/full-attn layer cat9-vs-native max_abs/argmax-match from BANKED captures: which 0.0, which first-diverge, magnitude' },
    measuredVsAssumed: { type: 'string', description: 'explicit split: what is a real GPU measurement vs what is the asserted ~1-ULP/32x narrative' },
    growthModel: { type: 'string', description: 'quantified: constant-per-layer / gate-1/rms-amplified (the ACTUAL factor from banked norms) / fp8-bucket-cross; one layer traced end-to-end' },
    dominantKernel: { type: 'string', description: 'attribution of the per-layer floor: FA2-fork (deterministic?) / fp8 in_proj-o_proj / gate / GDN recurrent op-order - with magnitude' },
    fixableOrDiffuse: { type: 'string', description: 'is it 1-3 dominant alignable layers/kernels (a seam) or genuinely ~48 comparable diffuse contributions (only topology/accept reduces)? with numbers. native=3 = not irreducible' },
    e2eFlipMechanism: { type: 'string', description: 'how the per-layer floor amplifies to the ~16 structural-boundary (dev~10) flips - reconcile the floor magnitude with dev~10 or revise' },
    minimalRecaptureIfNeeded: { type: 'string', description: 'the EXACT minimal GPU per-layer capture to close any gap (if the banked ladders are insufficient/stale)' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (DeepDive, no GPU, read-only). Do steps 1-5 QUANTITATIVELY from banked captures + code. Write '
  + 'FR13_DIFFUSION_DEEP_DIVE.md, commit pathspec. Return the schema.',
  { label: 'diffusion-deep-dive', phase: 'DeepDive', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','tableGrounded','growthQuantified','attributionSound','diffuseVerdictHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    tableGrounded: { type: 'string', description: 'is the per-layer table from ACTUAL banked captures (spot-check 2-3), not re-asserted from the hand-wave?' },
    growthQuantified: { type: 'string', description: 'is the growth model an actual number (gate amplification re-derived from real norms) or still "~32x" hand-wave?' },
    attributionSound: { type: 'string', description: 'is the dominant-kernel attribution grounded, consistent with scan-ruled-out + FA2-fork-floor?' },
    diffuseVerdictHonest: { type: 'string', description: 'is "fixable seam vs genuinely diffuse" backed by the per-layer numbers, not asserted either way?' },
    recommendation: { type: 'string', description: 'single recommendation: a dominant alignable layer to test, or confirm genuinely-diffuse (=> topology/accept is the only lever). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the per-layer table is '
  + 're-asserted from the "diffuse" narrative rather than read from actual banked captures (spot-check), or the '
  + 'growth model is still a hand-wave ("~32x") not a re-derived number, or the fixable-vs-diffuse verdict is '
  + 'asserted without the per-layer contribution magnitudes. The whole point is to REPLACE hand-waving with '
  + 'measurement. No close/pass-fail; no reward-hack.',
  { label: 'verify-diffusion-deep-dive', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
