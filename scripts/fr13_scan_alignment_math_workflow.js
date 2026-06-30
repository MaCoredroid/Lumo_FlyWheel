export const meta = {
  name: 'fr13-scan-alignment-math',
  description: 'CPU read-ahead (parallel to the decisive scan-vs-native-packed A/B we834923g): read OUR scan kernel _tree_gdn_kernel/_gdn_node_step vs native packed-decode fused_recurrent_gated_delta_rule_packed_decode op-by-op, DO THE MATH on the candidate bit-exact divergence seams (fp32 accum order/FMA, cast boundaries, l2norm, raw-g, gate order, BV/warps codegen, chunk-vs-recurrent realization), pre-design the build-our-kernel alignment per the A/B branch (geometry-seam → recompute-from-spine; kernel-math → fp32/op-order/l2norm/raw-g). ANSWER: is native packed-decode RECURRENT-like-ours (so chunk-vs-recurrent gap was vs the WRONG reference) or genuinely different? Adversarial verify.',
  phases: [
    { title: 'Math' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (the decisive',
'scan-vs-native-packed A/B we834923g runs concurrently + its Build edits fr10_gdn_tree_kernel.py with an',
'ADDITIVE test-only launch override - do NOT edit code; read the kernel MATH (the _gdn_node_step body + native',
'kernel), write ONLY FR13_SCAN_ALIGNMENT_MATH.md). Pathspec commits only.',
'',
'CONTEXT: the 21-flip carrier hunt eliminated every cross-event STATE channel (FA2/full-attn, SSM-recurrent,',
'conv = all closed) and converged on the WITHIN-FORWARD DEPTH-SCALED GDN scan STATE-FEED realization gap',
'(chunk-vs-recurrent ~1-ULP born at L0, amplified ~32x by gate 1/rms over L41-L63). The decisive A/B (we834923g)',
'tests whether our deployed scan (BV16/w8) is INT-VIEW bit-exact to native\'s ACTUAL decode kernel',
'fused_recurrent_gated_delta_rule_packed_decode (the no-spec oracle\'s kernel) - which was NEVER tested',
'(scan only ever checked vs native_update_serial_per_path = a serial TORCH ref, bug-class #10). THIS workflow',
'does the code+math so the FIX (or the analytical confirmation) is ready when the A/B lands.',
'',
'THE LOSSLESS BAR (bind it): bit-exact to the INCUMBENT SASS (native packed-decode), NOT R-correct vs a serial',
'ref (feedback_math_correct_vs_bitexact). Aligning OUR scan kernel to native packed-decode = build-our-kernel-',
'bit-exact (feedback_no_reroute_reward_hacking); do NOT propose calling native in the served path (the banned',
'splice). The non-WY sub-levers are open (WY is PARKED - failed abs-0.0 not the within-floor bar).',
'',
'YOUR JOB (read-only, do the math):',
'1. THE PIVOTAL QUESTION FIRST: read native fused_recurrent_gated_delta_rule_packed_decode',
'   (/tmp/vllm_live_019/.../fla/ops/) - is it a RECURRENT rank-1 realization (sequential per-token state',
'   update, like OUR _gdn_node_step) or a CHUNKED realization? If native DECODE is recurrent-like-ours, then',
'   the "chunk-vs-recurrent gap" that the conv-doc named as the carrier was measured vs the CHUNKED-PREFILL',
'   realization (the WRONG reference for decode) - and our scan should be ~bit-exact to native-DECODE modulo',
'   codegen. State this clearly (it reframes the whole "diffuse irreducible" pessimism).',
'2. OP-BY-OP MAP: OUR scan _tree_gdn_kernel (src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py, _gdn_node_step',
'   ~:330-383, h_cache :458, launch :812-844) vs native packed-decode. Map: in_proj/load + dtype, beta/g',
'   (softplus + sigmoid gate), q/k l2norm (in-kernel vs pre), the rank-1 delta-rule recurrent update',
'   (state*=exp(g); v-=sum(state*k); v*=beta; state+=v*k), the output gate, the reduction axis (K=128), dtypes',
'   at each boundary. Cite file:line for both.',
'3. LOCATE + SCORE THE CANDIDATE BIT-EXACT SEAMS (the math): (a) fp32-accumulation OP-ORDER / FMA in the rank-1',
'   update + the two tl.sum(axis=1) over K; (b) the BV/warps/num_stages codegen (deployed BV16/w8 vs native',
'   geom BV32/w4 - the SRAM-flagged seam; does the K-reduction tree differ?); (c) bf16<->fp32 CAST boundaries;',
'   (d) l2norm in-kernel vs pre-norm + the 1e-6 eps; (e) raw-g vs pre-activated g, softplus threshold; (f) gate',
'   application order. For EACH: ALIGNABLE (->0.0, like the conv bf16-tap / scan static_range fixes) vs',
'   STRUCTURAL (needs a kernel rewrite). Map each to the A/B branch it would explain (geometry-seam vs',
'   kernel-math).',
'4. PRE-DESIGN THE ALIGNMENT per branch (ready-to-apply once the A/B says which): (branch geometry) recompute-',
'   from-spine @BV32/w4 (SRAM EXIT-2, already designed in FR13_SCAN_BITEXACT_SRAM_NPAD_VERDICT_BIND - cross-',
'   reference, do not redo); (branch kernel-math) the specific numerics edits to OUR _gdn_node_step in',
'   dependency order (upstream proj/norm before the recurrent update before the gate). Note cheap (cast/op-',
'   order) vs structural. Keep BUILD-OUR-KERNEL.',
'',
'Be SKEPTICAL (this session overturned BV/warps + FA2-tile + the BV=4 lead + the chunked-prefill reference as',
'overstated/wrong). If our scan is ALREADY recurrent-identical to native-decode such that bit-exactness is just',
'codegen (alignable), say so; if there is a genuine structural recurrent-vs-chunked difference, say that too.',
'Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-identity, #12 depth/co-residency). Write',
'FR13_SCAN_ALIGNMENT_MATH.md, commit pathspec.'
].join('\n');

phase('Math');
const M_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['nativeDecodeRecurrentOrChunked','kernelMap','candidateSeams','alignableVsStructural','alignmentDesign','buildOurKernelNotReroute','committed','notes'],
  properties: {
    nativeDecodeRecurrentOrChunked: { type: 'string', description: 'THE pivotal answer: is native packed-decode recurrent-like-ours or chunked? (reframes the chunk-vs-recurrent carrier), with file:line' },
    kernelMap: { type: 'string', description: 'op-by-op OUR _gdn_node_step vs native packed-decode (proj/beta/g/l2norm/recurrent-update/gate/dtypes/reduction), file:line both' },
    candidateSeams: { type: 'string', description: 'the located seams (fp-order, BV/warps codegen, cast boundaries, l2norm, raw-g, gate order) each scored + mapped to the A/B branch' },
    alignableVsStructural: { type: 'string', description: 'per seam: alignable (->0.0) vs structural (rewrite)' },
    alignmentDesign: { type: 'string', description: 'the per-branch ready alignment (geometry → recompute-from-spine x-ref; kernel-math → the _gdn_node_step numerics edits in dependency order)' },
    buildOurKernelNotReroute: { type: 'string', description: 'confirm no native call in the live path; native is A/B oracle only' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const m = await agent(
  CTX + '\n\nTASK (Math, no GPU, read-only). Do steps 1-4. Write FR13_SCAN_ALIGNMENT_MATH.md, commit pathspec. '
  + 'Return the schema.',
  { label: 'scan-alignment-math', phase: 'Math', schema: M_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','recurrentAnswerGrounded','mapGrounded','seamsComplete','rerouteCheck','structuralRisk','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    recurrentAnswerGrounded: { type: 'string', description: 'is the native-decode recurrent-vs-chunked answer grounded in the actual native kernel code (the pivotal claim)?' },
    mapGrounded: { type: 'string', description: 'is the op-by-op map grounded in BOTH kernels actual code?' },
    seamsComplete: { type: 'string', description: 'are the seams the right class + complete, or one missed?' },
    rerouteCheck: { type: 'string', description: 'does the alignment keep OUR kernel (no native call in live path)?' },
    structuralRisk: { type: 'string', description: 'real risk of a structural recurrent-vs-chunked rewrite, or just codegen alignment?' },
    recommendation: { type: 'string', description: 'single recommendation for applying once the A/B branch lands. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(m) + '. Default holds=false if the recurrent-vs-chunked '
  + 'answer or the op-by-op map is not grounded in BOTH kernels actual code, or if the alignment proposes a '
  + 'native call in the live path (reroute, banned). The recurrent-vs-chunked answer is load-bearing (it '
  + 'reframes the carrier) - hold it to grounded-in-code. No close/pass-fail; no reward-hack.',
  { label: 'verify-scan-alignment-math', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { m, v };
