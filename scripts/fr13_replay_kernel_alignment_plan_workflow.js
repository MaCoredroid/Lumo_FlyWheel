export const meta = {
  name: 'fr13-replay-kernel-alignment-plan',
  description: 'READ-AHEAD (CPU, while the replay A/B w2vaqcsmx boots GPU): pre-design HOW to make OUR _tree_gdn_replay_kernel bit-exact to the sequential reference (native fused_sigmoid_gating_delta_rule_update) IF the A/B shows divergence. Read both kernels, locate the EXACT candidate divergence seams (op-order/FMA, bf16<->fp32 cast boundaries, l2norm/raw-g, gate application, reduction order, rank-1-vs-sequential formulation), design the build-our-kernel-bit-exact alignment (NOT a reroute). Also serves as the analytical companion if the A/B lands ~0. Adversarial verify.',
  phases: [
    { title: 'Plan' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (the replay-durable-state A/B workflow w2vaqcsmx is',
'booting GPU from the SAME patcher concurrently - do NOT edit any code; write ONLY',
'FR13_REPLAY_KERNEL_ALIGNMENT_PLAN.md). Pathspec commits only.',
'',
'CONTEXT: the PRIME 21-flip carrier hypothesis (triply-reinforced: reanalysis back-loading norm-mean 0.696 +',
'never-A/B-vs-native + the FA2 correction ruling out the attention backend) is the CROSS-EVENT durable-state',
'handoff. The locked build runs FR13_REPLAY_ROUTE=1 (always-on): at commit it re-executes the accepted chain',
'from h0 via OUR _tree_gdn_replay_kernel (rank-1 Triton kernel; fr10_gdn_tree_kernel.py ~:546, patcher',
'_fr13_replay_launch ~L7348) and writes the durable next-event GDN recurrent state. Native MTP produces ITS',
'durable state with the SEQUENTIAL recurrent kernel fused_sigmoid_gating_delta_rule_update',
'(reference_gdn_verify_sequential_dispatch). The running A/B (w2vaqcsmx) MEASURES max_abs(H_ours - H_native_',
'seq) per-layer per-event. THIS workflow pre-designs the FIX so it is ready the moment that verdict lands.',
'',
'THE LOSSLESS SPEC FOR THE DURABLE STATE (bind it): the accepted tokens ARE committed/correct, so the durable',
'next-event h0 must EXACTLY equal the SEQUENTIAL recurrent state after those accepted tokens (what a no-spec',
'decode would have). So the reference is native fused_sigmoid_gating_delta_rule_update (sequential rank-1),',
'and aligning OUR replay kernel to it bit-exact is BUILD-OUR-KERNEL-BIT-EXACT-TO-THE-INCUMBENT',
'(feedback_no_reroute_reward_hacking, feedback_math_correct_vs_bitexact: the bar is BIT-EXACT not R-correct).',
'NOT a reroute: the reroute ban is about routing OUR compute THROUGH native to pass a metric while our kernel',
'stays unused; here we KEEP our kernel and make IT match. Do NOT propose "just call native fused_sigmoid_',
'gating for the durable state" as the fix (that would be the splice/reroute) - propose the numerics alignment',
'of OUR kernel.',
'',
'YOUR JOB (read-only, conditional pre-design):',
'1. READ BOTH KERNELS in full: OUR _tree_gdn_replay_kernel (fr10_gdn_tree_kernel.py ~:546 + the shared',
'   _gdn_node_step body ~:330-383 + _fr13_replay_launch patcher ~L7348) AND native fused_sigmoid_gating_delta_',
'   rule_update (find it in the live tree /tmp/vllm_live_019/... or the installed vllm; the FLA/mamba ops).',
'   Map them op-by-op: input projection split, beta/g (gate) computation, l2norm of q/k, the delta-rule',
'   recurrent update U_t = U_{t-1}(I - beta k k^T) + beta v k^T (or the exact form each uses), the conv-state',
'   handoff, the output gate, dtype at each boundary.',
'2. LOCATE THE CANDIDATE BIT-EXACT DIVERGENCE SEAMS (the same class that the GDN scan grind found - reference',
'   feedback_math_correct_vs_bitexact + feedback_fr12_subkernel_zero_gate): (a) fp32-accumulation OP-ORDER /',
'   FMA contraction in the rank-1 update; (b) bf16<->fp32 CAST BOUNDARIES (where each kernel rounds); (c)',
'   l2norm in-kernel vs pre-norm + raw-g vs pre-activated g; (d) the GATE application order (sigmoid gating);',
'   (e) reduction order over the chain (our rank-1 sweep vs native sequential); (f) whether our replay is a',
'   DIFFERENT mathematical formulation (chunked/rank-1) than native sequential - if so, bit-exactness may need',
'   our kernel to adopt native\'s exact sequential op sequence. For EACH seam: is it ALIGNABLE (->0.0, like the',
'   scan tl.range/bf16-tap fixes) or a real formulation difference (needs our kernel rewritten to native\'s',
'   sequence)?',
'3. DESIGN THE ALIGNMENT (ready-to-apply once the A/B confirms which seam(s) are live): the concrete numerics',
'   edits to OUR replay kernel to make it bit-exact to native sequential, seam by seam, in dependency order',
'   (fix upstream first - input proj/norm before the recurrent update before the gate). Note which are cheap',
'   (cast-boundary/op-order) vs structural (formulation rewrite). Keep it BUILD-OUR-KERNEL (no native call in',
'   the live path).',
'4. CAVEAT the conditionality: this is the FIX IF the A/B (w2vaqcsmx) shows nonzero+growing divergence. If the',
'   A/B lands ~0, the plan is the analytical confirmation that our replay is already faithful. Do NOT presume',
'   the outcome; present the seam map + the conditional fix.',
'',
'DELIVERABLE: FR13_REPLAY_KERNEL_ALIGNMENT_PLAN.md = op-by-op kernel map + the candidate seams (alignable vs',
'structural) + the ready-to-apply numerics alignment of OUR kernel + the dependency order. Be SKEPTICAL (this',
'session overturned several single-seam overstates); if our replay is a fundamentally different formulation',
'than native sequential such that bit-exactness needs a rewrite, say so plainly. Quote FR13_BUG_CLASS_PLAYBOOK',
'rows (#10 codegen-identity-not-spec-guaranteed, #12 cross-event). Commit pathspec.'
].join('\n');

phase('Plan');
const P_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['kernelMap','candidateSeams','alignmentDesign','alignableVsStructural','buildOurKernelNotReroute','conditionality','committed','notes'],
  properties: {
    kernelMap: { type: 'string', description: 'op-by-op map of OUR _tree_gdn_replay_kernel vs native fused_sigmoid_gating_delta_rule_update (proj/beta/g/l2norm/recurrent-update/conv-handoff/output-gate/dtypes), with file:line for both' },
    candidateSeams: { type: 'string', description: 'the located candidate bit-exact divergence seams (op-order/FMA, cast boundaries, l2norm/raw-g, gate order, reduction order, formulation) - each with WHERE in code' },
    alignmentDesign: { type: 'string', description: 'the concrete numerics edits to OUR replay kernel to reach bit-exact to native sequential, seam by seam in dependency order' },
    alignableVsStructural: { type: 'string', description: 'per seam: alignable (->0.0, cast/op-order) vs structural (needs our kernel rewritten to native sequence)' },
    buildOurKernelNotReroute: { type: 'string', description: 'confirm the plan keeps OUR kernel (no native call in the live path) = build-our-kernel-bit-exact, not splice/reroute' },
    conditionality: { type: 'string', description: 'this is the fix IF the A/B shows divergence; if ~0 it is the faithfulness confirmation - not presuming outcome' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const p = await agent(
  CTX + '\n\nTASK (Plan, no GPU, read-only). Do steps 1-4. Write FR13_REPLAY_KERNEL_ALIGNMENT_PLAN.md, commit '
  + 'pathspec. Return the schema.',
  { label: 'replay-kernel-alignment-plan', phase: 'Plan', schema: P_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','mapGrounded','seamsComplete','rerouteCheck','structuralRisk','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    mapGrounded: { type: 'string', description: 'is the op-by-op map grounded in BOTH kernels actual code (not guessed)?' },
    seamsComplete: { type: 'string', description: 'are the candidate seams the right class (matches the scan-grind lessons) + complete, or is a seam missed?' },
    rerouteCheck: { type: 'string', description: 'does the alignment keep OUR kernel (no native call in live path) - confirm not a splice/reroute' },
    structuralRisk: { type: 'string', description: 'is there a real risk our replay is a fundamentally different formulation needing a rewrite (not a cheap alignment)?' },
    recommendation: { type: 'string', description: 'single recommendation for applying the plan once the A/B verdict lands. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(p) + '. Default holds=false if the kernel map is not '
  + 'grounded in BOTH kernels actual code, or if the plan proposes calling native fused_sigmoid_gating in the '
  + 'live path (that is the reroute/splice, banned). Flag any missed seam + the structural-rewrite risk. No '
  + 'close/pass-fail; no reward-hack.',
  { label: 'verify-alignment-plan', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { p, v };
