export const meta = {
  name: 'fr13-scan-bitexact-sram-npad-tension',
  description: 'USER (concrete follow-up to FR13_BV_SPILL_VERDICT, which deferred this): IF bit-exactness to native forces the 4/32 (num_warps=4, BV=32) reduction op-order, how do we resolve the h_cache=[N_PAD,BV,DIM_K] fp32 SRAM/register-spill tension at deep trees, GIVEN trees MUST be >4 nodes (N_PAD up to 16) for speculation to pay off? Chase BV=4/num_warps=4 (predicted bit-exact + 32KB no-spill at N_PAD=16) + re-confirm whether num_warps=8 is ACTUALLY bit-exact (dissolves the tension). Concrete register/SRAM arithmetic + ranked resolutions + GPU test plan. Adversarial verify.',
  phases: [
    { title: 'Design' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121): 128 fp32 lanes/CTA at 4 warps, ~255 fp32 reg/thread (lane) cap, ~99KB',
'shared-mem (SRAM)/CTA, 273 GB/s LPDDR5X (B=1 decode is BANDWIDTH-BOUND on this pool, so any register spill',
'to global = a real TPS tax on the same saturated bus). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY,',
'READ-ONLY (a GPU workflow may run concurrently; do NOT edit code; write ONLY',
'FR13_SCAN_BITEXACT_SRAM_NPAD.md). Pathspec commits only.',
'',
'THE TENSION (user, concrete): for the cat9 verify SCAN kernel (src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py,',
'the _gdn_node_step body ~:330-383 + its launch ~:812-844/:1245-1284, h_cache = [N_PAD, BV, DIM_K] fp32 at',
'~L277), bit-exactness to NATIVE may require matching native fused_sigmoid_gating geometry BV=32 / num_warps=4',
'/ num_stages=3 (native [32,128] -> sizePerThread=[1,4] threadsPerWarp=[1,32] warpsPerCTA=[4,1] = 4 warps on',
'ROWS, 32 lanes on the K=128 contraction). BUT the deployed cat9 uses BV=16 / num_warps=8 (an interim SPILL',
'fix). The h_cache spill arithmetic (FR13_BV_SPILL_VERDICT.md, w921xvgzx 2026-06-09, which RESEARCHED this then',
'DEFERRED it to the TPS gate): h_cache fp32 held in registers; N_PAD families = 2,4,8,8,16 (cat families',
'2/3/6/8/14-node); DIM_K=128. At N_PAD=16,BV=16,4warps = 128KB / 128 lanes = 256 fp32 regs/lane > 255 cap ->',
'SPILLS to LPDDR5X (+ 128KB > 99KB SRAM so cannot park in shared). num_warps=8 halves regs/lane to 128 (kills',
'the hard spill) BUT FR13_BV_SPILL_VERDICT flags it as a [LIVE prediction]: "thread-mapping change could',
'perturb the layout... re-run the gate to confirm still bit-exact" - i.e. num_warps=8 bit-exactness was NEVER',
'CONFIRMED. The decouple hope (BV=1 reduce + separate state) was ABANDONED there ("you cannot give the reduce',
'M>=2 rows without the state being M>=2 rows; BLOCK_V=1 unreachable bit-exact = 1.19e-7").',
'',
'HARD CONSTRAINT (user): "we must go beyond 4 node for tree to make sense" - the tree MUST be >4 nodes (cat9 =',
'9, deeper families up to 14-node / N_PAD=16) for the speculative speedup to be worth it. So "shrink the tree',
'to fit SRAM" is OFF the table. N_PAD must scale to 16.',
'',
'THE PROMISING LEAD to chase HARD (verify rigorously, do not just assert): FR13_BV_SPILL_VERDICT predicted',
'"BV=4 reliably bit-exact" (the K=128 reduction needs leading extent >= num_warps=4; BV=4 keeps the 2-D',
'[M>=2,128] reduction tree, only BV=1 collapses). At N_PAD=16, BV=4, 4 warps: h_cache = 16*4*128*4 = 32KB ->',
'32KB/128 lanes = 256 bytes/lane = 64 fp32 regs/lane < 255 cap = NO SPILL, and 32KB < 99KB SRAM. CRUCIALLY:',
'BV only sets the ROW count of the [BV,128] state tile; the K=128 CONTRACTION (the bit-exact axis: tl.sum over',
'DIM_K) order is set by the 32-lanes-on-K mapping, which BV=4 vs BV=32 may share (4 K-elems/lane both) -> the',
'per-row K-reduction order is IDENTICAL -> BV=4 plausibly bit-exact to native BV=32. IF TRUE: BV=4/num_warps=4',
'= bit-exact (native K-order) + spill-free at N_PAD=16 + scales to >4-node trees = resolves the WHOLE tension.',
'',
'NOTE the SCAN vs REPLAY distinction: the h_cache[N_PAD,BV,DIM_K] spill is the per-forward verify SCAN kernel',
'(caches all tree-node states). The REPLAY kernel (durable-state regen, _tree_gdn_replay_kernel) is SEQUENTIAL',
'over the LINEAR accepted chain = ONE [BV,DIM_K] state at a time, NO N_PAD cache -> the alignment-plan STEP 0',
'(replay -> 4/32) does NOT spill. Confirm that. The SRAM/N_PAD tension is the SCAN deployment problem; resolve',
'it for the SCAN.',
'',
'YOUR JOB (concrete, beyond the prior deferred research):',
'1. STATE THE TENSION WITH EXACT ARITHMETIC: for each N_PAD family (2,4,8,8,16) x candidate BV (4,16,32) x',
'   num_warps (4,8): h_cache bytes, regs/lane, spill? (>255), SRAM-parkable? (<99KB), occupancy. A concrete',
'   table. Identify which (BV,warps) are bit-exact-CANDIDATES (preserve the native 32-lanes-on-K reduction).',
'2. RIGOROUSLY EVALUATE BV=4/num_warps=4: (a) is the K=128 reduction order bit-identical to native BV=32 (read',
'   the actual tl.sum/reduction in _gdn_node_step + how Triton maps [4,128] vs [32,128] to warps/lanes - does',
'   BV=4 keep 32 lanes on K with 4 K-elems/lane, or does it re-warp onto K and change the tree)? (b) the',
'   no-spill arithmetic at N_PAD=16. (c) the PERF cost: BV=4 = more programs (smaller V-tile) - at B=1',
'   bandwidth-bound decode is that hidden behind the weight DMA, or a real launch/occupancy tax? Quantify.',
'3. EVALUATE THE ALTERNATIVES (rank by bit-exact x spill-free x scales-to-N_PAD16 x speed): (i) RE-CONFIRM',
'   num_warps=8 bit-exactness empirically (if 8/16 IS bit-exact, the tension DISSOLVES - cheapest; design the',
'   int-view re-measure, NEVER atol); (ii) recompute-from-spine (FR13_CACHE_SCALING_FUTURE - do not cache all',
'   N_PAD branch states, recompute from the spine on demand; bit-exact? compute cost?); (iii) N_PAD streaming/',
'   tiling (process N_PAD in SRAM-fitting chunks; re-load cost); (iv) two-pass (spill-friendly state pass +',
'   bit-exact reduction pass); (v) accept the spill + QUANTIFY the TPS tax (spill to the decode-saturated',
'   273 GB/s bus). For each: does it preserve native-bit-exact op-order, fit N_PAD=16, and keep >4-node trees.',
'4. RECOMMEND a single concrete deployable path + the dependency/fallback order + the GPU test plan (the minimal',
'   boots to confirm bit-exactness AND no-spill AND TPS at N_PAD=16). Online research: GB10/Triton register-',
'   pressure + N_PAD/sequence tiling + recompute-vs-cache in linear-attention/SSM kernels.',
'',
'Be SKEPTICAL (this session overturned BV/warps + FA2-tile + width-H1 as overstated). If BV=4 does NOT in fact',
'preserve the native K-reduction (re-warps onto K), say so and fall to the next option. The deliverable is a',
'CONCRETE resolution with arithmetic, not a survey. Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-identity,',
'#12 cross-event/co-residency). Write FR13_SCAN_BITEXACT_SRAM_NPAD.md, commit pathspec.'
].join('\n');

phase('Design');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['tensionArithmetic','bv4Verdict','numWarps8Recheck','alternatives','recommendation','gpuTestPlan','scanVsReplay','committed','notes'],
  properties: {
    tensionArithmetic: { type: 'string', description: 'the concrete table: N_PAD family x BV x num_warps -> h_cache bytes, regs/lane, spill?, SRAM-parkable?, bit-exact-candidate?' },
    bv4Verdict: { type: 'string', description: 'rigorous: does BV=4/num_warps=4 preserve native BV=32 K=128 reduction order (bit-exact) AND no-spill at N_PAD=16 AND acceptable B=1 perf? grounded in the actual reduction code + Triton warp/lane mapping' },
    numWarps8Recheck: { type: 'string', description: 'the empirical int-view re-measure design for whether num_warps=8 (deployed) is ACTUALLY bit-exact to native 4 - if yes the tension dissolves' },
    alternatives: { type: 'string', description: 'recompute-from-spine / N_PAD-tiling / two-pass / accept-spill+TPS-tax, each scored bit-exact x spill-free x scales-N_PAD16 x speed' },
    recommendation: { type: 'string', description: 'the single concrete deployable path + dependency/fallback order' },
    gpuTestPlan: { type: 'string', description: 'minimal GPU boots to confirm bit-exact AND no-spill AND TPS at N_PAD=16 (>4-node tree)' },
    scanVsReplay: { type: 'string', description: 'confirm the spill is the SCAN h_cache problem; the replay (sequential, no N_PAD cache) STEP-0 4/32 does not spill' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (Design, no GPU, read-only). Do steps 1-4. Write FR13_SCAN_BITEXACT_SRAM_NPAD.md, commit '
  + 'pathspec. Return the schema.',
  { label: 'design-scan-bitexact-sram-npad', phase: 'Design', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','arithmeticCorrect','bv4Grounded','recommendationSound','missedOption','rewardHackCheck','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    arithmeticCorrect: { type: 'string', description: 'is the regs/lane + SRAM arithmetic right (re-derive a couple cells)?' },
    bv4Grounded: { type: 'string', description: 'is the BV=4-preserves-native-K-reduction claim grounded in the actual reduction code + Triton mapping, or asserted? (this is the load-bearing claim)' },
    recommendationSound: { type: 'string', description: 'does the recommended path actually satisfy bit-exact AND no-spill AND >4-node trees?' },
    missedOption: { type: 'string', description: 'any resolution missed' },
    rewardHackCheck: { type: 'string', description: 'is the recommendation build-our-kernel (not a reroute / not shrinking the tree below usefulness)?' },
    recommendation: { type: 'string', description: 'single recommendation for the GPU test order. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the regs/lane arithmetic '
  + 'is wrong or the BV=4-bit-exact claim is asserted rather than grounded in the reduction code + Triton '
  + 'warp/lane mapping (the load-bearing claim - re-raised BV hypotheses have been overstated before). Confirm '
  + 'the recommendation keeps >4-node trees (does not shrink to dodge spill). No close/pass-fail; no reward-hack.',
  { label: 'verify-scan-bitexact-sram-npad', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
