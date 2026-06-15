export const meta = {
  name: 'fr13-npad-invariant-test',
  description: 'N_PAD-INVARIANT REDUCTION-ORDER TEST (user approved 2026-06-15). The math-rounding research (FR13_MATH_ROUNDING_NOFLIP, verify HOLDS) found the GDN co-residency "leak" = adding leaves moves N_PAD 4->8, RECOMPILING tl.static_range(0,N_PAD) + the tl.where reduction so the SAME spine nodes get a different FMA/accumulation ORDER (measured state gap 0.0289) = a pure rounding-order effect at all ~48 GDN layers. The apple-to-apple investigation (verify HOLDS) confirmed the committer is CLEAN and the genuine residual = R=8 confident verify-vs-decode flips = this diffuse GDN seam. This is UNTESTED in pure form: the earlier recompute "refutation" (flips rose 23->32) was DOUBLY CONFOUNDED (changed geometry to BV32/w1/s3 AND still looped N_PAD so never canonicalized order AND re-rolled trajectory). FIX = a constexpr FR13_NPAD_INVARIANT making the scan loop bound + offs_n lane count a FIXED N_FIXED for ALL tree sizes (masked by existing <N_ACTUAL), num_warps PINNED to cat9-deployed=8 (NOT recompute BV32), default-OFF byte-identical. Test = 1 GPU boot, K1 ON. GATE 1 (mechanism): byte-A/B spine node states cat9(leaves-on) vs leaf-free spine, int-view equality (NEVER atol) -> PASS = N_PAD seam closed + geometry held (recompute confound excluded). GATE 2 (verdict): re-score served cat9 vs the NATIVE RECURRENT oracle, de-cascaded clear flips + accept/event vs OFF=18/K1=12/native=3. G1 PASS+G2 drops toward native (holding accept) = lossless+fast no-copy SHIP; G1 PASS+G2 flat/rises = de-confounded NON-causal = irreducibly diffuse -> relax. Adversarial verify.',
  phases: [
    { title: 'Apply' },
    { title: 'BootGate' },
    { title: 'Verdict' },
  ],
}

const CAT9_TREE = '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]';
const CHAIN5_TREE = '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0)]';

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel. GPU',
'SERIALIZED. Pre-boot hygiene EVERY boot: source .venv; recover_host_memory(); assert MemAvailable>=100GiB +',
'docker ps empty. Teardown trap: docker rm -f + recover_host_memory after each boot. boot ENFORCE_EAGER=1.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS. int-view NEVER atol (#10).',
'',
'COMPARE TARGET (user, MANDATORY): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle. Oracle =',
'deployment-correct RECURRENT single-step decode (scripts/fr13_recurrent_decode_oracle.py). BAR native-E5=3',
'[0,0,2,1]; cat9 OFF=18 de-cascaded; cat9+K1=12 de-cascaded (banked output/fr13_scan_align_rerun/logs/ +',
'FR13_K1_STORE_BOUNDARY_BIND). clear-margin = deviation_nat>1.0 gold-margin. de-cascade = FR13_PLUS2 gap<=2.',
'',
'THE FIX (the math lead, FR13_MATH_ROUNDING_NOFLIP.md, verify HOLDS): in src/lumo_flywheel_serving/fr10_gdn_',
'tree_kernel.py _tree_gdn_kernel, the scan reduction order DEPENDS on N_PAD = 1<<(n-1).bit_length() (~L159-163):',
'the loop `for i in tl.static_range(0,N_PAD)` (~L582) + the parent reduction `tl.sum(tl.where((offs_n==j),',
'h_cache,0.0),axis=0)` (~L586-590) with `offs_n=tl.arange(0,N_PAD)` (~L550) all range over N_PAD lanes, so a',
'larger tree (more leaves) recompiles them to a different FMA/reduction tree and the SAME spine nodes get a',
'different rounding order (bug-class #10 codegen-identity; MEASURED state gap 0.0289). FIX = make the order',
'N_PAD-INVARIANT: a constexpr FR13_NPAD_INVARIANT that sets the loop bound + offs_n lane count to a FIXED',
'N_FIXED (= the deployed cat9 N_PAD = 1<<(9-1).bit_length() = 16) for ALL tree sizes, keeping the EXISTING',
'`<N_ACTUAL` masks so inactive lanes contribute exact 0.0 (tl.where already does this). num_warps PINNED to the',
'cat9-deployed value (8) - do NOT inherit the refuted recompute geometry (BV32/w1/s3, that was the confound).',
'Default-OFF (FR13_NPAD_INVARIANT unset) => constexpr threads dead => byte-identical served path (#10).',
'',
'WHY THIS IS NOT THE REFUTED RECOMPUTE: recompute (FR13_SCAN_NOT_E2E, flips rose 23->32) was DOUBLY CONFOUNDED -',
'it changed geometry to native BV32/w1/s3 (kernel comment ~L708-717) AND still looped tl.static_range(0,N_PAD)',
'(~L765/771) so it NEVER canonicalized the N_PAD order, AND re-rolled the LCP trajectory (369 tok diffs). This',
'test changes ONE constexpr on the DEPLOYED kernel at the SAME geometry (num_warps pinned), no new kernel, no',
'trajectory re-roll => de-confounded.',
'',
'YOUR JOB:',
'PHASE 1 (Apply, no GPU): READ _tree_gdn_kernel (the N_PAD def ~L159-163, the scan loop ~L582, the offs_n/',
'  reduction ~L550/L586-590) + ALL call sites. Add the FR13_NPAD_INVARIANT constexpr (a flag fn like scan_align_',
'  on(), reads FR13_NPAD_INVARIANT) that, when ON, replaces the N_PAD loop bound + offs_n lane count with the',
'  fixed N_FIXED=16 for the scan, KEEPING the <N_ACTUAL masks, num_warps PINNED to 8 (the deployed value, NOT',
'  recompute geometry). VERIFY default-OFF byte-identity (constexpr dead => locked path byte-unchanged, #10). Add',
'  `-e FR13_NPAD_INVARIANT` to the FORKED launcher scripts/fr13_launch_forked_fa2_tree_server.sh (the proven',
'  365da33b SCAN_ALIGN-style worker passthrough; the locked launcher does NOT forward it). Commit pathspec',
'  (kernel + launcher), default-OFF.',
'PHASE 2 (BootGate, GPU): hygiene + boot cat9 via the FORKED launcher (TREE=' + CAT9_TREE + ', locked pipeline',
'  flags) WITH FR13_NPAD_INVARIANT=1 + FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body (K1 ON), ENFORCE_EAGER=1,',
'  temp 0.0 seed 1313 prompts_swe4. NON-VACUITY (mandatory, fail-loud, #9): (i) flags LIVE - bridge-needle',
'  worker /proc/<pid>/environ for FR13_NPAD_INVARIANT=1 AND FR13_SCAN_ALIGN=1 (fail loud if absent); (ii)',
'  RECURRENT_PATH_ENGAGED=True on the rescore; (iii) within-boot det.',
'  GATE 1 (MECHANISM - the de-confound proof): capture the GDN SPINE node states (spine nodes 0,1,3,5,7) for',
'    cat9 (leaves ON, N_PAD would be 16) and for a LEAF-FREE spine tree chain5 (TREE=' + CHAIN5_TREE + ', N_PAD',
'    would be 8) - both with FR13_NPAD_INVARIANT=1 so both use N_FIXED=16 - and INT-VIEW compare the spine',
'    states (reuse scripts/fr13_gdn_scan_warp_gate.py methodology, hardened neg-control both-norms>0, NEVER',
'    atol). PASS = int-view 0.0 (the leaves no longer perturb the spine reduction order = N_PAD seam closed,',
'    geometry held). Also capture WITHOUT the flag (default) to show the gap is 0.0289 (neg-control powered).',
'  GATE 2 (VERDICT): re-score the served cat9 (NPAD-inv + K1) stream vs scripts/fr13_recurrent_decode_oracle.py:',
'    clear-margin flips raw + per-prompt + de-cascaded (FR13_PLUS2) + accept/event, vs OFF=18 / K1=12 /',
'    native=3. Teardown + recover after every boot.',
'PHASE 3 (Verdict). DECISION (do NOT bake): G1 PASS (spine states int-view 0.0 with flag, 0.0289 without =',
'  mechanism proven, geometry held) AND G2 de-cascaded flips DROP toward native (<=~7) holding accept/event',
'  ~3.0 => the N_PAD-invariant reduction order + K1 is the lossless+fast NO-COPY/NO-HBM math fix - bring the',
'  flips+accept table to the user (bake = user call). G1 PASS but G2 flat/rises => the rounding-order seam is',
'  closed yet NON-CAUSAL (the de-confounded recompute verdict) => the residual is irreducibly diffuse',
'  amplification => relax to accept/event-parity. G1 FAIL (states not 0.0 with flag) => the constexpr did not',
'  canonicalize the order (spill->num_warps or a missed lane) - report + do not trust G2. Default-OFF locked',
'  path byte-unchanged either way. Reward-hacks BANNED: this is OUR kernel reduction-order canonicalization',
'  (compute-only, no copy/HBM, geometry HELD); native = A/B oracle only; NOT recompute (geometry change), NOT',
'  WY (parked), NOT spine-bonus (user rejected). Quote FR13_BUG_CLASS_PLAYBOOK (#9 vacuous, #10 codegen/order,',
'  #12 trajectory).',
].join('\n');

phase('Apply');
const AP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['kernelRead','constexprInsertion','numWarpsPinned','defaultOffByteIdentical','launcherPassthrough','committed','notes'],
  properties: {
    kernelRead: { type: 'string', description: 'the N_PAD def + scan loop + offs_n/reduction lines read; where the order depends on N_PAD' },
    constexprInsertion: { type: 'string', description: 'where FR13_NPAD_INVARIANT sets loop bound + offs_n to N_FIXED=16, masks kept <N_ACTUAL' },
    numWarpsPinned: { type: ['boolean','string'], description: 'num_warps pinned to deployed 8 (NOT recompute BV32) - geometry held' },
    defaultOffByteIdentical: { type: ['boolean','string'], description: 'PROVEN default-OFF constexpr-dead byte-identical' },
    launcherPassthrough: { type: 'string', description: '-e FR13_NPAD_INVARIANT added to forked launcher' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const ap = await agent(
  CTX + '\n\nTASK (Apply, no GPU). Add FR13_NPAD_INVARIANT constexpr (fixed N_FIXED=16 reduction order, masks '
  + 'kept, num_warps pinned 8), prove default-OFF byte-identity, add forked-launcher passthrough, commit '
  + 'pathspec. Return the schema.',
  { label: 'npad-apply', phase: 'Apply', schema: AP_SCHEMA, model: 'opus' }
);

phase('BootGate');
const BG_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['flagsLive','oracleEngaged','within_boot_det','gate1_spine_intview_flagOn','gate1_spine_intview_flagOff','gate1_pass','npad_flips_raw','npad_flips_decascaded','npad_per_prompt','npad_accept_per_event','baselines','ok','notes'],
  properties: {
    flagsLive: { type: ['boolean','null'], description: 'FR13_NPAD_INVARIANT=1 + FR13_SCAN_ALIGN=1 in worker /proc/environ' },
    oracleEngaged: { type: ['boolean','null'] },
    within_boot_det: { type: 'string' },
    gate1_spine_intview_flagOn: { type: ['string','null'], description: 'cat9 vs chain5 spine states int-view WITH flag (expect 0.0)' },
    gate1_spine_intview_flagOff: { type: ['string','null'], description: 'WITHOUT flag (expect ~0.0289, neg-control powered)' },
    gate1_pass: { type: ['boolean','null'], description: 'spine states int-view 0.0 with flag (seam closed, geometry held)?' },
    npad_flips_raw: { type: ['integer','null'] },
    npad_flips_decascaded: { type: ['integer','string','null'], description: 'NPAD+K1 de-cascaded flips vs OFF=18/K1=12/native=3' },
    npad_per_prompt: { type: ['array','string','null'] },
    npad_accept_per_event: { type: ['number','null'] },
    baselines: { type: 'string', description: 'OFF=18/K1=12/native=3 reused' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const bg = await agent(
  CTX + '\n\nTASK (BootGate, GPU). Boot cat9 NPAD-inv + K1, PROVE flags-live + oracle-engaged. GATE 1 = spine '
  + 'states int-view cat9 vs chain5 (flag ON expect 0.0, OFF expect 0.0289). GATE 2 = e2e flips + accept/event '
  + 'vs OFF=18/K1=12/native=3. Teardown + recover. Return the schema.',
  { label: 'npad-boot-gate', phase: 'BootGate', schema: BG_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','gate1Mechanism','gate2Verdict','causalOrDiffuse','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'flags-live + oracle-engaged + GATE1 neg-control powered (0.0289 without flag) all proven?' },
    gate1Mechanism: { type: 'string', description: 'did GATE 1 PASS (spine int-view 0.0 with flag, 0.0289 without) = seam closed + geometry held = recompute confound excluded?' },
    gate2Verdict: { type: 'string', description: 'GATE 2 de-cascaded flips vs OFF 18 / K1 12 / native 3, accept/event' },
    causalOrDiffuse: { type: ['string','null'], description: 'G1 PASS+G2 drops = N_PAD-order CAUSAL (ship); G1 PASS+G2 flat/rises = de-confounded NON-causal = irreducibly diffuse (relax)' },
    nextAction: { type: 'string', description: 'if causal: bring flips+accept table (user bake). if diffuse: relax to accept/event-parity. No decision here.' },
    rewardHackCheck: { type: 'string', description: 'OUR kernel order-canonicalization, geometry HELD (num_warps 8 not BV32), no copy/HBM; native=oracle only; NOT recompute/WY/bonus; default-OFF byte-unchanged' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(bg) + ' (apply: ' + JSON.stringify(ap) + '). Default '
  + 'holds=false if flags not proven live, GATE 1 neg-control not powered (must show 0.0289 WITHOUT the flag '
  + 'else the int-view is vacuous), GATE 1 used atol not int-view, num_warps NOT held at 8 (geometry confound '
  + 'reintroduced = it becomes recompute), or flips from streamed logprobs. Conclude honestly: G1-pass+G2-drop = '
  + 'N_PAD-order causal (the math fix works); G1-pass+G2-flat = de-confounded non-causal = diffuse (relax). '
  + 'Confirm default-OFF byte-unchanged + no reward-hack (not recompute/WY/bonus). No close/pass-fail decision.',
  { label: 'npad-verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { ap, bg, v };
