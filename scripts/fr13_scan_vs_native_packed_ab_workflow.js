export const meta = {
  name: 'fr13-scan-vs-native-packed-ab',
  description: 'THE DECISIVE carrier test (converged from conv-investigate + SRAM EXIT-1, both verify-HOLDS). Carrier hunt eliminated both cross-event STATE channels (SSM-recurrent + conv = closed/flat); carrier = within-forward DEPTH-SCALED GDN scan state-feed (chunk-vs-recurrent). BUT the scan was NEVER tested vs native PACKED-decode SASS - only vs native_update_serial_per_path (serial torch ref, bug-class #10). Build the int-view A/B: deployed BV16/w8 scan out_i vs native fused_recurrent_gated_delta_rule_packed_decode (the kernel the no-spec oracle dispatches), + arms BV32/w4 (native geom), at N_PAD=1 AND 16, spine + branch winner, with rel_err + powered neg-control. MATCH ⇒ scan faithful to incumbent (the chunk-vs-recurrent gap was vs the WRONG ref; 21-vs-oracle is the dispatch reframe + SRAM dissolves). DIVERGE ⇒ FIXABLE per-forward carrier (align: recompute-from-spine / fp32-accum / op-order / l2norm / raw-g). Anti-premature-no-go.',
  phases: [
    { title: 'Build' },
    { title: 'Boot' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'WHY THIS IS THE DECISIVE TEST: the 21-flip carrier hunt has ELIMINATED every cross-event STATE channel -',
'FA2/TREE_ATTN full-attn (FA2-fork is the deployed decode kernel at its 0.0039 lossless floor), SSM-recurrent',
'durable-state (replay A/B w2vaqcsmx: grows=FALSE, L0 4.17 = ring-gather artifact), conv-state (verify HOLDS:',
'FIXED+CLOSED, 3-tap FIFO cannot accumulate). Both conv-investigate AND the SRAM-tension workflow converge:',
'the carrier is the WITHIN-FORWARD, DEPTH-SCALED GDN recurrent SCAN STATE-FEED realization gap (our rank-1',
'tree-scan vs the chunked-prefill/recurrent realization of the same logical GDN state; chunk-vs-recurrent ~1',
'bf16-ULP born at L0, amplified ~32x by gate 1/rms over L41-L63). The back-loading is DEPTH-correlated (longer',
'accepted chains later in the stream), NOT event-count.',
'',
'THE GAP THIS CLOSES (bug-class #10, codegen-identity not spec-guaranteed): the deployed scan _tree_gdn_kernel',
'(src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py) was only ever bit-exact-checked vs',
'native_update_serial_per_path - a SERIAL TORCH reference (scripts/fr13_gdn_scan_warp_gate.py:43,120, banked',
'0.0) - NEVER vs native\'s ACTUAL Triton kernel that the no-spec DECODE oracle dispatches:',
'fused_recurrent_gated_delta_rule_packed_decode (vllm fla ops; the SRAM verify named it as the kernel live',
'verify/decode actually dispatches). bit-exact-to-a-serial-ref != bit-exact-to-the-incumbent-SASS. So "scan',
'ruled out / per-forward kernels HOLD" rests on the WRONG reference. AND the conv-doc\'s "diffuse within-floor,',
'not a single fixable seam" pessimism rests on comparing to the chunked-PREFILL realization - also not the',
'decode incumbent. THIS TEST uses the RIGHT reference.',
'',
'YOUR JOB:',
'PHASE 1 (Build, no GPU): FIRST CONFIRM THE REFERENCE (load-bearing, do NOT trust the name - the FA2 naming-',
'slip + the serial-ref trap are this exact class): read the native GDN decode dispatch in the live image',
'(/tmp/vllm_live_019/.../fla/ops/) + the Qwen3-Next GDN layer forward - what kernel does NATIVE no-spec B=1',
'decode ACTUALLY call for the GDN recurrent update (fused_recurrent_gated_delta_rule_packed_decode? or another',
'fused_recurrent variant)? Use THAT as the reference. THEN extend scripts/fr13_gdn_scan_warp_gate.py: add a',
'reference arm = the ACTUAL native decode kernel (captured/run SAME payload), int-view comparison',
'(a.view(int32).eq(b.view(int32)).all(), NEVER atol) + raw max_abs + rel_err=max_abs/(||native||_inf+eps) +',
'norm_ours/norm_native + first-mismatch idx. Add a TEST-ONLY BV/num_warps/num_stages override param to',
'launch_tree_gdn_prepared (fr10_gdn_tree_kernel.py) - FLAG-GATED, default path BYTE-IDENTICAL (PROVE it: the',
'locked cat9 scan launch unchanged when the override env is unset). Arms: {BV16/w8 DEPLOYED, BV32/w4 native-',
'geom, BV8/w8, BV8/w4} all vs the native decode kernel at N_PAD=1 AND 16, on the SPINE and a BRANCH winner',
'([0,2],[0,1,4]; branch ref = native-on-path-to-root oracle, reference_gdn_tree_branch_oracle_losslessness).',
'POWERED NEGATIVE CONTROL (bug-class #9): a deliberately-perturbed arm that MUST int-view-mismatch, so an',
'all-match is not vacuous. CPU wiring tests + AST-parse. Commit pathspec (gate + kernel + tests). ready=false',
'with the blocker if the native decode reference cannot be invoked on the captured payload.',
'PHASE 2 (Boot, GPU - the ONLY boot): hygiene; run the extended gate against the captured payload',
'(FR10_TREE_GDN_CAPTURE_PAYLOAD; capture a fresh paired payload from a quick cat9 eager forward if the banked',
'one is stale/absent - PIN its prompt, reference_capture_once_native_pin_prompt). Record per-arm int32-equality',
'+ raw max_abs + rel_err + first-mismatch + n_regs/n_spills (ptxas) at N_PAD 1 AND 16, spine + branch, +',
'negative_control_powered=true. Teardown + recover; never leak.',
'DISCRIMINATOR: (1) deployed BV16/w8 INT-VIEW == native-decode at BOTH N_PAD + spine&branch => the scan is',
'FAITHFUL to the incumbent SASS. The chunk-vs-recurrent "carrier" was measured vs the WRONG reference; the',
'21-vs-no-spec-oracle gap is then the EXPECTED tree-verify-vs-sequential-decode numerical-dispatch gap (a',
'close/pass-fail REFRAME for the user, NOT a kernel bug) AND the SRAM tension dissolves (ship deployed as-is).',
'(2) BV16/w8 != native-decode but BV32/w4 == => GEOMETRY is the seam => recompute-from-spine @BV32/w4 (SRAM',
'EXIT-2, bit-exact-by-construction + spill-free). (3) NO arm == native-decode => the divergence is kernel MATH',
'=> align the named sub-levers (fp32 state accumulation / op-order / l2norm / raw-g) - FIXABLE, not irreducible',
'(feedback_research_before_deadend: this REFUTES the "diffuse within-floor irreducible" framing). Reward-hacks',
'BANNED: native kernel is the A/B ORACLE only (observe), the override is test-only geometry not a value change,',
'do NOT splice native into the served path. Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-identity, #9',
'silent/vacuous, #12 co-residency).'
].join('\n');

phase('Build');
const B_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['nativeRefKernelConfirmed','gateExtension','overrideDefaultOffProof','negControl','cpuTests','committed','ready','notes'],
  properties: {
    nativeRefKernelConfirmed: { type: 'string', description: 'the ACTUAL native GDN decode kernel confirmed from the live dispatch (name + file:line), NOT trusted from a name' },
    gateExtension: { type: 'string', description: 'how the gate now runs the native-decode reference + int-view + rel_err + the 4 BV/warps arms at N_PAD 1+16, spine+branch' },
    overrideDefaultOffProof: { type: 'string', description: 'the test-only BV/warps override is flag-gated; locked cat9 scan launch BYTE-IDENTICAL when unset' },
    negControl: { type: 'string', description: 'the powered negative control that must int-view-mismatch' },
    cpuTests: { type: 'string' },
    committed: { type: 'string' },
    ready: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const b = await agent(
  CTX + '\n\nTASK (Build, no GPU). Do PHASE 1. Commit pathspec. Return the schema. ready=false with the blocker '
  + 'if the native decode reference cannot be invoked on the captured payload.',
  { label: 'build-scan-vs-native-packed', phase: 'Build', schema: B_SCHEMA, model: 'opus' }
);

phase('Boot');
const BOOT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['booted','negControlPowered','deployed_bv16w8_intview_eq','bv32w4_intview_eq','results_table','first_mismatch','rel_err_deployed','n_spills','ok','notes'],
  properties: {
    booted: { type: 'boolean' },
    negControlPowered: { type: ['boolean','null'], description: 'did the negative-control arm int-view-MISMATCH (proving non-vacuity)?' },
    deployed_bv16w8_intview_eq: { type: ['boolean','string','null'], description: 'deployed BV16/w8 == native-decode INT-VIEW at N_PAD 1 AND 16, spine AND branch?' },
    bv32w4_intview_eq: { type: ['boolean','string','null'], description: 'native-geom BV32/w4 == native-decode INT-VIEW?' },
    results_table: { type: 'string', description: 'per-arm x N_PAD x spine/branch: int32-equal + raw max_abs + rel_err' },
    first_mismatch: { type: ['string','null'] },
    rel_err_deployed: { type: ['number','string','null'] },
    n_spills: { type: ['string','null'], description: 'ptxas n_spills per arm (BV16/w8 vs BV32/w4)' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
let boot = null;
if (b && b.ready) {
  boot = await agent(
    CTX + '\n\nTASK (Boot, GPU). Build: ' + JSON.stringify(b) + '. Hygiene + run the extended gate against the '
    + 'captured payload (fresh paired capture if needed, PIN the prompt). Record the per-arm int-view table + '
    + 'rel_err + first-mismatch + n_spills + negControlPowered. Teardown + recover. Return the schema.',
    { label: 'boot-scan-vs-native-packed', phase: 'Boot', schema: BOOT_SCHEMA, model: 'opus' }
  );
} else {
  log('Build not ready (native decode reference not invocable) -> SKIP boot; Verdict reports the blocker.');
}

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','scanFaithfulToIncumbent','branch','fixableOrReframe','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    scanFaithfulToIncumbent: { type: ['boolean','null'], description: 'is deployed BV16/w8 INT-VIEW == native-decode (faithful) or not?' },
    branch: { type: 'string', description: 'which discriminator branch: (1) faithful→reframe+SRAM-dissolve, (2) geometry-seam→recompute-from-spine, (3) kernel-math→align sub-levers' },
    fixableOrReframe: { type: 'string', description: 'is the carrier a FIXABLE seam (align named lever) or does the result REFRAME the 21 as the expected dispatch gap (close/pass-fail to user)?' },
    nextAction: { type: 'string', description: 'single next action per the branch. If branch (1) reframe => this is a close/pass-fail (ask user). No close decision here.' },
    rewardHackCheck: { type: 'string', description: 'native kernel = A/B oracle only; override = test-only geometry; no splice into served path' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Build ' + JSON.stringify(b) + ' Boot ' + JSON.stringify(boot) + '. Default '
  + 'holds=false if negControlPowered is not true (vacuous), the native-decode reference is the wrong kernel '
  + '(re-check the dispatch), or int-view is replaced by atol. Conclude the branch (faithful-reframe / geometry-'
  + 'seam / kernel-math-align). If branch (1) faithful, the 21-vs-oracle is the dispatch reframe = a close/pass-'
  + 'fail to bring to the user, do NOT declare it closed here. No reward-hack.',
  { label: 'verdict-scan-vs-native-packed', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { b, boot, v };
