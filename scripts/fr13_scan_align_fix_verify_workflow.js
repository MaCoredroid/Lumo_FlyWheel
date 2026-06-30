export const meta = {
  name: 'fr13-scan-align-fix-verify',
  description: 'FIX + VERIFY (user): native packed-decode is RECURRENT rank-1 = SAME algorithm as our scan (re-verified on the REAL image), so the carrier is CODEGEN-alignable. Apply the alignment (try-order: geom BV32/w1/s3 pre-test → l2norm rsqrt→1/sqrt + beta bf16-cast body seams → recompute-from-spine if geometry-seam+spill) to make OUR scan bit-exact to native packed-decode, then VERIFY (a) int-view scan==native-packed at N_PAD=1 AND 16 spine+branch, AND (b) the BINDING e2e per-token argmax flips US-vs-its-no-spec-oracle drop 21→toward native 3. Flag-gated default-OFF byte-identical. build-our-kernel NOT splice.',
  phases: [
    { title: 'Apply' },
    { title: 'Verify' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'GROUNDING RULE (user, MANDATORY): read vLLM source DIRECTLY from the pinned running image via',
'`scripts/vllm_src.sh <relpath>` (cats fresh from vllm/vllm-openai@sha256:3dbe092e = cu130-nightly =',
'0.19.2rc1.dev134). NEVER read /tmp/vllm_live_019 or any /tmp cache - they DRIFT (the stale-0.19.0 confound).',
'Re-ground every native-kernel line citation against the image; the prior scan-math (FR13_SCAN_ALIGNMENT_MATH.md)',
'cited stale 0.19.0 line numbers - its CONCLUSIONS hold (re-verified on the image: packed-decode IS recurrent',
'rank-1, num_warps=1) but RE-CONFIRM each edit target on the live image.',
'',
'COMPARE TARGET (user, MANDATORY - feedback_fr13_lossless_compare_target): lossless = US (cat9) vs native-E5',
'measured each-vs-its-own-no-spec-decode oracle. native-E5 pays ~3 flips for its own spec+dispatch = THE BAR.',
'US currently pays 21. The binding e2e instrument = fr13_gold_margin_probe.py per-token argmax US-vs-no-spec-',
'oracle, teacher-forced on THIS boot stream (NOT banked). int-view NEVER atol for kernel bit-exact gates.',
'',
'THE FIX (FR13_SCAN_ALIGNMENT_MATH.md, w0n91rty5 verify HOLDS): native fused_recurrent_gated_delta_rule_packed_',
'decode_kernel (vllm_src.sh model_executor/layers/fla/ops/fused_recurrent.py) is recurrent rank-1, byte-for-byte',
'the same 5 ops as our _gdn_node_step (src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py). The divergence is',
'CODEGEN, all build-our-kernel (no native call in the served path - native is the A/B ORACLE only;',
'feedback_no_reroute_reward_hacking). Seams + try-order:',
'(1) GEOMETRY [the likely dominant seam]: deployed BV16/w8 vs native BV32/num_warps=1/num_stages=3 repartitions',
'    the 128-lane K-reduce. FREE value-neutral pre-test: a flag-gated launch override BV=32/w1/s3 on the',
'    UNCHANGED _tree_gdn_kernel, int-view vs native-packed at N_PAD=1 (one node, no h_cache spill). If it',
'    matches -> geometry IS the seam. DEPLOYABLE form = RECOMPUTE-FROM-SPINE @ BV32/w1/s3 (NOT the geom-override',
'    on the full tree: at N_PAD=16 BV32/w1 h_cache = ~2048 regs/lane = catastrophic spill). Recompute-from-spine',
'    = drop h_cache, hold ONE [32,128] fp32 tile, replay ancestry via the existing tl.where(strict_mask) on the',
'    shared _gdn_node_step (the existing _tree_gdn_replay_kernel :588 is already no-h_cache/one-tile - reuse its',
'    structure); bit-exact-by-construction + spill-free + ALSO removes co-residency (each node from spine,',
'    independent) which is the +9-13 leaf-co-residency carrier. This is SRAM EXIT-2, re-pinned to w1.',
'(2) BODY SEAM d (l2norm): ours tl.rsqrt(sum+1e-6) vs native /tl.sqrt(sum+1e-6) (eps MATCHES) - 1-line opcode',
'    swap on b_q and b_k. (3) BODY SEAM e (beta): ours tl.sigmoid(b_raw_b.to(fp32)) vs native append',
'    .to(bf16).to(fp32) - 1-line cast. (4) DO NOT force the fp32-carry->bf16 seam (c): ours is MORE precise;',
'    the bar is per-depth argmax within-floor, NOT abs-0.0 (user 2026-06-09). Softplus/gate already 0.0.',
'',
'YOUR JOB:',
'PHASE 1 (Apply, no GPU): re-ground citations via vllm_src.sh. Implement the alignment FLAG-GATED (FR13_SCAN_',
'ALIGN, default-OFF): the geom-override (diagnostic, value-neutral), the 2 body seams (d,e), AND the recompute-',
'from-spine variant (the deployable geometry fix, reusing _tree_gdn_replay_kernel structure at native w1/s3).',
'PROVE FR13_SCAN_ALIGN-OFF is BYTE-IDENTICAL to the locked cat9 (locked path unchanged until proven). Wire the',
'int-view scan-vs-native-packed gate (reuse/extend scripts/fr13_gdn_scan_warp_gate.py with the native PACKED-',
'decode reference run SAME-payload + a POWERED negative control, bug-class #9). CPU wiring tests + AST. Commit',
'pathspec. ready=false with the blocker if recompute-from-spine cannot reuse the replay scaffold cleanly (then',
'do the body-seams-only fix + flag recompute as a follow-up).',
'PHASE 2 (Verify, GPU eager - the ONLY boot): hygiene; ENFORCE_EAGER=1. (A) int-view try-order: geom pre-test',
'(BV32/w1/s3 @ N_PAD=1) -> body seams -> recompute-from-spine; record which makes the deployed scan INT-VIEW ==',
'native-packed at N_PAD=1 AND 16, SPINE and BRANCH winner ([0,2],[0,1,4]; branch ref = native-on-path-to-root),',
'+ rel_err + first-mismatch + neg-control-powered. (B) the BINDING e2e: boot cat9 with FR13_SCAN_ALIGN=1, run',
'fr13_gold_margin_probe.py per-token argmax US-vs-its-no-spec-oracle on prompts_swe4 (teacher-force THIS boot) -',
'does the flip count DROP from 21 toward native 3? + same-seed within-boot det [T,T,T,T] + regular-decode',
'pristine + accept/event preserved (must NOT collapse). Teardown + recover.',
'DISCRIMINATOR: flips 21 -> ~3 (within native band) AND int-view 0.0 AND lossless-gate holds => THE FIX',
'(the carrier WAS the codegen scan state-feed; align it = lossless). flips drop partway => quantify which seam',
'closed how much + the residual (next front). flips unchanged but int-view 0.0 => the scan was NOT the e2e',
'carrier after all (re-open). Reward-hacks BANNED (native = oracle only; the override/recompute are OUR kernel;',
'no served-path native call). Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-identity, #12 depth/co-residency,',
'#9 silent/vacuous).'
].join('\n');

phase('Apply');
const A_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['citationsRegrounded','seamsImplemented','recomputeFromSpine','defaultOffProof','intViewGateWired','cpuTests','committed','ready','notes'],
  properties: {
    citationsRegrounded: { type: 'string', description: 'native kernel edit targets re-confirmed on the IMAGE via vllm_src.sh (file:line on the running 0.19.2)' },
    seamsImplemented: { type: 'string', description: 'the flag-gated geom-override + body seams (d l2norm, e beta) implemented in _gdn_node_step/launch' },
    recomputeFromSpine: { type: ['string','null'], description: 'the recompute-from-spine variant (reusing _tree_gdn_replay_kernel @ w1/s3) or the blocker + body-seams-only fallback' },
    defaultOffProof: { type: 'string', description: 'FR13_SCAN_ALIGN-OFF BYTE-IDENTICAL to locked cat9' },
    intViewGateWired: { type: 'string', description: 'scan-vs-native-PACKED-decode int-view gate + powered neg-control wired (the right reference, not serial)' },
    cpuTests: { type: 'string' },
    committed: { type: 'string' },
    ready: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const a = await agent(
  CTX + '\n\nTASK (Apply, no GPU). Do PHASE 1. Commit pathspec. Return the schema.',
  { label: 'apply-scan-align', phase: 'Apply', schema: A_SCHEMA, model: 'opus' }
);

phase('Verify');
const Vf_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['booted','eager','negControlPowered','intview_eq_which_seam','intview_table','flips_aligned','flips_baseline','within_boot_det','regularDecodePristine','accept_per_event','ok','notes'],
  properties: {
    booted: { type: 'boolean' }, eager: { type: ['boolean','null'] },
    negControlPowered: { type: ['boolean','null'] },
    intview_eq_which_seam: { type: ['string','null'], description: 'which seam (geom / body d+e / recompute-from-spine) makes the deployed scan INT-VIEW == native-packed' },
    intview_table: { type: ['string','null'], description: 'per-seam x N_PAD(1,16) x spine/branch: int32-equal + rel_err' },
    flips_aligned: { type: ['integer','null'], description: 'the BINDING e2e per-token argmax flip count US-vs-no-spec-oracle with FR13_SCAN_ALIGN=1 (target ~3)' },
    flips_baseline: { type: 'string', description: 'the OFF/locked baseline (21) for context' },
    within_boot_det: { type: 'string' },
    regularDecodePristine: { type: ['boolean','null'] },
    accept_per_event: { type: ['number','null'], description: 'must NOT collapse (~native 3.07)' },
    ok: { type: 'boolean' }, notes: { type: 'string' },
  },
};
let vf = null;
if (a && a.ready) {
  vf = await agent(
    CTX + '\n\nTASK (Verify, GPU). Apply: ' + JSON.stringify(a) + '. Hygiene + boot ENFORCE_EAGER=1. Run (A) the '
    + 'int-view try-order then (B) the BINDING e2e flips US-vs-no-spec-oracle with FR13_SCAN_ALIGN=1 + lossless '
    + 'gate. Teardown + recover. Return the schema.',
    { label: 'verify-scan-align', phase: 'Verify', schema: Vf_SCHEMA, model: 'opus' }
  );
} else {
  log('Apply not ready -> SKIP boot; Verdict reports the blocker.');
}

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','intViewBitExact','flipsDroveToNative','whichSeam','isTheFix','deployable','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    intViewBitExact: { type: ['boolean','null'], description: 'did a seam make the scan int-view == native-packed (0.0)?' },
    flipsDroveToNative: { type: ['boolean','string','null'], description: 'did the BINDING e2e flips drop 21 -> ~3 (native band)?' },
    whichSeam: { type: ['string','null'], description: 'geom / body d+e / recompute-from-spine' },
    isTheFix: { type: ['boolean','null'], description: 'flips~3 + int-view 0.0 + lossless-gate held = THE carrier fix?' },
    deployable: { type: 'string', description: 'is the winning seam deployable (recompute-from-spine spill-free / body-seam baked) or geom-override-spills?' },
    nextAction: { type: 'string', description: 'if flips~3: bake + proceed to speed/B=4 (the endgame). if partway: the residual seam. if unchanged@int-view-0.0: scan re-opened. No close decision without user.' },
    rewardHackCheck: { type: 'string', description: 'native = A/B oracle only; the fix is OUR kernel; no served-path splice' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Apply ' + JSON.stringify(a) + ' Verify ' + JSON.stringify(vf) + '. Default '
  + 'holds=false if neg-control not powered (vacuous), atol used instead of int-view, the reference is not the '
  + 'native PACKED-decode kernel, or FR13_SCAN_ALIGN-OFF is not byte-identical. Conclude: is this THE fix '
  + '(flips~3 + int-view 0.0 + lossless held), partway, or scan-re-opened. If flips~3, that is a major step to '
  + 'the lossless goal but bake/close is the user call. No reward-hack (build-our-kernel, native oracle-only).',
  { label: 'verdict-scan-align', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { a, vf, v };
