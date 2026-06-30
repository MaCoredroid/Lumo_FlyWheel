export const meta = {
  name: 'fr13-scan-align-verify-boot',
  description: 'PHASE-2 GPU VERIFY of the committed scan-alignment (5e56b7aa) - the prior fix-verify ran Apply only (set ready=false, skipped the boot). The alignment is in HEAD: FR13_SCAN_ALIGN flag (l2norm rsqrt→1/sqrt + beta bf16-cast seams in _gdn_node_step) + _tree_gdn_recompute_kernel (recompute-from-spine, native BV32/w1/s3, drop-in, spill-free, removes co-residency) via FR13_SCAN_ALIGN_MODE=recompute, + the int-view gate (fr13_gdn_scan_warp_gate.py vs native-packed, powered neg-control). Boot eager + measure: (a) int-view aligned-scan == native packed-decode @N_PAD 1+16 spine+branch (which seam closes it), (b) THE BINDING e2e flips US-vs-no-spec-oracle drop 21→toward native 3, (c) lossless gate + default-OFF binary byte-identical.',
  phases: [
    { title: 'Verify' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory. boot ENFORCE_EAGER=1 (the gate/hooks are eager-only).',
'',
'GROUNDING RULE (user, MANDATORY): read vLLM source DIRECTLY from the pinned running image via',
'`scripts/vllm_src.sh <relpath>` (vllm/vllm-openai@sha256:3dbe092e = 0.19.2rc1.dev134), NEVER a /tmp cache.',
'COMPARE TARGET (user, MANDATORY, feedback_fr13_lossless_compare_target): lossless = US(cat9) vs native-E5',
'each-vs-its-own-no-spec-oracle; native-E5 pays ~3 flips = THE BAR; US pays 21 = a real gap (NOT a free',
'dispatch pass). Binding instrument = fr13_gold_margin_probe.py per-token argmax vs the no-spec oracle',
'(teacher-force THIS boot stream, NOT banked; oracle=no-spec NOT prefill). int-view NEVER atol for kernel gates.',
'',
'WHAT IS ALREADY COMMITTED (5e56b7aa, verify holds=True, do NOT re-apply): in',
'src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py - FR13_SCAN_ALIGN constexpr (default False, DCE-d off =',
'byte-id locked path) threading 2 body seams in _gdn_node_step: (d) l2norm `b_q/b_k / tl.sqrt(sum+1e-6)` (on)',
'vs `* tl.rsqrt(...)` (off); (e) beta `tl.sigmoid(b_raw_b.to(fp32)).to(bf16).to(fp32)` (on) vs plain fp32',
'(off). PLUS _tree_gdn_recompute_kernel (recompute-from-spine: no h_cache, one [BLOCK_V,DIM_K] fp32 tile,',
'each output node replays path-to-root from h0 via tl.where(strict_mask) in topological order; native',
'BV32/num_warps=1/num_stages=3 = _NATIVE_PACKED_* constants; signature-identical drop-in for the served scan;',
'spill-free + removes leaf co-residency) dispatched when FR13_SCAN_ALIGN=1 + FR13_SCAN_ALIGN_MODE=recompute.',
'The fp32->bf16 carry seam (c) was deliberately NOT forced (ours more precise; bar = within-floor per-depth',
'argmax NOT abs-0.0). scripts/fr13_gdn_scan_warp_gate.py = int-view gate vs native PACKED-decode reference',
'(native_packed_decode_per_path, branch ref = native-on-path-to-root) + POWERED neg-control (root +0.5 must',
'int-view-mismatch). The geom-override (FR13_TREE_GDN_GEOM_OVERRIDE=BV32/w1/s3) is value-neutral DIAGNOSTIC',
'only - it SPILLS ~2048 reg/lane at N_PAD=16 so is NOT deployable; recompute-from-spine is the deployable',
'geometry fix. Reward-hacks BANNED: native packed-decode = A/B ORACLE only, the seams/recompute are OUR',
'kernel (no served-path native call - already verified in Apply).',
'',
'YOUR JOB:',
'PHASE 1 (Verify, GPU - the ONLY boot, ENFORCE_EAGER=1): hygiene + boot cat9.',
'(0) DEFAULT-OFF BINARY BYTE CHECK (the AST/DCE claim, now on GPU): with FR13_SCAN_ALIGN UNSET, confirm the',
'    deployed-arm scan output is byte-identical to the locked baseline (the gate deployed arm / a same-boot',
'    OFF capture) - the in-process byte check that the off path is unchanged.',
'(A) INT-VIEW TRY-ORDER via scripts/fr13_gdn_scan_warp_gate.py vs native PACKED-decode (int-view NEVER atol,',
'    + rel_err + first-mismatch + neg-control-powered): (1) geom pre-test BV32/w1/s3 @ N_PAD=1 (one node, no',
'    spill) - is geometry the seam? (2) body seams (l2norm then beta) @ N_PAD=1 AND 16, SPINE + BRANCH',
'    ([0,2],[0,1,4]); (3) recompute-from-spine @ N_PAD=16. RECORD which makes the scan INT-VIEW == native-packed',
'    (0.0) and the rel_err of each. NOTE caveat: recompute is bit-exact to native PER-PATH so it may differ',
'    from our own h_cache scan by the -0.0->+0.0 handoff bit - the gate compares vs NATIVE-PACKED (correct).',
'(B) THE BINDING e2e (the whole point): boot cat9 with the DEPLOYABLE alignment ON (FR13_SCAN_ALIGN=1',
'    FR13_SCAN_ALIGN_MODE=recompute), run fr13_gold_margin_probe.py per-token argmax US-vs-its-own-no-spec-',
'    oracle on prompts_swe4 (teacher-force THIS boot) - does the flip count DROP from 21 toward native 3? ALSO',
'    the lossless gate: same-seed within-boot det [T,T,T,T], regular-decode (non-spec) byte-pristine,',
'    accept/event preserved (~native 3.0-3.15, must NOT collapse). If the body-seams route (MODE unset, just',
'    FR13_SCAN_ALIGN=1) is cheaper/also-bit-exact, measure its flips too for comparison.',
'    Teardown + recover; never leak.',
'PHASE 2 (Verdict). DISCRIMINATOR: flips 21 -> ~3 (within E5 floor) AND int-view 0.0 AND lossless-gate holds',
'=> THE FIX (the carrier WAS the codegen scan state-feed; report the deployable seam + the new flip count =',
'the LOSSLESS WIN, bake is the user call). flips drop partway => quantify which seam closed how much + the',
'residual (e.g. co-residency vs op-order). flips unchanged but int-view 0.0 => the scan was NOT the e2e carrier',
'(re-open the hunt). Quote FR13_BUG_CLASS_PLAYBOOK rows (#10 codegen-identity, #12 depth/co-residency, #9',
'silent/vacuous).'
].join('\n');

phase('Verify');
const VR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['booted','eager','defaultOff_byteIdentical','negControlPowered','geom_intview_npad1','body_intview','recompute_intview','winningSeam','flips_before','flips_after_recompute','flips_after_body','within_boot_det','regularDecodePristine','accept_per_event','ok','notes'],
  properties: {
    booted: { type: 'boolean' }, eager: { type: ['boolean','null'] },
    defaultOff_byteIdentical: { type: ['boolean','null'], description: 'FR13_SCAN_ALIGN unset => deployed scan byte-identical to locked baseline (GPU binary check of the AST claim)' },
    negControlPowered: { type: ['boolean','null'] },
    geom_intview_npad1: { type: ['string','null'], description: 'geom BV32/w1/s3 int-view vs native-packed @N_PAD=1: == or rel_err' },
    body_intview: { type: ['string','null'], description: 'after l2norm+beta: int-view @N_PAD 1+16 spine+branch' },
    recompute_intview: { type: ['string','null'], description: 'recompute-from-spine int-view vs native-packed @N_PAD 1+16 spine+branch' },
    winningSeam: { type: ['string','null'], description: 'which alignment drove int-view to 0.0 (geom/body/recompute), or none' },
    flips_before: { type: ['integer','null'], description: 'baseline cat9 flips vs no-spec oracle (~21)' },
    flips_after_recompute: { type: ['integer','null'], description: 'flips with FR13_SCAN_ALIGN=1 MODE=recompute (the deployable route) - THE BINDING NUMBER (native bar 3)' },
    flips_after_body: { type: ['integer','null'], description: 'flips with body-seams-only (if measured)' },
    within_boot_det: { type: 'string' },
    regularDecodePristine: { type: ['boolean','null'] },
    accept_per_event: { type: ['number','null'] },
    ok: { type: 'boolean' }, notes: { type: 'string' },
  },
};
const vr = await agent(
  CTX + '\n\nTASK (Verify, GPU - the ONLY boot). Do PHASE 1 steps 0/A/B on the COMMITTED alignment (5e56b7aa, '
  + 'do NOT re-apply). Report defaultOff_byteIdentical + the int-view try-order + flips_before/flips_after + '
  + 'the lossless gate. Teardown + recover. Return the schema.',
  { label: 'verify-scan-align-boot', phase: 'Verify', schema: VR_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','carrierClosed','flips_after','deployableSeam','losslessGateHeld','coResidencyResidual','isLosslessWin','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    carrierClosed: { type: ['boolean','null'], description: 'flips → ~3 (within E5 floor) WITH the lossless gate holding?' },
    flips_after: { type: ['integer','null'], description: 'the binding flip count with the deployable fix ON' },
    deployableSeam: { type: 'string', description: 'the deployable winning alignment (recompute-from-spine / body seams) + int-view result' },
    losslessGateHeld: { type: 'string', description: 'default-off byte-id + det + regular-decode pristine + accept preserved?' },
    coResidencyResidual: { type: 'string', description: 'if flips dropped but not to 3: op-order vs co-residency split (recompute removes co-residency - did it?)' },
    isLosslessWin: { type: ['boolean','null'], description: 'flips~3 + int-view 0.0 + gate held = the lossless win to report to the user?' },
    nextAction: { type: 'string', description: 'if win: report to user, bake = user call → then speed → B=1. if partway: the residual. if unchanged: scan re-opened. No close decision here.' },
    rewardHackCheck: { type: 'string', description: 'native = oracle only; seams/recompute are OUR kernel; no served-path splice' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(vr) + '. Default holds=false if negControlPowered!=true '
  + '(vacuous), atol used instead of int-view, the ref is not native packed-decode, defaultOff is NOT byte-'
  + 'identical (the locked path moved = a bug), or the e2e flip gate was not run on THIS boot stream. The '
  + 'BINDING result is flips_after vs the native bar of 3 WITH the lossless gate holding. If flips~3 + gate '
  + 'held, this is the LOSSLESS WIN to report to the user (carrier fixed); bake/close is the user call. No '
  + 'reward-hack.',
  { label: 'verdict-scan-align-boot', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { vr, v };
