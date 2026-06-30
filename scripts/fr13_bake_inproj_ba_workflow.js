export const meta = {
  name: 'fr13-bake-inproj-ba',
  description: 'BAKE the confirmed in_proj_ba M-invariance fix into the LOCKED cat9 build: flip LUMO_FB_KERNEL_ROWS=1 + LUMO_FB_PROJ_PAD_ROWS=16 in fr13_launch_locked.sh (CPU), then GPU-verify with the standing same-boot lossless gate (same-seed byte-identical det, accept/event preserved, regular-decode pristine) + B=1 SWE-gold flip count + new fingerprint. Hold => keep; fail => revert.',
  phases: [
    { title: 'BakeFlag' },
    { title: 'VerifyGate' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'THE FIX TO BAKE (CONFIRMED, user-approved): the bf16 in_proj_ba GEMM M-keying is the largest single fixable',
'channel of the +17 leaf co-residency. The LUMO_FB pad-to-fixed-M (LUMO_FB_KERNEL_ROWS=1, LUMO_FB_PROJ_PAD_ROWS=16)',
'pads in_proj_ba (+ out_proj) to a tree_n-independent M => M-invariant a/b => removes ~8 flips. Confirmed:',
'same-boot flag-OFF=26 vs flag-ON=18 = -8 (real, not cross-boot); lossless (det [T,T,T,T], CPU lossless check',
'max_abs=0.0, flips DOWN = more lossless); accept/event 3.017 ~ native; default-OFF byte-identical to the',
'locked build (proven, the patcher pad path is gated on LUMO_FB_KERNEL_ROWS==1, commit 9a99bb44 after the',
'8b73be1c class-dedent crash was caught + fixed). The patcher (fr10_phase4_patch_vllm_tree_gdn.py) + the',
'forked launcher env passthrough (e376aec3) are already in HEAD; the ONLY missing step is enabling the flag',
'in the LOCKED launcher.',
'',
'BAKE = set, in scripts/fr13_launch_locked.sh, export LUMO_FB_KERNEL_ROWS=1 and export LUMO_FB_PROJ_PAD_ROWS=16',
'(additive; near the existing BATCH_INVARIANT=0 export). This makes locked cat9 run WITH the in_proj_ba pad by',
'default. The fingerprint WILL change (the fix changes cat9 output) - that is a LOSSLESS change (per the',
'directive BAKE-IN IS FINE; record the new fingerprint, do not gate on reproducing the old one - feedback_',
'no_cross_boot_byte_gate).',
'',
'VERIFY GATE (the standing same-boot lossless gate before trusting the locked path):',
'1. ENGAGEMENT + DETERMINISM (class-9/8): boot the baked locked cat9, assert tok/draft==9 (engaged) AND',
'   same-seed within_boot_det [T,T,T,T] (the in-process same-boot gate, NOT a cross-boot byte reproduction).',
'2. WORKER-ENV: confirm LUMO_FB_KERNEL_ROWS=1 actually reached the EngineCore worker (/proc/<pid>/environ) -',
'   the LUMO_FB vars DID propagate for the ba-proj test (they are passed -e + are read at the right scope),',
'   but re-confirm (the FR13_GDN_SUBOP_MAB env-allowlist lesson). If absent, FAIL LOUD.',
'3. LOSSLESS (the binding per-token argmax): the baked cat9 per-token argmax flip count vs its OWN no-spec',
'   decode oracle (fr13_oracle_stream_teacher_force.py thr 1.0 prompts_swe4) should be ~18 (<= the ~22-26',
'   unbaked, lossless-IMPROVING). Confirm it did NOT go UP (a regression would mean the pad introduced a',
'   defect).',
'4. ACCEPT/EVENT preserved (class-12, report RAW with the cross-boot caveat): ~native (the leaf edge intact);',
'   the fix must NOT collapse accept toward the leaf-free spine 2.66.',
'5. REGULAR-DECODE PRISTINE: the in_proj_ba pad gate is num_spec_decodes>1 (verify path only); confirm a',
'   NON-spec (regular decode) request is byte-unaffected (the fix is spec-path-only).',
'6. B=1 SWE-gold: record the new fingerprint (the [6,6,4,6]-class per-prompt vector) as the new locked baseline.',
'',
'HOLD => keep the bake (commit pushed). FAIL (flips UP / accept collapsed / non-spec changed / disengaged) =>',
'REVERT the locked-launcher flip + report. Reward-hacks BANNED. Quote FR13_BUG_CLASS_PLAYBOOK.md rows.'
].join('\n');

phase('BakeFlag');
const BAKE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lockedLauncherEdit','additiveProof','committed','notes'],
  properties: {
    lockedLauncherEdit: { type: 'string', description: 'the exact lines added to fr13_launch_locked.sh (LUMO_FB_KERNEL_ROWS=1 + PAD_ROWS=16)' },
    additiveProof: { type: 'string', description: 'additive (no existing line changed); the forked launcher already -e passes these' },
    committed: { type: 'string', description: 'the pathspec commit hash' },
    notes: { type: 'string' },
  },
};
const bake = await agent(
  CTX + '\n\nTASK (BakeFlag, no GPU). Add export LUMO_FB_KERNEL_ROWS=1 + export LUMO_FB_PROJ_PAD_ROWS=16 to '
  + 'scripts/fr13_launch_locked.sh (additive). Verify the forked launcher passes them to the container. Commit '
  + 'pathspec (only fr13_launch_locked.sh). Return the schema.',
  { label: 'bake-flag', phase: 'BakeFlag', schema: BAKE_SCHEMA, model: 'opus' }
);

phase('VerifyGate');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['booted','workerEnvConfirmed','engaged','tok_per_draft','within_boot_det','flips','flips_baseline','accept_per_event','regularDecodePristine','newFingerprint','decode_tps','s_per_forward','speedVsOff','ok','notes'],
  properties: {
    booted: { type: 'boolean' },
    workerEnvConfirmed: { type: ['boolean','null'], description: 'LUMO_FB_KERNEL_ROWS=1 in /proc/<EngineCore pid>/environ?' },
    engaged: { type: 'boolean' },
    tok_per_draft: { type: ['number','null'] },
    within_boot_det: { type: 'string' },
    flips: { type: ['integer','null'], description: 'baked cat9 per-token argmax flip count vs its own no-spec oracle (~18 expected, must be <= unbaked ~22)' },
    flips_baseline: { type: 'string', description: 'the unbaked baseline for context (banked 22 / same-boot OFF 26)' },
    accept_per_event: { type: ['number','null'] },
    regularDecodePristine: { type: ['boolean','null'], description: 'is a non-spec regular-decode request byte-unaffected (the fix is spec-path-only)?' },
    newFingerprint: { type: 'string', description: 'the new B=1 SWE-gold per-prompt fingerprint' },
    decode_tps: { type: ['number','null'], description: 'baked cat9 B=1 decode throughput (tokens/s, per-request like E5 NOT aggregate; metrics OFF for the speed read)' },
    s_per_forward: { type: ['number','null'], description: 'baked cat9 B=1 s-per-forward (ms) - the in_proj_ba pad processes 16*row_len rows vs ~33 real; measure if it regresses' },
    speedVsOff: { type: 'string', description: 'baked (LUMO_FB ON) decode TPS / s-per-forward vs the SAME-BOOT or banked flag-OFF cat9 - is the pad speed-neutral (GB10 bandwidth-bound, extra compute hidden behind the weight read) or a regression?' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const verify = await agent(
  CTX + '\n\nTASK (VerifyGate, GPU - the ONLY boot). Bake: ' + JSON.stringify(bake) + '. Hygiene + boot the '
  + 'BAKED locked cat9 (LUMO_FB_KERNEL_ROWS now =1 in fr13_launch_locked.sh). Confirm LUMO_FB_KERNEL_ROWS=1 in '
  + 'the WORKER /proc/<pid>/environ (FAIL LOUD if absent). Run gates 1-6: engagement tok/draft==9 + det '
  + '[T,T,T,T], the per-token argmax flip count (must be <= ~22, expect ~18), accept/event, regular-decode '
  + 'pristine, B=1 fingerprint. PLUS the SPEED measurement (user: measure speed after the fix): in the SAME '
  + 'boot, measure B=1 decode TPS + s-per-forward (per-request basis like E5, NOT aggregate decode_seconds; '
  + 'metrics OFF for the clean speed read per FR10_SPEED_MEASUREMENT_PITFALLS + feedback_flag_gate_metrics_reuse_infra) '
  + 'and compare to the flag-OFF cat9 baseline (same-boot if cheap, else banked) - the in_proj_ba pad runs '
  + '16*row_len rows vs ~33 real, so confirm it is speed-NEUTRAL (GB10 is B=1 bandwidth-bound on the weights, '
  + 'so the extra GEMM compute may be hidden behind the weight read) or quantify the regression. Teardown + '
  + 'recover; never leak. Return the schema (incl decode_tps/s_per_forward/speedVsOff).',
  { label: 'verify-bake', phase: 'VerifyGate', schema: VERIFY_SCHEMA, model: 'opus' }
);

phase('Verdict');
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','bakeKept','losslessConfirmed','reverted','newBaseline','nextAction','issues'],
  properties: {
    holds: { type: 'boolean' },
    bakeKept: { type: ['boolean','null'], description: 'is the bake kept (lossless + accept preserved + non-spec pristine) or reverted?' },
    losslessConfirmed: { type: 'string', description: 'flips down/same (not up), det clean, regular-decode pristine' },
    reverted: { type: ['boolean','null'], description: 'true if the flip was reverted due to a failed gate' },
    newBaseline: { type: 'string', description: 'the new locked cat9 baseline (flips + fingerprint + accept)' },
    nextAction: { type: 'string', description: 'single next action. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verdict = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: Bake ' + JSON.stringify(bake) + ' VerifyGate ' + JSON.stringify(verify) + '. '
  + 'Default holds=false if worker-env unconfirmed / flips went UP / accept collapsed / regular-decode changed '
  + '/ disengaged. If any gate failed, the BakeFlag flip must be REVERTED (commit the revert pathspec) and '
  + 'reverted=true. Else bakeKept=true with the new baseline. No close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }
);

return { bake, verify, verdict };
