export const meta = {
  name: 'fr13-k1-rerun',
  description: 'K1 RE-RUN (the first K1 boot waao62oj0 FAILED non-vacuity, fail-loud: FR13_SCAN_ALIGN=1 absent in worker /proc/environ because it booted via the LOCKED launcher fr13_launch_locked.sh which has NO -e FR13_SCAN_ALIGN docker passthrough; bare FR13_* env is curated out of the mp/spawn EngineCore worker, bug-class #9). FIX = boot via the FORKED launcher fr13_launch_forked_fa2_tree_server.sh (L306-307 pass -e FR13_SCAN_ALIGN / -e FR13_SCAN_ALIGN_MODE = the PROVEN channel that put it in /proc/175/environ for the recompute run) with TREE=cat9 caterpillar + FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body in shell env. The kernel seam is ALREADY committed (b91c1bc0, K1 = per-node bf16 b_h store-reload in _gdn_node_step gated on SCAN_ALIGN, default-OFF byte-identical) so NO Apply phase. Just boot cat9 + K1, PROVE flag-live (bridge-needle /proc/environ) + served-diverges-from-OFF + oracle-engaged, rescore vs the RECURRENT oracle (binding-23 frame), verdict. Does the in-place bf16 store-boundary drop cat9 flips toward native-3 at unchanged accept 3.198, or not (recompute-rose prior holds -> kernel-align dead -> relax). Single GPU boot. Adversarial verify.',
  phases: [
    { title: 'BootRescore' },
    { title: 'Verdict' },
  ],
}

const CAT9_TREE = '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1), (0, 0, 0, 0, 1)]';

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel.',
'GPU SERIALIZED. Pre-boot hygiene: source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps',
'empty. Teardown trap: docker rm -f the container + recover_host_memory after the boot. boot ENFORCE_EAGER=1',
'(hooks eager-only). conv-fused + eager-pack require replay (baked ON in the locked pipeline flags).',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.dev134),',
'NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS before reading any output.',
'',
'COMPARE TARGET (user, MANDATORY): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle. The oracle is',
'the deployment-correct RECURRENT single-step decode (scripts/fr13_recurrent_decode_oracle.py,',
'_forward_core_decode_non_spec). BAR: native-E5 = 3 clear-margin flips [0,0,2,1]; cat9 OFF = 23 [5,4,5,9]',
'(output/fr13_scan_align_rerun/logs/{native_recur_flips.json, off_recur_flips.json}, the binding numbers, SAME',
'oracle frame). int-view NEVER atol. clear-margin = deviation_nat>1.0 gold-margin, NOT streamed top_logprobs.',
'',
'WHY THIS RE-RUN (the first boot was a VACUOUS-instrument fail, correctly caught fail-loud): the K1 boot driver',
'scripts/fr13_k1_boot_capture.sh used LAUNCHER=scripts/fr13_launch_locked.sh, which does NOT pass FR13_SCAN_ALIGN',
'to the container (grep: no -e FR13_SCAN_ALIGN). The bridge-needle found hit_align=[] -> "FLAG NOT LIVE" -> the K1',
'seam never engaged -> no number produced (good fail-loud, playbook #9). The committed kernel seam is CORRECT',
'(b91c1bc0): _gdn_node_step has `if SCAN_ALIGN: state_i = state_i.to(tl.bfloat16).to(tl.float32)` after out_i is',
'taken from the precise fp32 state (the (2) store boundary), default-OFF = dead code = byte-identical locked path.',
'',
'THE PROVEN FLAG-DELIVERY CHANNEL (USE THIS): fr13_launch_forked_fa2_tree_server.sh passes',
'`-e FR13_SCAN_ALIGN="${FR13_SCAN_ALIGN:-0}" -e FR13_SCAN_ALIGN_MODE="${FR13_SCAN_ALIGN_MODE:-body}"` to docker',
'(L306-307) + `-e PYTHONPATH=/workspace/src` (L302). This is the channel that put FR13_SCAN_ALIGN in the',
'recompute run\'s worker /proc/175/environ (FR13_SCAN_NOT_E2E_CARRIER_BIND). The reshape A/B (fr13_reshape_boot_',
'capture.sh) booted cat9-class trees THROUGH this forked launcher successfully (recurrent oracle engaged). So:',
'boot via the FORKED launcher, NOT the locked one.',
'',
'THE INCUMBENT PRIOR (respect + distinguish): recompute (FR13_SCAN_NOT_E2E) made the per-node scan STATE',
'bit-exact to native packed-decode (a STRONGER condition than K1) yet e2e flips ROSE 23->32 (it changed the',
'trajectory: recompute geometry BV32/w1 + ancestry replay -> 369-tok diff, non-lossless). K1 is DIFFERENT: it',
'keeps the EXACT cat9 tree-scan geometry/h_cache + inserts ONLY the per-node bf16 round-trip (byte-deterministic,',
'no path fork). The question: does the store-boundary rounding ALONE move flips toward native-3, or not.',
'',
'YOUR JOB:',
'PHASE 1 (BootRescore, GPU): adapt scripts/fr13_k1_boot_capture.sh (REUSE its non-vacuity gates) but FIX the',
'launcher: set LAUNCHER=scripts/fr13_launch_forked_fa2_tree_server.sh and pass TREE=the cat9 caterpillar',
'  TREE="' + CAT9_TREE + '"',
'in the launch env (the forked launcher derives NUM_SPECULATIVE_TOKENS from TREE), WITH the IDENTICAL locked',
'pipeline flags the reshape boot used (DRAFTER_SINGLE_LOGITS, EAGER_PACK, TREE_CONV_FUSED, TREE_SAMPLE_ROW,',
'REPLAY_ROUTE, FA2_TREE_BIAS, FA2_PREFILL_NATIVE, EXP2_SOFTMAX, CONV_COMMITTED_PATH, LUMO_FB_KERNEL_ROWS=1',
'PAD_ROWS=16, BATCH_INVARIANT=0) + ENFORCE_EAGER=1 + FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body. (Model the boot',
'on scripts/fr13_reshape_boot_capture.sh which already drives this forked launcher correctly; just set TREE=cat9',
'and add the two FR13_SCAN_ALIGN env vars.) temp 0.0 seed 1313 prompts_swe4.',
'NON-VACUITY (mandatory, fail-loud, playbook #9 - the WHOLE point of this re-run):',
'  (i) FLAG LIVE: bridge-needle the worker /proc/<pid>/environ (docker exec, scan all python/Vllm/EngineCore',
'      pids) for FR13_SCAN_ALIGN=1 AND FR13_SCAN_ALIGN_MODE=body. If hit_align=[] -> FAIL LOUD (the bug we are',
'      fixing). Record the matching PID + environ line.',
'  (ii) SERVED DIVERGES FROM OFF: the d+e+K1 body seams perturb the scan (d/e known ~2.86e-6) so the served',
'      stream MUST differ from the banked OFF cat9 stream; if byte-identical to OFF -> the constexpr threaded',
'      dead (bug-class #10) -> FAIL LOUD.',
'  (iii) ORACLE ENGAGED: rescore vs scripts/fr13_recurrent_decode_oracle.py (reuse fr13_recur_rescore_in_',
'      container.sh) -> RECURRENT_PATH_ENGAGED=True + nonzero _forward_core_decode_non_spec; flips GOLD-MARGIN.',
'  (iv) within-boot det [T,T,T,T].',
'Report K1 clear-margin flips raw + per-prompt + de-cascaded (FR13_PLUS2 gap<=2) vs OFF=23/18 + native=3, AND',
'accept/event (K1 keeps cat9 geometry so should stay ~3.198; if it craters, K1 is lossy/not-drop-in). Teardown +',
'recover. Reuse banked OFF/native (SAME oracle frame).',
'PHASE 2 (Verdict). DISCRIMINATOR: (K1 IS THE LEVER) de-cascaded clear-margin flips DROP toward native 3 (vs OFF',
'18) AND accept/event holds ~3.198 -> the in-place store-boundary alignment IS the lossless lever at unchanged',
'speed -> bring the flips+accept table to the user (bake = user call). (K1 DEAD) flips stay ~23 / rise / accept',
'craters -> kernel-align confirmed dead (recompute prior holds), disagreement is topology/trajectory-intrinsic ->',
'the relax-to-accept/event-parity decision stands. Default-OFF locked path byte-unchanged either way. NO close/',
'pass-fail (user call). Reward-hacks BANNED: K1 = OUR kernel adopting (2) rounding (authorized numerics-align);',
'native = A/B oracle ONLY; no copy/dense/forced-spine/multi-spine (CLOSED); recompute is a SEPARATE non-lossless',
'route. Quote FR13_BUG_CLASS_PLAYBOOK rows (#9 vacuous/flag-not-live, #10 constexpr-dead, #12 trajectory).',
].join('\n');

phase('BootRescore');
const BR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['launcherUsed','flagLive','flagLivePid','servedDivergesFromOff','oracleEngaged','within_boot_det','k1_flips_raw','k1_flips_decascaded','k1_per_prompt','k1_accept_per_event','off_baseline','native_bar','ok','notes'],
  properties: {
    launcherUsed: { type: 'string', description: 'CONFIRM the FORKED launcher was used (fr13_launch_forked_fa2_tree_server.sh), not locked' },
    flagLive: { type: ['boolean','null'], description: 'bridge-needle /proc/<pid>/environ: FR13_SCAN_ALIGN=1 + MODE=body PRESENT in worker environ (the bug we fixed)?' },
    flagLivePid: { type: ['string','null'], description: 'the worker PID + environ line proving flag live' },
    servedDivergesFromOff: { type: ['boolean','null'], description: 'served stream DIVERGES from banked OFF (d+e+K1 perturb); byte-identical=constexpr-dead=fail-loud' },
    oracleEngaged: { type: ['boolean','null'], description: 'RECURRENT_PATH_ENGAGED=True + nonzero _forward_core_decode_non_spec' },
    within_boot_det: { type: 'string' },
    k1_flips_raw: { type: ['integer','null'] },
    k1_flips_decascaded: { type: ['integer','string','null'], description: 'K1 de-cascaded clear-margin flips vs OFF=18 / native=3' },
    k1_per_prompt: { type: ['array','string','null'] },
    k1_accept_per_event: { type: ['number','null'], description: 'should stay ~3.198 (geometry unchanged); craters => lossy' },
    off_baseline: { type: 'string' },
    native_bar: { type: 'string' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const br = await agent(
  CTX + '\n\nTASK (BootRescore, GPU). Boot cat9 via the FORKED launcher with FR13_SCAN_ALIGN=1 MODE=body, PROVE '
  + 'flag-live (the fix) + served-diverges-from-OFF + oracle-engaged BEFORE any number. Report K1 flips + '
  + 'accept/event vs OFF=23/18 + native=3. Teardown + recover. Return the schema.',
  { label: 'k1-rerun-boot', phase: 'BootRescore', schema: BR_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','flipsDropped','acceptHeld','k1IsLever','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'flag-live (PROVEN in worker environ this time) + served-diverges + oracle-engaged all confirmed? (the prior boot failed exactly here)' },
    flipsDropped: { type: ['boolean','string','null'], description: 'did de-cascaded clear-margin flips DROP toward native 3 (vs OFF 18)?' },
    acceptHeld: { type: ['boolean','string','null'], description: 'did accept/event hold ~3.198 (drop-in) or crater (lossy)?' },
    k1IsLever: { type: ['boolean','null'], description: 'is the in-place bf16 store-boundary the lossless lever (falsifies recompute incumbent) or dead?' },
    nextAction: { type: 'string', description: 'if K1 lever: bring flips+accept table (user bake call). if dead: relax-to-accept/event-parity stands. No decision here.' },
    rewardHackCheck: { type: 'string', description: 'K1 = our kernel adopting (2) rounding (authorized); native=A/B oracle only; default-OFF byte-unchanged; no splice/copy/dense' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(br) + '. Default holds=false if flag NOT proven live in the '
  + 'WORKER environ this time (the exact failure of the prior boot - require the PID+environ line), served stream '
  + 'NOT proven to diverge from OFF, oracle not engaged, or flips from streamed top_logprobs. Conclude honestly: '
  + 'did the in-place store-boundary alignment drop flips toward native-3 (K1 lever) or not (kernel-align dead, '
  + 'relax stands). Confirm default-OFF byte-unchanged + no reward-hack. No close/pass-fail decision.',
  { label: 'k1-rerun-verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { br, v };
