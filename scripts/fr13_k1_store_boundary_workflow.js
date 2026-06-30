export const meta = {
  name: 'fr13-k1-store-boundary',
  description: 'K1 TEST (the lone residual kernel-align doubt before the relax decision): does adopting native decode-oracle (2)\'s per-token bf16 b_h store-reload into our fp32-carried tree-scan drop cat9 flips toward native-3? Our default _gdn_node_step carries state_i FP32 across the whole spine (no bf16 round-trip between nodes); the deployment recurrent-decode oracle RELOADS state from bf16 cache EVERY token. K1 = add a per-node state_i.to(bf16).to(fp32) round-trip as a NEW body seam under FR13_SCAN_ALIGN MODE=body (combined with existing seam d=l2norm-div + e=beta-round-trip), keeping cat9 geometry EXACTLY (NOT recompute mode, which changed trajectory + rose 23->32). DISTINCT from recompute: in-place, byte-deterministic, drop-in-able. Prior (incumbent): recompute = a STRONGER state-bit-exact form already rose 23->32, so predicted K1<=0 -> kernel-align dead -> relax. Falsify-incumbent: flips drop toward 3 -> K1 IS the systematic lever (one op-order applied 48x to the correlated diffuse floor). Single GPU boot, re-score vs the RECURRENT oracle (binding-23 frame). Default-OFF flag = locked path byte-unchanged. Adversarial verify.',
  phases: [
    { title: 'Apply' },
    { title: 'BootRescore' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next). Repo /home/mark/shared/lumoFlyWheel.',
'GPU SERIALIZED. Pre-boot hygiene EVERY boot: source .venv; recover_host_memory(); assert MemAvailable>=100GiB',
'+ docker ps empty. Teardown trap: docker rm -f the container + recover_host_memory between/after boots.',
'boot ENFORCE_EAGER=1 (hooks are eager-only). conv-fused + eager-pack require replay (baked ON).',
'',
'GROUNDING RULE (user): read vLLM source DIRECTLY via scripts/vllm_src.sh (pinned image 3dbe092e = 0.19.2rc1.',
'dev134), NEVER a /tmp cache. CHECK ARTIFACT TIMESTAMPS before reading any output.',
'',
'COMPARE TARGET (user, MANDATORY): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle. The oracle is',
'the deployment-correct RECURRENT single-step decode (scripts/fr13_recurrent_decode_oracle.py,',
'_forward_core_decode_non_spec), NOT prefill/chunked/streamed-logprobs. BAR: native-E5 = 3 clear-margin flips',
'[0,0,2,1]. cat9 OFF (deployed) = 23 [5,4,5,9] (FR13_SCAN_NOT_E2E_CARRIER_BIND, the binding number). int-view',
'NEVER atol. clear-margin = deviation_nat>1.0 gold-margin (full oracle_topk), NOT streamed top_logprobs.',
'',
'WHY K1 (FR13_REALIZATION_AGREEMENT.md f6fb01b4, verify HOLDS): op-by-op, our default _gdn_node_step',
'(src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py) actually matches the native SPEC-UPDATE kernel (2\',',
'fused_sigmoid_gating: rsqrt l2norm, no beta round-trip, fp32 b_h carried across the T-loop), NOT the',
'packed-decode ORACLE (2, fused_recurrent.py L313-336: div l2norm, beta bf16 round-trip, state LOADED bf16->',
'fp32 + STORED bf16 EVERY token = one program/token). We are SCORED against (2). Of 6 op-level seams, 5',
'(K2 l2norm-div=seam d, K3 beta-round-trip=seam e, K4 gate-order=identical, K5 conv-tap=fixed, K6 scan-tree=',
'topology) have NO depth-growth and are provably ~0 flip-impact. ONLY K1 (the per-token bf16 b_h store-reload',
'(2) realizes) has the depth-growth the diffuse carrier needs. SUBTLETY: our fp32/rsqrt/no-round-trip is MORE',
'precise but DISAGREES with the bf16-store deployment path (2); the fix is to make our kernel LESS precise to',
'MATCH (2) (numerics-alignment = AUTHORIZED, reference_no_reroute_reward_hacking).',
'',
'THE INCUMBENT PRIOR you must respect + distinguish from: recompute (FR13_SCAN_NOT_E2E_CARRIER_BIND) made the',
'per-node scan STATE bit-exact (int-view 0.0) to native packed-decode -- a STRONGER condition than K1 -- and',
'e2e clear-margin flips ROSE 23->32 ([10,9,7,6]) because it CHANGED THE TRAJECTORY (recompute geometry BV32/w1',
'+ ancestry replay -> different LCP-max path -> 369-tok diff, non-lossless). K1 is DIFFERENT: it keeps the',
'EXACT cat9 tree-scan geometry/h_cache and inserts ONLY the per-node bf16 round-trip (byte-deterministic, no',
'path fork). The question this test settles: does the store-boundary rounding ALONE (without the recompute',
'trajectory change) move flips toward native-3, or not.',
'',
'CURRENT SCAN_ALIGN PLUMBING (read fr10_gdn_tree_kernel.py to confirm live line#s - they drift):',
'  scan_align_on() (~L72) reads FR13_SCAN_ALIGN (default OFF = served scan BYTE-IDENTICAL to locked cat9, the',
'  SCAN_ALIGN constexpr threads dead). scan_align_mode() (~L86) reads FR13_SCAN_ALIGN_MODE in {body, recompute}',
'  (default body). _gdn_node_step (~L423) takes SCAN_ALIGN: tl.constexpr; seam e (beta bf16 round-trip) is the',
'  `if SCAN_ALIGN:` block ~L465-466 (b_beta = tl.sigmoid(b_raw_b.to(fp32)).to(bf16).to(fp32)). SCAN_ALIGN is',
'  passed at the call sites (~L1132/L1576/L1898 = scan_align_on()). MODE=recompute geometry route ~L1817.',
'',
'YOUR JOB:',
'PHASE 1 (Apply, no GPU): READ _gdn_node_step + the b_h/state_i update fully (where state_i is carried node-to-',
'  node, where the rank-1 update lands). Add the K1 seam: under the SAME `if SCAN_ALIGN:` body path (MODE=body,',
'  so d+e+K1 are all on together = full (2)-oracle alignment), insert a per-node round-trip of the carried',
'  recurrent state: state_i = state_i.to(tl.bfloat16).to(tl.float32) at the (2) store boundary (AFTER the rank-1',
'  state update, BEFORE it is carried to the next node / read for the output) -- match exactly WHERE (2) stores+',
'  reloads bf16 (read fused_recurrent.py L313-336 via vllm_src.sh to place it correctly: (2) stores b_h.to(bf16)',
'  then the NEXT token reloads it, so the round-trip is on the state CARRIED forward, not the output). Gate it on',
'  the SAME SCAN_ALIGN constexpr (no new env needed; MODE=body already implies the body seams). VERIFY default-',
'  OFF byte-identity: when FR13_SCAN_ALIGN unset, the constexpr is False -> the K1 round-trip is threaded DEAD ->',
'  the locked cat9 path is byte-unchanged (this is the reward-hack-clean invariant; the committed kernel must',
'  stay byte-identical on the default path). Commit pathspec (the single kernel file) to main, default-OFF.',
'PHASE 2 (BootRescore, GPU): hygiene + boot cat9 EAGER, FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=body, temp 0.0',
'  seed 1313, the SAME 4 prompts_swe4. Capture served streams. Then re-score every served position vs',
'  scripts/fr13_recurrent_decode_oracle.py (the binding-23 frame): clear-margin flips (deviation_nat>1.0) raw +',
'  per-prompt + de-cascaded (gap<=2 FR13_PLUS2 rule); AND accept/event (K1 keeps cat9 geometry so accept should',
'  stay ~3.15 - if it CRATERS, K1 is lossy/not-drop-in, report it). NON-VACUITY (mandatory, 4 instruments burned',
'  this session, playbook #9): (i) FLAG LIVE - bridge-needle the worker /proc/<pid>/environ for FR13_SCAN_ALIGN=1',
'  AND confirm the SCAN_ALIGN constexpr is TRUE in the compiled kernel (the d+e seams are KNOWN to perturb state',
'  ~2.86e-6, so the served stream MUST DIVERGE from the OFF stream; if byte-identical to OFF -> constexpr threaded',
'  dead per bug-class #10 -> fail loud); (ii) ORACLE ENGAGED - RECURRENT_PATH_ENGAGED=True + nonzero',
'  _forward_core_decode_non_spec on all arms; (iii) within-boot det [T,T,T,T]. Reuse banked OFF=23 + native-E5=3',
'  from output/fr13_scan_align_rerun/logs/ (SAME oracle, comparable) for the baseline/bar.',
'PHASE 3 (Verdict). DISCRIMINATOR: (K1 IS THE LEVER) de-cascaded clear-margin flips DROP toward native 3 (vs',
'  OFF 18-23) AND accept/event holds ~3.15 -> the kernel-align lever LIVES (the diffuse floor IS a single',
'  systematic store-boundary op-order applied 48x); bring the flips+accept table to the user (bake = user call).',
'  (K1 DEAD - incumbent) flips stay ~23 or rise / accept craters -> kernel-align confirmed dead (recompute prior',
'  holds), the disagreement is topology/trajectory-intrinsic -> the relax-to-accept/event-parity decision stands.',
'  Either way the default-OFF locked path is byte-unchanged. NO close/pass-fail decision here (user\'s call).',
'  Reward-hacks BANNED: K1 is OUR kernel adopting (2)\'s rounding (authorized numerics-alignment); native = A/B',
'  oracle ONLY (no served-path splice); no copy-recurrent multi-spine (CLOSED) / dense / forced-spine; recompute',
'  is a SEPARATE non-lossless route (do not conflate). Quote FR13_BUG_CLASS_PLAYBOOK rows (#9 vacuous, #10',
'  codegen-identity/constexpr-dead, #12 trajectory).',
].join('\n');

phase('Apply');
const AP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['nodeStepRead','k1InsertionPoint','defaultOffByteIdentical','committed','notes'],
  properties: {
    nodeStepRead: { type: 'string', description: 'where state_i is carried node-to-node + the rank-1 update + where (2) stores/reloads bf16 (fused_recurrent.py L313-336 via vllm_src.sh)' },
    k1InsertionPoint: { type: 'string', description: 'the exact line the K1 state_i.to(bf16).to(fp32) round-trip was inserted, under the SCAN_ALIGN constexpr body path' },
    defaultOffByteIdentical: { type: ['boolean','string'], description: 'PROVEN: FR13_SCAN_ALIGN unset -> constexpr False -> K1 threaded dead -> locked path byte-unchanged (how verified)' },
    committed: { type: 'string', description: 'pathspec commit hash (single kernel file, default-OFF)' },
    notes: { type: 'string' },
  },
};
const ap = await agent(
  CTX + '\n\nTASK (Apply, no GPU). Add the K1 body seam, prove default-OFF byte-identity, commit pathspec. '
  + 'Return the schema.',
  { label: 'k1-apply', phase: 'Apply', schema: AP_SCHEMA, model: 'opus' }
);

phase('BootRescore');
const BR_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['flagLive','servedDivergesFromOff','oracleEngaged','within_boot_det','k1_flips_raw','k1_flips_decascaded','k1_per_prompt','k1_accept_per_event','off_baseline','native_bar','ok','notes'],
  properties: {
    flagLive: { type: ['boolean','null'], description: 'bridge-needle /proc/<pid>/environ FR13_SCAN_ALIGN=1 + SCAN_ALIGN constexpr TRUE in compiled kernel' },
    servedDivergesFromOff: { type: ['boolean','null'], description: 'served stream DIVERGES from OFF (d+e+K1 perturb); byte-identical=constexpr-dead=fail-loud' },
    oracleEngaged: { type: ['boolean','null'], description: 'RECURRENT_PATH_ENGAGED=True + nonzero _forward_core_decode_non_spec all arms' },
    within_boot_det: { type: 'string' },
    k1_flips_raw: { type: ['integer','null'] },
    k1_flips_decascaded: { type: ['integer','string','null'], description: 'K1 de-cascaded clear-margin flips vs OFF=18-23 / native=3' },
    k1_per_prompt: { type: ['array','string','null'] },
    k1_accept_per_event: { type: ['number','null'], description: 'should stay ~3.15 (geometry unchanged); craters => lossy' },
    off_baseline: { type: 'string' },
    native_bar: { type: 'string' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const br = await agent(
  CTX + '\n\nTASK (BootRescore, GPU). Boot cat9 EAGER with FR13_SCAN_ALIGN=1 MODE=body, rescore vs the recurrent '
  + 'oracle. PROVE flag-live + served-diverges-from-OFF + oracle-engaged BEFORE any number. Report K1 flips '
  + '(raw+de-cascaded+per-prompt) + accept/event vs OFF=23/18 + native=3. Teardown + recover. Return the schema.',
  { label: 'k1-boot-rescore', phase: 'BootRescore', schema: BR_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','nonVacuous','flipsDropped','acceptHeld','k1IsLever','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    nonVacuous: { type: 'string', description: 'flag-live (constexpr TRUE) + served-diverges + oracle-engaged all proven? not another vacuous instrument?' },
    flipsDropped: { type: ['boolean','string','null'], description: 'did de-cascaded clear-margin flips DROP toward native 3 (vs OFF 18-23)?' },
    acceptHeld: { type: ['boolean','string','null'], description: 'did accept/event hold ~3.15 (K1 drop-in) or crater (lossy)?' },
    k1IsLever: { type: ['boolean','null'], description: 'is the per-token bf16 store-boundary the lossless lever (falsifies incumbent) or dead (recompute prior holds, topology-intrinsic)?' },
    nextAction: { type: 'string', description: 'if K1 lever: bring flips+accept table (user bake call). if dead: relax-to-accept/event-parity stands - bring close/pass-fail. No decision here.' },
    rewardHackCheck: { type: 'string', description: 'K1 = our kernel adopting (2) rounding (authorized); native=A/B oracle only; default-OFF byte-unchanged; no splice/copy/dense' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(br) + ' (apply: ' + JSON.stringify(ap) + '). Default '
  + 'holds=false if flag not proven live (constexpr could be threaded dead = vacuous #10), served stream NOT '
  + 'proven to diverge from OFF, oracle not engaged, or flips from streamed top_logprobs. Conclude honestly: did '
  + 'the in-place store-boundary alignment drop flips toward native-3 (K1 lever, falsifies the recompute '
  + 'incumbent) or not (kernel-align dead, relax stands). Confirm default-OFF locked path byte-unchanged + no '
  + 'reward-hack. No close/pass-fail decision.',
  { label: 'k1-verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { ap, br, v };
