export const meta = {
  name: 'fr13-extend-measure-infra',
  description: 'USER convention (extend the canonical fr13_measure infra, never hand-roll): close the 2 gaps the GPU-validation flagged on the just-built infra (572b623e/e2c82ab5). GAP-1 temp06-drift TV is VACUOUS as-shipped (fr13_measure keys q by token-STRING but the recurrent oracle p by token-ID -> n_positions_scored=0, all id_string_mismatch; the 0.738 number was an out-of-band re-key = a key-misalignment ARTIFACT not a lossless signal). FIX: emit q keyed by token-ID at capture time (served_token_ids already recorded) + capture the recurrent p over the FULL served stream (currently top-K on flip positions only) -> a REAL per-position TV(softmax(q/0.6),softmax(p/0.6)) cat9 vs native, each-vs-own-oracle, vs the native floor. GAP-2 accept/event is TRAJECTORY-BOUND + forks cross-boot (the GB10 token-6 greedy fork = autotune floor; oracle proved the gold 3.161 trajectory correct by 11 nats), so a free-running cross-boot cat9-vs-native accept is NOT apple-to-apple. FIX: add a PAIRED TEACHER-FORCED accept mode to fr13_measure - force BOTH cat9 and native onto a COMMON reference trajectory (the no-spec recurrent oracle greedy stream) + measure accept/event per arm on that fixed trajectory = the fork-immune verification-efficiency edge (document deployment-accept free-running vs paired-accept apples-to-apple). Both EXTEND fr13_measure.py (one canonical module). Phase1 = extend (CPU). Phase2 = USE THE GPU to re-validate (real temp06-drift TV + paired accept, reconcile). Phase3 = adversarial verify. Output FR13_MEASURE_INFRA_GAPS_CLOSED.md.',
  phases: [
    { title: 'Extend' },
    { title: 'GPUValidate' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, B=1 decode HBM-bound). Repo /home/mark/shared/lumoFlyWheel. Read code',
'+ git history + vLLM source via scripts/vllm_src.sh (pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec',
'commit (git commit -m ".." -- <files>, NEVER git add -A). GPU SERIALIZED + FREE; this wf is the only GPU user.',
'MAX 2 concurrent workflows.',
'',
'THE CANONICAL INFRA (extend it, do NOT fork a new probe - feedback_canonical_speed_lossless_infra): scripts/',
'fr13_measure.py (subcommands speed(OFF)/capture-q(ON)/temp06-drift/bag-tv/diag-residue/reconcile, canonical',
'regime baked: prompts_swe4 seed1313 max128, /v1/completions raw, one raw self-warm, MAX_NUM_SEQS=1) + scripts/',
'fr13_measure_orchestrate.sh (serialized boots: native fr10_launch_speed_server num_spec=N, cat9 fr13_launch_',
'locked, TREE fr13_launch_forked_fa2_tree_server; prelaunch recover_host_memory+assert>=95GiB+docker-empty +',
'teardown). FR13_SPEED_MEASURE_INFRA.md + FR13_MEASURE_RECONCILE_GPU.md document it. SPEED reproduces (native s/',
'fwd 0.21732 vs banked 0.21816, cat9 0.22636 vs 0.2248, B-invariant); accept forks cross-boot (trajectory-bound,',
'bound to served_stream_fingerprint, each-arm-vs-own-oracle never cross-boot).',
'',
'GAP-1 (temp06-drift TV vacuous): fr13_measure temp06-drift compares q (token-STRING keys from the /v1/completions',
'top_logprobs) against the recurrent oracle p (token-ID keys) with NO tokenizer re-key -> n_positions_scored=0,',
'all positions align_status=id_string_mismatch (native_e5_temp06_drift.json). The 0.738/p95 0.9998 in *_idkeyed',
'.json was an OUT-OF-BAND re-key (q top-20 STRINGS vs p top-20 IDs only partially overlap = an artifact, NOT a',
'lossless verdict). FIX: (a) capture-q must emit q keyed by TOKEN-ID (the served_token_ids are already recorded;',
'use the served-position true token-id + the top-K candidate token-ids, not detok strings - request the',
'completions logprobs with token ids, or re-key via the SAME tokenizer the server uses, vllm_src.sh); (b) the',
'recurrent oracle p must be captured over the FULL served stream top-20 (not flip positions only). Then temp06-',
'drift reduces a REAL per-position TV(softmax(q/0.6),softmax(p/0.6)) + KL + over-floor vector, cat9 vs native each',
'-vs-OWN-oracle, paired with the per-position view (reference_scalar_metric_per_token_blindspot), vs the native',
'temp-0.6 floor. Record per-position truncated tail mass as the error bar.',
'',
'GAP-2 (paired teacher-forced accept): free-running accept/event FORKS cross-boot at the token-6 GB10 near-tie',
'(autotune floor, feedback_no_cross_boot_byte_gate; the no-spec oracle ranks the gold token argmax by 11 nats =',
'gold trajectory is correct, our boot landed the other side). So a cross-boot cat9-vs-native accept is NOT apple-',
'to-apple (bug-class #12). FIX: add a PAIRED TEACHER-FORCED accept mode to fr13_measure: pick ONE reference',
'trajectory (the no-spec RECURRENT oracle greedy stream = the deployment-correct ground truth) and force BOTH',
'cat9 AND native to verify that SAME fixed token sequence, counting accept/event per arm = the fork-immune',
'verification-efficiency edge (cat9 superset vs native linear on identical content). DOCUMENT the distinction:',
'paired-accept = apples-to-apple structural edge (use for the break-even); deployment-accept = free-running,',
'trajectory-variable (the floor). Reuse the per-event committer/spec counters; bind to the reference trajectory',
'fingerprint. Each arm scored on the SAME forced tokens. (If true teacher-forcing of the spec verify is infra-',
'heavy, the minimum is: anchor both arms to the oracle stream + report accept conditioned on landing on it.)',
].join('\n');

phase('Extend');
const E_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['gap1Fix','gap2Fix','codeChanged','committed','notes'],
  properties: {
    gap1Fix: { type: 'string', description: 'the temp06-drift q-by-token-id fix: how capture-q now emits id-keyed q + full-stream recurrent p, so the TV reduce aligns by id (not string) and scores all positions' },
    gap2Fix: { type: 'string', description: 'the paired-teacher-forced accept mode added to fr13_measure: how both arms are forced onto the common oracle reference trajectory + accept/event measured per arm = fork-immune; the deployment-vs-paired distinction documented' },
    codeChanged: { type: 'string', description: 'the exact fr13_measure.py / orchestrate.sh changes (extend, not fork) + how to invoke; CPU-validated bits' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const e = await agent(
  BASE + '\n\nTASK (Extend, CPU, no GPU yet). Close GAP-1 (id-keyed q + full-stream p -> real TV) and GAP-2 '
  + '(paired teacher-forced accept) by EXTENDING fr13_measure.py (+ orchestrate.sh). CPU-validate the reducers on '
  + 'banked/synthetic data. Write FR13_MEASURE_INFRA_GAPS_CLOSED.md, commit pathspec. Return the schema.',
  { label: 'extend-measure-infra', phase: 'Extend', schema: E_SCHEMA, model: 'opus' }
);

phase('GPUValidate');
const G_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['temp06DriftReal','pairedAccept','committed','notes'],
  properties: {
    temp06DriftReal: { type: 'string', description: 'GPU: re-run capture-q (id-keyed) + full-stream recurrent p on cat9 + native, the REAL per-position TV(softmax(q/0.6),softmax(p/0.6)) - n_positions_scored>0 (not the 0/vacuum), cat9 mean/p95 vs native floor each-vs-own-oracle; is it non-vacuous + interpretable now?' },
    pairedAccept: { type: 'string', description: 'GPU: paired teacher-forced accept - cat9 vs native on the COMMON oracle trajectory, accept/event per arm (the fork-immune edge), vs the forked free-running numbers; does cat9 show a real apples-to-apple accept edge?' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const g = await agent(
  BASE + '\n\nTASK (GPUValidate - USE THE GPU, serialized boots, prelaunch recover_host_memory+assert>=95GiB+'
  + 'docker-empty per boot + teardown). The infra was extended: ' + JSON.stringify(e) + '\nRe-run the temp06-drift '
  + '(id-keyed q + full p -> REAL TV, not the n=0 vacuum) + the paired teacher-forced accept (cat9 vs native on the '
  + 'common oracle trajectory). Commit the results. Return the schema.',
  { label: 'gpu-validate-gaps', phase: 'GPUValidate', schema: G_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','temp06NonVacuous','pairedAccountSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    temp06NonVacuous: { type: 'string', description: 'is the temp06-drift TV now REAL (id-aligned, n_positions_scored>0, not the string/id key-misalignment artifact) + interpretable vs the native floor?' },
    pairedAccountSound: { type: 'string', description: 'is the paired teacher-forced accept genuinely fork-immune (both arms on the SAME forced tokens, apples-to-apple) + the deployment-vs-paired distinction honest?' },
    recommendation: { type: 'string', description: 'single: are both gaps closed so the infra now drives the campaign (speed + accept + temp-0.6 lossless) trustworthily? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify({ extend: e, gpu: g }) + '. Default holds=false if the '
  + 'temp06-drift TV is still the string/id key-misalignment artifact (n_positions_scored must be >0 with id-'
  + 'aligned q,p, not the 0.738 out-of-band re-key), if the paired accept is not truly on a COMMON forced '
  + 'trajectory for both arms (still cross-boot/free-running), or if a fix forked a new probe instead of extending '
  + 'fr13_measure. research-before-deadend. No close/pass-fail; no reward-hack.',
  { label: 'verify-gaps-closed', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { e, g, v };
