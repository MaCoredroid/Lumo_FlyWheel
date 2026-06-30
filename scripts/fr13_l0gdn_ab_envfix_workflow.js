export const meta = {
  name: 'fr13-l0gdn-ab-envfix-rerun',
  description: 'Re-run the L0-GDN sub-op A/B after fixing the env-propagation (FR13_GDN_SUBOP_MAB must reach the EngineCore worker via VLLM_RAY_EXTRA_ENV_VARS_TO_COPY) with a HARD worker-env gate (/proc/<pid>/environ) BEFORE capture so it cannot be vacuous again. Decisive discriminator for the +13 residual: deep-spine conv1d_out/scan_out M10-vs-M5 nonzero (paddable carrier) vs ~0 (depth-intrinsic, BI exhausted at in_proj_ba). Also the FIRST real exercise of Front B bounds-guard fix 8cdda4c4.',
  phases: [
    { title: 'BootCapture' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory.',
'',
'PRIOR FAILURE (task wc0gyx2za, root-caused): the L0-GDN sub-op A/B (FR13_GDN_SUBOP_MAB) booted cat9 fine',
'(engaged tok/draft=9, det [T,T,T,T]) but the hook fired 0 times => 0 records, residual UNANSWERED. ROOT:',
'the GDN forward runs in the EngineCore WORKER process (pid ~176 = VLLM::EngineCore), and the bare',
'FR13_GDN_SUBOP_MAB master switch is ABSENT from that worker environ. vLLM ray-worker spawn',
'(vllm/ray/ray_env.py get_env_vars_to_copy) copies ONLY vllm.envs registered vars + DEFAULT_ENV_VAR_PREFIXES',
'={VLLM_} (+ VLLM_RAY_EXTRA_ENV_VARS_TO_COPY). FR13_* is neither VLLM_-prefixed nor registered => stripped.',
'/proc/176/environ confirmed: 14/66 FR13 vars reached the worker; the SUBOP siblings _LAYER/_LIMIT/_SKIP got',
'through but the bare master switch FR13_GDN_SUBOP_MAB + _DUMP/_EXPECT_TREE_N/_THRESHOLD did NOT. The gate',
'reads os.environ.get(FR13_GDN_SUBOP_MAB,0)==1 INSIDE the worker forward (gdn_linear_attn.py conv-site +',
'verify-call), so both sites are inert. (This is NOT the prior CUDA-assert; that boot DID engage the hook',
'pre-fix. Front B bounds-guard fix 8cdda4c4/d30755c8 is in HEAD but was NEVER exercised - this re-run is its',
'first real test: watch for a clean Python bounds-raise (fail-loud) vs a recurrence of the CUDA assert.)',
'',
'THE FIX (do this, default-safe, additive): make the master switch + its siblings reach the EngineCore',
'worker. CLEANEST = set VLLM_RAY_EXTRA_ENV_VARS_TO_COPY to include FR13_GDN_SUBOP_MAB,FR13_GDN_SUBOP_MAB_DUMP,',
'FR13_GDN_SUBOP_MAB_EXPECT_TREE_N,FR13_GDN_SUBOP_MAB_THRESHOLD (a comma-list) AND -e pass it in the forked',
'launcher (scripts/fr13_launch_forked_fa2_tree_server.sh, additive, only when FR13_GDN_SUBOP_MAB is set; do',
'NOT change the locked default path). Read ray_env.py to confirm the exact var name + format. (Fallback if',
'VLLM_RAY_EXTRA_ENV_VARS_TO_COPY does not exist in this vLLM version: front the gate read with a VLLM_-prefixed',
'alias, or set the vars at patcher import-time the way the 14 propagated FR13 vars got through.) Commit the',
'launcher change pathspec.',
'',
'HARD WORKER-ENV GATE (the lesson from the failure - class-9): BEFORE running the capture, find the EngineCore',
'worker pid (ps / pgrep for VLLM::EngineCore or the engine-core process) and assert FR13_GDN_SUBOP_MAB=1 is',
'present in /proc/<EngineCore pid>/environ. If ABSENT, FAIL LOUD (ok=false) and do NOT run the teacher-force',
'(do not waste the boot on a vacuous instrument). Only proceed to capture once the var is confirmed in the',
'WORKER environ, not just the container Config.Env.',
'',
'THEN CAPTURE: engagement tok/draft==9 (class-9) + within_boot_det [T,T,T,T] (class-8). Read',
'output/fr13_gdn_subop_mab/*.jsonl: deep-spine conv1d_out + scan_out M10-vs-M5 RAW max_abs + first-nonzero.',
'gpuRunPlan env: FR13_GDN_SUBOP_MAB=1 FR13_GDN_SUBOP_MAB_LAYER=language_model.model.layers.0.linear_attn',
'FR13_GDN_SUBOP_MAB_EXPECT_TREE_N=10 FR13_GDN_SUBOP_MAB_LIMIT=1 FR13_GDN_SUBOP_MAB_DUMP=output/fr13_gdn_subop_mab/fr13_gdn_subop_mab.jsonl.',
'DISCRIMINATOR: conv1d_out OR scan_out M10-vs-M5 != 0 => paddable M-keyed L0-GDN carrier in the residual',
'(align it); both ~0 => residual depth-intrinsic + FA2-downstream (batch-invariance exhausted at in_proj_ba).',
'Reward-hacks BANNED (observe-only). Quote FR13_BUG_CLASS_PLAYBOOK.md rows (class-9 worker-env gate).'
].join('\n');

phase('BootCapture');
const CAP_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['envFix','workerEnvConfirmed','booted','engaged','tok_per_draft','within_boot_det','recordsWritten','conv1d_out_m10_vs_m5','scan_out_m10_vs_m5','boundsGuardOrCrash','firstNonzeroSubOp','ok','notes'],
  properties: {
    envFix: { type: 'string', description: 'how FR13_GDN_SUBOP_MAB was made to reach the worker (VLLM_RAY_EXTRA_ENV_VARS_TO_COPY / alias / import-time) + the launcher commit' },
    workerEnvConfirmed: { type: ['boolean','null'], description: 'is FR13_GDN_SUBOP_MAB=1 confirmed in /proc/<EngineCore pid>/environ BEFORE capture?' },
    booted: { type: 'boolean' },
    engaged: { type: 'boolean' },
    tok_per_draft: { type: ['number','null'] },
    within_boot_det: { type: 'string' },
    recordsWritten: { type: ['integer','null'] },
    conv1d_out_m10_vs_m5: { type: ['number','string','null'] },
    scan_out_m10_vs_m5: { type: ['number','string','null'] },
    boundsGuardOrCrash: { type: 'string', description: 'did Front B bounds-guard fire (clean Python raise) / did the hook capture cleanly / did the prior CUDA assert recur?' },
    firstNonzeroSubOp: { type: ['string','null'] },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const cap = await agent(
  CTX + '\n\nTASK (BootCapture, GPU - the ONLY boot). Apply the env fix (commit pathspec). Hygiene + boot cat9 '
  + 'with FR13_GDN_SUBOP_MAB=1 + siblings. ASSERT the master switch in /proc/<EngineCore pid>/environ BEFORE '
  + 'capture (FAIL LOUD ok=false + teardown if absent - no vacuous boot). Then engagement tok/draft==9 + '
  + 'within_boot_det [T,T,T,T], capture deep-spine conv1d_out/scan_out M10-vs-M5. Report boundsGuardOrCrash '
  + '(first real exercise of 8cdda4c4). Teardown + recover; never leak. Return the schema.',
  { label: 'bootcapture-l0gdn', phase: 'BootCapture', schema: CAP_SCHEMA, model: 'opus' }
);

phase('Verdict');
const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','paddableCarrier','residualNature','nextLever','issues'],
  properties: {
    holds: { type: 'boolean', description: 'false if worker-env unconfirmed / 0 records / engaged/det failed' },
    paddableCarrier: { type: ['boolean','null'], description: 'conv1d/scan M10-vs-M5 != 0?' },
    residualNature: { type: 'string', description: 'nonzero => align that op; both ~0 => depth-intrinsic + FA2-downstream, BI exhausted at in_proj_ba' },
    nextLever: { type: 'string', description: 'single next lever. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verdict = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(cap) + '. Read output/fr13_gdn_subop_mab/*.jsonl. Default '
  + 'holds=false if workerEnvConfirmed!=true OR recordsWritten==0 OR engaged/det failed (do NOT report a '
  + 'depth-intrinsic verdict from a vacuous boot). Conclude paddable carrier vs depth-intrinsic. No '
  + 'close/pass-fail; no reward-hack.',
  { label: 'verdict', phase: 'Verdict', schema: VERDICT_SCHEMA, agentType: 'general-purpose' }
);

return { cap, verdict };
