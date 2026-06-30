export const meta = {
  name: 'fr13-subop-mab-gate-rebuild',
  description: 'CPU-only: root-cause WHY the FR13_GDN_SUBOP_MAB L0-GDN sub-op A/B failed 4x (CUDA crash -> env-not-in-worker -> env-fix-failed -> hook-body-not-executing) and design the COMPLETE rebuilt gate (robust env-to-worker + call-site engagement + fail-loud at every stage so future chases are never vacuous). READ-ONLY on the live patcher (the bake is booting from it); write the ready-to-apply rebuild to a doc. Adversarial verify.',
  phases: [
    { title: 'RootCause' },
    { title: 'RebuildDesign' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY on the live',
'patcher scripts/fr10_phase4_patch_vllm_tree_gdn.py (a GPU bake workflow is booting from it concurrently -',
'do NOT edit it; write the rebuild to FR13_SUBOP_MAB_REBUILD.md as a ready-to-apply implementation). Pathspec',
'commits only.',
'',
'THE INSTRUMENT: FR13_GDN_SUBOP_MAB = the decoherence-free in-process L0-GDN sub-op A/B. On a captured',
'deep-spine carrier verify event it re-runs the GDN sub-ops (conv1d_out, scan_out) at M=10 (full cat9 tree)',
'vs M=5 (spine-slice) vs M=1 (decode) reusing the SAME captured input + h0 + conv-state, and records the',
'deep-spine-row RAW max_abs M10-vs-M5 (the clean co-residency verdict). It is the discriminator for whether',
'a paddable M-keyed L0-GDN op remains in a residual. It has FAILED FOUR TIMES, all infrastructure not science:',
'',
'FAILURE 1 (task w68z6gxgy): flag-ON CRASHED the engine with a CUDA device-side assert on the first cat9',
'tree-verify - the M5/M1 reduced-row arm (_scan_arm ~:1465-1485 / _conv_arm) passed the deep node prior state',
'with an INVALID BANK (ssi0[0,deep_row] = a branch/leaf cache bank). Front B fix 8cdda4c4/d30755c8: reduced',
'arms use the valid committed prior bank ssi0[0,0] as a 1-row init-state at col0 + every index bounds-guarded',
'with a clean Python raise. 11/11 CPU tests, default byte-identical. BUT NEVER GPU-VALIDATED (every later',
'boot disengaged before the GDN forward, so the bounds-guard never fired).',
'FAILURE 2 (task wc0gyx2za): booted+served+engaged cat9 (tok/draft=9) but the hook fired 0 times = 0 records.',
'ROOT: FR13_GDN_SUBOP_MAB is read INSIDE the GDN forward = the EngineCore WORKER process (pid ~176), but',
'vLLM ray-worker spawn (vllm/ray/ray_env.py get_env_vars_to_copy) copies ONLY vllm.envs-registered vars +',
'DEFAULT_ENV_VAR_PREFIXES={VLLM_}. The bare FR13_GDN_SUBOP_MAB master switch (+ _DUMP/_EXPECT_TREE_N/_THRESHOLD)',
'is NOT VLLM_-prefixed => stripped from the worker. /proc/176/environ confirmed: only 14/66 FR13 vars reached',
'the worker (the SUBOP siblings _LAYER/_LIMIT/_SKIP DID get through - figure out WHY those and not the master).',
'FAILURE 3 (task wl043ivfu first boot): the VLLM_RAY_EXTRA_ENV_VARS_TO_COPY env-fix attempt STILL did not get',
'the master switch into the worker (worker-env gate /proc/<pid>/environ FAIL-LOUD, no wasted teacher-force).',
'FAILURE 4 (task wl043ivfu off-script re-boot, patcher-import-time attempt): even with the env reportedly',
'present, "No subop activity, no jsonl, no error. The _fr13_gdn_subop_mab call is genuinely not executing its',
'body. Given use_fr10_tree..." => the CALL SITE itself is not reached (the verify-site call is inside an',
'`if use_fr10_tree:` / `if spec_sequence_masks is not None:` branch, and/or the conv-site stash did not run,',
'so the gate never even gets evaluated, OR an engagement assert raised+was swallowed silently).',
'',
'YOUR JOB:',
'1. ROOT-CAUSE precisely (read the patcher hook: the helper def ~:1295, the conv-site stash ~:1856, the',
'   verify-call ~:3901/:4138, the gates, the use_fr10_tree / spec_sequence_masks branch conditions). Explain',
'   #4 definitively: under what cat9-verify condition is the _fr13_gdn_subop_mab call site reached vs skipped?',
'   Why did the 14 sibling FR13 vars propagate to the worker but not the master? (the patcher-import-time path.)',
'2. DESIGN THE COMPLETE REBUILT GATE (ready-to-apply, AST-valid patcher edits) that makes the instrument',
'   RELIABLE + NEVER VACUOUS:',
'   (a) ROBUST ENV-TO-WORKER: get FR13_GDN_SUBOP_MAB into the EngineCore worker reliably - the robust way is',
'       PATCHER-IMPORT-TIME (the patcher runs at vLLM import in BOTH pid 1 and the worker; have it set',
'       os.environ master from a propagated signal, OR via a VLLM_-prefixed alias the allowlist copies, OR',
'       register it in vllm.envs). Pick the one that the 14 siblings already use.',
'   (b) CALL-SITE ENGAGEMENT: ensure the hook body executes on the cat9 deep-spine verify event (fix the',
'       use_fr10_tree / spec_sequence_masks gating so the call site is reached, or move the gate).',
'   (c) FAIL-LOUD AT EVERY STAGE (class-9, the whole point): a single observable signal at each stage so a',
'       future chase is NEVER silently vacuous - (i) env-not-in-worker (log the /proc check), (ii) call-site-',
'       not-reached (a one-time log if the env is ON but the call site is skipped), (iii) engagement-assert-',
'       fail (a clear raise/log, not swallowed), (iv) capture-written (record count). The prior failures were',
'       all "no log line of any kind" - the rebuild must emit a stage marker so the failing stage is obvious.',
'   Keep the FR13_GDN_SUBOP_MAB default-OFF + the locked cat9 path byte-identical when OFF. Front B bounds-',
'   guard fix (8cdda4c4) stays.',
'3. Write FR13_SUBOP_MAB_REBUILD.md = the root-cause + the complete ready-to-apply patcher edits (exact',
'   needles + replacements, AST-valid) + the CPU wiring tests to add + the next-GPU-run plan (worker-env',
'   gate + stage-marker checks). I will APPLY it to the patcher AFTER the bake frees the patcher.',
'Reward-hacks BANNED (observe-only instrument). Quote FR13_BUG_CLASS_PLAYBOOK.md row 9 (silent/vacuous).'
].join('\n');

phase('RootCause');
const RC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['callSiteReachedWhen','whyBodyNotExecuting','whySiblingsPropagatedNotMaster','engagementAssertPath','notes'],
  properties: {
    callSiteReachedWhen: { type: 'string', description: 'under what cat9-verify condition is the _fr13_gdn_subop_mab call site reached vs skipped (the use_fr10_tree / spec_sequence_masks branch), with file:line' },
    whyBodyNotExecuting: { type: 'string', description: 'failure 4 definitive root cause: env present but body not executing - which gate/branch/assert' },
    whySiblingsPropagatedNotMaster: { type: 'string', description: 'why the 14 FR13 siblings reached the worker but the master switch did not (the allowlist / patcher-import path)' },
    engagementAssertPath: { type: 'string', description: 'do the in-hook engagement asserts raise loud or get swallowed? cite the try/except' },
    notes: { type: 'string' },
  },
};
const rc = await agent(
  CTX + '\n\nTASK (RootCause, no GPU, read-only). Do step 1. Return the schema (do NOT write the doc yet).',
  { label: 'rootcause-subop-mab', phase: 'RootCause', schema: RC_SCHEMA, model: 'opus' }
);

phase('RebuildDesign');
const RB_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['envToWorkerFix','callSiteFix','failLoudStages','readyToApplyEdits','defaultOffPreserved','cpuTests','notes'],
  properties: {
    envToWorkerFix: { type: 'string', description: 'the robust env-to-worker mechanism (patcher-import-time / VLLM_ alias / vllm.envs register) + why it reliably reaches the worker' },
    callSiteFix: { type: 'string', description: 'how the call site is made to execute on the cat9 verify event' },
    failLoudStages: { type: 'string', description: 'the stage markers added (env / call-site / engagement / capture) so a future run is never silently vacuous' },
    readyToApplyEdits: { type: 'string', description: 'the exact patcher needles + replacements (AST-valid), ready to apply post-bake' },
    defaultOffPreserved: { type: 'string', description: 'proof the FR13_GDN_SUBOP_MAB-OFF path stays byte-identical (locked cat9 unaffected)' },
    cpuTests: { type: 'string', description: 'the CPU wiring tests to add' },
    notes: { type: 'string' },
  },
};
const rb = await agent(
  CTX + '\n\nTASK (RebuildDesign, no GPU, read-only on the patcher). Given RootCause: ' + JSON.stringify(rc)
  + '. Do steps 2-3. Write FR13_SUBOP_MAB_REBUILD.md (the complete ready-to-apply rebuild), commit pathspec. '
  + 'Return the schema. Do NOT edit scripts/fr10_phase4_patch_vllm_tree_gdn.py (the bake is booting from it).',
  { label: 'rebuild-design-subop-mab', phase: 'RebuildDesign', schema: RB_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','rootCauseSound','rebuildWillEngage','failLoudComplete','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    rootCauseSound: { type: 'string', description: 'is the 4-failure root cause grounded in actual patcher + vLLM ray_env code?' },
    rebuildWillEngage: { type: ['boolean','null'], description: 'will the rebuilt gate reliably get the env into the worker AND reach the call site AND capture?' },
    failLoudComplete: { type: 'string', description: 'does the rebuild emit a stage marker at EVERY failure point so a future chase cannot be silently vacuous?' },
    recommendation: { type: 'string', description: 'single recommendation for applying the rebuild post-bake. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: RootCause ' + JSON.stringify(rc) + ' Rebuild ' + JSON.stringify(rb) + '. '
  + 'Default holds=false if the root cause is not grounded in code or the rebuild leaves any silent-vacuous '
  + 'path (a stage with no marker). Re-check the use_fr10_tree branch + the env-to-worker mechanism against '
  + 'the actual vLLM ray_env allowlist. No close/pass-fail; no reward-hack.',
  { label: 'verify-rebuild', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { rc, rb, v };
