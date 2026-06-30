export const meta = {
  name: 'fr13-diffuse-bigdenom-design',
  description: 'USER (2026-06-15): design a test that says whether the cat9 drift is genuinely DIFFUSE, run on a SWE-Verified ~30-min-bounded task so we get a BIG DENOMINATOR (the 4-prompt/~450-token frame is small-sample: ±a few flips of cross-boot noise dominates, so 23-vs-3 has loose error bars). Research + read code + GIT HISTORY to design: (1) the big-denominator run = cat9 AND native-E5 on a real SWE-Verified task (the FR9/FR10 SWE harness), ~30-min wall bound, scoring per-token clear-margin flips vs the deployment-correct RECURRENT oracle over the whole (thousands-of-tokens) generation = a statistically tight flip RATE; (2) the DIFFUSE-vs-FIXABLE discriminator at scale (rate stability, structural-boundary clustering, per-layer attribution if feasible, the binding gold-margin teacher-forced probe); (3) the SWE-QUALITY gate (does cat9 SOLVE the task correctly vs native despite the flips = the deployable answer). Produce a READY-TO-RUN GPU spec + the diffuse-test methodology. CPU read-only, code + git-history, adversarial verify. Output FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md.',
  phases: [
    { title: 'Design' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN + 16 full-attn). GB10 273 GB/s,',
'B=1 decode ~17 TPS. Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (the K1 mechanism proof runs',
'concurrently on GPU; do NOT edit code/boot). Read our code + GIT HISTORY (git log/show, how the SWE-Verified',
'gate + the 22->3 chase ran) + binds. Write ONLY FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e). int-view NEVER atol.',
'',
'THE PROBLEM TO SOLVE (user, this turn): the banked flip numbers are SMALL-SAMPLE. Denominator = 4 prompts_swe4',
'x up to 128 tokens, varying by EOS: native-E5 = 3 flips / 512 tokens = 0.59%; cat9 OFF = 23 / 435 = 5.29%;',
'cat9+K1 = 20 / 466 = 4.29%. Cross-boot autotune noise is ±a few flips (feedback_no_cross_boot_byte_gate), which',
'is large relative to ~3-23 counts over ~450 tokens -> loose error bars, the numbers wobble (21/22/23), and K1',
'18->12 cannot be certified above noise. The user wants a BIG DENOMINATOR via a SWE-Verified ~30-min task to get',
'a tight RATE and a real diffuse-vs-fixable verdict.',
'',
'WHAT IS BANKED (the diffuse hypothesis to TEST, not assume): the cat9 residual (~12-23 flips) is claimed',
'genuinely DIFFUSE = the multi-layer GDN/full-attn realization gap (FR13_DIFFUSION_DEEP_DIVE: geometric 1.166x/',
'layer, no single dominant layer) + the LCP-committer trajectory-fork superset (FR13_LEAF_CORESIDENCY_PATH;',
'apple-to-apple R=8 genuine confident verify-vs-decode flips + C=8 near-ties + W=7 instrument-artifacts), NOT a',
'single per-forward seam (scan-recompute refuted e2e, K1 partial ~1/3, N_PAD no-op/null, K2-K5 ~0 - all clean).',
'The flip = committer serves the VERIFY-forward argmax, oracle wants the DECODE-forward argmax, they differ',
'(verify-vs-decode realization gap); flips cluster at small-clean-margin structural boundaries (codefence/prose/',
'tool-call). COMPARE TARGET: cat9 vs native-E5 each-vs-its-own no-spec RECURRENT decode oracle (fr13_recurrent_',
'decode_oracle, deployment-correct, NOT prefill/streamed-logprobs). native-E5=3 = the within-floor BAR.',
'',
'YOUR JOB - design the big-denominator diffuse test + ready-to-run GPU spec:',
'1. THE SWE-VERIFIED BIG RUN: read the FR9/FR10 SWE-Verified harness (git history + scripts: how B=4 SWE-Verified',
'   4-task ran, the swe_bench harness in .cache/swe_bench_repos, the deliverable gate). Design cat9 AND native-E5',
'   each on a real SWE-Verified task (or a few), ~30-min wall bound, capturing the FULL served stream (the agent',
'   patch-generation = thousands of tokens across turns = the big denominator, NOT 30k - a SWE patch is hundreds-',
'   to-few-thousand tokens). Specify: which task(s), the wall bound, how to capture every served token.',
'2. FLIP SCORING AT SCALE: the recurrent oracle teacher-forces the served stream one forward/token. For a few-',
'   thousand-token denominator this is feasible but is the bottleneck - specify how to score the WHOLE stream',
'   (incremental or post-hoc, bounded), per-token clear-margin flips (deviation_nat>1.0 gold-margin, NOT streamed',
'   logprobs), for BOTH cat9 and native-E5 -> the tight flip RATE with a real denominator + a binomial CI.',
'3. DIFFUSE-vs-FIXABLE DISCRIMINATOR (the actual test): over the big sample, design the measurements that',
'   DISTINGUISH diffuse (verify-vs-decode realization gap) from a fixable seam: (a) is the cat9 rate stable +',
'   significantly above native (CI non-overlapping)? (b) are the flips structural-boundary-clustered (token-class',
'   distribution: codefence/prose/tool-call/JSON vs format-fixed)? (c) the leaf-fork vs spine-realization split',
'   at scale (reuse the apple-to-apple/fork-margin classify, FIXED reducer per FR13_APPLE_TO_APPLE_FORK); (d) if',
'   feasible, a per-layer first-nonzero attribution on a sample of flips (diffuse = no single dominant layer).',
'   State the DECISION RULE: what big-denominator result = DIFFUSE (no fixable seam, relax) vs FIXABLE (a',
'   concentration that re-opens a lever).',
'4. SWE-QUALITY GATE (the deployable answer): does cat9 SOLVE the SWE-Verified task correctly (patch applies +',
'   tests pass) vs native-E5, DESPITE the ~5% flips? If yes, the flips are quality-irrelevant structural-boundary',
'   token choices = the deployable lossless-enough answer. Specify the pass/fail capture.',
'',
'DELIVERABLE: FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md = the READY-TO-RUN GPU spec (exact SWE task(s), wall bound,',
'flags = the deployed cat9 pipeline + native-E5, capture, scoring) + the diffuse-vs-fixable discriminator with a',
'DECISION RULE + the SWE-quality gate + non-vacuity hooks (oracle engaged, big denominator real, flags live,',
'NOT streamed logprobs). Distinguish what is RUNNABLE-NOW from what needs a new harness. Quote FR13_BUG_CLASS_',
'PLAYBOOK (#9 vacuous, #12 length/denominator). Honest about GPU cost (the rescore is the bottleneck). Commit',
'pathspec.',
].join('\n');

phase('Design');
const D_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['sweRunSpec','flipScoringAtScale','diffuseDiscriminator','decisionRule','sweQualityGate','gpuCostEstimate','runnableNowVsNewHarness','committed','notes'],
  properties: {
    sweRunSpec: { type: 'string', description: 'the exact SWE-Verified task(s), 30-min wall bound, cat9 + native-E5 deployed flags, full-served-stream capture - from the FR9/FR10 SWE harness + git history' },
    flipScoringAtScale: { type: 'string', description: 'how to score per-token clear-margin flips vs the recurrent oracle over the few-thousand-token denominator (feasible, bounded), for both arms + the binomial CI' },
    diffuseDiscriminator: { type: 'string', description: 'the measurements distinguishing diffuse (realization gap, structural-boundary, no dominant layer) from a fixable seam at scale: rate-CI, token-class distribution, leaf-fork-vs-spine split, per-layer sample' },
    decisionRule: { type: 'string', description: 'the explicit DECISION RULE: what big-denominator result = DIFFUSE (relax) vs FIXABLE (re-open a lever)' },
    sweQualityGate: { type: 'string', description: 'does cat9 SOLVE the task (patch+tests pass) vs native despite flips - the deployable answer; pass/fail capture' },
    gpuCostEstimate: { type: 'string', description: 'honest GPU cost (generation + rescore bottleneck) within the 30-min frame' },
    runnableNowVsNewHarness: { type: 'string', description: 'what is runnable with existing scripts vs needs a new harness' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const d = await agent(
  CTX + '\n\nTASK (Design, no GPU, read-only). Read the FR9/FR10 SWE harness + git history + the recurrent oracle '
  + '+ the diffusion/apple-to-apple binds. Design the big-denominator diffuse test + ready-to-run GPU spec. Write '
  + 'FR13_DIFFUSE_BIGDENOM_TEST_PLAN.md, commit pathspec. Return the schema.',
  { label: 'diffuse-bigdenom-design', phase: 'Design', schema: D_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','specRunnable','scoringFeasible','discriminatorReal','decisionRuleSound','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    specRunnable: { type: 'string', description: 'is the SWE run spec grounded in the ACTUAL harness (cited scripts/git), runnable, with a real big denominator?' },
    scoringFeasible: { type: 'string', description: 'is the at-scale recurrent-oracle scoring feasible + non-vacuous (oracle engaged, gold-margin not streamed logprobs) within the GPU budget?' },
    discriminatorReal: { type: 'string', description: 'does the diffuse-vs-fixable discriminator actually distinguish them (not just re-measure the rate)? is the decision rule falsifiable?' },
    decisionRuleSound: { type: 'string', description: 'is the DIFFUSE-vs-FIXABLE decision rule sound + does the SWE-quality gate give the deployable answer?' },
    recommendation: { type: 'string', description: 'single: is the plan ready to run as the GPU big-denominator test, or what to fix first. No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(d) + '. Default holds=false if the SWE spec is not grounded '
  + 'in the real harness (cite scripts/git), the at-scale scoring is vacuous/infeasible or uses streamed '
  + 'logprobs, the diffuse-vs-fixable discriminator just re-measures the rate without distinguishing the two, or '
  + 'the decision rule is not falsifiable. No close/pass-fail; no reward-hack.',
  { label: 'verify-diffuse-bigdenom-design', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { d, v };
