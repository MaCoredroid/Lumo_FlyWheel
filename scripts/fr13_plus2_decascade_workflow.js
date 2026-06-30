export const meta = {
  name: 'fr13-plus2-spine-decascade',
  description: 'CPU-only: rigorously de-cascade the +2 spine gap (our-spine chain5/chain3 = 5 flips vs native 3) into INDEPENDENT divergence events vs cascade-inflated raw count, to settle whether the +2 is RESOLVED (a measuring/cascade artifact, our-spine ~ native in independent crossings) or a REAL per-forward defect (and if real, where - given scan/conv/fp8 are bit-exact and FA2 inert on a branchless spine). Adversarial verify.',
  phases: [
    { title: 'Decascade' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. READ-ONLY, CPU-ONLY (read the banked',
'flip records + code/binds, write ONLY FR13_PLUS2_DECASCADE.md; NO GPU boot). A GPU workflow runs concurrently',
'- do NOT modify any kernel/patcher; pathspec commits only.',
'',
'THE QUESTION (user): is the +2 spine drift RESOLVED (a measuring/cascade issue, our-spine ~ native) or a',
'REAL per-forward defect? Two clues TENSION:',
'- The recurrent re-score (FR13_ORACLE_FRAME_CLOSED_BIND, task wd0ok777i) found our-spine chain5 = 5 flips vs',
'  BOTH the chunked AND the recurrent oracle (5/5 byte-identical), native = 3/3. => the RAW count of 5 is',
'  consistent across oracles = NOT a chunk-vs-recurrent frame artifact.',
'- The +2-align research (FR13_PLUS2_BV_FIX_REFUTED_BIND, task w9dpm62la) found chain5 5 flips are ALL in',
'  prompt 2 at pos 25,26,27,28 (a CONTIGUOUS CASCADE: pos 25 = a tool-call format fork bash-vs-tool_call;',
'  once pos 25 commits a different token, the oracle for 26-28 is conditioned on the now-diverged served',
'  prefix, so 26-28 read as flips but only ONE independent decision diverged) + pos 43 (brace order). =>',
'  5 RAW = ~2 INDEPENDENT events. Native = 3, one-per-prompt at DIFFERENT boundaries.',
'',
'So the RAW 5-vs-3 may overstate a per-forward gap that, in INDEPENDENT crossings, is ~2 vs ~3 = AT OR BELOW',
'native. RIGOROUSLY settle it:',
'1. DE-CASCADE: read output/fr13_shape_sweep/chain5_flips.json + chain3_flips.json + q3_native_classify.json.',
'   For each flip at pos i, determine if it is INDEPENDENT (the served prefix up to i-1 matches the oracle',
'   stream, i.e. no upstream flip in this prompt within a short window) or a CASCADE consequence (an upstream',
'   flip at i-k diverged the served prefix the oracle conditions on). Use the per-position served_token vs',
'   oracle_argmax + the context_excerpt. Count INDEPENDENT events for chain5, chain3, AND native (native',
'   may also have micro-cascades - apply the SAME rule to be fair).',
'2. COMPARE independent-event counts: chain5 vs chain3 vs native. If chain5/chain3 independent ~ native',
'   independent => the +2 is a CASCADE/MEASURING artifact => RESOLVED (our-spine ~ native, no real',
'   per-forward defect). If chain5/chain3 independent > native independent => a REAL residual.',
'3. IF REAL: where? Everything on the branchless spine is bit-exact or inert - scan bit-exact to native at',
'   BOTH BV geometries (FR13_BV_GEOMETRY_NOT_THE_SEAM, RAW 0.0 D16=D32=N_PAD1=16), conv done (ex2-silu',
'   bf16-tap grind SUCCEEDED), fp8 M-invariant, FA2 tree-bias ARITHMETICALLY INERT on a branchless spine',
'   (_prepare_tree_attn_bias(chain5)==lower-triangular-causal). So a REAL +2 would be a PUZZLE (an unfound',
'   seam: the FA2-fork 2-ULP MMA-grouping floor on the spine? the tree-GDN-vs-native-MTP recurrence',
'   realization? the q1 deep-row finding L0 GDN 0.000854 vs recurrent oracle = ~1 ULP at floor). Reconcile:',
'   the re-score "real 5/5" is the RAW count (cascade-inflated); the independent count is the honest defect',
'   measure. Decide which the evidence supports.',
'',
'IMPORTANT: the BINDING lossless metric is per-token argmax vs the no-spec oracle (reference_scalar_metric_',
'per_token_blindspot), and accept/event is the deployable arbiter. The cascade is exactly the class-12',
'whole-window trap (scalar count inflated by a single upstream fork). Quote FR13_BUG_CLASS_PLAYBOOK.md rows.',
'Reward-hacks BANNED. Do NOT conclude "resolved" just because it is convenient - de-cascade rigorously and',
'apply the SAME rule to native.'
].join('\n');

phase('Decascade');
const DEC_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['chain5Independent','chain3Independent','nativeIndependent','cascadeEvidence','resolvedOrReal','ifRealWhere','notes'],
  properties: {
    chain5Independent: { type: 'string', description: 'chain5: raw flips (5) vs INDEPENDENT events (de-cascaded), with the per-pos evidence' },
    chain3Independent: { type: 'string', description: 'chain3: raw (5) vs independent, per-pos' },
    nativeIndependent: { type: 'string', description: 'native: raw (3) vs independent (apply the SAME de-cascade rule)' },
    cascadeEvidence: { type: 'string', description: 'the pos25-28 cascade chain proof (served prefix divergence) + pos43; native cascades if any' },
    resolvedOrReal: { type: 'string', description: 'VERDICT: is the +2 a cascade/measuring artifact (independent ~ native = RESOLVED) or a real residual (independent > native)?' },
    ifRealWhere: { type: 'string', description: 'if real: the candidate unfound seam (FA2 2-ULP floor / tree-GDN-vs-MTP recurrence / other), given scan/conv/fp8 bit-exact + FA2 inert' },
    notes: { type: 'string' },
  },
};
const dec = await agent(
  CTX + '\n\nTASK (Decascade, no GPU). Do steps 1-3. Write FR13_PLUS2_DECASCADE.md, commit pathspec. Return the schema.',
  { label: 'decascade-plus2', phase: 'Decascade', schema: DEC_SCHEMA, model: 'opus' }
);

phase('Verify');
const VERIFY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','independentVerdict','fairnessCheck','rewardHackCheck','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    independentVerdict: { type: 'string', description: 'the verified independent-event counts (chain5/chain3/native) + resolved-vs-real' },
    fairnessCheck: { type: 'string', description: 'ADVERSARIAL: was the SAME de-cascade rule applied to native (not a rule tuned to make ours look good)? re-check native for micro-cascades' },
    rewardHackCheck: { type: 'string', description: 'is "resolved (cascade)" an honest finding or a convenient re-definition of the metric to dismiss a real gap? the per-token argmax + accept/event must back it' },
    recommendation: { type: 'string', description: 'single recommendation: +2 resolved (move on / fold into the bake) OR real (the candidate seam to chase, a cheap GPU test). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const verify = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(dec) + '. Default holds=false if the de-cascade rule was '
  + 'NOT applied symmetrically to native (the reward-hack trap: a rule that de-cascades ours but not native). '
  + 'Re-check native for micro-cascades. Is "resolved" backed by per-token argmax + accept/event, or a convenient '
  + 're-definition? No close/pass-fail; no reward-hack.',
  { label: 'verify-decascade', phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' }
);

return { dec, verify };
