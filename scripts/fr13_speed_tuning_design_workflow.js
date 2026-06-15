export const meta = {
  name: 'fr13-speed-tuning-design',
  description: 'USER (2026-06-15): pivot to speed tuning toward cat9 B=1 decode-TPS STRICTLY > native E5, with the LONG-ESTABLISHED LOSSLESS GATE held as a per-change gate throughout. THREE lever families: (A) OPT-1 GPU-resident committer = build the UNBUILT G2 sync-kill (side-stream the committer .cpu().tolist() readback + CUDA event so the main thread stops blocking 91.9% -> restore run-ahead; the first-draft 10ebccac is GPU-resident but still syncs); tree-only, pure-integer lossless. (B) OPT-A GB10-tuned fp8 GEMV (e90de7ef, user ruled the bit-identical shared-kernel tune in-scope) = whole-system bandwidth win. (C) TOPOLOGY RESHAPE (user: "revive cat10 with ATTENTION TO THE ACCOUNTING ISSUE + other trees that REMOVE deep leaves but ADD a root leaf") = a speed+accept lever: deep leaves are expensive verify rows + drift carriers w/ low accept value; root siblings rescue the ~62% step-0 rejects (L3 conf-gated root sibling). cat10 prior accept 2.932 was a CLASS-12 trajectory + sibling-stop DENOMINATOR ARTIFACT (NOT a real loss) - measure accept depth-matched + PAIRED teacher-forced + confound-free this time. Design all three (CPU): the OPT-1 sync-kill patch, the candidate reshape trees + accounting-correct measurement, and the sequenced GPU campaign with the lossless gate held. CPU read-only (GPU free, do NOT boot - measurements after). Output FR13_SPEED_TUNING_PLAN.md.',
  phases: [
    { title: 'Design' },
    { title: 'Synthesize' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s LPDDR5X, B=1 decode HBM-bandwidth-bound). Repo',
'/home/mark/shared/lumoFlyWheel. CPU read-only (GPU free but do NOT boot - measurements run AFTER, under the',
'prelaunch host-mem protocol). Read our code + git history + the banked binds + vLLM source via scripts/vllm_src.sh',
'(pinned 3dbe092e, NEVER /tmp). Write ONLY FR13_SPEED_TUNING_PLAN.md + the design. Pathspec commit.',
'',
'CURRENT STATE (MEASURED, decode_seconds basis, FR13_SPEED_HISTORY_RECONCILE): native E5 0.2182 s/fwd / accept',
'3.161 / 18.93 TPS; cat9 0.2248 = 1.030x native (+6.5 ms/fwd) at accept ~3.18 (slight edge); lossless-vs-native',
'CONFIRMED at scale (big-denom cat9 13.55% ~= native 13.99%). GOAL: cat9 B=1 decode-TPS STRICTLY > native E5.',
'',
'THE LOSSLESS GATE (long-established, HELD per-change throughout speed tuning - feedback_fr13_lossless_compare_',
'target): every speed change must hold lossless = same-seed byte-identical served streams (greedy + t0.6) +',
'accept/event unchanged + regular-decode pristine + the per-token argmax-vs-clean-recurrent-oracle probe (fr13_',
'gold_margin_probe / the confound-free instrument). Compare = US vs native-E5 each-vs-its-own no-spec RECURRENT',
'decode oracle, NEVER chunked-prefill/streamed/serial/backend-name (int-view never atol).',
'',
'MEASUREMENT RULES (reference_fr10_speed_measurement_pitfalls + feedback_dont_handroll_speed): SPEED basis =',
'decode_seconds RAW /metrics counter (vllm:request_decode_time_seconds_sum / vllm:spec_decode_num_drafts_total),',
'NEVER TPS/accept (banned) nor wall; metrics OFF; per-request/per-event; BI=0 pinned identical both arms; pinned',
'prompts_swe4 seed 1313 greedy temp 0.0; engagement asserts (tok/draft, has_tree_parent_indices, tree_sample_',
'accept) BEFORE any number. ACCEPT must be DEPTH-MATCHED (feedback_depth_matched_accept_compare: cat9/cat10 d5 ->',
'native E5; depth-3 shapes -> native E3, which is UNMEASURED - capture before judging any d3 arm) and PAIRED',
'teacher-forced (the aggregate cross-trajectory accept is class-12 confounded). No hand-rolled TPS decomposition',
'as a MEASURED fact (all per-forward ms are INFERRED until the clean GPU measurement).',
'',
'TWO MEASUREMENT TIERS (user 2026-06-15): (1) DEV-ITERATION = B=1 EAGER decode_seconds (above) for fast cheap',
'tuning of each change during the build loop. (2) FINAL JUDGMENT / candidate comparison = B=4 + CUDA-GRAPH-',
'CAPTURED + 4 SWE-Verified tasks + ~30min PER CANDIDATE (the DEPLOYABLE gate; B=4 changes co-residency, CUDA-',
'graph capture is the deployed mode, SWE-Verified 4 tasks = real workload, 30min = denominator). EVERY candidate',
'(OPT-1 OFF/ON, OPT-A OFF/ON, EACH reshape tree incl cat10) gets the B=4/CUDA-captured/4-SWE-task/30min final',
'comparison, and the LOSSLESS gate is RE-CONFIRMED at B=4 (co-residency changes it - bit-exact at B=1 does NOT',
'imply bit-exact at B=4). CONSEQUENCE FOR DESIGN: each lever MUST be CUDA-GRAPH-CAPTURABLE and B=4-safe - the',
'OPT-1 sync-kill side-stream/CUDA-event must be capture-safe (no host sync inside the captured region), OPT-A\'s',
'config must capture at B=4 M-tiles, and each reshape tree must capture + behave at B=4 co-residency. A lever',
'that cannot B=4/CUDA-capture is NOT shippable - flag it in the design.',
].join('\n');

phase('Design');

const A_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['firstDraftState','syncKillDesign','losslessArg','measurementPlan','citations'],
  properties: {
    firstDraftState: { type: 'string', description: 'the OPT-1 first-draft (10ebccac FR13_GPU_COMMITTER) current state: what is GPU-resident, where the host .cpu().tolist() readback + main-thread sync still is, cited' },
    syncKillDesign: { type: 'string', description: 'the exact G2 sync-kill patch: side-stream the readback + CUDA event so committer inputs stay device-resident + the main thread does not block -> restore native-style run-ahead; default-OFF flag' },
    losslessArg: { type: 'string', description: 'why it is lossless (pure-integer, location-only, no float/reduction reorder, byte-identical when OFF) + the per-change lossless gate' },
    measurementPlan: { type: 'string', description: 'the OFF-vs-ON decode_seconds measurement (does the sync-kill reclaim the ~4-6ms + restore run-ahead) + the byte-A/B-on-GPU check' },
    citations: { type: 'string' },
  },
};

const B_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['cat10ReviveAccounting','removeDeepAddRootTrees','speedAcceptPrediction','accountingCorrectMeasurement','citations'],
  properties: {
    cat10ReviveAccounting: { type: 'string', description: 'cat10 revived: the shape (e.g. [2,6,8,6]), WHY its prior accept 2.932 was a class-12 trajectory + sibling-stop DENOMINATOR artifact (not a real loss), and how to measure it correctly (depth-matched E5, paired teacher-forced, confound-free instrument)' },
    removeDeepAddRootTrees: { type: 'string', description: 'candidate REMOVE-DEEP-LEAF / ADD-ROOT-LEAF trees (move depth-4/5 leaves to root siblings; root-heavy caterpillars; the L3 conf-gated root sibling emit (1,) when root margin<tau, 62% rejects step-0) - the exact tree_choices, reusing the committed reshape infra (_fr10_cat3w/_chain3/SPEC_CONFIG tree_choices auto-adapt)' },
    speedAcceptPrediction: { type: 'string', description: 'per candidate tree: predicted SPEED (fewer/shallower verify rows = cheaper s/fwd, lm-head GEMV over N rows) + ACCEPT (root rescue of step-0 rejects) + the lossless-held expectation; depth-matched, INFERRED-labeled' },
    accountingCorrectMeasurement: { type: 'string', description: 'the accounting-correct measurement protocol that AVOIDS the cat10 artifact: depth-matched (d5->E5, d3->E3-must-capture), PAIRED teacher-forced accept, decode_seconds for speed, the confound-free instrument + lossless gate, per-event not cross-trajectory' },
    citations: { type: 'string' },
  },
};

const [a, b] = await parallel([
  () => agent(
    BASE + '\n\nTASK (Design branch A = OPT-1 G2 SYNC-KILL). Read the OPT-1 first-draft (FR13_GPU_COMMITTER, '
    + '10ebccac; the gpu-committer patch fns + the kernel host .cpu().tolist() readback ~L471-474) + the committer '
    + '(fr10_phase4_patch_vllm_tree_gdn.py LCP path decision ~:5780-5879) + the vLLM model-runner decode loop (where '
    + 'the committer runs + where the sync blocks, via vllm_src.sh) + how native avoids it (run-ahead). Design the '
    + 'G2 sync-kill patch + lossless arg + OFF-vs-ON measurement. Return the schema.',
    { label: 'design:opt1-synckill', phase: 'Design', schema: A_SCHEMA, model: 'opus' }
  ).catch(() => null),
  () => agent(
    BASE + '\n\nTASK (Design branch B = TOPOLOGY RESHAPE, user "revive cat10 w/ accounting attention + remove-deep-'
    + 'add-root trees"). Read the cat10 history (cat10_gate_wf_f4c5b6f8, cat10_investigate_wf_59bf2440; the [2,6,8,6] '
    + 'shape FLAT 22 / accept 2.932 = class-12 trajectory + sibling-stop DENOM artifact per feedback_check_artifact_'
    + 'before_concluding) + project_fr13_tree_reshape_unifying_lever (shallower+root-sibling cuts verify rows = speed '
    + '+ rescues d0 accept, 813cb9fd) + the L3 conf-gated root sibling (wgb0yegin) + the committed reshape infra '
    + '(_fr10_cat3w_choices/_fr10_chain3_choices, SPEC_CONFIG tree_choices auto-adapt, fr10_phase4...:10747). Design '
    + 'the cat10 revive (accounting-correct) + remove-deep-add-root candidate trees + the speed/accept prediction + '
    + 'the accounting-correct measurement that AVOIDS the cat10 artifact. Return the schema.',
    { label: 'design:topology-reshape', phase: 'Design', schema: B_SCHEMA, model: 'opus' }
  ).catch(() => null),
]);

phase('Synthesize');
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['sequencedCampaign','sharedMeasurementProtocol','losslessGatePerChange','committed','notes'],
  properties: {
    sequencedCampaign: { type: 'string', description: 'the sequenced speed-tuning GPU campaign across OPT-1 (sync-kill) + OPT-A (fp8 tune) + topology reshape (cat10 + remove-deep-add-root), the order + dependencies + which to measure first' },
    sharedMeasurementProtocol: { type: 'string', description: 'the TWO-TIER measurement: (1) B=1 eager decode_seconds for dev-iteration; (2) FINAL JUDGMENT = B=4 + CUDA-graph-captured + 4 SWE-Verified tasks + ~30min PER candidate (lossless re-confirmed at B=4). depth-matched + paired-teacher-forced accept, confound-free instrument, prelaunch, BI=0, engagement asserts - explicitly avoiding the cat10 denominator/trajectory artifact + each lever\'s CUDA-capture/B=4 compat' },
    losslessGatePerChange: { type: 'string', description: 'the per-change lossless gate held throughout (byte-identical streams + accept/event unchanged + per-token argmax probe + regular-decode pristine), how it is applied to each lever' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const synth = await agent(
  BASE + '\n\nTASK (Synthesize). Combine branch A (OPT-1 sync-kill): ' + JSON.stringify(a) + '\nand branch B '
  + '(topology reshape): ' + JSON.stringify(b) + '\ninto the sequenced speed-tuning campaign + the shared '
  + 'accounting-correct measurement + the per-change lossless gate. Write FR13_SPEED_TUNING_PLAN.md, commit '
  + 'pathspec. Return the schema.',
  { label: 'synthesize-speed-tuning', phase: 'Synthesize', schema: SYNTH_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','syncKillSound','reshapeAccountingSound','losslessHeld','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    syncKillSound: { type: 'string', description: 'is the OPT-1 sync-kill design real (removes the main-thread block / restores run-ahead, not just relocating the sync) + lossless (pure-integer location-only)?' },
    reshapeAccountingSound: { type: 'string', description: 'does the cat10/reshape measurement AVOID the prior artifact (depth-matched, paired teacher-forced, per-event not cross-trajectory denom), not re-make the class-12 mistake?' },
    losslessHeld: { type: 'string', description: 'is the long-established lossless gate genuinely held per-change for every lever (not relaxed for speed)?' },
    recommendation: { type: 'string', description: 'single: the sequenced speed campaign (which lever/measurement first) - OPT-1 sync-kill, reshape, OPT-A. No premature STOP; no reward-hack (WY parked, no deviate-shared beyond the ruled-in OPT-A).' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify({ synth }) + '. Default holds=false if the sync-kill just '
  + 'relocates the sync (does not restore run-ahead) or is not pure-integer lossless; if the cat10/reshape '
  + 'measurement re-makes the class-12 trajectory/denominator artifact (must be depth-matched + paired teacher-'
  + 'forced + per-event) or is not depth-matched (d3->E3 unmeasured); if the lossless gate is relaxed for speed; '
  + 'if the FINAL-JUDGMENT tier is missing or weaker than B=4 + CUDA-graph-captured + 4 SWE-Verified tasks + ~30min '
  + 'per candidate (with lossless re-confirmed at B=4), or a lever\'s CUDA-capture/B=4-safety is not addressed; '
  + 'or if a lever is a reward-hack (WY parked; multispine/copy/dense banned; OPT-A is the only ruled-in shared '
  + 'tune). research-before-deadend. No close/pass-fail.',
  { label: 'verify-speed-tuning', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { a, b, synth, v };
