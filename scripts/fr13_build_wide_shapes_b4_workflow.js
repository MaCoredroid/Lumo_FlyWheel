export const meta = {
  name: 'fr13-build-wide-shapes-b4',
  description: 'USER (2026-06-15): try MORE tree shapes like a "3-3-3" (depth-3, 3-wide = top-3 candidates per depth) compared to E3, depth-matched likewise; and starting now measure at B=4 (deployable co-residency regime, ~30min wall anyway). The B=1 campaign found B=1 is HBM-bound so per-forward cost barely moves + accept dominates - so WIDER trees (more candidates -> more accept) are the right thing to explore, and B=4 co-residency is the FR13 crux that may change the picture. BUILD the 3-3-3 shape (depth-3 width-3 = [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),(0,0,1),(0,0,2)], 9 nodes pad16) - this needs DRAFTER RANK-2 support (the MTP drafter currently only emits child-rank<=1 = top-2; extend it to read the 3rd-place token top-3 from the SAME spine logits, like cat3w but with a rank-2 leaf at each depth; child-rank>=2 was "never drafted" = a packing limit, the logits are already there). Default-OFF exact-match, lossless-by-construction (drafter-packing-only, masks/committer auto-derive). + the B=4 DEPTH-MATCHED measurement plan: depth-3 family {chain3, cat3w, 3-3-3} -> native E3; depth-5 family {chain5, cat6root, cat9, cat10} -> native E5; capture native E3/E4/E5 at B=4 as the depth bars (E3/E4 UNMEASURED per the trust ledger), deploy-speed + deploy-lossless at B=4 co-residency, lossless within the depth-matched native floor. CPU build on fr13-speedfix (parallel to OPT-1 GPU); the B=4 GPU sweep runs after. Phase1 build (CPU) -> Phase2 verify. Output FR13_WIDE_SHAPES_B4_PLAN.md.',
  phases: [
    { title: 'Build' },
    { title: 'Verify' },
  ],
}

const BASE = [
'FR13 on DGX Spark GB10 (sm_121, 273 GB/s, B-decode HBM-bound). Repo /home/mark/shared/lumoFlyWheel, BRANCH',
'fr13-speedfix (the reshape builds live here). Read code + git history + vLLM source via scripts/vllm_src.sh',
'(pinned 3dbe092e, NEVER /tmp). int-view never atol. Pathspec commit on fr13-speedfix. NO GPU (OPT-1 wf w7vhfiwno',
'holds the GPU; this is the CPU build, the B=4 sweep runs after). MAX 2 concurrent workflows.',
'',
'CONTEXT: the B=1 deployment campaign found B=1 decode is HBM-bound (98.6ms weight-read floor) so verify-forward',
'cost scales ~linearly with node count but is NOT the TPS lever - accept/event + the codex trajectory dominate;',
'cat6root (pad8, fewer nodes) only shaved -1.7% s/fwd but LOST accept -> net -4.7% TPS. So the right direction is',
'WIDER trees (MORE candidates -> MORE accept), and at B=4 (the deployable co-residency regime, where the FR13',
'co-residency effects live + which the B=1 screen could not see). USER: more shapes like 3-3-3 vs E3, B=4 going',
'forward.',
'',
'THE EXISTING SHAPES (already built, exact-match, default cat9): chain5/chain3 (pure spines), cat3w [(0,),(1,),',
'(0,0),(0,1),(0,0,0)] (depth-3, 5 nodes, root sib + 1 depth-2 leaf), cat6root [(0,),(1,),(0,0),(0,0,0),(0,0,0,0),',
'(0,0,0,0,0)] (depth-5, 6 nodes pad8), cat9 (9 nodes pad16), cat10 (10 nodes pad16). The packing pattern is at',
'fr10_phase4_patch_vllm_tree_gdn.py (cat3w :11005/:11026/:11515-11538; cat6root/cat10 :11009-11644). The drafter',
'reads top-1 (spine) + top-2 (the rank-1 leaf) from each spine MTP step; child-rank>=2 is NOT currently drafted.',
'',
'BUILD: the 3-3-3 shape = depth-3, 3 candidates per depth: tree_choices [(0,),(1,),(2,),(0,0),(0,1),(0,2),(0,0,0),',
'(0,0,1),(0,0,2)] (9 nodes, depth-3, pad16). At each of the 3 spine depths: spine (top-1) + rank-1 leaf (top-2) +',
'RANK-2 leaf (top-3). This needs the DRAFTER to emit the 3rd-place token (top-3 / child-rank-2) at each spine',
'position - extend the caterpillar packing to read torch.topk(logits, 3)[:, 2] from the SAME spine logits (no',
'extra lm-head, the logits are already computed) and pack the rank-2 leaves in the (len,path)-sorted slots. New',
'exact-match guard _fr10_is_333 (n==9 AND sorted-tree match AND mode tree_mtp), default cat9/cat3w/cat6root/cat10',
'untouched, FAIL-LOUD on disengagement. Lossless-by-construction: drafter-packing-only (the rank-2 token is a',
'runner-up read from the same logits, never fed into a forward/recurrent state; parent/ancestry masks, committer',
'path enum, eager-pack replay, conv prior-windows ALL auto-derive from SPEC_CONFIG tree_choices; verified by its',
'own recurrent oracle, depth-matched d3 -> native E3). If rank-2 drafter support is infeasibly invasive, FLAG it',
'+ propose the nearest rank-<=1 depth-3 width-3 alternative; do NOT reward-hack.',
'',
'B=4 MEASUREMENT PLAN (the deliverable measurement, runs after on GPU): deploy at B=4 (MAX_NUM_SEQS=4 co-residency,',
'the deployable regime, ~30min wall codex) via fr13_measure deploy-speed (s/fwd + accept + derived TPS, INSTRUMENT',
'OFF) + deploy-lossless (flip-rate vs the recurrent oracle within the DEPTH-MATCHED native floor). DEPTH-MATCH:',
'depth-3 {chain3, cat3w, 3-3-3} vs native E3 (MTP-3); depth-5 {chain5, cat6root, cat9, cat10} vs native E5. Native',
'E3 AND E4 are UNMEASURED (trust ledger gap) - CAPTURE native E3/E4/E5 at B=4 as the depth bars FIRST. accept is',
'B-dependent + trajectory-bound (bind to served_stream_fingerprint, like-for-like trajectory, exclude degenerate',
'forks = the cat6root-r1 #12 lesson). The question: at B=4, does a WIDER tree (3-3-3) net-beat its depth-matched',
'native on TPS (accept gain > co-residency cost) where the narrow B=1 reshapes did not?',
].join('\n');

phase('Build');
const B_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['shape333Built','rank2Support','losslessByConstruction','offlineGate','b4Plan','committed','notes'],
  properties: {
    shape333Built: { type: 'string', description: 'the 3-3-3 shape built (the exact tree_choices, the guard, default-OFF) + how to engage on a boot' },
    rank2Support: { type: 'string', description: 'how the drafter rank-2 (top-3) support was added (read topk(logits,3)[:,2] from the same spine logits, no extra lm-head) - or FLAGGED infeasible + the rank-<=1 alternative' },
    losslessByConstruction: { type: 'string', description: 'default-OFF byte-identical + drafter-packing-only (rank-2 token is a runner-up read, never enters a forward/recurrent state; masks/committer auto-derive)' },
    offlineGate: { type: 'string', description: 'the offline byte-A/B gate (default path byte-identical, 3-3-3 engages at its exact tree, fail-loud unbuilt, n_pad<=16) - CPU-validated' },
    b4Plan: { type: 'string', description: 'the B=4 depth-matched measurement plan: capture native E3/E4/E5 at B=4 as the depth bars, deploy-speed + deploy-lossless at B=4, depth-3 family->E3 / depth-5 family->E5, the exact arms + order' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const b = await agent(
  BASE + '\n\nTASK (Build, CPU, no GPU). Build the 3-3-3 shape + the drafter rank-2 support (or flag infeasible + '
  + 'alternative) in fr10_phase4_patch_vllm_tree_gdn.py, default-OFF, offline-gate it; write the B=4 depth-matched '
  + 'measurement plan. Write FR13_WIDE_SHAPES_B4_PLAN.md, commit pathspec on fr13-speedfix. Return the schema.',
  { label: 'build-333-b4plan', phase: 'Build', schema: B_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','shapeCorrect','rank2Lossless','depthMatched','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    shapeCorrect: { type: 'string', description: 'is the 3-3-3 tree_choices correct (depth-3, 3 candidates/depth) + default-OFF byte-identical + fail-loud unbuilt?' },
    rank2Lossless: { type: 'string', description: 'is the rank-2 (top-3) drafter support lossless-by-construction (a runner-up read from the same logits, never entering a forward/recurrent state) - not a reward-hack/contamination? OR honestly flagged infeasible?' },
    depthMatched: { type: 'string', description: 'is the B=4 plan correctly depth-matched (3-3-3->E3, native E3/E4/E5 captured at B=4 as the bars) + B-dependent-accept handled?' },
    recommendation: { type: 'string', description: 'single: is the 3-3-3 build + the B=4 depth-matched plan ready for the GPU sweep after OPT-1? No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  BASE + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(b) + '. Default holds=false if the 3-3-3 tree is wrong/not '
  + 'default-OFF-byte-identical, if the rank-2 drafter support is a reward-hack or contaminates a forward/recurrent '
  + 'state (it must be a pure runner-up logit read) rather than honestly flagged-infeasible, or if the B=4 plan is '
  + 'not depth-matched (3-3-3 must compare to E3 not E5) or omits capturing native E3/E4 at B=4. research-before-'
  + 'deadend. No close/pass-fail; no reward-hack.',
  { label: 'verify-333-b4', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { b, v };
