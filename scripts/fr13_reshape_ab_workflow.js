export const meta = {
  name: 'fr13-reshape-ab',
  description: 'RESHAPE A/B (user chose option 1): the carrier-reopen H-FORK-AMPLIFICATION says the cat9 23-vs-native-3 flip gap is a small two-kernel verify-forward floor AMPLIFIED by tree topology. Test the ONLY remaining flip-count lever: boot cat9 with a SHALLOWER tree - cat3w (depth-3 spine + root sibling, _fr10_cat3w_choices) AND chain3 (pure depth-3, _fr10_chain3_choices) to separate depth-from-width - temp0 seed1313 same prompts_swe4, re-score vs the SAME proven recurrent oracle (fr13_recurrent_decode_oracle). Report BOTH (a) de-cascaded independent flips vs OFF=18/native=3 AND (b) e2e accept/event vs E5 (the BINDING arbiter; raw flip = fork-inflated class #12). PROVE non-vacuous (oracle engaged + reshape actually applied) before any number. Discriminator: flips DROP toward native = topology IS the lever; NO drop = H refuted, carrier is depth-independent floor. Adversarial verify.',
  phases: [
    { title: 'ReshapeBoot' },
    { title: 'Verdict' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121). Repo /home/mark/shared/lumoFlyWheel. GPU SERIALIZED. Pre-boot hygiene:',
'source .venv; recover_host_memory(); assert MemAvailable>=100GiB + docker ps empty. Teardown trap: docker',
'rm -f the container + recover_host_memory between every boot. boot ENFORCE_EAGER=1.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (0.19.2rc1.dev134), NEVER a /tmp cache.',
'COMPARE TARGET (user): lossless = cat9 vs native-E5 each-vs-its-own-no-spec-oracle; the BINDING arbiter is',
'e2e accept/event vs E5 (cat9 OFF already ~3.0/native 3.08); the raw flip count is fork-inflated + length-',
'sensitive (class #12) so report it de-cascaded + as a per-1000 rate, NOT raw. int-view NEVER atol. oracle =',
'the RECURRENT no-spec decode (fr13_recurrent_decode_oracle), NOT prefill/chunked.',
'',
'CONTEXT (FR13_CARRIER_REOPEN.md, verify HOLDS): the cat9 23-vs-native-3 flip gap is NOT a single per-forward',
'kernel seam (GDN scan ruled out e2e by FR13_SCAN_NOT_E2E_CARRIER_BIND; FA2-fork = 2-ULP floor). It is ~16',
'confident structural-boundary forks + ~5 ctrl-basin progeny + ~2 true near-ties, driven by cat9 running TWO',
'extra divergent verify kernels (forked-FA2 tree-bias + GDN tree-scan) vs native single FLASH spine, AMPLIFIED',
'by the tree topology (more divergent forwards/step + crossings forking into degenerate basins). H-FORK-',
'AMPLIFICATION predicts a SHALLOWER/root-sibling tree reduces the amplification → fewer de-cascaded flips.',
'MONITOR RED-TEAM CAVEAT to test honestly: reshape was previously banked EXHAUSTED, chain5(deep)→2-de-cascaded',
'BEAT chain3(shallow)→5-dispersed (deeper was better, which CONTRADICTS shallower-helps), and width adds',
'co-residency (cat3w=25 in a prior raw count). So this A/B may REFUTE H - report the honest result either way.',
'',
'NON-VACUITY (mandatory, the session burned 4 vacuous instruments): PROVE before any number - (a) the RESHAPE',
'actually applied: the served tok/draft per step matches the reshaped node count (cat3w/chain3 have FEWER',
'nodes than cat9 caterpillar 9), assert the speculative_token_tree the engine logs == the reshaped choices,',
'fail-loud if it is still cat9; (b) the recurrent oracle ENGAGED: RECURRENT_PATH_ENGAGED=True +',
'_forward_core_decode_non_spec counter increments (reuse scripts/fr13_recurrent_decode_oracle.py, PROVEN); (c)',
'within-boot det [T,T,T,T].',
'',
'YOUR JOB:',
'PHASE 1 (ReshapeBoot, GPU): FIRST read how the tree topology is selected - the launcher fr13_launch_locked.sh',
'speculative_token_tree (the cat9 caterpillar string) + the committed _fr10_cat3w_choices=[(0,),(1,),(0,0),',
'(0,1),(0,0,0)] / _fr10_chain3_choices in fr10_phase4_patch_vllm_tree_gdn.py (~:10747) and whatever env/flag',
'selects them (or directly set the speculative_token_tree to the reshaped tuple list). Then, hygiene + boot,',
'EAGER, temp0 seed1313 prompts_swe4:',
'  ARM cat3w (depth-3 spine + root sibling): capture served streams; assert reshape applied + det.',
'  ARM chain3 (pure depth-3, no width): capture served streams; assert reshape applied + det.',
'  (Reuse the banked OFF cat9 = 23 raw / 18 de-cascaded and native-E5 = 3 from output/fr13_scan_align_rerun/',
'   logs/ for the baseline/bar - SAME recurrent oracle, so comparable; re-capture OFF only if a fresh same-',
'   session baseline is needed.)',
'Rescore cat3w + chain3 served streams vs the SAME fr13_recurrent_decode_oracle: per-token argmax clear-margin',
'flips (deviation_nat>1.0, gold-margin NOT streamed top_logprobs) -> raw + de-cascaded (gap<=2, the FR13_PLUS2',
'rule) + per-1000 rate; AND the e2e accept/event for each arm (from the served stream / metrics). Teardown +',
'recover after every boot.',
'PHASE 2 (Verdict). DISCRIMINATOR: (H CONFIRMED) cat3w/chain3 de-cascaded flips DROP toward native 3 (vs OFF',
'18) AND accept/event holds or improves vs E5 => tree topology IS the lossless lever -> the path is reshape',
'(bring the accept/event + flip table to the user; bake = user call). (H REFUTED) flips do NOT drop (or width',
'cat3w is worse via co-residency) => the carrier is the depth-independent two-kernel verify-forward floor, NOT',
'topology -> reshape is dead, the remaining lever is the FA2-fork/diffuse-GDN floor (option 3) or accept the',
'accept/event-parity (option 2) - bring to user. Separate DEPTH (chain3) from WIDTH (cat3w root-sibling) in the',
'conclusion. Reward-hacks BANNED (vary ONLY topology, kernels/seed/prompts fixed; no copy/dense/forced-spine;',
'NOT-lossless multi-spine stays CLOSED). Quote FR13_BUG_CLASS_PLAYBOOK rows (#12 co-residency/trajectory, #9).'
].join('\n');

phase('ReshapeBoot');
const RB_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['topologySelectMechanism','reshapeApplied_cat3w','reshapeApplied_chain3','oracleEngaged','within_boot_det','cat3w_flips_raw','cat3w_flips_decascaded','cat3w_accept_per_event','chain3_flips_raw','chain3_flips_decascaded','chain3_accept_per_event','off_baseline','native_bar','ok','notes'],
  properties: {
    topologySelectMechanism: { type: 'string', description: 'how the tree topology was set to cat3w/chain3 (env/flag/launcher edit) + the actual choices used' },
    reshapeApplied_cat3w: { type: ['boolean','null'], description: 'cat3w reshape PROVEN applied (engine speculative_token_tree == cat3w, tok/draft matches node count, not still cat9)?' },
    reshapeApplied_chain3: { type: ['boolean','null'] },
    oracleEngaged: { type: ['boolean','null'], description: 'recurrent oracle RECURRENT_PATH_ENGAGED=True + counter incremented?' },
    within_boot_det: { type: 'string' },
    cat3w_flips_raw: { type: ['integer','null'] },
    cat3w_flips_decascaded: { type: ['integer','string','null'], description: 'cat3w de-cascaded indep flips vs OFF=18/native=3' },
    cat3w_accept_per_event: { type: ['number','null'] },
    chain3_flips_raw: { type: ['integer','null'] },
    chain3_flips_decascaded: { type: ['integer','string','null'] },
    chain3_accept_per_event: { type: ['number','null'] },
    off_baseline: { type: 'string', description: 'OFF cat9 baseline used (banked 23 raw/18 de-cascaded, or re-captured)' },
    native_bar: { type: 'string', description: 'native-E5 bar (banked 3)' },
    ok: { type: 'boolean' },
    notes: { type: 'string' },
  },
};
const rb = await agent(
  CTX + '\n\nTASK (ReshapeBoot, GPU). Do PHASE 1. Prove reshape applied + oracle engaged BEFORE any number. '
  + 'Report cat3w + chain3 flips (raw + de-cascaded + rate) + accept/event vs the OFF=18/native=3 baseline. '
  + 'Teardown + recover. Return the schema.',
  { label: 'reshape-boot', phase: 'ReshapeBoot', schema: RB_SCHEMA, model: 'opus' }
);

phase('Verdict');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','reshapeProven','flipsDropped','depthVsWidth','topologyIsLever','accept_per_event_summary','nextAction','rewardHackCheck','issues'],
  properties: {
    holds: { type: 'boolean' },
    reshapeProven: { type: 'string', description: 'was the reshape PROVEN applied (not still cat9) + oracle engaged - non-vacuous?' },
    flipsDropped: { type: ['boolean','string','null'], description: 'did cat3w/chain3 de-cascaded flips DROP toward native 3 (vs OFF 18)?' },
    depthVsWidth: { type: 'string', description: 'chain3 (depth) vs cat3w (depth+width) - which helped/hurt; does width add co-residency?' },
    topologyIsLever: { type: ['boolean','null'], description: 'is tree topology the lossless lever (H confirmed) or refuted (carrier is the depth-independent floor)?' },
    accept_per_event_summary: { type: 'string', description: 'accept/event for cat3w/chain3 vs E5 (the binding arbiter)' },
    nextAction: { type: 'string', description: 'if H confirmed: pursue reshape (user bake call). if refuted: option 2 (accept parity) or 3 (chase floor) - bring to user. No close decision here.' },
    rewardHackCheck: { type: 'string', description: 'only topology varied; no copy/dense/forced-spine; multi-spine stays closed' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(rb) + '. Default holds=false if the reshape was NOT proven '
  + 'applied (engine still cat9 = vacuous) / oracle not engaged / flip number from streamed top_logprobs. '
  + 'Conclude honestly: did topology drop the flips (H confirmed) or not (refuted - reshape dead, despite being '
  + 'the user-chosen test). Separate depth from width. Report accept/event (the binding arbiter). No close/pass-'
  + 'fail decision; no reward-hack.',
  { label: 'reshape-verdict', phase: 'Verdict', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { rb, v };
