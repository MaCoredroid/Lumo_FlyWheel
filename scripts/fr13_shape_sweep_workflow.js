export const meta = {
  name: 'fr13-tree-shape-sweep-continuation',
  description: 'Boot leaner cat-tree subsets via the driver (detached+polled) and gate each through the full-stream flip-count probe + accept/event, mapping the flips-vs-topology frontier vs native(3)/cat9(22)/chain5',
  phases: [
    { title: 'Gate' },
    { title: 'Synthesize' },
  ],
}

// Shapes passed via args (so chain5's result picks them at launch). Fallback default = lean subsets of cat9.
const SHAPES = (Array.isArray(args) && args.length) ? args : [
  { name: 'cat8',   tree: '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1), (0, 0, 0, 1)]', nodes: 8, note: 'cat9 minus the deepest leaf (0,0,0,0,1)' },
  { name: 'cat7',   tree: '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0, 0), (0, 1), (0, 0, 1)]', nodes: 7, note: 'cat9 minus the two deepest leaves' },
  { name: 'chain4', tree: '[(0,), (0, 0), (0, 0, 0), (0, 0, 0, 0)]', nodes: 4, note: 'shorter pure spine (depth 4)' },
];

const CTX = `
FR13 tree-shape sweep CONTINUATION on DGX Spark GB10. GPU SERIALIZED — shapes run ONE AT A TIME. Repo
/home/mark/shared/lumoFlyWheel. Baselines (clean q3 full-stream per-token argmax-vs-no-spec-decode-oracle,
threshold 1.0 nat, same prompts): native E5 = 3 flips; cat9 = 22 flips, accept/event 3.198 (raw). chain5
(pure 5-spine) already gated separately. GOAL: map flip-count (the comparable lossless metric: each shape
vs its OWN no-spec oracle) across leaner subsets to see if any topology cuts flips toward native 3.

KEY CAVEATS (do not repeat prior errors):
- The FLIP COUNT (each shape vs its own oracle, same boot) IS comparable across shapes (a deviation rate).
- accept/event is TRAJECTORY-CONFOUNDED across boots (class-12: different greedy streams have different
  lengths/denominators). Report it as a RAW number with that flag; do NOT conclude "shape X hurts/helps
  accept" from a cross-boot whole-window delta. Per-depth rates have the sibling-stop denominator caveat.
- The 22-flip may be a per-forward channel-2 GDN-diffuse defect ORTHOGONAL to topology (cat9=22, cat10=22
  on disjoint positions). If every leaner shape also ~22, that CONFIRMS topology-independence (reshape is a
  dead end for the flip count) — a real, reportable result.

THE DRIVER (robust, reuse): scripts/fr13_shape_gate.sh <name> "<TREE>" does hygiene
(recover_host_memory + assert MemAvailable~>=100GiB + docker ps empty) -> boot forked server with TREE +
the locked FIX flags (num_spec auto-derived from len TREE) -> engagement gate (tok/draft==len TREE, FAIL
LOUD else) -> capture (within_boot_det must be [T,T,T,T], class-8 same-boot, NEVER cross-boot) -> full-stream
flip count (fr13_oracle_stream_teacher_force.py --threshold 1.0, asserts spec_metrics_delta==0) ->
accept/event from /metrics -> teardown (docker rm -f + recover) -> writes output/fr13_shape_sweep/<name>_result.json
(one JSON line) + a teardown EXIT trap. The flip step is SLOW (~20-25 min: 512 teacher-force re-prefills).
Reward-hacks BANNED; the default locked build is NOT modified (TREE-override boots only).
`;

const GATE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name','tree','num_nodes','total_clear_margin_flips','per_prompt','accept_per_event','within_boot_det','tok_per_draft','engaged','ok','notes'],
  properties: {
    name:{type:'string'}, tree:{type:'string'}, num_nodes:{type:'integer'},
    total_clear_margin_flips:{type:['integer','null']},
    per_prompt:{type:'string'}, accept_per_event:{type:['number','null']},
    within_boot_det:{type:'string'}, tok_per_draft:{type:['number','null']},
    engaged:{type:'boolean'}, ok:{type:'boolean'}, notes:{type:'string'},
  },
};

phase('Gate');
const gateResults = [];
for (const s of SHAPES) {
  const r = await agent(
    CTX + `\nTASK (GPU, the ONLY GPU boot running now — serialized). Gate shape "${s.name}" (${s.note}),
TREE=${s.tree}, expected len/num_nodes=${s.nodes}.

DO NOT wait inline on the slow gate (the prior sweep died that way). Instead:
1. First assert the GPU is free: \`docker ps\` empty + \`free -g\` available ~>=100GiB; if a container is up
   from a prior shape, the previous teardown failed — recover before booting (source .venv;
   PYTHONPATH=/home/mark/shared/lumoFlyWheel/src python3 -c 'from lumo_flywheel_serving.model_server import recover_host_memory; recover_host_memory()').
2. Launch the driver DETACHED:
   cd /home/mark/shared/lumoFlyWheel && nohup bash scripts/fr13_shape_gate.sh "${s.name}" "${s.tree}" > output/fr13_shape_sweep/${s.name}_driver.log 2>&1 &
   (returns immediately; the driver self-manages boot->gate->teardown and writes output/fr13_shape_sweep/${s.name}_result.json).
3. POLL across short bash calls (NEVER one long blocking call): repeat { bash: sleep 540; then check
   \`test -f output/fr13_shape_sweep/${s.name}_result.json && cat it\` and \`tail -3 the driver.log\` } up to
   ~4 times (~36 min cap). The flip step is ~20-25 min so expect the result by poll 3-4.
4. When ${s.name}_result.json appears, READ it and RETURN its fields as the StructuredOutput. Assert
   engaged (tok/draft==${s.nodes}) and within_boot_det [T,T,T,T]; if either fails, ok=false with notes.
5. If after the cap there is no result: tear down (docker rm -f fr13-forked-fa2-tree; recover_host_memory),
   return ok=false with the driver.log tail in notes. NEVER leave a container leaked for the next shape.`,
    { label: `gate:${s.name}`, phase: 'Gate', schema: GATE_SCHEMA, model: 'opus' }
  );
  gateResults.push(r);
  // serial: do not start the next shape until this one's container is down (the driver tears down on exit)
}

phase('Synthesize');
const SYN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','frontier','flipVerdict','acceptNote','nextAction','issues'],
  properties: {
    holds:{type:'boolean'},
    frontier:{type:'string',description:'Table: shape | nodes | total flips | per-prompt | accept/event(raw,confounded) | engaged | det. Plus the fixed points: native 3, cat9 22, chain5 (read output/fr13_shape_sweep/chain5_flips.json).'},
    flipVerdict:{type:'string',description:'Does any leaner shape cut flips toward native 3, or are they all ~22 (=> 22-flip is topology-independent / per-forward, reshape dead for the flip count)? Use the FLIP count (comparable), not accept/event.'},
    acceptNote:{type:'string',description:'Report accept/event raw per shape but FLAG it trajectory-confounded (class-12); do NOT draw a help/hurt conclusion from cross-boot whole-window deltas.'},
    nextAction:{type:'string',description:'Single next action given the frontier. No close/pass-fail.'},
    issues:{type:'string'},
  },
};
const syn = await agent(
  CTX + `\nADVERSARIALLY SYNTHESIZE. Gate results:\n${JSON.stringify(gateResults)}\n
Also read output/fr13_shape_sweep/chain5_flips.json + output/fr13_verify_decisive/q3_tree_classify.json
(cat9=22) + q3_native_classify.json (native=3) for the fixed frontier points. Build the flips-vs-topology
frontier using the FLIP COUNT as the comparable metric; flag accept/event as confounded. Conclude whether
any topology cuts flips toward 3 (=> reshape viable) or all ~22 (=> topology-independent, reshape dead for
flips, fix is the per-forward/deep-diffusion axis). Default holds=false if any shape's engagement/det gate
failed. No close/pass-fail.`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYN_SCHEMA, agentType: 'general-purpose' }
);

return { gateResults, syn };
