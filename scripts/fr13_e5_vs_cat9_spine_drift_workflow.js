export const meta = {
  name: 'fr13-e5-vs-cat9-spine-drift',
  description: 'USER (2026-06-15, scientific spirit): rigorously study WHY E5 drifts only ~3 vs cat9 ~23 (precisely: E5-SPINE 3 vs cat9-SPINE 17, after the superset gate stripped the 6 net-positive leaf-saves). Is our math FUNDAMENTALLY harder, or is there still a LEVER? Both verify the SAME MTP spine, but E5 = FLASH attention + native MTP-5 GDN scan, cat9 = forked-FA2 tree-bias + GDN tree-scan = TWO different kernels. The GDN scan is now near-native (K1 proof rel-err 2e-4), so the prime suspects are the FA2-FORK (vs FLASH) + the inter-layer connection-amplification. ALSO refine the "diffuse" claim PRECISELY: is the drift IN the GDN-layer compute, or AMPLIFIED through the RESIDUAL-STREAM CONNECTIONS between layers (the diffusion deep-dive found 1.166x/layer signal-proportional growth = residual stream, NOT each layer wrong in isolation), or the deep full-attn? Per-component attribution (GDN compute / gate 1/rms / residual adds / layer-norms / full-attn / lm-head) of cat9-spine-vs-native + E5-spine-vs-native, then the LEVER verdict (fixable seam to drive 17->3, or fundamental). research-before-deadend (E5=3 exists = a 3-flip realization is reachable). CPU read-only, code + git-history + per-layer banked captures + online, adversarial verify. Output FR13_E5_VS_CAT9_SPINE_DRIFT.md.',
  phases: [
    { title: 'Study' },
    { title: 'Verify' },
  ],
}

const CTX = [
'FR13 on DGX Spark GB10 (sm_121, bf16/fp8 GDN-hybrid Qwen3-Next: ~48 GDN linear-attn + 16 full-attn layers).',
'Repo /home/mark/shared/lumoFlyWheel. CPU-ONLY, READ-ONLY (a big-denom GPU serve runs concurrently; do NOT edit',
'code/boot). Read our code + GIT HISTORY + the BANKED per-layer ladders + vLLM source via scripts/vllm_src.sh +',
'search online. Write ONLY FR13_E5_VS_CAT9_SPINE_DRIFT.md. Pathspec.',
'',
'GROUNDING RULE (user): read vLLM source via scripts/vllm_src.sh (pinned 3dbe092e), NEVER /tmp. int-view NEVER',
'atol. Compare target = the deployment-correct RECURRENT decode oracle (fr13_recurrent_decode_oracle).',
'',
'THE QUESTION (user, scientific): E5 (native MTP-5) drifts ~3 clear-margin flips vs its no-spec decode; cat9',
'drifts ~23 = 17 SPINE + 6 leaf (FR13_PEREVENT_SUPERSET_GATE_RESULT: the 6 leaf-saves are NET-POSITIVE/lossless,',
'so the REAL residual is the 17 SPINE-realization flips). Both verify the SAME MTP spine. So the precise puzzle:',
'WHY does cat9-SPINE drift ~17 vs E5-SPINE ~3 (a ~14 excess)? Is the math FUNDAMENTALLY harder (the tree-batched',
'verify forces it) or is there a LEVER? E5=3 PROVES a 3-flip realization at this model/fp8 exists = NOT',
'irreducible. AND: the drift is called "diffuse" - locate it PRECISELY (in the GDN-layer compute vs the inter-',
'layer RESIDUAL-STREAM CONNECTIONS vs the full-attn), not hand-waved.',
'',
'WHAT IS BANKED (build on, do not redo): FR13_DIFFUSION_DEEP_DIVE (per-layer node5/node7 ladders, MEASURED:',
'first-nonzero L0 GDN, geometric ~1.166x/layer riding the RESIDUAL STREAM (signal-proportional, NOT additive-',
'ULP-per-layer), largest jumps deep FULL-ATTN L35/47/51/62, crystallizes L60/61); K1 mechanism proof (the',
'deployed GDN SCAN STATE is ALREADY near-native rel-err 2.2e-4, so the scan is NOT the main 17-source); N_PAD',
'null (spine reduction order co-residency-invariant); FA2-fork = "2-ULP floor, no depth growth" (project_fr13_',
'fa2_fork_nocopy_floor, FR13_FA2_CARRIER_OVERTURNED) BUT that floor could COMPOUND through the residual stream;',
'fp8 in_proj/o_proj M-invariant; conv anchor-row 1-ULP. The two extra cat9 kernels vs E5 = forked-FA2 tree-bias',
'(scripts/fr13_patch_fa2_tree_bias.py, vs native FLASH_ATTN) + GDN tree-scan (vs native MTP-5 chunk scan).',
'',
'YOUR JOB:',
'1. THE 3-vs-17 OP-BY-OP: compare E5\'s spine verify forward (FLASH_ATTN + native MTP-5 GDN multi-token scan,',
'   read via vllm_src.sh + the banked native captures output/fr10_native_mtp5_same8_*) vs cat9\'s spine verify',
'   (forked-FA2 tree-bias + GDN tree-scan, our kernels). On the SAME spine, which kernel/op makes cat9 drift',
'   MORE from the decode oracle than E5 does? Candidates with the banked evidence: (a) FA2-fork vs FLASH (the',
'   full-attn realization - is the forked additive -inf bias / exp2 softmax / MMA grouping farther from decode',
'   than FLASH? the 2-ULP floor x how-many-full-attn-layers x residual amplification); (b) GDN tree-scan vs',
'   native MTP-5 scan (near-native per K1 but quantify the residual); (c) the COMPOUNDING (E5 and cat9 both have',
'   a per-layer floor, but does cat9\'s start bigger at L0 and ride the same 1.166x, or amplify faster?). Pin the',
'   dominant source of the ~14 excess.',
'2. THE WHERE (diffuse, precisely): from the banked per-layer ladders, is the drift (a) BORN in the GDN-layer',
'   COMPUTE (in_proj/conv/scan/gate/o_proj - which?), (b) AMPLIFIED in the inter-layer RESIDUAL-STREAM',
'   CONNECTIONS (the residual add + the next layer\'s input-norm 1/rms reading a drifted residual = the 1.166x/',
'   layer; this is the "connection" the user asks about), or (c) the deep FULL-ATTN layers (the L35/47/51/62',
'   jumps). Decompose the 1.166x/layer: is it the layer-output ratio or the residual-stream ratio? Is the',
'   amplification a CONNECTION effect (gate 1/rms reading a drifted residual) or a per-layer compute effect?',
'   MEASURED from the ladders, distinguish from INFERRED.',
'3. THE LEVER VERDICT (research-before-deadend, do NOT conclude fundamental without it): given the dominant',
'   source, is there a fixable lever to drive cat9-spine 17 -> ~3 (E5 floor)? e.g. align forked-FA2 closer to',
'   FLASH (find the exact divergent op: softmax-scale / accum-dtype / online-rescale / qk-cast - alignable to',
'   0.0 or real algorithmic diff?); a compensated residual/connection fix; the deep-full-attn realization. OR is',
'   it fundamental (the tree-batched verify intrinsically drifts more, no E5-level realization for a TREE)? Be',
'   quantitative: name the lever + expected reach, or show why no lever exists with the per-component numbers.',
'   Note: NOT K1/N_PAD (done), NOT WY (parked), NOT bonus (rejected), NOT copy/dense.',
'',
'DELIVERABLE: FR13_E5_VS_CAT9_SPINE_DRIFT.md = the 3-vs-17 op-by-op attribution (dominant source of the ~14',
'excess), the precise WHERE (GDN-compute vs residual-stream-connection vs full-attn, with the 1.166x decomposed),',
'and the LEVER verdict (a named fixable seam + expected reach, or fundamental-with-numbers). Distinguish MEASURED',
'/CODE-READ from INFERRED. Quote FR13_BUG_CLASS_PLAYBOOK (#12 depth-accumulation, #10 codegen). Commit pathspec.',
].join('\n');

phase('Study');
const S_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['threeVs17Attribution','whereDiffuse','leverVerdict','fundamentalOrFixable','minimalValidatingExperiment','committed','notes'],
  properties: {
    threeVs17Attribution: { type: 'string', description: 'op-by-op E5-spine (FLASH+native-scan) vs cat9-spine (forked-FA2+tree-scan): which kernel/op makes the ~14 excess; FA2-fork vs FLASH / tree-scan residual / compounding, with banked evidence' },
    whereDiffuse: { type: 'string', description: 'precise WHERE: GDN-layer compute vs inter-layer RESIDUAL-STREAM connection (the 1.166x decomposed - layer-output vs residual ratio, is the amp a gate-1/rms-reading-drifted-residual connection effect) vs deep full-attn; MEASURED from the ladders' },
    leverVerdict: { type: 'string', description: 'is there a fixable lever to drive cat9-spine 17->3? named seam (align forked-FA2 to FLASH - the exact divergent op / compensated connection / full-attn) + expected reach, OR fundamental-with-numbers' },
    fundamentalOrFixable: { type: 'string', description: 'FIXABLE (a lever exists, E5=3 reachable) or FUNDAMENTAL (tree-batched verify intrinsically drifts more), with the per-component numbers' },
    minimalValidatingExperiment: { type: 'string', description: 'the EXACT minimal GPU experiment to validate the top lever if one is found' },
    committed: { type: 'string' },
    notes: { type: 'string' },
  },
};
const s = await agent(
  CTX + '\n\nTASK (Study, no GPU, read-only). SEARCH ONLINE for FLASH-vs-tree-attn + SSM realization-floor work, '
  + 'then op-by-op E5-vs-cat9 spine + the per-layer ladder WHERE + the lever verdict. Write FR13_E5_VS_CAT9_SPINE'
  + '_DRIFT.md, commit pathspec. Return the schema.',
  { label: 'e5-vs-cat9-spine-drift', phase: 'Study', schema: S_SCHEMA, model: 'opus' }
);

phase('Verify');
const V_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['holds','attributionGrounded','whereSound','leverHonest','recommendation','issues'],
  properties: {
    holds: { type: 'boolean' },
    attributionGrounded: { type: 'string', description: 'is the 3-vs-17 attribution from actual code+banked captures (cited), not narrative? does it correctly exclude the now-near-native scan?' },
    whereSound: { type: 'string', description: 'is the GDN-compute-vs-residual-connection-vs-full-attn WHERE backed by the ladder numbers (1.166x decomposed), not re-asserting "diffuse"?' },
    leverHonest: { type: 'string', description: 'is the lever verdict honest - a real named seam with reach, OR fundamental backed by per-component numbers (not a premature no-go, research-before-deadend)?' },
    recommendation: { type: 'string', description: 'single: is there a lever to drive 17->3 (pursue it) or is it fundamental (the per-event superset + SWE-quality is the deliverable). No close/pass-fail.' },
    issues: { type: 'string' },
  },
};
const v = await agent(
  CTX + '\n\nADVERSARIALLY VERIFY: ' + JSON.stringify(s) + '. Default holds=false if the 3-vs-17 attribution is '
  + 'narrative not code/capture-grounded, the WHERE just re-asserts "diffuse" without decomposing the 1.166x '
  + '(GDN-compute vs residual-connection vs full-attn), the lever verdict is a premature "fundamental" without '
  + 'per-component numbers (research-before-deadend: E5=3 proves reachable), or it re-proposes K1/N_PAD/WY/bonus. '
  + 'No close/pass-fail; no reward-hack.',
  { label: 'verify-e5-vs-cat9-spine-drift', phase: 'Verify', schema: V_SCHEMA, agentType: 'general-purpose' }
);

return { s, v };
