# FR13 Step 3 B=4 Speed Forensics

Date: 2026-06-11 UTC

Status after user pivot: frozen as evidence for pausing B=4 chase-down. The
active next gate is B=1 speed + lossless first; return to B=4 only after B=1
is near-native forward speed and lossless.

Scope: read-only speed attribution from existing Step 3 artifacts and banked FR13 docs. No GPU, Docker, or new profiling run was used.

## Verdict

The current replay-on tree arm is measurably slower per speculation forward than native, independent of its lower accept/event:

| arm | route | active draft width | decode_seconds | spec_drafts | decode_s/spec_drafts | accept/event | warm TPS |
|---|---|---:|---:|---:|---:|---:|---:|
| tree | replay-on | 9 | 255.860496 | 621 | 0.412013682 | 2.132045 | 7.566623 |
| native | native MTP-5 | 5 | 143.072108 | 544 | 0.263000198 | 2.783088 | 14.314460 |
| tree_diag_replay_off | replay-off/legacy publish | 9 | 285.237044 | 594 | 0.480197044 | 2.149832 | 6.513880 |

Measured ratios:

- Tree replay-on / native = `1.566590764x` per speculation forward.
- Tree replay-off / native = `1.825842896x` per speculation forward.
- Tree replay-on / replay-off = `0.858009617x` per speculation forward, so replay-on is about 14.2% faster than replay-off on this surface.

This is diagnostic only. Step 3 is quality-failing (`FR13_STEP3_POST_HANDOFF_BIND.md`, `FR13_STEP3_FAILURE_LOCALIZATION.md`), so these numbers are not a deployment speed verdict.

## What Is Measured

The comparable per-forward basis is the vLLM `/metrics` delta:

`request_decode_time_seconds_sum / spec_decode_num_drafts_total`

The denominators are engaged and self-consistent:

- Tree replay-on: `spec_draft_tokens/spec_drafts = 5589/621 = 9.0`.
- Native: `2720/544 = 5.0`.
- Tree replay-off: `5346/594 = 9.0`.

So this is not the banned `TPS/accept` hand-roll. It is also not a matched node-shape comparison: one tree speculation event verifies 9 draft nodes, while one native event verifies 5 draft tokens. The metric is "time per speculation event", not "time per verified row".

## Replay-On Engagement

Replay-on appears genuinely engaged in the primary Step 3 tree arm:

- `output/fr13_step3_b4_gate/tree/engagement_asserts.json` has `FR13_REPLAY_ROUTE=1`, `FR10_METRICS=0`, `VLLM_BATCH_INVARIANT=0`, `FR13_BI_TREE_ATTN=0`, and no failures.
- `output/fr13_step3_b4_gate/tree/cuda_graph_proof.txt` shows `TREE_ATTN/tree_mtp`, `num_spec_tokens=9`, `enforce_eager=False`, `PIECEWISE=8 (largest=80), FULL=4 (largest=40)`, and graph capture completion.
- Native also FULL-captured: `PIECEWISE=7 (largest=48), FULL=4 (largest=24)`.

The tree is not losing FULL capture in this run. Tree's larger FULL shape is expected from B=4 times 10 verify rows (root + 9 draft nodes), versus native B=4 times 6 rows (root + 5 draft tokens).

## HBM Tax Check

Do not collapse this into the old replay-off HBM-state tax. Current evidence says replay-on removed the large legacy materialization path:

- `FR13_FLAGS.md` documents replay-on as compiling out per-node scratch export, removing the 201 MB/layer per-step allocation, using committer replay for accepted durable state, and skipping the SSM half of next-step remap.
- `scripts/fr10_phase4_patch_vllm_tree_gdn.py` sets `tree_state_all = None` under `FR13_REPLAY_ROUTE=1`, passes `state=None`, and calls `launch_tree_gdn_prepared(..., store_node_states=not _fr13_replay_route_on)`.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` documents `store_node_states=False` as eliding the per-node HBM state export and skipping the scratch allocation. The Triton scan's state store is guarded by `STORE_NODE_STATES`.
- The post-scan all-row `ssm_state.index_copy_` publish is skipped when replay route is on. The committer instead launches `launch_tree_gdn_replay` for registered GDN layers.
- Direct artifact comparison agrees: replay-on is faster than replay-off by about 14.2% per speculation forward, and graph capture memory is lower (`1.28 GiB` replay-on vs `1.66 GiB` replay-off).

Residual HBM work still exists, but it is not the known legacy all-N state materialization tax:

- Replay-on stages persistent activation rings (`k`, `v`, `raw_a`, `raw_b`) for all tree nodes. The code comment sizes this at about `16.2 KiB/node`, versus about `3.146 MiB` per state row.
- Replay-on replays root + accepted path into the durable state bank after commit.
- The verify scan still writes tree outputs and computes all nodes. Replay-on changes durable-state logistics, not the fact that the tree verifier has a wider forward.

Current artifacts do not prove a hidden full-state HBM export remains in replay-on.

## Likely Slower Area

The most likely remaining source is tree-width work in the GDN/linear-attention path, not accept/event and not a FULL-capture failure.

Evidence:

- Draft trace shapes are `[*, 9]` for tree and `[*, 5]` for native. The tree graph's FULL capture largest is 40, native largest is 24, matching the 10-row vs 6-row verify shape at B=4.
- The tree GDN scan still loops over `N_PAD=16` with an `h_cache` in `_tree_gdn_kernel`, and it is called once per spec-decode row in the patched GDN forward. Replay-on removes state export but not the pad16/tree-node scan work.
- Banked `FR13_CONV_PRIOR_SLOT_FIX_BIND.md` measured a clean `/metrics`-only prompt0 profile at `1.412x` tree/native per-forward and attributed the measured gap mostly to node count (`9/6 = 1.50x`), while explicitly saying GDN state traffic versus FA2/tree-attn could not be isolated without a kernel trace.
- `FR13_OVERHEAD_DECOMP_PLAN.md` keeps TREE_ATTN-vs-FLASH and pad16/GDN costs as unmeasured suspects. It names a top-kernels profiler table as the discriminator.

So the likely module is:

`language_model.model.layers.*.linear_attn` / `vllm::gdn_attention_core`, specifically `_tree_gdn_kernel` scan work plus replay-route staging/replay overhead.

Full-attention `TREE_ATTN` may also contribute because it handles the same 10-row tree shape instead of native's 6 rows, but no current artifact separates that from the 48 GDN layers. Exact layer/module attribution is not measured.

## Artifact That Would Prove It

Smallest proof artifact: one short diagnostic profiler trace under the Step 3 shape, not a broad gate:

- B=4, BI=0, `FR10_METRICS=0`, same tree/native configs, one pinned prompt, short `max_tokens` window.
- Capture either an eager `torch.profiler` trace or an `nsys`/NVTX trace that can group CUDA time by kernel name and layer/module.
- Required table: cumulative CUDA time and call count for `_tree_gdn_kernel`, `_tree_gdn_replay_kernel`, ring `copy_`/memcpy work, native `fused_sigmoid_gating_delta_rule_update`, TREE_ATTN/FLASH_ATTN kernels, and sampler/committer kernels.

Interpretation:

- If `_tree_gdn_kernel` + replay/staging accounts for the delta, the remaining tax is GDN/tree-width/kernel-count, not legacy HBM state export.
- If TREE_ATTN kernels account for the delta, the full-attn backend/row-shape is slower.
- If NCU/NSYS shows large DRAM writes matching all-node fp32 state rows under replay-on, that would overturn this doc and prove residual HBM-state tax. Current code/artifacts do not show that.

Secondary useful artifact: vLLM per-step CUDA graph dispatch metrics (`cudagraph_metrics=true`, `enable_logging_iteration_details=true`) for the same configs. That would prove runtime FULL/PIECEWISE/eager dispatch mix, but it still would not name the slow layer without the kernel table.

## Remaining Risk

Step 3 quality is failing and no Step 3 layer profiler exists. The current evidence is enough to reject "replay-on is just paying the old replay-off HBM tax" and to point at tree-width GDN/linear-attention as the likely slower surface. It is not enough to name the exact layer or quantify GDN versus TREE_ATTN versus replay committer overhead.
