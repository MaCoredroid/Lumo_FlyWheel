# Optimization route audit — 2026-09-22

**Decision:** fresh E1 measures Cat10 with single-logits draft reuse and the device multidraft committer active. It does **not** measure the fixed32 folded GDN candidate, batched TAW, or forked-FA2 grouping/spine-first storage. The smallest compatible new attribution experiment is single-logits reuse ON/OFF, after an explicit source switch and bounded same-input correctness qualification. No GPU work, inference, source changes, or automation changes were performed for this review. Only this report was written.

## Evidence names and scope

All paths below are relative to `papers/gdn-tree-scan-mlsys/v2`, except paths explicitly labeled repository-relative.

- `R = experiments/out-20260922T100028Z-e1-18cells`.
- `C1 = R/cell_05_b1_tree_B1_a1`; `C4 = R/cell_06_b1_tree_B4_a1` (the first timed tree cells).
- `L = C1/loaded_backend`; these are the actual loaded Python modules, not current patcher assumptions.
- `D = p0/monitor/e1-source-20260922T1150Z`; its preserved repository files are bound to the campaign identity.

Both Docker inspections and boot logs establish stock image `sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776`, TREE_ATTN, eager, prefix-cache off, synchronous scheduling, Cat10 (nine draft nodes plus root; depth five), and actual configured B1/B4. The actual occupancy/API/interval audit remains `p0/monitor/e1-remaining-cells-redteam.{md,json}`. Boot arguments are at both `docker_logs.txt:11`, backend selection at line 43. E1 is a patched serving implementation inside the stock image, not pristine upstream vLLM.

| Configuration in both tree cells | Observed value |
|---|---|
| `FR13_ATTN_KV_REMAP / FR13_SLOT_REORDER / FR13_KV_REMAP_SYNCFREE` | `1 / 0 / 1` (policy B, flat storage) |
| `FR13_REPLAY_ROUTE / FR13_EAGER_PACK / FR13_TREE_CONV_FUSED` | `1 / 1 / 1` |
| `FR13_DRAFTER_SINGLE_LOGITS` | `1`; loaded source also bakes it on |
| `FR13_DEVICE_MULTIDRAFT` | absent; loaded default is **1** |
| `FR13_FIXED32_MODE`, `FR13_TAW`, `FR13_DM_DEPTHSYNC`, `FR13_STEP_GRAPH` | absent |

## Design → implementation → measured route

| Candidate | Exact implementation/guard | E1 disposition and missing evidence |
|---|---|---|
| Single-launch GDN fold | `D/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:16994–17045`: folded candidate requires validated fixed32 schedule, exact 32 actual/padded nodes, BV8 and candidate contract; batched selection at `4270–4287`. | **Inactive, different geometry.** `L/gdn_linear_attn.py:14632–14648` enters batched fixed32 only with `_FR13_FIXED32_MODE`; E1 instead calls the generic per-request scan at `14765–14848`. E1 Cat10 is actual10/pad16. Porting the fixed32 fold is not a switch-only ablation. |
| Batched tree accept walk (TAW) | `D/scripts/fr13_device_multidraft_kernel.py:1034–1058` dispatches generic TAW/depthsync only when `not all_greedy`; fixed32 entry at `1010–1023`, with explicit greedy refusal at `6514–6518`. | **Inactive/ineligible for the present greedy route.** E1 enters canonical multidraft with `all_greedy=True` at `L/rejection_sampler.py:3762–3775`. Setting TAW=1 alone is vacuous. A greedy batched implementation would need a new qualified route. |
| FA2 grouping / spine-first storage | Repository `scripts/fr13_launch_forked_fa2_tree_server.sh:1745–1759` requires fixed32 B4/concurrency4 and a sealed dual gate before selecting grouped FA2; candidate scope is declared at `1699–1700`. | **Inactive, different backend.** `L/tree_attn.py:30,1152–1172` executes Triton `unified_attention` with tree bias and block table, not the forked FA2 grouped kernel. `SLOT_REORDER=0` preserves flat storage. The drafter's causal spine is active, but that does not imply spine-first attention storage. |
| Device multidraft | `L/rejection_sampler.py:2602–2666,2725–2761` defaults device committer on for deterministic drafts and dispatches the preserved `D/scripts/fr13_device_multidraft_kernel.py`. | **Active B1/B4.** Engagement at `C1/docker_logs.txt:283` and `C4/docker_logs.txt:282`. Probability/point-mass tensor arithmetic stays on the device; host control remains. No fresh isolated cost attribution exists. |
| Single-logits draft reuse | `L/eagle.py:1486–1495` is baked `True and not use_local_argmax_reduction`. Root/loop reuse at `4719–4785` and `5273–5329`; `_greedy_sample` recomputes logits at `568–572`. | **Active B1/B4.** Both logs line280 report `single_logits=True`, local argmax reduction false. Root/loop reuse avoids the legacy duplicate head evaluation. There is no fresh ON/OFF attribution; an environment-only OFF is ineffective. |
| EAGER_PACK stacked buffers | `L/gdn_linear_attn.py:8501` and `L/rejection_sampler.py:882` bake packing on. The forward uses its path at `gdn_linear_attn.py:11797` and avoids a dead cast at `11913–11918`. | **Stacked storage/preparation active.** Both tree boot logs `210,212` report stacked rings rebuilt to a union of 48 layers. **All-layer replay is a separate dispatch and is not established by this allocation log.** See the guard distinction below. |
| Fused tree convolution | `L/gdn_linear_attn.py:8514` bakes fusion on; its actual first-decode needle at `11877–11904` reports the runtime branch, prepared rows, and static tables. The imported helper is `D/src/lumo_flywheel_serving/fr13_tree_conv_fused.py`. | **Active B1/B4.** Both logs line281 report `fused=1 tree_n=10 width=4 state_len=12 prepared_rows=1 static_tables=1 zero_row_cached=1`; preparation for all three groups appears at `209,211,213`. Fresh E2 helper/composition evidence supports the scoped route, but no isolated fusion timing contrast exists. |

The GDN terminology needs care: the **fixed32 fold** is inactive, but E1's ordinary Cat10 monolithic scan itself launches `_tree_gdn_kernel` once per request/layer (`D/.../fr10_gdn_tree_kernel.py:17356–17362,17763–17765`). The subtree alternative has an explicit opt-in (`1070–1094,3222–3224`). Do not imply that E1 has the fixed32 two-level reference schedule merely because the fold is absent, or that E1 B4 uses one folded launch for the whole batch. Replay of accepted states is additional work.

The packed-buffer/all-layer-replay distinction is similarly material. Greedy now uses canonical multidraft (`L/rejection_sampler.py:3762–3775`). In that function, `_fr13_sbr_active` at `3219–3232` additionally requires SAMPLED_REPLAY_BATCHED/COMMITTER_NATIVE_BATCHED/COMMITTER_GRAPH or a named marker, plus prepared stacks and exclusions. Both inspected E1 Docker environments have none of these flags; neither archived cell contains an `.arm` file. The unarmed route enumerates layers at `3517–3522,3559` and calls per-layer replay at `3615–3630`; the all-layer call at `3287` is guarded. The old nearby comment that greedy “already batches” is not a substitute for following the unified call route. Do not claim the fresh E1 committer executes one replay launch over all layers from EAGER_PACK=1 alone. These are source/config observations, not a newly acquired profiler launch census.

## Material wording/ablation traps

1. **The active committer is not a zero-readback or host-loop-free walk.** Its engagement log says “no per-node Python loop,” but `D/scripts/fr13_device_multidraft_kernel.py:1075–1079` materializes drafts/counts, `1092` loops requests, `1116–1152` loops nodes and selects Python indices, and `1132–1134` materializes the sampled bonus. The node helper also uses scalar `.item()` decisions. In greedy mode it uses one-hot argmax rows (`1087–1089`), so the stochastic softmax-transfer rationale must not be reported as a measured E1 softmax saving. The host greedy fallback already computes argmax before CPU transfer (`L/rejection_sampler.py:2650–2663`).

2. **Device ON/OFF can change selected duplicate branches.** Greedy token selection does not guarantee a unique accepted path: duplicate draft siblings use a random source selection. `p0/monitor/e2-b4-duplicate-path-review.md:12–36` records the actual p085 witness; its CPU reproducer is at `88–102`. Device generator construction is at `D/scripts/fr13_device_multidraft_kernel.py:1098–1108`; host fallback instead creates a NumPy generator from a Torch draw (`L/rejection_sampler.py:2783–2806`). A naive ON/OFF timing comparison can change accepted length/work as well as placement. The old `scripts/fr13_greedy_pointmass_dup_gate.py:10–18` FIRST-child assumption is not the present contract. Do not use that gate as proof of matched paths. Retain validation of the actual published path used by `experiments/e2/e7b_b4_capture_gate.py`.

3. **A spine-first flag is not a legal “slow/fast” pair under stock TREE_ATTN.** The fresh policy-A failure and corrected flat-map policy-B evidence are retained at `results/e2-policyA-first-forward-audit.json` and `p0/monitor/e2-b4-final-redteam.md`. Re-enabling reordered writes without a matching read-address contract changes semantics. Forked FA2 qualification cannot be transferred to this backend by naming the same flag.

4. **Current repository bytes are not all the E1 bytes.** The current patcher and GDN kernel differ from `D`; source assertions about E1 above use `D` and `L`. Any new ablation must preserve and identify its own emitted source. Existing wiring tests passed on the current patcher, but do not prove that an environment OFF selects a different branch.

## Smallest necessary next experiment

**Prioritize single-logits reuse only.** Keep Cat10, stock TREE_ATTN policy B, synchronous eager/cache-off execution, model/input identity, precision and all other tree mechanisms fixed. Introduce one explicitly named source-controlled ON/OFF selector that reaches the existing legacy double-logits branch while keeping `use_local_argmax_reduction=False`. Do not use that reduction option as a substitute switch: it selects `get_top_tokens` and changes selection semantics. Do not reuse old timing or the completed E1 campaign as the new control arm.

Eligibility/stopping rule, frozen before new timing:

- First prove the emitted ON/OFF guards and call sites differ only as intended, and record both source hashes plus engagement. `tests/test_fr13_drafter_single_logits_wiring.py` is an existing CPU wiring starting point. Its five current checks passed, including the assertion that the current path is baked on; update the bounded switch expectations in the future ablation revision.
- Use the existing **untimed** `FR13_FIX1_SELFCHECK` at `L/eagle.py:1497–1509,5325–5329` to compare legacy and reused draft IDs on the same root/loop inputs. Add same-input full-logit/candidate equality and duplicate-path handling to the declared eligibility contract if those invariants are asserted; the current selfcheck establishes draft-token equality, not automatically full-logit or leaf-list equality. Validate actual Cat10 B1/B4 and retain the current direct-API/commit/state qualification surface.
- `R/campaign_snapshot/e7a_capture_launch.v7.sh:324` forwards SINGLE_LOGITS, but this frozen launcher does **not** forward FIX1_SELFCHECK, its dump setting, or DEVICE_MULTIDRAFT. Explicit Docker pass-through is required in the new launcher; changing a parent-shell environment alone is insufficient. Selfcheck/capture instrumentation must be off in retained timing.
- **Abort timing if any required logit, candidate, token, published-path/state invariant fails, or if either branch does not demonstrably engage.** Preserve failures and characterize them; do not tune the correctness threshold from the result. CPU/wiring success alone is not qualification of FP8 GPU head recomputation.
- Only after that gate, run the smallest fresh paired ON/OFF comparison: **B1, three paired blocks, six timing cells**, using the established event-matched token/wall recorder, shape-matched warmup, exact B1 support and balanced blocks. B4 correctness is relevant only if the new mechanism claim is expanded to B4; B4 timing is not automatically required. This tests one component inside Cat10; it does not require rerunning both native baselines or a five-factor campaign. Prespecify the timing/repetition stopping rule and report an unresolved or negative effect honestly.

The existing E2 logits/draft/path captures can support a **CPU replay check of device/host committer legality** before considering that separate ablation. They do not contain a paired duplicate evaluation of the drafter head, and cannot supply missing single-logits GPU cost evidence. For a clean committer timing contrast, first establish the same five products on matched inputs, including duplicate-source choice, or explicitly scope it as different executed workloads; do not silently enforce FIRST/LAST. That extra complication makes it a lower priority than single-logits reuse. No port of the GDN fold, greedy TAW, or forked FA2 is necessary merely to describe those built designs with an explicit “not active/not freshly ablated” boundary.

Fresh E1's negative comparison against depth-matched native5 at both B1/B4, the longer-chain native11 comparator distinction, T1's tree/native engine-seed deviation, divergent continuations, and the fresh E7a numerical failures remain unchanged. The mechanism reframe supplies no causal optimization speedup or full-model equivalence evidence. This report imports **no historical quantitative performance results**.

## Executed bounded CPU checks

Ran local `python3 -B` with `ast.parse` of `L/eagle.py` and `D/scripts/fr13_device_multidraft_kernel.py`; compiled/evaluated only the actual `_fr13_single_logits` expression and TAW/depthsync `if` predicates. No serving module import or model execution:

```text
single_logits_env 0 local_argmax_false True
single_logits_env 1 local_argmax_false True
FR13_TAW greedy_true_dispatch False greedy_false_dispatch True
FR13_DM_DEPTHSYNC greedy_true_dispatch False greedy_false_dispatch True
```

Loaded the existing wiring test file through `importlib.util.spec_from_file_location`, invoked all five `test_*` functions directly under `python3 -B`: **5/5 PASS**. These inspect/compile source strings; no GPU inference or timing. No numerical equivalence was newly established here.

## Checked identities (SHA-256)

```text
98fbcdb5b367dbe8b171d8b9b98daab74bd94c55a2ef26852417d9f4898e842c  C1/docker_inspect.json
c902265545110b4cb2342563ef8be854e292ed646524ae65d7a2ab684608aab5  C4/docker_inspect.json
8b7678970e0af0bd51a83bce59c88be82532c8e7efe837e3bd3ab536f2ad49bf  C1/docker_logs.txt
1131efb004788720c7188884d9fe4d9b3f6488ec5e362f9183fb65467980dea1  C4/docker_logs.txt
a5c398ffff43d1d8ff4678e8df3750987cd520e7710d044cb4cbeb3c1bee7f72  R/campaign_snapshot/e7a_capture_launch.v7.sh
aa022e2a1fc7993d0f7d4223f8366471e02e3c0e2740934c7a9490f7b45b01d7  L/eagle.py
5be4ad43409da82b63bfd0177926467e9a2b59da553ee6dd04d3130a5d77b12f  L/rejection_sampler.py
723974c703d1ad1545399b78526e5d50ae7f7f362e434bad384a0a3327595b28  L/gdn_linear_attn.py
a6d4ae3000f248ec2b9fd37546a1f90b96145c3078e5a98e696b5bda49a1af97  L/tree_attn.py
b2975681fe888b72854a25ba377ceb5c1287aa53e16327feab80e3155f25ef40  L/gpu_model_runner.py
648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9  D/scripts/fr13_device_multidraft_kernel.py
564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e  D/src/lumo_flywheel_serving/fr13_tree_conv_fused.py
98f7c8511c1fa991fd36526e69a8b753a5205716bf5b0a53c058177c965cdb20  D/src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
a7f8f0943d1a13ac4742e824ea4e126c6ff341342aba677db5169e712c7e7281  D/scripts/fr10_phase4_patch_vllm_tree_gdn.py
c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e  current repository scripts/fr10_phase4_patch_vllm_tree_gdn.py
d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8  current repository src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
55c1b67fb4ae7e3c177638cbfcae2ec03ea17f6b49568e68a8676e700416d4b3  current repository scripts/fr13_launch_forked_fa2_tree_server.sh
eeb071a7d1733d39f7160c26f12086056d96b25e4629356eda0cd6fe6846d0a1  current repository tests/test_fr13_drafter_single_logits_wiring.py
dbd4542ee9899da3eb592fdf8a0b935010f5c8035de9d5cb75e581343c7deb1d  R/aggregate.json
264cbfc51181c6853dec9a90b0177aec57faef16bbdc35a397376dd870defab4  p0/monitor/e2-b4-duplicate-path-review.md
```
