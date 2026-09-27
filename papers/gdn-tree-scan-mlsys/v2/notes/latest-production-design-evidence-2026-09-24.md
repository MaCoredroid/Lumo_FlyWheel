# Latest production design and evidence audit — 2026-09-24

**Disposition:** the latest *completed served* lineage is fixed-32 Hydra27, full-vocabulary Qwen3.8-27B NVFP4, with the patched FA2 GQA-pair split-K arm. The September 22 Cat10/Triton research runs revisit a different route and cannot qualify this production design. Current source also contains later candidates; the presence of a candidate in HEAD does not establish a completed serving qualification. No inference, GPU work, or source changes were performed for this audit.

Paths below are relative to repository root unless prefixed `v2/`, which means `papers/gdn-tree-scan-mlsys/v2/`. The inspection covered the June–August commit sequence, its later corrections, current implementation, and the original safe Cqc10 receipts. Head at review: `ae3dfbe99` (the September paper commit); production files remain from the August lineage. Dates alone are not used as evidence of design currency.

## 1. Supersession and promotion lineage

| Commits / period | What changed and how to interpret it |
|---|---|
| `e5f7b4f94`, then `219d41de`, `a09ef5b5`, `45dc05a2` | Patched FA2 tree integration and replay/eager buffers/fused convolution/single-logits entered the FP8 serving lineage. FA2 is not determined by weight precision. The current NVFP4 route descends from this route. |
| `316c6390b`, `362133b64`, `a35389eef` | Spine-first KV mapping, retirement of leaf-map machinery, and missing-node guards. Read later source rather than treating an early “stock”/default-off comment as the current launcher setting. |
| `e21d626f1`, `8bca48c2b` (July) | Bake the integrated TAW, parent-gather, committer-graph, conv-pregather, flags and subtree-parallel route. Retire the old locked launcher and explicit node-bank burn; running-row initialization is mandatory. |
| `9d8095ea0`, `6ca558f59`, `766d278ea` | Logical speculative state columns alias a scratch page; reservation is right-sized. Device commit filling joins the captured continuation path. |
| `99a511319`, `32e240e15`, `1d4258b36` (August) | GQA-pair FA2 promotion and FP8 fixed32 closeout. These do not themselves prove later NVFP4 quality or every new numerical arm. |
| `4bdf4dd14`, `fa2705196`, `e5e2a6137`, `8f0b6ab85` | Move to Qwen3.8 NVFP4, then the RadixArk aggressive NVFP4 checkpoint, including the NVFP4 vocabulary head; full-vocabulary profiles replace the FP8 restricted-head assumption. |
| `8fe896720` | Fused full-vocabulary draft selection is promoted. Its pinned binary remains the one recorded in Cqc10. |
| `fd728e2b3` → `e562d15be` → `8dd868805`, `fe1294207` | Earlier alleged split-K round 6 is withdrawn: it served incumbent versus incumbent. Round 12 establishes actual engagement. Split-K becomes the Hydra27 default through the explicitly Tier-B route, after the user waived the exact16-before-promotion ordering. This is not a byte-exact Tier-A promotion. |
| `6301e7efe`, `d390ed5e7`, `8dfeaaacf`, `e7af6b595` | Preserve the NVFP4 degeneration and interrupted quality lineage; put the response ceiling on the actual client path; reseal workload identity; complete Cqc10. Cqc10's source and binary receipts, not the misleading composite score, are the relevant completed deployment record. |
| `39b7afcb3`, `d75601a81`, `80bb1b45a`, `90ecf6ced` | Later source derives profile geometry and repairs the prospective Hydra31 launch path. `REDTEAM_20260816.md:7283–7313` explicitly ends with boot twelve still required after pause. Hydra31 therefore does not replace Hydra27 as a completed served result. |

`FR13_PIPELINE_LOCK.md`, `FR13_KERNEL_STATUS.md`, the initial `FR13_STATELESS_TREE_DESIGN.md`, and the beginning of the FR14 README/split-K note are earlier checkpoints. Read their superseding commits and appended decisions before citing them as current contracts. In particular, `splitk_fa2.md:967–1027` supersedes its initial “not promoted” status while retaining the Tier-B distinction.

## 2. Current implementation, bound to actual Cqc10 settings

The safe actual environment is `v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/container_env.SANITIZED.txt`; the FA2 engagement receipt is its `logs/fr13_fa2_qrow32_b1_production_engagement.json`.

- **Topology:** `scripts/fr13_fixed32_topology.py:62–129` defines Hydra27's 27 active drafts in 31 physical draft slots plus root, with four masked slots. Five head depths supply 15 MTP candidates; suffix paths supply the principal tail and rescue nodes. The maximum draft depth is eleven. Later Hydra31 geometry is a separate profile, not a measured replacement.
- **Actual switches:** TAW, parent gather, subtree parallel, fused convolution, single-logits, drafter graph, native committer graph, fixed32 device fill/publication, KV remap, slot reorder and syncfree remap are enabled. Conversely `FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=0`, `GDN_GQA_GROUP3_PRODUCTION=0`, `COMMITTER_LAYER_BATCH=0`, both TAW native-precompute selectors are zero, and `STEP_GRAPH=0`. Do not turn these optional candidates into deployed optimizations.
- **GDN scan and cached state:** `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:10409–10491` carries recurrence state in fp32, evaluates raw gates and query/key normalization, and applies rank-one updates in ancestry order. Actual `SCAN_ALIGN=0`; the optional native-alignment round trips are not active. `:16800–16817` requires the operand rings to be byte copies in the input dtypes. The active two-level path schedule (`:17270–17335`) passes `st["export"]`: one root path exports cut states, then terminal paths read them. This is temporary inter-launch state traffic, not durable branch state. It must not be called “zero exported states” or “a single GDN launch.”
- **Native replay and precision:** fixed32 replay resolves a preseeded fixed16 graph (`:13966–14008`, `:15480–15510`, `:16127–16230`). Its body pads/masks unused accepted positions and calls the native fused GDN update once per layer (`:14551–14693`). All replay positions point to the request's column-zero running row (`:14653–14657`). Layer batching is off in the actual environment. One graph replay therefore contains the sequence of native layer calls, not one all-layer kernel. Shared rounded operands do not imply bit identity between scan and native replay: normalization, gate/beta evaluation, casting and update schedules remain numerically relevant.
- **Running-row lifecycle, no burn:** `_fr13_fixed32_committer_fast_state` requires `burn_node_bank=False`. The patcher `scripts/fr10_phase4_patch_vllm_tree_gdn.py:25231–25501` retains a durable column zero plus one scratch page aliased across logical speculative columns. The fixed32 path disables legacy full-node convolution writeback and per-node convolution remap; it publishes the chosen convolution window into the native row. Request allocation/retirement and prefix-cache state remain real persistent state. “Stateless” can describe the absence of cross-step branch recurrence state, not absence of request state, draft suffix caches, RNG or reusable graph buffers.
- **KV mapping:** patcher `:38709–38742` writes tree KV into spine-first physical suffix slots while query rows/positions remain logically ordered. `scripts/fr13_patch_fa2_tree_bias.py:9430` permutes the ancestry-mask **key columns** consistently. The generic `FR13_FA2_SPINE_REORDER` hybrid query-reorder branch is a separate mechanism. Accepted-path remapping reads from the permuted source slots but writes flat native destination slots (`fr10_phase4_patch_vllm_tree_gdn.py:41768–41800`); mapping is restored before the drafter copies it. The fresh published path is request-keyed and guarded against a stale commit. This supports the concrete addressing design, not a universal cache-correctness theorem.
- **Acceptance:** the active fixed32 TAW path is distinct from the old generic per-node Python walk and from the optional all-parent precompute candidate. `scripts/fr13_device_multidraft_kernel.py:4483–4531` explicitly refuses per-request generator maps and fills uniforms from one bulk device generator. Do not claim request-seed or cross-boot stream invariance. The earlier source-v7 all-parent B1/B4 credential was explicitly invalidated by the August 15 widened gate/source identity; it cannot qualify an unarmed candidate in Cqc10.

**Important contradiction in an old audit:** `REDTEAM_20260816.md:6594–6624` attributes zero state export to single-launch kernels and then calls Cqc10 a live attestation. Actual Cqc10 disables that candidate; the active two-level source exports cut states. Retain the audit as history, but do not repeat this inference.

## 3. Strongest reusable current-route evidence and its limits

1. **Positive loaded FA2 engagement.** The Cqc10 receipt identifies `gqa_pair_splitk`, `candidate_served=true`, FULL graph, all sixteen full-attention layers, four splits, no fallback, source `e7af6b595...`, FA2 head `29210221863736a08f71a866459e368ad1ac4a95`, binary `28570f...`, patcher `8f61b9...`, and Tier-B credential `37ed4ff...`. This is stronger than an enabled environment flag. Final flush has matching complete census and no pending forward/commit intervals; it is lifecycle evidence, not a full state oracle.
2. **Fused candidate output parity.** `results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe_result.json` records 1,368 input cases × five block settings, zero byte mismatches, powered negatives and 24 captured four-level replays with no mismatch. Reference operations are separate argmax and `torch.topk(...,3)`. `fused_draft_topk.md:116–188` explains the pinned tie behavior: selected set ties use ascending index, emitted top-k ties use descending index, while argmax picks the lowest maximal index. Do not substitute an invented universal “stable top-k” policy. The .so hash equals Cqc10's actual `FR14_FUSED_DRAFT_TOPK_SHA256`. These are component checks, not proof that every model run or PyTorch version is equivalent.
3. **Numerical split-K credential.** `fr14_splitk_tierb_credential.json` passes nine checks and binds the exact binary, source closure, FA2 head, patcher and bounds used by Cqc10. `measurements`, `bounds_evaluation`, `determinism`, `probe` are the primary keys. Repeated results agree within and across two processes. Characterization is **synthetic Q/K/V at measured operand scales**, not original task tensors: `fr14_splitk_fa2_probe.py:128–149` uses `torch.randn`. The auxiliary argmax experiment also uses random projection/head weights (`:389–407`); it is not a next-token oracle. Current paper's bounded attention error statements are supportable; byte equality against incumbent and downstream token/quality preservation are not.
4. **Application behavior.** Original Cqc10 task metadata/evaluations plus the saved trace-check projection support the already scoped ten-task completion/nonempty-patch/zero-recorded-flags statement. The earlier campaign includes genuine thinking-only/capped outcomes. `REDTEAM_20260816.md:6556–6592` explicitly parks causal degeneration attribution; native probes do not establish equality of rates. A response ceiling and detector account for failures; they do not prove their absence. Keep these limits and the original adverse task outcomes.

No new experiment is indispensable to describe this source-selected design and these bounded component/workload observations. A new route-specific numerical/state or repeated quality experiment becomes necessary only for stronger claims such as native full-model equivalence, distribution preservation, no degeneration generally, or a newly promoted candidate. Removing superseded-route measurements must not be replaced by transplanting their proof claims.

## 4. Required manuscript corrections and review boundary

Remove September Cat10/E2/E7/E8 as current methods/results; retain their immutable archive. Remove old burn/leaf-state lifecycle, generic Python-walk characterization of the active fixed32 path, stock-Triton/precision-to-backend implication, K64 assumptions for this full-vocabulary deployment, and any statement that all implemented optimization candidates are enabled.

During the current rewrite I identified two remaining source-specific errors and sent them to the parent: (a) query/position permutation was conflated with default KV slot reorder in two passages; (b) “captured-scale” needed to say synthetic operands at measured scales. Parent reports both corrected. Final source/hash closure is recorded separately in `notes/latest-production-paper-redteam-2026-09-24.md`.

## Reviewed file hashes

```text
c02703b4bc777e5edfa01623b446a289cba8f1023e31d8efd7fc8126666b38bc  scripts/fr13_fixed32_topology.py
55c1b67fb4ae7e3c177638cbfcae2ec03ea17f6b49568e68a8676e700416d4b3  scripts/fr13_launch_forked_fa2_tree_server.sh
8f61b9cd6a079d0605daedb545b8a2e95225714693c775d396059bab2ce626b2  scripts/fr13_patch_fa2_tree_bias.py
c382ef22a5a8d4e84209e6723f8eea21c3819543c6e942c62951385b5c1c003e  scripts/fr10_phase4_patch_vllm_tree_gdn.py
648ff0e33e8a7cdd0fbe8ebb3d2befaa27002a3347f94f9bdc0668365ed2e8e9  scripts/fr13_device_multidraft_kernel.py
d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8  src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py
564c7dc7ba61ccd9579f2fb7a4af9a2cdaaa1753dfb98434f2011457d90d9a6e  src/lumo_flywheel_serving/fr13_tree_conv_fused.py
902814907c5642ce917fbda8f7ca955b3d859d4caae2d22570b448c4121d2736  results/fr14_nvfp4_port_20260816/REDTEAM_20260816.md
86d72bc2a50295ec161f2ccd1cbcb84bf6fa13d104ed3527914ca7db3abeab2b  results/fr14_nvfp4_port_20260816/splitk_fa2.md
da61cd05219fb20c3e98cc976caf9ba0db24613a43b67d34c11315e45e31503e  results/fr14_nvfp4_port_20260816/fused_draft_topk.md
b9ea3059f6e7400e79b4ee2a2842c17a35f2b2b1424fbfb9db8c2964be5fd6fb  results/fr14_nvfp4_port_20260816/fr14_fused_draft_topk_probe_result.json
37ed4ff59f6c0d6f9ae482916e0c5a5939f38f70d80756fdb43a44101d6dde19  results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_credential.json
ee49c3a712971f81509617bbd3f7cabffa5f43c74cd9db852459a9a758c1958e  results/fr14_nvfp4_port_20260816/fr14_splitk_tierb_bounds.json
44101abdb31292bfd702a2729bee2d5e115b79405eebe981ba246f9696bb7fcb  results/fr14_nvfp4_port_20260816/fr14_splitk_fa2_probe.py
a5042a1a03e08cc2e7f54ff18b64a6beeb8a7f9fc3a456fd492eac9889d4e44d  results/fr13_taw_widegate_shape_pin_20260815/README.md
15f1e8ea4a9bbc33d8837dfbeddce3e2b9bab62f63411c090b5b8a1da48550d4  papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/container_env.SANITIZED.txt
c8f3c761e9cd3ee97a1c3c421f28488154ae13df2504bb42aa026966dad1da71  papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/logs/fr13_fa2_qrow32_b1_production_engagement.json
4492d14ec659cb91d0476293cf4027715928975e91d45eba96d735af9f7cd001  papers/gdn-tree-scan-mlsys/v2/results/agent-workload/raw/cqc10/hydra27_fixed32_promoab_Cqc10/fixed32_final_flush.json
```

### Additional positive boot binding

The locally preserved Cqc10 `docker_full.log:30` explicitly sets `mamba_ssm_cache_dtype=float32` and prefix caching. Lines 127–128 announce fused convolution and the fixed32 subtree schedule; lines 186–187 announce the fixed16 graph with 48 native fused calls and `layer_batch=0`; line 292 announces actual fixed16 one-replay engagement. These records agree with the source/environment distinctions above. Hash: `97a5431d2890d71d8335979a75b4d9470b4fc22c25c2a511d728459956c31a9d`.
