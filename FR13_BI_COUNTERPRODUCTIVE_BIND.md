# FR13 — BI batch-verify: BI is COUNTERPRODUCTIVE (cat9+BI=34 > 22). BI dead, C3 moot, op still unlocalized.

Date 2026-06-14. GPU batch-verify `wtgz14vv4` (killed after the decisive cat9_bi boot; chain5_bi control
skipped as low-value once BI was disproven). Result: `research/fr13_workflows/cat9_bi_flips_34_decisive.json`.

## Decisive result (TRUSTWORTHY)
cat9 + Method-A BI on (BATCH_INVARIANT=1 + FR13_BI_TREE_ATTN=1, both wired + the boot needle self-asserted):
- **total_clear_margin_flips = 34, per-prompt [10,11,3,10], rate 34/510 = 6.67%**
- vs cat9 noBI = 22/462 = **4.76%**; chain5 = 5/512 = 0.98%; native = 3/512 = 0.59%.
- engaged tok/draft=9.0, within_boot_det [T,T,T,T] (stream AND oracle), spec_delta=0, accept/event 3.109.

=> **BI does NOT fix the spine perturbation — it makes lossless WORSE (6.67% vs 4.76%).** Not just
ineffective: counterproductive. BI is DEAD as the 22-flip fix.

## Why (hypothesis, unverified): partial BI coverage on GB10
The branch-fix workflow (w0tokpq9g) found: GB10 (sm_121/family-120) is NOT is_device_capability_family(100),
so enable_batch_invariant_mode takes the REDUCED override branch; AND the fp8 block GEMM is a CUSTOM op
(w8a8_triton_block_scaled_mm_func) NOT covered by the aten-only BI overrides. So BI pins SOME ops (aten::mm,
GDN bf16 in_proj/conv) to a fixed order while leaving the fp8 GEMMs (the dominant compute) at their default
order — creating a NEW inconsistency between the tree-verify path (M=9) and the decode oracle path (M=1) that
INCREASES divergence rather than removing it. (To verify if pursued.)

## C3 (targeted fp8-GEMM M-invariant override) is ALSO MOOT
The fp8 config lookup returns None on GB10 (no Spark JSON) -> the DEFAULT config is already M-INVARIANT
(branch-fix Verify). Single program owns each output row's full K loop, BLOCK_SIZE_K fixed -> the fp8 GEMM
row value is already M-independent. So forcing it M-invariant is a no-op; the fp8 GEMM is NOT the carrier.

## So the actual batch-variant op is UNLOCALIZED
SPINE_PERTURBATION is real (chain5=5 vs cat9=22, empirical), but every obvious op is refuted/M-invariant:
fp8 GEMM (M-invariant default), TREE_ATTN (strict -inf ancestry mask isolates spine), GDN scan (select-by-mask,
no branch folds in), norms (per-row). The op that the 4 co-resident branch rows perturb the spine through is
NOT yet pinned. BI worsening it deepens the puzzle.

## Disposition (lossless 22->3, keeping the accept edge):
1. **Deeper localization (next):** a CONTROLLED cat9-spine-row vs chain5-spine-row per-layer comparison on the
   SAME teacher-forced prefix (identical spine input, only the 4 co-resident branch rows differ) -> find the
   FIRST op/layer where cat9's spine row diverges from chain5's. That pins the actual batch-variant carrier
   (or proves it diffuse = no single op). Plus: why did BI worsen (the partial-coverage mechanism)?
2. **Reshape/lean fallback (proven):** chain5=5 shows fewer branches = fewer flips; the deployable question is
   which lean subset cuts flips most while keeping accept > native (the cat7/cat8 frontier, NOT global lean).
CPU-design first (per user), then GPU-verify. Pairs with [[feedback_check_artifact_before_concluding]],
[[reference_diffuse_gdn_accumulation_explained]], [[project_l0c_triton_autotune_drift]],
[[feedback_kill_wrong_gpu_task_immediately]].
