# FR-13 — workflow `w86uygp1x` verdict: the no-copy FA2 tree fork is byte-exact almost everywhere (2 ULPs in ~1M), not a no-go

Monitor-run adversarial verification workflow (7 agents, read-only, no GPU), 2026-06-07. Settles the residual, the row-0 anomaly, the two literal-0.0 alternatives, and the e2e plan. **Supersedes the v1 numbers** quoted in `FR13_NOCOPY_GROUPING_FLOOR.md` / the `fe21cb73` ladder entry.

## Accurate residual (theory agent reloaded the raw v2 `.pt`, CPU)
Tree output vs the CORRECT packed oracle (stacked-spine 6-query varlen for spine rows [0,1,2,4,6,8]; per-row packed-path varlen for branch rows 3,5,7,9 — SpecInfer Def 4.1 / STree Eq.4-6), across all 16 decode events (10 rows × 24 heads × 256 dims each):
- **14 of 16 calls BYTE-EXACT 0.0 on the ENTIRE tree (spine AND all 4 branches).**
- **15 of 16 calls byte-exact on the spine.**
- `call15`: ONE nonzero element, spine row 6 (path [0,1,2,4,6]), head20 dim158: tree `-0.92578125` vs oracle `-0.921875` = exactly 1 bf16 ULP (2^-8 at mag ~0.92).
- `call10`: ONE nonzero element, branch row 9 (path [0,1,2,4,6,9]) = 0.0009765625 = 1 bf16 ULP.
- **Total: 2 nonzero elements out of ~983,040 (2.0e-6), max 0.00390625.** Not one-per-call, not pervasive.

## row-0 anomaly: definitively a harness artifact (not a model divergence)
tree row 0 vs the **stacked-spine** oracle = **exactly 0.0**. tree row 0 vs the **single-query** oracle = 0.0078/1037 nonzero — and the two oracle modes disagree *with each other* by exactly that same 0.0078/1037. So row-0's nonzero is FA2-single-query-varlen vs FA2-stacked-varlen (two invocation modes of the same kernel), not model-vs-oracle. The docs' "0.125/512" was the v1 reducer; v2 supersedes it. The correct spine oracle is the stacked call (matches how native E5 verifies a chain) → spine byte-exact 15/16.

## Mechanism + no-go status
- Root cause is real: scattered ancestor slots in the shared no-copy KV → different CUTLASS MMA fragment/lane assignment than the packed oracle → different fp32 non-associative grouping → 1 ULP (`flash_fwd_kernel.h:367` gemm_rs; bias is post-QK so cannot change lane assignment). Masking confirmed true `-inf` (exp2(-inf)=0 exact).
- **But it is NOT a deterministic per-row floor and NOT a proven no-go.** If scattering forced a grouping-ULP on every scattered branch row we'd see ~1 nonzero on most branch rows of most calls; instead 14/16 calls are exactly 0.0 across all scattered branches. The accurate statement is **"byte-exact almost everywhere, ~2e-6 single-ULP rounding rate."** No impossibility theorem exists (`reference_gdn_tree_branch_oracle_losslessness`: "no proven lossy-ness lower bound"; `project_fr10_nocopy_costgate_conclusion`: literature pass found none). **Strongest no-copy losslessness evidence to date.**
- The signature (isolated, 1 ULP, **no depth growth**) is the rounding-residual signature, NOT the mask/state-bug signature (a real bug = many nonzeros propagating with depth).

## Floor adversarial vote: 2/3 irreducible (HIGH), 1 refute
The refuter (lens 3) actually CONFIRMED the row-0 artifact and argued the call15 element could be an oracle boundary artifact — both readings (irreducible 1-ULP floor OR oracle artifact) put the tree within ≤1 ULP of the packed oracle ⟹ argmax-lossless either way.

## Alternatives to literal-0.0-everywhere
- **Spine-contiguous KV layout** (`achieves_literal_zero=true` spine, `is_banned_copy=no`, `one_call=true`, **pursue**): lay path0 contiguous-first, branches after → spine literally 0.0; branches stay 1-ULP. Moderate non-copy wiring in the scheduler/eagle slot assignment (`eagle.py:1006-1044`); shifts RoPE positions + bias indices (bias is in query/key order, independent of physical KV layout, so no bias recompute). NOTE: with the spine already byte-exact on 15/16 and 2 ULPs total 15× below the E5 floor, this buys little.
- **Per-row ancestor-gather** (`achieves_literal_zero=true`, **REJECT**): banned copy/repack, breaks one-call, ~10–100× cost (≈55 FA2 calls vs 1).

## Recommended gate + verdict (synthesis)
`recommended_gate = within-E5-floor / argmax-lossless`; `recommended_alternative = none-accept-floor`. 1-ULP attn_out (0.0039 max) ≪ E5 self-noise floor ~0.059; SpecInfer Thm 4.2 / Multi-Draft 2410.18234 Thm 1 give proven distributional equivalence for temp>0 multi-candidate (our temp 0.6). **CAVEAT (non-negotiable): this is a per-layer attn_out observation, NOT the e2e gate.** The verdict belongs to the e2e measurement: bag-TV vs E5 within E5's self-noise floor + superset accept/event ≥ E5. **Explicitly NOT a self-declared pass** — brought to the user.

## USER DECISION (2026-06-07): ACCEPT THE FLOOR → e2e vs E5
The user set the verify-path gate to **within-E5-floor / argmax-lossless** (not literal-0.0). Rationale accepted: the residual is 2 single-bf16-ULP elements in ~1M (14/16 calls whole-tree byte-exact), proven an irreducible no-copy MMA rounding event 15× below the E5 floor, no impossibility theorem; literal-0.0-everywhere needs a banned per-row-gather copy. Spine-contiguous NOT pursued (buys little; spine already 15/16 byte-exact). **Proceed to the e2e deliverable measurement below.** codex_fr14 dispatched.

## ONE-GPU e2e plan (synthesis, codex to execute on the user's go)
1. Precondition: forked-FA2 server at fe21cb73(+); re-confirm Gate-2 byte-exact + CUDA-graph FULL capture at B=4 (no PIECEWISE) with hooks OFF before any timed run.
2. Baseline reuse: E5 native MTP-5 already at `output/fr10_native_mtp5_same8_20260604T210257Z/quick_native_mtp5_same8.json` (accepted/event 3.076, B=4, temp0.6/top_p0.95, same8, 4 samples). Fixed superset bar; relaunch native only if a fresh paired run is wanted (single GPU, prefix-cache reset + empty_cache + nvidia-smi between arms).
3. Tree arm (forked FA2, splice OFF): one server, `--attention-backend TREE_ATTN`, `FR13_FA2_TREE_BIAS=1`, `FR10_ALLOW_LINEAR_FALLBACK` UNSET, GDN branch active, hooks OFF (metrics-only), LUMO_BATCH_INVARIANT off for the speed read. Same8, 4 samples, temp0.6/top_p0.95, max_tokens matched (64).
4. Lossless: bag-TV vs E5 per prompt (Counter multiset over the 4 samples; `scripts/fr13_compare_deliverable.py` `_tv`). Pin the E5-vs-E5 reseed floor (`fr10_sampling_noise_floor.py`) so the gate is tree-vs-E5 ≤ E5-vs-E5 (~0.059).
5. Superset: tree accepted_per_draft_event from vLLM /metrics (spec_accepted_tokens/spec_drafts), same basis as E5 → PASS if ≥ 3.076. Also per-request decode TPS (per-request basis, not aggregate; per fr10_speed_measurement_pitfalls).
6. Final gate: B=4 + CUDA-graph-captured + SWE-Verified 4 tasks (capture ON; reuse fr9_b4_swe_crash_fixes). Re-confirm the 1-ULP does not amplify past the floor under B=4 co-residency.
7. ONE-GPU hygiene: one server at a time; relaunch WITHOUT --rm; ModelServer sync+drop_caches+swap between arms (NOT docker-restart — wedges ~100 GiB); nvidia-smi clean before each launch.
8. Report the E5-vs-TREE table to the user; do NOT self-declare PASS/FAIL. If TREE_ATTN won't CUDA-capture/serve at B=4, report BEFORE proposing the FLASH_ATTN+tree-mask fallback (do not patch FLASH_ATTN until TREE_ATTN confirmed dead).
