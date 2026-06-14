# FR13 bf16-vs-fp32 seam scan — conv tap is NOT the seam; the GDN-scan LAUNCH GEOMETRY is

Workflow `wf_55f0d466-176` (5 agents, CPU static scan vs native vLLM 0.19 source read from
`/tmp/vllm_live_019`). Raw: `research/fr13_workflows/bf16_fp32_seam_scan_wf_55f0d466.raw.json`.
Red-team **holds=True**. 2026-06-14. **This CORRECTS the earlier "conv bf16-tap is the fixable
carrier" read** (which came from a cross-boot/contaminated localize-v2 conv diff).

## THE BIG CORRECTION: the conv tap 0.0146 is MATCHED to native — do NOT "fix" it
The loud conv tap divergence (fp32 vs bf16 tap products = 0.0146 ~1 bf16 ULP) is a **SYMPTOM,
not a fixable seam**. **Native ALSO rounds the tap product to bf16** before the fp32 add:
`causal_conv1d.py:1044` `acc += matrix_x*matrix_w` with both operands bf16-loaded (no fp32 cast)
⇒ bf16*bf16 then fp32 add. Our two bf16-tap arms (patcher `:2114-2121`
`(x.to(bf16)*w.to(bf16)).to(bf16).to(fp32)`; fused `fr13_tree_conv_fused.py:234-235`) **MATCH
native exactly.** FR12 settled this empirically (`FR12_PARITY_RESULTS.md`: fp32 products → 0.125
max-abs MISMATCH vs native; bf16-rounded → 0.0 BYTE-EXACT; live ttgir shows native emits
`mulf:bf16 → extf→f32 → addf`). **So converting taps to fp32 (the naive "align to fp32" hint)
is WRONG — it would re-introduce the 0.125 mismatch. DO NOT change the conv taps.** The scan
saved us from shipping a wrong fix.

## THE ACTUAL OPEN SEAM (S1, top carrier): GDN-scan LAUNCH GEOMETRY codegen
Our GDN rank-1 scan **op body** `_gdn_node_step` (`fr10_gdn_tree_kernel.py:364-383`) is
**byte-identical to native** `fused_sigmoid_gating_delta_rule_update_kernel` op-for-op (fp32,
softplus/b_g/sigmoid/l2norm/*scale/*=exp(g)/-=sum/*=beta/+=outer/sum). The SEAM is the **launch
geometry**: ours **BV=16 / num_warps=8** (`fr10_gdn_tree_kernel.py:18`, num_warps=8 at
`:843,:1284,:1536`) vs native **BV=32 / num_warps=4** (`fused_sigmoid_gating.py:223,:227`).
Different warp/lane map → different ptxas FMA scheduling + `tl.sum(axis=1)` reduction tree →
**~1 bf16-ULP per node, on EVERY GDN node on EVERY ~48 GDN layers → uniform diffuse compounding**
(exactly the "diffuse L0-L58 GDN accumulation" signature), amplified ~32× by the gate 1/rms.
Never gated below atol=1e-3 vs native. S2 (replay bank-pointer layout, `:931`) is the same
codegen class; gate/o_proj are unchanged native module calls (no seam).

**Tension with the standing "NO BV change / BV=16 is bit-exact" note:** `FR13_BV_SPILL_VERDICT`
compared **BV=16 vs BV=8 (both OURS)** and found them equivalent — it did NOT compare ours vs
**native's BV=32/warps=4**. So "BV=16 bit-exact" was a different question; vs native, our
BV=16/warps=8 geometry IS the open seam.

## The fix + its REAL blocker (possibly-irreducible)
FIX = set our tree/replay scan launches to **BV=32, num_warps=4** to match native (changes only
launch constexprs/codegen of OUR fp32 kernel — NOT a reroute, NOT a reward-hack). **BLOCKER
(genuine):** `h_cache = tl.zeros((N_PAD, BLOCK_V, DIM_K))` (`:458`) grows with BLOCK_V; at
deployed **N_PAD=16 with BV=32 it may SPILL** (compile/launch fail). If it spills, the fallback
(reproduce native's reduction tree at BV=16) may be impossible → **S1 could be the irreducible
floor**. Also: matching geometry only makes the reduction tree identical IF ptxas tiles the same
— codegen-dependent, must gate on **RAW max_abs == 0.0 (NOT atol=1e-3)**.

## Cheapest verify (the decisive next step, ONE boot)
Single-boot, single-layer-0, **in-process A/B** of the GDN scan ROOT output at deployed config
(N_PAD=16, replay ON): our `_tree_gdn_kernel` at shipped **BV=16/warps=8** vs a second launch at
**BV=32/warps=4** on bit-identical captured inputs, compare **RAW max_abs of out** (must be 0.0)
against the native `fused_sigmoid_gating` reference captured in the SAME boot. If BV=32 → native
RAW 0.0 while BV=16 ≠ → S1 confirmed as a carrier. If BV=32 spills → irreducible-floor branch.
Reuses the FR12 paired-capture harness. NOTE: CPU cannot prove S1 is THE dominant carrier (only
A carrier) — this GPU A/B binds it.

## Status of the scan: EXHAUSTIVE — no new seams
conv tap (matched), scan reduction (op-body bit-identical; geometry = S1 open), l2norm, g/decay,
silu, gate (native), o_proj (native) all enumerated. **No new unlisted bf16-vs-fp32 op seams
beyond S1/S2.** So if S1 (scan geometry) is the carrier and is fixable, that is the whole fix.
Pairs with [[reference_diffuse_gdn_accumulation_explained]], [[feedback_math_correct_vs_bitexact]],
[[feedback_fr12_subkernel_zero_gate]], [[project_fr13_cache_scaling_future]] (h_cache spill),
[[feedback_no_reroute_reward_hacking]].
