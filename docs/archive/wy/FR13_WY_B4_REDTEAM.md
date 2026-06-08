# FR13 WY B=4 red-team — the 1.199 is most likely COMMITTER/TOPOLOGY wiring, NOT the verify (batch-invariance agent a1ca90, 2026-06-08)

## PREDICTION (confidence ~0.9, READ-ONLY static — B=4 Gate-A ladder is the decisive GPU test)
The WY scan is **batch-invariant**: B=4 drift == B=1 drift (1 bf16 ULP, argmax-lossless). So the 1.199 e2e accept/event is **NOT** GDN-scan batch-drift / a lossy verify at B=4. It localizes to the **committer/topology wiring**.

## Why the WY scan cannot be batch-dependent (file:line)
- Grid `(num_vh, cdiv(dim_v,BV))`, no `pid_batch` (fr10_gdn_tree_kernel.py:795,437-438) — each program self-contained.
- Launched per-element: `for fr10_b in range(num_spec_decodes)` + per-element slice `start=fr10_b*tree_n` (fr10_phase4_patch:1956,1960-61), `launch_tree_gdn_prepared(...)` (2389-2405). B=4 element-0 = byte-identical inputs + identical-shape launch to B=1 (n_actual=tree_n≤16). Cross-element atomics are diagnostic-only, gated `(pid_vh==0)&(pid_v==0)` (252,447).
- NO autotune (bare `@triton.jit`:400, `BV=16` const:17) → #42960/autotune-drift class EXCLUDED.
- Reductions `tl.static_range(0,N_PAD≤16)` (535,547,567,572), extents only constexpr N_PAD/DIM_K/DIM_V → N-independent.
- Native FLA autotune keys exclude N/B/T (chunk_delta_h.py:39, chunk_o.py:39, solve_tril.py:35,110,235) → also batch-invariant.

## Therefore the 1.199 lead (after the B=4 ladder exonerates the verify) = COMMITTER/TOPOLOGY
- **Drafter topology mismatch** (`project_fr10_drafter_topology_mismatch`): stock `propose_tree` gives `child_drafts=[1,2,1,1,1]` = two parallel chains NOT a caterpillar → leaf slots get chain-2.
- **Committer root/parent wiring**: a 56% step-0 reject = signature of WRONG root/parent wiring, not a lossy linear-recurrent verify.
- NO in-kernel batch-invariant fix to make (the scan already is). Do not grind the verify further.

## Decisive test (codex, in progress): the per-element B=4 Gate-A ladder
- If WY scan stays ~1 ULP / argmax-lossless at B=4 (PREDICTED) → the 1.199 is committer/topology → fix the drafter topology + committer wiring → re-e2e.
- If WY scan shows >1 ULP at B=4 → the only remaining batch-coupling is an UPSTREAM producer of g/beta/a/b/h0 (in_proj/conv/gating, vLLM-side), NOT the WY reduction. (Memory: in_proj bit-exact, conv fixed, fp8 GEMMs batch-invariant.)
Empirical confirmation tool: scripts/fr12_scan_batch_invariance_probe.py.
