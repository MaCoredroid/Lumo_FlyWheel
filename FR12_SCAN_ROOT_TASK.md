# FR-12 (codex_fr12) — drive GDN scan → gate → o_proj to BIT-EXACT 0.0, then lossless

**Branch:** `fr12-wy-tree-kernel` · **Driver:** codex gpt-5.5-high (codex_fr12) · **Red-team + loop:** Claude (Opus), 10-min. **ONE GPU — strictly serial.** Read `FR12_REDTEAM_ARGMAX_LAG.md` + `FR12_PARITY_RESULTS.md` (full state) and the committed probe scripts before starting.

## POLICY: proceed continuously, fix until zero-drift, THEN measure e2e (user, 2026-06-06)
Do NOT stop to ask between fixes. Keep fixing each divergence — **UPSTREAM-first, in OUR kernel, verifying SPINE *and* BRANCHES** — until there is **ZERO drift everywhere** (spine+branches bit-exact 0.0 across ALL 64 layers AND the final logits). ONLY THEN measure e2e: the **B=4 + CUDA-graph-captured + SWE-Verified-4** gate = lossless (drift within native self-noise floor) + SUPERSET (accept/event ≥ native). Bring the user the e2e numbers. Still ASK before any **close/pass-fail verdict** and before any **copy/dense/re-stream/reward-hack** shortcut (those remain BANNED). Each fix: find the FIRST diverging stage → fix its root in our kernel → re-verify that stage=0 + that layer 0.0 (spine+branches, eager, splice-OFF, ALIGNED so input_hidden~0) → continue propagation to the next divergence and fix it too.

**LOCATE PRECISELY (wiring vs kernel) before fixing (user, 2026-06-06):** for each divergence, determine whether it is **WIRING** (mask / position / layout / backend-config — fix the wiring in our code) or a **KERNEL** issue (a vLLM native kernel computes the WRONG thing for the tree case). If kernel-wrong-for-tree → **BUILD OUR OWN correct version and plug it** (like the GDN scan `tl.range`) — no decision to bring the user. Do NOT build a kernel to paper over a wiring bug, and do NOT reward-hack (splice native-on-spine to pass). When a test changes BOTH mask and kernel at once (e.g. switching attention backend), attribute the result carefully: backend→0.0 = wiring/mask (confirm the prior mask was actually wrong); backend still nonzero with correct mask = kernel → build ours.

## THE HARD GATE (user, non-negotiable)
Drive the per-layer GDN sub-kernels **(3) gdn_scan, (4) RMSNormGated gate, (5) o_proj** to **bit-exact 0.0** (below fp8 bucket resolution so no byte flips) **BEFORE any other kernel work** (speed, other model parts). Do NOT "let small residuals go" — they compound over 64 layers and ARE the whole drift.

**"0 vs native" must verify the SPINE *and* the BRANCHES — not just path0 (user, 2026-06-06).** All the scan probes so far compare only the spine rows `[0,1,2,4,6]` vs native's linear chain. That is NECESSARY but NOT sufficient: the off-spine branch nodes (rows 3,5,7,…) are what give the superset, and accepting them losslessly requires their verify output to be correct too. Native MTP-5 has NO branch counterpart, so the branch oracle is **native run on each branch's path-to-root** (the linear ancestor-path root→…→parent→branch), no-MTP, **depth-based positions** — this is theorem-backed (SpecInfer Def 4.1 / STree §3 Eq.4-6; see `reference_gdn_tree_branch_oracle_losslessness`). For each sub-kernel (scan/gate/o_proj), verify **both**: (a) spine row == native linear chain (0.0), AND (b) each branch node == native-on-its-branch-path oracle (0.0). The branch check also validates the **ancestry mask** (a branch must fold in EXACTLY its ancestors, no sibling/non-ancestor bleed). Do NOT declare "0 vs native" until spine AND branches both pass.

## The drift is LOCALIZED (proven, codex boot-free probes 2026-06-06 — build on these, don't re-litigate)
- **conv: FIXED** (0.0) via our-kernel bf16 tap rounding (`FR12_TREE_CONV_NATIVE_BF16_TAPS=1`).
- **in_proj: BIT-EXACT** (tree_vs_native in+out = 0.0) and **batch-invariant**.
- **o_proj fp8 GEMM: BATCH-INVARIANT** (full-batch vs spine-row-only vs reversed-context = 0.0, both tree+native; `fp8_full_gemm_batch_invariance_l0.json`). Its activation quant is row-independent too.
- ⟹ the ONLY L0 source is the **GDN SCAN: 5.96e-8 fp32 reduction-order** → amplified ~32× by the gate (1.9e-6) → quantized to ~2-4 fp8 bytes → **o_proj output 1.2e-4**. **o_proj (5) is a SYMPTOM**: fix the scan+gate input to 0.0 and o_proj → 0.0 automatically (identical fp8 input → identical bytes → identical GEMM out).

## The scan root = its BATCH-DEPENDENCE (#42960's uncovered GDN-scan case = authorized lever #2)
vLLM batch-invariance covers attn+GEMM but NOT the GDN scan. Co-resident branch rows shift the spine's fp32 reduction order → 5.96e-8. **Fix = make the scan reduction N-INDEPENDENT** (fixed accumulation order regardless of co-resident rows) so the spine row computes identically to native's linear chain → 0.0. Match native FLA exact rounding: chunk 64, bf16 cast boundaries, accumulation/op order, raw-g vs exp, l2norm-in-kernel (read live `/tmp/vllm-0.22-src/.../fla/ops/` chunk.py / chunk_delta_h.py / fused_recurrent.py / solve_tril.py / wy_fast.py / fused_sigmoid_gating.py).

## Plan (boot-free first, ONE GPU)
1. **Boot-free probe the scan batch-invariance** (mirror the GEMM probe): full-tree-batch vs spine-row-only vs reversed-context — does the SPINE output change with co-resident rows? (expect ~5e-8). This pins the mechanism.
2. **Fix the scan reduction to be N-independent / native-op-order** in OUR kernel → spine == native → 0.0.
3. **Drive the gate → 0.0** (op-order; the gate's own RMS reduction + the now-fixed scan input).
4. **Verify the cascade**: scan 0.0 → gate 0.0 → o_proj 0.0 → L0 layer-output 0.0 (splice OFF).
5. **Propagate** across all 64 layers (+ check the 16 full-attn layers have no own residual).
6. **Re-measure many-event** per-depth argmax + TV/accept; lossless = TV within a **native-vs-native** verify-dist noise floor (measure it as the bar); superset = accept/event ≥ native.
7. **Sweep every sub-kernel for fp32↔bf16 boundary mistakes** (the conv class — never repeat).

## HARD CONSTRAINTS
- **REWARD-HACKING RULE:** routing/splicing the spine through native (`causal_conv1d_update`/FLA) to pass a metric is REWARD HACKING. The splice (`FR12_NATIVE_SPINE_ORACLE=1`) is an **oracle only**; every "0.0" must be verified **splice OFF** (our kernel computing). Reject any green number that comes from calling native.
- **ONE GPU JOB AT A TIME.** No concurrent `docker run --gpus`; kill leftover containers; `torch.cuda.empty_cache()`; boot-free (no server) whenever possible.
- **Read LIVE vLLM source** (`/tmp/vllm-0.22-src`, `/tmp/vllm_live_019`) before patching — do not guess from numbers.
- **NO copy / state-copy / weight-re-stream / dense path.**
- **Commit + push every real step** to `fr12-wy-tree-kernel`; numbers in committed docs.
- **DO NOT close (pass/fail) without asking the user.**

## Measurement regime (user, 2026-06-06) — diagnostics eager/B1; the GATE is B=4 CUDA-captured SWE-4
- **Diagnostic captures run EAGER (`ENFORCE_EAGER=1`)** — the parity-capture hooks (tensor dumps) aren't Dynamo-traceable, so the Dynamo/graph crash during profiling is the **capture instrumentation, NOT the kernel** (the Triton scan is a Dynamo-opaque custom op; it ran clean in every spine capture). Defer the capture-graph crash; keep diagnostics eager.
- **The FINAL lossless+superset gate must be B=4, CUDA-graph-captured, SWE-Verified 4 tasks** (the FR9 `swe-bench-agentic-b4-four-verified` workload — NOT eager / B=1 / toy 8-prompt). Reasons: (a) B=4 changes co-residency — the scan's N-independence must hold under 4-request concurrency, not just full-tree-vs-spine-only at B=1; (b) the CUDA-captured path can round/behave differently than eager; (c) the toy prompts aren't the real workload.
- **The DELIVERABLE comparison (user, 2026-06-06): E5 (FLASH_ATTN native MTP-5, the standard baseline — `output/fr10_native_mtp5_same8_*`) VS TREE_ATTN+tree-verify (our system)**, for BOTH (a) **LOSSLESS** = our accepted distribution within E5's self-noise floor, and (b) **SPEED/SUPERSET** = accept/event + decode-TPS vs E5. The per-layer bit-exact 0.0 (incl. tree-TREE_ATTN vs native-TREE_ATTN) is a **DEV correctness check only** — it proves our kernels/wiring are right under a fixed backend; it is NOT the deliverable bar. Do NOT declare lossless by comparing tree-TREE_ATTN vs native-TREE_ATTN (trivially 0, dodges the question) — the lossless number MUST be vs E5/FLASH_ATTN. The full-attn `attn_out` 0.00195 (TREE_ATTN vs FLASH_ATTN) is a backend-numerics difference judged here: within E5 floor → acceptable (TREE_ATTN deploy wins); beyond → fall back to FLASH_ATTN+tree-mask (bit-exact full-attn vs E5).
- **Bit-exact 0.0 (spine + branches) must be re-confirmed at B=4 CUDA-captured**, not only eager-B1. And **confirm the kernel itself CUDA-graph-captures cleanly with capture-hooks OFF + graphs ON** (tree shapes / N_PAD in the captured sizes) BEFORE the final run — that property is untested (eager-opaque ≠ graph-capturable).

## Definition of done
scan + gate + o_proj == 0.0 at L0 (splice OFF) **on BOTH the spine (vs native linear chain) AND every branch node (vs the native-on-branch-path oracle)** — eager-B1 diagnostic — then all 64 layers, then **re-confirm 0.0 at B=4 CUDA-captured**, then the verdict: **B=4, CUDA-graph-captured, SWE-Verified 4 tasks** — many-event per-depth argmax match + TV within native-vs-native floor + accept/event ≥ native. Bring numbers to the user before any closeout.
