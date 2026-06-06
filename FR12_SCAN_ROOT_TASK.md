# FR-12 (codex_fr12) — drive GDN scan → gate → o_proj to BIT-EXACT 0.0, then lossless

**Branch:** `fr12-wy-tree-kernel` · **Driver:** codex gpt-5.5-high (codex_fr12) · **Red-team + loop:** Claude (Opus), 10-min. **ONE GPU — strictly serial.** Read `FR12_REDTEAM_ARGMAX_LAG.md` + `FR12_PARITY_RESULTS.md` (full state) and the committed probe scripts before starting.

## THE HARD GATE (user, non-negotiable)
Drive the per-layer GDN sub-kernels **(3) gdn_scan, (4) RMSNormGated gate, (5) o_proj** to **bit-exact 0.0** (tree spine == native; below fp8 bucket resolution so no byte flips) **BEFORE any other kernel work** (speed, branches, other model parts). Do NOT "let small residuals go" — they compound over 64 layers and ARE the whole drift.

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

## Definition of done
scan + gate + o_proj == 0.0 at L0 (splice OFF), then all 64 layers, then many-event per-depth argmax match + TV within native-vs-native floor + accept/event ≥ native. Bring numbers to the user before any closeout.
