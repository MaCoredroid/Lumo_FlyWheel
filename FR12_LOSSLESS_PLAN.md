# FR-12 (revised) — Lossless tree verify FIRST, no-copy, no slow paths

**Direction (user, 2026-06-06):** achieve **LOSSLESS + SUPERSET first, speed later**. **De-prioritize the WY kernel** (the scan was already exact; it's the *speed* lever, not the lossless one). **Chase lossless by making each decoder layer's tree-spine output equal native MTP-5, replacing whatever kernels are needed.** **Avoid copy / obviously-slow paths** (no per-spine state copy, no separate native-shape launch / weight re-stream, no dense O(N²) per-node solve).

## The hypothesis we're testing (user's, and it's correct)
> If every layer's verified output for the spine equals native, the tree verify is lossless.

True. The obstacle (FR11) is that it *doesn't* equal native today: the tree-spine per-layer output drifts from native ~0.0156 @ GDN L0 → ~53 @ L63. The scan is NOT the cause (byte-native 7.45e-9). The cause is **shape-dependent per-layer drift** — the tree's branched row-layout makes each layer's fp8 GEMM / chunk / attn round differently than native's linear chain.

## The crux: "match native" ≠ "be self-consistent"
Native MTP-5 runs the **default** kernels. So the fix must make the tree-spine's per-layer output equal **native's actual output** — not merely batch-invariant. This is why `BATCH_INVARIANT=1` made it **worse** before: it changed the tree's rounding *away* from native's default path, increasing the gap. The correct fix makes each tree row's computation **independent of co-resident rows AND reproducing native's per-row rounding**. The clean no-copy realization of that is **per-row (per-token) consistency**, e.g.:
- **fp8 GEMMs (in/out/MLP proj):** per-token quantization scaling (each row's scale depends only on that row), not per-tensor (which depends on the co-resident batch max → drift). Row i's output then equals native's row i.
- **GDN scan:** the spine path's recurrence already uses only its ancestors (tree mask). Make the chunk layout/reduction for the spine row native-aligned & co-resident-independent — no state copy.
- **Full-attn (16 layers):** tree-mask softmax is per-row exact; ensure the kernel's per-row output is co-resident-independent.
- **conv1d:** per-row causal window over ancestors (already tree-aware); fix any batch/dtype seam (FR12 found the tap-dtype one).

All of these are **no-copy, not-slow** (per-token scaling / per-row kernels), explicitly avoiding the FR9 copy and the re-stream paths.

## Plan (measurable, layer-by-layer)
1. **Per-layer parity harness** (reuse `scripts/fr10_layer_hidden_spine_compare.py`): tree-verify + native MTP-5 on the SAME decode event; capture each decoder layer's spine-row output; diff; find the **first divergence** and **which sub-kernel** (read source: in/out/mlp fp8 GEMM? conv? scan? full-attn?). Use the original (non-WY) scan for stability.
2. **No-copy fix that matches native:** make that kernel's spine-row output equal native's (per-token scaling / row-independent), reading source to confirm it reproduces native's path — not just self-consistency. No copy, no re-stream, no dense solve.
3. **Propagate** layer-by-layer until all 64 match → **lossless**.
4. **Then superset:** accept/event ≥ E5 (native MTP-5 ~3.08-3.34 on SWE-Verified).
5. **Then speed:** only after lossless+superset (this is where the WY kernel + matmul pipeline come back).

## Guardrails
- ONE GPU job at a time; read code + prior commits (FR11 per-layer diagnostic, conv tap-dtype seam, **why BATCH_INVARIANT failed**); don't guess from numbers.
- Verify against E5 (native MTP-5): lossless = acceptance-length TV vs E5 within self-noise floor (~0.0188); superset = accept/event ≥ E5.
- Do NOT close (pass/fail) or adopt a copy/re-stream/dense path without asking the user.
