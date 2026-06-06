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

## Branch losslessness — the bar for nodes native MTP-5 never computes (user, 2026-06-06)
Native MTP-5 computes ONLY the spine (path0 linear chain), so the off-spine branch nodes have **no native MTP-5 counterpart** to diff against. The spine isn't the whole gate. The bar for the whole tree:
> **Every node's verify logits (spine AND branch) == the TARGET MODEL's true logits for that node's ancestor-path.**
> - spine path → native (its own chain),
> - each branch path → the target model run on **that branch's linear ancestor-path** (the "branch-path oracle").

Why accepting branches stays lossless: the rejection-sampler / multi-draft committer emits the **target distribution** given correct per-node verify logits, for ANY number of candidates. A branch is accepted with `min(1, p_target/q_draft)` using ITS path's correct logits → output stays target-distributed (lossless), we just reach more tokens/event (the **superset**). A 1-ULP-wrong or argmax-flipped branch logit → accept with the wrong probability → NOT lossless. So branches carry the **same** bit-exact bar as the spine, measured against their **own path's** oracle.

**Validation (whole tree, our kernel doing the compute, splice OFF):** (1) spine == native; (2) for each distinct branch path, our kernel's branch logits == native run on that path. Both bit-exact / within floor + per-depth argmax.

**CAVEAT — do NOT take the branch-path-oracle for granted (user, 2026-06-06; online research + think-more required before trusting it):**
- **Oracle must be NO-MTP native, not MTP-5.** Lossless bar is vs the true target dist (no-MTP); MTP-5 itself drifts ~6e-5 from no-MTP. Run native **no-MTP** on each branch path.
- **RoPE positions must be depth-based** (FR10 depth-RoPE finding): the branch token sits at its tree depth; the linear branch-path oracle must place it at the same depth, else the oracle is the wrong ground truth.
- **Re-running native on the linear branch path only reproduces the tree's shared-ancestor computation IFF the shared prefix is bit-exact AND batch-invariant.** The branch-path oracle therefore also exercises the #42960 co-residency seam — a feature (it catches it) but means a branch "mismatch" can be an ancestor-state problem, not a branch-update problem; attribute carefully.
- **Per-node-correct logits is the NECESSARY (kernel) half; the committer (multi-round tree rejection sampling) is the other half** — confirm the committer's losslessness theorem holds for temp>0 (we run temp0.6), not just greedy. (Open research item — see branch-oracle research.)
- Branch-path oracle is **validation-only / offline** (one native run per distinct branch path); it is NOT a runtime path and must never leak into the served kernel.

## Guardrails
- ONE GPU job at a time; read code + prior commits (FR11 per-layer diagnostic, conv tap-dtype seam, **why BATCH_INVARIANT failed**); don't guess from numbers.
- Verify against E5 (native MTP-5): lossless = acceptance-length TV vs E5 within self-noise floor (~0.0188); superset = accept/event ≥ E5.
- Do NOT close (pass/fail) or adopt a copy/re-stream/dense path without asking the user.
