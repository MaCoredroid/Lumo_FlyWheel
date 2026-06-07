# FR-13 red-team: WHERE the 2.61→0.92 spine-accept gap lives (no hand-wave)

**Date:** 2026-06-07 · Claude red-team of the FR-13 E5 deliverable, driven by a 4-agent localization workflow (source-readers, no GPU) + my own source resolution of an agent disagreement. Branch: main.

## The result being explained
E5 (FLASH_ATTN native MTP-5, num_spec=5) vs our tree (TREE_ATTN tree-verify, num_spec=9), B=4, temp0.6, top_p0.95, SWE-4, CUDA-graph captured:
- accept/event: **E5 2.61 / tree 0.92** (2.8x deficit). acc/draft-token: E5 0.52 / tree 0.10.
- warm decode TPS: **E5 16.5 / tree 4.8** (3.4x slower; the gap exceeds the accept gap → tree per-forward is also heavier).
- bag_TV(tree,E5)=**0.558** vs E5 self-noise floor **0.059** (~9x). tree self-noise bag_TV=0.252 (4x floor). first_token_TV=0.0 everywhere.
- finish_reason: E5 64/64 `length`, **0 early stops** (128 samples). tree **16/64 `stop` (EOS)**, one at **4 tokens**. → spurious EOS the target ~never emits.

## Verdict: the tree verify is LOSSY (not lossless-but-slow). The cause is VERIFY-side.

### 1. Drafter — EXONERATED (source-proven, overturns a workflow agent)
The deliverable config (mode=tree_mtp, num_spec=9, the launch `TREE`) **fires the FR10_CATERPILLAR_NATIVE_SPINE_TOP2 drafter** (fr10_phase4_patch_vllm_tree_gdn.py:4514-4665), NOT stock `propose_tree`. Proof of the gate firing:
- gate requires `self.tree_choices == [(0,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]` (sorted by (depth,lex)).
- `config/speculative.py:594-595` **sorts the tree and writes it back**: `self.speculative_token_tree = str(sorted(tree_choices, key=lambda t:(len(t),t)))`; `eagle.py:259-260` reads that sorted string into `self.tree_choices`.
- The launch `TREE` (fr10_launch_speed_server.sh:64) sorted == the gate verbatim → `_fr10_is_caterpillar=True` → returns `_fr10_packed` before stock propose_tree (L4750) is reached.
- So path0 == native MTP-5 depth-5 chain (post-fix parity 32/32 all depths, output/fr10_drafter_fix_confirm_20260604T200639Z), and the spine slots [0,1,3,5,7] match the sorted-tree verify slots (no drafter↔verify slot mismatch). **A 2.8x collapse CANNOT come from the drafter.**

### 2. Co-residency / GDN batch-invariance numerics — EXCLUDED
GDN scan is per-sequence-independent (chunk_delta_h.py grid (cdiv(V,BV), N*H); per-seq cu_seqlens recurrence; autotune keys don't depend on N). Co-residency shifts only fp32 reduction order ~5.96e-8 → ~1.9e-6 gate → ~1.2e-4 o_proj (FR12_SCAN_ROOT_TASK.md:19-25). Empirical ceiling: E5 self-noise moves accept ~3% (2.61 vs 2.69). **A few-percent effect cannot produce a 180% (2.8x) drop.**

### 3. The gap is the FULL-ATTENTION tree-verify divergence (PRIME, consistent with all evidence)
- GDN (linear-attn) sub-kernels are bit-exact 0.0 spine+branch (FR12 gate). But the stack **first diverges at layer-3 = first full_attention layer, max_abs 0.0040 (eager-B1, FR12_PARITY_RESULTS.md:1064-1090)** and was **NEVER driven to 0 through all 64 layers / final logits**.
- The proven-0.0 was eager / B=1 / 8-toy-prompt / one decode event — a regime **structurally excluded** from the B=4 CUDA-captured SWE-4 deliverable (capture hook does in-forward .cpu() copies, fr10_phase4_patch_gdn_capture.py:77-86 → forces ENFORCE_EAGER; compare scripts index one request). The exp2 TREE_ATTN full-attn path's per-layer parity vs native FLASH_ATTN was **never measured** (FR13_RESULTS.md:121).
- Signature match: first_token_TV=0.0 (prefill/depth-0 fine) + divergence accumulating with depth + spurious EOS at later positions + spine rejected at depth≥1 = a verify whose logits drift wronger as decode proceeds. base-e (0.918) ≈ exp2 (0.922) → not an exp2-specific bug; the underlying full-attn tree divergence is present in both.

## Open magnitude question (the decisive experiment)
0.00195 (TREE_ATTN base-e vs FLASH_ATTN, eager-B1, depth~0) → bag_TV 0.558 + 65% spine rejection is a BIG jump. Must confirm the compounding, not assume it. **Decisive measurement (codex, ONE GPU, serial):** per-layer + FINAL-LOGITS spine parity (tree-verify spine vs native MTP-5 chain) over MULTIPLE decode positions, **B=1 eager first** (isolates the fundamental full-attn divergence from co-residency+capture). If lossy even at B=1 eager → the full-attn tree divergence is the confirmed root; localize the EXACT divergent op in tree_attn.py(TREE_ATTN) vs flash_attn.py(FLASH_ATTN) — softmax-scale placement / accum dtype / tiling-reduction order / online-softmax rescale / qk dtype / cast boundary — then decide alignable(→0.0, fix our kernel/wiring) vs real algorithmic diff (→ FLASH_ATTN + tree-mask, per the user's own within-floor decision rule). Do NOT patch FLASH_ATTN until TREE_ATTN is confirmed dead.

## Decision rule (user's own, now triggered)
"our dist within E5 floor → TREE_ATTN deploys; beyond → FLASH_ATTN+tree-mask." Deliverable bag_TV 0.558 >> floor 0.059 → **TREE_ATTN does NOT deploy within floor as-is.** But CONFIRM the localization (full-attn is the whole gap) before committing to FLASH_ATTN+tree-mask — the magnitude jump warrants the per-layer+final-logit measurement first.
