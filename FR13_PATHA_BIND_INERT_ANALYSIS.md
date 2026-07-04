# FR13 Path A (FR13_APC_BLOCK_REFOLD) — bind diagnostic verdict + why the bind fix is likely a dead end

Date: 2026-07-04. Task: astropy-13453, cat8 cache-ON + EXACT_SEED + FR13_APC_BLOCK_REFOLD=1 + FR13_SERVE_LOG=1, nudge-free qwen-code. Container fr13-bigdenom-cat8_refold_diag (torn down after capture).

## 1. Bind diagnostic verdict (the direct question)

"Does Path A's restore read the folded value or fall back?" — **it falls back, 100%.**

| marker | count | meaning |
|---|---|---|
| FR13_REFOLD_APPLIED | 720 | the fold fires (a lot) |
| FR13_REFOLD_BIND bound=False | 60 / 60 | the write NEVER binds |
| FR13_REFOLD_RESTORE_USED | 0 | the restore NEVER reads Path A's fold |
| FR13_REFOLD_RESTORE_OTHER | 30 | restore reads the OTHER channel (EXACT_SEED prefill-capture) |
| FR13_REFOLD_SKIP | 816 | skipped (mostly zero_accept) |

## 2. The bind bug, precisely (proven)

- Path A folds + labels boundaries at **64-token** granularity: `_rf_bs = 64` (patch:8050), publishes PENDING at `_rf_blk_end = _rf_abs_base + 64` and binds at that pos (patch:8204-8237).
- The block-hash keys the bind reads are recorded at **runtime block_size** boundaries: `_fr13_es_pos = (num_cached + i + 1) * block_size` (patch:18567-18569), and the runtime block_size is **832** (`ES_GATE bs=832`; ES_WRITE pos ∈ {832,1664,…,39936} = multiples of 832).
- 832 = 13×64, so a 64-labelled pos is a block boundary only 1-in-13 times. The logged BIND was at **pos=21696** (NOT a multiple of 832); the restore fell to OTHER at **pos=21632 = 832×26** (a real boundary). 60/60 miss.

Fix for THIS bug (if pursued): publish/bind only when `_rf_blk_end % _FR13_ES_BLOCK_SIZE == 0`, and gate on a genuine HIT-seeded abs_base (E3 seeds it to the restored 832-multiple; MISS defaults to 0 → would false-bind decode content onto prompt-block hashes). Keep the 64-fold + rolling checkpoint every 64 (bit-exact).

## 3. Why fixing the bind is likely a DEAD END for the give-up

The ES engagement log shows the sequence grows 832 → 39936 across turns, and **restores land at deep decode-region boundaries** (ES_SEED_APPLIED pos = 21632, 24128, 24960, 28288, 29120, 31616, 32448, 33280, 34944, 36608, 37440). Crucially, **every one of those positions carries an EXACT_SEED prefill-capture ES_WRITE** (96–192× each), i.e. each restored boundary is re-folded FAITHFULLY (chunked-FLA seeded from the restored base) on the turn that re-prefills it.

- Path A's fold = `chunk_gated_delta_rule(k,v,g,beta, initial_state=restored_base)` over the accepted tokens.
- Prefill-capture's write = the SAME `chunk_gated_delta_rule` over the SAME tokens (they are next turn's prompt) from the SAME restored base.
- → **identical value.** Binding Path A's fold changes the restored state by ~0.

So the drift that causes the give-up is NOT in the cached boundary states — those are faithful. It is the **decode-kernel discontinuity**: the co-resident tree-scan decodes on a drifted trajectory (0.0289 state gap, BV16/w8 spill geometry), while the restored/cached states are faithful chunked-FLA. The model decodes off one trajectory and gets re-based onto another at each restore.

**Path A operates at the committer (post-accept) and writes the cache — it cannot change the decode kernel.** recompute fixes the decode kernel at the source (native BV32/w1 geometry, bit-exact to the per-path serial recurrence) and ENGAGES 46 min real-task. That is why recompute works and Path A (even bind-fixed) is expected not to.

## 4. Residual uncertainty (the ~30% case)

If a decode-region boundary gets its hash recorded DURING decode (drifted state cached) and is HIT on a later turn WITHOUT being re-prefilled, prefill-capture never launders it and Path A's faithful fold WOULD differ (help). The log shows prefill-capture DID write those boundaries (argues against this), but write-vs-restore ordering per turn is not fully pinned. The only way to settle it is to implement the bind fix and re-run — RESTORE_USED>0 AND give-up resolves ⇒ Path A helps; RESTORE_USED≈0 or give-up persists ⇒ confirmed dead end.

## 5. Recommendation

Pivot to **recompute** (FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=recompute): run the native-MTP-cache-ON vs cat8-recompute-cache-ON 16-task lossless comparison (the real PASS gate). The bind fix is a delicate multi-site edit whose own diagnostic data predicts no behavioral change.
