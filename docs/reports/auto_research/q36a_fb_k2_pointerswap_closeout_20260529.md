# Q36-A F_b K=2 pointer-swap session — close-out summary

Date closed: 2026-05-29
Session arc: ~40 hours of monitored codex iteration across two codex sessions (codex_fb, codex_swap).

## Headline

F_b K≥2 with strict no-copy is **architecturally complete** at commit `7761de95` (HEAD), but **K=2 acceptance stalls at K=1 parity** for this workload's event geometry. The decisive structural finding: vLLM `block_size = 816` tokens vs typical accepted-suffix length ~5 tokens → block-aligned pointer-swaps almost never fire, so the K=2 winner-suffix never materially propagates into K=1's trunk. Pointer-swap is now correctness-preserving; not the performance lever.

## What was achieved (vs the three user constraints)

| Constraint | Status |
|---|---|
| (1) **FP8 production** | ✅ Restored. Earlier silent `weight_scale_inv` skip-loading was identified and fixed (`1eb3389d`, `1a9321e1`); clean E3 baseline reproduces 17.36 tps / 2.17 acc/ev (May-27 reference 17.523/2.197). `cd46d2fe` prevents non-Fb launches from inheriting F_b patches. |
| (2) **Add a row, NOT copy** | ✅ Verified zero-copy. `LUMO_FB_NO_KV_PREFIX_COPY=1` + split-attention path (`ce6543ab`) + RID-keyed pointer transfer (`4f49afd1`, `21434aa8`, `9654358f`, `3e287c7e`) yields telemetry: `fb_state_copy_bytes=0`, `fb_kv_blocks_copied=0`, zero `split_kv_suffix_commit_copy` events with non-zero bytes. Partial-head events safe-fallback to parent-row in-RAM mirror (no bytes moved). |
| (3) **K=2 strict superset of K=1** | ⚠️ K=2 ≈ K=1 parity within sample noise; not strictly exceeding K=1 on mean acc/ev. Best paired same-runtime measurement (commit `21434aa8` rerun): K=1 18.46/3.10, K=2 12.73/3.03 (K=2 accept-all-5 slightly higher than K=1 by ~2 pp). |

## Final measurements (production FP8, 256-tok temp 0.6, 3-prompt diag, gpu_mem 0.87 unless noted)

| Build | tps | acc/ev | accept-all-5 | notes |
|---|---|---|---|---|
| Clean E3 baseline (re-verified) | 17.36 | 2.17 | depth-3 ceiling | in-session fresh-E reference |
| F_b K=1 single-flip (21434aa8 rerun) | 18.46 / 18.81 / 19.50 | 3.10 / 3.16 / 3.32 | 38-47% | beats fresh-E |
| F_b K=2 single-flip (21434aa8 rerun) | 12.73 / 13.01 / 13.21 | 3.03 / 3.10 / 3.16 | 40-43% | K=1 parity |
| F_b K=2 pointer-swap (3e287c7e, gpu0.86 non-gate) | 12.78 | 3.03 | — | true zero-copy verified |
| User floor (debug 64-tok one-off, never reproduced at scale) | 14.999 | 3.500 | 53% | not cleared |

## Why K=2 plateaus at K=1 parity (the structural finding)

Codex_swap's diagnostic (post-`9654358f` helper fix + post-`3e287c7e` partial-head fallback):

- vLLM `block_size = 816` tokens per page-table entry.
- F_b K=2 internal-winner *accepted* suffix lengths at depth-5 spec-decode are ~5 tokens.
- Suffix starts at arbitrary offsets (37, 45, 58, …) — never on a 816-token block boundary.
- Therefore: **zero block-aligned pointer swaps fire** in 256-tok diag windows; every K=2 winner falls back to parent-row promotion (the safe baseline path). The pointer-swap mechanism is wired correctly but the workload never triggers it usefully.

The pointer-swap is a correctness-preserving zero-copy mechanism for *rare* aligned events; for this workload at depth 5 + block_size 816, it cannot be the performance lever.

## Significant bugs found and fixed along the way (24 commits since session start)

- FP8 metadata: vision-skip enumeration (`1a9321e1`), inject quant metadata before launch (`1eb3389d`)
- Launcher contamination prevention (`cd46d2fe`) — non-Fb launches no longer inherit F_b source patches
- Batch-shape-dependent GDN projections — padded batched projection (`1ee4930e`), batch-invariant FP8 (`e96cd945`), qkvz padded (`b50b3eb3`)
- F_b sampler / commit logic — parent double-promotion (`53aeff72`), share target samples across prefix rows (`c0e9be50`), preserve active rows for pre-update pointer swap (`7334f03e`), runner kernel-row helper (`9654358f`), partial-head fallback (`3e287c7e`)
- KV layout: split prefix/suffix attention (`ce6543ab`), pointer transfer wiring (`4f49afd1`, `21434aa8`)

## Artifacts on disk

- 30+ probe JSONs in `output/spec_speed_probe/Fb_n5_*` (each iteration's measurement preserved).
- `output/diagnostics/vllm_live_vs_pristine_0.19.0_20260528.diff` — 5,701-line enumeration of 14 modified vLLM source files.
- `output/diagnostics/container_cache_state_20260528.txt`
- `/tmp/pointerswap_brief.md` — original brief that drove codex_swap's work.
- `/tmp/pivot.md` — user-directive history (final F_b K≥2 + 3 constraints + floor).

## Recommended next step (when work resumes)

Codex_swap delivered a path-(c) design doc proposing 4 alternative K=2 tree shapes:

1. **Adaptive Root-Diversity K=2** (recommended #1) — top-2 at pos 0 only (2 rows), enabled only when pos-0 confidence is low. Best chance to keep tps ≥ 14.999 while improving low-confidence events.
2. **Fixed Beam-Search K=2, 2 rows** — best chance to reach acc/ev ≥ 3.5 if verify cost stays close to root-diversity.
3. **Adaptive Beam-Search K=2** — combine adaptive triggering with beam.
4. Wider K at fewer positions — second-tier.

Pointer-swap stays implemented (correctness contribution) but the next investigation is tree-shape design, not more KV-manager work.

## Open SWE-Bench round

Task #11 (Launch q36a SWE round on F_b K≥2 winner) remains pending. The current K=2 build (K=1 parity, verified zero-copy) is *shippable* if you accept that K=2 doesn't strictly exceed K=1 at the per-prompt level — running the round would tell whether the row-1 contribution helps at suite level. Not launched in this session per the "do not auto-ship" rule.

## Operational state at close

- HEAD: `7761de95 Record K2 diag3 after partial-head fallback`
- Container: `lumo-vllm-track-b-suffix Up ~4h` (gpu0.86 non-gate diagnostic build), can be torn down.
- Tmux sessions: `codex_fb` killed; `codex_swap` idle since 1h 32m completion report; `codex_fa` idle throughout.
- Cron job `3dda92d1` (10-min monitoring loop) deleted.

End of session.
