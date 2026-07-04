# FR13 APC — Session Digest & Cherry-Pick / Phase-4 Plan

Scope: FR13 mamba/GDN prefix-cache (APC) losslessness + give-up work, 2026-06-28 → 2026-07-04.
Sources: 7 session digest chunks (06-28 18:50 → 07-01 02:14, truncated) + PICKPLAN + VERIFY_PICKTREE +
VERIFY_DIGEST + VERIFY_LEAKTEST. Verifier verdicts: DIGEST holds (no refutations), PICKTREE holds (no
required fixes), LEAKTEST holds **with 7 attribution/monitoring refutations** (see §5).

Convention: `REFUTED`/`CORRECTED` tags mark claims a verifier or a later same-session finding overturned.

---

## 1. PREVIOUS SESSION SUMMARY (chronological)

### Phase A — align-mode carrier & block-size band-aid (06-28 18:50 → 23:09)
- Killed a vacuous conv-redirect gate; conv thread **CLOSED/banked** — GDN conv+ssm restore 48/48 faithful,
  conv redirect was on the WRONG carrier (compaction-summary stale framing). [06-28 19:31]
- Global-shadow experiment proved the shadow bypass is complete/non-vacuous: **carrier = APC MACHINERY, not
  the cached VALUE**. Decisive differing path = `mamba_cache_mode='align'`. [06-28 21:49]
- `mamba_cache_mode='align'` = 0/6 solve; `'none'` = 5-6/6. no-spec+cache tolerates align 2/2 → **failure
  requires align AND spec together (spec-amplified, not spec-caused)**. [06-28 21:55]
- Config lever found: `mamba_block_size 1024→8192` = ~8x fewer align boundaries (~30→~3 per 30K prefix),
  config-only. block=8192 rollout1 RESOLVED 12907, cached_max 28560, no char-8, no runaway — first spec+cache
  solve with both speedups live. [06-28 22:43] **CORRECTED later: 1/1 → actually 1/2** (r2 char-8 failed).
- `REFUTED (same session):` "runaway = cache residual" — char-8 tool-call runaway reproduces with cache fully
  bypassed → **cache-INDEPENDENT**. [06-28 21:31]
- Bug-hunt workflow claimed a LARGE wrong-state bug in `_patch_mamba_utils_tree_accept_bias`; Claude
  red-teamed it as **real-but-not-the-carrier** (SNAP_FIX rides collect_mamba_copy_meta, fires ~1/1024 =
  vacuous; shadow already proved cached VALUE isn't the carrier). [06-28 22:52 / 23:00]

### Phase B — drift root-cause = kernel-realization mismatch; redesign chosen (06-28 23:09 → 06-29 14:59)
- Drift curve (fixed-replay diagnostic): 1024 = **77.96** state_max, 2048 = 49.70; >>fp ceiling 0.0078.
  `CORRECTED:` 77.96 is inflated — per-element mean ~0.0069 (fp-level), 77.96 = capture position/turn
  misalignment + thin outlier tail. [06-29 01:01 / 04:19]
- Banked block_size=8192 to config (FR13_APC_BLOCKSIZE_FINDING.md `0cd261c0`; MAMBA_BLOCK_SIZE:-8192 at
  launcher `47a221d0`). Framed as **band-aid**: cuts boundary COUNT not per-boundary error. [06-28 23:57 / 00:01]
- Root cause pinned: align IS a true continuation, but **prefill writes via `chunk_gated_delta_rule` (chunked
  WY/UT) while decode/spec write via `fused_sigmoid_gating_delta_rule_update` (sequential rank-1)** →
  ~0.0078 per boundary. block-size = band-aid. [06-29 04:18]
- `REFUTED (same session):` "GDN restore 48/48 faithful" likely **vacuous** — capture plumbing was dead on
  cfg/on arms until fixed. [06-29 01:01]
- `REFUTED:` PRE_SNAP_FIX 3-line wiring fix = **VACUOUS** (77.9571 vs 77.96 baseline, literally identical =
  3rd vacuous redirect after conv-redirect and baked SNAP_FIX). [06-29 02:31]
- Redesign chosen = **single-realization invariant** (SGLang MambaRadixCache / Marconi): cache the chunked
  realization at 64-aligned positions, restore through the SAME chunked kernel. `mamba_cache_mode='all'` is
  **hard-blocked 3 ways** for Qwen3-Next+spec (config.py:348 demote; qwen3_next.py:707 NotImplementedError;
  config.py:330 spec→align). [06-29 04:18 / 14:59]
- EXACT_SEED pivot loop iter1-7 (this window): per-req/slot keying **doomed** (req_id changes prefill→decode,
  0 overlap); pivot to prefix-hash cache-block storage + prefill-capture compute. [06-29 06:54 / 09:10]
- Discovered vLLM forces requested 1024 → **actual 816** (page alignment, 816%64=48≠0) → the %64 bit-exact
  gate can never pass; iter1-4 stall root. [06-29 13:55]

### Phase C — 64-align + EXACT_SEED chain lands; L0 gate passes (06-29 15:02 → 20:34)
- Design pivot corrected: make block_size a **multiple of 64** (`--block-size 1024` passthrough,
  `APC_BLOCK_SIZE=1024`), NOT waive the %64 gate. 816 = pure memory-layout floor (16×51), forcing 64-aligned
  832/1024 proven SAFE by workflow. [06-29 15:04 / 15:39]
- Six-fix EXACT_SEED chain: 64-align block (`4b007ac3`) + postproc-relay-disable (`62202fd4`) + context_lens
  b0 (`0b84a2f3`) + bidirectional (req,pos) join (`8dc92f50`) + per-layer keying + bare-hash key (`cd703875`).
  Drift chain **77.96(816) → 38.36(1024 no-fix) → 30.11(sub-baseline, restore correct)**. [06-29 18:09]
- Layer-0 "residual" proven a **metric artifact** (L0 forget-gate ≈1, state magnitude ~90; REF-vs-REF
  self-compare gives L0 max 46.58 > cache's 30.11 → restore is bit-exact, flat 0.0078 threshold can't judge a
  magnitude-90 state). L0 state-diff gate **PASSED** (`efb9445b`). [06-29 18:07 / 18:09]
- Paper bank `0fb18c80`, main FF; L1 live gate launched (cache-ON EXACT_SEED+1024 vs cache-OFF, temp 0.6,
  PIECEWISE, N=3). [06-29 19:29 / 19:03]

### Phase D — give-up root-cause (thinking template) + full cuda graph (06-29 20:35 → 06-30 05:52)
- L1 v1 "failure" = **own GPU guard** SIGKILL at 8966 MiB (34 MiB under 9000 floor), exit137/OOMKilled=false —
  NOT cache-lossiness (user hint "make sure it snot your memlimit tripping" correct). Floor 9000→4000. [06-29 20:35]
- **~100% attempt-1 give-up root cause = thinking-template field-name mismatch**: vLLM emits
  `{'role':'assistant','reasoning':…}` but `qwen3-openai-codex.jinja` reads `.reasoning_content` → chain
  dropped at render → model re-derives blind, ends turn with no tool call. Two-layer fix (field-name + anchor
  thinking-visibility at FIRST user query) baked default-on, `9fd1b402`. v4 give-ups 100%→0 (cache-OFF). [06-29 22:57 / 23:40 / 00:43]
- **Full cuda graph confirmed lossless**: EXACT_SEED already makes full-graph cache-ON garble-free (12907
  replay clean, CJK=0). Garble was a SYMPTOM of the lossy cache, not a separate CUDA bug; **task #8 closed**,
  PIECEWISE was only the pre-EXACT_SEED mitigation. [06-30 03:24 / 03:47]
- `REFUTED:` full-graph seed-row fix "B-1" **disproven** before shipping (kernel already reads correct
  committed-leaf column; B-1 would corrupt multi-accept decode). [06-30 01:53]
- `REFUTED:` char-8 workflow headline (cache AND spec, Fisher p=0.004) did **not** survive its own red-team;
  ~50/64 char-8 traces were COLD/no-cache at the breaking request; ~3.8% base-rate qwen tool-call flake,
  cache-independent, guided-decoding-fixable. [06-30 04:02 / 04:22]

### Phase E — the memory leak, committer deletion, speed numbers (06-30 05:52 → 18:04)
- Speed gate exit137 = gpu_oom_guard killing on a **~0.7 GiB/min unified-mem serving-path bleed** (not
  steady-state, not host-OOM). Leak A = committer per-slot GPU dicts + EXACT_SEED pending/ckpt dicts, never
  popped at `_free_request`. [06-30 05:55 / 06:08]
- Committer-prune extended survival 34→58min, cut bleed 0.7→0.4 GiB/min (`1f7b1301`) but **not fully gone**.
  Fixed-buffer port (FR13_APC_FIXED_BUFFER, default OFF, unit bit-exact 9018 chunks, `11ed5304`). [06-30 07:29 / 09:25]
- Key structural finding: **committer drain NEVER fires live** (ES_CHAIN_PUBLISH=0); prefill-capture is the
  sole source of cache-block WRITEs. So committer + its .cpu() syncs are dead weight → **deleted** (`98a809b7`).
  Deleting committer also fixes the leak (#14) and the fixed-buffer port (#15) becomes MOOT. [06-30 10:37 / 16:07 / 15:41]
- Speed (cache-ON, full-graph, EXACT_SEED): cat6root ~16.8 tok/s (3-task TW), cat10 14.1 < cat6root — **wide
  tree hurts at B=1** (drafts 10 vs 6, accept 35% vs 59%). `CORRECTED:` the "30% tax" and "9% gap" **dissolved
  into basis-mismatch + task-mix** — matched per-forward: current cat6root 17.87 vs historic 17.61 (current
  slightly faster), no real regression. [06-30 09:17 / 16:14]
- HBM-traffic workflow: model is dense fp8, weights = 92% of traffic; cache tax +8.28 MB/step = 0.028% → under-
  explains any gap ~300x; prime non-byte suspect = 48 device→host col0 `.cpu()` syncs/step. [06-30 15:14]
- `REFUTED (noisy):` 2×2 "cache-OFF resolves, cache-ON fails" 4/4 — n=4, ~6% by chance. [06-30 17:34]

### Phase F — give-up cap matrix; 2×2 flips to noise (06-30 18:04 → 07-01 02:14)
- Drift-by-block quantified: 1024 = 78, 2048 = 49.7, 8192 = 4.29 (~18× drop); "bad zone" is a number, drift
  threshold between 4 and 50. [06-30 18:04]
- `REFUTED:` the 4/4 "OFF resolves / ON fails" flipped — e5_ON RESOLVED, e5_OFF FAILED → **resolve is sampling
  noise (temp-0.6 char-8 flake), consistent with cache lossless**. [06-30 20:52]
- Per-turn cap experiments: cap=500 force-closes >500-tok thinking without breaking resolve; **6/6 arms
  (chain5/cat6/cat8 × ON/OFF) all resolved → losslessness intact**. [07-01 00:54 / 02:11]

### Phase G — native-MTP-5 anchors, leak regression, feature branch, close-out (07-01 → 07-04, from close-out/pickplan)
*(not in the digest chunks; reconstructed from VERIFY_DIGEST + PICKPLAN + close-out bbd9619c.)*
- Clean long native-MTP-5 + EXACT_SEED resolves: `m_nat_exseed @ 703f9af4` (07-02, 13453 16min, 4/5),
  `m_nat16 @ a76f2553` (77min, no OOM), `4b68c8af` (37min). [VERIFY_DIGEST]
- **Leak regression window pinned a76f2553..0d12cdbf; first OOM = 0d12cdbf (07-03 18:23)**, ~10 GB/min (a.k.a
  ~3.5 GB/min serving-phase + a ~40 GB first-big-prefill step). Root cause **STILL UNKNOWN**.
- Leak FIX A (`1e1df386`), FIX B (`a2c1a585`), FIX C (`52818d74`), A/B-disable (`f972681e`) **all wrong, same
  failing trajectory; B/C + disable reverted** to dd5578b4 pre-leak state via `837236d0`. CONV_SNAP_FIX ruled
  out (disabling it still leaks). [close-out §4]
- `CORRECTED (memory error):` "recompute engages 46min = works" was a **memory error** — the sole recompute-on-
  13453 artifact is a confounded codex give-up (exit137 OOM, patch_apply_failed); n=1, unresolved.
- Path A v4 engaged 8min+ before the leak kill (`698e01cc`); **bind fix diagnosed but INERT** for the give-up
  (restored HIT boundaries were already faithful via prefill-capture). [close-out §2]
- Close-out FR13_CLOSEOUT_20260704.md (`bbd9619c`): give-up 2×2 premise **shaky** (char-8/N=1/agent confounds).

---

## 2. CURRENT TRUTH TABLE

### Leak
| Item | Status |
|---|---|
| Regression window | **KNOWN** = a76f2553..0d12cdbf; first OOM 0d12cdbf (07-03 18:23) |
| Root cause | **UNKNOWN** — in the window, in-container (freed on death), ~3.5 GB/min bleed + ~40 GB step |
| ES checkpoint maps (FIX A/B) | **EXCLUDED** — bounding them didn't stop it |
| faecc88d `.cpu()` (FIX C) | **EXCLUDED** — reverting it didn't stop it |
| CONV_SNAP_FIX (1053c604) | **EXCLUDED** — disabling still leaks |
| Committer per-slot dicts | **Partially explained** earlier (Phase E, `98a809b7` deletion); NOT proven the 07-03 carrier |
| main (1053c604) position | main is **INSIDE** the window (a76f2553 ancestor of 1053c604 ancestor of 0d12cdbf) → picked build inherits whatever main-baked EXACT_SEED-path leak exists |

### Give-up evidence quality
| Claim | Quality |
|---|---|
| tree+cache/13453 baseline = 2-turn give-up | Real, but **baseline SHIFTED**: CONV_SNAP_FIX (baked @1053c604, commit 4063b346) already moved it 2→~8 turns |
| cache-OFF & native+cache resolve | Real (703f9af4 native resolves; cache-OFF engages 12-22 turns) |
| recompute "works @46min" | **REFUTED — memory error**; n=1 confounded give-up (OOM) |
| Path A rescues give-up | **INERT** (bind-fixed, restored boundaries already faithful) |
| "cache-ON fails / OFF resolves" 2×2 | **NOISE** (temp-0.6 char-8 flake; flipped on re-run) |
| give-up premise overall | **SHAKY** (char-8 n=1, agent-behavior confound, single-task low power) |

### Feature status
| Feature | Status |
|---|---|
| SCAN_ALIGN recompute (plain, NP=0) | On main; default-OFF; plain recompute mode already lands cleanly |
| Recompute node-parallel (NP=1) | Built (kernel grid-z dispatch), default-OFF; **GPU-UNTESTED** (cuda-graph capture + NP=0-vs-NP=1 bit-exactness unproven) |
| Path A block-refold (BLOCK_REFOLD) | Built incl. 832-boundary bind fix, default-OFF; **fires but INERT** for the give-up |
| bind fix | Present (publish/bind at runtime block_size 832, not 64); proven to FIRE, not proven to CHANGE outcome |

### Git state
- Working tree at `/home/mark/shared/lumoFlyWheel` = **main @ 1053c604** (do not touch — live leak run executing from it).
- Branch `fr13-apc-ssm-shadow` tip = `bbd9619c` (close-out); 27 commits above main (6 recompute+NP, 8 Path A+bind,
  4 reverted leak-fix, 8 control-matrix/Track-A/B drivers, 1 close-out).
- Cherry-pick scratch worktree `fr13-mainpick @ 08b629ef` = **10 commits above main** (clean, unpushed).

---

## 3. CHERRY-PICK RECIPE (VERIFIED)

Goal: bring exactly **2 default-OFF features** (recompute node-parallel + Path A block-refold) from
`fr13-apc-ssm-shadow` onto `main` (1053c604), leaving the default path byte-identical. Branch is LINEAR on
main (merge-base == main), so all picks are strict descendants. Method: `git cherry-pick -x` (provenance line
recorded). VERIFY_PICKTREE verdict: **holds, no required fixes**.

### Ordered commit list (oldest → newest; new-sha ⟵ branch-sha)
1. `3a3764a5 ⟵ 7f7f70a2` — [recompute] kernel `scan_align_mode()` docstring "park" note (comment-only, +12).
2. `8a3cc779 ⟵ 56b2a88f` — [Path A base] +332 patcher (refold fn + module globals + E0-E3 sites) + 4 launcher (-e FR13_APC_BLOCK_REFOLD).
3. `39300518 ⟵ 0f1f9234` — [Path A] pre-alloc refold g/beta buffers on `self` in live eager-pack init branch (fixes AttributeError under full graph).
4. `6a5d84bd ⟵ e48b57b7` — [Path A] accepted-path walk fix (`_rf_cols=[0]+node_path[:acc_len-1]` clamped to N_PAD) + SERVE_LOG diagnostics.
5. `263cd092 ⟵ 9ef57cf8` — [Path A] cast fold k/v/g/beta to bf16 (chunk_gated_delta_rule asserts fp32); bit-exact upcast round-trip.
6. `b7f5be90 ⟵ 4a8fa679` — [Path A] restore-source diagnostic tags (`_FR13_REFOLD_WROTE/REFOLD_BIND/RESTORE_USED`). **Path-A dev diagnostic (79b76c84 depends on it; in 837236d0 ground truth) — correctly INCLUDED, NOT a Track A/B diag.**
7. `fd75600e ⟵ 79b76c84` — [Path A] bind fix: publish/bind at runtime block_size **832** boundary, not 64.
8. `798f2cbf ⟵ 4a635805` — [recompute] design doc FR13_RECOMPUTE_NODE_PARALLEL_DESIGN.md (harmless).
9. `6bb5eb61 ⟵ dd5578b4` — [recompute] node-parallel impl (+133 kernel): `recompute_node_parallel_on()` gate + grid-z=node dispatch, default OFF.
10. `08b629ef ⟵ e6d0214a` — [plumbing] launcher `-e FR13_RECOMPUTE_NODE_PARALLEL` + serve_variant `FR13_ALLOW_SCAN_ALIGN` guard-bypass.

### Deliberately EXCLUDED (not features)
- Track A/B diagnostics: 22be5610, d782983d, 2957ea4a, 0f9c5298, 71a0a7a3, c978ed34, 0d12cdbf, ba61bed4.
- Leak-fix churn a2c1a585/52818d74/f972681e + its revert 837236d0 (net-zero on patcher — Path A chain stops at
  79b76c84 which already reproduces 837236d0's patcher content).
- Pure experiment-output logs 6c70ed8c/5845ab56/698e01cc; analysis doc c9deb112; close-out bbd9619c.

### Conflicts + resolutions
- **NONE.** All 10 `cherry-pick -x` applied with zero conflicts / zero manual resolution.
- Clean despite skipping intervening Track A/B diags because: (a) d782983d touches only an env-bridge proof-log
  disjoint from Path A + recompute; (b) 2957ea4a (+89) is fully cancelled by revert 71a0a7a3 (-89), net-zero;
  (c) remaining Track A/B commits touch only `output/` logs or standalone files; (d) main already carries the
  SCAN_ALIGN base + recompute MODE + launcher -e passthrough.
- Recorded decisions: 4a8fa679 INCLUDED (real Path-A prereq, not a named Track diag); 7f7f70a2 INCLUDED (keeps
  kernel byte-exact to 837236d0/dd5578b4); 837236d0 + its churn BOTH skipped (net-zero).

### Residual vs GPU-validated ground truth (VERIFY_PICKTREE re-ran the diffs)
- **kernel** `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` vs dd5578b4 AND vs 837236d0 = **byte-identical** (diff exit 0).
- **launcher** `scripts/fr13_launch_forked_fa2_tree_server.sh` vs e6d0214a = **byte-identical**.
- **serve_variant** `scripts/fr13_bigdenom_swe_serve_variant.sh` vs e6d0214a = **byte-identical**.
- **patcher** `scripts/fr10_phase4_patch_vllm_tree_gdn.py` vs 837236d0 = **3-line delta only**, the excluded
  d782983d env-bridge proof-log at ~L1562 (worker log won't print a `CONV_SNAP_FIX=` field). Non-functional:
  it only reads the already-baked-on-main FR13_APC_CONV_SNAP_FIX flag, gates/alters nothing.
- ast.parse OK (patcher+kernel); bash -n OK (launcher+serve_variant); `git range-diff` shows all 9 feature
  picks reproduce originals with only the added provenance line; single definitions of `_fr13_pathA_refold`,
  `_FR13_REFOLD_ON`, `recompute_node_parallel_on` (no dupes); git status clean; exactly 5 files changed vs main.

### What remains to audit (VERIFY_PICKTREE `required_fixes` = **EMPTY**)
- Nothing at the pick level. The one disclosed residual (d782983d log field) is intentional and inert.
- Operational cautions (not pick fixes) are the §4 interpretation guards + §6 risks: NP=1 GPU-untested,
  Path A likely inert, non-vacuity worker-env needle, leak inheritance.

### Worktree
`/tmp/claude-1000/-home-mark-shared/1297dd77-e0da-41fe-aceb-175500c156f5/scratchpad/picktest`
(local branch `fr13-mainpick`, HEAD `08b629ef`, 10 commits above main 1053c604, clean, **nothing pushed**;
the repo checkout at `/home/mark/shared/lumoFlyWheel` was NOT touched).

---

## 4. PHASE-4 RUNBOOK — 3-arm give-up matrix

Common wrapper (from `matrix_clean.sh` `run()`):
`SWE_AGENT=qwen_code OFFLOAD_CODEX=1 OFFLOAD_HOST=alienware SWE_EMPTY_PATCH_RETRIES=0 FR13_SERVE_LOG=1
FR13_LEAK_PROBE=1 RUNROOT=output/fr13_matrix_clean` →
`bash scripts/fr13_bigdenom_swe_serve_variant.sh <arm> cat8 subset_one_13453.json` (KIND=cat8 ⇒ forked
launcher, EXPECT_RATIO=8).
`<CACHE>` block (all cache arms): `FR13_ENABLE_APC=1 FR13_APC_EXACT_SEED=1 MAMBA_BLOCK_SIZE=1024
MAMBA_SSM_CACHE_DTYPE=float32`.

### Arm 1 — `m_tree_cache_base` (cat8 + cache, NO fixes = give-up control)
- Env: `<CACHE>` only. No SCAN_ALIGN, no BLOCK_REFOLD, no NODE_PARALLEL.
- Runtime-confirm: `container_env.txt` has `^FR13_APC_EXACT_SEED=1$` and `^MAMBA_BLOCK_SIZE=1024$`; cache
  engaged = `ES_GATE bs=832` + `ES_SEED_APPLIED>0`; **MUST NOT** contain `FR13_SCAN_ALIGN=1` /
  `FR13_APC_BLOCK_REFOLD=1`; no `FR13_REFOLD_APPLIED` lines.

### Arm 2 — `m_tree_recompute_np` (cat8 + cache + recompute + NP=1)
- Env: `<CACHE> FR13_SCAN_ALIGN=1 FR13_SCAN_ALIGN_MODE=recompute FR13_ALLOW_SCAN_ALIGN=1 FR13_RECOMPUTE_NODE_PARALLEL=1`.
- Runtime-confirm:
  (a) guard-bypass fires — `FR13_ALLOW_SCAN_ALIGN=1` must be in the **shell** env (host-side gate at
  serve_variant L304; NOT a container -e → will NOT appear in container_env.txt);
  (b) `container_env.txt` has `^FR13_SCAN_ALIGN=1$`, `^FR13_SCAN_ALIGN_MODE=recompute$`,
  `^FR13_RECOMPUTE_NODE_PARALLEL=1$` (via forked-launcher -e);
  (c) **CRITICAL non-vacuity (bug-class #9):** bridge-needle the **WORKER `/proc/<pid>/environ`** for all three
  flags — container_env alone is insufficient (bare FR13_* has been curated out of workers before);
  (d) cache still engaged: `ES_SEED_APPLIED>0` (`ES_GATE bs=832`).

### Arm 3 — `m_tree_patha` (cat8 + cache + BLOCK_REFOLD)
- Env: `<CACHE> FR13_APC_BLOCK_REFOLD=1`.
- Runtime-confirm: `container_env.txt` has `^FR13_APC_BLOCK_REFOLD=1$` (launcher L446 -e); fold firing =
  `FR13_REFOLD_APPLIED>0` (needs FR13_SERVE_LOG=1, set by wrapper; marker patcher L8330); cache engaged
  `ES_SEED_APPLIED>0`; **MUST NOT** contain `FR13_SCAN_ALIGN=1`. Deeper: `FR13_REFOLD_BIND(bound=T)` /
  `RESTORE_USED` tags (b7f5be90) to confirm the fold reaches the restore vs falling back to the co-resident leaf.

### Interpretation guards
- **CONV_SNAP_FIX baseline shift:** main @1053c604 bakes `FR13_APC_CONV_SNAP_FIX=1`, which already moved 13453
  from 2-turn give-up to ~8-turn engagement. `m_tree_cache_base` here is **NOT** the historical 2-turn baseline
  — judge recompute/Path A against the improved ~8-turn base; **a small/null delta is expected, not "broken."**
- **Path A likely inert:** `REFOLD_APPLIED>0` proves the fold FIRES, not that it CHANGES the outcome (branch
  close-out found it inert because restored HIT boundaries were already faithful via prefill-capture). Confirm
  real effect via `RESTORE_USED` vs `FALLBACK`, not `REFOLD_APPLIED` alone.
- **NP=1 unproven:** gate `m_tree_recompute_np` behind a same-input NP=0-vs-NP=1 argmax/state parity check
  before trusting its result; it may fail cuda-graph capture or not be bit-exact.
- **Give-up premise shaky:** char-8 signal is n=1, single task (13453), low power. Per the bug-class playbook,
  **run B≥1 same-seed repeat first** and treat outcomes as directional.
- **Leak interaction:** all 3 cache arms exercise the main-baked EXACT_SEED path → **expect possible OOM137
  mid-run** (leak root unknown, no fix beyond main). Watch per-arm MemAvailable directly; FR13_LEAK_PROBE only
  counts pend_kvab/ckpt, NOT the `_FR13_REFOLD_*` maps (Path A).
- Standing rules: temp 0.6 only (never temp-0/greedy); live SWE-Verified task only (no static/replay gate);
  concurrency = 1; commit/push only when asked.

---

## 5. LEAK-TEST READING GUIDE (running main leak test)

VERIFY_LEAKTEST verdict: **holds=true** (leak signature genuine + monotonic) **but with 7 refutations** — do
NOT trust the monitor's DONE/OOM verdict; read authoritative signals and beware attribution.

### Exact greps to validate before trusting the verdict
1. **Do NOT trust the monitor's OOM/DONE line** — teardown `docker rm -f`'d the Exit(137) container, so
   `docker ps -a … | grep 'mainleak.*Exited (137)'` now returns nothing and the monitor falls through to a
   false "DONE, no OOM". Instead:
   - `cat output/fr13_leak_main/RUNINFO.txt` → expect **rc=143** (≠0).
   - `ls output/fr13_leak_main/mainleak/swe_out/verified/per_task/astropy__astropy-13453/{eval_report.json,runner_metadata.json}`
     → **both ABSENT ⇒ task did not complete ⇒ FAIL/leak.**
   - `grep -n 'gpu_oom_guard\|floor' output/fr13_leak_main/mainleak/launch.log`.
2. **Prove engagement was non-vacuous** (ES=0 trap):
   - `grep -c 'ES_SEED_APPLIED' …/logs/fr13_apc_exact_seed_eng.log` (>0; observed 3216).
   - `grep -c 'ES_GATE bs=1024' …/logs/fr13_apc_exact_seed_eng.log` (>0; observed 50).
   - `grep -x 'ATTENTION_BACKEND=FLASH_ATTN' …/container_env.txt`.
3. **Attribute the leak — don't assume "cache leaks":** FR13_LEAK_PROBE dict-size lines are **NOT present**
   under `output/fr13_leak_main/`. Grep the engine/boot docker log (`…/boot_log_snapshot.txt` + live container
   log) for the leak-probe dict-size output to prove the ES checkpoint dict is what grows (ES_WRITE=7152,
   hashes at pos 13K→37K).
4. **Run the paired cache-OFF control** (KIND `flash_ns5_nocache`, same task + boot era) — currently absent.
   Without it you cannot separate "exact-seed cache leaks" from "native-MTP-5 + a 37K-token multi-turn agent
   working set."
5. **Config parity for any main-vs-branch A/B:** `grep -E 'CONV_SNAP_FIX|REPLAY_ROUTE' <anchor>/container_env.txt`
   vs branch arm — 1053c604 bakes CONV_SNAP_FIX=1 and 60d7170c changed the REPLAY_ROUTE default; both must
   match or the A/B is confounded.
6. **Reproduce:** repeat the B=1 same-seed run ≥1× (playbook first gate) — one OOM is not a reproduced leak.
7. **Sanity-check trajectory:**
   `awk 'NR==1{f=$2}{if(m==""||$2<m)m=$2;l=$2}END{printf "first=%.1f min=%.1f last=%.1f\n",f/1048576,m/1048576,l/1048576}' output/fr13_leak_main/memavail.log`
   → confirm min (~8.8 GB) coincides with guard floor + death timestamp (recovery to 15 GB is **post-death**).

### Confounds (why a naive "cache leaks" read is unsafe)
- **Attribution not isolated:** no paired cache-OFF control; probe dict-size output not captured. The 66 GB
  drain (75→8.8 GB) could be native-MTP-5 + a growing 37K-token agent working set, not the cache. Restore is
  under-exercised (all 50 ES_GATE samples `row0_hit=False` ⇒ growth is write-side checkpoint accumulation).
- **Monitor false-DONE:** teardown races away the Exit(137) container; monitor also conflates script-DONE
  (rc=143 written regardless of OOM) with leak-absent.
- **Script asserts miss vacuity:** boot-time worker-env checks pass even for a vacuous ES=0 run; only the manual
  eng-log grep distinguishes it (here it engaged, 3216 applies).
- **Config drift:** anchors (≤4b68c8af) predate CONV_SNAP_FIX bake + REPLAY_ROUTE default change; anchor output
  dirs not present locally ⇒ parity unverified.
- **N=1:** single task / single boot / no same-seed repeat.

### What IS solid (LEAKTEST confirmations)
- Config correct + fail-loud asserts passed (FLASH_ATTN, EXACT_SEED=1, SNAP_FIX=1, CONV_SNAP_FIX=1,
  REPLAY_ROUTE=0, spec ns=5, LEAK_PROBE=1); OFFLOAD isolates GB10 MemAvailable as a clean GPU-pool sensor.
- Leak signature **genuine + monotonic**: 111.6 → 75 (model load) → monotonic drain to 8.8 GB over ~9 min on
  ONE task → guard floor 9000 MiB → Exit(137) → task incomplete → recovery only post-death.

---

## 6. OPEN QUESTIONS / RISKS

1. **Leak root cause UNKNOWN** — in window a76f2553..0d12cdbf, in-container, ~3.5 GB/min bleed + ~40 GB
   first-prefill step. FIX A/B/C all wrong; CONV_SNAP_FIX + ES checkpoint maps + faecc88d `.cpu()` all
   EXCLUDED. main (and therefore the picked build) is INSIDE the window and carries NO leak fix beyond main.
   → **Expect OOM137 mid-run on any cache arm; MemAvailable is the guardrail.**
2. **Is the leak the cache at all?** LEAKTEST cannot yet attribute it (no cache-OFF control, no probe dict-size
   capture). Could be native-MTP-5 + agent working set. Resolve before any "cache leaks" claim.
3. **Give-up premise is shaky** — char-8 is n=1, agent-behavior-confounded, single-task low power; the 2×2
   "cache-ON fails" collapsed to temp-0.6 sampling noise. Losslessness looks intact (6/6 cap-matrix arms
   resolved). Phase-4 is a **research probe** (does tree+cache stop giving up), not a ship gate.
4. **Baseline shift** — CONV_SNAP_FIX already moved 13453 2→~8 turns, so a null recompute/Path A delta is the
   expected prior, not evidence of breakage.
5. **NP=1 (node-parallel) GPU-UNTESTED** — cuda-graph capture + NP=0-vs-NP=1 bit-exactness unproven; gate
   behind a same-input parity check.
6. **Path A likely INERT** for the give-up (restored boundaries already faithful via prefill-capture); confirm
   with RESTORE_USED vs FALLBACK, not REFOLD_APPLIED.
7. **Non-vacuity risk (bug-class #9)** — recompute arm's flags must be proven live in the mp/spawn EngineCore
   worker's `/proc/<pid>/environ`, not just container_env.txt.
8. **Path A per-request GPU state** (`_FR13_REFOLD_TAIL/_CKPT/_SEEDED/_WROTE/_ABS`) — pruned at `_free_request`
   (patcher L7532-7541) and tail-ring trimmed <64 wide, but FR13_LEAK_PROBE does NOT track the REFOLD maps →
   watch MemAvailable directly on `m_tree_patha`.
9. **Env/scope** — the live give-up run needs the GPU + NVIDIA container (currently BUSY with the leak test)
   and alienware offload reachable; neither exercised in the pick audit. Deliverable here is the **trialed,
   audited, static build** (fr13-mainpick @ 08b629ef, unpushed) + this plan.
10. **Disclosed pick residual** — worker env-bridge log won't print a `CONV_SNAP_FIX=` field (d782983d omitted);
    the flag itself stays active via main's baked default. Cosmetic only.

**Recommendation (from PICKTREE):** PROCEED. Pick is clean, minimal, byte-exact to GPU-validated ground truth
(patcher==837236d0 minus the one non-functional log line; kernel==dd5578b4; launcher/serve_variant==e6d0214a);
both features default-OFF and inert with flags unset. Keep on branch `fr13-mainpick`; treat as a research probe,
not a ship candidate.
