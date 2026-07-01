# FR13 astropy b4 Longitudinal Solve-Rate History

> **⚠️ CORRECTION (2026-07-01): the "current run is IN-LINE, not a regression" verdict below is WRONG.**
> This survey never scanned the `fr9_*` dirs and missed `fr9_b4temp06_lowmem088_mtp5_s1` which resolved
> **8/16 on the identical astropy-16** (2026-06-02). The current pipeline's 1–2/16 IS a real ~6-task
> **char-8 regression**. The b4 CORE-4 analysis here is still valid (13033/13236/13398 are hard), but the
> full-16 discriminates. See **FR13_CHAR8_REGRESSION_FINDINGS.md** for the corrected picture.

Scope: astropy b4 CORE gate = **12907 / 13033 / 13236 / 13398**. Verdicts sourced from
`per_task/astropy__astropy-<id>/eval/eval_report.json` (`resolved` => passed). Config inferred
from run-dir names + `boot_log_snapshot.txt`. Cell legend: `R`=resolved, `T`=failed:tests_failed,
`P`=failed:patch_apply_failed, `-`=task absent from run, `?`=ungraded (no eval_report).

READ-ONLY analysis; live inference arm untouched.

---

## (1) Chronological table (sorted by date)

| Date | Run | Config (short) | b4 X/4 | 12907 | 13033 | 13236 | 13398 | 16-subset |
|------|-----|----------------|--------|-------|-------|-------|-------|-----------|
| 20260531 | fb2spine_*norepair_fix1..5 (5 runs) | fb 2-spine spec — BROKEN infra (patch never applied) | 0/4 | P | P | P | P | — |
| 20260531 | fb2spine_optionC_indep_seg * (5 graded runs) | fb 2-spine optionC — BROKEN infra | 0/4 | P | P | P | P | — |
| 20260531 | fb2spine_optionC_parentrefresh / waitingreq | fb 2-spine optionC — BROKEN infra | 0/4 | P | P | P | P | — |
| 20260531 | fb2spine_parentnative_gate1..16 (graded: 1,2,3,4,5,7,12-16) | fb 2-spine parentnative — BROKEN infra | 0/4 | P | P | P | P | — |
| 20260531 | fb2spine_* (draftfree, waitingpair, gate6/8/9/10/11) | UNGRADED (no eval dir) | ?/? | ? | ? | ? | ? | — |
| 20260612 | fr13_b1_gold_swe / b1_swe_gold task1 native_a/b, tree_a/b | E5-native mtp5 + TREE cat9 mtp9, cache OFF (dup seeds/mirror) | 1/1 | R | - | - | - | — |
| 20260612 | fr13_b1_gold_swe / b1_swe_gold task2 native+tree | E5-native mtp5 / TREE cat9, cache OFF | 0/1 | - | T | - | - | — |
| 20260612 | fr13_b1_fix1_confirm | APC off-vs-on stream compare — UNGRADED | ?/1 | ? | - | - | - | — |
| 20260615 | cat10_w600 | cat10 tree spec ON, 600s cap | 1/1 | R | - | - | - | — |
| 20260615 | cat6root_w600 | cat6 spine-root, 600s | 0/1 | P | - | - | - | — |
| 20260615 | cat6root_w600r2 | cat6 spine-root, 600s rerun | 1/1 | R | - | - | - | — |
| 20260615 | cat9_a / cat9_w600 / cat9_dev / opta_w600 | cat9 TREE_ATTN mtp9 spec ON, warmup arms | 1/1 (dev=0/1) | R/R/P/R | - | - | - | — |
| 20260615 | native_a | native vllm (no forked-fa2), spec ON | 1/1 | R | - | - | - | — |
| 20260616 | .stale/cat6root_b4 | cat6 spine-root tree, spec ON | 0/2 | P | - | P | - | — |
| 20260616 | .stale/cat9_b4 | cat9 TREE mtp9, pre-APC | 1/4 | R | T | T | P | — |
| 20260616 | .stale/nativeE5_b4 | native e5-tree, spec ON | 1/4 | R | T | P | T | — |
| 20260616 | cat9_b4_600s_screen | cat9 TREE mtp9, 600s screen | 1/4 | R | T | P | P | — |
| 20260616 | dm_device / dm_hostref | cat9 TREE mtp9, device-map variants | 1/4 | R | T/P | P | P | — |
| 20260616 | nativeE3/E4/E5_b4_600s_screen | native e3/e4/e5-tree, 600s screen | 1/4 | R | T | P | P | — |
| 20260616 | validate_clean / validate_contam | cat9 TREE mtp9, validation arms | 0-1/1 | P / R | - | - | - | — |
| 20260616 (snap) | .b4_deleted/nativeE5_b4 | E5-native mtp5, cache OFF | 1/4 | R | T | T | T | 4/16 (12907,14309,14365,14508,14539,14995 R) |
| 20260616 (snap) | .b4_deleted/cat9_b4 | TREE cat9 mtp9, cache OFF | 1/2 | R | - | T | - | partial |
| 20260617 | cat555_b1 / cat6root_b1 / cat9_b1 | cat55x / cat6 / cat9 tree, spec ON, b1 | 1/4 | R | T | T | T | — |
| 20260618 | cat55221_b1 | cat55221 tree variant, spec ON | 1/4 | R | P | P | P | — |
| 20260618 | nativeE3_b1 / nativeE5_b1 | native e3/e5-tree, spec ON, b1 | 1/4 | R | T | T | P | — |
| 20260619 | cat9_apc / cat9_apc_fix / cat9_apc_snap | cat9 TREE mtp9, **cache ON (APC)** | 1/4 | R | P | P | P | — |
| 20260620 | cat9_apc_12907* (clean, 45477, wt) | cat9 TREE, cache ON, 12907 probes | 0/1 | P | - | - | - | — |
| 20260620 | cat9_apc_* (ssm/wt/leaf/ov/tap/uni diag ×~18) | cat9 TREE, cache ON, 13033 diagnostics | 0/1 | - | P | - | - | — |
| 20260620 | cat9_apc_wt_deploy | cat9 TREE, cache ON, 13236 | 0/1 | - | - | P | - | — |
| 20260620 | **cat9_gate_apcON** | cat9 TREE mtp9, **cache ON**, full b4 gate | **0/4** | P | P | P | P | — |
| 20260621 | cat6root_apcoff_b1 | cat6 spine-root, **cache OFF** | 1/1 | R | - | - | - | — |
| 20260621 | nativeapc_spine | native spine, **cache ON** | 1/1 | R | - | - | - | — |
| 20260621 | shipgate/cat6root_apcon_b1 | cat6root, **cache ON** | 0/2 | P | T | - | - | — |
| 20260622 | shipgate/cat6root_apcon_b1 | cat6root, cache ON | 0/1 | P | - | - | - | — |
| 20260622 | shipgate_plainalign/apcoff_b1 | cat6root, **cache OFF** | 1/2 | R | T | - | - | — |
| 20260622 | shipgate_plainalign/plainalign_b1 | cat6root, cache ON (plainalign) | 0/4 | P | P | P | P | — |
| 20260624 | rategate/rg_OFF_r1,r2 | cat6, **cache OFF** | 1/1 | R | - | - | - | — |
| 20260624 | rategate/rg_ON_r1,r2 | cat6, **cache ON** | 0/1 | P | - | - | - | — |
| 20260625 | rategate/rg_CFG*,OFF* (6 runs) | cat6, **cache OFF** (cfg variants) | 1/1 | R | - | - | - | — |
| 20260625 | rategate/rg_ON_r1,r2,r3 | cat6, **cache ON** | 0/1 | P | - | - | - | — |
| 20260626 | shipgate/apcoff_b1 (×3) | cat6root, **cache OFF** | 1/1 | R | - | - | - | — |
| 20260626 | shipgate/apcon_b1 (×5) | cat6root, **cache ON** | 0/1 | P | - | - | - | — |
| 20260626 | bigN_solve/cacheoff_r1,r2,r3 | cat6, **cache OFF** | 1/1 | R | - | - | - | — |
| 20260626 | bigN_solve/tree_r1..r3 (2 runs) | cat6, **cache ON** | 0/1 | P | - | - | - | — |
| 20260627 | matched_control/matched_r1 | cat6, **cache OFF** | 1/1 | R | - | - | - | — |
| 20260627 | nospec_cache/nospecoff/nospecon (×6, 2 dates) | **FLASH_ATTN no-spec**, cache OFF & ON | 1/1 | R | - | - | - | — |
| 20260627 | apc_hrs0_swe/hrs0_r1,r2,r3 | cat6, cache ON, HRS=0 hitcap 1e6 | 0/1 | P | - | - | - | — |
| 20260627 | apc_spine_hrs0 / spine_cache_engaged / bigN spine,tree | spine e5 & cat6, cache ON, HRS=0/1 | 0/1 | P | - | - | - | — |
| 20260627 | zeroaccept/za (034122=0/1, 040652=1/1 then 0/1) | cat6, cache ON, zeroaccept probe | 0-1/1 | P/R/P | - | - | - | — |
| 20260628 | apc_e2e_gate/194610 eg_ON | cat6, cache ON, SNAP_FIX+ZEROACCEPT | 0/1 | P | - | - | - | — |
| 20260628 | apc_e2e_gate/220251 eg_ON r1,r2 | cat6, cache ON | 0-1/1 | R/P | - | - | - | — |
| 20260628 | apc_stale_or_not/* (fresh/stale/shadow ×9) | cat6 spine-tree, cache ON | 0/1 | P | - | - | - | — |
| 20260629 | apc_e2e_gate/203425 eg_OFF/ON | cat6, cache OFF & ON, HRS=0 hitcap64 | 1/1 | R | - | - | - | — |
| 20260629 | apc_e2e_gate/230614 eg_ON | cat6, cache ON, HRS=0 hitcap64 | 1/1 | R | - | - | - | — |
| 20260629 | apc_e2e_gate/234136 eg_OFF r1,r2 / eg_ON r1,r2 | cat6, cache OFF & ON, HRS=0 hitcap64 | 1/1 | R | - | - | - | — |
| 20260630 | arm_cat6_OFF | cat6 spine, **cache OFF** (EXACT_SEED) | 1/1 | R | - | - | - | — |
| 20260630 | arm_e5_ON / sg_cat6on_clean | e5 & cat6, **cache ON** (EXACT_SEED) | 0/1 | P | - | - | - | — |
| 20260630 | speedgate/040226 sg_cat6root_ON | cat6root, cache ON, EXACT_SEED HRS=0 hitcap64 | 1/2 | R | T | - | - | — |
| 20260630 | speedgate/051031 sg_cat6root_ON | cat6root, cache ON, EXACT_SEED HRS=0 | 1/1 | R | - | - | - | — |
| 20260630 | speedgate/064631 sg_cat10_ON | cat10, cache ON, EXACT_SEED HRS=0 | 1/4 | R | P | P | P | — |
| 20260630 | speedgate/064631 sg_cat6root_ON | cat6root, cache ON, EXACT_SEED HRS=0 | 1/4 | R | P | T | P | — |
| 20260630 | tree_cache_matrix/185234 m_e5_OFF | e5 spine, **cache OFF** EXACT_SEED=0 | 0/1 | T | - | - | - | — |
| 20260630 | tree_cache_matrix/185234 m_e5_ON | e5 spine, cache ON EXACT_SEED=1 HRS=0 | 1/1 | R | - | - | - | — |
| 20260630 | tree_cache_matrix/224939 m_cat6_ON | cat6, cache ON EXACT_SEED=1 HRS=0 | 1/1 | R | - | - | - | — |
| 20260701 | tree_cache_matrix/001109 m_cat6_OFF/ON, cat8_OFF/ON, e5_OFF/ON | cat6/cat8/e5, cache OFF & ON, EXACT_SEED HRS=0 hitcap64 | 1/1 (all 6) | R | - | - | - | — |
| 20260701 | tree_cache_matrix/051923 m_e5_ON | e5 spine, cache ON EXACT_SEED HRS=0 | 0/2 | P | T | - | - | — |
| 20260701 | tree_cache_matrix/060531 m_e5_ON | e5 spine, cache ON EXACT_SEED HRS=0 | 1/4 | R | P | T | P | — |
| 20260701 | tree_cache_matrix/063919 m_e5_ON | e5 spine, cache ON EXACT_SEED HRS=0 | 1/4 | R | P | T | P | — |
| **20260701** | **072605 m_cat8_OFF (CURRENT)** | **cat8, cache OFF EXACT_SEED=0 hitcap64; 16-subset partial (9)** | **1/4** | **R** | P | T | P | **1/9** (14096 R) |
| **20260701** | **072605 m_e5_ON (CURRENT)** | **e5 spine, cache ON EXACT_SEED HRS=0 hitcap64; 16-subset full** | **0/4** | **P** | T | P | P | **1/16** (14309 R) |

Note: many single-task probe/diagnostic dirs (esp. the 20260620 cat9_apc SSM/leaf/wt/overlap diag family, ~18 dirs) only ran one non-12907 task and are collapsed above.

---

## (2) 12907 (the char-8-prone task) across time

Restricting to runs where **12907 was actually attempted and GRADED** (excludes absent, ungraded,
and the 20260531 fb2spine BROKEN-infra block which is an INFRA `patch_apply_failed`, not a model signal):

- **cache OFF / no-cache-marker (spec ON) arms:** 12907 resolves overwhelmingly. Every gold/native/cat6-OFF/rategate-OFF/bigN-cacheoff/matched-control/shipgate-apcoff run = **R**. A handful of OFF-side flakes exist (cat6root_w600 P, validate_clean P, tree_cache_matrix/185234 m_e5_OFF **T**), i.e. even cache-OFF 12907 is not a hard 4/4 — it bounces.
- **cache ON (APC) arms:** 12907 is the coin-flip. Pre-EXACT_SEED APC (20260619-20260628) 12907 was almost always **P** (rategate ON 0/5, shipgate ON 0/8, bigN tree 0/5, hrs0 0/3, stale 0/9). Post-EXACT_SEED APC (20260629-0701) it flips to mostly **R** (e2e_gate 6/6 R, tree_cache_matrix 001109 all R, speedgate R) but STILL flakes back to **P** (arm_e5_ON, sg_cat6on_clean, tree_cache_matrix 051923 & the CURRENT 072605 m_e5_ON).
- **no-spec (FLASH_ATTN) arms:** 12907 = **R** in all 6 (cache ON and OFF alike).

Tally (graded, non-infra attempts of 12907): roughly **~40 R vs ~40 P/T** — it is genuinely ~50/50 and the failure mode is `patch_apply_failed`, matching the known char-8 tool-call derail signature. It resolves and fails under the SAME nominal config on different seeds/runs (e.g. zeroaccept 040652 r1=R r2=P; e2e_gate 220251 r1=R r2=P).

---

## (3) Best-performing config for b4

No config has ever cleared better than **1/4** on the full 4-task b4 core (13033/13236/13398 have
NEVER co-resolved with 12907 in any single graded run — they fail `tests_failed` or `patch_apply_failed`
everywhere). So "best b4" is really "best at the achievable ceiling of 1/4, i.e. reliably landing 12907":

- **Best full-b4 config: cache-OFF spec-ON tree (native-E5 / cat9 / cat6), pre/post-EXACT_SEED = 1/4**, driven entirely by 12907. This is the modal outcome across 20260616-18 and the OFF side of every A/B.
- **Best 1-task 12907 reliability: EXACT_SEED cache-ON HRS=0 hitcap64 (e2e_gate 20260629 + tree_cache_matrix 20260701T001109)** — went 12907=R across ALL cache-ON and cache-OFF arms in those batches (cat6/cat8/e5), the only regime where cache-ON matched cache-OFF on 12907.
- **no-spec FLASH_ATTN** is the only regime with zero 12907 failures observed (6/6 R), but it disables the whole spec-decode subject-under-test.
- 16-subset ceiling: the historical best is **.b4_deleted/nativeE5_b4 = 4/16** (cache OFF, native mtp5) — 12907,14309,14365,14508,14539,14995 resolved (6 R over 16 actually, listed 4/4 core). That is the high-water full-subset mark.

**Bottom line: cache is not the lever.** The best achievable b4 is 1/4 regardless of cache, gated by
13033/13236/13398 being structurally unsolvable and 12907 being a flaky char-8 coin-flip.

---

## (4) Where the CURRENT run sits vs history

Current run `run_20260701T072605Z`:
- `m_cat8_OFF` = **1/4** core (12907 R, +14096 R in subset) → **1/9 on the partial subset**. This is EXACTLY the modal/best historical b4 (1/4, 12907-carried). **In-line, not a regression.**
- `m_e5_ON` = **0/4** core (12907 **P**), **1/16 subset** (14309 R). The 0/4 is driven solely by 12907 flaking to `patch_apply_failed` this seed.

Is 0/4 / 1/16 anomalously low? **No — it is normal variance:**
- 12907=P under cache-ON EXACT_SEED e5 has direct precedent 3 days running: arm_e5_ON (0630), sg_cat6on_clean (0630), tree_cache_matrix/051923 (0701) — all the same P flake.
- The immediately preceding e5_ON runs the SAME morning (060531, 063919) both landed 12907=R → 1/4. So 072605 m_e5_ON=0/4 is the DOWN side of a config that oscillates 0/4 ↔ 1/4 on consecutive runs. Classic seed variance, not a new failure.
- 1/16 on the full subset is consistent with the only other full-16 cache-ON e5 data point (this run) and below only the cache-OFF native mtp5 4/16 high-water — expected, since cache-ON e5 lost 12907 to the char-8 flake this seed while OFF-native kept it.

---

## (5) Does history support "char-8 is the dominant, cache-independent, cap-independent blocker"?

**Yes, strongly.**
- **Cache-independent:** 12907 resolves AND fails on both cache OFF (185234 m_e5_OFF T; validate_clean P vs the many OFF R) and cache ON (post-EXACT_SEED R vs arm_e5_ON/051923/current P). The OFF↔ON gap collapses entirely under EXACT_SEED (20260629 + 001109 batches: OFF and ON both R). The pre-EXACT_SEED ON=all-P vs OFF=all-R gap was an APC-lossiness artifact that EXACT_SEED closed — after which the residual failures are the cache-independent char-8 flake.
- **Cap-independent:** w600 (600s cap) arms flake both ways (cat6root_w600 P, w600r2 R, cat10_w600 R); the cap-500 current run and cap-1e6/64 hitcap variants all land in the same 12907 coin-flip band.
- **Spec-adjacent but not spec-caused:** the only regime that never fails 12907 is no-spec FLASH_ATTN — consistent with char-8 being a spec-decode/tool-call derail, not model incapacity.
- **Dominant blocker:** b4 solve rate bounces in the narrow 0/4 ↔ 1/4 band entirely on 12907's toggle, while 13033/13236/13398 stay pinned failed regardless of config. The whole visible b4 signal IS the char-8 12907 flake.

---

## VERDICT

The CURRENT run `run_20260701T072605Z` is **IN-LINE, not a regression.** `m_cat8_OFF=1/4`
is the modal best-case historical b4; `m_e5_ON=0/4 / 1/16` is the normal DOWN-swing of the
cache-ON e5 config, which oscillates 0/4↔1/4 on consecutive same-morning seeds (060531/063919=R,
051923/072605=P). Historically **no config ever beats 1/4** on the b4 core — 13033/13236/13398 are
structurally unsolvable across every cache/spec/cap setting, and the entire b4 signal is the
char-8-prone **12907** flipping ~50/50 on `patch_apply_failed`. Cache is NOT the lever: post-EXACT_SEED
the ON vs OFF gap disappears (both landed 12907=R in the 20260629 + 20260701T001109 batches). The
data therefore **confirms the standing finding**: char-8 is the dominant, cache-independent,
cap-independent blocker, and low b4 on any single run is seed variance on 12907, not a config regression.
Best reliable config = **EXACT_SEED cache-ON HRS=0 hitcap64** (only regime where cache-ON matches
cache-OFF on 12907); best full-subset high-water = cache-OFF native-mtp5 (4/16).
