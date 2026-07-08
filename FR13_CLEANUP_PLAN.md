# FR13_CLEANUP_PLAN.md — Dead-Flag Surgical Cleanup (EXACT_SEED / HRS / BLOCK_REFOLD / REFOLD_TO_SNAPSHOT)

## 0. Scope & the one fact that makes this safe

Four flag families are dead and **already force-off at gdn import** — the emitted loop at
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:1037-1038`:

```
for _fr13_dep_k in ("FR13_APC_EXACT_SEED", "FR13_APC_HIT_RECURRENT_SUFFIX",
                    "FR13_APC_BLOCK_REFOLD", "FR13_APC_REFOLD_TO_SNAPSHOT"):
    os.environ[_fr13_dep_k] = "0"
```

Because every request-time gate reads these env vars **after** this loop has zeroed them,
**every removal in this plan is behavior-preserving** — no numerical change, no gate flips.

### Two-level architecture (why a mis-cut boot-crashes)
The patcher edits in-container vLLM source via `PATH.read_text()` → Python `str.replace()` →
`PATH.write_text()`. So each dead flag lives at **two levels**:
- **L1 — patcher control-flow**: the Python that decides what string to emit / which dispatch entry to run.
- **L2 — emitted vLLM source**: code *inside* triple-quoted heredocs / `"...\n"` concat literals that
  references the flag at request time.

Removing a flag = cutting **both** levels while leaving the emitted template a **valid parse**. The
**only** guard is `py_compile.compile(path, doraise=True)` at ~L20776, which runs *after* write — an
orphaned `if:`/`try:`/`else:` or a broken implicit-concat = worker boot crash.

### The force-off loop is removed LAST, not first
`scripts/fr10_phase4_patch_vllm_tree_gdn.py:1037-1038` (+ its DEPRECATION comment 1029-1036) neutralizes
**all four** flags. It must **stay** until every guarded impl of all four families is gone. Removing it
early would re-arm any stray env export against still-present dead code. It is the **final** edit.

---

## 1. KEEP LIST — DO NOT TOUCH (stated loudly)

These are the working, baked, shipped machinery. Any edit that grazes them is a regression, not a cleanup.

| KEEP | Where | Why |
|---|---|---|
| **FR13_APC_SNAP_FIX** + **SNAP_FIX_ZEROACCEPT** | `_patch_worker_mamba_snap_fidelity` (~14372-14925); launcher `-e` 517-518 | Working baked node-bank/SSM snapshot fix. Its `get_temporal_copy_spec` override resolves to `state[leaf].data_ptr()` (~14814-14816). The dead ES/REFOLD branches are **fused into it** — preserve the whole function. |
| **FR13_APC_CONV_FIX / CONV_SNAPSHOT / CONV_SNAP_FIX / PRE_SNAP_FIX / CONV_LEAF_COMPLETE** | launcher 308-313, 335; `-e` 519-520, 527 | Baked conv node-bank fix. |
| **FR13_ENABLE_APC** | launcher master switch | Gates the whole APC branch. |
| **Stateless-tree trio**: COMMIT_TO_RUNNING_ROW / TREE_RUNROW_INIT / BURN_NODE_BANK | committer args ~10875-10877 / 11604-11606; `h0_use_accepted_column` ~5367-5371; `_fr13_conv_commit_to_col0` ~8991; all-together assert ~10565-10578 / 11485-11498 | Current stateless-tree design; not a dead flag. |
| **FR13_TREE_GDN_SLOT_PIN** | launcher `-e` **524** (interleaved between EXACT_SEED@523 and BLOCK_REFOLD@525) | Verified, default-off, NOT dead. Easy to eat by mistake. |
| **The force-off loop** 1037-1038 + comment 1029-1036 | gdn import preamble | Neutralizes all four families; remove LAST. |
| **`_fr13_write_apc_env_sidecar`** 19630-19669 + its `keys` list 19646-19649 | patcher | Forwards SNAP_FIX/CONV_FIX/etc. `keys` list does NOT contain HRS/REFOLD — no functional removal here; docstring prose 19634 is optional cosmetic only. |
| **`_patch_scheduler_fr13_freereq_cleanup`** leak-fix (~7874+) | patcher | KEEP the function. Its pop-list names some `_FR13_ES_*`/REFOLD dicts but via `getattr(...,None)`+`isinstance` guards — see edit S-FREE for the surgical entry drops. |
| **`_FR13_APC_SSM_CHUNKED_PTR_BY_REQ`** map + its EXACT_SEED legacy readers (6782, 6829, 14632-14641, 14794) | patcher | Written only by refold (removed) but READ by ES legacy slot arm — map stays (lazily created), just empty. |
| **`# FR13_APC_ALIGN_TREE_AWARE`** idempotency sentinel | postprocess inject | Re-apply guard; must survive so re-patch stays a no-op. |
| **fr13_bigdenom_swe_serve_variant.sh** (whole file) | scripts | EXACT_SEED=0 shares the XFLAGS line with KEEP `FR13_ENABLE_APC=1` (L114/136), names the `nativemtp5_exseed` arm, and the assert L292-293 **enforces** the =0 deprecation. Leave whole. |
| **8 live seq/launch scripts** exporting `FR13_APC_EXACT_SEED=0` next to stateless-trio config | fr13_stateless_4arm_nowall_seq, _cachefirst_seq, _speed_seq, remaining3_seq, cat6_cache_seq, b1_cache_seq, native_cache_seq, native_nocache_b4_seq | Redundant `=0` matches force-off; multi-site + entangled with keep-config. Leave. |

**EXACT_SEED note**: `FR13_APC_EXACT_SEED` machinery is in this pass, but its restore/capture blocks **host** HRS and REFOLD sub-blocks. That is why the entangled HRS/REFOLD cuts are folded into the EXACT_SEED pass (Phase 5) rather than done standalone.

---

## 2. WHOLE SCRIPTS TO DELETE (`git rm`) — pure dead-flag diagnostics, zero live callers

| # | File | Subject |
|---|---|---|
| 1 | `scripts/fr13_apc_exactseed_statediff.sh` | EXACT_SEED state-diff harness (boots EXACT_SEED=1, greps ES_* markers). |
| 2 | `scripts/fr13_apc_greedy_divergence.sh` | HRS 0-vs-1 greedy first-divergence (also greedy temp=0, itself banned). |
| 3 | `scripts/fr13_apc_hrs0_swe_test.sh` | HRS=0 subtractive SWE, cat6root. |
| 4 | `scripts/fr13_apc_spine_hrs0_test.sh` | HRS=0 subtractive SWE, chain5 spine. |
| 5 | `scripts/fr13_apc_remeasure_ab.sh` | HRS 0-vs-1 residual A/B. |

Repo-wide caller check: **no live executable** invokes any of them; the only refs are doc prose
(`FR13_APC_EXACT_SEED_PIVOT_CHECKPOINT.md:37`, `FR13_TREE_CACHE_LOSSY_MECHANISM.md:12`), comments
inside the delete-set itself, and the already-staged `research/fr13_workflows/k2_cleanup.diff` /
`fr13_rename_codex.patch`. **Cross-check that staged diff before `git rm`** to avoid a double-removal
conflict. Doc pointers go stale but are non-blocking — annotate or fix later.

---

## 3. SCRIPTS-LEVEL FLAG-LINE EDITS (keep the file, delete the lines)

Target: `scripts/fr13_launch_forked_fa2_tree_server.sh`. Each dead flag appears at **three** sites that
must be cut together. All anchored by the unambiguous `DEPRECATED 2026-07-07` comment tags.

| ID | Site | Action | Risk / hazard |
|---|---|---|---|
| S-SETUP | setup block (the `FR13_APC_HIT_RECURRENT_SUFFIX=0` … `FR13_APC_REFOLD_TO_SNAPSHOT=0` DEPRECATED lines + HRS/EXACT_SEED history comments + `FR13_APC_HIT_SUFFIX_CAP:=64`) | delete the contiguous dead-flag block | **MED** — bracketed by KEEP `PRE_SNAP_FIX:=0` above and `CONV_LEAF_COMPLETE` comment below. Do NOT eat the CONV/SNAP_FIX group (308-313) or the CONV_LEAF_COMPLETE/ZERO_MAMBA/COPY_SRC/FREE_TREE group. |
| S-EXPORT | the `export FR13_APC_... ` name-list (the line beginning `export FR13_APC_CONV_FIX ...`) | drop the 5 contiguous dead names: `FR13_APC_HIT_RECURRENT_SUFFIX FR13_APC_HIT_SUFFIX_CAP FR13_APC_EXACT_SEED FR13_APC_BLOCK_REFOLD FR13_APC_REFOLD_TO_SNAPSHOT` | **LOW** — keep the 10 surviving KEEP names on the same line. Must be dropped **with** S-SETUP or bash exports unset vars (harmless but stale). |
| S-DOCKER | docker `-e` forwards **521, 522, 523, 525, 526** | delete those 5 lines | **MED** — **line 524 `-e FR13_TREE_GDN_SLOT_PIN` is INTERLEAVED** between EXACT_SEED(523) and BLOCK_REFOLD(525) — PRESERVE it. Keep SNAP_FIX group (517-520) above and CONV_LEAF_COMPLETE (527) below. |

`FR13_APC_HIT_SUFFIX_CAP` is the HRS companion (inert while HRS=0), not in the patcher force-off list but
defaults to 64 both places — removing its forward is still behavior-preserving; treat as coupled-to-HRS.

There is **no `es_ckpt` code** in any `.sh` (only prose/asserts) — nothing to cut there beyond the EXACT_SEED forwards above.

---

## 4. PATCHER EDITS — `scripts/fr10_phase4_patch_vllm_tree_gdn.py`

> Real footprint ≫ the grep-hit counts (HRS "~9", refold "~159", exact_seed "~585" are grep hits, not lines).
> `_fr13_pathA_refold` alone is ~553 lines (8437-8989); the HRS prefill if-branch is ~168 lines (6034-6201);
> the ES capture tail is ~789 lines (6616-7404). Map **whole units**, not the estimate.

### 4A. HRS — `FR13_APC_HIT_RECURRENT_SUFFIX` (clean pieces here; the big if-branch is deferred to 4C/Phase 5)

| ID | Lines | Level | Action | Risk |
|---|---|---|---|---|
| H1 | 1760 (`_fr13apc_hrs = ...`), 1762 print fragment, 1766 marker-write fragment | emitted (`mab_helper` heredoc) | delete the `_fr13apc_hrs` var + the ` + " HIT_RECURRENT_SUFFIX=" + _fr13apc_hrs` fragments from print() and write(); keep the concat otherwise valid | LOW — KEEP `_fr13apc_es`(1761)+SNAP_FIX/ZEROACCEPT/CAP/CONV fragments (engagement proof). |
| H2 | 6027 `_fr13_es_on=...`; 6028-6033 `_fr13_apc_active=bool(not _fr13_es_on and ...HIT_RECURRENT_SUFFIX=="1"...)`; **if-body 6034-6201** | emitted (`prefill_scan_replacement`) | **DEFER to Phase 5** — this if/else is shared with the EXACT_SEED RESTORE(c) else (6202+) which wraps the LIVE `chunk_gated_delta_rule`. Collapse `if <HRS> else <ES restore>` → dedented `<ES restore>` only; `_fr13_es_on` dies with it. | **HIGH** — mis-dedent = boot crash. |
| H3 | 19634 docstring prose | patcher (non-load-bearing) | optional cosmetic; `keys` list unaffected | NONE |

### 4B. REFOLD — `FR13_APC_BLOCK_REFOLD` + `FR13_APC_REFOLD_TO_SNAPSHOT`

| ID | Lines | Level | Action | Risk |
|---|---|---|---|---|
| R-ALLOC | 479-514, 527-532 (stacked); 591-614 (per-layer) | emitted (gdn metadata-builder) | delete PATH-A g/beta alloc + FR13_REFOLD_ALLOC print + per-row g_step/beta_step wiring + the `if BLOCK_REFOLD:` twin + its `else: ...=None` | MED — nested in EAGER_PACK ring alloc; KEEP `_fr13_ep_ring_*`, `_FR13_EAGER_PACK_STACKS`, `_fr13_replay_*`. |
| R-E1 | 5325-5350 | emitted (tree replay-ring copy) | delete the `if globals().get("_FR13_REFOLD_ON",False) and ...g_step...:` g_tree/beta_tree stash | LOW — KEEP the 4 `_fr13_replay_ring_{k,v,a,b}.copy_` above, `launch_tree_gdn_prepared` below, and `FR13_TREE_RUNROW_INIT` h0 arg 5367-5371. |
| R-PATHA | **8437-8989** (whole `def _fr13_pathA_refold`) | emitted (inside `helper = r'''...'''` @8029-11691) | delete the entire function | MED — writes `_FR13_ES_PENDING_BY_REQ`/calls `_fr13_es_try_bind`/publishes `_FR13_APC_SSM_CHUNKED_PTR_BY_REQ` (all shared, all getattr-guarded readers). KEEP `_fr13_conv_commit_to_col0`(8991) below + the `r'''`/`'''` fences. |
| R-CALL | 10887-10913 (greedy), 11614-11633 (canonical) | emitted (`helper`) | delete both `if getattr(..., "_FR13_REFOLD_ON", False):` fold loops | LOW — KEEP `_fr13_publish_apc_ssm_leaf` above and `_fr13_flags[0].fill_(0)` / `_fr13_boundary_replay_post` below. |
| R-PUBGATE | 10687-10691 | emitted (`helper`, greedy committer) | drop the `or ...BLOCK_REFOLD...=="1"` OR-term + comment from `_fr13_apc_publish_on` | LOW — KEEP SNAP_FIX(10682) + EXACT_SEED(10686) terms + closing `)`. |
| R-STATEP | 7165-7255 | emitted (`prefill_scan_replacement`, ES capture branch) | delete the `if ...REFOLD_TO_SNAPSHOT...=="1":` state@P fold-and-stash | MED — inside ES per-row capture loop; keep `_fr13_es_end_abs`(7164) above and `_fr13_es_first_bnd`(7256-7259) below. |
| R-FREE | 7902-7903, 7905, 7911-7915 | patcher list-literal → scheduler.py | drop `_FR13_REFOLD_TAIL/CKPT/SEEDED/WROTE`, `_FR13_APC_SSM_RUNNING_POS_BY_REQ`, and the whole `for _fr13_tk_nm in ('_FR13_REFOLD_ABS','_FR13_REFOLD_PUB_OK'):` loop | LOW-MED — KEEP `_FR13_APC_SSM_LEAF/CONV_LEAF/ALIGNED_POS_BY_REQ`, `_FR13_ES_*`, `_FR13_APC_SSM_CHUNKED_PTR_BY_REQ`(7904, ES still reads). |
| R-REWIRE | 14580-14700 (specifically drop REFOLD arm 14599-14628 + `else:` wrapper 14629, keep legacy 14632-14641; drop `refold_pub_miss` 14696) | emitted (SNAP_FIX `get_temporal_copy_spec` override) | **DEFER to Phase 5** — collapse `if REFOLD_TO_SNAPSHOT: <rank-1> else: <legacy slot read>` → dedented legacy only | **HIGH** — inside KEEP SNAP_FIX + fused with ES chunked-ptr. KEEP `_fr13_fx_slotkey/cm`(14591-97), GAP-2 guard(14650-14700), LEAF_CROSSCHECK(14701-15). |
| R-VALIDATE | 14759-14793 | emitted (SNAP_FIX override) | delete the `if _fr13_fx_chunked_state is not None:` FR13_REFOLD_VALIDATE diagnostic | LOW-MED — KEEP redirect_engaged/used bumps(14756-58) and the chunked-ptr APPLY(14794). |
| R-ABSPUB | 14956-14981 | patcher `inject = (...)` concat | drop the `+ "if ...REFOLD_TO_SNAPSHOT..."` publisher; leave `inject = (anchor + "# FR13_APC_ALIGN_TREE_AWARE\n")` | LOW — KEEP anchor + `# FR13_APC_ALIGN_TREE_AWARE` sentinel + `text.replace(anchor, inject, 1)`(14983). |
| R-INTERLEAVE | 6218-6224, 6281-6310, 6337-6373, 6374-6439 | emitted (`prefill_scan_replacement`, inside ES restore else) | **DEFER to Phase 5** — remove the `or globals().get("_FR13_REFOLD_ON",False)` gate term + 3 REFOLD diagnostic/seed sub-blocks; keep the ES restore scaffolding + balanced parens | MED — sits INSIDE the ES restore loop; KEEP `_fr13_es_rmap/nsr2/st2/segbase`. |
| R-GLOB | **1048-1068** (`_FR13_REFOLD_TAIL/CKPT/ABS/SEEDED/WROTE={}` + `_FR13_REFOLD_ON=...` + comment) | emitted (import preamble) | delete — **LAST, after all readers gone** | MED — `_FR13_REFOLD_ON` read at 5339, 10896, 11618, 6224/6282/6338/6384, 8475. Remove readers first or NameError at gdn import. |
| R-TUPLE | 1037 tuple entries `"FR13_APC_BLOCK_REFOLD"`, `"FR13_APC_REFOLD_TO_SNAPSHOT"` | emitted force-off loop | drop these 2 entries — **only after 4A+4B+4C all done** | LOW — see §5 ordering. |

### 4C. EXACT_SEED — `FR13_APC_EXACT_SEED` (the riskiest family; done last)

**Clean wholesale removals (do these first within the ES pass):**

| ID | Lines | Level | Action | Risk |
|---|---|---|---|---|
| E-FN1 | def 19671-19863 + dispatch 20750 `(BLOCK_POOL_PATH, _patch_block_pool_exact_seed()),` | patcher fn + callsite | delete both | none — self-contained; getattr-guarded reach. KEEP `_fr13_write_apc_env_sidecar`(ends 19669) + fp8 dispatch 20745. |
| E-FN2 | def 19866-19997 + dispatch 20751 | patcher fn + callsite | delete both | none |
| E-FN3 | def 19999-20210 + dispatch 20752 + the 4-line ES dispatch comment 20746-20749 | patcher fn + callsite | delete all | none — KEEP other MAMBA_UTILS_PATH steps + ZERO_MAMBA at 20753+. |
| E-PATH | header comment 65-72 + `BLOCK_POOL_PATH`(73-75) + `KV_CACHE_MANAGER_PATH`(76-78) + blank 79 | patcher constants | delete | LOW — grep-confirmed referenced ONLY by E-FN1/2. Remove ONLY with E-FN1/2/3. KEEP `FP8_UTILS_PATH`(61-64), `QWEN3_5_PATH`(58-60), `SINGLE_TYPE_MANAGER_PATH`. |
| E-CTXLENS | 845-878 (`text.replace(` context-lens stash) | mixed (patcher `.replace` unit) | delete whole replace unit | none — KEEP preceding replace ending 844 and `FR13_TREE_GDN_SLOT_PIN` block at 879. |
| E-CAP | 6616-7404 (comment 6616-6652 + `if ...EXACT_SEED...=="1":` capture tail through 7404) | emitted (`prefill_scan_replacement`) | delete complete line units, keep 6615 + 7405 (`'''`) | **HIGH-mechanical** — 789-line if-block; last thing in the string, live neighbors above/below. (Note R-STATEP 7165-7255 lives inside this — if E-CAP deletes wholesale, R-STATEP is subsumed.) |
| E-GAPA | 13200-13288 (`es_sentinel`/`es_anchor`/`es_inject` unit) | mixed (patcher `.replace`) | delete whole unit | none — KEEP prior `text.replace(anchor,inject,1)`(13198) + `write_text`(13290). |
| E-GAP2 | 13998-14024 (`if ...EXACT_SEED...=="1":` SSM_ALIGNED_POS stash) | emitted (postprocess_new) | delete whole if-block | LOW — KEEP accept_token_bias(13991-97) above + `src_block_idx=...`(14025) below. Consumer is dead-preserved SNAP_FIX redirect (getattr-guarded). |

**Surgical / boolean edits:**

| ID | Lines | Action | Risk |
|---|---|---|---|
| E-PUBDISJ | 10688-10690 | drop the `or __import__('os').environ.get("FR13_APC_EXACT_SEED",...)=="1"` disjunct (+comment 10685-87) from `_fr13_apc_publish_on` | LOW — redundant (SNAP_FIX baked '1'). KEEP SNAP_FIX(10682-83)+REFOLD(handled by R-PUBGATE). |
| E-NONSPEC | 13038-13073 | **OPTIONAL** — `_LUMO_FA_NONSPEC_ROW_REQ_IDS` publish + partition assert; value-inert once ES gone. Interleaved with live reqkey publish → **leave by default** | LOW/OPTIONAL |

**Doc/comment-only (optional cosmetic):** APC env-bridge print of EXACT_SEED (1761-62, leave — engagement proof); OBS comment 1311 (leave); stale `vestigial EXACT_SEED committer call REMOVED` comments 10914/11634 (optional).

**Entangled collapses — the Phase-5 core (do WITH the ES pass, never standalone):**

| ID | Lines | Action | Risk |
|---|---|---|---|
| E-ESON | 6027 + clause 6029 | delete `_fr13_es_on` and the `not _fr13_es_on\n and` clause; this is the HRS side of H2 | MED — resolved together with H2 collapse. |
| E-RESTORE | 6202-6468 (ES restore else) | **PRESERVE the else-wrapper (it nests the LIVE `chunk_gated_delta_rule` 6469-6484)**. Only remove the EXACT_SEED disjunct at 6217 + ES log lines 6451-6466 + the R-INTERLEAVE REFOLD sub-blocks. If REFOLD is co-removed (this plan), collapse the whole HRS-if/ES-else to the live-scan path. | **HIGHEST** — see §6. |

**ES module globals + try_bind + force-off tuple (LAST):**

| ID | Lines | Action | Risk |
|---|---|---|---|
| E-GLOBDOC | 1013-1028 (ES globals doc + SHAPES comment) | delete complete `"...\n"` literals; **STOP at 1028** (force-off begins 1029) | LOW — two-cut straddle, see §6. |
| E-GLOBDEF | 1039-1047 (9 `_FR13_ES_*` global defs) | **RESUME at 1039** (after force-off 1029-1038). Remove after all ES/REFOLD readers gone. `_FR13_ES_PENDING_BY_REQ` + `_FR13_ES_RESTORE_BY_REQ` are REFOLD-shared → only removable because R-PATHA/R-INTERLEAVE are also being cut. | MED |
| E-TRYBIND | 1069-1123 (`def _fr13_es_try_bind`) | delete — REFOLD-shared (8801-8863 in R-PATHA), block_pool insert (E-FN1), capture (E-CAP); all removed. | MED |
| E-TUPLE | 1037 tuple entry `"FR13_APC_EXACT_SEED"` + (with R-TUPLE) the 2 REFOLD entries + HRS entry | **Remove the whole force-off loop 1029-1038 ONLY when all four families' impl are gone.** Until then KEEP it. | LOW-once-clean |

---

## 5. EXECUTION ORDER (lowest-risk → highest; readers before globals)

The golden rule: **remove every reader of a name before removing the name's definition**, and **remove
the force-off loop only after all four families' guarded impl are gone.**

1. **Scripts delete** (§2): `git rm` the 5 pure-diagnostic scripts (cross-check `k2_cleanup.diff` first).
2. **Script flag-lines** (§3): launcher S-SETUP, S-EXPORT, S-DOCKER (guard SLOT_PIN@524).
3. **HRS clean** (4A): H1 (env-bridge fragments), H3 (docstring, optional). *(H2 deferred to step 6.)*
4. **REFOLD clean** (4B): R-E1 → R-CALL → R-PUBGATE → R-VALIDATE → R-ABSPUB → R-FREE → R-ALLOC → R-STATEP → **R-PATHA** (big self-contained def). *(R-REWIRE, R-INTERLEAVE deferred to step 6.)*
5. **EXACT_SEED clean wholesale** (4C top): E-FN1, E-FN2, E-FN3 (+ dispatch/comment) → E-PATH → E-CTXLENS → E-GAPA → E-GAP2 → **E-CAP** (789-line capture tail; subsumes R-STATEP if not already cut) → E-PUBDISJ.
6. **ENTANGLED collapses** (Phase 5 core — all touch the same two emitted templates, so do them as one careful sitting): **H2 + E-ESON + E-RESTORE + R-INTERLEAVE** (the prefill if/else → live-scan collapse) and **R-REWIRE** (SNAP_FIX override → legacy arm). `py_compile` after **each** edit.
7. **Globals + force-off, LAST**: E-GLOBDOC (stop@1028) → E-GLOBDEF (1039-1047) → E-TRYBIND (1069-1123) → R-GLOB (1048-1068) → **E-TUPLE/R-TUPLE**: now all four families are gone, so delete the whole force-off loop 1029-1038.
8. Final `python -c "import ast,sys; ast.parse(open(PATH).read())"`, then a live boot gate.

> If you prefer a smaller, lower-risk landing: stop after step 5. Steps 6-7 are the only cuts that touch
> live-scan-adjacent emitted templates; leaving the force-off loop + inert globals in place is fully
> behavior-preserving and defers the boot-crash-prone edits.

---

## 6. HIGHEST-RISK CUTS (emit-template breakers — read before touching)

1. **E-RESTORE / H2 collapse (6027-6468)** — the LIVE default `self.chunk_gated_delta_rule(...)`
   (6469-6484) is nested inside the `else:` of the HRS `if _fr13_apc_active:` gate, *after* the shared
   ES/REFOLD restore loop. This is where HRS (6034-6201), the ES restore disjunct (6217), and 4 REFOLD
   sub-blocks (6218-6439) all converge. **Never delete the `else` or the HRS `if` wholesale** — collapse
   `if <HRS> else <ES-restore-with-REFOLD-subblocks>` to the dedented live-scan path. A single mis-dedent
   here boot-crashes the worker.
2. **R-REWIRE (14580-14700)** — inside the **KEEP SNAP_FIX** `get_temporal_copy_spec` override, fused
   with the ES chunked-ptr redirect + GAP-2 position guard. Collapse to the **legacy slot-keyed `else`
   arm only** (keep 14632-14641); preserve `_fr13_fx_*` setup, GAP-2 guard, LEAF_CROSSCHECK, and the
   chunked-ptr APPLY at 14794. Do NOT treat as a block delete.
3. **E-CAP (6616-7404)** — 789-line `if EXACT_SEED=="1":` block that is the *last* thing in the
   `prefill_scan_replacement` string. Delete **complete line units only**, keeping 6615 and the string
   terminator `'''` at 7405. Safe *because* live neighbors bound it — but a partial cut orphans a
   `try:`/`except`.
4. **Two-cut straddle in the gdn_linear globals string** — E-GLOBDOC (1013-1028) and E-GLOBDEF
   (1039-1047) sit on **opposite sides** of the KEEP force-off loop (1029-1038). **Never delete 1013-1047
   as one range** — you would delete the force-off. Two separate cuts: one ends at 1028, the next starts
   at 1039.
5. **E-GLOBDEF / E-TRYBIND ordering** — `_FR13_ES_PENDING_BY_REQ`, `_FR13_ES_RESTORE_BY_REQ`, and
   `_fr13_es_try_bind` are read by REFOLD (R-PATHA 8801-8863), the restore else (E-RESTORE), and popped by
   the freereq leak-fix (getattr-guarded). Remove those readers **first** or hit a NameError at gdn import.

---

## 7. RESIDUAL / follow-up

Applying the full plan retires all four dead-flag families and the force-off loop. The only "left as-is"
items are the KEEP-list shared maps that become inert-empty (`_FR13_APC_SSM_CHUNKED_PTR_BY_REQ` and the
getattr-guarded freereq/SNAP_FIX cross-references), which are correct-by-guard and cost nothing. Doc
pointers to the 5 deleted scripts should be annotated. `research/fr13_workflows/k2_cleanup.diff` already
drafts the script cut — reconcile, don't double-apply.
