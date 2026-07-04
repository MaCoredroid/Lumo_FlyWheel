# FR13 Flag / Dead-Code Audit (synthesized + spot-check-corrected)

## Basis & scope
- **Frozen copies** in `scratchpad/audit/`: `patcher_HEAD.py` (= `scripts/fr10_phase4_patch_vllm_tree_gdn.py`), `fr13_launch_forked_fa2_tree_server.sh` (674 L), `fr13_bigdenom_swe_serve_variant.sh` (635 L).
- **HEAD MOVED**: prompt cited `0fafd978` (patcher 19158 L); actual HEAD at freeze = **`53368a3b`**, patcher = **19301 L** (+143 L net). The concurrent workflow's **FR13_OBS counter registry landed** (injected block `patcher_HEAD.py:1190-1250`, `_fr13_obs_bump`/`_FR13_OBS`). Consequence: **launcher & serve_variant line refs are STABLE** (both files byte-identical to the audit basis — 674 / 635 L). **PATCHER line refs are VOLATILE**: net +143 L but the OBS work plus edits landed at multiple points, so per-site displacement varies and is NOT a uniform offset (verified: e.g. GPU_COMMITTER predicate `:8772`→**`:7774`**, LUMO_FB gate `:7132`→**`:7257`**, marker `:1571`→**`:1637`**). **Treat every patcher line number in the tables below as an approximate anchor — re-grep by flag/symbol name against the frozen copy.** The load-bearing items (Phase-A deletion candidates + all §2 dead-code blocks) are re-verified with EXACT frozen-`53368a3b` line numbers.
- Producer/consumer re-derivation grepped the **whole repo** `scripts/` + `src/` (not only the 3 audited files) — this is where audit-R's two wrong verdicts were caught.
- **Byte-identity bar**: on the locked cat9 default path `FR13_ENABLE_APC` defaults 0 (launcher :208) ⇒ APC_FLAGS empty ⇒ no `--enable-prefix-caching` ⇒ the ENTIRE APC/ES machinery is inert. So every APC/ES flag is inert on the golden path; the deletion candidates below additionally touch no arm at all.

## Corrections applied vs the two input audits
1. **FR13_APC_REQUIRE_HIT_SUFFIX_CAP** — audit-R said DEAD (`deletion_candidate=true`, "no producer"). **WRONG → LIVE.** `scripts/fr13_apc_engaged_test.sh:35` and `scripts/fr13_spine_cache_engaged_test.sh:32` both `export FR13_APC_REQUIRE_HIT_SUFFIX_CAP=1000000` then drive serve_variant, so the gate `serve_variant:428-430` fires and validates the worker marker's `HIT_SUFFIX_CAP` field (emitted unconditionally at `patcher:1638`). Root cause of miss: only the 3 audited files were searched for producers.
2. **FR13_SCAN_ALIGN / FR13_SCAN_ALIGN_MODE / FR13_RECOMPUTE_NODE_PARALLEL / FR13_NPAD_INVARIANT** — audit-R (conv/speed block) said DEAD (`deletion_candidate=true`, "no consumer in patcher"). **WRONG → LIVE-DEFAULT-OFF.** They have **0 patcher hits** but are read live in the kernel source `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:82 / :110 / :128 / :162`. Do NOT delete. (This is exactly the "audit all three: patcher + kernel + fa2 patch" lesson.)
3. **FR13_APC_FIXED_BUFFER** — confirmed DEAD everywhere (0 Python consumers repo-wide). Deletion candidate = **TRUE** (see Phase A).
4. **FR13_APC_REQUIRE_SHADOW** — confirmed doubly-dead (no producer anywhere; marker `patcher:1638` writes `pid SNAP_FIX HIT_SUFFIX_CAP ZEROACCEPT CONV_FIX HIT_RECURRENT_SUFFIX` — **no `SHADOW=` field**, so `serve_variant:433 grep -q "SHADOW="` can never match). Deletion candidate = **TRUE** (serve_variant gate only).

---

## 1. Flag table (grouped by family; verdicts corrected)

### Family A — APC / prefix-cache carrier & EXACT_SEED (all inert on locked path; live only when FR13_ENABLE_APC=1)
| Flag | Default | Verdict | Producer | Consumer | Evidence |
|---|---|---|---|---|---|
| FR13_ENABLE_APC | 0 | LIVE gate (master) | launcher :208 | launcher :230 (chooses APC_FLAGS) | =0 ⇒ whole family inert; the golden-path guarantee. |
| FR13_APC_EXACT_SEED | 0 | LIVE-OFF, **PARTIAL** | launcher :304/:457; bridge :1544/:1570 | patcher :6432 (prefill capture), :13623/:13736 (redirect), :18604 (blockpool), worker post/preprocess | Prefill+restore wired; **decode-side chunked-ptr redirect DEAD** (see §2). |
| FR13_APC_SNAP_FIX | arm 1 / locked 0 (code "1") | LIVE-ON (arms) | launcher :282 / :451 (-e :-0) | patcher :8383, :13485/:13561 | Working SSM node-bank fix, verify3b FAITHFUL 240/240; asserted by REQUIRE_SNAP_FIX. |
| FR13_APC_SNAP_FIX_ZEROACCEPT | arm 1 / locked 0 | LIVE-ON (arms) | launcher :283 / :452 | patcher :8386/:8419/:8433 | BAKED 2026-06-27; publishes committed-root row on accepted_len==0. |
| FR13_APC_CONV_FIX | 1 | LIVE-ON | launcher :280 / :448 (-e :-1) | patcher :13280/:13309/:13324/:13419 | Tree conv node-copy + whole-row snapshot; inert when APC off. |
| FR13_APC_CONV_SNAPSHOT | arm 1 / locked 0 | LIVE-ON (arms) | launcher :281 / :449 | patcher :13308/:13323 | Whole-row conv snapshot (SGLang #25587). code "1" vs docker `-e :-0` divergence. |
| FR13_APC_CONV_SNAP_FIX | arm 1 / locked 0 | LIVE-ON (arms), **PARTIAL** | launcher :284 / :453 | patcher :8384/:8403, :13486/:13568 | BAKED 2026-07-03; conv twin of SNAP_FIX, redirect still falls back on some rows. |
| FR13_APC_PRE_SNAP_FIX | 0 | LIVE-OFF | launcher :285 / :454 | patcher :8385/:8404, :13487/:13557 | CLR preprocess SSM redirect; never baked ON. |
| FR13_APC_HIT_RECURRENT_SUFFIX | 0 | SUPERSEDED (by EXACT_SEED) | launcher :295 / :455; **also =1 in engaged tests** | patcher :5877 (`_fr13_apc_active` needs not-ES & HRS==1) | Un-baked 2026-06-27; still exercised by `fr13_apc_engaged_test.sh`/`fr13_spine_cache_engaged_test.sh` (HRS=1). Togglable, not unreachable. |
| FR13_APC_HIT_SUFFIX_CAP | 64 | LIVE (marker+REQUIRE) / HRS-read SUPERSEDED | launcher :296 / :456; tests set =1000000 | patcher :5877-region (recompute cap) **AND** :1638 (unconditional marker write) → REQUIRE gate serve_variant:428-430 | Refined from "inert while HRS=0": value is marker-surfaced + REQUIRE-gated even at HRS=0. |
| FR13_APC_BLOCK_REFOLD | 0 | LIVE-OFF (orphaned producer) | launcher :306 / :458 | patcher :948 (`_FR13_REFOLD_ON`), publish → `_FR13_ES_PENDING_BY_REQ` :8268 | Fold fires (REFOLD_APPLIED) but publishes to a channel the snapshot never reads (RESTORE_USED=0). RETRY candidate, not dead. |
| FR13_APC_BLOCK_ALIGN_45477 | 1 | LIVE-ON (APC-align only) | launcher :450 | patcher :7596 | vLLM PR#45477 backport; inert on locked path. |
| FR13_APC_CONFIG_ONLY | 0 | LIVE-OFF (launcher-only selector) | env/serve_variant XFLAGS | launcher :230 | Cache-OFF matched-config A/B arm. |
| FR13_APC_LEAF_CROSSCHECK | 0 | LIVE-OFF (diag) | launcher :462 | patcher :13689 | Read-only SSM-vs-conv leaf mismatch logger. |
| FR13_APC_CACHEROW_DUMP(+_LIMIT 80) | ""/80 | LIVE-OFF (diag) | launcher :463/:464 | patcher :13787/:13799/:13790 | torch.save, capture-guarded. |
| FR13_APC_CONV_RESTORE_CAPTURE | 0 | LIVE-OFF (diag) | launcher :587 | patcher :5732 | Clones restored conv seed; OFF ⇒ byte-identical. |
| FR13_APC_EXACT_SEED_ENG_LOG | /logs/…eng.log | LIVE-within-ES | (constant) | patcher :6449/:13738/worker | Opened only under ES+SERVE_LOG. |
| FR13_ES_CKPT_CAP | 64 | LIVE-within-ES | (constant) | patcher (BlockPool ckpt LRU) | Per-blockhash chunked ckpt store cap. |
| FR13_APC_ENV_FLAG_FILE | /logs/fr13_apc_env.flag | LIVE (inject is no-op) | patcher :18520 write | patcher :1544 (only if SNAP_FIX absent from worker env) | Belt-and-suspenders; inject never fires (worker inherits os.environ). |
| FR13_APC_BRIDGE_MARKER_FILE | /logs/fr13_apc_bridge_loaded.flag | **LIVE (real engagement gate)** | patcher :1637 (UNCONDITIONAL) | serve_variant :415/:425 | Load-bearing worker-engagement proof; missing bridge can't masquerade as engaged. |
| FR13_APC_BRIDGE_ERR_FILE | /logs/…error.flag | LIVE (fail-loud sink) | (constant) | patcher :1555-region | Written only on inject/marker exception. |
| FR13_SERVE_LOG | 0 | LIVE-OFF (log throttle) | launcher :459 | patcher :501/:987/:6227/:6442/:13759/… | Silences ~5/s hot-path ES logging; functional restore runs regardless. |
| FR13_LEAK_PROBE | 0 | LIVE-OFF (diag) | launcher :461 | patcher :7552 | Host-RSS leak instrument. |
| FR13_APC_REQUIRE_SNAP_FIX | unset | LIVE (post-boot gate) | serve_variant :116/:138; +4 engagement .sh | serve_variant :424-426 | Guards vacuous ES run. |
| **FR13_APC_REQUIRE_HIT_SUFFIX_CAP** | unset | **LIVE (CORRECTED from DEAD)** | `fr13_apc_engaged_test.sh:35`, `fr13_spine_cache_engaged_test.sh:32` | serve_variant :428-430 | Verdict flipped (see Corrections #1). |
| **FR13_APC_FIXED_BUFFER** | 0 | **DEAD (del=TRUE)** | launcher :305/:307/:460; also `fr13_apc_fb_speedgate.sh:55`, `fr13_apc_multiturn_one_arm.sh:63` | **NONE** (0 Python reads repo-wide) | Never consumed; `fr13_apc_fb_speedgate.sh` A/B over it is VACUOUS (both arms byte-identical). |
| **FR13_APC_REQUIRE_SHADOW** | unset | **DEAD/VACUOUS (del=TRUE)** | NONE (no arm sets it) | serve_variant :414/:432-434 | Marker never emits `SHADOW=` (:1638) ⇒ gate can never pass. serve_variant-only. |

### Family B — FR10 tree-decode core (locked cat9 serving path)
| Flag | Default | Verdict | Producer | Consumer |
|---|---|---|---|---|
| FR10_DECODE_MODE_DEFAULT | tree_mtp | LIVE-ON (master switch) | launcher :30/:504; serve_variant native arms → naive_mtp | patcher :902/:7310/:11246/… |
| FR10_ENABLE_TREE_GDN | 1 | LIVE-ON | launcher :502 | patcher :2280/:4306/:12897/:13277 |
| FR13_REPLAY_ROUTE | 1 | LIVE-ON (runtime read) | launcher :475; serve_variant NEEDS-gate | patcher :4438 (init-side `if True` bakes 331/817 dead) |
| FR13_DEVICE_MULTIDRAFT | 1 | **LIVE-ON, distribution-lossless (not byte)** | launcher :496 | patcher :10288 | ← flag for lossless-gate owner (not in PIPELINE_LOCK) |
| FR13_DEVICE_MULTIDRAFT_KERNEL | /workspace/…kernel.py | LIVE-ON (dependent) | launcher :497 | patcher :10362 |
| FR10_ALLOW_LINEAR_FALLBACK | unset→0 | LIVE-OFF (fail-loud policy) | launcher :604 (`unset`) | patcher :2491/:2835/…/:10203 (`!= "1"` raise) |
| SPEC_CONFIG | tree topology | LIVE-ON | launcher :199; native arms strip tree | vLLM --speculative-config |
| FR13_TREE_BONUS_SELF | 1 | LIVE-ON | launcher :446 | patcher :8750 (=0 = legacy bug path) |
| **Baked-ON, env value now INERT** (SUPERSEDED; export pinned by NEEDS-gate/tests — do NOT delete without updating gate): FR13_CONV_COMMITTED_PATH (patcher literal True :2474; dep-guards `if False` :1045/:1049), FR13_TREE_CONV_FUSED (:1017; `if True` :675), FR13_EAGER_PACK (:1008/:350), FR13_DRAFTER_SINGLE_LOGITS (:14460), FR13_TREE_SAMPLE_ROW (:14079; `if False` :14086), FR13_TREE_PER_REQ_GEN (:10449), FR13_TREE_REQKEY (live read survives at :12199; dep-guard `if False` :14086), FR13_TREE_ATTN_EXP2_SOFTMAX (alt dead `if False` :16179), FR13_TREE_REMAP_SEQ (kernel `if True` :356). ||||

### Family C — GDN-scan / kernel-mode selectors (consumer = kernel src, NOT patcher) — **all LIVE-DEFAULT-OFF (CORRECTED from DEAD)**
| Flag | Default | Producer | Consumer |
|---|---|---|---|
| FR13_SCAN_ALIGN | 0 | launcher :505; serve_variant K1 guard :304-306 (`FR13_ALLOW_SCAN_ALIGN`) | `fr10_gdn_tree_kernel.py:82` |
| FR13_SCAN_ALIGN_MODE | body | launcher :506 | `fr10_gdn_tree_kernel.py:110` |
| FR13_RECOMPUTE_NODE_PARALLEL | 0 | launcher :507 | `fr10_gdn_tree_kernel.py:128` |
| FR13_NPAD_INVARIANT | 0 | launcher :508 | `fr10_gdn_tree_kernel.py:162` |

### Family D — speed A/B arms (default-OFF except noted; lossless-by-construction)
FR13_GPU_COMMITTER 0 (serve_variant cat9-opt1 :81 →1; patcher :8772), FR13_COMMITTER_SYNCKILL 0 (paired; patcher :7648), FR13_GPU_COMMITTER_KERNEL (patcher :18364), FR13_GB10_FP8_GEMV_CFG 0 (cat9-opta :80; patcher :18246), LUMO_FB_KERNEL_ROWS unset**→forced =1 by serve_variant :247/:303** (patcher :7132 — *this one DOES change decode numerics on the locked SWE campaign*), LUMO_FB_PROJ_PAD_ROWS 16 (:7167), FR13_SFWD_GPU_TIMER(+_MAXPENDING/_JSON) 0, VLLM_BATCH_INVARIANT 0 (serve_variant asserts =0), LUMO_BATCH_INVARIANT_VLLM 0 (consumer outside patcher).

### Family E — FA2 sibling-patch flags (consumer = `scripts/fr13_patch_fa2_tree_bias.py`, NOT the audited patcher)
FR13_FA2_TREE_BIAS 1 LIVE-ON (:570/:575/:714), FR13_FA2_PREFILL_NATIVE 1 LIVE-ON (:499), FR13_BI_TREE_ATTN 0 LIVE-OFF (cost-gated BLOCKED, :781).

### Family F — diagnostic / capture family (~60 vars, all LIVE-DEFAULT-OFF, empty/0, launcher :519-596 bulk passthrough, none on locked path)
FR12_FULL_ATTN_CAPTURE, FR12_SUBKERNEL_CAPTURE, FR13_TREE_ATTN_OP_CAPTURE, FR13_FLASH_ATTN_OP_CAPTURE, FR13_FINAL_LOGIT_CAPTURE, FR13_PREPROCESS_INPUT_CAPTURE, FR13_PREFILL_GDN_CAPTURE, FR10_*_HIDDEN/LOGIT_CAPTURE, FR10_TREE_GDN_CAPTURE_PAYLOAD/COMMIT_HANDOFF_LOG/SRC_NATIVE_PAYLOAD, FR12_TREE_SCAN_NATIVE_SPINE/CONV_NATIVE_SPINE/CONV_NATIVE_PRIOR_READ, FR12_NATIVE_SPINE_ORACLE, FR13_FA2_MAB, FR13_GDN_SUBOP_MAB, FR13_CHASE_DIAG, FR13_COMMIT_ARGMAX_GATE, FR13_FORK_MARGIN_DUMP, FR13_HIDDEN_SUBSTITUTE, FR13_CONV_REPLAY_NODES (manual-only, no launcher producer), FR13_TCF_SELFCHECK, FR10_METRICS (each `==\"1\"`/`bool(env)` gated + torch.save side-effect only; several raise class-9 if combined with REPLAY_ROUTE=1). **Deletion-candidate as a class only after the campaign closes** — they are the live diagnostic tooling. FR12_TREE_CONV_NATIVE_BF16_TAPS 1 is **NOT** diagnostic: required-ON losslessness invariant, RAISES if =0 (patcher :1052-region).

---

## 2. Dead-code inventory (frozen `53368a3b`; verified line ranges)
1. `patcher_HEAD.py:19051-19205` — `_fr13_es_worker_postprocess` has UNCONDITIONAL early `return` after the docstring (comment ":19059 iter8: postprocess relay DISABLED"); the entire relay body is unreachable. block_pool insert now binds pos→hash directly from `_FR13_ES_PENDING_BY_REQ`.
2. `patcher_HEAD.py` — module globals `_FR13_APC_SSM_CHUNKED_PTR_BY_REQ` / `_FR13_APC_SSM_CHUNKED_POS_BY_REQ` are **consumer-only** (reads at :6676/:6681/:13781/:13797; NO assignment anywhere). ⇒ EXACT_SEED decode-side chunked-ckpt redirect always gets None → falls back to the recurrent bank leaf (campaign §14/§16 "orphaned producer never wired"). Dead until task-7 fold→ptr wiring lands.
3. `patcher_HEAD.py` — `_FR13_APC_SSM_ALIGNED_POS_BY_REQ`: producer :13223 but only consumer :13800 is nested under `if _fr13_fx_chunked_ptr is not None:` (dead per #2) ⇒ write-only ⇒ effectively dead.
4. `patcher_HEAD.py:922` — `_FR13_ES_BLOCK_PENDING` declared "(legacy; iter8 relay disabled, kept inert)"; written only inside the dead relay (:19092/:19095) ⇒ never populated.
5. `patcher_HEAD.py:13984/:14000` — `# FR13_APC_ALIGN_TREE_AWARE` sentinel; VERBATIM+STALENESS_AUDIT bodies deleted. **The string is load-bearing as the re-injection idempotency guard — do NOT delete the comment alone.**
6. `patcher_HEAD.py:5852-region` (HRS recurrent-suffix recompute, gate `not ES and HRS==1` :5877) — SUPERSEDED/dormant; togglable + still exercised by engaged tests, so "dormant on locked default", not unreachable.
7. Baked-flag dead dep-guards: `if False` at :1045, :1049, :14086, and the exp2 alt-mask `if False` at :16179 — never fire.
8. Launcher orphans on **main HEAD**: `fr13_launch...:305/:307/:460` (FIXED_BUFFER, no consumer anywhere). NOTE the SCAN_ALIGN/RECOMPUTE/NPAD launcher exports (:505-508) are **NOT** orphans — kernel src consumes them (Corrections #2).

---

## 3. PHASE-A DELETION LIST (spot-check-confirmed, behavior-preserving; launcher/serve_variant ONLY — patcher is mid-edit)
> Scope deliberately excludes patcher-internal dead code (concurrent edit + load-bearing sentinels). Only orphan env plumbing that touches NO consumer on ANY path.

**A1. FR13_APC_FIXED_BUFFER** — remove `fr13_launch_forked_fa2_tree_server.sh:305` (`: "${FR13_APC_FIXED_BUFFER:=0}"`), drop the token from the `export` list `:307`, and remove the `-e FR13_APC_FIXED_BUFFER=…` line `:460`. (Leave `fr13_apc_fb_speedgate.sh`/`fr13_apc_multiturn_one_arm.sh` refs OR clean them too — either way inert; if you keep the speedgate, annotate that its A/B is vacuous.)
- Verify: (1) `bash -n fr13_launch_forked_fa2_tree_server.sh` parses. (2) `grep -rc FIXED_BUFFER src/ scripts/*.py` = 0 (no consumer regressed). (3) Golden-flag fingerprint (`[6,6,4,6]` integration test, PIPELINE_LOCK) unchanged. (4) Same-boot in-process byte gate (greedy + t0.6 identical streams) — trivially preserved since the var reaches no code.

**A2. FR13_APC_REQUIRE_SHADOW gate** — in `fr13_bigdenom_swe_serve_variant.sh` drop the `${FR13_APC_REQUIRE_SHADOW:-}` term from the `-n` guard `:414` and delete the block `:432-434`.
- Verify: (1) `bash -n fr13_bigdenom_swe_serve_variant.sh`. (2) Confirm no repo script sets `FR13_APC_REQUIRE_SHADOW` (`grep -rn REQUIRE_SHADOW scripts/` = only these 3 lines). (3) Confirm marker format `patcher:1638` has no `SHADOW=` (gate was unreachable). (4) serve_variant is off the served path ⇒ cat9 byte-identity trivially preserved. (5) The two OTHER REQUIRE gates (SNAP_FIX, HIT_SUFFIX_CAP) must still fire — re-run `fr13_apc_engaged_test.sh` to confirm exit-4 vacuity guards still trip.

**Explicitly NOT Phase-A (rejected from audit-R):** SCAN_ALIGN, SCAN_ALIGN_MODE, RECOMPUTE_NODE_PARALLEL, NPAD_INVARIANT, REQUIRE_HIT_SUFFIX_CAP — all LIVE (kernel/test consumers). Deleting them would break the kernel-mode routes or the engaged-test vacuity gates.

**Deferred to Phase-A′ (after the OBS edit settles + campaign green-lights, patcher-internal):** dead blocks §2.1-4, §2.7 dep-guards. Each needs: extract injected string → `python -c "import py_compile"` on the generated module → smoke-boot the patched worker → per-layer 0.0 + same-boot byte gate. NOT now (mid-edit).

---

## 4. PHASE-B SPLIT SKETCH — decompose `fr10_phase4_patch_vllm_tree_gdn.py` (19.3k L) by patched target
The patcher is one monolith that string-injects into many vLLM classes. Split into a package `fr10_tree_patch/` where each module owns ONE patched target and exposes `apply(vllm_mod)`; a top-level `patch_steps.py` orders them. **Ordering constraints (must preserve):**
- `steps[0]` env/observability bootstrap — `_FR13_OBS` registry, `_fr13_obs_bump`, baked-flag constants (`_FR13_EAGER_PACK` etc.), fail-loud guards. Everything else imports these.
- `steps[1]` **GPUModelRunner** — depth-position/MRoPE remap (`mrope_base = num_computed_tokens + depth_offsets`), tree_state scratch alloc (REPLAY_ROUTE), input-buffer capture hooks. Must run before attention/GDN (they read tree metadata it installs).
- `steps[2]` **Qwen3Next GDN linear_attn** — conv (bf16 taps, whole-row snapshot), sequential rank-1 tree-scan, gate, o_proj; APC SSM node-bank (SNAP_FIX family) + restore. Depends on step1 tree descriptors.
- `steps[3]` **full-attention** — TREE_ATTN ancestry-mask path + depth-RoPE; the FA2 fork lives in the sibling `fr13_patch_fa2_tree_bias.py` (keep separate module, ordered after GDN).
- `steps[4]` **committer / rejection-sampler** — device multidraft (temp>0), per-req RNG, spine/tree commit, GPU_COMMITTER opt arm.
- `steps[5]` **BlockPool / scheduler (EXACT_SEED only)** — chunked-ckpt capture/store/restore, bridge sidecar+marker. Gate the whole module on `FR13_ENABLE_APC`.
- `steps[6]` diagnostics/capture family — all default-OFF torch.save taps; isolate so the serving modules carry no capture branches.
Cross-module invariants to assert at load: (a) no consumer without producer (see Rule 1); (b) baked constants centralized in step0; (c) sentinel strings that guard re-injection stay verbatim.

---

## 5. MAINTENANCE RULES (for future agents)
1. **Never leave a consumer without a producer, or a producer without a consumer.** Every `_FR13_*_BY_REQ` map and every env flag must have ≥1 live writer AND ≥1 reachable reader on the intended path. The FR13_APC chunked-ptr dead half (§2.2-3) and audit-R's two wrong verdicts both trace to a broken producer/consumer pair. When you add a redirect map, wire its producer in the SAME change or it silently falls back.
2. **Search the WHOLE repo, not just the file you're patching.** Consumers span `patcher` + `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py` + `scripts/fr13_patch_fa2_tree_bias.py` + the `fr13_*_test.sh`/`*_gate.sh` harnesses. A "no consumer" claim from grepping one file is how SCAN_ALIGN got mis-called DEAD.
3. **Marker/flag ABSENCE must be self-describing.** A REQUIRE-gate that greps for a field the marker never emits (REQUIRE_SHADOW) is silently vacuous. Rule: the marker writer (`patcher:1637-1638`) and every REQUIRE grep in serve_variant must reference the SAME field set; add a field to the marker in lockstep with any new `FR13_APC_REQUIRE_*` gate, and make a missing field a loud FAIL, never a silent pass.
4. **Throttled/observability logging carries registry keys — route it through the FR13_OBS layer.** The new `_FR13_OBS` registry (`patcher:1190-1250`, `_fr13_obs_bump("<key>")`, `FR13_OBS_FINAL_PATH=/logs/fr13_obs_final.json`) gives an unthrottled count even when `FR13_SERVE_LOG=0` throttles the human line. Any new hot-path event MUST `_fr13_obs_bump` a stable key (don't invent a parallel counter); SERVE_LOG only gates the text line, never the count. This is how you tell "fell back 384×" from "never ran".
5. **Baked flags stay as pinned exports until the gate is updated in lockstep.** SUPERSEDED flags (Family B baked list) are inert at runtime but their launcher exports are pinned by serve_variant NEEDS-gates and `tests/test_fr13_nondet_chase_fixes.py`. Delete the export ONLY together with the gate/test line, and NEVER delete the baked CODE (it is the locked default behavior). Note the stale test at `test_fr13_nondet_chase_fixes.py:327` still greps for the removed `FR13_TREE_REMAP_SEQ` env read (now `if True:` :356) — fix when touching that suite.
6. **Latent footgun:** launcher `-e` lines default SNAP_FIX/ZEROACCEPT/CONV_SNAPSHOT to `:-0` while the injected code defaults `os.environ.get(...,"1")`. Any invocation reaching the patched vLLM WITHOUT these `-e` lines silently defaults several APC fixes ON. Keep the `-e` explicit-0 lines; if you refactor the launcher, preserve them.
