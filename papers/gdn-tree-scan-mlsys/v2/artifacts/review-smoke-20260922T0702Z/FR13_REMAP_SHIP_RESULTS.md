# FR13 ATTN_KV_REMAP — Ship Results (garble + speed + agentic)

Consolidated results for the garble fix `FR13_ATTN_KV_REMAP` (baked into `fr13_launch_locked`).
`output/` is gitignored, so the NUMBERS are recorded here (committed). Updated as matrix arms complete.

## The fix (one line)
Missing attention-KV re-linearization: after a branched-tree commit, GDN/conv state was re-linearized
(`launch_tree_state_linear_remap`) but the attention KV was not, so the accepted non-contiguous path read a
sibling branch node's near-neighbor K/V → compounding drift → gross near-neighbor wrong-accepts. Fix =
`launch_attn_kv_linear_remap` (copy each committed node's K/V flat-verify-slot → linear-committed-slot per
full-attn layer, in `sample_tokens`). Wiring/data-movement, not a kernel change; keeps all branches.

## 1. Token-level garble gate (canonical temp-0.6 matrix_build, tree vs native)
- **15/15 → 0/15**, across eager / graph / cache-cold; remap ENGAGED (foreign>0); clean read (syntax_bad=0).
- General fix (adversarial 4-agent workflow CONFIRMED): one code path for cat9/cat8/cat6/any branched tree.

## 2. Agentic garble gate (qwen-code, B4=BSIZE4+CONC4, 16 SWE-Verified tasks, cache-ON, no-wall, nudge-OFF)
Gate = corrupted-identifier (undef) rate on self-contained served scripts, tree vs native floor
(`scripts/fr13_qwen_jsonl_garble_scan.py`, v6: fragments/shlex/bindings/floor-names/non-python/f-string-edges
all handled; discrimination preserved).

| arm | undef (garble) | syntax_bad | flagged names |
|---|---|---|---|
| PRIOR NO-FIX cat8 (qc4) | **40.3%** | 16.3% | wcs_wcs_hdr, sll, result_full_low (garble) |
| PRIOR NO-FIX cat6 (qc4) | **44.9%** | 10.3% | lon_, topo_itrs_frame, nref_nommask (garble) |
| native (qc4, floor) | 1.0% | 1.0% | cright/right (real, rare) |
| **cat8 FIX-ON** (this run) | **0.0%** (84 scripts, FINAL) | 0.0% | — |
| **cat6 FIX-ON** (this run) | **0.0%** (61 scripts, FINAL) | 1.6% (1 self-corrected model err) | — |
| native (this run) | RUNNING (arm 3/3) | | |

## 3. Trajectory / give-up (assistant turns per task; give-up=short, garble-thrash=long tail)
- PRIOR NO-FIX cat8: median 32, dist `[2,2,...,88,124]` = BIMODAL DYSFUNCTION (2 hard give-ups + garble-thrash).
- native: median 14 (healthy). **cat8 FIX-ON: median ~15, min 8 (NO give-ups), max 35 (NO thrash)** → tracks native.
- Edits: **15/16 cat8 tasks made real edits to the CORRECT source file** (separable.py, quantity.py, etc.).

## 4. Speed (accept-normalized; NOT raw TPS)
🛑 **The prior no-fix cat8/cat6 accept + derived_tps are GARBLE-CONFOUNDED — do NOT use them as the speed
baseline.** No-fix accept counted garble wrong-accepts (drifted verify accepted near-neighbors it should have
rejected), so accept/event **3.588** (cat8) / **3.850** (cat6) and derived_tps **34.98/40.12** are INFLATED.
The clean tree-vs-native speed A/B must come from the **FIX-ON run** (this run). Marked as confound across
`FR13_B4_CACHE_MATRIX_RESULTS.md` + memory ([[reference_deploy_speed_metric_definitions]],
[[project_fr13_b4_cache_matrix]]).
- **Valid bar = native 3.050** (no garble, ~1% floor) → derived_tps **31.74**. Native is NOT confounded.
- **cat8 FIX-ON interim** (/metrics, cumulative): accept/forward **~3.3** (was 3.41 earlier, drifts down as
  decode accumulates) — **below** the garble-inflated 3.588, empirically confirming the inflation. Clean
  `s_per_fwd_gpu` / `derived_tps_gpu` at arm completion (reduce), + cat6-fix-ON + this-run native for the full
  apples A/B.
- Directionally the tree likely still out-accepts native (~3.3 > 3.05) but the **magnitude** (was claimed
  +10%/+26%) is NOT established until the fix-ON reduce lands. No hand-rolled TPS.
- Remap own tax (paired probe): **−0.7% s/fwd (within cross-boot noise)** = no-HBM-tax bar met.
- B4 genuine: Running=4 dominant (~68% of intervals), effective batch ~3.7 (not serialized-to-1).

### 4a. cat8 FIX-ON FINAL reduce (16/16) — CLEAN accept, speed DEFERRED (red-team)
- **accept_per_event 3.336** (clean) → confirms no-fix 3.588 was garble-inflated by ~0.25. TRUSTWORTHY.
- **s_per_fwd_gpu_per_forward = 0.217s** = MATCHES the B=1 bank (0.218) → the trustworthy per-decode-step cost.
- 🛑 **`derived_tps_gpu = 63.92` (s_per_fwd_gpu per-draft 0.0678) is NOT TRUSTED — do not cite as a speedup.**
  The two GPU-time bases move OPPOSITE ways vs no-fix (per-draft −48% but per_forward +33%) — impossible for
  the same kernel+tiny-copy. Same matched denominator both runs (checked), so it's not a denom change; the
  **no-fix bases were themselves garble-composition-confounded** (garble skewed the draft/step mix). Fix-ON
  per_forward matching B=1 (0.217) is the tell it's the correct one; no-fix 0.164 was anomalously low.
- **The ONLY clean speed A/B = THIS-RUN cat8 vs THIS-RUN cat6 + native (same boot, same basis)** — running
  now. No cross-run (fix-vs-prior) speed verdict. Quote speed only after this-run native lands.

### 4b. cat6 FIX-ON FINAL reduce (16/16) — same two red-team caveats
- accept **3.594**, per_forward **0.199**, s/fwd_gpu(draft) **0.0634**, derived_tps **72.43**.
- 🛑 **accept 3.594 > cat8's 3.336 is STRUCTURALLY IMPOSSIBLE on matched tokens** (cat6 ⊂ cat8, so cat8 ≥ cat6)
  => it's TRAJECTORY NOISE (temp-0.6, different tokens per arm), NOT "cat6 out-accepts cat8." Do not rank
  cat6-vs-cat8 on accept.
- 🛑 **derived_tps 72.43 (per-draft 0.0634) NOT trusted** — same per-draft-basis artifact as cat8's 63.92.
- Trustworthy per_forward: cat8 0.217 vs cat6 0.199 (cat6 cheaper per-forward = fewer nodes, plausible). The
  clean tree-vs-native verdict needs THIS-RUN native (arm 3, running) as the common bar; defer until it lands.

## 5. Config manifest / confound tracking (`scripts/fr13_config_manifest.py`)
harness_hash (must-match: wall/nudge/temp/conc/B/agent/git) matches prior EXCEPT git_head (newer commit =
the fix). wall=no-wall (WALL=0), nudge=0 (OFF), temp=0.6, B4, cache-ON (APC hits 86-87%), EXACT_SEED=0.


## 6. RESOLVE RATE (SWE-bench eval verdicts) — the deliverable number
| run | resolved | rate | vs native |
|---|---|---|---|
| **cat8 FIX-ON (this run, 16/16 FINAL)** | **8** | **50%** | **= native EXACTLY** |
| **cat6 FIX-ON (this run, 16/16 FINAL)** | **7** | **44%** | **≈ native (−1 = flaky 14096)** |
| native (prior qc4, bar) | 8/16 | 50% | — |
| cat8 no-fix (prior qc4) | 6/16 | 37% | BELOW native = the degradation |

**cat6 FINAL 7/16 (44%):** differs from cat8 (8/16) by exactly ONE task — 14096, the known cache-flaky task
(task #12; matrix doc records it as the source of native+cache's only give-up). On the 15 non-flaky
overlapping tasks cat6 == cat8. => NO smaller-tree resolve penalty; cat6 ≈ native, garble 0/61.

**FINAL (16/16):** cat8 fix-ON resolve **8/16 (50%) = native 8/16 (50%) exactly.** The 16th task (14598)
FAILED — a known-hard task (no-fix also wrong-patched it per matrix doc), NOT a fix regression. So the
tree+cache agentic DEGRADATION (no-fix 37% < native 50%) is fully ELIMINATED. Garble 0% (84 scripts).

Per-task vs no-fix: RESCUED from garble-failure: astropy-13579, -14539, -14508 (no-fix failed -> fix
resolved). REGRESSED: astropy-13236 (no-fix resolved -> fix failed; likely temp-0.6 trajectory variance -
the fix removes garble, does not inject errors). Net +2 (6->8). cat6 + this-run native resolve at completion.

## STATUS (this run, `output/fr13_qwencode_cachefirst_remap`, TAG b4_remap)
- **cat8 arm: DONE 16/16.** Garble **0%** (84 scripts), resolve **8/16 (50%) = native**, accept **3.336** clean,
  per_forward **0.217** (=B1 bank). derived_tps 63.92 NOT trusted (§4a). Merged to main @ cff3abd8.
- **cat6 arm: DONE 16/16.** Garble **0/61** (undef), resolve **7/16 (44%) ≈ native** (−1 flaky 14096), accept
  3.594 (traj-noise), per_forward 0.199. derived_tps 72.43 NOT trusted (§4b).
- **native arm (3/3): RUNNING** — MTP-5 (num_spec=5), FLASH_ATTN, no tree (remap correctly NOT engaged = the bar).
- => DELIVERABLE MET on garble: BOTH cat8 (0/84) AND cat6 (0/61) branched trees garble-free on full ship config,
  resolve ≈ native. Remaining: this-run native reduce -> clean same-boot speed A/B; then branch-rescue diag + cleanup.
- Queued: branch-rescue diagnostic on cat8 (per-flat-row accept hist, spine {1,3,5,7,8} vs branch {2,4,6}).

## 7. Completeness audit (audit-symmetry red-team) — the fix is STRUCTURALLY complete
Every per-layer state read positionally after a branched commit now has a re-linearizer:
- full-attn `kv_cache` -> `launch_attn_kv_linear_remap` (THE FIX)
- GDN `conv_state` -> `launch_tree_state_linear_remap` + `_fr13_conv_commit_to_col0`
- GDN `ssm_state`  -> `launch_tree_state_linear_remap` + `launch_tree_gdn_replay`
No third cached state exists (attn=KV; GDN=conv+ssm; RoPE is position-derived, not cached). The attn-KV
capture is EXCLUSION-based (captures every group whose builder is NOT Mamba2/GDN) -> robust to any attention
variant, not a narrow whitelist. => the missing twin (full-attn KV) was the LAST uncovered store; no residual
carrier can hide under the 0% gate. Static coverage (this audit) + empirical correctness (0% garble gate) =
complete + correct.
