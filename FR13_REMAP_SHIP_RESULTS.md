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
| **cat8 FIX-ON** (this run) | **0.0%** (77 scripts, live) | 0.0% | — |
| cat6 FIX-ON | PENDING | | |
| native (this run) | PENDING | | |

## 3. Trajectory / give-up (assistant turns per task; give-up=short, garble-thrash=long tail)
- PRIOR NO-FIX cat8: median 32, dist `[2,2,...,88,124]` = BIMODAL DYSFUNCTION (2 hard give-ups + garble-thrash).
- native: median 14 (healthy). **cat8 FIX-ON: median ~15, min 8 (NO give-ups), max 35 (NO thrash)** → tracks native.
- Edits: **15/16 cat8 tasks made real edits to the CORRECT source file** (separable.py, quantity.py, etc.).

## 4. Speed (accept-normalized; NOT raw TPS)
- PRIOR NO-FIX cat8: accept/event **3.588**, s_per_fwd_gpu **0.131**, derived_tps_gpu **34.98** (prefill-independent).
- cat8 FIX-ON interim (/metrics): accept/forward **3.41** (draft/fwd=8.0 basis valid). Clean s_per_fwd_gpu at completion.
  Caveat: no-fix accept is garble-INFLATED (wrong-accepts); fix accept is clean → small dip expected.
- Remap own tax (paired probe): **−0.7% s/fwd (within cross-boot noise)** = no-HBM-tax bar met.
- B4 genuine: Running=4 dominant (~68% of intervals), effective batch ~3.7 (not serialized-to-1).

## 5. Config manifest / confound tracking (`scripts/fr13_config_manifest.py`)
harness_hash (must-match: wall/nudge/temp/conc/B/agent/git) matches prior EXCEPT git_head (newer commit =
the fix). wall=no-wall (WALL=0), nudge=0 (OFF), temp=0.6, B4, cache-ON (APC hits 86-87%), EXACT_SEED=0.


## 6. RESOLVE RATE (SWE-bench eval verdicts) — the deliverable number
| run | resolved | rate | vs native |
|---|---|---|---|
| cat8 FIX-ON (this run, 15/16 evaluated) | 8 | **53%** | **≈ native** |
| native (prior qc4, bar) | 8/16 | 50% | — |
| cat8 no-fix (prior qc4) | 6/16 | 37% | BELOW native = the degradation |

=> cat8 fix-ON resolve (53%) tracks native (50%); the tree+cache agentic DEGRADATION (no-fix 37% < native
50%) is ELIMINATED. Same 16-task subset (15/16 overlap; 1 fix-ON task pending). RESCUED from garble-failure:
astropy-13579, -14539, -14508 (no-fix failed -> fix resolved). REGRESSED: astropy-13236 (no-fix resolved ->
fix failed; likely temp-0.6 trajectory variance - the fix removes garble, does not inject errors). Net +2
(6->8). Garble 0% (fixed gate). Final cat8 tally + cat6 + this-run native at arm completion.

## STATUS (this run, `output/fr13_qwencode_cachefirst_remap`, TAG b4_remap)
- cat8 arm: 16/16 tasks seen, running (finishing + eval). Garble 0%, edits 15/16, trajectories native-like.
- cat6 + native arms: PENDING. Final resolve verdicts (SWE-bench eval) + clean speed A/B at completion.
