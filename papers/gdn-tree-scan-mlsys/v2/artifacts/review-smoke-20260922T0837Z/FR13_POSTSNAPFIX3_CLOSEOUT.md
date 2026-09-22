# FR13 postsnapfix3 — 3-arm GPU verification closeout (2026-07-22)

**Campaign:** tail6 (SNAP_FIX-deletion verification) vs native-MTP5 control vs native-MTP11
depth-matched control. All cache-ON (`FR13_ENABLE_APC=1`), B=4, temp 0.6, qwen-code
nudge-free, live SWE-bench-Verified 16-task subset
(`output/fr13_b1_gold_swe/subset_b4_sixteen.json`). Run root:
`output/fr13_kvremap_tail6/{kvremap_tail6_kvr1,native5_control_kvr1,native11_control_kvr1}/`.

## Why this campaign existed

Two prior attempts (`sfv2`, `sfv3`, same day) showed severe repeat-loop/garble on tail6
cache-ON — up to 126 consecutive identical tool calls on one task. Root-caused:
`FR13_ATTN_KV_REMAP` and `FR13_SLOT_REORDER` — both previously proven and "baked" per
project docs — were defaulting OFF in the general B4 agentic campaign launcher
(`fr13_launch_forked_fa2_tree_server.sh`), despite being hardcoded ON in the narrow
`fr13_launch_locked.sh` and a dozen one-off diagnostic scripts. Weeks of tail6/cat8/cat6
campaigns through the general launcher ran without the fixes.

Fixed at the source this session:
- `scripts/fr13_required_tree_flags.sh` (new) — single source of truth for the two flags,
  sourced by all 3 tree-launcher entry points instead of each hardcoding its own copy.
- Fail-loud post-boot assertion gate added to the orchestration script.
- 17 dead one-off diagnostic scripts deleted.
- Committed `316c6390b`, pushed, fast-forward merged to `main`.

Also independently confirmed the SNAP_FIX/leaf-map deletion itself (commits `362133b64`,
`5fc2446ec`, `6ce55e972`, all 2026-07-22 ~02:25-02:28 UTC) is a no-op under the current
STATELESS-TREE (`runrow_commit=1`) config — it removes an already-dead redirect layer;
the stock `get_temporal_copy_spec`/`get_conv_copy_spec` branches are untouched.

## Results

| arm | resolve | accept/event | committed/event | measured decode TPS (prefill-indep) | wall-clock (16 tasks, B=4) | repeat-loop |
|---|---|---|---|---|---|---|
| tail6 (SNAP_FIX-deleted, fixes baked) | 10/16 (62.5%) | 4.286 | 5.286 | 32.85 tok/s | 3h09m45s | 0/16 |
| native5 (MTP-5 control) | 10/16 (62.5%) | 3.422 | 4.422 | 42.74 tok/s | 2h10m21s | 0/16 |
| native11 (MTP-11 depth-matched control) | 8/16 (50.0%) | 4.923 | 5.923 | 39.95 tok/s | 3h07m58s | 0/16 |

Historical base rate for this exact 16-task subset (all prior `eval_report.json` for these
instance IDs, across `output/`): ~28%. All three arms clear that comfortably.

## Verdict 1 — SNAP_FIX deletion verified behavior-preserving

Zero repeat-loop instances across all 48 task-runs (3 arms × 16 tasks) — a clean sweep,
versus the pre-fix campaign's severe garble (`sfv3`: 7/16 tasks with maxrun≥3, one at 126).
Confirms both (a) the deletion didn't introduce a regression, and (b)
`FR13_ATTN_KV_REMAP`+`FR13_SLOT_REORDER` were the real fix for the garble contaminating
this campaign for weeks.

## Verdict 2 — tail6's historical "accept~5" figures were garble artifacts

Investigated via vLLM's own `Per-position acceptance rate:` telemetry. Within the SAME
pre-fix code/config, a time window dominated by a known-garbled task (14598, 126
consecutive identical tool calls) showed deep-tail acceptance 40-124% higher than a
healthy window in the same run — direct evidence that degenerate repeat-loop trajectories
inflate the accept metric (repetitive continuations are trivially predictable, so the
drafter "accepts" its own corruption). Every historical near-5 tail6 number checked was
either cache-OFF or garble-contaminated. `kvremap_tail6_kvr1`'s 4.286 is the first clean,
garble-free cache-ON accept number for this tail geometry.

## Verdict 3 — tail6 loses to both native controls on raw decode speed: committer/verify tax, not accept

Per-event GPU component breakdown (ms/event; sums match `wall_s_per_event` closely):

| component | tail6 | native5 | native11 |
|---|---|---|---|
| verify | 106.28 | 66.68 | 73.95 |
| drafter | 35.04 | 31.72 | 68.26 |
| committer | 16.89 | 2.49 | 3.84 |
| host-other | 2.67 | 2.55 | 2.24 |
| **total** | **160.88ms** | **103.45ms** | **148.28ms** |

tail6's committer is 4-7x either native arm's — it commits a non-contiguous tree path
(GDN/conv replay per layer) vs a straight linear copy. Despite committing more tokens per
event than native5 (+19.5%), the +55% per-event cost overwhelms it.

native11 is the interesting middle case: highest accept of the three (real per-depth MTP
heads, not tail6's heuristic arctic-tail fill) and second-fastest decode — but its drafter
cost is more than double native5's (68.26ms vs 31.72ms, proportional to 11 sequential head
positions vs 5), so it still loses to native5 on raw tok/s. It also has the **lowest**
task-resolve rate of the three (50% vs 62.5%/62.5%) despite the best token-level numbers —
token-level speed/accept doesn't automatically predict agentic task success.

## Verdict 4 — per-position acceptance: where tail6's branching wins, where it doesn't

| depth | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tail6 | .973 | .822 | .666 | .535 | .444 | .230 | .180 | .152 | .134 | .120 | .111 |
| native5 | .865 | .729 | .618 | .528 | .459 | — | — | — | — | — | — |
| native11 | .850 | .706 | .591 | .503 | .435 | .379 | .337 | .301 | .274 | .249 | .226 |

- **Depths 0-2**: tail6's branching head (3-wide comb, `[(0,),(1,),(2,)]` etc. at each of
  depths 1-5) beats native5 outright — sibling candidates genuinely rescue mismatches.
- **Depths 3-4**: margin shrinks to parity then inverts slightly (-0.015 at depth 4).
  Leading hypothesis, not yet directly verified: numerical drift compounding across the
  tree kernel's batched/padded attention path (BV=8 packing, KV_REMAP re-linearization,
  SLOT_REORDER permutation), consistent with the project's prior finding that the tree
  kernel drifts ~7x more per-layer than native's clean path.
- **Depths 5-10** (tail6's arctic tail): tail6 is roughly HALF native11's rate at the same
  depth (depth 10: .111 vs .226). A genuine model-quality gap — tail6's tail runs a cheap
  heuristic suffix predictor (`FR13_DRAFT_SOURCE=merged`), native11 uses real trained
  MTP-11 heads at every depth.
- Structural note: every sibling node in tail6's tree is a dead-end leaf (no children) —
  only the pure primary path continues past depth 5, so reaching the tail at all requires
  5-for-5 primary hits.

## Open paths to beat both native controls (not yet built)

1. **Speed** — R3 (verify-tree-tax trim, 103→~70ms) and R4 (drafter CUDA-graph capture,
   106→~50ms), tasks #49/#50, already queued, target exactly tail6's committer/verify tax.
   If both land near target, tail6's per-event cost drops toward ~125ms, pushing measured
   TPS from 32.85 toward ~42-43 — closing the gap to native5 without touching accept.
2. **Accept** — replace the arctic tail's heuristic merged-draft source (depths 6-11) with
   genuine deeper MTP-style heads or a better-trained suffix predictor, closing the gap to
   native11's per-position rate at the same depths. Would let tail6 combine its proven
   shallow-depth branching advantage with native11-grade deep-depth acceptance.
3. Unresolved: whether the depth 3-4 margin-inversion in the branching head is a fixable
   numerical-drift bug or an inherent tree-kernel cost — needs a same-seed primary-only
   bit/argmax diff vs native5 per depth to localize.

## Artifacts

Full per-task traces, patches, eval reports, `deploy_speed*.json`, `container_env.txt`,
docker logs for all 3 arms under `output/fr13_kvremap_tail6/`. Committed in full to git
(force-added past the `output/` gitignore, per explicit user directive) alongside this
closeout.
