# FR13_COMMIT_ARGMAX_BIND — in-process committer-row argmax gate (DRAFT, not committed)

Date 2026-06-13 UTC. Tree boot `fr13-forked-fa2-tree` cat9 / TREE_ATTN / num_spec=9,
**ENFORCE_EAGER=1**, all FIX default ON, `FR13_COMMIT_ARGMAX_GATE=1`. Probes =
`output/fr13_acceptance_ladder/prompts_swe4.json` (4 pinned), greedy seed 1313 top_p 1.0
max_tokens 128, rep1+rep2. In-process gate jsonl =
`output/fr13_commit_argmax/fr13_commit_argmax_gate.jsonl` (944 served records).

## CHANNEL VERDICT: **CHANNEL 2 (verify-forward losslessness gap).** Not channel 1.

The committer is **bit-faithful**: at EVERY served position with a real margin,
`committed_token_id == argmax(verify_logits[the exact flat row the committer indexed])`.
The gold-gate clear-margin flips are produced by the **tree verify forward itself**
computing argmax-flipping logits vs a clean single forward — the committer then correctly
serves that (wrong) verify-forward argmax.

## Within-boot determinism (class 8): PASS
rep1 == rep2 served ids byte-identical on all 4 prompts (`within_boot_det_rep1_eq_rep2 =
[true,true,true,true]`). The gate's own per-record ch1 results are deterministic: each of
the 5 unique zero-margin "violations" appears exactly twice (once per rep).

## Engagement: PASS
- Gate armed: boot log `FR13_COMMIT_ARGMAX_GATE committer-row gate: armed=1
  (armed-eager-only)`. Eager confirmed (`enforce_eager: True`), cat9 tree
  (`num_speculative_tokens: 9`, TREE_ATTN). 944 in-process committer-row records written.
- All `row_known=True`, `oob=False`: the committer indexed a valid per-node flat verify
  row at every served token (no `path0_native_bonus` legacy last-row case fired).

## CHANNEL-1 (committer row-mapping) — CLEAN at clear margin
944 served records. **0 clear-margin channel-1 violations.** 10 `ch1_match=false` records
exist but ALL are exact zero-margin argmax ties: `ch1_margin = 0.0`, `verify_top2_margin =
0.0`, and `verify_runnerup_id == committed_token_id` in every one (the committer's argmax,
computed at patch L6887 `target_logits.argmax`, and the gate's row-stats `topk` broke a
two-way logit tie the other way). These are gate self-noise, not a row bug. By node_type:
spine 654/6-ties, reject_correction 98/4-ties, leaf 58/0, bonus_self_spine 76/0,
bonus_self_leaf 58/0.

=> The committer's row map (`scripts/fr10_phase4_patch_vllm_tree_gdn.py` L5707-5859:
accepted prefix → `target_logits[start+node]`; reject/root → parent-edge target row;
tree-self bonus → `self_logits[start+leaf]`) is correct. FIX-A / eager-pack-replay /
conv-fusion-committed-path row selection are NOT serving wrong rows.

## CHANNEL-2 (verify-forward losslessness) — CONFIRMED, the real gap
At the two gold-gate clear-margin forks the committer is faithful (`ch1_match=true`) but a
CLEAN max_tokens=1 teacher-force on the byte-identical served prefix (same eager tree boot,
`output/fr13_commit_argmax/ch2_teacher_force_tree.json`) flips the argmax vs the captured
verify-forward argmax:

| fork | committed_row (flat) | node | node_type / na / accept_run | verify-forward argmax (served) | verify top-2 margin | CLEAN teacher-force argmax | clean top-2 |
|---|---|---|---|---|---|---|---|
| p2 pos21 (` code` vs ` files`) | `start+7` | node 7 (deep spine tail) | spine / na=5 / arun=4 | ` code` (1970) @24.625 | 0.5 over ` files`(3425) | **` files` (3425)** @-0.146 | ` code` @-2.021 (rank2) |
| p3 pos73 (`Let` vs `\`\`\``) | `start+7` | node 7 (deep spine tail) | reject_correction / na=4 | `Let` (9764) @25.375 | 7.25 over 44780 (`\`\`\`` not even rank-2) | **`\`\`\`` (71093)** @-0.158 | `Let` @-2.033 (rank2) |

Both flips land on the **deepest tree row (flat `start + node 7`**, the depth-4/5 spine
tail of `best_path=[0,1,3,5,7]`), at **code-fence / template boundaries** (p3 served
context `...| sort\n\`\`\`\n\n` then served `Let`; p2 `...locate the relevant ` then served
` code`). p2 pos15 (an earlier ` code`) clean-forces to ` code` => genuine, confirming the
gap is at pos21 not pos15 (consistent with the gold-gate fork-21).

The verify-forward argmax at the deep node-7 row is grossly wrong (p3: `\`\`\`` is ~85%
under a clean forward but absent from the verify row's top-2; the verify forward ranks
`Let` first). This is NOT a near-tie cross-backend numerics flip — it is a **verify-forward
logit divergence at deep tree rows** ([[feedback_math_correct_vs_bitexact]]: ℝ-correct but
not bit-exact, flipping argmax at structural boundaries).

## NAMED SEAM (do NOT fix here)
The tree **verify forward** (TREE_ATTN + fp8 GEMM + conv-fusion committed-prior window +
GDN recurrent scan + eager-pack replay), at the **deepest tree row** (flat `start + node 7`
= depth-4/5 spine tail), produces argmax-flipping logits vs a clean single forward. The
divergence concentrates at the deep accumulation point of the masked tree attention /
recurrent scan and surfaces at structural/template boundaries. Candidate sub-ops to localize
next (top-down per-layer logit ladder, [[feedback_top_down_per_layer_lossless_gate]]):
1. **TREE_ATTN deep-row attention** (ancestry-mask accumulation / online-softmax rescale at
   the deepest path row) vs FLASH_ATTN clean — the row-7 attention output diverges most.
2. **conv-fusion committed-prior window** (`FR13_CONV_COMMITTED_PATH`,
   patch L797-818 region) feeding the GDN scan at the deep node — wrong prior-window column
   at the deep tail row would corrupt that row's hidden state → lm-head logits.
3. **GDN recurrent-scan accumulation depth** (the row-7 state is the most-accumulated).

The decisive next instrument is a per-layer hidden/logit capture at flat row `start+node 7`
for these two steps (tree-boot step 62 served_pos4; step 103 served_pos4), comparing to a
clean teacher-forced forward at the same prefix — first nonzero layer at that row = root.
NOT channel 1; the committer row map is correct and needs no change.

## Artifacts (`output/fr13_commit_argmax/`)
- `fr13_commit_argmax_gate.jsonl` — 944 in-process committer-row records (the primary gate).
- `tree_capture.json` — served streams rep1+rep2 (within-boot det source).
- `ch2_teacher_force_tree.json` + `ch2_tf_probe.py` — clean teacher-force at the flips.
- `channel_split_summary.json` — violation table + channel-2 flip records.

## NOTE (boot ops): do NOT set `FR10_METRICS=1` with the gate. With empty
`LUMO_TREE_PATH_LCP_LOG` (launcher default ''), FR10_METRICS=1 makes the committer's
independent-winner log block `open('')` → FileNotFoundError → EngineDead (first boot died
this way at L6465 region). The gate jsonl is the instrument; metrics are not needed.
