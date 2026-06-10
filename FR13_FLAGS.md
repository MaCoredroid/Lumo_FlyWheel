# FR-13 flag reference — which flags wire our code vs do drift-measure

Authoritative source: `scripts/fr13_launch_forked_fa2_tree_server.sh` (tree arm) + `scripts/fr10_launch_speed_server.sh` (native/speed arm). This doc just documents them clearly (no refactor). Captures default empty/off; the deliverable speed run turns them OFF.

## A. WIRE-OUR-CODE — always ON for the tree-verify deliverable
These select OUR verify path (forked FA2 tree-bias + GDN tree-kernel + MTP drafter). Always on for both the drift gate and the speed run.

| flag | value | what it wires |
|---|---|---|
| `--attention-backend` | `TREE_ATTN` | the tree attention backend (carries the tree ancestry bias) |
| `FR13_FA2_TREE_BIAS` | `1` | route TREE_ATTN through the **forked FA2** with the `-inf` ancestry bias (the kernel) |
| `FR13_TREE_ATTN_EXP2_SOFTMAX` | `1` | base-2 (`exp2`) softmax in tree attn, matching FA2 |
| `FR10_ENABLE_TREE_GDN` | `1` | the **GDN tree kernel** (`_tree_gdn_kernel` ancestor-replay + tree-aware conv) |
| `FR10_DECODE_MODE_DEFAULT` | `tree_mtp` | decode mode = tree verify (vs `naive_mtp`/`non_mtp`) |
| `FR11_TREE_CONV_NATIVE_BF16_TAPS` | `1` | conv tap-dtype alignment (FR11) |
| `FR12_TREE_CONV_NATIVE_BF16_TAPS` | `1` | conv tap-dtype alignment (FR12) — **note: the GATE-A residual is a remaining 1-ULP in this manual conv vs native `causal_conv1d_update`** (`FR13_GATEA_DEEP_DIVERGENCE.md`) |
| `FR13_CONV_COMMITTED_PATH` | `1` | committed-path conv prior window: the next event's prior window is gathered from the accepted path's LEAF NODE column (node-indexed, BEFORE the in-place remap), so BRANCH winners ([0,2], [0,1,4]) commit a window built from committed-path tokens only (m1 seam 1, `FR13_S1S2S3_DISCRIMINATE_BIND.md`). Spine winners byte-identical to legacy (tested). `=0` restores the legacy post-remap linear-column read |
| `SPEC_CONFIG` | `{method:qwen3_5_mtp, num_speculative_tokens:N, speculative_token_tree:TREE}` | the **MTP-5 drafter** + the tree topology (spine=path0=the MTP chain) |
| `VLLM_BATCH_INVARIANT` | `0` | batch-invariance off (speed). **For B=4 drift this matters** — #42960 batch-dependence; the conv fix must be batch-invariant regardless |
| forked `.so` + patchers | — | copy `_vllm_fa2_C.abi3.so`; run `fr10_phase4_patch_vllm_tree_gdn.py` + `fr13_patch_fa2_tree_bias.py --skip-source` at launch |

**FOOTGUN — must stay UNSET:** `FR10_ALLOW_LINEAR_FALLBACK`. The launcher explicitly `unset FR10_ALLOW_LINEAR_FALLBACK` (line 144). Set ⟹ GDN silently falls back to LINEAR ⟹ DIAGNOSTIC ONLY, never a valid gate result, never bound to a commit.

**FOOTGUN — must stay `0` for serving/gates:** `FR13_FORCE_SPINE_COMMIT` (default OFF). Same class as `FR10_ALLOW_LINEAR_FALLBACK`: the greedy committer scores ALL paths (alts verified in `path_scores`) but always COMMITS the spine path's own prefix, so alt paths never win. Exists ONLY for the S3/m1 decisive A/B (caterpillar forced-spine vs chain-only boot, `FR13_S1S2S3_DISCRIMINATE_BIND.md`): if forced-spine next-event drafts match the chain boot token-for-token, the m1 contamination is entirely in the branch-commit state advance. NEVER bind `=1` into a committed serving config or a gate result; the sampled (temp>0) committer fails loud if it sees the flag. Log rows carry `forced_spine_commit` + policy `greedy_tree_lcp_max_FORCED_SPINE_DIAGNOSTIC` so captures self-describe.

## B. DRIFT-MEASURE — capture flags (set per measurement; default empty/off)
Top-down ladder vs E5 (eager, hooks ON). All point to a `/logs/*.pt` path; each has `_NUM_TOKENS/_SKIP/_LIMIT` (+ `_LAYER_PREFIX`/`_ROWS`/`_Z`/`_INPUT` where noted).

| flag | captures |
|---|---|
| `FR13_PREPROCESS_INPUT_CAPTURE` | verifier INPUT hidden / embeds (stage 0) |
| `FR10_LAYER_HIDDEN_CAPTURE` (`_ROWS`) | per-layer `hidden`+`residual` (all 64 layers) — the per-layer drift curve |
| `FR12_FULL_ATTN_CAPTURE` (`_LAYER_PREFIX`) | full_attn stage outputs (qkv/rope/attn_out/o_proj) for one layer |
| `FR12_SUBKERNEL_CAPTURE` (`_LAYER_PREFIX`,`_INPUT`,`_Z`) | GDN sub-ops: input_hidden, pre_conv, conv1d_out, h0_state_in, scan_out, gate_z, gate_out, o_proj_out |
| `FR10_SPINE_LOGIT_CAPTURE` / `FR13_FINAL_LOGIT_CAPTURE` (`_ROWS`) | spine / final logits |
| `FR13_TREE_ATTN_OP_CAPTURE` / `FR13_FLASH_ATTN_OP_CAPTURE` (`_LAYER`) | low-level attn op q/k/v/out (for the FA2-on-path oracle) |
| `FR10_METRICS` | `1` for measure (diag counters); `0` for the speed run |
| `FR12_TREE_CONV_STATE_FULL_CAPTURE`, `FR12_NATIVE_SPINE_ORACLE`, `FR12_TREE_CONV_NATIVE_SPINE`, `FR12_TREE_SCAN_NATIVE_SPINE`, `FR12_TREE_CONV_NATIVE_PRIOR_READ` | conv ring-buffer full state + native-spine conv/scan oracle (distinguish conv-state vs conv-kernel) |

Reducers (offline, no GPU): `scripts/fr13_gdn_subop_diff.py` (per-stage tree-vs-native), `fr13_fa2_tree_path_ref.py` (FA2-on-path oracle), the layer-hidden ladder, `fr13_fa2_no_bias_pristine_compare.py` (gate-2).

## C. The two strict gates (flag config)
- **Gate 1 VERIFY-PATH** (drift→0 vs E5, spine+branch): wire flags (A) ON + measure flags (B) ON, `FR10_ALLOW_LINEAR_FALLBACK` UNSET, eager. Compare tree vs native (`naive_mtp` FLASH_ATTN). Across **eager-B1 / eager-B4 / graphed-B4** (graphed-B4 is capture-free → e2e). See `FR13_GATEA_DEEP_DIVERGENCE.md`.
- **Gate 2 REGULAR-DECODE** (verifier-only): forked `.so` vs **pristine** `.so`, plain decode, no tree/spec/bias. `fr13_fa2_no_bias_pristine_compare.py` → 0.0 every layer. Bound `d2f1ba18`.

## D. SPEED / e2e regime — captures OFF (run once Gate A passes)
All capture flags (B) empty/unset; `FR10_METRICS=0`; `VLLM_BATCH_INVARIANT=0`; **B=4** (`MAX_NUM_SEQS=4`), **CUDA-graph** (no `--enforce-eager`). Measure via vLLM `/metrics` (accept/event) + bag-TV on generated tokens vs E5 (`output/fr10_native_mtp5_same8_*`). This run is the **graphed-B4 drift proof** AND the deliverable (lossless ≤ floor + superset accept/event ≥ 3.076). **Gated on Gate A = drift 0 across all three regimes.**

## Comparison structure + action items (user 2026-06-08)
- **Two servers:** (1) TREE = `scripts/fr13_launch_forked_fa2_tree_server.sh` (our forked FA2 + tree GDN + ex2 conv, `ATTENTION_BACKEND=TREE_ATTN`, `tree_mtp`); (2) NATIVE drift-reference = `scripts/fr10_launch_speed_server.sh` run as `naive_mtp` (`FLASH_ATTN`, native MTP-5). Drift gate = tree vs native-arm; e2e deliverable = tree vs the saved **E5** baseline `output/fr10_native_mtp5_same8_20260604T210257Z` (`naive_mtp`, B=4).
- **ACTION (validity):** the drift native-arm (B=1) and the E5 baseline (B=4) are SEPARATE runs; E5's artifact does NOT record its backend/`--gdn-prefill-backend`. **Pin E5's exact launch config (git history / launch log) and make BOTH the tree run and the drift native-arm match it** — else "lossless to native-arm" ≠ "lossless to E5". Both current launchers use `--gdn-prefill-backend triton` (= GDN linear-attn prefill kernel, separate from full-attn FA2).
- **ACTION (naming):** rename `TREE_ATTN` → `FA2_TREE_ATTN` (our forked FA2, NOT Triton tree-attn) across launcher/code/docs — the label misled framing. Part of the cleanup.
