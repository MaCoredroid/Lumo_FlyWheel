# FR13 replay route — BUILD record (branch `fr13-replay-route`, 2026-06-10)

Implements the verified full-route design: cut GDN tree state traffic from
36 rows/forward (21.74 GB, 5.14x actual native) to 6 rows (3.62 GB, 0.86x
native) by deleting the scan's per-node HBM export and re-executing the
committed accepted path from a tiny activation ring (~16.2 KiB/node vs the
3.146 MB state row), publishing directly to LINEAR bank columns.

## Design references
- `research/fr13_workflows/replay_full_1x_w78aq6xum.raw.json` — the
  verify-passed design (traffic accounting, pure-export proof, ±0.0 handoff
  proof, Option-1 prev-lens gap, zero-accept row-0 correction).
- `FR13_ACCEPT_ONLY_GATE4_FAIL_BIND.md` — gate-4 root causes bound into this
  build: (1) zero-accept row-0 publish path implemented in-kernel (root
  replays unconditionally into column 0); (2) NO dict-pinned per-step
  buffers — persistent preallocated staging only. The failed accept-only
  patch stays PARKED on `fr13-accept-only-wip`; its dict mechanism is NOT
  reused (test-enforced), its gate-script ideas inform the GPU gates below.
- `project_fr13_conv_priorwindow_root` — the conv remap half is the FR13
  conv-prior-window carrier; it is KEPT (only the ssm half is replaced).

## What was built (commits, all `FR13 replay-route:` prefixed)
1. `c2e84054` kernel — `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py`
   - `_gdn_node_step`: the per-node GDN rank-1 update factored into ONE
     shared `@triton.jit` body, inlined by BOTH kernels with identical
     constexprs (`DIM_K=128`, `BLOCK_V=16`, `OUTPUT_SCALE`,
     `USE_QK_L2NORM_IN_KERNEL`, `RAW_GATING`) and `num_warps=8`.
   - `_tree_gdn_kernel` + `STORE_NODE_STATES: tl.constexpr`: guards the
     per-node HBM export (pure export — children resume from `h_cache`
     registers; nothing in-kernel reads the store). Default `True`.
   - `_tree_gdn_replay_kernel`: chain replay, NO `h_cache` (one
     `(BLOCK_V, DIM_K)` register tile, spill-free at any tree size),
     `tl.static_range(0, PATH_COLS+1)` with `t-1 < accepted_len` masking,
     indirect node load from `accepted_paths`, recompute of
     decay/beta/l2norm from ring activations, `state = state + 0.0` per
     post-root parent-handoff edge (the scan's masked-sum `-0.0 -> +0.0`
     flip; the root consumes h0 UNnormalized), h0 from the SCAN-TIME
     prev-lens snapshot column (`clamp(prev_len-1, 0)`), h0 tile read
     before any store (publish-overwrites-h0-row safe), post-step states
     stored to bank LINEAR column `t-1` (root always to column 0 = the
     zero-accept publish). Native gate-folding basis — no rescaled-exp.
   - `launch_tree_gdn_replay`: validating launcher (fp32 bank, contiguous
     persistent rings, `PATH_COLS <= SPEC_COLS`, `RAW_GATING=True` pinned).
   - `launch_tree_gdn_prepared(store_node_states=False)`: skips the
     201.3 MB/layer scratch alloc (capture-blocking), dummy state pointer,
     returns `(out, None)`; raises if a state buffer is passed anyway.
2. `ff9e90e8` ring+glue — `scripts/fr10_phase4_patch_vllm_tree_gdn.py`
   - Forward (gdn_linear_attn replacement), under `FR13_REPLAY_ROUTE=1`:
     scratch alloc skipped; scan launched with `store_node_states=False`;
     PERSISTENT per-layer activation ring (`B_max x N_PAD x heads`, B_max =
     the request-keyed accepted-lens buffer batch) byte-copies of k
     pre-l2norm / v (value_tree) / raw_a / raw_b at consumed precision;
     prev accepted lens + spec indices SNAPSHOTTED AT SCAN TIME into
     persistent buffers (the committer refills
     `_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR` BEFORE its publish block — the
     verify-rider Option-1 gap); layer registered in `_FR13_REPLAY_LAYERS`;
     all-rows `ssm_state.index_copy_` publish SKIPPED; tree_state-consuming
     diagnostics (`FR10_TREE_GDN_CAPTURE_PAYLOAD`,
     `FR10_TREE_GDN_COMMIT_HANDOFF_LOG`, `FR10_TREE_GDN_SRC_NATIVE_PAYLOAD`,
     `FR12_TREE_SCAN_NATIVE_SPINE`) raise loudly; diag[12]/[13]
     (`scan_state_staging`) skipped = explicitly VACUOUS under the flag (do
     not gate on it; gate on the replay A/B + durable-state diff instead).
   - Committers (greedy `_lumo_tree_path_lcp_max_greedy_sample` + sampled
     `_lumo_tree_canonical_multidraft_sample`): after the accepted
     paths/lens device-buffer refill + REQKEY publish, launch
     `launch_tree_gdn_replay` per registered layer with fresh-staging and
     row-count asserts (inside the committers' existing fail-loud try).
   - Next-step remap: `ssm_state=None` under the flag (replay already wrote
     linear columns; the node-column remap would corrupt); conv half KEPT.
3. `406ab614` tests — reference module + CPU battery + GPU A/B scaffold
   (details below).

## Flag matrix
| Flag | Default | Behavior |
|---|---|---|
| `FR13_REPLAY_ROUTE` | **0 (OFF)** | Unset/0 = legacy path (all-rows publish + ssm remap; scratch alloc; STORE_NODE_STATES=True), **source-inert**: every new behavior is flag-gated and the legacy text is intact, but the shared `_gdn_node_step` body refactor RECOMPILES the default scan, so flag-OFF **compile-identity is pending the refactored-scan byte A/B (GPU TODO #2)** — NOT claimed byte-identical until that A/B passes. 1 = replay route. |
| `FR13_TREE_REQKEY` | 1 (ON) | Unchanged; the replay reuses the request-keyed persistent paths/lens buffers (within-step positional fill is what the replay consumes, snapshot covers the cross-refill hazard). |
| `FR13_TREE_REMAP_SEQ` | 1 (ON) | Unchanged; under the replay route the ssm half is not launched at all, conv half still uses the race-free gather remap. |
| `FR13_TREE_PER_REQ_GEN` | 1 (ON) | Untouched (sampler rng; orthogonal). |
| `FR10_TREE_GDN_CAPTURE_PAYLOAD` / `FR10_TREE_GDN_COMMIT_HANDOFF_LOG` / `FR10_TREE_GDN_SRC_NATIVE_PAYLOAD` / `FR12_TREE_SCAN_NATIVE_SPINE` | off | RAISE if set together with `FR13_REPLAY_ROUTE=1` (they consume/splice the deleted scratch). Capture payloads with the flag OFF. |
| `FR10_METRICS` diag[12]/[13] | off | VACUOUS under the replay route (skipped, stays at zeros-init) — `fr10_serving_wiring_gate.py` now REFUSES to run (raises) with `FR13_REPLAY_ROUTE=1` instead of silently passing the vacuous `scan_state_staging` check (see AMENDMENT #3). |

## CPU-PROVEN (all green on CPU torch 2.4.1, no triton; 23 new tests + 15 existing after the AMENDMENT remediation)
- `tests/test_fr13_replay_reference_bitexact.py` — 9 passed:
  - `test_handoff_masked_sum_equals_plus_zero_and_flips_neg_zero` — the
    kernel's masked-sum handoff == `+0.0` emulation, bit-level, and the
    flip is real (non-vacuous).
  - `test_replay_chain_matches_scan_chain_all_paths_deployed_tree` —
    bitwise (int32-view) identity of every replay step vs the scan's
    h_cache states, every root-to-node path, deployed 10-node tree + bushy
    14-node tree, bf16-roundtripped activations.
  - `test_replay_bank_columns_match_legacy_publish_plus_remap` — final
    linear-column contents == legacy publish+remap consumed columns.
  - `test_zero_accept_replays_root_into_column_zero` — empty accepted path
    publishes exactly the root state to column 0.
  - `test_replay_chain_every_accepted_len_on_spine` — accepted_len 0..MAX.
  - `test_neg_zero_in_parent_state_handoff_is_bit_exact` — crafted -0.0
    surviving the root update, handoff flip identical in both chains.
  - `test_node_step_softplus_branch_and_q_independence` — x<=20 branch both
    sides of 20.0; zero-q/real-q/no-q produce bit-identical STATE (licenses
    the replay's discarded-out zero-q).
  - `test_activation_ring_bf16_roundtrip_is_byte_exact` — ring store/load
    byte identity (int16 view) + exact fp32 widen, incl. -0.0/denormal/max.
  - `test_replay_differs_without_plus_zero_normalization` — dropping the
    +0.0 BREAKS identity (the normalization is load-bearing).
- `tests/test_fr13_replay_route_wiring.py` — 13 passed (remediated battery,
  see AMENDMENT): flag defaults OFF at every read; kernel export gate +
  shared body single-source (called by both kernels, identical constexpr
  plumbing, num_warps=8 on the replay); scratch alloc/publish/diag
  flag-gated with legacy text intact; scan-time snapshot precedes the
  launch and the committer refill precedes the replay launch; staging
  allocated at METADATA-BUILDER INIT (forward only writes, raises if
  missing, no allocation calls in the flagged path); broadened per-step
  object-creation ban in the flagged forward path (not just
  `_FR10_PENDING_TREE_STATE_PUBLISH` by name) + capture-safe
  `_fr13_replay_flags` handshake asserted; `fr10_serving_wiring_gate.py`
  refuses loudly under the flag; replay launch in BOTH committers after the
  REQKEY block; remap conv half kept / ssm half None; embedded patch
  fragments (incl. the builder-init allocation fragment) AST-parse;
  `py_compile` of patcher + kernel + reference + gate script.
- `tests/test_fr10_phase4_sampled_committer_wiring.py` — 15 passed
  (unchanged; flag default OFF preserves all legacy surfaces).
- `tests/test_fr10_serving_wiring_gate.py` — 9 passed (8 existing + 1 new:
  `evaluate_wiring` raises with FR13_REPLAY_ROUTE=1, runs with it unset/0).
- Full suite: 968 passed, 8 skipped, 31 failed — the 31 failures are
  byte-identical to unmodified main HEAD `8587396d` (diffed failure lists:
  IDENTICAL), i.e. pre-existing and unrelated.
- `py_compile`: kernel module, patcher, reference, all new tests OK.

## NOT CPU-PROVABLE — GPU TODO (declared, NOT claimed)
1. **Triton codegen identity byte A/B** (THE bit-exactness gate): one-time
   byte A/B on captured payloads via the `FR10_TREE_GDN_CAPTURE_PAYLOAD`
   harness (capture with flag OFF), all 48 layers, all accepted lens,
   including a -0.0-in-parent case, a publish-dst==h0-row case, and a
   zero-accept case. Unit-level scaffold ready:
   `tests/test_fr13_replay_gpu_byte_ab.py` (5 cases, skip-on-cpu).
2. **Refactored-scan A/B**: the scan body was factored into `_gdn_node_step`
   (inlined; expected identical) — byte-compare the refactored scan's
   out/state vs pre-refactor captures before trusting ANY flag-OFF serving
   number (protects the live lossless chase).
3. **Live serving gates** (B=1 bit-identical repeat; durable accepted-state
   diff=0 vs legacy on CONSUMED columns 0..max(len-1,0) — unconsumed
   columns legitimately differ, legacy leaves node-state bytes there;
   regular-decode == pristine; the live single-step ordering probe from the
   gate-4 lesson; B=4 corruption gate; measured TPS/traffic).
4. **LIVE zero-accept gate**: force/observe a zero-accept event in live
   serving and assert the NEXT event h0 read consumes the replayed root
   state (column 0).
5. **CUDA-graph capture** of the replay kernel + the no-scratch forward
   (the alloc removal is the capture-blocker fix; verify capture, then the
   final regime = B=4 CUDA-captured SWE-4 vs E5 per standing policy).
6. **ptxas spill check**: replay kernel spill-bytes==0 (and the scan at
   N_PAD=16/num_warps=8 — the standing FR13_CACHE_SCALING_FUTURE gate).
7. **Preemption/resume invariant**: "no pending replay => bank state is
   final" — re-read live vLLM preemption source first (read-source-first).
8. Container deployment detail: the live image must ship this updated
   `fr10_gdn_tree_kernel.py` (the patcher imports `launch_tree_gdn_replay`
   inside the committer only under the flag, so stale images fail loudly,
   not silently).

## Traffic target (from the verified accounting, to be measured at gate time)
CURRENT 36 rows = 21.74 GB/fwd (scratch 9 + publish 18 + remap 2a + h0 1,
a=4) -> replay route 2+a = 6 rows = 3.62 GB/fwd = 0.86x actual native E5
(4.23 GB); replay FLOPs ~5-33 us vs the ~99 ms weight-bound forward.

## AMENDMENT (2026-06-10) — verify violations acknowledged + remediated
Workflow `w89pmmka9` verify returned holds=FALSE on the original build
(commits `c2e84054`/`ff9e90e8`/`406ab614`/`6929bdb9`). The gate-4 review
violations are acknowledged here and remediated on this branch:

1. **Lazy allocation was a gate-4-class capture landmine (NOW FIXED).** The
   original build allocated the activation rings + prev-lens/spec-idx
   snapshot buffers LAZILY on the first flagged forward
   (`if getattr(self, "_fr13_replay_ring_k", None) is None: torch.zeros(...)`
   inside the forward). Under CUDA FULL capture, a first-flagged-forward
   allocation inside the captured region = stale-pointer aliasing — exactly
   gate-4 root cause #2, the very failure mode this build claimed to bind.
   FIX: all replay staging buffers are now allocated at GDN
   METADATA-BUILDER INIT (mirroring the persistent accepted-paths/lens
   buffer pattern in the same `__init__`), sized `B_max x N_PAD`
   (`B_max = max(decode_cudagraph_max_bs, max_num_seqs)`); the forward only
   WRITES and RAISES if the buffers are missing (no fallback allocation).
2. **`_fr13_replay_meta` was a NEW per-step Python dict handshake (NOW
   FIXED).** Replaced with a capture-safe persistent mechanism: a
   preallocated per-layer int32 flag tensor `_fr13_replay_flags`
   (`[0]`=fresh, `[1]`=staged spec-decode rows) written by CAPTURED device
   ops in the forward — so a CUDA-graph REPLAY re-arms freshness even
   though the Python in the captured region never re-runs — and cleared by
   the committer after the replay launch; `output_scale` is a fixed
   init-time attribute; the ssm bank ref is a fixed attribute written per
   step without object creation; layer registration in
   `_FR13_REPLAY_LAYERS` moved to builder init. NO per-step dict/object
   creation remains in the flagged replay path (test-enforced, broadened
   beyond the old `_FR10_PENDING_TREE_STATE_PUBLISH`-by-name ban).
3. **diag[12]/[13] vacuity left `scripts/fr10_serving_wiring_gate.py`
   unmodified (NOW FIXED).** The original build declared diag[12]/[13]
   vacuous under the flag but did not touch the gate script, so its
   `scan_state_staging` check would have PASSED silently-vacuous with
   `FR13_REPLAY_ROUTE=1` — the gate-transfer matrix rider's "silently
   vacuous" item. FIX: the gate now FAILS LOUDLY (raises, in both
   `evaluate_wiring` and `main`) when `FR13_REPLAY_ROUTE=1` instead of
   emitting a vacuous PASS; gate the replay route on the replay byte A/B +
   durable accepted-state diff instead.
4. **Flag-OFF claim overreach (DOWNGRADED).** The original text claimed
   flag-OFF is BYTE-identical unconditionally. The shared `_gdn_node_step`
   body refactor RECOMPILES the default scan, so the honest claim is:
   flag-OFF is **source-inert**; **compile-identity is pending the
   refactored-scan byte A/B (GPU TODO #2)**. Downgraded in the flag matrix
   above and in the wiring tests.
5. **GPU TODO was missing an explicit LIVE zero-accept gate (ADDED).** See
   GPU TODO #4: force/observe a zero-accept event in live serving and
   assert the NEXT event h0 read consumes the replayed root state
   (column 0).
