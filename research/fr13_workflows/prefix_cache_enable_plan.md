# Prefix caching ENABLE + lossless-verify plan (SGLang-informed) — 2026-06-19

Builds on `prefix_cache_gdn_hybrid_research.md` (the WHY-off / YELLOW report). Goal: enable
prefix caching for Qwen3-Next-27B fp8 GDN-hybrid on our vLLM tree-spec serve and LOSSLESSLY
verify it. The codex+SWE deploy re-prefills its append-only ~11–14k context every turn, so APC
is a big prefill lever.

## A. SGLang reference (MambaRadixCache) — what to borrow
SGLang supports APC for Qwen3-Next GDN-hybrid via `MambaRadixCache` (~v0.5.5): FULL SSM-state
snapshot at radix-tree nodes (not block-table). Match = copy the deepest prefix node's state into
a fresh per-req buffer; insert = fork a checkpoint per node; dual-LRU evict (KV leaf-to-root,
state elastic). `extra_buffer` impl (ping-pong mamba slots) is the spec-decode-compatible one.
- **BORROW #1 — fp32 state cache by default.** SGLang ships `--mamba-ssm-dtype float32` BY DEFAULT
  ("float32 gets more accurate results, default float32"); bf16 is cold-vs-warm lossy. vLLM's
  `--mamba-ssm-cache-dtype` defaults to `auto`/bf16 (less safe). → set **float32 from the start**
  of our gate (matches vLLM #26807).
- **BORROW #2 — the conv-non-invertibility frame as a VERIFY TARGET.** SGLang #25587 "Hybrid-GDN MTP
  not lossless" root-causes conv1d+SiLU being non-invertible ⇒ a position-shift rollback can't
  reconstruct the K-1 window after accepting k tokens; fix (NVIDIA #10335) = per-step intermediate
  conv-window snapshots. This is the SAME carrier as our `project_fr13_conv_priorwindow_root` and
  our `FR13_CONV_COMMITTED_PATH` is the same shape of fix. NOTE: #25587 ran with radix-cache OFF →
  this conv carrier is **spec-decode-intrinsic, present with OR without APC**. Gate must re-prove
  `conv1d_out row-0 = 0.0` under APC's reconstructed window.
- **BORROW #3 — measure hit-rate.** SGLang radix keeps many checkpoints; vLLM align keeps ONE
  (last block boundary), so we must measure hits + tune `mamba_block_size` (no automatic density).
- DON'T borrow the architecture (SGLang-internal; we serve on vLLM). Transferable = fp32 dtype +
  measure-hit-rate + verify-conv-window disciplines.

## B. vLLM enable path + red-flag status (re-checked 2026-06-19)
- Enable = `--enable-prefix-caching` → spec-on auto-forces `mamba_cache_mode=align` (config.py:326-360)
  → align HARD-ASSERTS `--enable-chunked-prefill` (a 2nd behavioral change to control). Qwen3-Next
  rejects 'all' (qwen3_next.py:1161). Plumbing in our build (#30877 GDN align, #33705 spec+align).
- **#43559** (~20% acc drop, APC+MTP-spec on Qwen3 GDN-hybrid = OUR COMBO): **OPEN**. Observed 0.21.0
  (ahead of our pin gfe9c3d6c5).
- **#45477** (candidate fix: keep prefill chunks mamba-block-aligned w/ spec): **OPEN, NOT MERGED** →
  NOT in our build. Root cause = eagle pruning zeros `last_cache_position` for prompts < 2× mamba
  block size → unaligned chunk-ends → GDN writes mid-block state to slot-0 → later lookups hash it
  as a boundary snapshot → cache poisoning (stray </think>, malformed tool calls, runaway gens).
  Our ~11–14k append-only context is LONG (≫ 2× block) so we LIKELY avoid the specific #45477
  trigger — confirm empirically, don't assume.
- **OUR PATCHER HARD-RAISES**: `fr10_phase4_patch_vllm_tree_gdn.py:11056-11062` raises
  NotImplementedError("DS conv state layout … align … num_accepted_tokens > 1") when
  `is_conv_state_dim_first()` AND offset>0. If our conv layout is dim-first, APC+spec CRASHES at the
  first multi-token accept. Gate 0 must confirm survival of a num_accepted>1 step (or we resolve the
  DS-layout branch first).

## C. Concrete plan
### Flags (flag-gate behind FR13_ENABLE_APC=1; default cat9 path byte-identical)
Add to the `vllm serve` line at `fr13_launch_forked_fa2_tree_server.sh:476-483`:
```
--enable-prefix-caching --enable-chunked-prefill \
--mamba-block-size <M>            # multiple of 8 (cache.py:108); start 512–1024
--mamba-ssm-cache-dtype float32   # SGLang default + #26807 lossless lever
--max-num-batched-tokens <B>      # >= mamba_block_size (config/vllm.py:1865)
```
SPEC_CONFIG unchanged (:181). Native-E5 arm (fr13_bigdenom_swe_serve.sh:132-138) gets same flags
for cache-ON-native (#43559 check); OFF for the floor arm.

### Gate 0 (FIRST — non-vacuous + no-crash)
Scrape /metrics after a warm turn, require ALL > 0: `prefix_cache_queries_total`,
`prefix_cache_hits_total`, `prompt_tokens_cached_total` (currently all 0). 0-hit ⇒ #45238 silent
trap ⇒ re-tune mamba_block_size BEFORE any lossless claim. ALSO confirm the server survives a
`num_accepted>1` step (the :11058 raise). Extend `fr13_gold_margin_probe.py:189` `_metrics_excerpt`.

### Same-boot A/B lossless gate (proven infra; US vs no-spec recurrent oracle, never a proxy)
4 arms in ONE boot (reset via /reset_prefix_cache, fr13_bigdenom_swe_serve.sh:255):
1. **B** = cache-OFF chunked-OFF (current deploy = floor).
2. **CONTROL** = cache-OFF chunked-ON (isolate chunked-prefill confound).
3. **A** = cache-ON (align+chunked+fp32) — under test.
4. **native-E5 cache-ON** — does #43559 reproduce on our commit.
Gates: (1) greedy byte-identity int-view; (2) BINDING per-token clear-margin argmax flip-rate vs
the no-spec RECURRENT oracle (fr13_b1_lossless_prescore.sh + fr13_recurrent_decode_oracle.py,
seed1313 topk20 thresh1.0) — cache-ON LOSSLESS iff within native E5 floor (Wilson-CI, not above);
(3) temp-0.6 bag-TV vs multi-seed native p95 floor. Re-verify `conv1d_out row-0 = 0.0` under APC.

### Sequencing + cost
Gate 0 (hits + no-crash, tune mamba_block_size) FIRST. Then the 4-arm same-boot Gates 1→2→3 with
fp32 state. GPU ≈ one B=1 serve-boot (~30min–couple hours). VERDICT YELLOW, measurement-gated.

Sources: SGLang MambaRadixCache (pytorch.org/blog/hybrid-models-meet-sglang, alibabacloud hybrid
blog, docs.sglang.io qwen3, sgl #22326, #25587); vLLM #43559/#45477/#45238/#26807/#33705/#30877.
