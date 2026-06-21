# APC (GDN prefix-cache) fix state — CANONICAL (2026-06-21)

Branch `fr13-prefix-cache`. Current, honest state after the gross poison was root-caused and fixed.
Supersedes every earlier carrier theory in this dir (`apc_ssm_carrier_FIXED.md`, the #43559 /
chunk-vs-recurrent / WT-overshoot framings — all refuted below).

## THE FIX (BAKED, default-ON whenever APC is on) — commit d228c76b
`scripts/fr13_launch_forked_fa2_tree_server.sh` line ~205:
```
APC_MAX_NUM_BATCHED_TOKENS=${APC_MAX_NUM_BATCHED_TOKENS:-$MAMBA_BLOCK_SIZE}
```
i.e. **default max-num-batched-tokens = the mamba block size**, so each chunked-prefill scheduler step
crosses **at most one** mamba block boundary. APC-scoped (consumed only inside `APC_FLAGS`), so the
non-APC locked cat9 serve command is **byte-identical**.

## ROOT CAUSE (proven, run_20260621T013300Z)
vLLM mamba **`align` caches ONE checkpoint per scheduler step — the LAST block boundary it crosses**
(vLLM #45238 "align keeps only one checkpoint"). With `max_num_batched=2048 > block_size=1024` a step
spans ~2 blocks, so an **intermediate** boundary (e.g. 3072, crossed in the same step as 4096) is cached
with the **step-END (overshoot) recurrent state**. A later cache hit at that boundary restores a grossly
wrong GDN state → **total collapse at token 1** (`</think>\n\noutput_text\n\noutput_text…` degenerate
loop). Forcing each step to ≤1 block makes step-end == boundary state → correct checkpoint → poison gone.
Result: cache-ON output flips from `output_text` garbage to **coherent + on-task**; **TTFT cache-hit 3.96×**
(miss 4.29s → hit 1.08s); GATE-A/B/C/D pass.

## REFUTED CARRIER THEORIES (do not revisit)
- **#43650 drop-final-block** — REFUTED + stays `FR13_APC_DROP_FINAL_BLOCK` default-OFF. The GDN restore
  reads `ssm_state[block_table[:,0]]` anchored to `(seq_len-1)//block_size`, NOT the matched-block count
  (prior red-team reader 2), and empirically still garbled at `max_num_batched=2048` — dropping a matched
  block is a no-op on the restored state.
- **#43559 fp-nondeterminism (autotune/shape variance)** — REFUTED from source: the FLA chunk kernel is
  deterministic + length-invariant (autotune key `['H','K','V','BT']`, `T` is `do_not_specialize`,
  sequential chunk accumulation, no split-K/atomics). batch-invariant mode doesn't touch the FLA Triton
  kernels.
- **WT-leaf overshoot** — the baked SSM write-through is INERT in the prefill reproduction (`map_leaf=None`,
  `did=False`): a warm-req prefill has no decode-committed leaf. So WT was not the carrier here.

## RESIDUAL (small, being closed) — NOT yet fully lossless
Temp-0.6 precheck (1 prompt, 512 tok, same boot; clear-margin argmax-flip vs the no-spec RECURRENT oracle;
`output/fr13_apc_temp06/PRECHECK_VERDICT.md`):
- **cache-ON 9.18%** (Wilson95 [6.97, 11.99]) — **within the absolute 12.90% E5 floor**
- **cache-OFF 6.05%** (Wilson95 [4.30, 8.47])
- +3.12pp; clear-margin CIs overlap (n=512), but raw-flip is 2× (19.5% vs 9.6%, significant).
Cause: cache-OFF prefills [0..end] continuously (no restart); cache-ON restores the boundary state and
restarts the FLA chunk scan over the suffix — the FIRST suffix chunk's fold of the restored state is an
artifact cache-OFF lacks. **Fix in progress (user ruling: close it now)** = bounded single-first-suffix-chunk
recurrent recompute (`FR13_APC_HIT_RECURRENT_SUFFIX`, gated default-0): recompute only the first ≤64 suffix
tokens via the bit-exact sequential rank-1 scan (`fr10_gdn_tree_kernel.py`), FLA the rest. O(64),
prefill-only, NO whole-suffix serial scan (BARRED), NO WY (PARKED). Design workflow w0by0i7nn.

## OTHER BAKED FLAGS (default-ON with APC; partial / for the decode-snapshot case)
`FR13_APC_SSM_WRITE_THROUGH`, `FR13_APC_SSM_SNAPSHOT`, `FR13_APC_CONV_FIX`/`CONV_SNAPSHOT`,
`FR13_APC_BLOCK_ALIGN_45477` — kept; they target the decode-committed-leaf snapshot (multi-turn), not the
prefill overshoot that `max_num_batched=block_size` fixes. Conv is FUSED for our GDN so the conv override
is ~no-op. All only fire under `--enable-prefix-caching`, so non-APC is byte-identical.

## MEASURED WINS (banked)
- Gross poison eliminated; cache-ON coherent + on-task.
- **TTFT cache-hit 3.96×**; decode-TPS unaffected (APC is prefill-only).
- cache-ON within the absolute 12.90% E5 lossless floor (residual to be closed to ≤ cache-OFF).

## INSTRUMENTS (reusable)
- `scripts/fr13_apc_prefill_after_hit.sh` — greedy gross-poison reproduction (DIFFER/MATCH).
- `scripts/fr13_apc_temp06_precheck.sh` — temp-0.6 cache-ON + cache-OFF same-boot capture → oracle src.
- `scripts/fr13_recur_rescore_in_container.sh` + `fr13_recurrent_decode_oracle.py` — no-spec RECURRENT
  oracle rescore → `total_clear_margin_flips`/`total_positions` (the binding lossless metric).

Ship gate still pending: temp-0.6 4-task (12907/13033/13236/13398) cache-OFF vs fixed-cache-ON —
decode-TPS≈ + TTFT-win + coding-quality parity + cache-ON clear-margin ≤ cache-OFF.
