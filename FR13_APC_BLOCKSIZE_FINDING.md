# FR13 APC — mamba_block_size makes cache-ON + spec-decode work (lead, 2026-06-28)

## Finding (one line)
Raising `mamba_block_size` from the deployed **1024** to **8192** (keeping
`max_num_batched_tokens = block_size`) makes **cache-ON (APC prefix caching) +
TREE speculative decoding (cat6root) RESOLVE** astropy-12907 — where the deployed
1024 config fails every time.

## Scope (be precise)
- Validated for the **cat6root TREE** drafter (caterpillar tree spec).
- **SPINE (chain) is UNTESTED at 8192** — do not claim it yet.
- Single task so far: **astropy-12907**, temp 0.6, live agentic SWE.

## Evidence
| config | mamba_block_size | result on 12907 (cache-ON + cat6root spec, temp 0.6) |
|---|---|---|
| deployed | 1024 | **0/3** — all failed with the char-8 malformed-JSON tool-call runaway |
| this finding | 8192 | **resolved** (rollout 1): `cached_tokens=28560` (cache genuinely engaged), `spec_ok=1` (spec engaged), `char8=0` (no runaway) |

This is the first time spec(cat6root)+cache resolved 12907 with **both** speedups
live (TTFT prefix-cache hit AND decode-spec engaged).

## Why (carrier, proven this session)
`--enable-prefix-caching` flips vLLM's `mamba_cache_mode` `none → align`, which forces
the chunked prefill to **save/reload + restart the GDN recurrent scan at every
`mamba_block_size` boundary**. That boundary save/reload is a cross-chunk fp
re-accumulation (not bit-exact vs a single continuous pass). Over a ~30K-token
prefix that's ~30 boundaries at block=1024 vs ~3 at block=8192. Speculative
decoding's continuous accept/reject amplifies the cumulative diff into the runaway;
no-spec tolerates it (no-spec+cache resolves). Coarsening the block ⇒ fewer
boundaries ⇒ less cumulative diff ⇒ spec tolerates it.

Proof the cached *value* is NOT the carrier: a scheduler-level "shadow" that bypasses
the restore (re-prefill) still fails — and the bypass provably wipes the cached state.
So the carrier is the **align-mode block-aligned chunked-prefill machinery**, not the
restored value. (See memory `project_fr13_apc_spec_specific_carrier`.)

## Tradeoff (it's a dial, not free)
Larger `mamba_block_size` = **coarser mamba-state cache** ⇒ on a cache hit, the tail
back to the last block boundary re-prefills (up to `block_size` tokens of GDN
recurrence recomputed) ⇒ some mamba-TTFT win is lost. The **KV-cache TTFT win** (per
token) and the **decode-spec TPS win** are preserved. The shipping value should be the
**smallest block that's still lossless** (most TTFT kept) — TBD from the drift curve.

## Open / not-yet-closed (do NOT over-claim)
1. **Only 1/1 rollout** — big-N solve-rate not yet confirmed.
2. **bug-vs-fp not settled.** A bug-hunt flagged a possible *large* wrong-state bug in
   the FR13 `accept_token_bias` snapshot patch (~23 diff) — but that contradicts the
   shadow result (cache value isn't the carrier) and the deployed `SNAP_FIX=1` failing.
   The **drift-curve measurement** (`scripts/fr13_apc_drift_curve.sh` +
   `fr13_apc_drift_curve_reduce.py`) quantifies drift vs block_size: ~0.0078 ⇒ fp
   machinery; ~tens ⇒ the bug. Magnitude settles it and picks the sweet-spot block.
3. **Spine untested; only one task.**

## Recommended config (interim)
`MAMBA_BLOCK_SIZE=8192` + `APC_MAX_NUM_BATCHED_TOKENS=8192` for cache-ON + cat6root
spec. Treat as the validated-working value; the drift curve will pick the smallest
lossless block for the final shipping config.

## Next
Drift curve → smallest-lossless `mamba_block_size` → big-N (N≥6) validation on
**cat6root AND cat10** spec trees on 12907 (and add spine) = the lossless-APC ship gate.
