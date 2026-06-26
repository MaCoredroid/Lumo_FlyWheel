# SGLang `MambaRadixCache` — implementation-level design reference (2026-06-19)

Source-level dive of SGLang's prefix cache for Qwen3-Next / GDN-hybrid linear-attention
models (clone read at `/tmp/sglang`, ~v0.5.5-dev). The user asked specifically for "how they
design" the Qwen prefix cache as a reference. This is the *what they actually do in code* —
the design-level summary is in `prefix_cache_enable_plan.md` §A; this is the backing detail.

Files: `python/sglang/srt/mem_cache/mamba_radix_cache.py` (1353 L),
`mamba_checkpoint_pool.py` (373 L), `mamba_checkpoint_pool.py:Int8CheckpointStore`.

## Core idea
Recurrent SSM/conv state is NOT block-addressable like attention KV — there is no per-token
slot you can hash and reuse. So SGLang keeps, per radix-tree node, a **full snapshot of the
recurrent state at that prefix boundary**, in a separate compressed pool. A prefix-cache hit
copies the deepest matching node's snapshot into a fresh *active* slot (copy-on-write) and
decoding resumes from there. Two pools, decoupled:
- **Active `MambaPool`** — running requests, full precision, kernel-facing.
- **Cached checkpoint pool (`MambaCheckpointPool`)** — radix-owned, idle, compressed.

## The pieces that matter for US (vLLM tree-spec on GB10)

### 1. Cached state is int8-quantized — a CAPACITY trick, NOT a vLLM-transferable lossless lever
`Int8CheckpointStore`: the cached SSM **temporal state** is stored **int8, symmetric
per-(head, k-channel)** (scale axis reduces over d_v). Rationale verbatim from the docstring:
- Quantized ONCE on store, dequantized ONCE on a hit — it **never re-enters the recurrence as
  a quant→dequant loop**, so the only error is a single rounding of S, then decode continues at
  full precision.
- int8 beats fp8-e4m3 at the same 1 byte: the state is ~uniformly distributed, fp8 wastes bits
  on the exponent; the per-k-channel scale axis matches the decay `diag(alpha)` so large entries
  keep ~bf16 precision and error concentrates on small entries that barely affect the readout.
- Gives ~2× cached-prefix capacity at fixed memory; composes with host-offload (halves it too).

**CORRECTION to plan §A "BORROW #1 fp32 default":** that "fp32" is the *active/working* SSM
dtype (SGLang `--mamba-ssm-dtype float32`, the computation precision). The *cached snapshot* in
this SGLang build is int8, not fp32. vLLM `align` mode has **no separate quantized checkpoint
store** — it keeps the working state at the last block boundary in the working dtype. So the
int8 store is an SGLang-internal capacity optimization that **does not port to vLLM align**. The
transferable discipline stays: **fp32 working state** (`--mamba-ssm-cache-dtype float32`) for
computation accuracy. Don't chase an int8 cached store on our path — vLLM has no hook for it.

### 2. conv1d window kept at NATIVE dtype (full precision) — confirms our verify target
The conv window (W−1 tokens) is "tiny, not worth quantizing" → stored full-precision in the
checkpoint and copied verbatim on hit (`load_to_active`: `conv[:, active] = ckpt_conv[:, ckpt]`).
This lines up with `project_fr13_conv_priorwindow_root` / `FR13_CONV_COMMITTED_PATH`: the conv
prior-window is the fragile, non-invertible part (SGLang #25587). SGLang's answer is "snapshot
the actual window, never reconstruct it." OUR align gate must re-prove `conv1d_out row-0 = 0.0`
under whatever window vLLM reconstructs at the boundary — that is the spec-intrinsic carrier,
present with OR without APC (#25587 ran radix OFF).

### 3. Spec-decode safety = the `extra_buffer` ping-pong + DONATE-don't-copy
`cache_unfinished_req` (the on-the-fly snapshot taken mid-decode, the spec-relevant path):
- `cache_len = req.mamba_last_track_seqlen if enable_mamba_extra_buffer else len(token_ids)` —
  with the ping-pong track on (spec path), it caches only up to the **safely-committed prefix
  length**, NOT the speculative tail. (Don't snapshot un-verified tokens.)
- `donate_mamba_ping_pong_slot(req, new_slot)` — the active slot is **DONATED** to the radix,
  not copied: "avoids a data copy that would race with the forward stream." The ping-pong gives
  the spec kernel a fresh slot to keep writing into while the donated one becomes the checkpoint.
- snapshots are **page-aligned** (`page_aligned_len = len//page_size*page_size`) — the analog of
  vLLM's `mamba_block_size` boundary. SGLang also only snapshots at block/page boundaries.

The takeaway for us: vLLM's align mode is the *one-checkpoint-at-last-block-boundary* analog of
this. The danger is the boundary landing inside request-unique tokens → 0 reuse (#45238 silent
trap). SGLang sidesteps it with a per-node tree of checkpoints (many boundaries); vLLM keeps one
→ **we must measure hit-rate and tune `mamba_block_size`** (plan §B Gate 0).

### 4. Dual-LRU eviction (two independent budgets)
`evict_mamba(mamba_num)` and `evict_full(full_num_tokens)` are separate; `LRUList(mamba=True)`
vs the full/KV list. Mamba state slots and KV token budget are evicted independently (a node can
be KV-evicted while its mamba checkpoint survives, and vice versa), because they have different
sizes/lifetimes. vLLM align has no analog (single block pool) — informational only.

## Net: what ports to our vLLM path, what doesn't
PORTS → (a) **fp32 working SSM state** (`--mamba-ssm-cache-dtype float32`) for accuracy;
(b) **measure hit-rate + tune mamba_block_size** (vLLM's one-checkpoint design has no automatic
density, unlike SGLang's per-node tree); (c) **verify the conv window** under reconstruction
(`conv1d_out row-0 = 0.0`), the spec-intrinsic carrier SGLang snapshots rather than rebuilds.
DOES NOT PORT → int8 cached store (no vLLM align hook), per-node checkpoint tree, ping-pong
donate, dual-LRU (all SGLang-internal architecture; we serve on vLLM align mode).

Sources: SGLang `mem_cache/mamba_radix_cache.py`, `mamba_checkpoint_pool.py`,
`Int8CheckpointStore`; SGLang #25587 (Hybrid-GDN MTP not lossless, conv non-invertibility);
pytorch.org/blog hybrid-models-meet-sglang. vLLM cross-refs: #43559 (APC+MTP acc drop, OUR
combo, OPEN), #45477 (block-align fix, OPEN/unmerged), #45238 (silent 0-hit), #26807 (fp32).
