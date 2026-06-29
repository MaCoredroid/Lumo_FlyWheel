# FR13 — Lossless Mamba/GDN Prefix Cache: Code-Grounded Redesign

**Goal:** replace the `mamba_block_size=8192` band-aid (drift-suppression-by-coarsening) with a
*lossless-by-construction* GDN prefix cache, so that cache-ON reproduces cache-OFF state
bit-for-(near)-bit instead of merely keeping the chunked/recurrent disagreement under spec's
tolerance.

## 0. Grounding scope and sources

All line citations below are from the **real deployed** vLLM `0.19.2rc1.dev134` source extracted at
`/home/mark/.claude/jobs/22c39bb9/tmp/vllm_real_0192/`. The extracted subset is:

- `model_executor/layers/mamba/gdn_linear_attn.py` (the GDN attention layer + both kernels)
- `model_executor/layers/mamba/mamba_utils.py` (state shape/dtype/copy-spec calculators)
- `model_executor/models/qwen3_next.py` (model-level mamba-state hooks)
- `v1/attention/backends/gdn_attn.py` (`GDNAttentionMetadata` builder — slot assignment)
- `v1/core/sched/scheduler.py` (`_mamba_block_aligned_split`)
- `v1/core/single_type_kv_cache_manager.py` (`MambaManager`)
- `v1/worker/gpu_model_runner.py` (mamba pre/postprocess hookpoints)

**Deployment config** (from `scripts/fr13_launch_forked_fa2_tree_server.sh:216-243`):
`--mamba-block-size 8192`, `--mamba-ssm-cache-dtype float32`,
`--max-num-batched-tokens = MAMBA_BLOCK_SIZE` (8192), `--gdn-prefill-backend triton`
(line 567), spec-decode on → `mamba_cache_mode` auto-forced to `align`.

> **UNCERTAINTY / NOT in the extracted subset (flagged up front):**
> - `mamba_block_size` config plumbing and the `MambaSpec` dataclass (`block_size`,
>   `num_speculative_blocks`, `mamba_cache_mode`) are defined in `config/cache.py` and
>   `v1/kv_cache_interface.py`, **neither of which is extracted**. We infer their semantics from
>   their *use sites* in the extracted code.
> - `mamba_get_block_table_tensor(...)` (used at `gdn_attn.py:170`) is imported from
>   `v1/attention/backends/utils.py` — **not extracted**. It maps the per-request block table to the
>   single mamba state row used for restore. The launch-script comment
>   (`fr13_launch_forked_fa2_tree_server.sh:227-228`) asserts it anchors col-0 to
>   `(seq_len-1)//block_size`. **This must be confirmed against the running engine** before any code
>   change relies on the exact anchoring.
> - `mamba_utils.preprocess_mamba` / `postprocess_mamba` (the block→block state *copy* machinery, used
>   at `gpu_model_runner.py:3936` and `:1442`) live in `v1/worker/mamba_utils.py` — **not extracted**.
>   These are where align-mode physically copies a checkpoint into a fresh block. Any "re-chunk on
>   restore" landing point interacts with these and **must be read in the running tree** before
>   implementation.
> - `FLA_CHUNK_SIZE` is imported (`gdn_linear_attn.py:32`, `gdn_attn.py:326`) but its literal value is
>   not in the extracted files. The warmup docstring states the chunk kernels use a **fixed
>   `BT = chunk_size = 64`** (`gdn_linear_attn.py:699-701`). We treat `FLA_CHUNK_SIZE = 64` as the FLA
>   chunk granularity; **confirm the constant in `fla/ops/utils.py` on the running engine.**

---

## 1. Lifecycle map (real lines)

### 1.1 Where the mamba cache block / align mode is selected
- `scheduler.py:250-254` — `self.has_mamba_layers = kv_cache_config.has_mamba_layers`;
  `self.need_mamba_block_aligned_split = has_mamba_layers and cache_config.mamba_cache_mode == "align"`.
  So *align* mode is the toggle that turns on block-aligned chunked prefill for mamba.
- `qwen3_next.py:708-712` — Qwen3-Next **rejects `mamba_cache_mode == "all"`**; only `"align"` (or off)
  is supported. This is why the deployment is on align mode.
- `gdn_attn.py:170-175` — the metadata builder calls
  `mamba_get_block_table_tensor(m.block_table_tensor, m.seq_lens, self.kv_cache_spec, cache_config.mamba_cache_mode)`.
  The mamba-cache-mode reaches the per-step slot mapping here.

### 1.2 How mamba state slots are assigned per request (non_spec vs spec/node-bank)
In `gdn_attn.py` `GDNAttentionMetadataBuilder.build`:
- **No spec** (`spec_sequence_masks is None`, `:199-211`): `non_spec_state_indices_tensor = block_table_tensor[:, 0]`
  (`:207`). One state row per request = column 0 of the mamba block table. `spec_state_indices_tensor = None`.
- **Spec present** (`:212-314`):
  - `spec_state_indices_tensor = block_table_tensor[spec_sequence_masks_cpu, : self.num_spec + 1]`
    (`:255-257` and `:276-278`) — a **bank of `num_spec+1` rows per spec request** = the "node-bank".
  - `non_spec_state_indices_tensor = block_table_tensor[~spec_sequence_masks_cpu, 0]` (`:279-281`).
- State shape per row: `gated_delta_net_state_shape` → `temporal_state_shape =
  (num_v_heads/tp, head_v_dim, head_k_dim)` (`mamba_utils.py:232-237`). The conv state carries
  `conv_kernel_size - 1 + num_spec` width (`:227-230`) so spec lookahead fits in one conv block.
- The node-bank is allocated by `MambaManager` in `align` mode: `allocate_new_blocks`
  (`single_type_kv_cache_manager.py:934-1010`) always reserves
  `cdiv(num_tokens, block_size) + num_speculative_blocks` rows (`:956-958`), and records the
  *running-state* block via `last_state_block_idx` (`:968-978`).

### 1.3 Where the boundary checkpoint is written (chunked prefill write-back)
In `gdn_linear_attn.py` `_forward_core`, the **prefill** branch (`:982-1006`):
```
984   initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()   # READ restore
986   initial_state[~has_initial_state, ...] = 0                              # zero fresh rows
988-1002  self.chunk_gated_delta_rule(initial_state=..., output_final_state=True, ...)
1004  ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)  # WRITE chunked
```
So the **chunked** kernel both *seeds from* and *writes back to* the same `non_spec` state rows.
Because each scheduler step is forced ≤ 1 block (`max_num_batched = block_size`,
`fr13...:218-230`), the step-end recurrent state == the block-boundary state, and that boundary state
is what later gets cached/restored.

The actual *physical* checkpoint copy into a cacheable block (align mode) is performed by
`mamba_utils.preprocess_mamba`/`postprocess_mamba` (`gpu_model_runner.py:3936`, `:1442`) — **not
extracted; see §0 flag.**

### 1.4 How a cache HIT decides cached-prefix length and the re-prefill tail
- `MambaManager.find_longest_cache_hit` (`single_type_kv_cache_manager.py:785-831`) scans block hashes
  **right-to-left, early-stops on first match** (`:810-829`): it returns exactly one matched block plus
  `i` leading `null_block`s so that `hit_length = len(hit_blocks[0]) * other_block_size` (`:822-828`).
  Alignment is enforced by `(i+1)*block_size % alignment_tokens != 0 → continue` (`:817-821`).
- The cached-prefix length therefore lands on a **block_size boundary** (8192). Everything past that
  boundary is the **re-prefill tail**, re-run through the chunked kernel seeded from the restored
  checkpoint (`gdn_linear_attn.py:984`). A coarse 8192 block means the tail can be up to ~8191 tokens →
  the **TTFT tax**.
- `MambaManager.get_num_skipped_tokens` returns `num_computed_tokens - 1`
  (`single_type_kv_cache_manager.py:1018-1024`): mamba keeps only the **last** token's state, so only
  the boundary checkpoint is retained, not per-token states.

### 1.5 `_mamba_block_aligned_split` (scheduler.py:298-346)
Called at `:437-440` (running reqs) and `:709-717` (new/resumed reqs), gated by
`need_mamba_block_aligned_split` (`:252-254`). It forces `num_new_tokens` to a multiple of
`block_size` during prefill so each cached block is exactly `block_size` tokens:
- `block_size = self.cache_config.block_size` (`:327`) — note this is the *full-attn/global* block
  size, aligned to the mamba block via the cache config.
- `last_cache_position = num_tokens - num_tokens % block_size` (`:328`); eagle prunes one block (`:330-331`).
- If the step stays below the last boundary → `num_new_tokens = num_new_tokens // block_size * block_size`
  (`:333-335`); if it would cross the last boundary → snap to it (`:336-342`); else prefill the
  trailing remainder (`:343-345`).

---

## 2. The mismatch, pinned on real code

**Claim:** a continuation-prefill can seed the **chunked** kernel from a state row last written by the
**recurrent** kernel.

- **Restore READ (chunked seed):** `gdn_linear_attn.py:984`
  `initial_state = ssm_state[non_spec_state_indices_tensor].contiguous()`, fed directly into
  `self.chunk_gated_delta_rule(initial_state=initial_state, ...)` (`:988-996`).
- **Write-back A (chunked):** `gdn_linear_attn.py:1004`
  `ssm_state[non_spec_state_indices_tensor] = last_recurrent_state.to(ssm_state.dtype)` — produced by the
  **chunked** `chunk_gated_delta_rule` (`:988`).
- **Write-back B (recurrent, non-spec decode):** `gdn_linear_attn.py:1007-1026`
  `fused_sigmoid_gating_delta_rule_update(..., initial_state=ssm_state, inplace_final_state=True,
  ssm_state_indices=non_spec_state_indices_tensor, ...)` — writes the **recurrent** realization
  **in place** to the *same* `non_spec_state_indices_tensor` rows.
- **Write-back C (recurrent, spec / node-bank):** `gdn_linear_attn.py:957-977`
  `fused_sigmoid_gating_delta_rule_update(..., inplace_final_state=True,
  ssm_state_indices=spec_state_indices_tensor, ...)` — recurrent, written in place to the node-bank rows.

Because the restore READ at `:984` does **not** distinguish which kernel last wrote the row, a request
that decoded (B/C) and is then continued with a prefill chunk will seed `chunk_gated_delta_rule` from a
**recurrent-realization** state. The two realizations of the *same* gated-delta recurrence
(`chunk_gated_delta_rule` vs `fused_sigmoid_gating_delta_rule_update`) agree only to ~fp, and the
disagreement **accumulates across every boundary**. The third realization,
`fi_chunk_gated_delta_rule` (`:70-116`, the FlashInfer chunked variant), is yet another fp-distinct
path selectable at `:155-157`.

**Cache-OFF** is a single continuous chunked pass (one `chunk_gated_delta_rule` over the whole prefix,
no intermediate boundary reads). **Cache-ON** is a chunked-prefill **restart at each boundary** seeded
from a possibly-recurrent checkpoint — a chunked/recurrent **hybrid**. That is the root of the drift the
8192 band-aid merely *dilutes* (fewer boundaries → less accumulation, per `fr13...:209-215`).

Additional fp loss: `.to(ssm_state.dtype)` at `:1004`/`:1006` **truncates** to the cache dtype. Under
default `auto`, `gated_delta_net_state_dtype` → `_mamba_state_dtype` makes temporal state = conv state =
bf16 (`mamba_utils.py:86-99, 110-119`). The deployment pins `--mamba-ssm-cache-dtype float32`
(`fr13...:217`) so this specific truncation is fp32-wide (see R3).

---

## 3. The three moves (on real lines)

### R1 — single realization: cached rows hold ONLY the chunked realization, and pin ONE chunked kernel

**3a. Pin one chunked kernel.** `ChunkGatedDeltaRule.__init__` (`gdn_linear_attn.py:119-157`) chooses
`forward_cuda` (FlashInfer, `:159-183` → `fi_chunk_gated_delta_rule`) vs `forward_native`
(fla, `:185-211`) from `additional_config["gdn_prefill_backend"]` (`:125`). The deployment already
forces `triton` (`fr13...:567`) → `use_flashinfer = False` (`:140-141`) → fla. **Keep it pinned to
`triton`/fla.** FlashInfer (`:96-108`) casts state, g, beta to fp32 and exponentiates g
*outside* the kernel — a numerically *different* path from fla. Mixing them, or letting `auto`
(`:142-143`) pick FlashInfer on a cap-9 device, breaks bit-exactness. **Action:** assert/lock
`gdn_prefill_backend == "triton"` whenever prefix caching is on for GDN; reject `auto`/`flashinfer`.

**3b. Cached rows = chunked-only.** The invariant we want: *any state row that can be read back at
`:984` was last produced by `chunk_gated_delta_rule`.* Today that invariant is violated by write-backs
B (`:1007-1026`) and C (`:957-977`). R1 is **not** "stop the recurrent kernel from running" (impossible —
decode is inherently 1-token recurrent) but "**never cache a recurrent row; re-chunk before it becomes a
restore seed**." That is the crux in §4 — R1 by itself only guarantees correctness for the
**prefill-only** prefix (no decoded tokens cached), which is the common APC case (cache the *prompt*
prefix, not generated tokens).

### R2 — fine checkpoints: 64-aligned (FLA chunk size) instead of block_size=8192

- The chunked kernel already runs at `FLA_CHUNK_SIZE = 64` granularity internally
  (`gdn_linear_attn.py:699-701`; `chunk_indices`/`chunk_offsets` precomputed at
  `FLA_CHUNK_SIZE` in `gdn_attn.py:329-334`). So a checkpoint taken at **any multiple of 64** is
  exactly a chunk boundary the kernel already lands on — **no new kernel math** is needed to checkpoint
  more finely; the chunked pass is *identical* whether or not we snapshot at a given 64-boundary.
- What would change to make block_size = 64 (or any small multiple of 64):
  - `_mamba_block_aligned_split` (`scheduler.py:319-345`) keeps working unchanged — it just snaps to a
    smaller `block_size`, so steps cross more boundaries but each is a legal FLA chunk boundary.
  - `MambaManager.find_longest_cache_hit` (`single_type_kv_cache_manager.py:807-829`) keeps working;
    `block_size` smaller → finer hit granularity → **smaller re-prefill tail → lower TTFT** (the
    intended win over 8192).
  - **Cost:** more mamba state rows (one per 64 tokens vs one per 8192) → ~128× more mamba KV blocks.
    The temporal state row is tiny (`(num_v_heads/tp, head_v_dim, head_k_dim)`,
    `mamba_utils.py:232-236`), so this is feasible but must be **budgeted on the running engine**
    (it competes with the full-attn KV pool; the global `block_size` at `scheduler.py:327` is shared).
  - **Correctness:** finer checkpoints do **not** by themselves fix the chunked/recurrent mismatch —
    they make the *prefill* checkpoints denser and TTFT cheaper, but a *decoded* row cached at a 64
    boundary is still recurrent-realization. R2 is a TTFT/lossless-prefill lever; §4 is the
    decode-checkpoint lever. R2 and §4 are **complementary**.
- **Subtlety to confirm on the engine:** `_mamba_block_aligned_split` uses
  `self.cache_config.block_size` (`:327`), the *global* page size shared with full attention. Driving
  the mamba checkpoint to 64 may require decoupling the mamba block size from the global block size (the
  `mamba_block_size` config we could not see — §0). If they are forced equal, 64-aligned mamba
  checkpoints would also shrink the full-attn page to 64, which is undesirable. **This coupling is the
  single biggest open question and must be resolved against `config/cache.py` + `kv_cache_interface.py`
  on the running tree.**

### R3 — fp32: already enforced

`--mamba-ssm-cache-dtype float32` (`fr13...:217`) → `gated_delta_net_state_dtype` →
`_mamba_state_dtype`: `mamba_ssm_cache_dtype != "auto"` so
`temporal_state_dtype = STR_DTYPE_TO_TORCH_DTYPE["float32"]` (`mamba_utils.py:94-99`). Thus
`ssm_state.dtype == fp32`, and the `.to(ssm_state.dtype)` at `gdn_linear_attn.py:1004-1006` is a
**no-op-precision** cast (fp32→fp32). **R3 is done.** (Keep asserting it; if anyone drops the flag,
`auto` → bf16 truncation returns. The launch comment `fr13...:205` already documents this as the SGLang
default + vLLM #26807 lever.)

---

## 4. THE CRUX — the decode-checkpoint decision

**Problem restated on the code:** decode (`gdn_linear_attn.py:1007-1026`) and spec
(`:957-977`) produce state via `fused_sigmoid_gating_delta_rule_update` — the **recurrent**
realization — because a single token cannot be chunked. So any prefix that includes *generated* tokens
has **no chunked checkpoint** for the decoded region. If such a row is ever restored at `:984` and
continued by the chunked kernel, we get exactly the cache-ON drift.

> Note: in pure-APC (cache the **prompt** prefix only), the cached prefix is entirely prefill →
> chunked-realization → **already lossless under R1+R2+R3**. The crux only bites when generated tokens
> become part of a reusable prefix (multi-turn / agentic reuse of prior assistant output — exactly the
> FR13 tool-call workloads, `fr13...:211-213`).

### Option (a) — re-chunk the decoded tail through `chunk_gated_delta_rule` ON a cache hit (restore-time)
- **Where it lands:** `gpu_model_runner._prepare_inputs` mamba branch
  (`gpu_model_runner.py:3929-3954`, the align preprocess), and/or a new restore hook just before the
  first prefill `_forward_core` of the resumed request. Conceptually: on a hit whose checkpoint row was
  written by the recurrent kernel, **re-run `chunk_gated_delta_rule` over the decoded span** [last
  chunked checkpoint … hit boundary] to regenerate a chunked-realization seed, *then* restore.
- **Compute cost:** re-chunk = #decoded-tokens-in-the-cached-suffix, **once per cache hit**. With R2's
  64-granularity, at worst the tokens between the last *prefill* boundary and the hit boundary.
- **Correctness:** **This is the only option that reproduces the continuous chunked pass bit-for-bit**,
  *provided* the re-chunk is seeded from a genuine chunked checkpoint and run with the **same** fla
  kernel, same `cu_seqlens`/`chunk_indices`/`chunk_offsets`, same fp32 state. It literally re-derives the
  state cache-OFF would have computed.
- **Interaction with spec node-bank:** the decoded tail's *inputs* (q/k/v/g/beta per token) must be
  available to re-chunk. After decode they are **gone** — only the final recurrent state survives in the
  node-bank (`:973`). **So (a) as "re-chunk on restore" requires re-materializing the decoded tokens'
  projections**, i.e. re-running the conv + `fused_post_conv_prep` over the cached suffix from the raw
  token ids. That is feasible (the token ids are in the request) but is essentially a **re-prefill of
  the decoded suffix** — which is fine and is what cache-OFF does anyway.

### Option (b) — periodically re-chunk during decode to refresh a chunked checkpoint
- **Where it lands:** a counter in `_forward_core` decode branch; every N decoded tokens, run a
  chunked pass over the last N tokens (need to retain their q/k/v/g/beta or recompute from the conv
  state window) and overwrite the cacheable row with the chunked result.
- **Compute cost:** amortized N-token re-chunk every N decode steps = **+1 chunked pass per N tokens,
  on every request, always** — even requests whose output is never reused. Pure overhead on the hot
  decode path.
- **Correctness:** the refreshed checkpoint is chunked-realization at the refresh boundary, but the
  *running* decode state in between is still recurrent; on a hit you restore the last refresh boundary
  and re-prefill the small remainder — bit-exact at refresh boundaries only.
- **Interaction with spec:** spec writes the node-bank in place (`:957-977`); a periodic re-chunk would
  have to reconcile with the dynamic accepted-leaf row the committer writes (`fr13...:245-247` node-bank
  staleness fix). High blast radius on the spec path.

### Option (c) — store recurrent checkpoints but ALWAYS re-chunk on restore
- This is option (a) made unconditional (re-chunk every restored suffix regardless of which kernel wrote
  it). Simpler branch logic (no "was this recurrent?" test), but pays the re-chunk even for
  prefill-only prefixes where the row is *already* chunked — wasted work on the common APC path.

### RECOMMENDATION: **Option (a)** — restore-time re-chunk of the (decoded) suffix, gated on "is any cached token a generated token?"

Rationale:
1. **It is the only bit-exact option.** It reconstructs precisely the state the continuous chunked pass
   (cache-OFF) would produce, because re-chunking the suffix from a chunked seed *is* a piece of that
   continuous pass. (b) and (c) only guarantee exactness at refresh/checkpoint boundaries and add
   always-on overhead.
2. **Cost is paid only on reuse, scaled to the reused suffix** — zero overhead for non-reused requests
   and for prefill-only prefixes (the common case), unlike (b)/(c).
3. **Lowest spec blast radius.** It does not touch the decode hot path (`:1007-1026`) or the node-bank
   write path (`:957-977`); the re-chunk happens at scheduling/restore time on the `non_spec` rows.
4. **Composes with R2.** With 64-aligned checkpoints, the prefill portion of any prefix is already
   chunked-lossless; (a) only needs to re-chunk the *generated* span beyond the last prefill checkpoint.

**Concretely, (a) ≈ "on a hit whose suffix contains generated tokens, treat that suffix as a
re-prefill"**: feed the suffix token ids back through conv + `fused_post_conv_prep` +
`chunk_gated_delta_rule` seeded from the last clean (prefill) checkpoint, write the chunked result, then
continue. This reuses the *existing* prefill code path at `gdn_linear_attn.py:880-1006` — no new kernel.

---

## 5. Concrete change sketch (functions / lines to modify)

1. **Lock the chunked kernel (R1, low risk, stock-compatible).**
   In `ChunkGatedDeltaRule.__init__` (`gdn_linear_attn.py:119-157`): when prefix caching is enabled for
   a GDN layer, hard-require `backend == "triton"` (fla) and raise on `flashinfer`/`auto`. Prevents the
   `fi_chunk_gated_delta_rule` (`:70-116`) third realization from ever touching a cached row.

2. **Decouple + shrink the mamba checkpoint granularity to a multiple of `FLA_CHUNK_SIZE` (R2).**
   - Confirm/introduce a mamba-specific `block_size` (the `mamba_block_size` config, §0) so
     `MambaSpec.block_size` can be set to e.g. 64–512 *independently* of the global page size used at
     `scheduler.py:327`. This is the **TTFT** win and the actual replacement for `8192`.
   - No change needed in `_mamba_block_aligned_split` (`scheduler.py:298-346`),
     `MambaManager.find_longest_cache_hit` (`:785-831`), or the chunked kernel — they are already
     block-size-parametric and FLA already runs at 64.

3. **Restore-time re-chunk of generated suffixes (Option a, the lossless crux).**
   - **Landing point:** the align preprocess in `gpu_model_runner.py:3929-3954` (which already runs
     `mamba_utils.preprocess_mamba` to copy checkpoints into fresh blocks). Add a step that, for each
     resumed request whose cached suffix `[last_prefill_checkpoint … hit_boundary]` includes tokens
     produced by decode, schedules those suffix tokens as a **chunked re-prefill** (reusing the prefill
     branch `gdn_linear_attn.py:880-1006`) seeded from the last clean checkpoint, *before* the normal
     tail prefill.
   - **Bookkeeping:** tag each cached mamba checkpoint with the realization that produced it (a 1-bit
     "chunked vs recurrent" flag per cached block) so the re-chunk fires only when needed. The natural
     home is alongside `MambaManager.last_state_block_idx` (`single_type_kv_cache_manager.py:781`,
     `:968-978`) or the block metadata.
   - **Reuse, don't reinvent:** the re-chunk is literally the existing `:982-1006` path with
     `non_spec_query_start_loc`/`chunk_indices`/`chunk_offsets` rebuilt for the suffix
     (`gdn_attn.py:316-334`). No new kernel.

4. **Keep R3 (fp32) asserted.** Guard that `mamba_ssm_cache_dtype == "float32"` whenever GDN prefix
   caching is on, so the `:1004` cast stays lossless.

5. **Retire the band-aid.** Once (1)-(3) hold, drop `MAMBA_BLOCK_SIZE` back from 8192 to the small
   R2 value and remove `APC_MAX_NUM_BATCHED_TOKENS = MAMBA_BLOCK_SIZE` coupling
   (`fr13...:216-230`) — the ≤1-block-per-step constraint was only needed to make the recurrent-poisoned
   intermediate checkpoint correct; with (3) the checkpoints are chunked-clean regardless of step span.
   **Verify the multi-block-per-step checkpoint semantics on the engine before relaxing
   `max_num_batched`** (the comment at `fr13...:218-224` describes the #45238 overshoot hazard).

---

## 6. Feasibility, stock-vs-FR13, upstreamability, bit-exactness risks

**Stock vLLM already provides:** align-mode block-aligned mamba prefix caching
(`scheduler.py:252-254,298-346`; `MambaManager`), the per-step ≤1-block knob (via `max_num_batched`),
fp32 SSM cache dtype (`mamba_utils.py:94-99`), the triton/fla-vs-FlashInfer selector
(`gdn_linear_attn.py:119-157`), and 64-granular chunk metadata (`gdn_attn.py:316-334`). So **R1, R2,
R3 are configuration/guard changes on stock surfaces** — upstreamable as a "lossless mode" assert + a
mamba block size knob.

**FR13-patch territory:** Option (a)'s restore-time re-chunk of generated suffixes is **new behavior**
(no stock hook re-derives a chunked checkpoint for decoded tokens). It is the only genuinely new code and
the only part that needs the running engine to design (it sits on top of the **non-extracted**
`v1/worker/mamba_utils.py` copy machinery — §0).

**What could make it NOT bit-exact (must validate):**
- **FlashInfer vs fla** (`:155-157` vs `:70-116`): different fp path (fla in-kernel L2 norm + g; FI
  casts to fp32 and `exp(g)` outside, `:96-108`). **Pin fla.**
- **Chunk-boundary alignment:** a restored seed must land on a true FLA 64-boundary; any checkpoint not a
  multiple of `FLA_CHUNK_SIZE` would re-chunk from a mid-chunk state and diverge.
- **`cu_seqlens` / `chunk_indices` / `chunk_offsets`:** these are rebuilt per step from
  `non_spec_query_start_loc_cpu` (`gdn_attn.py:316-334`); a re-chunk over a suffix must rebuild them for
  the suffix or the chunk partitioning (hence the math) differs from cache-OFF.
- **`has_initial_state` zeroing** (`:985-986`): the re-chunk must mark `has_initial_state=True` (seed from
  the prior checkpoint), not zero it.
- **dtype truncation** (`:1004`): only lossless while fp32 is pinned (R3).
- **Mamba/global block-size coupling** (`scheduler.py:327`): if not decoupled, R2 can't shrink without
  shrinking full-attn pages (§4/R2 subtlety).

---

## 7. Validation test (concrete)

**Primary (state-level, the decisive one):** instrument `ssm_state` and diff cache-ON vs cache-OFF.
1. Boot two servers with **identical** seed/config except `--enable-prefix-caching` (use the existing
   `FR13_APC_CONFIG_ONLY` matched-config cache-OFF arm, `fr13...:237-243`, so chunked-prefill numerics
   are held constant and *only* the cache restore differs).
2. Drive a multi-turn prompt where turn 2 reuses turn 1's **generated** output as prefix (forces the
   decoded-suffix-in-cached-prefix case — the crux).
3. After the turn-2 prefill, dump the GDN temporal `ssm_state` row for the request
   (`non_spec_state_indices_tensor` row, `gdn_linear_attn.py:984`) on both servers and compute
   `max|Δ|` / `rel‖Δ‖` per layer.
4. **Pass criterion:** with R1+R2+R3+Option(a), the per-layer state diff is at fp32 round-off
   (≈1e-6 rel), **independent of `mamba_block_size`** — i.e. the diff no longer *grows* as block_size
   shrinks (today it does: `fr13...:209-213` shows 1024 drifts, 8192 doesn't). The block-size
   independence is the proof the fix is structural, not dilutional.

**Secondary (behavioral, cheap canary):** rerun the cat6root tree + cache-ON workload
(`fr13...:212-213`); confirm no tool-call runaway and real cache hits + spec engaged at the **small**
block size (where the band-aid previously failed). Behavioral pass is necessary but not sufficient — the
**state diff is the lossless proof.**

> Both tests require the **running engine** (the state dump, the re-chunk hook, and the
> `mamba_get_block_table_tensor` anchoring all live outside the extracted subset — §0).
