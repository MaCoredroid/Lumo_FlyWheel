# FR13 SPEED TUNING PLAN — Branch A = OPT-1 G2 SYNC-KILL

Date 2026-06-15. Branch `fr13-speedfix`. CPU-only design (NO GPU boot; measurements
run AFTER, under the prelaunch host-mem protocol). HEAD `f25075b8`. Goal: **cat9 B=1
decode-TPS STRICTLY > native E5** by reclaiming the committer-sync residual that
kills the tree path's run-ahead.

Grounded against the LIVE pinned image (`scripts/vllm_src.sh --sha` =
`3dbe092e`, vLLM 0.19.2rc1.dev134) — native rejection sampler +
`AsyncGPUModelRunnerOutput` read fresh, NOT the stale `/tmp` cache.

---

## 1. The OPT-1 first-draft state (MEASURED / CITED) — and why G2 is the real win

OPT-1 (`10ebccac`, FR13_GPU_COMMITTER, default-OFF) moved the committer's
**pure-integer** accept / path-LCP / bonus decision off the host Python loop onto a
Triton integer kernel. The decision logic + CPU byte-A/B gate are complete (52/52
byte-identical, exit 0). BUT the first draft does **not** actually kill the sync — it
is still gated by the same main-thread sync AND adds a second one:

1. **The packed DtoH + main-thread sync survives.** The committer (patched into
   vLLM `rejection_sample`) still runs the FR13_EAGER_PACK packed readback at
   `scripts/fr10_phase4_patch_vllm_tree_gdn.py:6745-6762`:
   - `:6758` `_ep_dev[...].copy_(_ep_s)` — device-side gather of all 6 committer
     inputs (parents / drafts / parent_targets / self_targets / bonus / counts) +
     the stacked 48×2 replay-flag matrix into one staging buffer;
   - `:6760` `_ep_cpu[...].copy_(_ep_dev[...], non_blocking=True)` — pinned async DtoH;
   - **`:6761` `torch.cuda.current_stream(...).synchronize()`** — the BLOCKING sync
     on the MAIN launching thread (census: chain5 blocks the main thread in
     memcpyAsync **91.9%** of the verify window vs native **0.8%**;
     FR13_BEAT_NATIVE_SPEED_DESIGN_BIND.md). The bind cites the stale line `:5674`;
     HEAD has drifted — the live sync is `:6761`.
   - `:6762` `.tolist()` → `parents_cpu`, `drafts_cpu`, … host lists.

2. **The GPU committer is fed those ALREADY-SYNCED host lists.** The hook
   (`_patch_rejection_sampler_gpu_committer`, `:14304-14355`) injects, just before
   the per-request loop (anchor `out_rows = [] … for req_i, node_count in
   enumerate(counts):`), a flag-guarded branch that calls
   `fr13_gpu_committer_full(parents_cpu, drafts_cpu, parent_targets_cpu,
   self_targets_cpu, bonus_targets_cpu, counts, max_spec_len)` — i.e. it consumes
   the host lists produced by the `:6761` sync. So the sync is NOT removed.

3. **The kernel re-uploads, then re-syncs.** `fr13_gpu_committer_triton`
   (`scripts/fr13_gpu_committer_kernel.py:407-484`) calls `_build_device_layout`
   (HtoD re-upload of the host lists it was just handed), launches
   `_fr13_committer_kernel[(n_req,)]`, then does the host readback at
   **`:471-474`**:
   ```
   out_tokens_h   = out_tokens.cpu().tolist()
   row_len_h      = row_len.cpu().tolist()
   accepted_row_h = accepted_row.cpu().tolist()
   best_lcp_h     = best_lcp.cpu().tolist()
   ```
   Each `.cpu()` is a **second** implicit main-thread sync. The first draft's own
   docstring (`:36-42`, kernel module) and `FR13_GPU_COMMITTER_BIND.md` G2 flag this
   as the unbuilt win: *"the win only lands when that readback moves to a non-gating
   side stream + CUDA event so the MAIN thread never blocks (restores run-ahead)."*

**Net:** first-draft OPT-1 with the flag ON would be SLOWER than OFF (sync at
`:6761` + re-upload + re-sync at `:471-474`). The decision is on-GPU and lossless,
but the run-ahead is not yet restored. **G2 is the patch that lands the speed.**

### What native does (the run-ahead bar to restore)
Native `RejectionSampler.__call__` (pinned-image
`v1/worker/gpu/spec_decode/rejection_sampler.py:159-232`) returns `sampled` /
`num_sampled` as **device tensors** from a Triton kernel (`strict_rejection_sample`)
— NO `.cpu()`, NO `.synchronize()`, NO host loop on the main thread. The host
readback is deferred to `AsyncGPUModelRunnerOutput`
(`v1/worker/gpu_model_runner.py:227-292`): on a dedicated
`async_output_copy_stream`, `sampled_token_ids.to("cpu", non_blocking=True)` +
`self.async_copy_ready_event.record()` (`:251-261`); the main thread keeps
launching the NEXT step; `get_output()` blocks on
`async_copy_ready_event.synchronize()` (`:269`) only LATER, overlapped with the next
forward. That is the 0.8% main-thread block. G2 restores exactly this shape.

### Where the committer output must land (so we know what stays on-device)
The committer's products are scattered back to **device** tensors regardless of
host lists, via the FR13_EAGER_PACK pinned HtoD writeback (`:7366-7410`):
`output_token_ids` (`:7403`) and `accepted_tree_rows` (`:7406`), plus the GDN
durable-state advance tensors `_LUMO_FA_ACCEPTED_TREE_PATHS_TENSOR` /
`_LUMO_FA_ACCEPTED_TREE_LENS_TENSOR` (`:7440-7489`) that the next step's GDN replay
reads ON DEVICE (`:5604`, `:2130`, `:4079`, …). The host lists
(`out_rows`/`accepted_rows`/`accepted_node_paths`) are needed ONLY as the host
SOURCE of those HtoD writes plus three diagnostic globals
(`_LUMO_TREE_LAST_ACCEPTED_{ROWS,LENS,NODE_PATHS}_KERNEL`, `:7420-7424`, consumers =
logging + the eager-only boundary tap). **The serving critical path needs the
accept decision ON DEVICE, not as host lists** — exactly what the kernel already
produces (`out_tokens` / `row_len` / `accepted_row` / `best_lcp` device tensors).

---

## 2. The G2 sync-kill patch (exact design; default-OFF flag `FR13_COMMITTER_SYNCKILL`)

New flag **`FR13_COMMITTER_SYNCKILL`** (default "0"), gated UNDER `FR13_GPU_COMMITTER`
(sync-kill is meaningless without the GPU committer). When BOTH ON: the committer
inputs stay device-resident, the kernel writes device outputs, and the
device→host readback rides the existing FR13_EAGER_PACK side-stream + event so the
main thread never blocks. When `FR13_COMMITTER_SYNCKILL=0` (default): byte-for-byte
the current path (`:6761` sync + first-draft kernel or legacy loop) — zero change.

### G2.a — feed the kernel DEVICE tensors; skip the `:6761` main-thread sync
The committer already has the device-resident sources in scope at the EAGER_PACK
block: `tree_parent_indices`, `draft_token_ids`, `parent_token_ids`,
`self_token_ids`, `bonus_token_ids`, `num_draft_tokens` (the `_ep_*_src` views at
`:6704-6719`). Add a NEW device entry to the GPU-committer module,
`fr13_gpu_committer_device(parents_dev, drafts_dev, ptgt_dev, stgt_dev, bonus_dev,
counts_dev_or_list, max_spec_len)`, that:
  - builds the fixed-stride device layout **on-device** from the device tensors
    (a device `cumsum`/scatter of `counts` → per-request node offsets; the leaves
    list is derived in-kernel from `parents`, NOT on host — the kernel already
    walks ancestry array-free). `counts` is the ONLY quantity the host packing
    needs; it is tiny (`n_req` ints, B=1 ⇒ 1) and on the serving path is ALREADY a
    host list from `num_draft_tokens` when that arrives as a list (`:6786`,
    `:6809`) — no committer-input sync required to read it.
  - launches `_fr13_committer_kernel[(n_req,)]` writing the SAME device outputs
    (`out_tokens`, `row_len`, `accepted_row`, `best_lcp`).
  - returns those four **device tensors** (NO `.cpu()`).
Wire it into the committer under the new flag so that when `SYNCKILL` is ON the
EAGER_PACK packed-readback block (`:6745-6795`, incl. the `:6761` sync) is SKIPPED
for the committer inputs — only the tiny `counts`/flag bits the GDN replay route
needs are read (those already have their own staged path, `_ep_stacks['flags']`,
and can move to the side-stream event the same way). **The 91.9% main-thread block
is gone because no committer-input DtoH+sync runs on the main thread.**

### G2.b — move the committer-output readback to the side-stream + event
Replace the kernel's `:471-474` `out_tokens.cpu().tolist()` (and the OFF-path host
loop's per-element scatter) with the existing run-ahead machinery:
  - Reuse `_fr13_eager_pack_stage('committer_out_synckill', n_elems, device,
    torch.int64)` (`:6222-6247`) — it already returns `(device_buf, pinned_cpu_buf,
    cuda_event, rec_flag)` with the exact pinned-reuse event guard native uses.
  - Pack the four device outputs into one staging device buffer (device-side
    slice copies, on the DEFAULT compute stream — no sync), then on a dedicated
    side stream (a module-level `torch.cuda.Stream()`, mirroring native's
    `async_output_copy_stream`): `side.wait_stream(default); pinned.copy_(staging,
    non_blocking=True); event.record()`.
  - The accept decision the SAME-STEP serving path needs (the HtoD scatter into
    `output_token_ids` / `accepted_tree_rows` / the GDN path/len tensors) is done
    **device→device** from the kernel's device outputs — it never needs the host
    copy. So the main thread launches the writeback + the next forward WITHOUT
    blocking.
  - The host lists (`out_rows`/`accepted_rows`/`accepted_node_paths` and the three
    diagnostic globals) are materialised LAZILY: a thin shim that, on first host
    access, does `event.synchronize()` then `.tolist()` (native's `get_output()`
    shape). On the pure serving path (metrics OFF, no boundary tap, no argmax gate)
    these host lists are needed only for the diagnostic globals at `:7420-7424` —
    move THAT block behind the event so it runs overlapped (or, when no diagnostic
    consumer is armed, skip the host materialisation entirely; the globals are
    logging-only). The HtoD writeback at `:7366-7410` becomes a device→device copy
    (kernel outputs are already on device) so it no longer sources from host lists.

### G2.c — what stays unchanged (lossless invariants)
- The kernel `_fr13_committer_kernel` is byte-for-byte the first-draft kernel
  (G1-validated separately). G2 changes only the INPUT plumbing (device vs synced
  host list) and the OUTPUT transport (side-stream event vs main-thread `.cpu()`),
  never the integer decision.
- The HtoD writeback values into `output_token_ids` / `accepted_tree_rows` /
  `_LUMO_FA_ACCEPTED_TREE_*` are bit-for-bit identical (same ints, device→device
  instead of host→device).
- Default `FR13_COMMITTER_SYNCKILL=0` ⇒ the committer takes the existing `:6761`
  path verbatim; the new device entry and side-stream are never touched.

### G2.d — per-change lossless gate (must pass BEFORE any speed number)
Extend `scripts/fr13_gpu_committer_byte_ab_gate.py` with a DEVICE arm
(`fr13_gpu_committer_device`) asserting its outputs (after the side-stream event
sync) are byte-identical to the ORACLE across the same 52-tree matrix, AND that the
device→device writeback into a mock `output_token_ids`/`accepted_tree_rows`/path
buffer equals the legacy host-scatter result element-for-element. Plus the
structural assert: `FR13_COMMITTER_SYNCKILL` read with default "0" and gated under
`FR13_GPU_COMMITTER` (OFF leaves `:6761` + the legacy/first-draft path intact).

---

## 3. Why it is lossless (the argument + the gate)

**Pure-integer, location-only, no float / no reduction / no reorder.** The accept /
path-LCP / bonus decision is integer token-id `==` compares, a parent walk, an LCP
scan, an earliest-leaf strict-`>` tie-break, and a 3-way bonus-source select (kernel
module `:13-18`). G2 moves NEITHER the math NOR the values — it changes only:
  (i) WHERE the committer inputs live when the kernel reads them (device tensor vs a
      host list that was a `.tolist()` of that same device tensor — bit-identical
      ints), and
  (ii) WHEN/WHERE the OUTPUT bytes cross to host (a side stream + event vs an inline
       `.cpu()` on the main thread — same bytes, later, off the critical thread).
No fp op, no accumulation order, no reduction is introduced or reordered. The GDN
durable-state advance reads the SAME device path/len tensors (`:5604`, `:2130`) —
its inputs are byte-identical, so the NEXT step's forward is byte-identical.

**Byte-identical when OFF:** `FR13_COMMITTER_SYNCKILL=0` (default) leaves the
`:6745-6795` packed sync, the first-draft kernel/legacy loop, and the `:7366-7410`
writeback verbatim — the default-ON serving path (the locked
SWE-gold-gate build) is untouched; the OFF/ON A/B is the instrument.

**Per-change lossless gate (HELD per-change, feedback_fr13_lossless_compare_target):**
the gate for THIS change is, in order:
  1. CPU/GPU byte-A/B (G2.d) — device arm + writeback equality vs the bit-exact
     oracle across 52 trees; structural default-OFF assert.
  2. Same-seed **byte-identical served streams** OFF vs ON, both greedy (t=0.0) and
     t=0.6, `prompts_swe4` seed 1313 — in the SAME boot (in-process A/B, NOT
     cross-boot; no_cross_boot_byte_gate).
  3. **accept/event unchanged** OFF vs ON (depth-matched: cat9 is d5 ⇒ vs native
     E5; paired teacher-forced, NOT the class-12-confounded aggregate).
  4. **regular-decode pristine** (non-spec requests unaffected).
  5. the per-token **argmax-vs-clean-recurrent-oracle** probe
     (`scripts/fr13_gold_margin_probe.py` / `fr13_recurrent_decode_oracle.py`):
     US vs native-E5, each vs its OWN no-spec RECURRENT decode oracle — NEVER a
     chunked-prefill / streamed / serial / backend-name proxy; int-view, never atol.
     (Scalar accept/event is necessary-not-sufficient,
     reference_scalar_metric_per_token_blindspot — the argmax probe is the binding
     instrument; G2 is a structural transport change so it MUST be a no-op on every
     token, which the probe verifies directly.)

---

## 4. Measurement plan (OFF vs ON; runs AFTER, under the prelaunch host-mem protocol)

**Boot once, in-process A/B (no cross-boot byte gate).** Single locked boot
(`scripts/fr13_launch_locked.sh`), `FR13_GPU_COMMITTER=1` fixed, toggle
`FR13_COMMITTER_SYNCKILL` 0→1. Pins (reference_fr10_speed_measurement_pitfalls +
feedback_dont_handroll_speed): `VLLM_BATCH_INVARIANT/LUMO_BATCH_INVARIANT_VLLM=0`
identical both arms, `FR10_METRICS=0`, `GPU_MEMORY_UTILIZATION=0.82`,
`MAX_NUM_SEQS=1`, `prompts_swe4` seed 1313 greedy temp 0.0. Reset prefix cache +
`torch.cuda.empty_cache` + verify `nvidia-smi` between arms
(gpu_mem_collection_between_experiments).

**Engagement asserts BEFORE any number** (fail_loud_assert_engagement): assert
tok/draft > 0, `has_tree_parent_indices`, `tree_sample_accept` fired, AND the new
FR13_COMMITTER_SYNCKILL engagement needle (fires once in BOTH flag states, like
`_fr13_eager_pack_needle` `:6265-6291`) confirms the device-input + side-stream
path actually engaged when ON. No number is recorded if any assert fails.

**SPEED basis = decode_seconds RAW /metrics counter, NEVER TPS/accept/wall**
(banned as MEASURED facts; all per-forward ms are INFERRED until this clean run):
  - per-forward s/fwd = `vllm:request_decode_time_seconds_sum` /
    `vllm:spec_decode_num_drafts_total`, per-request.
  - **Primary verdict:** does ON reclaim the ~4-6 ms of the cat9 +6.5 ms tax (bind
    arithmetic: conservative 2.5-4 ms → ~220.7-222.2 ms; optimistic 6 ms →
    ~217-218 ms at/just-below native 218.2 ms)? Report ON s/fwd vs OFF s/fwd vs
    native E5 0.2182.
  - **Run-ahead reclaim, the mechanism check:** census the main-thread block in
    memcpyAsync ON vs OFF (target: 91.9% → toward native's 0.8%). nsight via
    `scripts/join_nsight_decode_metrics.py`; this is the diagnostic that confirms
    the sync was actually killed (NOT a hand-rolled TPS decomposition).
  - **End-to-end verdict (the GOAL):** cat9 B=1 decode-TPS STRICTLY > native E5.
    Because the accept edge is unchanged by G2 (lossless ⇒ cat9 ~3.18 vs native
    ~3.07 still holds; depth-matched E5), if ON brings s/fwd to ≤ native, cat9 wins
    on TPS via the accept edge. Report TPS as a DERIVED end-state only, with the
    decode_seconds + accept basis shown — never as the measured speed primitive.

**The byte-A/B-on-GPU check (lossless, same boot):** with the SAME boot, dump the
served stream OFF vs ON (greedy + t0.6) and assert byte-identical (gate §3.2); run
the argmax probe ON vs its no-spec recurrent oracle and confirm within-floor vs
native-E5's own probe (gate §3.5). The decisive pass: **OFF≡ON byte-identical
streams + unchanged accept/event + sync-kill confirmed by the census drop**, AND
the ON arm's s/fwd ≤ native E5 so cat9 TPS clears native. Only on all-pass does G2
graduate from default-OFF instrument toward a bake-in (behavior-preserving,
default-ON path still byte-identical — feedback_flag_gate_metrics_reuse_infra).

---

## 5. Open / needs-the-live-boot (does NOT block the design)
- G2.a on-device `counts`→offset packing: if `num_draft_tokens` ever arrives as a
  device tensor on the serving path, the tiny `counts` DtoH must also ride the side
  stream (it is `n_req` ints — negligible, but keep it off the main thread for
  purity). B=1 serving has it as a host list already (`:6786`).
- G3 (CUDA-graph conditional-node / `torch.cond` in-capture accept) is the FOLLOW-ON
  after G2 lands the sync-kill; not in scope here. The committer is eager (class 6);
  G2 restores run-ahead without needing in-capture commit first.
- Kernel O(depth²) ancestor re-walk (G1): lossless either way; pick the
  shared-mem-path variant on measured kernel time at the live boot. Independent of
  G2's transport change.
