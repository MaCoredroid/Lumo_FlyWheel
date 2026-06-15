# FR13 SPEED-TUNING PLAN — SEQUENCED CAMPAIGN (OPT-1 sync-kill ⊕ OPT-A fp8 ⊕ topology reshape)

Date 2026-06-15. **CPU-only design** (GPU free; do NOT boot — measurements run AFTER,
under the prelaunch host-mem protocol). Repo `/home/mark/shared/lumoFlyWheel`.
GOAL (user): **cat9-family B=1 decode-TPS STRICTLY > native E5**, with the
long-established **LOSSLESS GATE HELD per-change** throughout speed tuning.

This is the SYNTHESIS of the two design branches:
- **Branch A = OPT-1 G2 sync-kill** (full design `FR13_SPEED_TUNING_PLAN_BRANCH_A.md`
  archive of the pre-synthesis doc; the detail is folded into §2 here).
- **Branch B = TOPOLOGY RESHAPE** (full design `FR13_SPEED_TUNING_PLAN_BRANCH_B.md`,
  HEAD `3c846d5d`; the detail is folded into §4 here).
- plus **OPT-A = GB10-tuned fp8 GEMV** (`e90de7ef`, `FR13_GB10_FP8_GEMV_CFG`, the
  user-ruled-in-scope bit-identical shared-kernel tune), folded into §3.

Grounded against the LIVE pinned image (`scripts/vllm_src.sh --sha` = `3dbe092e`,
vLLM 0.19.2rc1.dev134), read fresh — NOT the stale `/tmp` cache. Repo HEAD
`3c846d5d` on `main`; OPT-1 first draft `10ebccac`, OPT-A `e90de7ef`.

---

## 0. THE THREE LEVERS AT A GLANCE (independent, additive; each default-OFF, lossless-by-construction)

| lever | flag (default OFF) | what it touches | lossless mechanism | built? | B=4 / CUDA-capture posture |
|---|---|---|---|---|---|
| **OPT-1** sync-kill | `FR13_COMMITTER_SYNCKILL` (under `FR13_GPU_COMMITTER`) | committer INPUT plumbing + OUTPUT transport (NOT the math) | pure-integer decision unchanged; only WHERE inputs live + WHEN/WHERE outputs cross to host | first draft `10ebccac` (GPU-resident decision); **sync-kill UNBUILT** — this plan designs it | committer is EAGER (class 6, never captured); side-stream + event live OUTSIDE any captured region ⇒ capture-safe by construction; B=4 enlarges `n_elems` only |
| **OPT-A** fp8 GEMV | `FR13_GB10_FP8_GEMV_CFG` | GB10 w8a8 block-fp8 GEMM M-tiling / swizzle / warps / pipeline | `BLOCK_SIZE_K=128` PINNED ⇒ the fp32 K-accumulation loop is identical tiles in identical order; only M-axis/L2-swizzle/warp/pipeline move | **BUILT** `e90de7ef`; CPU byte-A/B PASS | the override fires on the SAME captured GEMM; must re-capture + re-confirm bit-exact at the B=4 M-tiles (M grows 6→up to ~40 at B=4) |
| **RESHAPE** trees | per-shape exact-match guard (e.g. `_fr10_is_cat6root`); default cat9 | the drafter PACKING + `tree_choices` (downstream auto-adapts) | shapes are integer topology; lossless HELD at the cat9 operating point (not improved); gated root sibling is strict-mask-invisible | cat3w/chain3 on HEAD; **R1/R2/R4/R5/cat10 UNBUILT** (~15-30 lines each) | each shape must CUDA-capture + behave at B=4 co-residency; conf-gate is a host-side pre-capture branch (variable pack realized pre-capture) ⇒ capture-safe; flag any shape that can't |

**Independence:** OPT-1 changes only the committer's host↔device transport; OPT-A
changes only the block-fp8 GEMM launch config; reshape changes only the drafter
topology. They compose (different subsystems, different flags) — but each is
measured and gated SEPARATELY first (§6), because the final-judgment tier costs
~30 min/candidate and confounds stack.

---

## 1. CURRENT STATE (MEASURED, decode_seconds basis) — the bar to beat

`FR13_SPEED_HISTORY_RECONCILE` canonical:
- **native E5** (FLASH MTP-5): **0.2182 s/fwd**, accept **3.161**, **18.93 TPS** —
  the SPEED bar AND the lossless bar (its own ~3-flip recurrent-oracle floor).
- **cat9** (LOCKED, depth-5 + 4 leaves): **0.2248 s/fwd = 1.030× native** (+6.5 ms/fwd)
  at accept **~3.18** (slight edge). Lossless-vs-native **CONFIRMED at scale**
  (big-denom cat9 13.55% ≈ native 13.99% flip rate).

⇒ cat9 already has the **accept edge** (3.18 > 3.07); it LOSES on TPS only because of
the **+6.5 ms/fwd s/fwd tax**. Two independent ways to erase that tax (and one
accept lever):
- **OPT-1** reclaims the committer-sync residual (~4-6 ms of the tax; the 91.9%
  main-thread block) → cat9 s/fwd toward native → cat9 wins on TPS via the accept edge.
- **OPT-A** is a whole-system bandwidth win on EVERY fp8 GEMM (native AND tree) —
  it lowers s/fwd for both arms; cat9 keeps its accept edge over the faster native.
- **RESHAPE** sheds the dominant **pad8→pad16 step (+42-46 ms/fwd)** by keeping
  N ≤ 7 (pad8), trading the deepest leaves for a free conf-gated root sibling.

> Honest label: every per-forward ms is INFERRED (census/literature-anchored; nsys
> per-kernel export empty). MEASURED facts = native/cat9 s/fwd above, the pad8/pad16
> step + M-invariant lm-head (+0.0019 ms/verify row; ptxas wp5hsu63v/fdf5ffa7),
> OPT-A's CPU byte-A/B PASS, OPT-1's 52/52 CPU byte-A/B. The clean B=1 `decode_seconds`
> boot is the arbiter (feedback_dont_handroll_speed).

---

## 2. OPT-1 — the G2 sync-kill patch (Branch A; design only, default-OFF `FR13_COMMITTER_SYNCKILL`)

### 2.1 First-draft state — the decision is on-GPU, but the sync is NOT killed (it adds a SECOND)
OPT-1 (`10ebccac`, `FR13_GPU_COMMITTER`, default-OFF) moved the committer's
**pure-integer** accept / path-LCP / bonus decision off the host Python loop onto the
Triton kernel `_fr13_committer_kernel` (`scripts/fr13_gpu_committer_kernel.py:13-18`),
which writes DEVICE outputs `out_tokens` / `row_len` / `accepted_row` / `best_lcp`. The
committer returns `output_token_ids` as a DEVICE tensor
(`scripts/fr10_phase4_patch_vllm_tree_gdn.py:8536`, `:7977`, `:8515`) exactly like
native `rejection_sample`, and the GDN durable-state advance reads device path/len
tensors `_LUMO_FA_ACCEPTED_TREE_{PATHS,LENS}_TENSOR` (`:7440-7494`, consumed `:5604`,
`:2130`, `:4079`). Decision + CPU byte-A/B gate are complete (52/52 byte-identical, exit 0).

**BUT the first draft does NOT kill the sync — three residual host-syncs survive on
the MAIN launching thread:**
1. **The FR13_EAGER_PACK packed-input readback.** `:6758` device-gather of the 6
   committer inputs + the 48×2 replay-flag matrix into staging; `:6760`
   `_ep_cpu.copy_(..., non_blocking=True)`; **`:6761`
   `torch.cuda.current_stream(...).synchronize()`** — the BLOCKING sync on the main
   thread (census: blocks the main thread in `memcpyAsync` **91.9%** of the verify
   window vs native **0.8%**; FR13_BEAT_NATIVE_SPEED_DESIGN_BIND); `:6762` `.tolist()`
   → `parents_cpu` / `drafts_cpu` / … host lists.
2. **The GPU-committer hook feeds those ALREADY-SYNCED host lists.** The hook
   `_patch_rejection_sampler_gpu_committer` (`:14248` def; injection `:14304-14358`)
   calls `_fr13_km.fr13_gpu_committer_full(parents_cpu, drafts_cpu, …)` at
   `:14342-14346` — it CONSUMES the host lists produced by the `:6761` sync, so the
   sync is not removed.
3. **The kernel re-uploads then re-syncs.** `fr13_gpu_committer_triton`
   (`kernel.py:407-484`) calls `_build_device_layout` (`:348-404`, host Python loop +
   `.to(device)` HtoD re-upload of the host lists it was just handed), launches the
   kernel, then does the host readback at **`:471-474`**: `out_tokens.cpu().tolist()` /
   `row_len.cpu().tolist()` / `accepted_row.cpu().tolist()` / `best_lcp.cpu().tolist()`
   — each `.cpu()` a SECOND implicit main-thread sync. AND the writeback at `:7366-7409`
   still sources HtoD from host lists (`_ep_a_np[req_i*cols:...] = row` built from
   `out_rows`/`accepted_rows`).

**Net:** flag-ON would be SLOWER than OFF (`:6761` sync + re-upload + `:471-474`
re-sync + host-sourced writeback). The decision is on-GPU and lossless, but native's
async run-ahead is NOT restored — that is exactly what G2 lands.

### 2.2 What native does (the run-ahead bar to restore)
Native `RejectionSampler.__call__` (pinned-image
`v1/worker/gpu/spec_decode/rejection_sampler.py:159-232`) returns `sampled` /
`num_sampled` as DEVICE tensors from `strict_rejection_sample` — no `.cpu()`, no
`.synchronize()`, no host loop. The host readback is deferred to
`AsyncGPUModelRunnerOutput` (`v1/worker/gpu_model_runner.py:227-292`) on
`async_output_copy_stream` (`wait_stream(default)` `:252`, `.to("cpu",
non_blocking=True)` `:253`, `async_copy_ready_event.record()` `:261`), with
`get_output()` blocking on `event.synchronize()` (`:269`) only LATER, overlapped with
the next forward = the 0.8% main-thread block. **G2 restores exactly this shape.**

### 2.3 G2.a — feed DEVICE tensors, skip the `:6761` sync
New flag **`FR13_COMMITTER_SYNCKILL`** (default "0"), gated UNDER `FR13_GPU_COMMITTER`
(meaningless without the GPU committer). When BOTH ON: committer inputs stay
device-resident, kernel writes device outputs, device→host readback rides the existing
FR13_EAGER_PACK side-stream + CUDA event so the main thread never blocks. When
`SYNCKILL=0` (default): byte-for-byte the current path (`:6761` + first-draft
kernel/legacy loop + `:7366-7409` host-sourced writeback) — zero change.

The committer already holds device-resident sources at the EAGER_PACK block — the
`_ep_*_src` views at `:6704-6719` (`tree_parent_indices` / `draft_token_ids` /
`parent_token_ids` / `self_token_ids` / `bonus_token_ids[:,0]` / `num_draft_tokens`).
Add a NEW device entry to the kernel module,
`fr13_gpu_committer_device(parents_dev, drafts_dev, ptgt_dev, stgt_dev, bonus_dev,
counts, max_spec_len)`, that:
- builds the fixed-stride layout ON-DEVICE (device `cumsum`/scatter of `counts` →
  per-request node offsets; derive `leaves` ON-DEVICE — a device scatter marking
  has-child from `parents`, OR have the kernel derive leaves array-free via the same
  ancestry walk it already does, removing the precomputed `leaves` input; either is
  lossless, pick on kernel time per G1);
- launches `_fr13_committer_kernel[(n_req,)]` writing the SAME four device outputs;
- returns those four DEVICE tensors (NO `.cpu()`).

Wire under the flag so `SYNCKILL=ON` SKIPS the `:6745-6795` packed-readback for
committer inputs — **the `:6761` sync is gone; no committer-input DtoH+sync runs on the
main thread.** `counts` is the ONLY host quantity packing needs; B=1 serving has it as
a host list already from `num_draft_tokens` (`:6786`) — no input sync. The replay-flag
matrix bits the GDN route needs (`_ep_stacks['flags']`) move to the same side-stream event.

### 2.4 G2.b — move the OUTPUT readback to the side-stream + event
Replace `kernel.py:471-474` `.cpu().tolist()` (and the OFF-path host scatter) with the
existing run-ahead machinery: reuse `_fr13_eager_pack_stage('committer_out_synckill',
n_elems, device, torch.int64)` (`:6222-6247`), which returns `(device_buf, pinned_cpu,
cuda_event, rec_flag)` with the exact pinned-reuse event guard native uses. Pack the
four device outputs into ONE staging device buffer via device-side slice copies on the
DEFAULT compute stream (no sync); then on a dedicated module-level side stream
(mirroring native's `async_output_copy_stream`): `side.wait_stream(default);
pinned.copy_(staging, non_blocking=True); event.record()`.

The same-step serving path's HtoD scatter into `output_token_ids` / `accepted_tree_rows`
AND the GDN path/len tensors becomes **device→device** from the kernel's device outputs
— so the `:7366-7409` writeback (currently host-sourced `_ep_a_np[...] = row`) never
sources from host lists. The main thread launches the writeback + the next forward
WITHOUT blocking.

The host lists (`out_rows` / `accepted_rows` / `accepted_node_paths` + the 3 diagnostic
globals `:7420-7424`) are materialised LAZILY: a thin shim that on first host access
does `event.synchronize()` then `.tolist()` (native `get_output()` shape). On the pure
serving path (metrics OFF, no boundary tap, no argmax gate) the only consumer is
logging/eval (`scripts/swe_x86_helpers/relaunch_qwen36_round.py:3625`,
`_LUMO_TREE_LAST_ACCEPTED_ROWS_KERNEL`) — so when no diagnostic consumer is armed the
host materialisation is SKIPPED entirely.

### 2.5 CAPTURE / B=4 safety (OPT-1)
The committer is **class-6 EAGER, never CUDA-graph captured**
(`_fr13_eager_pack_stage` docstring `:6226-6227` states this). The side-stream + event
live OUTSIDE any captured region ⇒ there is NO host sync inside a captured graph; the
device→device writeback and the kernel are Dynamo-opaque/eager and B=4-safe (the stage
realloc-on-grow keys on size, B=4 just enlarges `n_elems`). **G3** (in-capture
`torch.cond` / conditional-node commit) is the follow-on, NOT in scope.
DEFAULT `FR13_COMMITTER_SYNCKILL=0` leaves `:6761` + the first-draft kernel/legacy loop
+ `:7366-7409` verbatim.

### 2.6 Why OPT-1 is lossless (the argument)
Pure-integer, location-only, no float / no reduction / no reorder. The decision is
integer token-id `==` compares, a parent walk, an LCP scan, an earliest-leaf strict-`>`
tie-break, and a 3-way bonus-source select (committer loop `:6877-7137`; kernel module
`:13-18`). G2 moves NEITHER the math NOR the values — only (i) WHERE the committer
inputs live when the kernel reads them (a device tensor vs a host list that was a
`.tolist()` of that SAME device tensor — bit-identical ints), and (ii) WHEN/WHERE the
OUTPUT bytes cross to host (a side stream + event vs an inline `.cpu()` on the main
thread — same bytes, later, off the critical thread). The HtoD writeback values into
`output_token_ids` / `accepted_tree_rows` / `_LUMO_FA_ACCEPTED_TREE_*` are bit-for-bit
identical (same ints, device→device), so the GDN durable-state advance reads
byte-identical path/len tensors and the NEXT step's forward is byte-identical. The
kernel `_fr13_committer_kernel` is byte-for-byte the first-draft kernel (G1-validated
separately); G2 changes only INPUT plumbing and OUTPUT transport.

---

## 3. OPT-A — GB10-tuned fp8 GEMV config (BUILT `e90de7ef`, default-OFF `FR13_GB10_FP8_GEMV_CFG`)

### 3.1 What it is (whole-system bandwidth win on EVERY fp8 GEMM, native + tree)
vLLM ships no GB10/Spark JSON for `get_w8a8_block_fp8_configs` (`configs/` has
H100/H200/L40S/MI3xx only), so on GB10 `w8a8_triton_block_scaled_mm` always falls to
the generic DEFAULT (`BLOCK_SIZE_M=64, GROUP_SIZE_M=32, num_warps=4, num_stages=2`) —
tuned for fat-M server batches, wasteful at tree-verify decode M=6-10 and only
double-buffering the LPDDR5X weight DMA. B=1 decode is HBM-bandwidth-bound (273 GB/s
LPDDR5X) and the fp8 weight stream dominates, so a decode-M-tuned config is a
whole-system win on BOTH arms.

`_patch_fp8_utils_gb10_gemv_cfg` (registered in `main()`) injects a flag-gated override
into the else-branch: when `FR13_GB10_FP8_GEMV_CFG=1` AND device is GB10 AND M≤32 AND
`block_size[1]==128`, use **`BLOCK_SIZE_M=16, GROUP_SIZE_M=1, num_warps=8,
num_stages=4`**. `BLOCK_SIZE_N/K` stay pinned to `block_size[*]` (K=128).

### 3.2 LOSSLESS BY CONSTRUCTION (the user ruled this in-scope as a bit-identical shared-kernel tune)
`BLOCK_SIZE_K=128` PINNED ⇒ the fp32 K-accumulation loop runs the **identical number of
tiles in identical order**; only M-tiling / L2 group-swizzle / warp-count /
pipeline-depth move, NONE of which touch the reduction axis. Default-OFF / non-GB10 /
large-M leave the stock default dict byte-for-byte (FIX-1/2/3 gating mirror). This is
NOT a reward-hack reroute — it is the SAME native vLLM block-fp8 kernel with a different
launch config that provably preserves the K-reduction order.

CPU byte-A/B gate `fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py` (boot-free) ALL PASS: G0
patched compiles, G1 idempotent, G2 stock dict verbatim, G3 N/K pinned; A-arm
(flag-off / non-GB10 / large-M) byte-identical to pristine default, B-arm (flag-on GB10
decode) GB10 cfg with N/K pinned.

### 3.3 CAPTURE / B=4 caveat (the ONE thing OPT-A must re-confirm live)
The override fires on the SAME captured GEMM (it changes the launch config, not the
op), so it IS inside the captured region. Two live checks the design REQUIRES before
OPT-A ships:
- **(a) M-tile validity at B=4.** The gate fires only on M≤32. At B=1 tree-verify
  M=6-10; at B=4 the co-resident verify rows grow (up to ~40 across 4 sequences) — so
  some captured GEMMs may exceed M=32 and fall back to the stock config (correct,
  byte-identical, but no speedup there). Confirm the M-distribution at B=4 and that the
  flag-on path captures cleanly at the B=4 M-tiles (`BLOCK_SIZE_M=16` must tile the B=4
  M without a spill — re-run the ptxas / TRITON_KERNEL_DUMP spill check at the B=4 M).
- **(b) raw bit-exact RE-CONFIRMED at B=4.** Per BV-spill caveat #5, read RAW
  `out_vs_native_max_abs == 0.0` by hand (NOT the `atol=1e-3` exit code) at the B=4
  M-tiles, and re-run the served-stream byte-identical A/B at B=4 (co-residency changes
  losslessness; CPU byte-A/B + B=1 bit-exact do NOT imply B=4 bit-exact).

If (a) shows the B=4 M routinely exceeds 32 (no speedup) OR (b) shows any non-zero raw
drift at B=4, OPT-A is flagged NOT-shippable-as-is and the design re-tunes the M
threshold or config. The deploy gate is the live-GPU byte-identical-stream + B=4
capture confirm (documented in `e90de7ef` as the deploy-time gate).

### 3.4 Relation to the lm-head (the dominant B=1 GEMV)
The B=1 attribution (FR13_B1_SPEED_ATTRIBUTION_BIND) found the lm-head bf16 GEMV
(`internal::gemvx`, 2.543 GB weight/call, ~62% of 273 GB/s) is the dominant B=1 cost;
FIX-1 (`FR13_DRAFTER_SINGLE_LOGITS`, removing the double-head) already addressed the
DRAFTER double-compute. OPT-A targets the fp8 block-scaled GEMMs (in/out/MLP proj),
where the M=6-10 decode shape is mis-served by the fat-M default — a separate, additive
bandwidth win. (The lm-head itself is bf16, not block-fp8; the joint-lm-head split-N
sub-native lever is a SEPARATE future, not OPT-A.)

---

## 4. TOPOLOGY RESHAPE (Branch B; design only, per-shape exact-match guards, default cat9)

### 4.1 The speed accounting that makes "remove-deep / add-root" a real lever
Two cost laws (FR13_SPEED_HISTORY_RECONCILE; FR13_SPEED_TAX_BASELINE):
1. **lm-head verify rows are M-INVARIANT: +0.0019 ms/row** (539 rows/+1 ms; one
   `[M,5120]·[5120,248320]` bf16 GEMM, weight read once). Adding/removing a verify ROW
   is ~free at the lm-head ⇒ do NOT design around "skip lm-head for unservable nodes."
2. **The binding per-NODE cost is GDN state-row traffic + the N_PAD STEP.**
   `n_pad = next_pow2(N+1)`: N≤7 → **pad8** (chain5 N=5); N=8..15 → **pad16** (cat9 N=9:
   **+42-46 ms/fwd over chain5 = ~7× the 3N+2a+1 row-traffic floor**, dominated by the
   pad8→pad16 step; `h_cache=(N_PAD,BV,DIM_K)` register-bound — N_PAD=16 is the only
   0-spill cat9 geometry at BV=16/warps=8, 254/255 regs).

**Design key:** TPS = accept_tok / s_per_fwd. The cheapest way to push cat9 TPS over
native is NOT fewer lm-head rows (free) — it is to **keep N ≤ 7 (stay in pad8)** so
s/fwd drops toward chain5's regime, WHILE keeping enough accept (via a confidence-gated
root sibling) that accept/event holds the break-even. A pad8 tree that holds cat9-class
accept beats BOTH native AND cat9 on TPS.

### 4.2 What the prior reshape campaign settled (do NOT re-litigate)
The RECURRENT-frame A/B (`FR13_RESHAPE_AB_RECURRENT_BIND`, wf_0e61765e, verify HOLDS) is
canonical: native E5 = 3 flips / 3.08 accept (the BAR); cat9 = 23 flips (~18 de-casc) /
3.198 (lossy, FAST, the accept edge); chain3 (d3 leaf-free) = 1 flip / 2.266 (LOSSLESS,
SLOW); cat3w (d3 + width) = ~17 / 2.282 (lossy AND slow). **DEPTH is +1; WIDTH/leaf
co-residency is the flip CARRIER (+16)** — the leaves are BOTH the accept edge AND the
lossy co-residency, COUPLED. ⇒ **Reshape is NOT a flip-reduction lever** (topology alone
gives lossless-OR-fast, not both). Branch B is a **SPEED-with-lossless-HELD** lever: at
the CURRENT cat9 lossless operating point (held per change, NOT improved), find a
reshape that makes decode-TPS > native by trading N_PAD / GDN state-row-traffic against
accept. The 22-flip chase is the PARALLEL drift front, out of scope here per
speed-first order (project_fr13_speed_first_lossless_gate). WY PARKED, out of scope
(feedback_wy_parked_dont_revive).

### 4.3 cat10 revive — accounting-correct (the prior 2.932 was an ARTIFACT, not a loss)
cat10 = cat9 + the depth-0 root sibling `(1,)`, sorted `(len,path)` (10 nodes,
committed depth 5, N=10 → **pad16**):
`[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0),(0,0,0,1),(0,0,0,0,0),(0,0,0,0,1)]`.
The banked accept 2.932 (`[2,6,8,6]` / −0.27 vs cat9 3.198) was **artifact-dominated**
(FR13_CAT10_INVESTIGATE_BIND, adversarial verify holds=FALSE = CORRECTED;
feedback_check_artifact_before_concluding):
- **(i) Class-12 whole-window TRAJECTORY/DENOMINATOR confound (dominant):** cat9 vs
  cat10 generated DIFFERENT greedy streams (diverge p0@17/p1@11/p2@21/p3@61); cat10 p0
  hit EOS 25 tokens sooner (73 vs 98) ⇒ 25 fewer accepted tokens over ONE MORE event ⇒
  mechanically lower accept/event. Cross-trajectory accept/event is NOT apples-to-apple.
- **(ii) SIBLING-STOP DENOMINATOR artifact (the sharp d0→d1 drop):** a root-sibling win
  `(1,)` is `accepted_len=1` (caps at d0), swelling `per_pos[0]` but contributing 0 to
  d1+, deflating the d1|d0 conditional. De-confounding pos0 RECOVERS d1|d0 to ~0.84+;
  d2/d3/d4 conditionals are FLAT cat9-vs-cat10.
- **(iii) m1 (verify co-residency) STRUCTURALLY RULED OUT:** the verify `strict_mask`
  walks parents to root; `(1,)` is never a spine ancestor ⇒ NO spine row has
  `strict[spine,root_sib]=1`, so the root-sibling row is attention-INVISIBLE to every
  spine row (GDN tree-scan uses the same mask). Residual ≤ a sub-ULP fp reduction-order
  leak in the shared pad tile, NOT −0.27.
- **(iv) The d0 RESCUE is REAL:** d0 accept 0.871→0.906 (+0.035, ~+21/boot); root
  runner-up is truth **27%** when root top-1 misses (a 2-horse-race near-tie signature;
  random rank-2 ≈0%). The 62%-of-rejects-at-step-0 are exactly what the root sibling
  rescues.

**Net: cat10's real per-event yield ≈ cat9 + a small d0 rescue, once you strip the
trajectory + sibling-stop denominators.** cat10 is NOT on HEAD (grep 0 for
`FR13_CAT10_ROOT_SIBLING`/`is_cat10`; archived only on remote `origin/fr13-cat10-archive`)
so it must be re-derived as a NEW exact-match shape like cat3w (new `_fr10_is_cat10`
guard + `_fr10_cat10_choices` + packing branch, ~15-30 lines), NOT revived from the
archive flag.

### 4.4 The candidate trees (exact `tree_choices`, infra-reuse, ~15-30 lines each)
All sorted `(len,path)`; downstream consumers (parent/ancestry masks, committer path
enum, eager-pack replay rows, conv-fusion prior windows) AUTO-ADAPT off SPEC_CONFIG
`tree_choices` (patcher `:11000-11001`); only the drafter PACKING is hand-rolled. Each
new shape = one exact-match guard (`_fr10_is_<name>`, like `_fr10_is_cat3w` `:11026`) +
a `torch.stack` packing in `(len,path)` order (like cat3w `:11515-11538`); default cat9
untouched; disengagement RAISE (`:12005`) intact; launcher auto-derives
`num_speculative_tokens = len(TREE)`. The committed infra is code-verified
(`_fr10_cat3w_choices` `:11005`, `_fr10_chain3_choices` `:11002`,
`_fr10_caterpillar_choices` cat9 `:10984`, `_fr10_spine_steps`/`_fr10_leaf_steps`
`:11036-11045` auto-adapt, root-sibling capture `torch.topk(_fr10_logits,2)[:,1]`
`:11181`).

| rank | shape | TREE (sorted len,path) | N / pad | committed depth | role |
|---|---|---|---|---|---|
| **R4** `cat6root` | full spine + root sib, ZERO off-spine leaves | `[(0,),(1,),(0,0),(0,0,0),(0,0,0,0),(0,0,0,0,0)]` | 6 / **pad8** | 5 | **LEAD CANDIDATE** |
| R5 `cat6root_g` | R4 with root sibling CONF-GATED | (R4, gated) | 6→5 / pad8 | 5 | leanest deploy of R4 |
| R1 `cat7rd` | remove-deep (drop d5 spine + both d5 leaves) + add-root + 3 leaves | `[(0,),(1,),(0,0),(0,1),(0,0,0),(0,0,1),(0,0,0,0)]` | 7 / **pad8** | 4 | accept-vs-co-residency knee |
| R2 `cat7rd_g` | R1 root sibling CONF-GATED | (R1, gated) | 7→6 / pad8 | 4 | deploy of R1 |
| R3 `cat3w` (ON HEAD, no code) | spine3 + root + d1 | `[(0,),(1,),(0,0),(0,1),(0,0,0)]` | 5 / **pad8** | 3 | pad8-floor SPEED probe; LOSSY (~17 flips) — speed/depth ref ONLY, not a lossless cand |
| — cat10 (ungated) | cat9 + root sib | (§4.3) | 10 / pad16 | 5 | d0-rescue building block; no pad win |
| — cat10-gated | cat9 + gated root | (§4.3, gated) | 10→9 / pad16 | 5 | cat10 deploy form |

**R4 build:** `_fr10_spine_steps=4`, `_fr10_leaf_steps={}`, root sibling slot 1 (reuse
cat3w `_fr10_root_leaf_token` verbatim). **R1 build:** `_fr10_spine_steps=3`,
`_fr10_leaf_steps={1,2,3}`, root sibling slot 1. **cat10 build:** cat9 choices verbatim
(9 nodes) + `(1,)` in slot 1; ungated asserts `tok/draft==10`, gated asserts ∈ {9,10}.

### 4.5 The L3 conf-gated ROOT-SIBLING emit (the free lever; applies to R2/R5/cat10-gated)
Emit `(1,)` ONLY when root top-1 margin = `logit[rank1] − logit[rank2] < tau` (~ln2)
using the already-materialized `torch.topk(_fr10_logits, 2)` (patcher `:11166`/`:11181`)
= one scalar compare/event, **zero extra forward** (no extra lm-head). 62% of rejects
are step-0; root runner-up is truth 27% on those near-ties. On confident roots serve
clean cat9/spine (no sibling); on near-ties add the sibling that pays — keeps the
+0.035 d0 rescue WITHOUT the (mostly-artifact) deeper dilution.

**REALIZABILITY caveat (class 9):** all R1-R5 use **rank-2 width ONLY** (root + spine
runner-ups, all captured by `topk(...,2)`); NO rank-3 node exists in the drafter (only
`[:,1]` captured) — do NOT propose 3-way fans. Each new shape FAILS LOUD via the
disengagement RAISE (`:12005`) until its packing branch + guard is added; NEVER
`FR10_ALLOW_LINEAR_FALLBACK`.

### 4.6 CAPTURE / B=4 shippability (each reshape tree)
Each reshape tree MUST CUDA-graph-capture + behave at B=4 co-residency, and lossless
MUST be RE-CONFIRMED at B=4 (bit-exact at B=1 does NOT imply B=4). The conf-gated root
sibling makes draft-toks/event VARIABLE — but that is a **host-side branch on a logit
scalar BEFORE the captured forward** (the gate appends slot 1 to the drafter pack, no
host sync inside the captured region), so it is capture-safe AS LONG AS the
variable-shape pack is realized pre-capture. Flag any shape that cannot B=4/CUDA-capture
as NOT shippable. Gated shapes make the engagement assert a SET (`tok/draft ∈ {6,7}` or
`{9,10}`), NOT a fixed int.

### 4.7 Per-candidate speed/accept/lossless prediction (depth-matched, INFERRED-labeled)
All ms INFERRED (census/literature-anchored; per-forward nsys export empty). MEASURED =
the pad8/pad16 step + the M-invariant lm-head. Flips each-vs-OWN-recurrent-oracle
(depth-agnostic); accept depth-MATCHED + PAIRED. Break-even: for a pad8 shape to beat
native E5 on TPS it needs accept ≳ `(s_pad8/s_native) × native_accept ≈
(0.222/0.218) × 3.08 ≈ 3.14` = roughly cat9's accept.

- **R4 cat6root [LEAD]:** s/fwd ~chain5 pad8 (sheds pad16 step, −40+ ms vs cat9); accept
  = full-spine (cat9 d0-d4) + d0 rescue +0.035, root sib strict-invisible, vs **native
  E5**; lossless BEST (~chain5 regime 1-5 flips + strict-invisible sparse sib). **Most
  likely > native** (pad8 s/fwd + near-cat9 accept). OPEN risk: whether the full spine
  WITHOUT the 4 deep leaves holds accept ≥ break-even (the deep leaves carried SOME
  accept; the paired E5 capture decides).
- **R5 cat6root_g:** s/fwd ≤ R4; accept = R4 on near-ties, no dilution; lossless ≥ R4
  (sparser sib); net ≥ R4.
- **R1 cat7rd:** s/fwd ~chain5 pad8 + 1 row (free); accept = d0-d3 + 3 leaves + d0
  rescue, vs **native E4 (UNMEASURED)**; lossless held @ cat9 point, 3 shallower leaves
  = some co-residency (< cat9's 4 deep); likely > native.
- **R2 cat7rd_g:** s/fwd ≤ R1; vs **native E4 (UNMEASURED)**; net ≥ R1 (deploy form).
- **R3 cat3w (HEAD):** s/fwd ~chain5 pad8 (lowest rows); accept LOW (d0-d2), vs **native
  E3 (UNMEASURED; chain3 was 2.27)**; LOSSY (~17 flips) = SPEED ref only.
- **cat10 ungated:** s/fwd ~cat9 (+1 row free, SAME pad16 ≈ +2.9 ms vs cat9, the
  MEASURED +1-row tax); accept = cat9 + d0 rescue +0.035 (PAIRED,
  sibling-stop-de-confounded, NOT the artifact 2.932), vs **native E5**; lossless held =
  cat9's 22 (FLAT; root sib strict-invisible; prior boot confirmed 22==22). Net ~cat9
  (NO pad win) — value is as the d0-rescue building block BOLTED ONTO a pad8 shape.
- **cat10-gated:** s/fwd ≤ cat10 ~cat9; accept = cat9 + sparse d0 rescue, no dilution;
  net ~cat9 + a hair.

**The pad8 shapes (R1/R2/R4/R5) are the real TPS lever** (shed the dominant +42-46 ms
pad8→pad16 step); cat10 does NOT win on s/fwd (still pad16) — its only gain is the
+0.035 d0 rescue, valuable as a building block on a pad8 shape.

---

## 5. SHARED ACCOUNTING-CORRECT MEASUREMENT — TWO TIERS (the confound-free instrument)

Every candidate of every lever (OPT-1 OFF/ON, OPT-A OFF/ON, EACH reshape tree incl
cat10) is measured by the SAME two-tier protocol. Tier 1 = cheap dev iteration; Tier 2 =
the deployable judgment. Reuse `scripts/fr13_shape_gate.sh <name> "<TREE>"` (serialized
GPU, recover_host_memory + MemAvailable≥95-100 GiB + docker-empty before each boot,
locked pipeline flags, `num_spec` + `speculative_token_tree` AUTO-DERIVED from
`len(TREE)`, does NOT set `FR10_ALLOW_LINEAR_FALLBACK` so an unrealizable TREE fails
loud) and `scripts/fr13_launch_locked.sh` for the OPT-1/OPT-A flag toggles.

### 5.1 TIER 1 — DEV-ITERATION = B=1 EAGER `decode_seconds` (fast, cheap, per-change)
Boot once, **in-process A/B** (no cross-boot byte gate, feedback_no_cross_boot_byte_gate),
single locked boot; toggle the ONE flag under test (e.g. `FR13_COMMITTER_SYNCKILL` 0→1
with `FR13_GPU_COMMITTER=1` fixed; or `FR13_GB10_FP8_GEMV_CFG` 0→1; or the reshape TREE).
Pins identical both arms: `VLLM_BATCH_INVARIANT/LUMO_BATCH_INVARIANT_VLLM=0`,
`FR10_METRICS=0`, `GPU_MEMORY_UTILIZATION=0.82`, `MAX_NUM_SEQS=1`, `prompts_swe4` seed
1313 greedy temp 0.0; reset prefix cache + `torch.cuda.empty_cache` + verify
`nvidia-smi` between arms (gpu_mem_collection_between_experiments).

**ENGAGEMENT ASSERTS BEFORE ANY NUMBER (class 9, fail_loud_assert_engagement):**
tok/draft > 0 (`== len(TREE)` ungated, `∈ {realizable set}` gated — NOT a fixed int for
gated shapes), `has_tree_parent_indices`, `tree_sample_accept` fired; AND a per-lever
engagement needle (fires once in BOTH flag states, like `_fr13_eager_pack_needle`
`:6265-6291`): OPT-1 = confirm the device-input + side-stream path engaged when ON;
OPT-A = confirm the GB10 cfg dict was injected (not the stock default) when ON; reshape
= the tree topology realized. No number recorded if any assert fails. The patcher RAISES
"caterpillar drafter disengaged" (`:12005`) for any unbuilt shape (vacuous tree fails
loud, records nothing).

**WITHIN-BOOT DETERMINISM (class 8):** rep1≡rep2 byte-identical served streams, all
prompts, greedy AND t0.6.

**SPEED BASIS = `decode_seconds` RAW /metrics counter, NEVER TPS/accept/wall** (banned
as MEASURED facts; all per-forward ms are INFERRED until this clean run): per-forward
s/fwd = `vllm:request_decode_time_seconds_sum / vllm:spec_decode_num_drafts_total`,
per-request, metrics OFF, BI=0 pinned identical both arms. Record `verify rows` + `n_pad`
per arm to attribute the pad8-vs-pad16 step. `decode_seconds/draft` + paired per-depth
accept are the ONLY load-bearing numbers; TPS is DERIVED for reporting, NOT gated on.

**RUN-AHEAD MECHANISM CHECK (OPT-1 only):** census the main-thread block in
`memcpyAsync` ON vs OFF (target 91.9% → toward native 0.8%) via nsight
(`scripts/join_nsight_decode_metrics.py`, `LUMO_NSYS_TRACE=cuda,cuda-sw`) — the
diagnostic that confirms the sync was ACTUALLY killed (NOT a hand-rolled TPS
decomposition). OPT-A: optionally census the fp8 GEMM kernel-time delta; reshape: none.

**ACCEPT — DEPTH-MATCHED + PAIRED teacher-forced (the artifact-avoider):**
- **DEPTH-MATCH:** committed-depth-5 (cat9, cat10, R4/R5, OPT-1, OPT-A on cat9) →
  **native E5**; depth-4 (R1/R2) → **native E4**; depth-3 (R3) → **native E3** — E3/E4
  are UNMEASURED on the temp0/prompts_swe4/recurrent frame, **CAPTURE them before
  judging any d≤4 arm "slow"** (feedback_depth_matched_accept_compare; NEVER a d4 arm
  vs E5). The flip/lossless axis is depth-AGNOSTIC (each vs OWN oracle); only SPEED needs
  depth-match.
- **PAIRED teacher-forced per-event on a byte-identical served prefix** shared with the
  depth-matched native, NOT cross-trajectory whole-window (the cat10 trap, class-12
  confounded). Report PER-DEPTH accept RATE (d0..d4) + the within-arm **d0-rescue
  delta**, not whole-window accept/event.
- **De-confound the sibling-stop denominator** (gated/cat10 shapes): with `FR10_METRICS=1`
  + a per-node sibling-vs-spine d0 tag, REMOVE sibling-win (`accepted_len=1` root) events
  from the `per_pos[0]` denominator before computing d1|d0 (the per-node counter the
  prior cat10 boots LACKED).

### 5.2 TIER 2 — FINAL JUDGMENT = B=4 + CUDA-graph-captured + 4 SWE-Verified tasks + ~30 min/candidate
The DEPLOYABLE gate. EVERY candidate (OPT-1 OFF/ON, OPT-A OFF/ON, EACH reshape tree incl
cat10) gets this. B=4 changes co-residency; CUDA-graph capture is the deployed mode;
SWE-Verified 4 tasks = real workload; ~30 min = the denominator. The LOSSLESS gate is
RE-CONFIRMED at B=4 (co-residency changes it; bit-exact at B=1 does NOT imply B=4).
Before recording a Tier-2 number, confirm the lever CAPTURES at B=4 (hooks OFF) and
behaves at the B=4 M-tiles/co-residency:
- **OPT-1:** committer is eager-class (capture-safe by construction §2.5); confirm the
  side-stream/event runs outside the captured region at B=4 and the device→device
  writeback enlarges `n_elems` cleanly.
- **OPT-A:** confirm the GB10 cfg captures at the B=4 M-tiles (§3.3: M may exceed 32 →
  stock fallback; `BLOCK_SIZE_M=16` must tile B=4 M with 0 spill — re-run ptxas/dump);
  re-confirm RAW `out_vs_native_max_abs == 0.0` at the B=4 M (NOT the atol=1e-3 exit).
- **RESHAPE:** confirm each tree captures + behaves at B=4 co-residency; the conf-gate's
  variable pack is realized pre-capture (host-side scalar branch).

### 5.3 LOSSLESS GATE — held per-change throughout BOTH tiers (the binding instrument)
The gate for THIS change (in order), per feedback_fr13_lossless_compare_target:
1. **CPU/GPU byte-A/B** vs the bit-exact ORACLE. OPT-1: extend
   `scripts/fr13_gpu_committer_byte_ab_gate.py` with a DEVICE arm asserting
   `fr13_gpu_committer_device` outputs (after the side-stream event sync) byte-identical
   to the oracle across the 52-tree matrix, PLUS the device→device writeback into a mock
   `output_token_ids`/`accepted_tree_rows`/path buffer equals the legacy host-scatter
   element-for-element, PLUS the structural assert (`FR13_COMMITTER_SYNCKILL` default "0",
   gated under `FR13_GPU_COMMITTER`; OFF leaves `:6761` + legacy/first-draft + `:7366-7409`
   intact). OPT-A: `fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py` (PASSED CPU; re-read RAW
   max_abs at the live M-tiles). RESHAPE: structural — default cat9 untouched, new guard
   only fires on its exact shape.
2. **Same-seed BYTE-IDENTICAL served streams** OFF vs ON (or arm vs depth-matched
   reference), greedy t0.0 AND t0.6, `prompts_swe4` seed 1313, SAME boot (in-process
   A/B). For OPT-1/OPT-A this is OFF≡ON byte-identical (structural no-op). For reshape it
   is rep1≡rep2 within-boot determinism (the shape changes the stream by design; lossless
   is judged by the flip-vs-oracle in step 5, not stream-identity-to-cat9).
3. **accept/event unchanged** OFF vs ON, depth-matched, paired teacher-forced (NOT the
   class-12-confounded aggregate). OPT-1/OPT-A: unchanged (structural). Reshape: held at
   the cat9 operating point (Branch B's gate = "flips not regressed", need NOT improve).
4. **regular-decode pristine** (non-spec requests unaffected).
5. **The per-token argmax-vs-clean-recurrent-oracle probe** (`fr13_gold_margin_probe.py`
   / `fr13_recurrent_decode_oracle.py` / `fr13_oracle_stream_teacher_force.py`,
   `FR12_NO_SPECULATIVE_CONFIG=1`, FLASH_ATTN, teacher-forced max_tokens=1 per served
   position, thr 1.0 nat): **US vs native-E5, EACH vs its OWN no-spec RECURRENT decode
   oracle** — NEVER chunked-prefill / streamed-logprobs / serial-torch / a backend NAME
   (int-view, never atol). Scalar accept/event is necessary-not-sufficient
   (reference_scalar_metric_per_token_blindspot — it hid a real defect); the argmax probe
   is BINDING. OPT-1/OPT-A are structural transport/config changes so they MUST be a
   no-op on EVERY token (the probe verifies directly); reshape holds the cat9-operating-
   point flip count (each vs OWN oracle). Assert `spec_metrics_delta_during_oracle == 0`.
6. **RE-CONFIRMED at B=4** (Tier 2): co-residency changes losslessness; bit-exact at B=1
   does NOT imply B=4 — re-run the byte-A/B + served-stream + argmax probe at B=4.

**VERDICT:** TPS = (paired accepted tok/event) / (decode_seconds per fwd), arm vs native
E5 (GOAL = cat9-family strictly > native E5 at B=1). WIN iff **lossless HELD** (flips not
regressed vs the cat9 operating point, each-vs-own-oracle) AND **decode-TPS > native E5**
with accept ≥ the depth-matched break-even. Only on all-pass does a lever graduate from
default-OFF instrument toward a behavior-preserving bake-in (default-ON path still
byte-identical, feedback_flag_gate_metrics_reuse_infra).

---

## 6. THE SEQUENCED GPU CAMPAIGN (order + dependencies; GPU serialized, AFTER prelaunch host-mem protocol)

GPU is the bottleneck (serialized); CPU design is done. Sequence by (a) what is BUILT vs
needs-build, (b) cheap-to-measure first, (c) which result de-risks the others. MAX 2
concurrent workflows (typically 1 GPU + 1 CPU); the GPU campaign is ONE serialized
queue. ALL runs under the prelaunch host-mem protocol (recover_host_memory +
MemAvailable≥95-100 GiB + docker-empty per boot).

**PHASE 0 — fresh same-boot oracle re-baseline + the UNMEASURED depth references (PREREQUISITE).**
First boot establishes, on THIS image, the recurrent-oracle flip floor for native E5 +
cat9 (the lossless bar) AND captures **native E3 and native E4** (UNMEASURED today;
needed for R1/R2/R3 depth-matched accept — feedback_depth_matched_accept_compare). Until
E3/E4 exist, NO d≤4 arm can be judged "slow." Cheapest, blocks the reshape arms.

**PHASE 1 — OPT-A fp8 GEMV (BUILT, cheapest, lowest-risk, de-risks both arms).**
OPT-A is already built + CPU-byte-A/B-passed; it is a pure flag toggle on the locked
boot with no topology change. Measure FIRST after Phase 0: Tier-1 B=1 `decode_seconds`
OFF vs ON on cat9 AND native (it speeds BOTH), then the §3.3 live checks (M-distribution
+ RAW bit-exact at B=4). It is the cleanest whole-system bandwidth win and it shifts the
s/fwd baseline that OPT-1 and reshape are measured against — so establishing it first
gives the truest break-even. If OPT-A wins, it stacks UNDER every later candidate (keep
it ON as the new baseline once Tier-2 + B=4 bit-exact confirm).

**PHASE 2 — OPT-1 sync-kill (needs build: `FR13_COMMITTER_SYNCKILL`).**
Build the G2.a device-input entry + G2.b side-stream readback + the byte-A/B device arm
(§2.3-2.4, §5.3.1) — CPU-gated FIRST (52-tree device arm + writeback equality +
structural default-OFF). Then Tier-1 OFF vs ON on cat9: the PRIMARY verdict is whether
ON reclaims the ~4-6 ms of the cat9 +6.5 ms tax (conservative 2.5-4 ms → ~220.7-222.2
ms; optimistic 6 ms → ~217-218 ms at/below native 218.2 ms), CONFIRMED by the run-ahead
census drop (91.9% → toward 0.8%). OPT-1 is the lever most directly aimed at the
remaining cat9 tax once OPT-A has lowered the GEMM floor. Independent of reshape (it acts
on the committer transport, not topology) so it can be measured on cat9 with OPT-A
ON-or-OFF.

**PHASE 3 — TOPOLOGY RESHAPE sweep (needs build: per-shape guards; measure with OPT-A+OPT-1 ON).**
Build the guards/packing for the candidates (~15-30 lines each); each FAILS LOUD until
built. Sweep order (most-promising / cheapest-to-judge first), each Tier-1 then Tier-2:
1. **R4 `cat6root`** (full-spine pad8 floor — the LEAD candidate; isolates "does the d0
   root rescue ALONE net-beat native at pad8");
2. **R1 `cat7rd`** (pad8 + 3 leaves — the accept-vs-co-residency knee);
3. **cat10 UNGATED** (the artifact-corrected re-measure, per-node sibling counter ON —
   settles the 2.932-was-artifact claim);
4. then the GATED deploy forms (**R5 cat6root_g / R2 cat7rd_g / cat10-gated**).
Measure reshape with OPT-A (and OPT-1 if it won) ON, since those are the deploy baseline;
but on any surprising reshape result, re-isolate by toggling OPT-A/OPT-1 OFF to attribute.
`R3 cat3w` (on HEAD) only as the pad8-floor SPEED probe (LOSSY ~17 flips, not a lossless
candidate). chain5/cat9 only for the same-boot oracle re-baseline.

**PHASE 4 — FINAL JUDGMENT (Tier 2) on the winners + the stacked combination.**
For each candidate that cleared Tier-1 + lossless, run the B=4 + CUDA-captured + 4
SWE-Verified + ~30 min judgment with lossless RE-CONFIRMED at B=4. Then run the STACKED
best combination (OPT-A ON + OPT-1 ON + the winning reshape tree) at Tier-2 to confirm
the levers compose without a co-residency surprise and the combined arm clears native E5
on TPS with lossless held at B=4.

**Dependencies / why this order:**
- Phase 0 BLOCKS Phase 3 (E3/E4 references) and gives the lossless bar for all.
- OPT-A first because it is BUILT (zero build latency), cheapest (flag toggle), lowest
  risk (lossless by construction), and it RE-BASELINES s/fwd for everything downstream.
- OPT-1 second because it needs a build but is a focused transport change measured on the
  existing cat9 topology; it does not depend on reshape.
- Reshape last because each shape needs a build AND the depth-matched references from
  Phase 0; it is also where the accept/lossless trade is most uncertain (the paired E5/E4
  capture is decisive).
- All three are INDEPENDENT subsystems → measured separately first, stacked last (Phase 4).

**MEASURE-FIRST = OPT-A** (built + cheapest + re-baselines), gated behind the Phase-0
oracle/depth-reference boot.

---

## 7. SHIPPABILITY FLAGS (levers that cannot B=4/CUDA-capture are NOT shippable)
- **OPT-1:** capture-safe by construction (committer eager-class; side-stream outside any
  captured region). SHIPPABLE pending Tier-2 B=4 bit-exact re-confirm. NOT-shippable only
  if the device→device writeback or the lazy-host shim introduces a sync inside a captured
  region (it does not, by design) — verify at Phase 2 Tier-2.
- **OPT-A:** SHIPPABLE iff §3.3 (a) the B=4 M-tiles stay ≤32 often enough to win AND
  (b) RAW bit-exact == 0.0 at the B=4 M. FLAG NOT-shippable if B=4 M routinely exceeds 32
  (no speedup) or any non-zero raw drift at B=4.
- **RESHAPE:** each tree SHIPPABLE iff it CUDA-captures + behaves at B=4 co-residency +
  lossless re-confirmed at B=4. The conf-gate is capture-safe (pre-capture host-side
  branch). FLAG any shape that cannot.

---

## Cross-refs
`FR13_SPEED_TUNING_PLAN_BRANCH_B.md` (Branch B full design, HEAD 3c846d5d),
`FR13_GPU_COMMITTER_BIND.md` (OPT-1 G2 caveat), `FR13_BEAT_NATIVE_SPEED_DESIGN_BIND.md`
(91.9% vs 0.8% census, committer-sync residual), `FR13_B1_SPEED_ATTRIBUTION_BIND.md`
(per-kernel attribution, FIX-1 double-head), `FR13_BV_SPILL_VERDICT.md` (OPT-A/scan
num_warps + the raw-max_abs-not-atol caveat #5), `FR13_GB10_FP8_GEMV_CFG` /
`e90de7ef` (OPT-A config + CPU byte-A/B), `FR13_CAT10_BIND.md` /
`FR13_CAT10_INVESTIGATE_BIND.md` (cat10 artifact decomposition + conf-gate lever),
`FR13_RESHAPE_AB_RECURRENT_BIND.md` (depth+1/width+16 carrier), `FR13_RESHAPE_EXHAUSTED_BIND.md`,
`FR13_RESHAPE_SHAPE_DESIGN.md` (rank-2-only, infra), `FR13_SPEED_HISTORY_RECONCILE.md`
(M-invariant lm-head, N_PAD step), `FR13_SPEED_TAX_BASELINE.md` (per-node +42-46 ms).
CODE: `scripts/fr10_phase4_patch_vllm_tree_gdn.py` (EAGER_PACK `:6745`/`:6758`/`:6760`/
`:6761`/`:6762`, device sources `:6704-6719`, committer loop `:6877-7137`, writeback
`:7366-7409` + GDN `:7440-7513`, diag globals `:7420-7424`, committer returns device
`:8536`/`:7977`/`:8515`, hook `:14248`/`:14304-14358`, eager-pack stage `:6222-6247`,
needle `:6265-6291`, choices/leaf_steps `:10984-11045`, root capture `:11166`/`:11181`,
cat3w packing `:11515-11538`, disengage RAISE `:12005`), `scripts/fr13_gpu_committer_kernel.py`
(`:13-18`, `_build_device_layout :348-404`, triton `:407-484` re-sync `:471-474`,
dispatch `:577-622` — no `fr13_gpu_committer_device` yet),
`scripts/fr13_gpu_committer_byte_ab_gate.py` (add DEVICE arm),
`scripts/fr13_gb10_fp8_gemv_cfg_byte_ab_gate.py`, `scripts/fr13_shape_gate.sh`,
`scripts/fr13_launch_locked.sh`, `scripts/join_nsight_decode_metrics.py`. VLLM (pinned
3dbe092e via `scripts/vllm_src.sh`, NOT /tmp): `rejection_sampler.py:159-232`,
`gpu_model_runner.py:227-292`. MEMORY: feedback_fr13_lossless_compare_target,
reference_fr10_speed_measurement_pitfalls, feedback_dont_handroll_speed,
reference_scalar_metric_per_token_blindspot, feedback_no_cross_boot_byte_gate,
feedback_depth_matched_accept_compare, feedback_fail_loud_assert_engagement,
feedback_check_artifact_before_concluding, project_fr13_tree_reshape_unifying_lever,
gpu_mem_collection_between_experiments, feedback_wy_parked_dont_revive (WY out of scope),
reference_monitor_pathspec_commit_shared_branch.
NOTE: the plan's stale-cited hook line `:14304-14355` and committer `:5780-5879` are
CODE-GENERATOR string offsets; the live file anchors are committer loop `:6877` and hook
def `:14248` — verified live this session.
