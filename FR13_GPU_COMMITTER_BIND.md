# FR13 OPT-1 — GPU-resident tree committer (FR13_GPU_COMMITTER), FIRST DRAFT

Date 2026-06-14. Branch `fr13-speedfix`. CPU-only first draft (no GPU boot).
Design: [[FR13_BEAT_NATIVE_SPEED_DESIGN_BIND]] (the committer SYNC, not graph
nodes, is the dominant beat-native residual). Flag **FR13_GPU_COMMITTER**,
DEFAULT-OFF. Mirrors FIX-1/2/3 gating (default path byte-identical with the flag
off).

## Why
The greedy tree committer `_lumo_tree_path_lcp_max_greedy_sample`
(`scripts/fr10_phase4_patch_vllm_tree_gdn.py`, the integer block) decides
accept / path-LCP / bonus as a **pure-Python loop on the host** over the synced
committer inputs. That host loop is gated by a packed DtoH + `cuda.synchronize()`
that census-measured blocks the MAIN launching thread **91.9%** of the verify
window vs native's 0.8% — so the tree path loses native's async run-ahead.

The committer decision is **pure integer**: `drafts[node] == parent_targets[node]`
compares, a parent walk, an LCP scan, an earliest-leaf strict-`>` tie-break, and a
3-way bonus-source select. None of it is float / reduction / order-sensitive, so
it can move to the GPU with NO loss.

## What shipped (this commit)
1. `scripts/fr13_gpu_committer_kernel.py`
   - `fr13_gpu_committer_oracle` — bit-exact pure-Python re-statement of the
     committer SERVING logic (FR13_TREE_BONUS_SELF=1, no diagnostic/force-spine
     branches). The reference the kernel transcribes.
   - `_fr13_committer_kernel` — the **Triton integer kernel**: per request (grid
     = num_requests), walks every leaf's root→leaf path, scores LCP, keeps the
     strict-`>` earliest winner, emits the accepted draft prefix + the 3-way
     bonus token, and writes `accepted_rows` / `accepted_lens` / `out_tokens`.
     Pure integer loads/compares; no `tl.dot`, no float, no reduction.
   - `fr13_gpu_committer_triton` — host launcher: fixed-stride device layout
     (MAX_NODES) + kernel launch + side-stream-able readback.
   - `fr13_gpu_committer` / `fr13_gpu_committer_full` — dispatch: Triton when
     CUDA+Triton present, else the bit-exact CPU oracle (so the contract is
     CPU-testable). `_full` also returns `accepted_node_paths` /
     `accepted_token_rows` (re-derived from the same winning path) for the GDN
     durable-state advance downstream.
2. Hook: `_patch_rejection_sampler_gpu_committer()` in
   `scripts/fr10_phase4_patch_vllm_tree_gdn.py`, registered AFTER the LCP
   committer patcher (both on `REJECTION_SAMPLER_PATH`). Flag-gated, default-OFF:
   reads `FR13_GPU_COMMITTER` (default "0") just before the committer loop; when
   ON (and not in a diagnostic mode: not force-spine, not argmax-gate, bonus_self
   only) it fills the five committer products from the kernel module and makes
   the legacy loop iterate an EMPTY list (the legacy per-node Python body stays
   byte-for-byte present but never runs). **Flag-off, `_fr13_gpu_commit_counts IS
   counts` → the legacy loop runs unchanged → byte-identical default path.**
   Idempotent (sentinel `# FR13_GPU_COMMITTER`), version-guarded (raises if the
   committer loop anchor drifts).
3. Gate: `scripts/fr13_gpu_committer_byte_ab_gate.py` (CPU-only, boot-free).
   Compares ORACLE + FULL (and, if a GPU is present, TRITON) against a verbatim
   in-gate copy of the committer serving logic across a 52-tree matrix
   (production 9-node caterpillar at every accept depth, pure spine, root-fan,
   reject@2 / reject@root, LCP ties, multi-request batches, 40 randomised trees).
   It also asserts the hook is DEFAULT-OFF and the legacy block intact.
   **RAN: 52/52 byte-identical, exit 0** (TRITON arm skipped on this CPU host).

## NEEDS LIVE GPU ITERATION (the "first draft" caveats)
The decision logic + gate are complete and CPU-proven. These require a GB10 boot:

- **G1 — Triton kernel byte-A/B on GPU.** Run the gate on a CUDA host so the
  TRITON arm exercises `_fr13_committer_kernel` and confirms it matches the
  oracle/REF byte-for-byte. The kernel uses nested `while` walks (re-walking
  ancestors per LCP position to stay array-free per program); on GPU, profile
  whether the O(depth^2) re-walk is acceptable at the deployed tree depth (≤6)
  or whether a per-program shared-memory path array (depth ≤ MAX_NODES) is
  faster. Either is lossless; pick on measured kernel time.
- **G2 — kill the sync (the actual speed win).** This draft still does a host
  `.cpu().tolist()` readback of the kernel outputs. The win only lands when that
  readback moves to a **non-gating side stream + CUDA event** so the MAIN thread
  never blocks (restores run-ahead). Wire the committer-output copy onto the
  existing FR13_EAGER_PACK side-stream/event machinery (`_fr13_eager_pack_stage`)
  and confirm via census that main-thread memcpy-block drops from 91.9% toward
  native's ~0.8%.
- **G3 — CUDA-graph conditional-node / `torch.cond` accept branch.** To keep the
  commit inside the captured graph (OPT-C synergy), express the data-dependent
  accept/bonus select as a CUDA-12.4 graph conditional node or `torch.cond`.
  First draft runs eager (kernel launch + readback); the in-capture form is the
  follow-on. Confirm the kernel graph-captures (it is Dynamo-opaque Triton, so it
  should) at B=4.
- **G4 — variable node_count packing on the serving path.** The host layout
  builder (`_build_device_layout`) currently runs in Python; on the serving path
  the per-request node_count is small and bounded — fold the packing into the
  existing on-device tree-metadata builder so no host loop touches the inputs.
- **G5 — end-to-end gate.** The class-9/class-10 byte A/B used for FIX-1/2/3:
  boot the tree server twice, same seed/prompt, `FR13_GPU_COMMITTER=0` vs `=1`,
  assert byte-identical decode streams (greedy + t=0.6) and unchanged
  accept/event. Then measure s/fwd vs native E5 (FLASH_ATTN MTP-5) to confirm the
  reclaimed run-ahead.

## Losslessness summary
Pure-integer, location-only host→Triton move. No float, no reduction, no
reorder. The byte-A/B gate proves token-for-token / row-for-row identity to the
legacy committer's serving decision across 52 trees CPU-side; the GPU kernel is a
faithful transcription of that same oracle. DEFAULT-OFF, default path
byte-identical with the flag off.
