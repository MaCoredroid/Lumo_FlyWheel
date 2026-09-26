# A / re-run of P4' on the PORTED head of vllm-project/vllm PR #55122

Date: 2026-09-17. Working dir: `/home/mark/shared/p4prime-55122/port/`.
Device: NVIDIA GB10 (DGX Spark), sm_121, 48 SMs, `sharedMemPerBlockOptin` = 101,376 B,
driver 590.48.01, CUDA 13.0 V13.0.88. Read-only on GitHub; nothing posted, nothing pushed.
GPU wall time ~7 min (routing + probes + 90-cell matrix), all under
`flock /home/mark/shared/exp54928/gpu.lock` with bounded `timeout`s.
`nvidia-smi --query-compute-apps` showed **no other compute process** on the device during
this run (unlike 2026-09-11, which ran alongside another workload).

## RESULT: POSITIVE, and unchanged by the port

The port does not touch the code path we validated. The measured routing window, the
device-side fallback entry, and the full result matrix all reproduce the 2026-09-11 numbers
exactly. One separate **porting defect** was found by reading the port (see §8); it lives in
the FilteredTopK arm, which is unreachable on this device and was therefore not exercised.

---

## 1. Source

```
mkdir -p /home/mark/shared/p4prime-55122/port
git init -q vllm-pr55122-port && cd vllm-pr55122-port
git fetch --depth=1 https://github.com/vllm-project/vllm.git pull/55122/head
git checkout -q FETCH_HEAD
```

`git rev-parse HEAD` = **`85f61e24bdfd84c072f300df8512855d4b90a223`** — matches the pinned port head.
Subject: "[Kernel] Restore the hist4096 include and alias the port dropped", Jürgen Schmied,
2026-09-17 12:10:17 +0200. PR state OPEN, `headRefOid` 85f61e24…, 3 changed files
(`persistent_topk.cuh`, `topk.cu`, `tests/kernels/test_top_k_per_row.py`).

sha256 of the kernel sources as fetched, alongside the previously tested head `7cfd04a3`:

| file | 7cfd04a3 (tested 09-11) | 85f61e24b (this run) |
|---|---|---|
| `persistent_topk.cuh` | `aa48de44…88ee9cb1` | **`e5f24542f4471a85b067d9d425e999a9623c4eabf5fe161ff89259a9503494bb`** |
| `topk.cu` | `68b6bcd5…e2b6d7ff` | **`8f82caf8a9cd26440404311835cae3dd849b2f627813ee607ad708798ecec372`** |
| `sampled_topk.cuh` | (absent) | `31c06513b26ab0f1ebb95206c267d2840524f30af4cd468160aeff7a3906b89b` |
| `topk_histogram_4096.cuh` | `b1c8901d…0d333231bd3` | `b1c8901d5b580fade9cd57e8d0bd1920c8c36908a8ede174393c20d333231bd3` (unchanged) |
| `cooperative_topk.cuh` | `fbade41d…0e48b1a4d51` | `fbade41dc89181e661a11c0f1d7946a54bdd08e9792645529bad50e48b1a4d51` (unchanged) |
| `cooperative_topk.cu`  | `9224b4e2…9dd7daec33` | `9224b4e281ff5ab91e56701d716f2525799a5e4deb41aa4dff20499dd7daec33` (unchanged) |

Line counts: `persistent_topk.cuh` 1548 → 1793, `topk.cu` 390 → 406.
Diffs saved: `port/persistent_topk_cuh.diff`, `port/topk_cu.diff`.

## 2. What actually changed in the port (step 2 of the task)

**Nothing in the path under test.** Concretely:

* **`namespace vllm::persistent` is BYTE-IDENTICAL** between the two heads. Old lines 14–1321
  and port lines 20–1327 hash to the same sha256 `2def82a7919c14e084078716341c8fc407bd72d16c6d79fe519febacad7c89d9`.
  That region contains `persistent_topk_kernel`, `det_select_row`, `det_select_row_bytes`,
  `det_select_row_fixed_bytes`, `RadixRowState`, `PersistentTopKParams`, and every constant
  (`RADIX_THRESHOLD`, `kSmemMedium`, `kFixedSmemLarge`, `kDetMaxCtasPerGroup`, `kThreadsPerBlock`).
  **`force_single_cta`, `det_select_row`, `RADIX_THRESHOLD`, `max_chunk_elements` and the
  >64-CTA check did not move, rename, or change.**
* **The `else` (persistent) branch of `launch_persistent_topk()` in `topk.cu` is BYTE-IDENTICAL**
  — both heads' branch text hashes to `6970950770f87774a4f06fa948446992354ba0f9a4b6c10a340feb77d85015ea`
  (279 lines). The launcher did not move; it is still `launch_persistent_topk<TopK>` in the
  anonymous namespace of `csrc/libtorch_stable/topk.cu`.

The four real changes, all outside that path:

1. `topk.cu`: `#include "sampled_topk.cuh"` and a **new first arm** in the dispatch chain, ahead
   of FilteredTopK:
   `if (num_rows > 64 && max_seq_len >= vllm::sampled_topk::kMinSampledLength<TopK> && max_smem_per_block >= 144*1024)`
   → `sampled_topk_kernel<TopK>`. Unreachable here (101,376 < 147,456), and our grid uses rows ≤ 4.
2. `persistent_topk.cuh` line 12: `#include "topk_histogram_4096.cuh"` restored (commit 85f61e24b).
3. Inside `namespace filtered_topk`: `namespace hist4096 = topk_histogram_4096;` alias restored,
   plus the upstream refactor's **new `__device__` helper `filtered_topk_row<...>`** (+ its
   `FilteredTopKStorage<MAX_K>` struct, ~250 lines), retained intact with a `CheckOverflow`
   template parameter. Its only consumer is `sampled_topk.cuh` (lines 160, 164).
4. `FilteredTopKUnifiedKernel` — the thin `__global__` wrapper — now contains only the
   deterministic `det_select_row` call (the PR's substitution moved from the merged function to
   the wrapper). **Its signature lost two parameters**: `uint32_t max_seq_len` and
   `uint32_t smem_bytes` (8 params → 6). See §8.

Because the persistent launcher and kernel are byte-identical, the 2026-09-11 transcription is
still a valid line-by-line transcription; only the dispatch chain ahead of it gained an arm.

## 3. Harness (step 3)

`port/harness/p4_harness_port.cu`, sha256 `ee891afc1fb05ecc83aafc62d7d83f0b8ea6d2ba3c028a8b67795f9a6b901b16`,
505 lines, derived from the 09-11 `p4_harness.cu` (`4b24c447…`). It `#include`s the **port's
`persistent_topk.cuh` unmodified, byte-for-byte as fetched** (via `-I` at the clone, no copy),
plus `sampled_topk.cuh` so the new dispatch arm can be transcribed rather than assumed away.

Every substitution against `launch_persistent_topk()` is documented in the file header:

* **S1** `torch::stable::Tensor` accessors → raw device pointers + explicit `num_rows`, `stride`,
  `max_seq_len`, workspace-size scalars (`logits.size(0)`, `logits.stride(0)`, `workspace.numel()`
  become parameters).
* **S2** `STD_TORCH_CHECK(...)` → non-throwing `REJECT(...)` / `REJECT_EXPECTED(...)` recorder.
  **Only** the `ctas_per_group > kDetMaxCtasPerGroup` check is `REJECT_EXPECTED`; every other
  host rejection is UNEXPECTED, prints `REJECTED_PRELAUNCH_UNEXPECTED` and exits 4
  (Codex protocol correction — the 09-11 build classified all rejections alike).
* **S3** `DeviceGuard` + `get_current_cuda_stream()` → device 0 and the default stream;
  `get_device_prop()` → `cudaGetDeviceProperties(&prop, 0)`.
* **S4** the three arms unreachable on this device are not modelled and are recorded as
  UNEXPECTED rejections if ever selected: the new `sampled_topk` arm (`num_rows>64 &&
  optin>=144 KiB`), the FilteredTopK primary arm (`num_rows>32 && optin>=128 KiB`), and the
  FilteredTopK overflow fallback (`optin>=128 KiB`). None was ever selected.
* **S5** added routing diagnostics (the `Diag` struct) and an explicit `cudaDeviceSynchronize()`
  after launch. **No arithmetic is altered** — every step (effective_max_smem tiering, vec_size,
  `cudaFuncGetAttributes` static-smem subtraction, `max_chunk_elements`, `ctas_per_group`,
  `chunk_size`, det_want/dyn_cap raise, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`,
  headroom reservation, `num_groups`, `total_ctas`, the fallback edit, the `cudaMemsetAsync`
  of `RadixRowState`, `cudaFuncSetAttribute` + launch) is reproduced in the same order with the
  same constants.

Protocol carried over unchanged: output buffer poisoned with `-424242` before every launch
(residual poison = FAIL); finite float32 inputs only; exact stable reference
(`std::stable_sort` value-descending / index-ascending, first k, then `std::sort` ascending);
any mismatch fails; a CUDA error exits 3; bounded `timeout` on every subprocess; the sweep
preserves unfiltered stdout+stderr in `matrix_raw.log` and records every cell's process exit
status as a `CELL_EXIT ... rc=` line, aborting on any nonzero.

Build (same toolchain as 09-11):

```
nvcc -std=c++17 -O2 -arch=sm_121a -I <port>/csrc/libtorch_stable -o p4_harness_port p4_harness_port.cu
```
rc=0; only the pre-existing benign `ptxas warning: ... .minnctapersm will be ignored` lines.

| artifact | sha256 |
|---|---|
| `harness/p4_harness_port.cu` | `ee891afc1fb05ecc83aafc62d7d83f0b8ea6d2ba3c028a8b67795f9a6b901b16` |
| `harness/p4_harness_port` (pristine) | `8e5eb532489ae5e2ed0e14c2648d48cf5078107a4676108e626c4b9226d89613` |
| `harness/p4_harness_port_instr` | `50dcaf5350e426ba1187e9f7f8a68581937f0ff7108ca9b86630f59752267f96` |
| `harness/sweep_port.sh` | `51aa8b9070d127756042f938b3fbd84fbb34641433f3fc351335244b797e1495` |
| `instr/persistent_topk.cuh` | `bf181bad8c103db831d17bfe035821b612f49cdc9d7632685bf3bd8e8135bf7b` |

`strings p4_harness_port | grep -c INSTR` = **0**; on `p4_harness_port_instr` = **2**.
All validation numbers in §6 come from the pristine build.

## 4. Diagnostics read back from the built kernel

```
DEVICE name=NVIDIA GB10 cc=12.1 sms=48 sharedPerBlockOptin=101376 sharedPerBlock=49152
       sharedPerMultiprocessor=102400
CONST  kThreadsPerBlock=1024 RADIX_THRESHOLD=22016 kSmemMedium=35968 kFixedSmemLarge=2080
       kDetMaxCtasPerGroup=64 det_select_row_fixed_bytes=6400 (k=512/1024/2048 alike)
MEASURED (rows<=4):
       static __shared__ of persistent_topk_kernel<K,VS> = 4256 B (all K, all VS)
       effective_max_smem = min(101376, kSmemMedium) = 35968
       available_for_ordered = 35968 - 2080 - 4256 = 29632
       max_chunk_elements = 29632/4 = 7408  (identical for vec_size 1, 2 and 4, since 7408%4==0)
       det_want = 6400 + 4*22016 = 94464 ; dyn_cap = 101376 - 4256 = 97120 ; smem_size = 94464
       occupancy @1024 thr / 94464 B = 1 ; hw_resident_cap = 48*1 = 48 ; headroom = 1
```
Identical for k ∈ {512,1024,2048} and rows ∈ {1,4}. **Identical to 09-11.**

## 5. Routing window (step 4) — `logs/route_rows1_k2048.log` (157 widths), `logs/route_boundaries.log` (78 cells)

`ctas_per_group = ceil(active_width / 7408)` for `active_width > RADIX_THRESHOLD = 22016`.
Since occupancy == 1, the fallback predicate `total_ctas > hw_resident_cap` fires exactly when
`ctas_per_group > 48`.

| `ctas_per_group` | n (stride = max_seq_len = n) | selection |
|---|---|---|
| ≤ 48 | n ≤ 355,584 | cooperative multi-CTA, `force_single_cta = 0` |
| **49 … 64** | **355,585 … 474,112** | **LOW-SMEM FALLBACK: `force_single_cta = 1`, `ctas_per_group → 1`, `chunk_size → 7408`, `total_ctas = min(48, num_rows)`** |
| ≥ 65 | n ≥ 474,113 (first measured: 474,114) | pre-launch host rejection `persistent_topk: ctas_per_group 65 exceeds 64` |

Measured endpoints: last cooperative n = **355,584**; first fallback n = **355,585**; last
fallback n = **474,112**; first measured rejection n = **474,114** (474,113 is the arithmetic first, not probed). Verified for rows ∈ {1,4} × k ∈
{512,1024,2048} — 78 boundary cells, all agreeing.

This run improves on 09-11's boundary claim. Because `max_chunk_elements` = 7,408 for **all three
vec_sizes** (it is divisible by 4), the window is the same for every stride parity, not just
multiples of four: n = 355,585 (stride odd → vec_size 1) and n = 355,586 (vec_size 2) both route
to `force_single_cta = 1`, and n = 474,114 (vec_size 2) is already rejected. The 09-11 report's
hedge to "widths divisible by four" was therefore conservative; on this device the window is
n ∈ [355,585 , 474,112] for any stride. Still scoped to `stride == max_seq_len == n`,
float32, contiguous rows, rows ≤ 4.

## 6. Device-side proof of fallback entry (step 4) — `logs/instr_probe.log` (SAVED this time)

`instr/persistent_topk.cuh` is the port header plus **exactly 9 added lines**: one
`#include <cstdio>` and two `printf` probes — one immediately after
`const bool cached = ...` in `det_select_row`, one as the first statement inside
`if (params.force_single_cta || seq_len <= RADIX_THRESHOLD)` in `persistent_topk_kernel`.
No logic change. Verbatim probe output:

```
### PROBE rows=1 k=2048 n=355584 pat=random       (control, one CTA short)
  (no INSTR lines at all)  outcome=PASS force_single_cta=0
### PROBE rows=1 k=2048 n=355588 pat=random
INSTR single_cta_branch force_single_cta=1 seq_len=355588 ctas_per_group=1 gridDim=1 det_smem_bytes=94464
INSTR det_select_row n=355588 smem_bytes=94464 fixed=6400 CACHED=0
### PROBE rows=1 k=2048 n=400000 pat=random
INSTR single_cta_branch force_single_cta=1 seq_len=400000 ctas_per_group=1 gridDim=1 det_smem_bytes=94464
INSTR det_select_row n=400000 smem_bytes=94464 fixed=6400 CACHED=0
### PROBE rows=4 k=512 n=474112 pat=tie
INSTR single_cta_branch force_single_cta=1 seq_len=474112 ctas_per_group=1 gridDim=4 det_smem_bytes=94464
INSTR det_select_row n=474112 smem_bytes=94464 fixed=6400 CACHED=0
### PROBE rows=4 k=2048 n=474116 pat=random
  outcome=REJECTED_PRELAUNCH_EXPECTED
```

`CACHED=0` is the uncached global-rescan variant of `det_select_row` — the code the PR
substitutes for `top_k_per_row_decode`. The control width produces no probe output at all.

## 7. Result matrix (step 5) — pristine build, `logs/matrix_raw.log`

Grid: n ∈ {355584 (cooperative control), 355588 (first fallback ×4), 400000 (mid), 474112 (last
fallback), 474116 (first rejection ×4)} × rows ∈ {1,4} × k ∈ {512,1024,2048} × inputs ∈
{random U(−100,100) f32, tie-heavy randint(0,5), all-equal 1.5f} × 6 repeats.
**90 cells, 450 attempts (432 launches + 18 host rejections).** Seed 20260917.

| n | `force_single_cta` | cells | outcome |
|---|---|---|---|
| 355,584 | 0 (cooperative control) | 18 | **108 / 108 PASS** |
| 355,588 | **1** | 18 | **108 / 108 PASS** |
| 400,000 | **1** | 18 | **108 / 108 PASS** |
| 474,112 | **1** | 18 | **108 / 108 PASS** |
| 474,116 | n/a | 18 | **18 / 18 REJECTED_PRELAUNCH_EXPECTED**, message `persistent_topk: ctas_per_group 65 exceeds 64` (all 18 byte-identical) |

Totals: **432 PASS, 0 FAIL, 0 `exact=0`, 0 `repro=0`, 0 residual poison, 0 CUDA errors,
0 device faults, 0 UNEXPECTED rejections, 0 FATAL/STOP lines.** All 90 `CELL_EXIT` lines rc=0;
all 90 `SUMMARY` lines `all_pass=1`; `SWEEP_COMPLETE` reached.
"PASS" = output exactly equals the stable value-descending / index-ascending reference **and**
bit-identical to repeat 0 **and** zero surviving poison. This includes the all-equal rows (where
the answer is purely the index rank) and the tie-heavy rows at n = 474,112, where ~95 k elements
share the top value. No timings are reported (shared-device, no warm-up — they would not mean
anything).

## 8. Porting defect found by reading the port (NOT exercised — unreachable on this device)

`FilteredTopKUnifiedKernel`'s parameter list dropped `uint32_t max_seq_len` and
`uint32_t smem_bytes` (port `persistent_topk.cuh` lines ~1663–1668: 6 parameters, was 8), but
its launcher `filtered_topk::FilteredTopKRaggedTransform` — which the port does **not** touch —
still builds an **8-entry** argument array at port line 1756:

```cpp
void* args[] = {&input,     &output_indices, &lengths,    &num_rows,
                &top_k_val, &max_len,        &max_seq_len, &smem_size};
FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*)kernel, grid, block, args, smem_size, stream));
```

`cudaLaunchKernel` takes an untyped `void**`, so the two extra entries are silently ignored —
**no compile error, no runtime error**. Two consequences, both on ≥128 KiB devices only:

1. **The row-bound clamp is gone.** The 7cfd04a3 wrapper computed
   `row_bound = min(max_len, max_seq_len)` and clamped `lengths[bid]` into `[0, row_bound]`,
   with a comment stating that an oversized `lengths[bid]` "would otherwise read past the row".
   The port's wrapper is `const int length = lengths ? lengths[bid] : (int)max_len;` — unclamped
   and unguarded against negatives.
2. **The launcher's shared-memory sizing is dead.** The launcher still computes
   `smem_size = min(want, optin − static)` and launches with it — the comment directly above it
   explains this was added so `det_select_row` caches wider rows (~41 K keys on A100, ~57 K on
   H100 instead of ~32 K). The wrapper now passes the hardcoded `FILTERED_TOPK_SMEM_DYNAMIC`
   (131,072) to `det_select_row` instead of the launched size, so on H100/A100 that improvement
   no longer takes effect. In the narrow case where `optin − static < 131,072` (possible for a
   device whose optin is just at the 128 KiB gate) the kernel would also believe it has more
   shared memory than was allocated.

I could not test either: this device's optin is 101,376 B, so the FilteredTopK arm is
unreachable, and the harness records it as an UNEXPECTED rejection if ever selected (it never
was). Reported as a static reading of the port, with the caveat that it may be intentional.

## 9. Not done (deliberately)

No torch, no operator registration, no vLLM build, no CUDA graphs, no multi-stream or
multi-device, no performance claim, no model, no serving, no rows 8/64 sweep, no FilteredTopK
or sampled_topk execution (both unreachable at 101,376 B optin), no GitHub write.

## 10. Plain-language conclusion (scoped to this device and this harness)

The author's port onto upstream's `filtered_topk_row` split **does not change the code we
validated on 2026-09-11**. The entire `vllm::persistent` namespace and the persistent branch of
the launcher are byte-for-byte the same as at `7cfd04a3`; the port's edits are confined to the
FilteredTopK wrapper, a restored include and namespace alias, the retained upstream helper, and
a new `sampled_topk` dispatch arm that this hardware cannot select. Re-running the whole
protocol on the port head confirms that empirically: on this GB10 (48 SMs, 101,376 B opt-in
shared memory, CUDA 13.0, driver 590.48.01, no other compute process on the device), the
low-shared-memory cooperative-overflow fallback is still entered for row widths from 355,585 to
474,112 elements at 1 or 4 rows — i.e. whenever the schedule wants 49 to 64 CTAs per row group,
since occupancy here is exactly one block per SM. A separately instrumented copy of the header
shows the kernel really taking the `force_single_cta` path with one CTA per row and really
running `det_select_row` with its key cache disabled, and shows the control width not taking it.
Across 72 fallback-window cells and 18 cooperative-control cells — two row counts, three k
values, random, tie-heavy and all-equal inputs, six repeats each, output buffers poisoned before
every launch — all 432 launched repeats matched an exact stable value-descending /
index-ascending reference bit for bit, with no mismatch, no non-determinism and no CUDA error.
Just above the window, at 474,116 elements, the PR raises its intended host-side error before
any launch in all 18 cases. Within what was tested, the ported branch behaves exactly as the
pre-port branch did. This says nothing about performance, about widths, strides or row counts
outside the grid above, about torch/vLLM operator registration or CUDA graphs, or about the
FilteredTopK and sampled_topk paths, which this device cannot reach — and §8 records a
parameter-count mismatch in one of those unreachable paths that this run could not exercise.
