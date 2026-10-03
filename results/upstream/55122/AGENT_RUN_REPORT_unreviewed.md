# P4' run report — vllm-project/vllm PR #55122, low-shared-memory `force_single_cta` fallback on GB10

Date: 2026-09-11. Working dir: `/home/mark/shared/p4prime-55122/`.
Wall time: start 23:19:00Z, GPU work complete 23:37Z (~18 min of the 2 h budget).
No GitHub writes. No model, no serving, no performance claim. Nothing posted.

## RESULT: POSITIVE (covered)

The `max_smem_per_block < 128*1024` cooperative-overflow fallback **is reachable on GB10**,
the `force_single_cta` branch and the **uncached** `det_select_row` rescan **both execute on
device**, and in every case tested the result was exact against a stable
value-descending/index-ascending reference and bit-identical across six repeats.
432/432 launched repeats PASS; 18/18 over-width cases produced the expected *pre-launch host
rejection* (not a CUDA fault).

---

## 1. Source preparation

```
mkdir -p /home/mark/shared/p4prime-55122 && cd /home/mark/shared/p4prime-55122
git init -q vllm-pr55122
cd vllm-pr55122
git fetch --depth=1 https://github.com/vllm-project/vllm.git pull/55122/head
git checkout -q FETCH_HEAD
```

`git rev-parse FETCH_HEAD` = `7cfd04a395fab32d6ee3364327df297a393f35d2`
(subject: "[Kernel] Raise RADIX_THRESHOLD to 22016, the caching bound", Jürgen Schmied, 2026-09-08)
— matches the pinned PR head exactly.

SHA-256 of the PR-head kernel sources as fetched:

| file | sha256 |
|---|---|
| `csrc/libtorch_stable/topk.cu` | `68b6bcd583c231b3eb43c52a553e86be30d86de03a13eec5697b2eb1e2b6d7ff` |
| `csrc/libtorch_stable/persistent_topk.cuh` | `aa48de444e180069c79295b732c33f4e0de8ec24457f1964ac35d3de88ee9cb1` |
| `csrc/libtorch_stable/cooperative_topk.cuh` | `fbade41dc89181e661a11c0f1d7946a54bdd08e9792645529bad50e48b1a4d51` |
| `csrc/libtorch_stable/cooperative_topk.cu` | `9224b4e281ff5ab91e56701d716f2525799a5e4deb41aa4dff20499dd7daec33` |
| `csrc/libtorch_stable/topk_histogram_4096.cuh` | `b1c8901d5b580fade9cd57e8d0bd1920c8c36908a8ede174393c20d333231bd3` |

The stale standalone mirror referenced in the earlier investigation was **not** used.

## 2. The registration/launcher shim (documented)

`/home/mark/shared/p4prime-55122/harness/p4_harness.cu`
(sha256 `4b24c44754eb799955e98f7f50c8fc58b8ecded96d4e0e4ce2e8904972c4611d`, 461 lines).

* `#include "persistent_topk.cuh"` — the PR-head header **unmodified**, byte-for-byte as fetched.
  The header pulls in only `<cuda.h> <cuda_fp16.h> <cuda_runtime.h> <cub/cub.cuh> <cstdint>
  <type_traits>`; it has **no torch dependency**, so no vLLM build and no torch extension is needed.
* `launch_shim<TopK>()` is a **line-by-line transcription of the non-FilteredTopK (`else`) branch of
  `launch_persistent_topk()` in `csrc/libtorch_stable/topk.cu`**, with exactly two substitutions:
  1. `torch::stable::Tensor` accessors replaced by raw device pointers and scalars
     (`num_rows`, `stride`, `max_seq_len` passed explicitly);
  2. `STD_TORCH_CHECK(...)` replaced by a non-throwing `REJECT(...)` recorder so that the
     **expected `ctas_per_group > 64` host rejection is distinguishable from a CUDA failure**.
  Every arithmetic step (effective_max_smem tiering, vec_size, `cudaFuncGetAttributes` static-smem
  subtraction, `max_chunk_elements`, `ctas_per_group`, `chunk_size`, det_want/dyn_cap smem raise,
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, headroom reservation, `num_groups`,
  `total_ctas`, the fallback edit, the `cudaMemsetAsync` of RadixRowState, and the
  `cudaFuncSetAttribute` + launch) is reproduced in the same order with the same constants.
* The `num_rows > 32 && optin >= 128 KiB` FilteredTopK primary path and the `>= 128 KiB`
  FilteredTopK overflow fallback are **not modelled** (neither is reachable on GB10, optin =
  101,376 B); they are recorded as an explicit rejection if ever selected. They never were.
* Host reference: `std::stable_sort` by value descending (index-ascending tie order preserved),
  first k indices, then `std::sort` ascending — identical semantics to the in-tree test's
  `torch.argsort(descending=True, stable=True)[:k]` then `torch.sort(...)` (test file lines
  1330–1331). Inputs are finite float32 only.
* Output buffer is **poisoned with `-424242` before every launch**; residual poison counts as FAIL.
* Every result is retained; any mismatch fails; a CUDA error exits the process with code 3
  (no continuation in a poisoned context).

Toolchain / device:

```
nvcc: Cuda compilation tools, release 13.0, V13.0.88  (/usr/local/cuda)
build: nvcc -std=c++17 -O2 -arch=sm_121a -I <PR>/csrc/libtorch_stable -o p4_harness p4_harness.cu
driver 590.48.01
DEVICE name=NVIDIA GB10 cc=12.1 sms=48 sharedPerBlockOptin=101376 sharedPerBlock=49152 sharedPerMultiprocessor=102400
binary sha256 (p4_harness) 4e333d3df1437c428bb928b058f0b16d41ccc61849906d6943eeaa1e14c81895
```

Torch was not needed and was not used. Nothing was installed into `exp54928/.venv`.
Every GPU-touching command ran as `flock /home/mark/shared/exp54928/gpu.lock timeout <n> <cmd>`
(the lock file did not exist; it was created as an empty file — the only write outside
`/home/mark/shared/p4prime-55122/` and `/home/mark/shared/tmp-scratch/`).

## 3. Host-side diagnostics (constants read back from the built kernel)

```
CONST kThreadsPerBlock=1024 RADIX_THRESHOLD=22016 kSmemMedium=35968 kFixedSmemLarge=2080
      kDetMaxCtasPerGroup=64 det_select_row_fixed_bytes=6400 (identical for k=512/1024/2048)
MEASURED on GB10, rows<=4, stride%4==0 (vec_size=4):
      static __shared__ of persistent_topk_kernel<K,4> = 4256 B   (all three K)
      effective_max_smem = min(101376, kSmemMedium) = 35968       (num_rows <= 4)
      available_for_ordered = 35968 - 2080 - 4256 = 29632
      max_chunk_elements    = 29632 / 4 = 7408
      det_want = 6400 + 4*22016 = 94464 ; dyn_cap = 101376 - 4256 = 97120
      smem_size (launched)  = 94464
      occupancy (cudaOccupancyMaxActiveBlocksPerMultiprocessor @ 1024 thr, 94464 B) = 1
      hw_resident_cap = 48 SMs * 1 = 48 ; headroom = 1 (occupancy==1)
```

These values are identical for k ∈ {512, 1024, 2048} and rows ∈ {1, 4}.

## 4. Measured routing table (the answer to step 3 of the plan)

`ctas_per_group = ceil(n / 7408)` for `n > RADIX_THRESHOLD = 22016`.
Because `occupancy == 1`, the fallback predicate `total_ctas > hw_resident_cap` can only fire
when `num_groups` is clamped up from 0, i.e. when `ctas_per_group > max_resident_ctas`.

| `ctas_per_group` | n (stride = max_seq_len = n, multiple of 4) | selection |
|---|---|---|
| ≤ 47 | n ≤ 348,176 | cooperative multi-CTA, `force_single_cta = 0` |
| 48 | 348,180 … 355,584 | cooperative multi-CTA, `force_single_cta = 0` (`max_res` becomes 48, `48/48 = 1` group, `total_ctas = 48 == hw_cap`) |
| **49 … 64** | **355,588 … 474,112** | **LOW-SMEM FALLBACK: `force_single_cta = 1`, `ctas_per_group → 1`, `chunk_size → 7408`, `total_ctas = min(48, num_rows)`** |
| ≥ 65 | n ≥ 474,116 | **pre-launch host rejection**: `persistent_topk: ctas_per_group 65 exceeds 64` |

So the exact fallback window on this device is **n ∈ [355,585 , 474,112]** (measured to the
nearest multiple of 4: last cooperative n = 355,584; first fallback n = 355,588; last fallback
n = 474,112; first rejection n = 474,116). Identical for rows 1 and 4 and for all three k.

**The agent estimate "ctas_per_group ∈ [49,64] ⇒ n ≈ 343k–458k" was close but wrong in both
endpoints** — the true window is 355.6k–474.1k, because `max_chunk_elements` is 7,408, not the
estimated ~7.2k–8.5k (the kernel's static `__shared__` is 4,256 B, and `ctas_per_group == 48`
still fits under `hw_resident_cap`, so entry begins at 49 CTAs = 355,585 elements, not 48).

Raw logs: `route_rows1_k2048.log`, `route_boundaries.log`.

## 5. Device-side proof that the target branch and the uncached select actually execute

A **separate instrumented copy** of the header (`instr/persistent_topk.cuh`, sha256
`775cf361033ded26073cd9a436bad800f8ecbdde99d24c3e352a9303641c6c3a`) adds exactly two
`printf` probes and one `#include <cstdio>` (diff is 3 hunks, 8 added lines, no logic change):
one inside the `if (params.force_single_cta || seq_len <= RADIX_THRESHOLD)` branch of
`persistent_topk_kernel`, one immediately after `const bool cached = ...` in `det_select_row`.
Built as `p4_harness_instr` (sha256 `8c6bbc7e80219afd24bd4f2dc3ed5e2ca116343882f862b8979cb5d4a577e0da`).
**All validation numbers in §6 come from the pristine, uninstrumented build.**

```
n=400000 rows=1 k=2048:
  INSTR single_cta_branch force_single_cta=1 seq_len=400000 ctas_per_group=1 gridDim=1 det_smem_bytes=94464
  INSTR det_select_row n=400000 smem_bytes=94464 fixed=6400 CACHED=0
n=474112 rows=4 k=512:
  INSTR single_cta_branch force_single_cta=1 seq_len=474112 ctas_per_group=1 gridDim=4 det_smem_bytes=94464
  INSTR det_select_row n=474112 smem_bytes=94464 fixed=6400 CACHED=0
n=355584 rows=1 k=2048 (control, one CTA short of the window):
  (no INSTR lines at all — the single-CTA branch is never entered; the cooperative path runs)
```

This is the direct evidence asked for: the low-shared-memory fallback executes,
`det_select_row` runs with `cached == false`, i.e. the **uncached global-rescan branch** — the
only way to reach it on GB10 — and the control width does not enter it.

## 6. Result matrix (pristine build, `matrix_raw.log`, `matrix_summary.txt`)

Grid: n ∈ {355584 (control, cooperative), 355588 (first fallback), 400000 (mid fallback),
474112 (last fallback), 474116 (first rejection)} × rows ∈ {1, 4} × k ∈ {512, 1024, 2048} ×
inputs ∈ {random U(-100,100) float32, tie-heavy `randint(0,5)`, all-equal 1.5f} × **6 repeats**.
90 cases, 522 launches/attempts. Seed 20260911. Every launch preceded by output-buffer poisoning.

| n | `force_single_cta` | cases | repeats | outcome |
|---|---|---|---|---|
| 355,584 | 0 (cooperative control) | 18 | 6 | **108 / 108 PASS** |
| 355,588 | **1** | 18 | 6 | **108 / 108 PASS** |
| 400,000 | **1** | 18 | 6 | **108 / 108 PASS** |
| 474,112 | **1** | 18 | 6 | **108 / 108 PASS** |
| 474,116 | n/a | 18 | — | **18 / 18 REJECTED_PRELAUNCH**: `persistent_topk: ctas_per_group 65 exceeds 64` |

Total: **432 PASS, 0 FAIL, 0 CUDA errors, 0 device faults, 18 expected pre-launch rejections.**
"PASS" = (a) output exactly equals the stable value-descending / index-ascending reference,
(b) bit-identical to repeat 0, (c) zero poison values surviving. All three held on every repeat
of every case, including the all-equal rows (where the entire answer is the index rank) and the
tie-heavy `randint(0,5)` rows at n = 474,112, where ~95 k elements share the top value and the
k-th value is tied with ~93 k others — the exact structure the PR's determinism claim is about.

Kernel times (cudaEvent around the whole shim call incl. the first-call attribute setup):
min 0.037 ms, mean 8.6 ms, max 48 ms over 522 measurements. These are **not** performance
results — no warm-up, no repetition control, single stream, alongside another workload on a
shared GPU. Recorded only to show the runs completed and nothing hung.

## 7. Not done (deliberately, per plan)

* No comparison against a "stock"/main binary — its source and operator loading were not
  independently established, and it is unnecessary for the primary correctness claim.
* No FilteredTopK path exercised (unreachable on GB10: 101,376 B optin < 128 KiB).
* No rows = 8/64 sweep, no model, no serving, no performance claim, no GitHub write.

## 8. Plain-language conclusion

On this GB10 (48 SMs, 101,376 B opt-in shared memory, CUDA 13.0, driver 590.48.01), PR #55122's
low-shared-memory cooperative-overflow fallback is genuinely reachable, and we reached it. Built
straight from the pinned PR head with the kernel header untouched and the PR's own host launch
logic transcribed verbatim, the branch is selected for row widths from 355,585 up to 474,112
elements at 1 or 4 rows — that is, whenever the schedule wants 49 to 64 CTAs per row group, since
occupancy on this part is exactly one block per SM. Device-side probes confirm that inside that
window the kernel really takes the `force_single_cta` path with one CTA per row and really runs
`det_select_row` with its key cache disabled (the uncached global-rescan variant), which is the
code the PR substitutes for the old `top_k_per_row_decode` call and which nothing on H100/A100
can execute. Across 72 fallback-window cases and 18 control cases — two row counts, three k
values, random, tie-heavy and all-equal inputs, six repeats each, output buffers poisoned before
every launch — all 432 launched repeats matched an exact stable value-descending/index-ascending
reference bit for bit, with no mismatch, no non-determinism, and no CUDA error. Just above the
window, at 474,116 elements, the PR raises its intended hard host-side error
`persistent_topk: ctas_per_group 65 exceeds 64` before any launch, in all 18 cases — an explicit
rejection, cleanly distinguishable from a device fault. Within what was tested, the branch works
as its author intends; this says nothing about performance, about widths or row counts outside
the grid above, or about any end-to-end behaviour.
