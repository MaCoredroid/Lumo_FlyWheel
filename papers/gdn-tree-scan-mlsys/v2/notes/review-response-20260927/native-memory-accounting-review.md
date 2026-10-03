# Native smoke memory accounting: independent local review

**Disposition:** the retry failed its startup memory admission check; the retained observations support a large accounting gap, but do not identify its owner or prove a driver leak. No live owned engine remains in the captured lifecycle evidence. No further process cleanup is justified by these receipts. A prospectively frozen smaller-memory **instrumentation smoke** is defensible, subject to capacity/headroom checks; its fit is not established here and it cannot replace the full calibration design.

This review read local artifacts and source/configuration only. It launched no CUDA, GPU, Docker, SSH, model or remote process; changed no gate; and preserved unrelated processes and models. NVIDIA documentation research is being handled separately by the parent.

## Evidence and what it proves

The three diagnosis files under `experiments/review-response-20260927/runs/q1-native-smoke/diagnosis-q1-native-smoke-retry-20260928T025847Z/` hash as follows:

| File | SHA-256 |
| --- | --- |
| `startup-memory-diagnosis-20260928T030056Z.txt` | `01855c7e18db41ca28a8d88fc7cb5625b085a31617e2a8a0f069f1aed981412f` |
| `memory-holder-diagnosis-20260928T030134Z.txt` | `fc0554d20098af7d13ecdf5c3514f4e6d5d48caeaab7fabce20293e8296acbd2` |
| `driver-kernel-lifecycle-diagnosis-20260928T030329Z.txt` | `062c68c40f73fbb31a768aa752510d948d6c5c1ffd2843e543973d566e753832` |

The retained refusal is explicit: device free **40.26/117.51 GiB** was below utilization **0.6 × total = 70.51 GiB**. This is startup admission refusal, not evidence that a particular model forward exceeded memory. The later bounded query reports **37.5 GiB** free, with no model loaded; these are different timestamps/contexts, not inconsistent simultaneous observations.

The first container's saved state has exit 0, `OOMKilled: false`, `Running: false`, PID 0 and finish time 02:43:28Z. Docker events record stop/die/destroy; the retry records exit 1 and destroy. The later host process check reports no owned leftovers. NVIDIA process evidence shows only Xorg and GNOME graphics entries (237 and 186 MiB), and the visible device-handle list contains desktop clients. This supports successful owned-process cleanup. It does **not** establish complete device allocation reclamation or the absence of every driver allocation. The earlier ambiguous `(count: 4)` orphan-search line alone was not used as evidence of four engines.

At 03:01:34Z, `/proc/meminfo` reports:

| Counter | GiB |
| --- | ---: |
| MemTotal | 117.5099 |
| MemFree | 38.0144 |
| MemAvailable | 42.6747 |
| AnonPages | 3.2089 |
| Cached | 5.3895 |
| Buffers | 0.0728 |
| Slab | 0.7827 |
| SwapCached | 0.1288 |

Subtracting free, anonymous, cache, buffers and slab from total leaves about **70.04 GiB** outside those main counters. This subtraction is an approximate diagnostic, not an exhaustive disjoint allocation census: other counters overlap or classify different aspects of memory. Small page-table/stack/per-CPU/locked/hugepage/CMA values do not visibly account for that scale. Total process RSS is about 4.11 GiB; root cgroup `memory.stat` also shows only modest anonymous/file/kernel totals. Neither RSS summation nor this partial cgroup view proves ownership of the remaining pages. `Committed_AS` is not resident memory.

The first engine earlier advertised **44.97 GiB available KV cache** and was observed with **66,788 MiB** in the GPU process list. Their scale is consistent with a possible residual engine allocation, but temporal association is insufficient to establish causation. Driver-retained allocations, another accounting category, or a platform accounting issue remain hypotheses.

## Driver evidence and limits

Captured NVIDIA version/firmware is **590.48.01**, GB10, with persistence enabled. FB and BAR1 total/reserved/used/free counters are all `N/A`; this is missing accounting, not zero use. `nvidia_uvm` shows no module users, while graphics-related NVIDIA modules remain used. That does not establish the size or absence of retained driver pages. Visible handle enumeration is explicitly limited to the collecting user's visibility.

The filtered kernel-log excerpt since 02:40Z shows a denied ptrace-style access, and no displayed NVIDIA fault/OOM event. It is not an unfiltered, whole-boot proof that no fault occurred. No allocation-level driver heap/page ownership counter is captured. The supplied kernel version `6.14.0-1015-nvidia` is contextual information from the parent, not independently recovered from these three files.

## Recovery within the existing scope

Owned stop/removal has already completed. There is no evidenced target for another kill, and another new-process `empty_cache` cannot release a dead process's private allocator. Preserve the desktop, agents, models and unrelated containers. Removing model files or pruning disk artifacts does not address this observed RAM deficit.

Do not present cache clearing as a solution: even the entire visible ~5.4 GiB file cache is much smaller than the >30 GiB startup deficit. `MemAvailable` already reflects reclaimability estimates. About 8.4 GiB is swapped; swapoff would require bringing pages back into RAM rather than creating free RAM. The steady diagnostic vmstat samples show little current paging; the first vmstat row is a historical aggregate and must not be described as the current swap rate.

A later read-only stability sample is reasonable if needed to distinguish delayed reclamation, but these receipts do not support a guaranteed safe, scoped driver-memory release operation. GPU reset, unloading NVIDIA modules, restarting graphics, or rebooting are host maintenance actions with effects beyond the owned smoke; none is justified as automatic scoped cleanup here. Any such action needs a separate maintenance decision, followed by new baseline receipts.

## Defensible prospective smaller-memory smoke

The failed first run used `--gpu-memory-utilization 0.6 --max-model-len 131072` (`tools/q1_spec_off_engine_config_v2.py:45`). Its actual smoke case is the unchanged **13,487-token** `calibration-short_available` prefix followed by two forced inputs at positions 13,487 and 13,488. It does not need a 131,072-token admission capacity. A new **16,384 maximum-length, B1, root-only, R=2, one-process** smoke is therefore a defensible resource restriction without shortening/replacing the selected tokens.

Preserve the pinned model/image/patched FA2, aligned nonpacked recurrent operator, seed, BF16 attention KV, FP32 recurrent state, 1024 block sizes, prefix-cache policy, and existing chunk/prefill thresholds. Do not add eager execution, another attention backend, lower precision, truncation or another model as an unreviewed workaround. Keep the O0/O1/O2 archive and repeat requirements intact. Smaller cache allocation may change physical storage and must be recorded; this smoke does not qualify the full 84-case/long-context calibration.

For budgeting only, utilization **0.28** would request approximately **32.90 GiB**, leaving about **4.60 GiB** below the later 37.5 GiB free-device observation. Subtracting the earlier `70.51 − 44.97 = 25.54 GiB` non-KV estimate leaves roughly **7.36 GiB** for cache. These are planning estimates, not a proven fit: the new boot's model/activation/graph/workspace costs and hybrid-cache padding must be measured. Logical BF16 attention KV for 13,489 positions alone is about 0.823 GiB from the recorded 16 attention layers × K/V × four KV heads × 256 dimensions; that explicitly excludes the three Mamba groups, allocator/layout padding and workspace, so it is not a complete capacity bound.

Freeze the changed config as a new resource-bounded smoke identity before launch. Confirm actual resolved per-group block/page capacity covers the full prefix plus forced tokens, retain host/device headroom observations and stop on admission/profile/capacity failure rather than repeatedly reducing parameters. The current local bundle does not include the pinned `gpu_worker.py` implementation; this note does not assume that an unverified `kv_cache_memory_bytes` option bypasses the refusal. A smaller declared utilization plus appropriate maximum length is a proposal for source/runtime validation, not a command or gate change.

Finally, retain both existing failures separately: the first run reached the selected prefix but failed instrumentation (`prepared input_ids/positions unavailable`, missing O0/O1/O2; one request, zero valid), while the retry never became healthy. Neither is a native numerical qualification result. The resource adjustment cannot repair or erase the instrumentation failure.

## Addendum — pinned driver pool source and one reclaim operation

**Updated operational assessment:** independently reviewed primary source supports **one recorded `drop_caches=2` operation** as a reasonable diagnostic recovery in the present idle window. It does not require killing processes, unloading/resetting the driver, swapping pages back in, or deleting files. No remote operation was performed by this reviewer, and no scientific/experiment gate is changed.

**Correction to the earlier cache-size objection:** Linux 6.14's `fs/drop_caches.c:57–65` separates the page-cache branch (bit 1) from `drop_slab()` (bit 2). The latter invokes registered shrinkers and is not limited by the visible `Slab` or `Cached` totals. Thus those small totals do **not** bound the memory potentially returned by the NVIDIA pool. Value 2 avoids the explicit bulk page-cache branch; value 3 unnecessarily adds it. [Pinned Linux handler](https://raw.githubusercontent.com/torvalds/linux/v6.14/fs/drop_caches.c).

`drop_slab` walks online nodes and memory cgroups; global shrinkers are reached for the root cgroup. NVIDIA's registered pool shrinker is NUMA-aware, not memcg-aware. A child-cgroup-only reclaim request is therefore not an established replacement for this operation. The kernel controls iteration; one sysctl write is not a fixed-duration or fixed-byte operation. [Linux 6.14 reclaim traversal](https://raw.githubusercontent.com/torvalds/linux/v6.14/mm/vmscan.c), [shrinker dispatch](https://raw.githubusercontent.com/torvalds/linux/v6.14/mm/shrinker.c).

**Pool semantics:** the pinned callback removes returned entries from dirty/clean pool lists under a mutex, then frees them. Reallocation removes entries before returning them to a client. The count includes an entry temporarily held by the scrubber, so a scan need not free the entire reported count. Counts are order-sized allocations: bytes require multiplication by `PAGE_SIZE << order`. These paths support reclaiming unused pooled allocations, not live client buffers. [NVIDIA 590.48.01 `nv-vm.c`, pool implementation](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/590.48.01/kernel-open/nvidia/nv-vm.c).

**529 means a mask, not a memory quantity:** `529 = 0x211`, the driver's default selection of 4 KiB, 64 KiB and 2 MiB pools. These are shared across adapters. Orders 0/4/9 apply only with a 4 KiB base page; do not infer those orders without the host page size. [Pinned option definition, lines 896–912](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/590.48.01/kernel-open/nvidia/nv-reg.h).

**Narrower interface:** Linux documents per-shrinker `count`/`scan` files, but the parent now reports mounted debugfs with `CONFIG_SHRINKER_DEBUG` unset, explaining their absence. Mounting it again cannot supply that disabled feature. No other dedicated pool-drain endpoint was found in the inspected NVIDIA pool/procfs code. The module-parameter macro has sysfs permissions 0, and pool destruction is wired to module teardown, not an evidenced live parameter toggle. Do not improvise registry or module changes. [Shrinker interface](https://docs.kernel.org/admin-guide/mm/shrinker_debugfs.html), [NVIDIA procfs](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/590.48.01/kernel-open/nvidia/nv-procfs.c), [parameter macro](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/590.48.01/kernel-open/common/inc/nv-linux.h), [module lifecycle](https://raw.githubusercontent.com/NVIDIA/open-gpu-kernel-modules/590.48.01/kernel-open/nvidia/nv.c).

The proposed operation is **system-wide reclaimable-cache eviction**, not cleanup limited to the smoke's PID. Kernel documentation describes it as non-destructive to dirty objects but warns of subsequent CPU/I/O costs rebuilding caches. NVIDIA's list named `dirty` means unsanitized returned pool memory, not dirty filesystem data. No additional safety blocker was identified in this source review; that is not a guarantee against implementation bugs. [Linux operational guidance](https://docs.kernel.org/admin-guide/sysctl/vm.html#drop-caches).

The matching GB10/kernel/driver forum report increases plausibility but is a user's reproduction report, not independent proof of this host's exact allocation owner. Its observed value-3 workaround is consistent with the bit-2 mechanism established above. [NVIDIA-hosted report](https://forums.developer.nvidia.com/t/driver-590-48-01-regression-uma-memory-not-released-after-cuda-process-exit-works-on-580-126-09/359969).

Recommendation: record the single operation's start/end/return status and before/after host/device availability, process/container identities, cache/swap counters and relevant kernel messages. Add no `sync`, swapoff, reset, unload, repeated memory-pressure loop or automatic second write. Preserve all existing failure receipts. Later numerical calibration/timing must begin after reclamation finishes, with normal attestation and the same declared warmup policy across compared arms. Do not treat recovery duration or a cold-cache boot as method timing; do not claim a driver leak proved or a precise pool size solely from the global free-memory delta. If availability recovers, reconsider the smaller-memory workaround prospectively rather than changing the already frozen smoke silently.
