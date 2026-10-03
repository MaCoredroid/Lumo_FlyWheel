# Memory v2.5 terminal result — independent operational review

**Disposition: verified, correctly non-admitting `INSUFFICIENT_CAPACITY`; no candidate boot or scientific result.** The one authorized recovery write completed, finalization and owned query cleanup completed, and the one-use authorization is consumed. The query returned **70.0641 GiB**, below the unchanged **82.26 GiB** reservation by **12.1959 GiB**. The parent may close the scientific gate; this receipt must not authorize a launch or another recovery.

Run: `q1-candidate-stage1-20260928T165700Z`; recovery ended **2026-09-28 16:58:19 UTC**. All evidence is preserved under `p0/monitor/review-response-20260927/memory-v25-insufficient-reviewed-20260928T165819Z/`. Its `raw/` contains **71 files, 353,525 bytes**, all matched to the captured remote hash/size index. The final receipt authenticates **70 members**; its own SHA is `78b94ca4bdb66430b7bc373d75358cc0125477b75cd11dec3c2423dc79e5286e`. The previous successful recovery remains unchanged; its receipt SHA is `260e8a206ac432ae2fc9d7364010bb52edf8f4d534a2a22944112bb809e9cea8`, and its **39 members** also match.

## Authority, execution and cleanup

Before gate closure, the reviewed producer's **read-only** `--verify-receipt` was invoked against approved gate SHA `b582795968acf7c24208879184e1a97651c37ebaf5def3eb4f63208e2b3e0c48`. It returned **rc 13, `admit:false`**, specifically because the terminal outcome/free-memory check does not admit a boot; `finalization_ok:true`. Gate bytes were identical before and after verification. `PRESERVATION.json` retains the exact command/result and copied approved gate/auth bytes.

- Runner SHA is the reviewed v2.5 `973e06783c222ece34df97246ab70d6f766c2222f48e3addf525472fa9d74ac8`. Authorization SHA `a4dcb7993d1aad75315c9a3ce15280e7acd17039398a255c766fdc264336781e` binds that runner, image, run, operational configuration and gate. The consumed marker `1c386c66472d8fa284ef22f7abe211baeccdc9f9d88f7963073e2fb183cd1530` was durably recorded before the write; an independent later read preserved its exact bytes. The authority lock was absent and `holds/` empty in that read.
- All **30 command records** match their raw stdout/stderr hashes and sizes. No command timed out. The only nonzero exits are the expected empty engine search and exact-CID absence check. There is **one** `sudo -n sh -c 'echo 2 > /proc/sys/vm/drop_caches'`, rc 0, duration **6.1789 s**; no retry. Kernel log records value 2. Independently recomputed **54 meminfo and 185 vmstat counters**, at both post-write checkpoints, match every retained value and delta. `drop_slab` increases once; `drop_pagecache` stays unchanged. OOM-kill, direct/kswapd steal and compaction-stall counters do not increase during this interval.
- Initial container, engine and NVIDIA compute inventories are empty. Kernel **6.14.0-1015-nvidia**, driver **590.48.01**, and pinned image `ffa30d66…` match the receipt. Before/after module listings and the five GPU-accessing desktop clients' PID/start-time/command records are byte-identical.
- Exactly one query process used the pinned image, CUDA 13.0 / Torch 2.11.0+cu130. Its raw JSON matches the receipt. Created, inspected, started and removed CID `498168cb11afed7b4264310b1d454729041d0297b3fa6ae615d1a398a1a98232` is consistent throughout, including image/nonce ownership and final exact-CID absence. No model is loaded by the recorded query command. The candidate wrapper did not run; candidate model boots and candidate observations remain **0**.

The local-only reproducer is `audit_recovery.py`; its output is `INDEPENDENT-AUDIT.json`. The reviewer performed only local reductions and read-only remote preservation, authority inspection, the explicitly allowed receipt verification, and file `stat` calls. No reclaim, query container, GPU/model execution, kill, settings change or gate mutation was performed by this review.

## Why this free-memory result differs from 03:13

The reclaim **worked**: immediate host MemFree rose **1.8614 → 71.4251 GiB**, a **69.5637 GiB** increase. The relevant distinction is what remained resident afterward. Matched ten-second checkpoints show:

| Host counter, GiB | Earlier recovery | Current recovery | Current minus earlier |
| --- | ---: | ---: | ---: |
| MemFree | 104.1065 | 71.3766 | −32.7299 |
| MemAvailable | 110.0760 | 109.4206 | −0.6554 |
| Buffers | 0.2980 | 7.6478 | +7.3498 |
| Cached | 6.4379 | 30.4486 | +24.0106 |
| AnonPages | 3.2886 | 3.8905 | +0.6019 |
| Slab | 0.8557 | 1.6745 | +0.8188 |

**Buffers plus Cached account for 31.3604 GiB of the 32.7299 GiB host-free difference.** Thus Linux's estimated available memory is almost unchanged, while immediately free memory is much lower. The separately timed CUDA queries are **103.5280** versus **70.0641 GiB**; their difference is not an exact simultaneous decomposition of the host counters. These records do not establish how a future CUDA allocation would trigger reclaim, nor permit substituting `MemAvailable` for the unchanged admission measurement.

Existing pinned Linux/NVIDIA source already supports the distinction: value 2 invokes registered shrinkers and does not request the explicit bit-1 bulk page-cache branch. A partial subtraction of Free/Buffers/Cached/Anon/Slab from total falls from roughly 68 GiB to roughly 2.5 GiB in the current run. This is consistent with successful unused-pool reclamation, but remains incomplete accounting, not a measurement of NVIDIA pool ownership or a proof that every reclaimed byte had the same source as before.

The worker's later read-only `mincore_residency_v2.py` and outputs are preserved in an **18-file diagnosis supplement** with hashes. Source inspection shows read-only libc mappings/mincore without reading mapped file contents. The output reports **20.31 GiB** resident under the served model directory and **1.95 GiB** under the M1 tensor-store run. Independent `stat` receipts confirm all three large weight shards share `(device,inode,size)` with their `-asshipped` counterparts and have link count 2: **count the large weight data once**. This corroborates substantial resident file data, not its uninterrupted residence since the native boot. The remaining cache and buffers are not attributed by this sample. The directory walker lacks an `onerror` handler, so the empty `/var/lib/docker` result is not proof that its complete contents were accessible or nonresident.

## Corrections to the worker's preliminary diagnosis

The preserved `Q1-CANDIDATE-CAPACITY-DIAGNOSIS-20260928T1700Z.md` should use **7.65 GiB Buffers**, not 2.52, and include Buffers in its explanation. Replace “never frees page cache” with “does not request the explicit bulk page-cache branch.” Limit the claim that device free “tracks MemFree, not MemAvailable” to these observations. Remove the unobserved eviction-history assertion and any assertion that the subtraction identifies an actual driver pool. The successful reclaim and high remaining resident cache support the factual diagnosis; they do not establish a permanent ~70 GiB ceiling or a guaranteed recovery remedy.

No further operational remedy is approved here. The consumed authorization remains consumed, the reservation/settings remain unchanged, and any prospective cache treatment requires its own explicit reviewed policy and authorization. No repeated slab reclaim, smaller reservation or scientific-gate advance follows from this result.
