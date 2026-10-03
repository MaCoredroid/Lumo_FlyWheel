# Global clean-cache recovery: actual result review

**Disposition: accepted as bounded operational recovery evidence. No discrepancy found.** This records the recovery and its capacity observation; it does not qualify a model, establish performance, increment scientific/workload counters, or authorize another recovery.

Run: `q1-candidate-stage1-20260929T040755Z`.
Receipt: `experiments/review-response-20260927/runs/memory-recovery/memory-recovery-v2-q1-candidate-stage1-20260929T040755Z-20260929T040809Z/RECEIPT.json`.
Receipt SHA-256: `d1c6072baeb38ce252ab0e58909418e239e84c6c48d6288fb6a6eaf547e4419c`.

Local, read-only reconstruction independently checked all **89 declared raw files** for exact SHA/size and all **39 command records** for retained stdout/stderr SHA/size. The raw evidence supports:

- Exactly one `sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'`, rc 0, no timeout or retry, with the durable consumed marker preceding the write. The write ran 04:08:09.714–04:08:17.810 UTC on September 29, 2026.
- Raw `drop_slab` 4290 → 4291 and `drop_pagecache` 4288 → 4289, unchanged at the second checkpoint 10 seconds after the first. Prewrite Dirty was 1,944 KiB and Writeback 0 KiB, within the frozen limits.
- Empty container/engine/compute inventories before and immediately before the action. The same five desktop-client PID/start-time/command identities appear in the retained before/prewrite/after raw identity records.
- Exactly one fixed-image, network-disabled CUDA query. Its create arguments exactly match the accepted runner's fixed query, with observed CID/image/nonce ownership before start and before removal. Only CID `9551cff6ff28003f60399a95459441688fcb0e525217db5286665c1ae9708c25` was removed, rc 0; the subsequent exact-CID inspect returned the retained `No such container` response.
- Raw query output: **114,153,811,968 bytes free (106.31 GiB)**, exceeding **88,326,002,442 bytes** by **25,827,809,526 bytes**. Device `NVIDIA GB10`, total 126,175,309,824 bytes, matched the fixed query contract. Host MemFree separately rose from 47.273 GiB to 110.170 GiB immediately and 110.081 GiB at the second checkpoint; host and CUDA free-memory figures are distinct observations.

The original local-mirrored authorization bytes exactly match `authorization.copy.json`; required extracted fields match that copy, including the fresh user approval text. The approved proposal hash, current gate, operational-config canonical hash, accepted runner source and durable marker bind consistently:

| Item | SHA-256 |
|---|---|
| Accepted runner v2.8 | `b85c13a77c9a6060d2f9b6bd221997851159f716efff8aa0c53751edfed22616` |
| Fresh authorization | `38e2a780589610e4edd768c375ce1ae28edf7e186eeed8f0d5eb8cf39659026e` |
| Scientific gate bound by authorization | `9866e64dd49d54136c0ffea259f8719bf8340e3f9ae5b4376fc15dcf075854c1` |
| Canonical operational configuration | `330b1d081e9b78af315a4da12276c9069419daf3a0c78621a905c90e5a6b2b16` |
| Consumed marker | `2fc47a4e78e74ea17fc6358ba1c2dc8d7f617fa85d29b6c85f4430a9d9c8aa5d` |
| Approved proposal v1 | `0a08c95f89f64459cc739f2eb9b351868f7cc43c4c69a3fb2537070273f87655` |

Terminal outcome is SUCCESS with all required boot-hold conditions true, released operation lock, no listed operational hold, and no unexpected/seal error. The separately retained `PARENT-LIVE-CONSUMER.json` (SHA-256 `f1565331925275fa24ff79f6ff68d840e5dbf469d1b1711de8dd2eb5baf91185`) records an admitting live check for this exact remote receipt and authority directory, with no lock, holds or unfinished reservation. That is the parent-observed prelaunch namespace state, not a fresh independent assertion that the host is idle after the candidate launch. The terminal receipt's `cleared_after_seal: null` is understood through its documented external no-hold consumer rule; the parent live check supplies that separate evidence.

This reviewer performed no SSH, cache write, CUDA query, Docker command, model operation or evidence mutation. The one-use authority remains consumed. The memory-capacity block was cleared at the recorded observation; later model success remains a separate result.
