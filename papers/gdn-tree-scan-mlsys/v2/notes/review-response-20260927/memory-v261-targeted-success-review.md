# Targeted model-cache readiness: independent result audit

**PASS for the recorded one-use operational result.** Run `q1-candidate-stage1-20260928T181822Z` executed three target-only advisory calls, zero slab writes and one capacity query, then sealed `SUCCESS`, exit 0. It ran from 18:20:32.568696 to 18:20:36.184528 UTC on 2026-09-28. This audit advances no experiment or numerical-qualification counters and does not infer successful candidate model startup.

## Evidence and authority

The read-only remote snapshot contains all **78 run files, 432922 bytes**, including all **77 receipt-indexed members**. Every member's hash and size matched the raw remote observation, and each indexed member matched the terminal receipt. All 31 command records were cross-checked against their raw stdout/stderr files. The independent local reduction is `audit_operational_result.py`; its result is `AUDIT.json` beside the preserved evidence.

| Binding | SHA256 |
| --- | --- |
| Terminal receipt | `a5e6b8add8eef73dc54b6a3bb8201a1673c10e5a52b045d3aab3464fc93a7f39` |
| Frozen v2.6.1 runner | `548c6c26c0f8d4791e8f398ef90220cadcf2174d4a5c2b40b0e8b670efc7b050` |
| Original approved scientific gate | `812f2c032c657820e3f5321030d9f47c87d95c227026f2c2a334cb1adf2874cf` |
| Approved authority / copied authorization | `1fe188cb78ffe63a0b72d3eb8f95b7784e9908df9feba579dd432da29fc9c793` |
| Canonical effective configuration | `1d00662ce2c3e8591412e7fadd6243b759cfb768b5f2aca2044bc620ad0c2e5a` |
| Consumed marker | `589a8b3a801cecc912833f92d685408d95d7a18e63a272bbf1a58a8043ed09b6` |

The consumed marker was durably recorded at 18:20:33.631721 UTC before the first advisory at 18:20:33.635370, binds the exact run/config/source/authority, and remained present in the independent observation. This authority must never be reused. The preceding configuration refusal and its revoked authority remain preserved separately.

## Advisory and state continuity

All three canonical targets remained the pinned device 2050 inodes, with unchanged size, mtime and ctime across open, recheck, immediately before advice and after advice. Each target opened once; all held descriptors closed. Calls executed sequentially as euid 1000 against those descriptors. Each worker's reported PID/start time matched the parent's live-child observation and the frozen worker source `bea5abcd4f25210c898f34fc76e2df9d6d71740ebba331adba8c9c11919ec7da`, before its release. Each worker exited 0, was reaped, and had neither timeout nor identity uncertainty.

| Inode | Worker PID / start ticks | Duration (s) | Pre/recheck resident bytes | Post resident bytes |
| --- | --- | ---: | ---: | ---: |
| 24511058 | 1471265 / 480710347 | 0.384762 | 9965652544 | 0 |
| 24511059 | 1471266 / 480710385 | 0.364842 | 9985757064 | 0 |
| 24511061 | 1471267 / 480710422 | 0.079128 | 1835995136 | 0 |

The two privileged read-only inventories covered 624/624 and 623/623 PIDs, with no unknown processes, writers, mappers or vanished entries. Their only target readers were the runner's three own read-only descriptors (PID 1471202, start ticks 480710234). Recheck completed about 4.47 ms before the first advisory. The enforced pre-advice global counters were Dirty 1104 KiB and Writeback 0, within the prospectively bound 65536/16384 KiB limits. Both global drop_slab and drop_pagecache counter deltas were zero. All five desktop GPU client PID/start-time/name identities and the recorded module inventory were preserved.

The observed total target residency fell from 21787404744 bytes to zero. Host MemFree increased from 65985876 to 87357100 KiB, while Cached decreased from 38009824 to 16733124 KiB. These are matched observations around the operation, not a claim that every host-memory change belongs exclusively to these inodes. No runtime full-weight content hash was performed; content continuity relies on the previously accepted model identity, exact pinned stat fields and held descriptors.

## Single capacity query and cleanup

Exactly one immutable-image query container was created and started. Its CID was `bb057e4100184ece39800d338929b9592dd516de6b251cef7c517129c76ce57a`. The creation stdout and cidfile agree. Both ownership inspections agree on that CID, image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc` and nonce `d6335bd019db2cf10e5081a6`; state progressed from created to exited. The raw query JSON reports NVIDIA GB10, torch 2.11.0+cu130, CUDA 13.0, total 126175309824 bytes and free **89038159872 bytes**. The integer range checks pass. Free memory exceeds the unchanged **88326002442-byte (82.26 GiB) requirement by 712157430 bytes, or 0.663248 GiB**.

The one removal targeted that verified CID and returned 0; the final exact-CID inspect returned the expected “No such container” result. Finalization released the campaign lock and sealed without errors. The independently captured authority state at 18:22:01.177035 UTC has no campaign lock and no holds, including no lingering finalization hold. The receipt's `cleared_after_seal: null` is resolved by that external absence check rather than treated as an asserted future cleanup.

## Scope of assurance

Worker evidence consists of the frozen runner's captured handshake, identity, classification, timing and reaping records; complete worker stdout is hash-labelled but not separately retained for an independent parser replay. The per-target mincore results and process inventories describe finite observation times. They neither guarantee permanently cold cache nor exclude a later external reader. DONTNEED remains advisory and may initiate writeback; the bounds and observed counters are not a filesystem side-effect prohibition. The pinned upstream Linux files are semantic references, not proof of the exact Ubuntu kernel source build.

The capacity query measured instantaneous CUDA-reported free memory; it did not reserve or stress-test that amount and cannot guarantee a subsequent model boot. The separately approved candidate wrapper must retain its own gate and readiness checks. No threshold or serving setting was changed; no additional query, reclaim, model launch, GPU probe or remote mutation was performed by this reviewer.
