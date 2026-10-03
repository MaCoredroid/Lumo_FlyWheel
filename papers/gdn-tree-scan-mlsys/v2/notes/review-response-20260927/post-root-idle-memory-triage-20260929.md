# Post-root idle memory: bounded operational triage

29 September 2026, live read-only sample at **08:27:28 UTC**. No CUDA query, Docker operation, cache write, cleanup, reset, reboot, kill or Claude interaction was performed. The only remote actions read `/proc`, cgroup counters, kernel configuration, process names/RSS and NVIDIA module/parameter metadata. The parent owns verification that the latest code/artifacts satisfy the user's Git-pushed backup condition.

**Recommendation: issue a fresh run identity/one-use authority for the unchanged reviewed v2.8 `global_clean_cache_readiness_v1` profile after the backup and idle/ownership checks.** That existing profile already invokes the original unused NVIDIA pool recovery mechanism. No different driver-reset operation is needed first. Its success is plausible from source and prior observations, not guaranteed from current counters. The 08:06 authority is consumed and must not be reused.

## Current evidence

The live read reports kernel `6.14.0-1015-nvidia`, NVIDIA open kernel module **590.48.01**, 4096-byte pages, `EnableSystemMemoryPools: 529`, and `nvidia_uvm` module users 0. The largest process RSS is GNOME shell at 1,700,424 KiB; the displayed agent/daemon processes are much smaller. This is not proof of complete allocation ownership or a reason to terminate them.

| `/proc/meminfo` counter | KiB | GiB |
| --- | ---: | ---: |
| MemTotal | 123218076 | 117.5099 |
| MemFree | 24974172 | 23.8172 |
| MemAvailable | 36321724 | 34.6391 |
| Buffers | 63704 | 0.0608 |
| Cached | 12112604 | 11.5515 |
| AnonPages | 2956968 | 2.8200 |
| Slab | 839940 | 0.8010 |

`Total − Free − Buffers − Cached − AnonPages − Slab` is **78.4594 GiB**. This is a partial accounting residual, **not a measured NVIDIA pool size**. Root cgroup `memory.stat` has about 3.03 GB anon, 12.47 GB file and 0.305 GB kernel accounting; visible page tables, stacks, locked pages and CMA likewise do not explain that scale. Dirty is 2,984 KiB and Writeback 0 at this observation. Those values still require the reviewed runner's fresh under-lock check before any action. Parent-reported empty compute/container ownership remains the separate lifecycle evidence; this triage did not create a CUDA context to recheck it.

`# CONFIG_SHRINKER_DEBUG is not set` is confirmed in the live boot configuration. The unprivileged `/sys/kernel/debug/shrinker` listing returned permission denied; the disabled kernel feature independently means this is not an available reviewed per-shrinker control path. Do not turn that listing error into a reason to mount or change debugfs.

## Original source-proven recovery mechanism

The retained source package is `p0/monitor/review-response-20260927/memory-recovery-20260928/`. Its `SOURCE-MANIFEST.json` SHA is `7783277b36844b30a626d3922980eb520963db49b48f08d72ab3d6835529309f` and supplies exact primary-source URLs.

- `nvidia-nv-vm-590.48.01.c`, SHA `51e79cb3e7a8c7478cbfa8d3f1f6e4fa07798bda2ede85bc2b8136ff8e233ec0`: lines **426–434** register NUMA-aware `nv-sysmem-alloc-node-%d-order-%u` shrinkers; **539–569** remove entries from returned dirty/clean pool lists under the pool mutex and free them through **490–503**; **755–759** bind count/scan callbacks. The allocation path removes entries for active use (**614–635**), and returned allocations enter the pool at **770–838**. This is a mechanism for unused pooled pages, not destruction of live client buffers. The source does not promise every counted entry is freed in one scan (the scrubber can temporarily hold an entry).
- `linux-v6.14-drop_caches.c`, SHA `4ec92618175611528850ae0a1839975d38326401158debc188f8486d858d4573`: **57–65** sends bit 1 to page-cache eviction and bit 2 to `drop_slab()`, which reaches registered shrinkers. Consequently **value 3 already contains the NVIDIA-pool reclaim path**. Small visible `Slab`/`Cached` counters do not bound what that shrinker can return. A value-1-only write or file-specific `POSIX_FADV_DONTNEED` is not a substitute for the original mechanism.

The historical operational evidence is directly relevant to this diagnosis. `runs/memory-recovery/memory-recovery-20260928T031357Z/RECEIPT.json` (under the campaign; SHA `260e8a206ac432ae2fc9d7364010bb52edf8f4d534a2a22944112bb809e9cea8`) records one **bit-2-only** write, host free **36.04 → 104.05 GiB**, and stable visible clients/modules. The separately reviewed full receipt supports a 68.0115 GiB host-free increase despite only about 0.322 GiB lower visible Slab. This demonstrates why the current low Slab count does not rule out useful pool reclaim. It does not assign every reclaimed byte to NVIDIA or establish a leak.

The latest preboot receipt is `experiments/review-response-20260927/runs/memory-recovery/memory-recovery-v2-q1-candidate-stage1-20260929T080536Z-20260929T080608Z/RECEIPT.json`, SHA `011296978110e0a36fa11d4e583a699c5f1b3c93cf3b007a510a85c712806a36`. It records `SUCCESS`, one value-3 write at **08:06:09.267–08:06:15.349Z**, both cache-drop counter deltas +1, and the then-observed CUDA free capacity **114,960,273,408 bytes**. Its prewrite deficit was mostly file cache (Cached 106,592,440 KiB); that historical capacity observation is not current capacity after the subsequent failed harness boot.

## Minimum next step and stop boundary

Current `tools/memory_recovery_v2_8.py` SHA is unchanged at `b85c13a77c9a6060d2f9b6bd221997851159f716efff8aa0c53751edfed22616`. Reuse its reviewed global profile rather than inventing a second remedy: fresh backup-bound run/authority, fresh idle and unchanged-client checks, one durable consumed reservation, **one** `drop_caches=3` write, immediate and ten-second postchecks, then at most the existing one owned fixed-image capacity query with exact-CID cleanup. Preserve its 88,326,002,442-byte capacity floor and all operational holds; an earlier success is not an admission receipt for the next boot.

The narrower bit-2 operation is the source mechanism for the suspected returned driver pool; the already-reviewed bit-3 profile additionally removes the current clean file cache and avoids introducing another implementation/configuration variant during this repair. It can make later file reads colder, which remains operational context rather than method timing. No `sync`, allocator-pressure loop, swapoff, driver parameter change, module unload, desktop restart or reboot is indicated before this bounded operation. If it fails or leaves insufficient capacity, preserve the result and stop for a separately justified maintenance decision; do not retry the consumed action or lower the scientific settings to hide the capacity failure.

This note supports a next operational attempt only. It neither proves the missing pages' exact owner nor advances numerical, workload or full-model qualification.
