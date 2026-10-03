# Memory recovery result: independent local audit

**Disposition: the observed resource-admission block for the unchanged 0.6 native instrumentation smoke is resolved.** The post-recovery CUDA query reports **103.5280 GiB free**, exceeding `0.6 × 117.5099 = 70.5059 GiB` by **33.0221 GiB**. No smaller-memory configuration is needed to address this recorded admission failure. This is a point-in-time resource finding, not a launch authorization, guaranteed later capacity, or numerical qualification.

Reviewed campaign artifact: `runs/memory-recovery/memory-recovery-20260928T031357Z/RECEIPT.json`, SHA-256 **`260e8a206ac432ae2fc9d7364010bb52edf8f4d534a2a22944112bb809e9cea8`**. All **39 listed local members** match their hashes and sizes. The retained recovery script matches `008249cebd29aaa2dccedea053161acaf7ff1d2d7d6ecc2b4bc3efbb04b98f9c`. Remote-copy verification remains the parent's separate check.

I independently parsed all receipt-listed meminfo and vmstat values from the raw before/after/10-second files and recomputed their deltas. The result agrees:

- One recorded `drop_caches=2` write, return code 0, empty stderr, reported duration **0.600161 s**; kernel log records the same value. `drop_slab` increments exactly once; `drop_pagecache` does not increment.
- Host MemFree rises **36.04 → 104.05 GiB**, a **68.0115 GiB** increase, and remains **104.1065 GiB** after ten seconds. MemAvailable rises **42.29 → 110.02 GiB**. Cached increases slightly; the Slab decline is only about 0.322 GiB. The freed-memory increase therefore cannot be attributed just to the visible slab decline.
- No OOM-kill, direct-reclaim, kswapd-steal or compaction-stall counter increase appears in the inspected interval. SwapTotal is unchanged and SwapFree changes only slightly. These observations support the claimed operation; they do not establish every background process's behavior.
- Before/after GPU-client PID files, full client listings and NVIDIA module listings are byte-identical. The five visible clients remain 4178, 4354, 4383, 4849 and 5132. Docker listings are empty before recovery and after the bounded query. The pre-compute match is the recovery wrapper's command text, not an engine; NVIDIA's compute-app listing contains only its header.
- The query JSON matches the receipt exactly. Image identity is the pinned `ffa30d66…`; query exit is 0 with empty stderr, and `--rm` plus the empty post-Docker listing supports completed query/context cleanup. The prior **37.5 GiB** query was at 03:03Z, so the **66.03 GiB** device-free difference is a comparison across times, distinct from the immediate host delta.

The result strongly supports successful reclamation through the registered-shrinker mechanism. It does **not** measure the exact NVIDIA pool contribution or prove which former allocation owned every reclaimed byte.

For the planned unchanged **two requests / one fresh process** smoke, retain the parent-verified source/configuration gate and recheck ordinary launch-time admission. Keep patched FA2, aligned native operator, tokens, cache configuration and observation requirements unchanged. Record recovery and normal warmup before interpreting later performance; no timing result was produced here. The earlier instrumentation failure and startup refusal remain preserved and must not be relabeled as passing native observations. Full calibration and qualification remain separate stages.

This audit used local file reads and standard-library reductions only; no GPU, remote operation, host mutation, configuration edit or gate change was performed.
