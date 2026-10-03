# Second native-smoke attempt: startup-resource failure confirmed

Run `q1-native-smoke-retry-20260928T025847Z` failed before model readiness. There was **one container/core-initialization attempt, zero model-weight load starts, zero healthy engines, zero driver requests, zero case seals, and zero raw objects**. Neither of the two planned observations ran. The instrumentation repair was installed but its request path was not exercised; this run neither validates nor falsifies that repair and supplies no numerical result.

## Bounded integrity audit

Terminal receipt SHA-256: `ce3760cb0f68acd848fdf520e5a16b389c8d58aff55015d0a6e4d166860f25fb`. All **25 indexed files** match their hashes and byte lengths, independently checked locally and remotely. Local membership has no unindexed artifacts other than the receipt itself. Remote driver/case/object counts are all zero.

`LAUNCH-BINDING.json` matches all **17 approved dependency/runtime hashes**, scope and run ID in gate `2bb8b76c195a93f0bd1de5139c3ceed3ac9346b172dbf9a25385bf040a519936`. Job/config hashes, rendered-command digest and actual argv/script match. The gate references freeze `81b94eb336080ec5fa3aa6043b136b8b1667454fc0dcc740eda16aa15568bdc5`; launcher identity is `df5e0c1982f12060a4277bcd79d9762c371ed009c8a5ed47d3cfc7062903a721`, hooks `1ac939405de01a4c22e5d73decb9994cfb3a3214fd838ed2a734b39110b2b8cf`, cleanup library `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524`. This audit checks recorded identities, not a fresh review of the repaired sources.

The actual container uses the pinned image `sha256:ffa30d66ff5c9346c6389507cc529827fc9934a6d2ee37855934f94fe1061cdc`. Scope remains one aligned/nonpacked native process, the shortest calibration root-only case, R=2; no candidate, held-out, mechanism or workload execution occurred.

## Failure and cleanup

The recorded lifecycle is 02:59:28Z start to 03:00:10Z cleanup finalization, about **42 seconds**. At 03:00:01Z vLLM `request_memory` rejected device initialization because CUDA reported **40.26 GiB free of 117.51 GiB**, below the requested utilization reservation **0.6 = 70.51 GiB**. It failed before the model-weight load stage. The receipt has `healthy_utc=null` and all driver timestamps/status null; there is no hook boot/per-layer runtime attestation or HTTP completion POST. An empty compute-process contention listing alone does not establish sufficient CUDA allocatable memory; this audit does not attribute the missing memory to any particular process or cache.

Terminal status is `ENGINE_NOT_HEALTHY_cleanup_rc=0`. Container inspection records exited, Running=false, ExitCode=1, RestartCount=0 and OOMKilled=false: this is an explicit startup capacity refusal, not a kernel numerical failure or recorded OOM kill. Cleanup state is `stopped_and_removed`, diagnostic error file empty, and `FINALIZED.txt` matches the indexed receipt. A fresh successful read-only Docker enumeration confirms the owned container is absent. Receipt hashes remain valid after exit.

Preserve this run separately from the first attempt's input-hook failure. There are now two different failure stages: the first engine became healthy and returned one HTTP request with an INVALID observation; this retry never reached model readiness or a request. A future launch requires its own parent decision after startup resources are addressed; this note changes no gate and grants no retry approval.

Only local CPU analysis and read-only remote hashing/counts/container enumeration were performed. No GPU/model launch, cleanup command, source edit, or evidence mutation occurred.
