# Bounded native-smoke launch wrapper review

Verdict: the scientific scope, dependency binding and unchanged seven-source repair closure are acceptable. One new wrapper lifecycle defect requires a small repair before treating the run as safely completed. The parent owns launch authorization; this review neither launches nor approves the gate.

## Exact evidence

Reviewed `FREEZE-Q1-FULLMODEL-DESIGN-v2.json`, frozen 2026-09-28T02:12:04Z, raw file SHA-256 `08ecf29a8a0a810f98016dc235c183e5fc2b9a055fdce2f33434834a069d9af2`; its declared canonical identity is `41cf48ee5759c1776b76c27c5e98db236750f97c6ecdc994a2572d761c7dc3e3` (independently verified by omitting `frozen_utc` and `freeze_canonical_sha256`, sorting keys and using compact JSON separators). Launcher SHA-256 `ac1740bd08d0364d7fb56c64a9a336f68add096f5209a7333543307a8245f352`. Static test SHA-256 `8d250e99c76d4cb1232af66d1b8f71fe957b691f71375cfb270706910a1db802`. Copies are preserved under `native-smoke-launcher-snapshot-20260928T021204Z/` next to this note.

All 25 campaign-file entries matched both size and SHA locally. All seven repaired sources remain byte-identical to the independently reviewed closure addendum (`8d40cf1d94c3f1d63c386a06b0d51d3486767c58415d1f283d69e71fac0b8c87`); their implementation was not reopened. The parent is independently checking remote/repository/binary identity.

I independently executed the four static launcher tests using the Python standard library: all passed, including shell syntax, no-action dry run and rejection of an unset run ID. I did not rerun the previously reported 58 package tests. A separate CPU-only shell mock exposes the cleanup defect below; neither Docker nor NVIDIA executables were called by this probe.

## Accepted bounded scope and wiring

The fixed launcher scope is aligned/nonpacked native process A, one calibration case `calibration-short_available__c0__root-only`, R=2, expected requests=2. The frozen prefix is 13,487 tokens; the forced chain is `[11352, 25559]` at `[13487, 13488]`. The selected fixture is explicitly not held out. The immutable image, patched FA2 fork digest/size, source dependencies, prefix bytes and reviewed scope are checked before launch. The generated job and exact rendered command are bound in `LAUNCH-BINDING.json`. Boot/source/per-layer dispatch and complete O0/O1/O2 evidence remain the already reviewed seven-source responsibility.

The wrapper calls the reviewed job builder with `--smoke`, uses the reviewed nonpacked config, then drives only the reviewed two-request native driver. It has no candidate, held-out, timing or workload expansion branch and no retry loop. Health polling is bounded by the default 2,400-second limit; request/seal timeouts remain in the unchanged reviewed driver. Output directories cannot be reused. The run archives gate/design/config/job, boot and patch receipts, driver verdict, raw object store, engine logs, launch binding and an indexed receipt.

## Required wrapper repair: cleanup cannot report success while the engine still runs

`run_q1_native_smoke_v2.sh:112-119` suppresses both `docker stop` and `docker rm` failures, unconditionally emits `engine_stopped_utc.txt`, then returns the status of the final `nvidia-smi` command. Lines 184-186 ignore cleanup outcome and exit only with the driver status. Thus a successful driver followed by failed cleanup can exit 0 and record a stopped timestamp while its owned container remains running. This directly violates the launcher's stated stop/failure contract; it is independent of scientific numerical criteria.

Reproduction: extract the unchanged `stop_engine` shell function; define shell `docker` mocks in which `ps -a` lists the owned container, `stop` and `rm` return 1, logs/inspect return 0, and shell `nvidia-smi` reports a still-running process with exit 0. Invoke in a temporary local output directory. Actual output:

```text
stop_engine_rc=0
emitted_stopped_timestamp_despite_failed_stop_and_rm
```

Minimal closure: after the bounded stop attempt, verify the owned container is terminal or absent; record stop/remove return codes and confirmed state; write the stopped timestamp only after confirmation; preserve a nonzero cleanup outcome in final/abort receipts and process status. Do not expand to other containers or add retries/experiments. Arrange receipt finalization after the final cleanup so the EXIT trap does not rewrite already hashed cleanup artifacts. CPU shell mocks should cover successful stop, failed stop with still-running state, and driver failure combined with cleanup failure. Once those wrapper-only bytes and tests are refrozen, equality to the unchanged seven sources suffices; no repeat semantic review is requested.

## Limits

This is a launch-wrapper review, not native smoke evidence or full-model/candidate qualification. No GPU/model/Docker execution, remote modification, implementation edits, or gate changes were performed.
