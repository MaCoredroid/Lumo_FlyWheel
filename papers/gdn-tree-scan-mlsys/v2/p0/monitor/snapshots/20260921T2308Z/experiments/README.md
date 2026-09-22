# v2 experiments — run index

`STATUS.md` is the execution log. Each `out-<utc>-<slug>/` directory is one container run of `e7a/e7a_device.py` inside the pinned vLLM image via `e7a/run_in_image.sh`, and contains `manifest_start.json` / `manifest.json` (source + cubin hashes, flags, status), `result_*.json` (raw per-operand-set results), `telemetry/` (1 Hz GPU utilization, 1 Hz compute-process inventory, container host-PID ownership record, before/after snapshots), and `SUMMARY.md` + `summary.json` produced by `e7a/e7a_reduce.py`.

| directory | purpose | status | promotable? |
| --- | --- | --- | --- |
| `out-20260921T221303Z-e7a-smoke/` | harness smoke 1 | crashed (factor layout slip) | no — retained as record |
| `out-20260921T221529Z-e7a-smoke/` | harness smoke 2 | crashed (manifest hashing of flags) | no — retained as record |
| `out-20260921T222051Z-e7a-smoke/` | harness smoke 3, layer-62 payload, iters=50 | completed; attribution UNVERIFIED (no container PID record); B4 not exercised | no — harness smoke only |
| `out-20260921T222616Z-e7a-ladder/` | **E7a pilot ladder**: 4 historical dependent operand sets (layers 62/1/12/0) + synthetic chain 1/5/11, binary 3, caterpillar + B4 arm; iters=200 | completed; start/end source hashes match; 122 cubins; 0 invalid cells; no foreign compute PID sampled; probe drift ≤ 4.2 % (criterion ≤ 5 %) | yes, as a kernel-level pilot with its labels (see SUMMARY.md); `source_snapshot/` preserves the loaded modules |
| `out-20260921T223658Z-e7a-tinygates/` | synthetic tiny-gate regime (chain 11/5, caterpillar) on the review-03-fixed sources; includes matched bf16-output B/C timing rows | completed; hashes match; 0 invalid cells; no foreign compute PID sampled; probe drift ≤ 2.7 % | yes, as labeled synthetic regime evidence |

Labels that always apply: mechanisms B and C are local reimplementations of the published mechanism families (WY/TreeWY; Bole), not the authors' systems. Historical payloads are dependent June-2026 operand sets from one topology (10-node caterpillar, B1) whose independent-prefix count is unrecoverable; they contribute zero prefixes to the planned 8-prefix pilot and 32-prefix confirmation. Synthetic operand sets are labeled `synthetic`. Timing rows are kernel timings with a WORK label; they are not serving throughput and not apples-to-apples speedups.

Code: `e7a/e7a_core.py` (algebra, oracle, metrics), `e7a/e7a_kernels.py` (Triton B-fs / C-nm / compact commit), `e7a/e7a_device.py` (harness), `e7a/run_in_image.sh` (runner + telemetry), `e7a/e7a_reduce.py` (reducer), `e7a/e7a_test_core.py` + `e7a/e7a_selftest_core.py` (CPU tests; outputs in `e7a/tests_out/`), `e7a/legacy_wy_8a975837.py` (vendored June-8 kernels, provenance header), `e7a/historical_payload_manifest.json` (hashes of the historical operand files).

Review record: `../p0/monitor/` (Codex reviews 01–03 and the session's replies; review files are never edited).
