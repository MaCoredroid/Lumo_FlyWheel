# Reply to initial independent review (2026-09-21T22:01:38Z)

Written 2026-09-21T22:09Z by the Claude session (`lumo-v2-20260921`). The review file is preserved unmodified.

## Finding 1 — timing qualification risk → ACCEPTED, rule adopted

Fresh telemetry (`experiments/e7a/telemetry/gpu_contention_20260921T220600Z.txt`, observed 22:06:00Z):

- `VLLM::EngineCore` resident as PID **2736995**, 34,135 MiB (the unrelated server has restarted at least twice since 21:47: PIDs 2729147 → 2735033 (reviewer) → 2736995). Launcher: `/home/mark/shared/exp54928/.venv/bin/vllm serve Qwen/Qwen3.8-27B-FP8 … --gpu-memory-utilization 0.50`.
- GPU utilization, five 1 s samples: **0 %, 72 %, 0 %, 0 %, 56 %** → the other server is intermittently executing work. Residency plus these samples establish active contention, not just residency.

Resolution:
- All E7a timing output is labeled **"shared-device diagnostic"** in the harness manifest (`e7a/e7a_device.py::manifest`, key `timing_label`) until a window is qualified uncontended.
- `e7a/run_in_image.sh` records host-side `nvidia-smi` compute-app snapshots before/after and a **1 Hz utilization log for the entire container lifetime** (`telemetry/gpu_util_1hz.csv`); the harness brackets each timing block with a fixed 4096² fp16 matmul probe (`probe_matmul4096_fp16_us_before/after`) as an in-band contention control.
- Qualification rule for any comparative timing claim: baseline utilization ≈ 0 % across the whole window in the 1 Hz log (excluding our own kernels' spikes, identified by the harness timestamps) AND probe-before ≈ probe-after. Otherwise the table stays diagnostic.
- The unrelated server is never stopped or signaled.

## Finding 2 — inaccurate chronology → ACCEPTED, corrected

The "~22:00 / ~22:20" stamps in the previous STATUS.md were guesses and are retracted. STATUS.md now carries a measured chronology (file mtimes, `date -u`, telemetry stamps) and marks unrecoverable earlier events *observed-by*. Host clock is UTC. Key anchors: STATUS first version before 21:47:36Z (`uptime` stamp), inventory conclusion 21:54:47Z (STATUS mtime), vendored WY 21:57:41Z, manifest 21:58:03Z, kernels 22:03:15Z, telemetry 22:06:00Z.

## Finding 3 — coverage narrower than file count → ACCEPTED, accounting fixed

Tensor-level lineage check across the 9 unique-hash payloads (transcript, 22:0xZ; reproducible with the snippet in the session):

| Observation | Result |
| --- | --- |
| Joint (q,k,v,a,b,h0) identical between any two files | none |
| Layer parameters (A_log, dt_bias) identical across the six layer-0 files | yes (expected; same weights) |
| `fr10_verify_residual` vs `fr10_preprocess_commit_fix` (both layer 0) | identical q,k,v; different a,b,h0 — debugging-era captures with murky lineage |
| `fr10_preprocess_commit_fix` vs `fr10_scan_capture_replay` | identical h0 only |
| layer-1 (`fr13_wy_l1_payload`) vs layer-12 (`fr13_l12_offline_replay`) | max abs q difference 0.87 (others 5–10): possibly the same forward pass, unprovable |
| Prompt sources recoverable from side files | `prompts_swe4.json` battery (sha256 `3e2b14ba…`), "Explain hash tables." (`fr13_wy_l0_localize/request.json`), unknown for fr10-era |
| Token ids / absolute positions stored in payloads | **no** |

Consequences applied:
- Historical payloads = **≤ 9 dependent historical operand sets**, **one topology** (10-node caterpillar `[-1,0,1,1,2,2,4,4,6,6]`, B1), from **≤ 3 known prompt sources**; the number of independent (prompt, position) prefixes is **unrecoverable** and is never reported as 9.
- They count **0** toward the 8-prefix diagnostic pilot and **0** toward the 32-prefix confirmation set; both require fresh captures under the pinned configuration with saved IDs/seeds.
- Synthetic depth-1/5/11, binary-branching, tiny-gate and B4 cases are labeled `synthetic=True` in every result JSON and establish algebra/kernel behavior only, not model-native B4 or fresh-prefix confirmation.
- Results are keyed by `provenance.sha256` (historical) or `provenance.seed/regime/parents` (synthetic); layer count is never used as a sample count.

## Status of numerical results

No GPU result is claimed yet. CPU fp64/fp32 algebra on the layer-62 payload is recorded in STATUS.md ("Results so far"); device runs start with the smoke run whose artifacts land under `experiments/out-<utc>-e7a-*/`.
