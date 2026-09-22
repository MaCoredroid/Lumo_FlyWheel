# Reply to review 03 (2026-09-21T22:31:54Z)

Written 2026-09-21T22:4xZ by the Claude session. Review files preserved unmodified. Both new P1 findings are accepted and fixed; the interpretation boundaries are adopted. Corrections to my earlier replies are listed explicitly.

## Corrections to earlier statements

- **Smoke B4 coverage (reply-02):** wrong. The smoke `out-20260921T222051Z-e7a-smoke` ran with `args.b4=false`; no B4 branch executed there. The B4 arm ran for the first time in the ladder (`out-20260921T222616Z-e7a-ladder/result_b4_synthetic.json`, `summary.json: b4_exercised=true`).
- **Legacy export "fix date" (STATUS, reply-02):** withdrawn. Inspected source of the vendored June-8 revision shows the export path skips the root update; the June-10 payload's exported state agrees with the oracle. *Which* binary changed the path and *when* is not established (no source-change/execution record), consistent with P0's unknown historical identities. STATUS.md and the reducer footnote now say exactly that.
- **Smoke attribution label (reducer v1 SUMMARY):** invalid and regenerated. With ownership-based attribution the smoke reads **ATTRIBUTION UNVERIFIED (container host PID not recorded; PIDs seen: ['2746563'])**. Its timing rows are not promoted.

## Finding 1 — non-finite statistics → FIXED

`compare()` now indexes both tensors by the finite-pair mask (no zero-filling), reports `n_finite` and `n`, returns NaN/None statistics when no finite pair remains, and the reducer renders **INVALID** for any row with `nonfinite_candidate > 0 OR nonfinite_reference > 0` (no error value shown; depth tables print `n/a (nonfinite)`; validity count printed in every SUMMARY).

Evidence (`experiments/e7a/tests_out/review03_regression_<utc>.json`, host torch 2.4.1, ALL PASS, 25 checks):

| case | before (reviewer repro) | after |
| --- | --- | --- |
| `compare([NaN],[1])` | max_abs 0, rms 0, exact 1.0 | `n_finite=0`, max_abs **NaN**, rms **NaN**, exact_frac **NaN** |
| `compare([2, inf],[1, 1])` | rms √½, exact 0.5 | `n_finite=1`, max_abs 1.0, **rms 1.0**, ref_max 1.0, **exact 0.0** |

Impact on existing data: the ladder's 585 stored comparison dicts contain **0** non-finite entries (scan over all `result_*.json`), so no ladder metric changes under the fix; the smoke likewise had 0. The tiny-gate synthetic pass (in flight) is the first run where non-finite values are expected, and it runs entirely on the fixed sources.

## Finding 2 — process ownership attribution → FIXED

- Runner records the container's **host PID** (`docker inspect .State.Pid`, = container PID 1 = the python3 entrypoint, the PID nvidia-smi reports) plus `ps --ppid` descendants into `telemetry/container_host_pid.txt`, re-sampled every 5 s (`container_host_pid_log.txt`). For the ladder, which was already running when the review arrived, the same record was taken manually at 22:33:37Z while the container was live (`pid=2748559`; nvidia-smi listed exactly that PID as the only compute app).
- `e7a_reduce.py::attribution()` classifies by **PID identity**, not name: PIDs ∉ recorded set ⇒ `SHARED-DEVICE (foreign compute PIDs …)`; missing record ⇒ `ATTRIBUTION UNVERIFIED`; no sampled process at all ⇒ `UNRESOLVED`; otherwise `NO FOREIGN COMPUTE PROCESS SAMPLED (… sampled observation, not a guarantee)`. Coverage gaps (> 2.5 s between 1 Hz samples) are listed and appended to the label. The **probe-drift criterion (≤ 5 %) is enforced** per timing block (PASS/FAIL printed; rows flagged ⚠ on FAIL).
- Fixture results (reviewer's shape, `attrib_fixture/`): two python3 PIDs with 1111 recorded as ours ⇒ **SHARED-DEVICE (foreign ['2222'])**; no record ⇒ **UNVERIFIED**; only ours + a 6 s sampling hole ⇒ `NO FOREIGN … — COVERAGE GAPS: 1`; probe drift 4 % ⇒ PASS, 10 % ⇒ FAIL.
- Ladder attribution: `NO FOREIGN COMPUTE PROCESS SAMPLED (only container PID 2748559 seen; 575/591 inventory samples; 0 coverage gaps; window 22:26:16–22:36:06Z)`. This remains a 1 Hz sampled observation.

## Interpretation boundaries → adopted

- Timing rows now carry a WORK column (out dtype, state exports, per-call allocations). The ladder's B/C verify rows use **fp32 output stores** (not matched to A's bf16 store) and the native helpers include per-call gathers/allocations; they are labeled as such and not presented as speedups. The harness now also times a **matched bf16-output** B/C variant (`*_verify_out16`) and **preallocated** native closures (native_sg's wrapper still allocates its outputs internally, labeled); these appear from the tiny-gate run onward.
- Loaded-source snapshot: `out-20260921T222616Z-e7a-ladder/source_snapshot/` holds byte copies of every module the ladder imported (taken while the container was running, before any patch); each matches `manifest_start.json` (README.json `match_manifest_start=true`), and the final manifest's end hashes equal the start hashes. All post-run patches are in files whose hashes differ from that snapshot and are tagged in STATUS.md.
- Failed/obsolete attempts retained: `out-20260921T221303Z-e7a-smoke/` (layout slip) and `out-20260921T221529Z-e7a-smoke/` (crashed manifest) are kept.

## Status

Ladder (`out-20260921T222616Z-e7a-ladder`, 4 historical dependent operand sets + 5 synthetic + B4) completed with `status=completed`, start/end hashes matching, 122 cubins hashed, 0 invalid cells; SUMMARY.md regenerated with the ownership-based reducer. It is a kernel-level E7a pilot on dependent historical operand sets and labeled synthetic inputs — not the fresh 8-prefix pilot, not E2/E7b, not E1. Tiny-gate synthetic pass running on the fixed sources.
