# Independent Q1 environment v2.3 closure and host prerequisite review

2026-09-28. **Accept the bounded v2.3 private-sidecar repair. The candidate remains not launch-ready.** The next steps are a receipt-bound concurrency value, a bound host validation interpreter, and a versioned diagnostic-only swap policy. No numerical, model, timing, workload, or launch approval is implied.

## Authenticated package and controls

Snapshot: `p0/monitor/review-response-20260927/candidate-env-v23-reviewed-20260928T1915Z/repo/`, 970 files / 5,428,135 bytes. All ten new freeze payloads and all 948 raw evidence-index files match their hashes. The delta's canonical digest also recomputes exactly. Unchanged unrelated v3 dependencies were not re-reviewed.

| Artifact | SHA-256 |
| --- | --- |
| `FREEZE-Q1-CANDIDATE-STAGE1-v3.1-DELTA.json` | `2b800746a6f54fa7c37d28886e68987395353cd221bb1b45cc3eea4f29994508` |
| Renderer v2.3 | `03686d3e6be2243bba5a7faa9ec97ffd0b8a6d2713ab8f2dc512138b84cb1315` |
| Host env v2.3 | `d71cc1fa780883a52ee860257207e36abcc1761a94097c096de1ebf8948f6f52` |
| Worker connected-control source | `9d1387446a730a464508cffac690c3ff8a419c877c34875e3d41790771627e38` |
| Worker test log (31 passed) | `eeed261364d636dfd1ef266947779259bdd815e86634cf4a1b07c3ad10bfcf00` |
| Independent control audit | `d9f8b353dc0d1ec4c7a513f9e9183f46272193d43cd3efbd5c2b5431fdfd333c` |

Independent pure rendering from both sealed v2.2 and v2.3 documents reproduces v2.3 byte-for-byte. All retained assignment values are identical; the four private assignments alone are replaced by guard/unset pairs. The actual diagnostic launcher, wrapper and derived credentials remain unchanged. The renderer distinguishes the four membership-based refusals from nonempty-only private guards; it correctly keeps `FR13_FIXED32_B1_FP8_QUANT_REGCACHE_SO` outside the sidecar set.

Six independent local source-boundary controls passed: clean and inherited-empty environments remove all four names; each nonempty inherited name exits 2 before the continuation marker and without disclosing its value. `exit 2` correctly closes the prior bare-`source` return-code gap.

The worker's 14 preserved connected controls use the actual wrapper and staged launcher with recording stubs. Their control receipts and logs support: original v2.2 refusal reproduced; clean/inherited-empty v2.3 passes the private guard and stops at the subsequent concurrency guard; all four nonempty cases stop in the wrapper, seal `ABORTED_PRELAUNCH_rc=2_no_owned_container`, and never execute the launcher; bypass controls demonstrate that the unchanged launcher guard still refuses. I verified 209 saved wrapper-receipt members. The harness deliberately omits 42 duplicate staged shell copies; their receipt/control hashes and byte sizes equal the already preserved pinned source copies. All 14 control summaries have no recorded audit violation, only stubbed Docker `ps` calls, and no created container.

This is a source/raw-evidence review plus six local shell controls, not an independent full Bash 5 rerun. The worker's labelled hypothetical concurrency probe substitutes the model-config read and stops at the missing interpreter; it establishes neither full pre-container readiness nor final Docker argv correctness.

## Minimal source-grounded prerequisites

**Concurrency.** The sealed production receipt contains `SWE_CONCURRENCY=1`; the renderer's prefix/extra-key filter omitted it. Launcher lines 2570–2576 require this value alongside Hydra27 and B1, and the container receives it explicitly. Restore and assert the receipt's value in versioned env/renderer v2.4. This restores the declared B1 geometry; it does not introduce a new task, batch, or concurrency experiment. Re-run the connected clean/empty/nonempty controls through the next boundaries, because this value also participates in ambient-environment protection.

**Host validation interpreter.** `.venv/bin/python` is a real executable dependency, including topology validation at line 4074 and contract validation at line 4742; there are 16 pre-Docker call sites, many conditional. Use an owned worktree environment from the already accepted host interpreter, with no dependency installations or shared-environment modifications. Bind the executable's resolved path, version/hash, environment configuration, and actual imported validation source paths/hashes; set `PYTHONDONTWRITEBYTECODE=1`. The first topology/contract entry imports are standard-library plus local topology, so a missing worktree environment does not justify installing model/GPU libraries. Any further active validation dependency should be established by the continued CPU control, not guessed or replaced with a stubbed successful validator.

**Existing swap: a defensible explicit diagnostic exception.** The current launcher's lines 6866–6874 literally require both `MemAvailable>=80GiB` and `swap_used==0`; retaining swap therefore changes an operational contract and must be stated and frozen as an exception. It is not an unchanged readiness policy. However, the Q1 protocol and full-model plan contain no zero-swap numerical premise. The accepted native smoke launcher v2.4 has no zero-swap guard; its v2.5 design freeze explicitly distinguishes zero engine swap from host swap belonging to other processes. A blanket zero-host-swap requirement is consequently not a necessary scientific condition for this narrowly scoped untimed numerical/instrumentation comparison.

For the single root-only, R=2 diagnostic, retain pre-existing host swap with a branch restricted to the exact diagnostic identity and separately bound policy. Preserve the production zero-swap branch; preserve `MemAvailable>=80GiB`, `MemFree>=GPU_UTIL*MemTotal`, GPU_UTIL 0.7, the 82.26-GiB reservation and pinned CUDA capacity check, all contention checks, serving settings, precision and numerical criteria. No swapoff/swapon, reclaim or unrelated-process intervention follows from this recommendation.

Record before/healthy/terminal swap occupancy, `pswpin`/`pswpout`, pressure-stall totals, and owned process/cgroup swap/OOM facts. Old occupied swap alone does not demonstrate current paging or imply a changed arithmetic operator. Conversely, host counters cannot attribute activity to the engine by themselves, and unchanged counters do not establish performance equivalence. Any pressure, timeout, missing output or OOM remains visible and cannot become a numerical pass by fallback or omission. Existing complete finite/structural/decision checks still determine the bounded diagnostic outcome. No latency, GPU timing, mechanism speed, or workload claim is supported by this exception.

Parent notified this reviewer that `Q1-DIAGNOSTIC-EXISTING-SWAP-POLICY-v1.json` and the concurrency/interpreter preparation have been adopted. This note supplies the source-grounded rationale and obligations; the successor implementation, dependency map and connected pre-`docker run` proof still require exact-source review. The final control must reach a refusing Docker stub and preserve the actual derived-route/model/image/FA2/job arguments, rather than claim readiness from the currently blocked controls.

No remote mutation, GPU query, cache operation, container or model execution occurred in this review. No gate was changed. Further polling stopped pending the parent-delivered successor freeze.
