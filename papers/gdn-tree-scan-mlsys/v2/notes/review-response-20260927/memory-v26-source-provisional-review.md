# Memory v2.6: bounded provisional source review

**Disposition: one small worker-identity repair is required; the rest of the inspected target-only path implements the six proposal corrections.** This is a source review, not an operation/launch approval. The captured source is provisional: no final v2.6 freeze or v2.6 test file was present at **17:35:57 UTC**.

Snapshot: `p0/monitor/review-response-20260927/memory-v26-source-reviewed-20260928T1735Z/`. `SNAPSHOT.json` authenticates the copied remote bytes:

- v2.6 runner: **216,293 bytes**, SHA `15b438d0180dc5746feaa8caaee50b768117073eb8fb95b1ff3df89e78628981`.
- Accepted v2.5 baseline: SHA `973e06783c222ece34df97246ab70d6f766c2222f48e3addf525472fa9d74ac8`.
- Target-only policy: SHA `05252fd2545cfd61f088be2ceedfff12bd238e8fa2f67a7fe19f523b1684530f`.

Line references below are to the frozen local copy of `tools/memory_recovery_v2_6.py`, not the changing remote file.

## F1 — missing exact worker identity can still proceed to the query

`_proc_starttime_real` at **809–813** returns `None` on any stat-read/parse error. `real_advise_worker` at **816–839** retains that value but continues. `classify_advice` at **842–869** checks reaping and an integer PID, then accepts the two-line successful result without validating start-time identity. `_targeted_advice_and_query` at **2253–2270** and `_targeted_conditions` at **2332–2349** consequently permit that worker to satisfy the termination condition.

This is a narrow evidence-validity gap, not a demonstrated wrong-process signal: reaping still proves that the owned child ended. However, policy expressly requires retaining the exact worker process identity, and the real read-error path can omit it while reporting success.

The retained CPU reproducer establishes both a valid control and this altered-evidence path. With only the second worker's `starttime_ticks` changed to `None`, the actual targeted orchestration method issues the three **stubbed** advice calls, reaches exactly one **stubbed** query and returns `SUCCESS`. Its recorded second-worker start time remains null. A standalone wrong-worker-hash probe is also accepted; the real worker currently emits the expected hash, so that second probe is a validation-hardening observation rather than an independently demonstrated producer fault.

**Smallest closure:** require positive, non-boolean PID and start-time ticks and the expected embedded-worker SHA before classifying any worker result as qualified. Missing/invalid identity must be non-admitting and preserve the already-issued attempt; do not retry its syscall. If possible, gate the child before advice when its identity cannot be obtained; otherwise the existing conservative unknown-result/hold path is sufficient to prevent admission. Add valid, null/invalid start time, and mismatched worker-hash controls, including the existing orchestration path. No reservation or numeric setting change is needed.

## Inspected closures that remain accepted

| Requirement | Source and result |
| --- | --- |
| Target-only operation, no automatic second slab write | `_flow` **1343–1359** selects the explicit target profile; **2230–2296** contains only advice/evidence/query. `cmd` **1257–1267** rejects the exact slab write on this profile. The receipt exposes `operation_profile`, `slab_writes:0` and an explicit zero-write record instead of fabricating v2.5 milestones. |
| Accurate effects and source status | **270–278, 336–355** pin the four previously reviewed upstream source hashes and retain semantic-reference versus exact Ubuntu-build limits, writeback/LRU effects, and advisory/non-cold-cache caveats. The filesystem check **1909–1920** binds the expected ext4 device. |
| Exactly three held file identities | `normalize_targeted_block` **999–1062** rejects a fourth, duplicate, alias or changed target. **1922–1953** opens once with the real read-only/no-follow seam, checks fstat against all pinned fields and the path's inode, then retains the descriptor. **2101–2129**, **2170–2184**, and **2276–2283** recheck the same descriptors and detect immediate path/identity changes. |
| Root-inclusive process visibility | **886–975** distinguishes permission/parsing/PID-reuse uncertainty from proved process exit and records mappers/writers. **2013–2077** invokes a bounded, separately privileged read-only helper and requires root, the initial PID namespace, the matching `/proc` view, matching target identities and complete coverage. Unknown coverage, mappings or writers refuse. This is a finite observation, not a lock on external processes. |
| Temporary mappings and content-hash order | **771–806** unmaps the mincore view before returning and rejects an unmap error. **1955–1984** records continuity limits and validates residency output; **2079–2099** finishes any optional hash hook before advice. Production `real_deps` supplies no full-weight hashing hook. |
| One-use attempts, partial failure and uncertainty | `_reserve` **1607–1626** records the targeted operation under a fresh consumed marker. **2131–2202** persists per-inode attempted state before dispatch. **2230–2296** stops after the first non-ok result, never reaches the query after failure, and retains hold/lock for timeout or unproved termination. Definite errors are retained as consumed failures. |
| Honest receipt/consumer path | **2298–2349** requires unchanged drop-caches counters, three completed attempts and post-residency evidence. **2831–2914** adds target-specific required conditions while retaining the established authority/hold/terminal checks. Final integration must select and hash-bind this consumer; prior consumed authorities remain unusable. |

The accepted baseline lock, owned query and cleanup mechanisms were compared by source; they were not reopened as a new review campaign. No additional scientific experiment is required for this repair.

## Independent CPU evidence and next boundary

`review_cpu_controls.py` and `CPU-CONTROLS.json` are saved next to the snapshot. **26 controls pass**, covering valid target/worker behavior; every pinned identity field; extra/duplicate/alias targets; permission and malformed/PID-reuse process scans; proved exit; mapper/writer observations; definite errors; timeouts; unreaped workers; exactly three calls and one query; and stopping after the second target with no third call/query and retained uncertainty holds. The worker-identity omission is recorded separately as an expected failing requirement. The orchestration controls enter the actual method at its post-reservation boundary with fake dependencies; they are not full-producer integration runs. Linux device-number decoding is explicitly stubbed because the reviewer runs on macOS.

After F1, freeze the exact repaired source and new tests/consumer dependencies and bind the final parent authority to **one three-inode target-only attempt, zero slab writes, at most one bounded current-image query, and the unchanged 82.26 GiB reservation**. A valid insufficient-capacity result remains non-admitting. This note neither issues that authority nor admits a subsequent model boot. No fadvise, madvise, reclaim, live process inventory, query/container, GPU/model execution, process kill, pressure allocation, remote edit or gate change was performed.
