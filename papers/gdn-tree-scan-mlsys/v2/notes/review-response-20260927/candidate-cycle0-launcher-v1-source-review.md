# Candidate cycle0 launcher/gate v1: bounded independent source review

Reviewed 2026-09-29 UTC. **Hold operational admission for R1 below.** No new blocker found in the new cycle0 gate or selected-job integration. This is a source/CPU review, not permission to launch, a native-result acceptance, or numerical qualification. No Docker, SSH, GPU, model, cache, or workload operation was performed; implementation and gates were not edited.

## Exact source binding

Paths below are relative to `v2/experiments/review-response-20260927/`.

| Artifact | SHA-256 |
|---|---|
| `tools/run_q1_candidate_cycle0_v1.sh` | `2e8d739ec3f32da9b564e7835db028868647d1c62bf67919c804e71fafc59c81` |
| `tools/q1_candidate_cycle0_gate_v1.py` | `3baea8b63d027897afdcc9fc39fbbe846be0d58b8dcc01f513171b74dc9e9bb9` |
| `tools/tests/test_q1_candidate_cycle0_gate_v1.py` | `53d37bad14677a1c9175e81af7dc602eabfc2e03d9c492d98e23c46a2360b4b0` |
| Preserved parent `tools/run_q1_candidate_stage1_v2_5.sh` | `8b475bffbae5087a62580b8bf17f6d9115349021d9a59b6f286447e5611f9de2` |
| Unchanged cleanup library `tools/q1_native_smoke_cleanup_v2_3.sh` | `44ae0909759e3338fd0c3031c447f07523b2d47072add2404e78aea15cbd0524` |

The inspected case binding, v2 builder/hooks/driver/patcher and generated diagnostic v3.3 launcher match their previously reviewed hashes; their unchanged scientific implementation was not reopened. Full hashes and focused control output are in `candidate-cycle0-launcher-v1-cpu-audit.json` beside this note.

## R1 — unresolved creation is incorrectly sealed as no owned container

The new wrapper's `seal_prelaunch` (lines 103–107) always writes `${status}_no_owned_container`. It is used after the staged launcher invocation at line 152 when the CID file is missing/invalid, ownership inspection fails, the launcher times out, or it exits unsuccessfully (lines 154–166). There is no attempted-creation distinction. A launcher can create an engine before losing its CID or before the wrapper can prove ownership. These branches correctly return nonzero and avoid unproven deletion, but the terminal evidence incorrectly states absence rather than unknown ownership/cleanup.

Independent local controls extracted the exact `seal_prelaunch` function and replaced only `write_receipt` with a recording stub. No launcher or external operational command was invoked. Results:

| Injected boundary | Return | Retained status |
|---|---:|---|
| Before creation, readiness refusal | 4 | `READINESS_REFUSED_no_owned_container` |
| After launcher, missing CID | 6 | `LAUNCHER_NO_CONTAINER_no_owned_container` |
| After launcher, unproven CID | 6 | `OWNERSHIP_UNPROVEN_rc=0_no_owned_container` |
| After launcher, timeout and no CID | 6 | `LAUNCHER_TIMEOUT_no_owned_container` |

Minimal correction: durably latch the launcher/possible-creation attempt immediately before invocation. After that latch, an absent or unproven CID must produce an explicit ownership/cleanup-unknown terminal disposition and nonzero cleanup result, retaining raw CID/inspection evidence; do not delete by an unproven identity. Pre-attempt refusal can retain its absence disposition. Add focused extracted controls for both sides of this boundary, including an interrupted launcher. This is the inherited root-wrapper gap identified in the review request, not an experimental failure.

## New integration checks that pass

- The wrapper checks the open gate and hashes every declared dependency, including the new gate and its campaign imports, before importing campaign Python (lines 46–63). It checks the real host and owned-interpreter validation bindings before claiming the output namespace. Atomic `mkdir` still enforces one use of that output namespace.
- Gate admission fixes one explicit A/B process, 84 calibration cases, two repeats, 168 requests, cycle 0, fixed request settings, and the unchanged 0.7 candidate memory setting. It requires accepted root instrumentation (two valid root observations), hashes/rebuilds the prepared job through the reviewed v2 builder, and compares the resulting canonical job exactly (gate lines 19–55). The existing builder retains the accepted full-native-corpus/reduction prerequisite and predetermined same-case aligned native A/r0 selection. No synthetic result can be inferred as a real native-result acceptance from these tests.
- The wrapper stages the approved v2 job, uses v2 hooks/patcher/driver and diagnostic launcher v3.3, retains the accepted readiness consumer and ownership proof before arming the cleanup library, and records raw collection rather than full qualification. The integration diff does not add workload attempts or alter numerical/model settings.
- Independently ran `python3 -B tools/tests/test_q1_candidate_cycle0_gate_v1.py -v`: all **3 test methods passed**, including the negative subcases. These are CPU admission controls; the builder is mocked in this suite and actual full-native acceptance remains future.
- Inspected the parent's retained target-Bash closed-gate control (`tools/test_log.candidate_cycle0_wrapper_closed_gate.json`, SHA `6a0ffe7c0bc083806dfbbce7647b57a0322464c6709dc4666f413aaa896b51b0`): parse 0, refusal 3, no external-command sentinel, and no output namespace. This is parent-produced evidence, not an independently repeated target-shell run.

After R1 is closed, this bounded integration review has no further identified admission blocker. Operational readiness, exact final-source gate binding and accepted native-corpus results still belong to the parent. No timing, continuous-cycle/MTP lifecycle, task quality, or full qualification claim is established here.
