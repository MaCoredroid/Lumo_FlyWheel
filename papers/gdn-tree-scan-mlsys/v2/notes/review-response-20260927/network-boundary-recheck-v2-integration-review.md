# Network boundary recheck v2: independent integration review

2026-09-29. **Bounded source/CPU PASS after one narrow repair. No remaining material defect found in the requested delta.** This is preparation review; final freeze/packet binding and runtime admission remain parent-owned and pending. No live SSH, network, Docker, GPU, agent, evaluator, cleanup, or scientific execution occurred. No implementation or gate was edited by this reviewer.

## Final reviewed bytes

Paths in this table are repository-relative. Canonical files matched the recorded hashes again after testing.

| File | SHA-256 |
|---|---|
| `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/network_boundary_recheck_v2.py` | `9793e44c059831045dd5dc07588ae9fa2545bb17b2dbcfb9e5fd7fe7969c7572` |
| `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/sole_executor_v3_11.py` | `76334a811892e6572aadbd670dda8ee6b99fe78598ea4eebe54ea48891bdfe9a` |
| `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/workload-plan/tools/attempt-runtime/two_host_v1.py` | `9af64943fa3c1398856e4af2c0382f8500226a617e7c5221e4322e792ae860f8` |
| `papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_network_boundary_recheck_v2.py` | `d50ddc6814517656ddd3c891774d2de7bd7869f8168683b8c8dee2366bf93254` |

## Preserved initial finding and closure

The initial helper `e214467b4c638032abee45bf4ab90195a155233db3201459976e1b30b51eec4f` authenticated only matching opened/closed clock IDs and a 0–60 second duration (`verify_receipt`, original lines 123–126). It omitted the authority and approved-window checks enforced by `Runtime.clock` (`runtime_v3_3.py:78–86`). Starting with a valid fake-observation PASS and then rehashing either (a) both IDs changed to a foreign authority or (b) both timestamps changed to 2000, one second apart, left `terminal=True` and passed `verify_prior`. These were receipt-integrity negatives; the unchanged producer was not observed to generate such clocks.

The repaired helper adds only the validation block at lines 110–122: frozen clock authority, approved wall window, present host boot ID, strict nonnegative integer monotonic clock, unchanged boot and ordered monotonic interval bounded to 60 seconds. Opened clocks are checked for partial REFUSED records; both clocks for PASSED. Existing wall ordering and duration checks remain. The helper diff is preserved. This matches the relevant existing runtime-clock contract and introduces no new live operation or scientific criterion.

The **same two independent mutations now mark the result incomplete and refuse prior-chain verification**. A valid PASS and partial KeyboardInterrupt/SystemExit refusals still authenticate. The supplied added controls also reject host reboot and monotonic rollback. Initial source snapshots, initial failing results, and the initial reviewer control script remain intact; closure is stored separately in `repair1/`.

## Integration verified

- The exact helper/caller pins and authenticated source evidence are required at `sole_executor_v3_11.py:562`, before STARTED retention at lines 586–591. Independent missing-helper and changed helper/caller pin controls refuse before any injected read.
- The attempted marker is appended at line 785 before `NETBOUND.run` at line 786. The live helper is before agent-host preflight and agent creation. Refusals cannot become successful pre-agent network evidence.
- The caller's actual nested `retain_result` function was AST-extracted and executed with fake budget/host-finalization/journal boundaries. It authenticates and binds the network receipt at line 711 before RESULT retention at line 726. Missing attempted receipt yields `terminal=False`. Its `finally` path still binds the network evidence if host finalization raises, and propagates that exception; this is not a re-audit of unchanged cleanup policy.
- Independent third-command `KeyboardInterrupt` and `SystemExit` controls preserve the first two observations as REFUSED, propagate the same cancellation type, and bind the exact receipt digest. Such a closed pre-agent failure remains a consumed attempt; it is not successful workload evidence or permission to create the agent.
- Rehashed wrong producer/baseline identities refuse. Supplied controls additionally cover wrong attempt/freeze, changed retained rules, and byte changes after ledger binding. The receipt enters the authenticated packet evidence map only after validation.
- `prior_evidence_map` invokes `NETBOUND.verify_prior` at line 399 after existing terminal, closure, budget and owned-cleanup checks. Valid PASS and cancellation receipts join; missing/altered evidence refuses. REFUSED is allowed only when the sequence contains no `create_agent`.

## CPU evidence

The final supplied suite passed **35 executions, 23 distinct controls** (12 base controls are repeated through class inheritance):

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests/test_network_boundary_recheck_v2.py -v
```

The independent 12-case seam check passed all expected outcomes, including the original two negative reproductions:

```sh
python3 -B papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/network-boundary-v2-integration-independent/repair1/independent_controls.py
```

Evidence directory: `papers/gdn-tree-scan-mlsys/v2/p0/monitor/review-response-20260927/network-boundary-v2-integration-independent/`. `SNAPSHOT.json` preserves the initial four files, `independent-results.json` preserves the initial two invalid acceptances, and `repair1/` contains the final source snapshot, supplied test output, same independent control script/results and exact helper diff. `AUDIT.json` binds all review artifacts and records final canonical source equality (SHA-256 `d2a3ac572b7ec833b4340b2728e442e7d95707e536fff2db073643237c0aee24`). Only fake observations in temporary directories and reviewer-owned artifacts were written.

## Required final binding, outside source closure

The final freeze must name the reviewed helper and caller hashes in `common.agent_host.agent_network.boundary_rules.live_recheck` and include their exact source evidence in the packet/dependency ledger, together with the frozen baseline and completed clock policy. The draft remains unapproved; missing final pins correctly refuse. This review does not qualify live SSH transport or forwarding. A successful receipt proves a fresh read-only comparison of the specified rules/sysctls and daemon identities at its recorded interval, not continued network invariance throughout a workload.
