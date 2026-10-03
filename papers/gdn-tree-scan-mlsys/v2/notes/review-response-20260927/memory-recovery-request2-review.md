# Memory recovery request 2: bounded operational review

**Recommendation:** preparing one new, guarded recovery for one specifically approved next model boot is reasonable under unchanged settings. The parent's prospective policy is suitable for CPU implementation review; **nothing is authorized to execute now**. The previous authorization must remain `approved:false`, `CONSUMED_SUCCESS`; neither its guard nor its receipt should be reset. Reclaim readiness cannot open a scientific gate.

## Evidence and interpretation

Reviewed request SHA `68c64ac30b461d96524878565b0d7ba7e76528bb176a26d8a9716cd0b61c16b9`; 03:42:50Z readiness capture SHA `7a7b734b10fc715a780b08111ab278629e17ffbb39c38b029834269e79777f60`. Recomputed current counters: MemFree **16.1199 GiB**, MemAvailable **44.4990 GiB**, Cached **28.5034 GiB**, AnonPages **3.5896 GiB**, Slab **1.0860 GiB**. The partial subtraction leaves **67.7727 GiB** outside the selected counters. This supports recurrent memory-retention concern; it does not prove the previous engine owns those bytes or measure current CUDA allocatable free memory. The request's causal phrase “the engine's device allocation is not returned” should remain a hypothesis. The readiness file does not independently enumerate desktop GPU clients, so its absence-of-compute summary is not a substitute for the next action's fresh inventory.

The prior operation's receipt still hashes to `260e8a206ac432ae2fc9d7364010bb52edf8f4d534a2a22944112bb809e9cea8`; the earlier result review documents one successful write, +68.0115 GiB host free memory, unchanged explicit page-cache-drop counter, and a subsequent CUDA query above the unchanged reservation. I checked that all four locally pinned Linux/NVIDIA source files still match their source manifest. Existing source review suffices: bit 2 invokes registered shrinkers, including the reviewed unused NVIDIA pool mechanism; it is system-wide reclaimable-cache eviction, not a PID-local cleanup. No repeated web research or new reclaim was performed.

## Why a renamed old script is insufficient

Old `tools/memory_recovery_once.sh` SHA `008249cebd29aaa2dccedea053161acaf7ff1d2d7d6ecc2b4bc3efbb04b98f9c` has specific fail-open gaps:

- Line 12 copies the authorization without validating approval, identity, expiry or source hash. Lines 17–20 print a source-hash mismatch without failing the process.
- Lines 24–28 can turn failed inventory commands into zero matches; their exit statuses are not mandatory evidence. The existing desktop-client check at line 51 is only a warning.
- Lines 11/42 check an old marker and create the new marker only **after** the sysctl write. A crash or concurrent runner can defeat that arrangement if it is generalized without atomic ownership.
- Lines 41–42 record write failure but continue; lines 56–59 record query failure without propagating a terminal failure. `timeout docker run --rm` does not establish that the container stopped when the client timed out. Final receipt production is not self-contained in that old runner.

These do not invalidate the recorded first successful recovery. They explain why subsequent authority needs the proposed new guarded runner, not removal of its campaign-wide consumed guard.

## Smallest closure and policy assessment

Reviewed `MEMORY-RECOVERY-PROSPECTIVE-POLICY-v2.md` SHA `0182385a43a98b091777a8ba695026e70359908d7706b29f5efb68019c062913`. Its new per-run authorization, source/image/scientific-gate binding, fail-closed inventory, exclusive pre-write attempt guard, single bit 2 write, postconditions, owned query cleanup and sealed failure receipts cover the concrete old-runner gaps. No additional experiment or source-research campaign is needed this tick.

For the next exact implementation, retain three small clarifications:

1. Acquire the shared operation lock **before the final inventory/revalidation**, keep the one-use attempt durably reserved before the write, and prevent the next owned boot from racing with unfinished recovery/query cleanup. An ambiguous attempt remains consumed/unknown rather than retryable.
2. “Bounded timeout” is a supervisory deadline, not proof that a kernel reclaim has returned. Likewise, “guarantee stop/removal” must mean **verify the exact owned query container is stopped/absent; otherwise fail and hold the next boot**. No successful result or next model launch may follow uncertain termination. This is the one wording limitation in the otherwise adequate policy, not a new operation or test scope.
3. Freeze the recovery decision rule prospectively for each compared run/cohort and record `performed` or `skipped` plus its pre-run reason. Do not decide from the method's speed, correctness outcome or a failed trial. For the immediate next boot, a new explicit one-use authorization can specify the single requested reclaim. All later boots still need their own authority; no standing loop follows from this note.

The candidate runner needs only the policy's proposed injected CPU controls: denied/malformed/expired or reused authority, source mismatch, inventory failure/active compute, race/ambiguous-write refusal, counter discrepancy, query failure/timeout and failed owned cleanup. Its exact bytes and receipts can then be reviewed; parent can issue one new operational authorization when the named scientific run is ready. No reclaim should be spent while its executor is still preparing code.

Preserve utilization 0.6 and all frozen model/cache/precision/observation settings. Admission uses a valid post-recovery query's exact free/total bytes against the unchanged reservation, with measured headroom recorded; host MemFree/MemAvailable and historical 103.528 GiB device free cannot substitute for that result. Failure stops the stage without lowering configuration or repeating recovery. Keep recovery/query/boot/warmup outside measured method intervals; record page-cache and warmup conditions because bit 2 does not promise identical cache contents. Never clear caches between the two repeats of an already declared native smoke.

This is an operational readiness recommendation only. No GPU, CUDA query, remote command, host mutation, source edit, gate change or reclaim was performed by this reviewer.
