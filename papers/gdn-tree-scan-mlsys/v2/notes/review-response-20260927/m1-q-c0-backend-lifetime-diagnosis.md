# M1-Q C0 second-cycle Backend lifetime diagnosis

2026-09-28. Read-only source diagnosis; no implementation or GPU execution by this reviewer. The existing run is `m1-q-c0-20260928T221747Z`. Eight status files were read over SSH and preserved byte-for-byte in `p0/monitor/review-response-20260927/m1q-c0-lifetime-diagnosis-20260928T222228Z/observed-status/`, with a read-only command/UTC/hash receipt.

Observed dispositions: Lumo repeat 0 is `complete`; Lumo repeat 1 is `failed` with `FR13_FIXED32_CONV_PREGATHER builder SSI group changed after preseed`, reported after zero scientific calls/zero recorded steps. All six Weaver/TreeWY repeats are `complete`. Zero recorded scientific calls does not mean zero GPU allocation or setup work: the traceback reaches Backend construction and boot registration.

## Cause and exact source chain

- `tools/m1/m1_q_c0_collect_v1.py` creates a fresh `ImageExecutorV3` for each cycle through its factory.
- `tools/m1/m1_executor_image_v3.py:49` initializes `backend=None`; `_ensure_backend`, lines 56–68, constructs `R.Backend` when it is absent.
- `tools/q1_component_runner_v2_2.py:356` allocates fresh builder-owned `ssi_groups`. Constructor line 360 calls `boot_lifecycle`; line 144 registers their pointers.
- `src/lumo_flywheel_serving/fr10_gdn_tree_kernel.py:7639` holds the process-global preseed state; lines 7711–7720 reject a new SSI pointer or stride after that state exists. Registration retains the actual old tensor at line 7729. Dropping an executor does not make its global contract unbound.

The local runner and kernel match the frozen hashes: runner `ab27fc9cf680bdd2b7c3ad994fc9f7bffe0f9ff1b0f35dc2cc36018f364dea29`, kernel `d9dd0c697b466180ab5d834a52555c1516edb534e0a3d1bd3094e0d1489c24c8`. The raw traceback follows these source lines. This is a repeated-construction/lifetime failure, not a reported numerical disagreement for the missing repeat.

## Smallest repair direction

Keep a single fully constructed Lumo Backend and its runner module for the two cycles. Give each cycle a fresh `ImageExecutorV3` facade and the existing fresh `CapturingExecutor`, binding the same Backend into the next facade. Retain only the Backend/module in the factory after a cycle; do not carry its old log, results, `_lifecycle`, `failed_calls`, capture callbacks, or previous tensor result references. Capture the Backend after `run_method` on all exit paths where construction completed; never accept a partially failed constructor as a usable Backend. Preserve the existing uninstrument-in-finally lifecycle.

This fits the frozen amendment: one fresh GPU process, two complete untimed cycles per method, and reset before each cycle (`M1-Q-STANDALONE-UNTIMED-SCHEDULING-AMENDMENT-v1.md:30`). It does not require fresh Backend storage per repeat. It needs a versioned collector/helper and parent-bound source/lifecycle clarification. The original run and executed sources must remain preserved; a new parent-authorized repair run is a disclosed implementation rerun on the same held-out inputs, not new independent qualification data or a replacement that erases earlier outcomes.

The author branches do not need this change. `m1_executor_image.py:93–108` allocates/resets author buffers and invokes author warmups. Only the Lumo branch at line 109 and Lumo bank-digest branch at line 136 request this Backend. The completed author statuses support that reading. Leave those branches unchanged.

## Reset and counter boundary

Continue calling the unchanged `load_instances` and `restore_O0` before every Lumo cycle. `restore_O0` (runner 373–379) restores S0, scratch/control rows and rings/flags/output in place. It omits constructor-zero pending metadata: `accepted_paths`, `accepted_lens`, and `prev_lens`. Zero those existing public buffers in place at the cycle boundary. `publish` overwrites accepted metadata at 407–410, but explicit reset matches the amendment and avoids a stale pre-publication state.

`spec_idx` is the fixed run/scratch-row mapping initialized at runner 344; builder SSI tensors are initialized to zero at 356. The reviewed scan/publish path reads these tensors, and boot warmup already promises input-state restoration. Retain their objects and assert expected values and pointer identity; do not re-register groups or replace tensors. `Backend.pointer_identity` covers the SSI groups, commit SSI, banks, rings and accepted metadata; separately retain/check `prev_lens` identity, which is absent from that map. Do not zero whole alias pages or reset private preseed/graph/metadata-lease dictionaries.

Keep process-global counters cumulative. `Backend.publish` already returns an observed after-minus-before replay count (runner 411–420). The reset receipt's `committer_counters_after_reset` is an absolute observation, not a promise of zero. Record before/after cycle counter snapshots and deltas, preserve the original once-per-process boot receipt, and label reuse explicitly. Do not claim a second boot, cleared counters, or method-isolated memory peaks. The existing memory observer should remain active; retained Backend storage is visible live memory.

## Bounded acceptance controls for the parent implementation

Use fake Backend/facade objects to establish one constructor across two Lumo cycles, identical Backend/SSI pointers, new logs/results/captures each time, pending-metadata reset before calls, preserved cumulative counters/deltas, and no Backend construction in author cycles. Test an exception after successful construction so the holder and partial records remain correct; test a constructor exception so no incomplete Backend is cached. Verify observer wrappers are removed after both normal and exceptional exits and no old facade/capture object stays alive solely through the holder.

Before any repaired candidate run, bind the new helper/entry source and prospective lifecycle clarification; preserve original evidence and one-use authority. At runtime retain exact reset-S0 capture, original input hashes, full two-repeat raw outputs/states and pointer/counter receipts. Score every retained observation under the unchanged frozen rule; do not alter tolerances, operands, accepted paths, method arithmetic or global scientific cache to obtain completion.

This note diagnoses and recommends the lifecycle repair. It does not certify unimplemented code, authorize a launch, or establish numerical qualification.
