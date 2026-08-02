# Fixed32 CFWD paired-ingress patch plan

Classification: source-only implementation plan; not a performance result.

## Decision

The requested qualification-to-timing rotation is materially broader than a
small ingress state-machine extension, so this branch does not implement it.
The current engine and proxy ingress objects each own one terminal
`preflight -> campaign -> finalized` lifecycle and one O_EXCL ledger. The
serve wrapper owns one begin/run/finalize window, while the runner's timing
handoff is deliberately process-local. Reopening either ingress in place would
mix identities and counters, and invoking the runner twice would lose the
handoff that is required to authorize timing.

This artifact defines the narrow implementation boundary needed for a later
change. It does not alter kernel math, launch a campaign, or authorize a
hardware-floor claim.

## Required design

### 1. Opt-in phase protocol

Add a v2 paired-phase protocol only for the existing fixed32 CFWD qualifier.
The default v1 ingress behavior must remain unchanged.

- Phase identities are exactly `qualification` and `timing`.
- Both phases bind concurrency B4, the canonical exact4 or exact16 task-set
  digest, source commit, server boot identity, mode, and flush generation.
- Phase identity is independent of the existing ledger lifecycle field. A row
  therefore identifies both its campaign phase and its lifecycle state.
- Reusing a SWE-Verified task ID is valid only when the authenticated phase
  identity differs. A repeated `(phase, task_id)` is rejected.
- Authentication covers the phase identity, canonical task-set digest, prior
  finalized ledger head, and transition nonce. The nonce and timing handoff
  remain process-local and are never published as credentials.

### 2. Fresh per-phase ingress state

Engine and proxy must rotate symmetrically after qualification finalizes.

- Finalize and close the qualification ledger.
- Require no active logical requests, attempts, or engine requests.
- Create a distinct timing ledger with O_EXCL and mode 0600. Never truncate,
  reopen, rename over, or append timing rows to the qualification ledger.
- Allocate fresh timing counters and active-request maps. Qualification
  counters remain immutable for validation and publication.
- Start timing at ledger sequence zero and zero admitted/completed/rejected
  counters.
- A one-sided rotation or begin poisons the paired run. There is no rollback
  and timing must not start.

The phase transition should be a new paired-mode method on
`Fixed32ProxyIngress` and `Fixed32EngineIngress`, exposed through authenticated
v2 control requests. It must not weaken the current terminal semantics for v1.

### 3. Persistent paired controller

Run qualification and timing from one Python controller invocation under one
server boot. Refactor the current runner body into an internal single-phase
callable; do not launch two independent runner processes.

The controller performs this sequence:

1. Begin authenticated qualification on proxy and engine.
2. Run canonical exact4 or exact16 qualification with the one B4 campaign
   marker and separate qualification output root.
3. Remove the marker, complete the post-qualification flush, and require full
   reachable B1-B4 accepted-length coverage (`0x0fff` for every B).
4. Finalize both qualification ledgers and bind their terminal heads into the
   process-local `_Fixed32CfwdSameServerTimingHandoff`.
5. Rotate both ingress objects to fresh timing state. Abort on any asymmetric
   result.
6. At the immediate timing pre-boundary, consume the handoff only after
   rechecking unchanged server boot, producer PID, mode, source commit,
   canonical task set, later generation, marker absence, coverage, and gate
   attempts.
7. Run the same canonical task set into a separate timing output root.
8. Finalize both timing ledgers and perform one final campaign teardown.

Qualification remains `performance_measurement=false`. Only the timing phase
may feed throughput, acceptance, or floor reducers.

### 4. Publication and validation

- Preserve separate engine and proxy qualification/timing ledgers and
  begin/finalize receipts.
- Publish only hashes for process, boot, nonce, and in-memory handoff identity;
  do not publish replayable credentials or raw SWE task output.
- Extend provenance validation to require matching engine/proxy phase identity,
  canonical task-set digest, source commit, and terminal ledger heads.
- Timing is eligible only when qualification coverage is complete, the
  in-memory handoff is consumed exactly once, both timing ledgers finalize, and
  the timing post-boundary reports zero gate-attempt delta.
- Timing reducers must read only the timing output root and timing metric
  window. Qualification samples must never enter a performance aggregate.
- Keep generic B4 qualification auto-publication disabled so raw prompts,
  responses, patches, traces, logs, and runtime identifiers are not swept into
  git.

## Fail-closed conditions

Reject timing on reboot, producer-PID change, generation regression, mode or
source drift, task-set drift, live marker, incomplete coverage, changed gate
attempt count, ledger-head mismatch, replayed transition, active requests, or
an engine/proxy phase mismatch. Once either ingress transition fails, the
paired run is terminal and the server must be torn down.

## Intended source scope

- `src/lumo_flywheel_serving/inference_proxy.py`: v2 phase-bound ledger and
  symmetric proxy/engine rotation, guarded by an explicit paired-mode option.
- `scripts/run_swe_bench_q36_a.py`: reusable single-phase runner and persistent
  qualification/timing controller that owns the live handoff.
- `scripts/fr13_bigdenom_swe_serve_variant.sh`: opt-in paired route, one server
  boot, separate phase roots, and one final teardown.
- Fixed32 ingress, wiring, campaign-provenance, and layer-batch test modules:
  protocol, replay, drift, isolation, and reducer tests.

No general harness path should observe a schema, state, ledger-path, counter,
or teardown change unless the paired CFWD route is explicitly selected.
