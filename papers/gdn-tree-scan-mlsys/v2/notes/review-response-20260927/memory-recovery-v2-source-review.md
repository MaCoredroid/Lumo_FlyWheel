# Memory recovery v2: bounded independent source review

**HOLD — not ready for final source-freeze acceptance.** The intended single bit-2 action is appropriately narrow, but this draft does not yet implement the policy's one-use authority or exact-owned-container cleanup. These are operational launch blockers; **no reclaim, query or scientific launch is authorized by this review**. Fix the existing guard paths and submit one settled version with focused CPU regressions; no broader experiment or recovery campaign is requested.

## Reviewed identities and scope

- Immutable snapshot: `p0/monitor/review-response-20260927/candidate-runtime-draft-20260928T0418Z/tools/memory_recovery_v2.py`, SHA256 `71afe729a0670558b519c4009d180c7cf4b3892d2c9d7150cc6b6b56f0a027a3`. All line numbers below refer to this file.
- Prospective policy: `MEMORY-RECOVERY-PROSPECTIVE-POLICY-v2.md`, SHA256 `0182385a43a98b091777a8ba695026e70359908d7706b29f5efb68019c062913`.
- Prior scope/timeout clarifications: `notes/review-response-20260927/memory-recovery-request2-review.md`, SHA256 `4169bb16216fad268435341832294026c585d49761e2291c8737146aba217e18`.
- Author log: `tools/test_log.memory_recovery_v2.attempt1.txt`, SHA256 `2098f7b757d2f6935443402e3c1513758034993bc6c0bb780347f5ec0f394733`. It records 69 CPU tests with source hash unchanged. I inspected the log; I did not independently rerun or adopt those tests as evidence of closure.

I read the complete 1,161-line source and executed only selected methods/parsers with in-memory fake command results and fake persistence. I never called `main`, `Recovery.__init__`, `real_deps`, `real_run`, SSH, Docker, CUDA, cleanup or sysctl. No filesystem mutation occurred except this review note. The module was imported with bytecode writes disabled.

## Required corrections

### F1. Cleanup can remove an unrelated same-name container

Lines **914–931** run Docker and then inspect the requested name regardless of creation success. The CID file is merely recorded. Lines **964–985** accept any successful inspection and call `docker rm -f <name>` without comparing the returned ID against this invocation's proven CID, image or ownership label. A pre-existing **stopped** same-name container is not covered by `docker ps` (723); Docker can reject creation while cleanup still removes that unrelated object. A name replacement after inspection is also acted on by name.

Independent in-memory replay returned `unrelated-container-id running` for the first inspection. The actual method emitted `docker rm -f collision`, then reported cleanup proven after a mocked missing-container response. It never required a CID match.

**Minimal fix:** establish creation ownership before any removal. Prefer create → capture full CID/label/image → bounded start/query → inspect/stop/remove that exact CID. Refuse existing names; do not clean anything when creation ownership is absent or mismatched. Preserve a pending/unknown state if a timed-out creation has not resolved; an immediate name absence alone must not clear a potentially still-arriving daemon creation. Add collision/wrong-CID/no-CID/late-creation and exact-owned timeout controls. No unrelated object may be removed.

### F2. One-use protection is output-directory-local, not authorization-wide

The lock and pre-write attempt marker are derived from freely supplied `--out-root` (**377–382**, **1126**). The authorization's shared `consumed_marker` is checked only during authentication (**578–591**) and created only during final sealing (**1048–1059**). Thus two invocations with the same unconsumed authorization/run ID and different output roots have different locks/attempt markers and can both reach the write before either seals. The same gap permits replay after an ambiguous/crashed attempt by changing output root. The existing per-instance write counter is not sufficient.

Independent in-memory calls to the actual `_write_attempt_marker` accepted both `/virtual/root-A/ATTEMPT-v2-same-approved-run.json` and `/virtual/root-B/ATTEMPT-v2-same-approved-run.json` under the same authorization hash. This demonstrates the namespace separation; no real write was issued.

**Minimal fix:** bind one canonical operation-lock/attempt namespace in the parent authorization (or immutable policy), and durably reserve the authorization-wide one-use marker with exclusive creation **before** the write. Any preexisting reservation refuses. Retain it after timeout/crash regardless of receipt success. Authenticate/recheck expiry and gate bytes under that lock immediately before action: currently expiry is tested only at **565–567**, before all potentially slow inventory/capture work. Keep uncertain write/query termination under a persistent operational hold; ordinary timeout/cleanup-unproven returns currently leave `retain_lock=False` and release the lock at **1077–1111**. The future scientific boot must honor that hold.

### F3. Authorization does not pin several effective policy inputs

Authorization binds run ID, script hash, image ID and scientific-gate hash (**547–590**), but the effective reservation, timeout bounds, driver/kernel expectations, source-manifest hash/directory, desktop-client allowlist and allowed container names remain caller-selected (**1126–1137**). For example, `--reservation-gib 1` is valid according to **1149–1152**, then directly becomes the success threshold at **916, 938–949**; this changes the unchanged reservation without changing any authorized field. An arbitrary desktop allowlist or expected source-manifest hash likewise changes what is admitted while the same authorization remains valid.

The memory query's device check only requires a nonempty string (**347–349**), not the authorized device identity/total. The pure validator accepted `{"device":"unexpected device","total_bytes":1000000000000,"free_bytes":999999999999}` with rc 0. A pinned image does not identify the GPU on which the query ran.

**Minimal fix:** bind a complete small operational-config hash (including canonical paths and allowlist bytes) or refuse overrides from frozen constants. Freeze the reservation and timeout limits, bind the approved PID/start/comm allowlist, and check the query against the approved GPU identity/expected total-memory contract. Keep the parent-issued scientific-gate hash/run binding authoritative; no new gate or permission is inferred here.

### F4. Some inventory and counter discrepancies are accepted

The declared fail-closed parsing has concrete holes:

- `parse_nvidia_compute_apps` (**322–326**) accepts the malformed single line `pid GARBAGE` as an empty compute inventory.
- `_inventory_gpu_clients` (**777–785**) accepts `fuser` rc 0 with empty stdout/stderr as zero clients; that inconsistent successful-accessor result is not rejected. The PID/start/comm comparison itself is useful when PIDs are actually observed.
- `_postconditions` (**875–905**) validates only the immediate before/after counters. The separately captured after-ten-second counters are ignored. An in-memory before slab/pagecache `(3,0)`, after `(4,0)`, after10s `(4,1)` produced `ok=True` with no discrepancy, then permits querying despite observed page-cache-drop evidence.

**Minimal fix:** require the exact expected compute CSV header/record grammar and consistent fuser return code/output, and evaluate both captured post-action counter checkpoints before admitting readiness. Keep malformed/discrepant evidence and stop; do not retry. These are focused negative cases, not a demand for additional host probes.

### F5. A cleanup/seal failure can leave a success receipt admitting the next boot

`_seal` computes `boot_hold` and serializes the receipt before lock release (**1069–1112**). If `_release_lock` fails, it appends an error and returns process exit 12, but the already-written receipt still reports exit 0, empty `seal_errors`, `released_after_seal:true` and `next_boot_permitted:true`. Independently replayed with in-memory persistence and an injected release failure: **returned rc 12; sealed exit 0; sealed boot permitted true; sealed lock released true**. A consumed-marker write failure can also leave `boot_hold` true because its calculation ignores seal errors.

**Minimal fix:** readiness must require successful durable one-use reservation and terminal bookkeeping. Record actual release/retention outcome before the final immutable receipt (or use a finalization receipt that supersedes only a clearly preliminary receipt); any failure sets boot hold and nonzero receipt status. Do not label a planned release as completed. Keep an exclusive hold through finalization so fixing this ordering does not introduce an operation race.

## What is already sound within this draft

The exact reclaim argv contains only `echo 2 > /proc/sys/vm/drop_caches`; the per-instance command counter prevents a second write; there is no model launch or setting adjustment. The attempt file uses exclusive creation plus file and parent-directory fsync. Lock acquisition precedes the inventories in `_flow`. Required command failures generally stop the stage, write timeout never triggers an automatic second write/query, and insufficient capacity is a terminal result. These pieces can be retained while repairing the authority, ownership and receipt paths above.

The next review should be of one new immutable source/test snapshot resolving F1–F5. No recovery should be performed using this draft, and the earlier consumed authorization must remain consumed.
