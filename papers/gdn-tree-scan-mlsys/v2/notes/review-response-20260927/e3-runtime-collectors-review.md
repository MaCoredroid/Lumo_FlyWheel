# E3 runtime collector review — 2026-09-27

**Disposition: sealed CPU preparation verified; scoped repairs required before accepting it as the runtime collector implementation.** This is not a runtime freeze or launch approval. Genuine instrumentation producers, their engine call sites, the sole-executor wrapper, distributed clock binding and final attempt closure remain future work, correctly disclosed by the package. No Docker subprocess, remote operation, server/model request, evaluator task or workload launch was performed in this review. Only this note was written; tests used temporary local fixtures.

## Verification and supported behavior

The requested hashes match: `MANIFEST.json` = `554b49adbfa6bcf230e21d8345b359bf49e8b3811c3a4357bd0d555b4171e56c`, `collectors.py` = `a80b05067a9bfb06052e905abc50b099b3f66149d13d6be924b2167a45489823`, `integration.py` = `1736fdff7664009db3ee7652491059130a1d0495b422a29cffd0f0fd9ff7b987`. Running `verify_bundle.py` verified all **10** sealed collector files, all **29** accepted identity-adapter files, inspected source bindings and known locks. The accepted adapter remains unchanged at manifest `bee950319dbb59244f2cec4b12081c9fda16eec8352a7f538032f453608f4e95` and source `1bc3be933b08b0eb99432a2b70d0a91aed863b798bf6d3f764775b4dd01e7a06`.

I reran `python3 -B -m unittest discover -s <runtime-collectors> -p test_collectors.py -q`: **51 tests passed**. This is a CPU test count, not an experiment count; actual experimental observations remain zero.

The code supports the following bounded claims:

- `RealHost.run` permits only inspect/diff operations. Process inspection requires the local namespace, validates the observed container cgroup, retains actual argv/stat bytes, and rechecks PID/start/running/argv. Expected hashes do not replace file reads; effective source members are observed under the process root and checked against all three source families.
- Runtime values come from six required source-approved probe files. Their hashes, attempt/boot/container/request identities, observed payloads and producer membership are checked. Missing probes, an intended-configuration origin, missing actual seed, missing route execution or wrong settings refuse. Changed observed settings remain changed and are rejected by the accepted adapter rather than overwritten with expected values.
- Evaluator preparation requires an owned `created`, nonrunning, never-started container, no mounts and empty Docker diff. The integration hook checks it before the accepted gate and again afterward. A late state change retains the earlier gate result and a subsequent hook refusal. There is no hidden start callback.
- Content-addressed evidence is retained with exclusive creation and hash verification. Phase locking and one-time attempt-directory reservation prevent implicit retry reuse when the future sole executor holds the lock across the whole attempt. Nonzero agent exit plus a present empty-patch prediction produces `FAILED`; neither successful exit nor a nonempty patch claims a solved task.

## Four grouped findings

All reproductions below used the existing `test_collectors.CollectorTests` injected fixture, calling `setUp()` and `runtime_fixture()` where relevant, with `doCleanups()` afterward. No implementation/test files were changed. `run_runtime()` invokes the collector; `f.check('agent')` invokes the accepted adapter on the resulting packet.

### 1. Raw probe freshness is not enforced at gate close

**Reproduction:** leave all raw probes at their fixture timestamp `00:00:14`, then set `host.now='2026-09-27T00:02:14+00:00'`; run `run_runtime()` and `f.check('agent')`. Result: **`IDENTITY_GATE_PASS` despite a 30-second frozen age limit and 120-second-old probes**.

`collectors.py:194–195` only requires each producer timestamp to fall between boot and collection. `finalize_stage` and `_envelope_at_clock` then stamp the enclosing observation with the new collection time. The accepted adapter's maximum-age check consequently sees a fresh envelope, not the age of the reported runtime behavior. A live process and unchanged argv do not establish that its earlier route/seed/accounting observations are fresh.

**Minimal repair:** at the actual closing gate clock, enforce the approved age/skew policy on each producer timestamp, retaining both production and collection times. Keep the bound raw bytes. Do not silently increase the window or replace the producer time with collection time; hashing large model files may legitimately require an explicitly reviewed age policy. Add the 120-versus-30-second fixture as a refusal case.

### 2. Agent-container ownership is weaker at admission than at completion

**Reproduction:** set `host.containers[agent]['Labels']['lumotree.configuration']='wrong-configuration'`, preserving other injected values. `run_runtime()` followed by `f.check('agent')` still returns **`IDENTITY_GATE_PASS`**.

`runtime` calls `agent_reader.owned(agent,packet)` without `boot=True` at line 201, checking campaign/attempt/phase only. Completion later checks boot/configuration/task as well. Thus a contradiction detectable before dependent agent work is deferred until completion. The runtime subject records the agent image but not its container ID, so completion is also not explicitly tied to the precise agent container inspected at admission.

**Minimal repair:** enforce the full boot/configuration/task ownership tuple during admission, retain the actual agent container ID in the bound runtime subject, and require the same ID/image at terminal collection. Preserve the explicit separate-host collector arrangement. Add admission refusal for contradictory labels and terminal refusal for a replacement container. These are collector/envelope changes, not authorization for real launcher insertion.

### 3. Delivered-seed presence is verified; agreement with the frozen seed is not

**Reproduction:** replace `sampling_seed_delivery.observed.effective_seed` with `7`, regenerate the fixture's raw probe hashes with `refresh_probes()`, then collect and validate. Result: **`IDENTITY_GATE_PASS`**, just as for the fixture's original `26092700`.

The collector checks that the delivered seed is a nonnegative integer and was applied to the bound probe request. It never compares that integer with an independently derived expected value. The integer is retained in raw evidence but not in the runtime subject validated against `effective_common`. The accepted adapter requires a nonpending `seed_derivation` string and a `VERIFIED` status; these do not execute or bind a per-request seed derivation.

**Required next binding:** before runtime freeze, define the reviewed executable derivation or explicit expected probe/request-seed mapping from the scheduled seed block/root, attempt and request identity. Compare the observed delivered value to that independently bound expectation and retain it in the subject. Do not manufacture the expectation from the observed probe itself. This may require a coordinated accepted-adapter/freeze schema and source update; **do not silently modify the accepted 29-file adapter**. Until then describe this collector as proving seed presence/request delivery, not correct frozen-seed use across workload arms.

### 4. Missing prediction preserves raw Docker bytes but lacks a terminal failure record

**Reproduction:** use the existing exited-agent fixture with actual `ExitCode=137`, then call `agent_terminal(...,'/missing-prediction.json')`. Result: `KeyError` from the injected host, **one new Docker evidence blob and zero terminal records**. On a real missing file the corresponding filesystem exception has the same control-flow consequence.

The existing nonzero-exit/empty-string-patch test is valid, but a crashed attempt with **no prediction** never reaches terminal envelope creation. `agent_terminal` has no journaled failure hook analogous to agent/evaluator gate refusal. It is therefore too broad to claim that this collector alone retains all failed attempts as terminal records.

**Minimal disposition:** preserve the actual exited-container observation and an attempt-bound missing/unreadable/malformed-prediction failure receipt in the owned journal even when evaluation cannot proceed. The future wrapper must close that attempted row as failed/infrastructure-failed as supported by real evidence, with prediction absence explicit. It must not invent an empty agent patch, drop the attempt, advance the schedule without its closure, or retry in place. If final `attempt_closure` remains a later wrapper responsibility, keep that obligation explicit and add a CPU failure-path test proving that the reservation plus failure receipt survive. No adapter weakening is needed to make missing prediction evaluable; refusing evaluation is correct.

## Scope after repair

These findings do not invalidate the byte seals or the 51 passing tests. They identify missing adversarial cases in the prepared implementation. The first two are narrow collector fixes; seed correctness requires its pending frozen protocol binding, and prediction absence requires a retained failed-attempt path through the future closure workflow. All are required before the corresponding runtime claim/gate use, without adding workload experiments.

The proposal file remains a source/role proposal, not producer approval. The package has no real producer and no actual call-site insertion in the historical SWE agent/evaluator runners. A future wrapper must still use the pinned dataset, immutable agent/evaluator images, one coordinator clock or reviewed bridge, shared evidence storage, exact pre-start evaluator boundary and all-attempt closure ledger. None of those future integration facts is established by injected-fixture success or by this review.
