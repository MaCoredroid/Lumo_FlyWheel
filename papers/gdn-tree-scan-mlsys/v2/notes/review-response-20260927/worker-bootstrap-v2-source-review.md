# Worker bootstrap v2 — bounded independent source review

**Verdict: PASS for the reviewed source integration and CPU controls; no new blocking source defect found.** This does not qualify a live boot or open the workload gate. No engine, GPU, model, Docker container, request, or workload was launched. Existing accepted allocation-observer and request-observer findings stay closed.

Reviewed `workload-plan/tools/runtime-collectors/` bytes:

| File | SHA256 |
| --- | --- |
| `worker_bootstrap_v2.py` | `bb4353bde916098c245a9503882b9de33be7aa835048fdea86c38d03024a9cf8` |
| `worker_sitecustomize_v2.py` | `2a06b5ca31a9e258652c43009815660a569902b645821b327624b7ce37fb20fb` |
| `test_worker_bootstrap_v2.py` | `4d70e1edbc6c8577c2809ad7073b4d388bf0c43599a29ac62fe457ccb5a7ce33` |
| `request_worker_observer_v1.py` | `653088a9b40e7ec54cc249e80d80e8faa8ba9fcc249309aeeb967a9f157c9c14` |

The seven original implementation/test/dependency files are preserved in `p0/monitor/review-response-20260927/worker-bootstrap-v2-independent-20260928T233629Z/source/`; `SOURCE-SNAPSHOT.json` binds all seven, including the unchanged accepted allocation observers. Review context: `worker-bootstrap-source-review.md` and `request-worker-observer-source-review.md`.

## Import and binding conclusions

- Both exact-name finders are installed together before engine imports (`worker_bootstrap_v2.py:191`). The request producer itself imports only standard-library modules. The vLLM runner callback installs allocation, request-state/execute, and graph wrappers before returning the imported class. The separate input-processor callback wraps the ID assignment before its importer proceeds. Either input-processor/runner import order works.
- SGLang's scheduler can import the model runner while its own module is still executing. The runner callback installs allocation observation first; the scheduler callback then installs the queue observer once `Scheduler` exists. The retained scheduler constructs `Scheduler` inside `run_scheduler_process` at line 4966, after import, rather than at module scope. Static inspection found no direct module-scope construction of any of the four target classes. `PINNED-SOURCE-AST.json` retains exact source paths/hashes and call scopes; this is not a claim of transitive engine execution.
- `AfterSource` hashes, compiles, and executes the exact target source bytes, then checks them again before installing wrappers. This retains the repaired stale-pyc defense. `checked_module` checks actual loader, path, source digest, and cache location for the observer and selected dependencies. The request policy's boot/attempt/engine and overlapping source bindings must equal the allocation/bootstrap policy. The unchanged actual request installers additionally validate each method's defining source, not only a class name.
- Failed activation aborts `sitecustomize` through `SystemExit`. A failed target callback prevents successful import and its success receipt. Installation receipts are emitted only after the corresponding installers return. A SGLang allocation receipt can precede the scheduler callback; it must not alone be interpreted as a complete request-instrumentation or boot qualification receipt.

## Retained controls

Reproduced **9 supplied CPU tests**, all passing, including nested scheduler imports, vLLM ordering, refusal of wrong source/reload, timestamp-valid stale bytecode, and fatal shim activation failure.

Added **8 independent CPU controls**, all passing. These use the actual `activate`, `checked_module`, `Observer`, and request-wrapper installers with synthetic source-backed engine/dependency modules. Four positive controls cover both vLLM import orders and both SGLang import orders (including the nested runner import), verify wrapper presence, preserve original return values, and produce the expected source-bound ID/sampling records. Four refusal controls cover another boot, divergent source bindings, changed target bytes before import, and a non-static ID method rejected by the real installer without a success receipt.

Allocation installers were fixture functions because their source integration is unchanged and previously reviewed. `statvfs` read-only flags and `/proc` process identity were injected fixtures; these controls do not prove actual mount or process isolation. No scientific library was imported. Exact scripts, unedited test output, interpreter, and source bindings are retained alongside the snapshot in `independent_controls.py`, `parent-tests.log`, `independent-controls.log`, and `TEST-RECEIPT.json`.

## Required future launcher evidence (unchanged integration boundary)

The owned launcher must establish the private **empty, read-only** `PYTHONPYCACHEPREFIX` and `-B`/no-bytecode-write policy **before interpreter startup**, and mount the selected bootstrap/shim, policy, observers, and actual post-patch engine sources immutably. It must propagate the frozen policy and private source path to each relevant interpreter; setting the cache prefix after imports does not establish the reviewed bytecode guarantee. The shim's selected bytes must also be frozen by that launcher, rather than inferred solely from a bootstrap receipt.

The host consumer must join the applicable API/worker/scheduler installation and observation receipts to the owned boot and actual process identities. vLLM API and worker roles need not load both target modules in each process; pending unused finders are not evidence that a hook ran. Require actual allocation/request evidence and the named request's HTTP completion before admission. Installation is not allocation, graph dispatch is not device completion/timing, and none of these CPU controls establish numerical correctness or serving performance. No additional experiment scope, runtime setting, or gate change is proposed.
