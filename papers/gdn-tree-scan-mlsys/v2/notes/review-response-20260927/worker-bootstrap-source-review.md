# Worker bootstrap — bounded independent source review

2026-09-28. **Bounded source closure PASS after the F1 repair below; the initial defective snapshot and reproduction remain preserved.** This review is CPU/source-only and confers no runtime qualification, workload admission, or launch authority. No engine, model, GPU, HTTP request, container, live worker, or actual `/proc` observation was invoked. The only spawned interpreters ran synthetic modules/sitecustomize refusal controls.

Paths are relative to `papers/gdn-tree-scan-mlsys/v2`; helper names denote `experiments/review-response-20260927/workload-plan/tools/runtime-collectors/`. The initial source/test/dependency bytes are preserved under `p0/monitor/review-response-20260927/worker-bootstrap-independent-v1/`, with `SNAPSHOT.json`.

## F1 — source hashes do not bind code executed from an existing bytecode cache

At initial `worker_bootstrap_v1.py` SHA `b9cc42b42d9fe8babc6d5b736acab7091dd97a2208f30c32ea5ee878d4d14d1d`, `AfterSource.exec_module` verifies the `.py` digest before and after delegating to `SourceFileLoader.exec_module`. That loader may execute an existing timestamp-valid `.pyc` rather than the source just hashed. `checked_module` similarly checks the source path/hash only after a normal import.

Independent stdlib reproduction: compile a synthetic module with `VALUE="evil"`; replace the source with same-length `VALUE="good"`, preserving its modification timestamp; freeze the new source hash; import through the exact `OnceAfterImport` hook. The executed module and successful post-import callback both observe `evil` while the final source hash still matches the frozen `good` file. The result is preserved in `stale-pyc-reproduction.json` and reproduced by `independent_cache_controls.py`. `PYTHONDONTWRITEBYTECODE=1` alone does not prevent existing-cache reads.

Minimal target repair: read and hash source bytes, compile those exact bytes with the original source filename, and execute that code in the original module namespace, preserving ordinary module/spec semantics. Do not infer executed source from a file hash around a cache-enabled import.

For dependencies imported by the runner before the callback, the proposed owned-launcher policy is conditionally sufficient for ordinary `SourceFileLoader` caching: establish an empty, dedicated, read-only `PYTHONPYCACHEPREFIX` **before Python starts**, together with `PYTHONDONTWRITEBYTECODE=1`; validate actual `sys.pycache_prefix`, `sys.dont_write_bytecode`, directory identity/emptiness and selected dependencies' regular source loader/spec/cache paths. Keep the selected source files immutable/read-only for that process. Do not reload or re-execute already-loaded scientific modules. Applying this policy only after imports, admitting a populated/writable prefix, or merely changing the module's `__cached__` attribute would not establish this guarantee.

One independent toy subprocess confirms the proposed Python behavior: with the stale sibling `__pycache__` left in place, startup private prefix plus `-B` loads `good`, locates `__cached__` under the private prefix, and leaves that prefix empty. This demonstrates cache selection only; it does not establish a real read-only mount, caller startup environment, or runtime admission. Source-level implementation of the proposed repair remains to be reviewed.

## Other checked boundaries

- The finder intercepts only the exact engine runner module, requires an ordinary source-module location/hash, and refuses already-imported targets, duplicate activation and reload. Its callback runs after the original module body and before subsequent runner initialization calls. The supplied synthetic test confirms that order while preserving the original allocation method's return value. The pinned native runner snapshot has no `GPUModelRunner(...)` module-body construction; the inspected SGLang runner exposes allocation as an instance method. No runtime claim follows from those static observations.
- Activation reads the approved policy/hash, safe attempt/boot identifiers and dedicated directories before installing its finder. The vLLM callback selects the already-reviewed v1.2 metadata hook, and the SGLang callback selects the reviewed v1.1 producer/dependency pair with explicit identity equality. Source/path bindings and initialization-method wrapping are retained; no forward/sampler/kernel implementation is edited by the bootstrap.
- The shim is inert when both private policy environment variables are absent. Partial/bad activation raises `SystemExit`, so Python's ordinary sitecustomize exception suppression cannot silently continue into the engine. The actual child-interpreter refusal control exits nonzero and never prints its engine-entry marker. Future target-import or callback exceptions propagate from the import; no successful installation receipt is written in that case.
- The bootstrap does not set unrelated environment variables or modify installed global files. It changes only the current interpreter's import finder and, upon the exact target import, its initialization wrappers. Installation receipts are write-once/fsynced and carry PID, attempt/boot, policy/bootstrap hashes and runner source identity. These are installation receipts, not allocation observations or scientific qualification.
- Final owned caller/source-policy/mount integration remains separate: install the shim on the new boot's private Python path before any target import, preserve existing scientific arguments, enforce source and dependency bindings, join allocation/ownership/installation receipts for the correct live worker, and refuse missing instrumentation. The new private-cache policy also needs that caller-level startup/read-only proof. No standalone helper can authorize an attempt.

The five supplied CPU tests pass against the preserved initial source. Command: `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_worker_bootstrap_v1`. Independent cache control: `PYTHONDONTWRITEBYTECODE=1 python3 independent_cache_controls.py`, run only inside the review snapshot.

| Initial reviewed source | SHA256 |
|---|---|
| `worker_bootstrap_v1.py` | `b9cc42b42d9fe8babc6d5b736acab7091dd97a2208f30c32ea5ee878d4d14d1d` |
| `worker_sitecustomize_v1.py` | `7a2fca8f17add346a01645217048d7662ab629947acd0acac46bbc96fa5a1807` |
| `test_worker_bootstrap_v1.py` | `ea9ac9800f7946c6a319a4bf5b9be8c682471115e05730f31baec297f98c718b` |
| `worker_metadata_v1_2.py` | `bf08b96a9f11adb38513f1d4376a6b32d438c464f9f073ae03a73a11eaa43875` |
| `sglang_worker_metadata_v1_1.py` | `08175a1a7a70857925012992295233294b046168c922122f97a8f9be26ef5988` |
| `worker_metadata_v1_1.py` | `e02fea266846223296b9d64bfa5e857524d98bb21c7d05a9c0c78c1c0907ec4b` |

## Repair closure — 2026-09-28

**F1 is closed for the checked source bytes. No remaining material blocker was found in this bounded delta.** `worker_bootstrap_v1.py` SHA `a1a46aa5861697e952075f3e883edec450af35c2d14bef01c721d30bfecee303` now reads, hashes, compiles and executes the same target bytes (lines 71–78), bypassing `SourceFileLoader.get_code` and its stale cache. The post-import callback still occurs after module execution. It does not reload scientific dependency modules.

The dependency-cache policy is checked at activation (line 110), before each selected dependency check (lines 52–60), and when sealing the installation receipt (lines 148–149). Lines 34–49 require the frozen absolute directory, matching interpreter/environment prefix, bytecode writing disabled, a read-only filesystem flag and an empty directory. Dependencies must have exactly `SourceFileLoader`, matching spec/file location, the pinned source hash/path and a cache location within that prefix. These checks address the original cached-code ambiguity under the prospective owned-launcher contract. They cannot independently prove when interpreter settings were established or that sources were immutable before their imports; the future caller must establish the empty read-only prefix **before Python startup** and mount selected source paths immutably. No late policy assignment or post-hoc cache-path attribute update is accepted as that startup evidence.

The exact repair source/test/dependency bytes and independent controls are preserved in `p0/monitor/review-response-20260927/worker-bootstrap-independent-v1-repair1/`. The current canonical copies were compared byte-for-byte with that snapshot at closure. Only the bootstrap and its supplied tests changed from the initial reviewed helper set; shim and three metadata producers are unchanged.

Validation in the repair snapshot:

- `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest -v test_worker_bootstrap_v1`: **7/7 supplied tests passed**.
- `PYTHONDONTWRITEBYTECODE=1 python3 independent_cache_controls.py`: **2 independent controls passed**. The original timestamp-valid stale-cache counterexample now executes the pinned `good` source. The actual toy subprocess with a startup private prefix also executes `good` and leaves the prefix empty.
- `PYTHONDONTWRITEBYTECODE=1 python3 independent_policy_controls.py`: **11 independent controls passed**: valid policy/dependency admitted; absent interpreter prefix, mismatched environment, enabled bytecode writes, writable/populated prefix, external cache path, mismatched spec/file, non-source loader and wrong source digest refused.

The policy controls use mocked `statvfs` read-only flags and synthetic interpreter settings. They prove the validator branches, **not** an actual read-only mount or live worker integration. The toy interpreter imports only synthetic source. No engine, model, GPU, container, HTTP request or workload was run. Existing owned-CID/process-epoch, target-layer/precision/parallel identity, exact source freeze and actual allocation-receipt integration remain required and unqualified by this review.

| Repair artifact | SHA256 |
|---|---|
| `worker_bootstrap_v1.py` | `a1a46aa5861697e952075f3e883edec450af35c2d14bef01c721d30bfecee303` |
| `test_worker_bootstrap_v1.py` | `772ef921389dc614c2d942f1d8afdf994420b9f79154c4a627e554cf4ee35093` |
| `independent-cache-results.json` | `16c7ce746873eb1a0da59f1c2c11517fd975348fe73bf2d0d86a83ac3f8342c5` |
| `independent-policy-results.json` | `7eb4107f0e193bfdef82b87826ba3c9c34ebfc2ea595ad7cbce00c622de8cad4` |

Disposition: **ready for a final source freeze review of this helper; not runtime admission or launch authorization.** Actual caller startup, read-only mounts and receipt joins are pending integration boundaries, not tests claimed complete here.
