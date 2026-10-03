# Candidate continuous dispatch — bounded independent review

**PASS for the new callback-dispatch source and local CPU controls.** Exact source `q1_candidate_sequence_dispatch_v1.py` SHA-256 `b55be8bb7a347f1898ff7e92895f047084f89aac1b995392e3fefdae036878f8`; accepted runtime remains `f8735399464c6f71ce400c7d13cba7c0fbdb6a082a71630ba1fe34e153b598d8`. No new blocking defect found in this delta. This is not a launcher, outer collector, gate or qualification approval.

## Verified behavior

- `_sequence_active` (lines14–23) checks enabled/sealed state, bound request, invalid-case reasons, sequence invariants and actual runner identity. Wrong runner and invalid binding fail before lower callbacks. Disabling callbacks does not bypass an already-latched unusable process.
- `on_pre_forward` (lines26–36) executes binding, active-request/runner validation, actual boot-attestation call, explicit `boot.fatal is False` / no-fault check, then the inner runtime callback. The positive control preserves exact argument objects and return value. Fatal, absent or faulted boot refuses before the inner callback/import seam; wrong binding, wrong runner and invalid case refuse even before the attestation seam.
- The actual served entry names route sampled output, draft proposals, target logits and acceptance products to the accepted runtime. Inactive sampler dispatch preserves the exact product object. Active acceptance without a prepared frame refuses. No cycle0 forcing callback is invoked by this mixin.
- `on_sealed` (lines56–68) permits a frameless callback only for initial prefill at cursor ready/index0, without calling the inner tree seal or finalizer. A frameless interior boundary refuses. The connected complete flow invokes finalization only after the terminal deferred event and passes the actual completed result to the outer finalizer.

## CPU evidence

**14 independent dispatch controls passed (4 valid/no-op flows, 10 refusals), and all8 supplied connected entry-name tests passed locally.** The connected suite includes complete three-cycle plus terminal ordering, publication-length failure, foreign runner, retained receipt/O2 corruption and exact finalization timing. It reuses the already-reviewed lower components; their neural outputs and model/cache snapshots retain the supplied fixture's explicit isolation. No new raw-state or model claim is made.

The independent pre-forward controls inject the future outer binder, boot attestor and inner call at their declared boundaries to establish ordering and refusal. They do not claim that an actual worker image was attested or a real job was admitted. The accepted runtime's real Maps/Joint/H4/import tests are reused by reference, rather than repeated.

Local Torch is Python3.9, so the harness retains the prior namespace-only strict-zip and positive-integer bit_count equivalents for preserved CPU fixtures. The dispatch and accepted runtime bytes were not edited. Source/test snapshots, `independent_controls.py`, `RESULTS.json`, `controls.log`, `SOURCE-MANIFEST.json` and `REVIEW-SEAL.json` are under `p0/monitor/review-response-20260927/sequence-dispatch-review/`.

## Remaining outer-owner obligations

`_sequence_bind_runner` and `_sequence_finalize_record` are intentionally absent. Their future implementation must bind exact admitted job/control/source/request state, keep binding free of model-state import before this dispatch's boot gate, serialize/authenticate the completed record and mark final case state appropriately. The future singleton/patcher/launcher must select this dispatch and pin all dependencies. These are existing integration boundaries, not newly completed evidence. No GPU, remote CPU, runtime process, cache operation, gate or qualification count changed in this review.
