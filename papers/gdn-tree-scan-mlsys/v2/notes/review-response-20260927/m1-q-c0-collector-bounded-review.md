# M1-Q untimed C0 collector: bounded source review

2026-09-28. Initial reviewed collector SHA-256: `7309d30d9307526ff142b990d72bf4edba3908ae92e2aa454afade9ea84518bc`. Snapshot: `p0/monitor/review-response-20260927/m1q-c0-collector-reviewed-20260928T215748Z/`. This review runs no Torch, candidate, fixture generator, GPU, Docker, or launcher. The six independent AST controls use only opaque CPU strings/dictionaries. All six behaved as expected, including two reproduced defects; that is not a clean implementation verdict.

1. **Final index hashing can suppress the final failure receipt.** The initial collector's `finally` directly calls `sha(store.index_path)` before setting `ended_utc` and persisting the receipt. A missing/unreadable index raises out of `finally`, leaving the primary error only in memory. The independent fixture reproduces an entered finalizer with a primary failure and zero final receipt writes. Protect this hash, append its failure, demote, and still attempt final receipt persistence. The parent has already added that protection plus an alternate receipt; repair verification follows separately.
2. **A started cycle is absent if `run_method` raises before returning.** The initial loop appends its method/repeat/path only after `S.run_method` returns. A bookkeeping exception escaping the inherited method collector produces one entered call but zero cycle records. The ordinary method collector catches most scientific exceptions, but this wrapper must also retain the actual attempted method/repeat/path when an unexpected exception escapes. Register the cycle before dispatch, then retain returned status or raised failure; do not retry it.

The positive controls confirm exact serial four-method/two-repeat enumeration with eight distinct output paths; a returned nonfinite status is persisted and stops later calls; a post-call pristine-input hash mismatch stops subsequent dispatch. Archived operand file hash and byte length precede `torch.load(..., map_location='cpu', weights_only=True)`; the exact eight input names and per-tensor hashes are checked, and the 48 layer identities are enumerated in order. There is no generator call in this entry.

Source binding uses the unchanged v3.5 identity preflight and a pinned freeze SHA-256 `9b3ea3f7e1d7472692f2336b4b0136f7a960a2f5bf88beae3e86713e306e4da3`. The freeze's source entries include the executor and cycle driver, so these imports are not merely trusted because of their filenames. The new decision replaces the held-out seed and adds two repeats/archive-origin metadata; method, topology, dtype policy and accepted lifecycle remain unchanged. Each reference seal is gate-hash-bound and must report eligibility, no C0 outputs and the exact held-out population. The final supplied gate and deployed files remain runtime admission facts, not observations made by this review.

The derivative `run_m1_q_c0_owned_v1.py` was reviewed as a diff against `run_m1_q_native_owned_v1.py`, without running either. Its substantive changes are the C0 stage/output namespace, four methods/two repeats, reference-seal and v3.5 bindings, the collector entry and gate/image-receipt mount, and Lumo environment. A stdlib AST comparison confirms all 14 `LUMO_ENV` values exactly equal `m1_adapters.py`'s `LUMO_REQUIRED_ENV`. The actual CID/image receipt is written after owned-container inspection and before start. The unchanged exclusive lock, one-use consumption, exact CID+nonce cleanup, contention checks and timeout lifecycle are not reopened here.

Reproduce the initial controls from the snapshot directory:

```sh
/Users/zhiyuanma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -B -m unittest -v reviewer_controls
```

No C0 launch is approved or performed by this note. Parent owns repairs, final source/gate binding, launch authority, and later numerical reduction. No held-out inputs or outputs were read by the independent fixtures.
