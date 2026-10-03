# Memory recovery v2.1: bounded closure review

**HOLD: F1–F4 closed within the reviewed scope; F5 remains open.** The repaired draft is not ready for final source-freeze acceptance. This is a source review, not execution authorization. No reclaim, process cleanup, container operation, remote command, GPU operation, or real runner entry point was executed.

Reviewed immutable snapshot: `p0/monitor/review-response-20260927/memory-v21-reviewed-20260928T0450Z`, relative to paper v2.

| Reviewed item | SHA-256 |
|---|---|
| `FREEZE-MEMORY-RECOVERY-RUNNER-v2.1.json` | `8c4a0fffa346497afe6f6427f93442811fe3bcf29c06a73c561555a924e9c2a6` |
| `tools/memory_recovery_v2.py` | `17d9c4453bbea247f0e3c60cb122859aee9e529806a5196c893066e04dac89e2` |
| `tools/tests/test_memory_recovery_v2.py` | `8ebd86c4129255dc8a4bd6619ccb8a4bd436d8c7c5876eac4e98bde28b3688eb` |
| Author CPU test log, attempt 2 | `60e909d3616a6214cbc8c9c44c503be130f34262ad1521dc2519fa20901bd8e1` |

Independently verified all 10 freeze members' sizes and hashes: zero mismatches. The author's 105-test result is retained as author evidence; it was not rerun here. Independent checks used Python standard-library imports and in-memory method stubs only, bypassing the constructor and entry point.

## Closed original findings

| Finding | Bounded closure evidence in the pinned runner |
|---|---|
| F1: unowned query cleanup | Lines 1159–1200 and 1203–1318 establish a full CID, pinned image and nonce, start by CID, recheck ownership before removal, and require completed CID-absence evidence. Failed/ambiguous creation places a hold. The old name-based removal is gone. Pure ownership controls reject a foreign nonce and accept the matching CID/image/nonce. |
| F2: one-use authority | Lines 735–813 bind the authority namespace and lock before inventory. Lines 853–900 recheck authorization bytes, expiry, gate, holds and reservation, then exclusively persist the consumed marker before the write; lines 574–578 refuse a write without it. Unknown write/query conditions retain a hold. Changing output roots no longer creates another authority namespace. The finalization exception described below remains F5, not a reopening of this pre-write closure. |
| F3: effective configuration | Lines 457–489 and 758–770 bind the operational configuration, including reservation/timeouts, source and allowlist inputs, expected driver/kernel/device/total and command policy. Lines 369–406 validate the query against the authorized device contract. The parent still owns the exact future issued configuration. |
| F4: inventory and counters | Lines 355–366 reject malformed compute headers; 1020–1021 reject successful-but-empty fuser output; 1112–1153 test both post-action checkpoints. Independent controls reject `pid GARBAGE` and detect an after-ten-second page-cache delta of one. |

## F5 remains: final receipt can fail after releasing the only authority-wide hold

The corrected ordering handles a failed **lock release** before constructing the final receipt, but not a failed **final receipt write**:

- `_finalize` releases the authority lock at **1476–1480** before the final write at **1501**.
- Its final-write exception handler at **1502–1505** returns 12 without creating a persistent hold or retaining/reacquiring the lock. A crash in that interval has the same missing-finalization-marker problem.
- The consumed marker remains durable and blocks reuse of the **same** authorization. It does not supply an authority-wide hold against a separately issued authorization using another marker; `_check_no_holds` at **804–809** sees no hold from this unfinished finalization.
- `atomic_write_bytes` does `os.replace` at **235**, then directory `fsync` at **236**. If that final fsync fails, a visible `RECEIPT.json` can already contain `outcome=SUCCESS`, `exit_code=0`, and `next_boot_permitted=true`, although `_finalize` returns 12. The assertion at 1503–1504 that only the preliminary receipt exists is therefore false for this ordinary failure point.

Independent in-memory replay of the actual `_finalize` and `_boot_hold` methods, with all persistence redirected to a dictionary and release simulated, produced:

| Injected condition | Return | Lock held | Holds | Final receipt |
|---|---:|---|---:|---|
| No fault, positive control | 0 | false | 0 | SUCCESS; exit 0; boot permitted true |
| Final write fails before replace | 12 | false | 0 | absent |
| Final write fails after replace, during directory fsync | 12 | false | 0 | **SUCCESS; exit 0; boot permitted true; no seal errors** |

All three preserve the same already-written consumed marker. No real filesystem write or operation was needed to reproduce the remaining failure.

**Minimal repair:** create a durable authority-wide in-progress finalization hold before releasing the lock, retain it on any incomplete/unknown seal, and clear it only after the terminal receipt is durably sealed. The readiness consumer must honor that hold, so a visible receipt from a failed or incomplete seal cannot admit a boot; a new authorization must also remain blocked. Do not make a success receipt independently authoritative before completion is proven. Add only focused failure-order controls: before final replace, after replace/before successful directory fsync, interruption between release and final seal, and the clean terminal path. Preserve this attempted source/freeze and the prior consumed authorization.

### Minimal independent reproduction seam

Run only as a standard-library in-memory check; this deliberately bypasses `Recovery.__init__` and never invokes `main`, commands, or actual persistence. From the snapshot directory, import the module under a non-main name with bytecode disabled; set `hash_dir` to return `{}`, `sha256_file` to return the known virtual reservation digest, and `os.path.isfile` to return true. Construct `r = object.__new__(Recovery)` with successful write/postcondition/query state, a written reservation, `lock_held=True`, no holds or seal errors, and no-op milestones. Use the actual `_boot_hold` inside a small receipt projection. The decisive injection is:

```python
stored = {}
def release():
    r.lock_held = False
    return True, None
def atomic(path, data):
    stored[path] = json.loads(data)  # models successful replace
    if path.endswith('/RECEIPT.json'):
        raise OSError('parent-directory fsync failure after replace')
r._release_lock = release
r._place_hold = lambda *a, **k: r.holds_placed.append({'path': '/virtual/hold'})
module.atomic_write_bytes = atomic
rc = r._finalize('SUCCESS', 'bounded query passed')
assert rc == 12 and not r.lock_held and not r.holds_placed
final = stored['/virtual/run/RECEIPT.json']
assert final['exit_code'] == 0
assert final['boot_hold']['next_boot_permitted'] is True
```

This report closes no scientific, operational or launch gate. Only the above F5 repair needs another bounded source review; F1–F4 need not be expanded or rerun as a broader campaign.
