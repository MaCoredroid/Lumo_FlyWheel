# FR13 fixed32 corrected hardware-floor propagation

This is a source-only contract artifact. No GPU or SWE-Verified task was run.
It applies to new fixed32 measurement and acceptance runs; immutable historical
result directories were not rewritten.

## Canonical lower bound

- Mandatory weight bytes per fixed32 speculative event: `32,666,638,208`.
- Assumed bandwidth: `273,000,000,000 bytes/s`.
- Optimistic mandatory-weight-read floor: `119.658015414 ms/event`.
- Acceptance multiplier: `1.15`.
- Weight-bound one-sided U95 cap: `137.606717726 ms/event`.

The floor counts the target-model read, verifier-head read, five MTP-forward
weight reads, and five 64K drafter-head reads. It is not a measured complete
hardware floor. It assumes ideal mandatory-weight streaming and excludes
KV/state/activation traffic, attention, scan, sampling, committer work, kernel
launches, synchronization, allocator/runtime effects, and host gaps. Therefore
it is intentionally optimistic: real wall time cannot be expected to equal it.

## Active bindings

- `fr13_hardware_floor_ledger.py` owns the derived Python constants.
- `fr13_fixed32_floor_timers_seq.sh` exports the corrected floor unconditionally.
- `fr13_launch_forked_fa2_tree_server.sh` rejects any other fixed32 floor value.
- `fr13_measure.py` defaults to the corrected floor and reports its byte and
  bandwidth basis plus the weight-read-only limitation.
- `fr13_floor_gate.py` imports the ledger constants, binds the runtime manifest
  to the corrected environment, and uses the exact weight-bound U95 cap.
- `fr13_runtime_manifest.py` includes the ledger in the fixed32 source closure.

The legacy `98.6 ms` and `113.39 ms` values are prohibited by regression tests
in all four active measurement/acceptance contract paths.

## Verification

- Targeted fixed32 pytest suite: `60 passed`.
- Python compile checks: passed.
- Shell syntax checks: passed.
- Runtime-manifest self-test: passed.
- Ruff checks on touched Python files: passed with the repository's pre-existing
  unrelated `F841` in `fr13_measure.py` explicitly ignored.

The monolithic `fr13_floor_gate.py --self-test` is not a clean verification
signal at this source tip: both the unchanged base implementation and this
branch stop at the same pre-existing fixture error, `v3 provenance does not
match strict trace request evidence`. The targeted floor and runtime-contract
tests above pass independently of that fixture failure.
