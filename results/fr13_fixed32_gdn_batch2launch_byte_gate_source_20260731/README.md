# Fixed32 batched-GDN real-event byte gate

## Source identity

- Exact-safe base: `47de87295387cf25e900c566bfd3c8002df4d264`
- Rebased two-launch candidate: `f75c023c94973fef62ba5c35d84fdbb42f16e754`
- Real-event byte-gate code: `d642d7067fbf1a83cf205f281cf7fde33f903137`
- Branch: `agent/fixed32-gdn-batch2launch`

This is source and CPU/static evidence only. No GPU byte-parity result and no
performance result are claimed here.

## Kernel contract

- B1 remains on `launch_tree_gdn_prepared`; the batched API rejects B1.
- The normal B2-B4 candidate issues exactly two physical path-kernel launches
  per GDN layer. Missing fixed32 preseed, a disarmed subtree route, or an armed
  nested subtree selfcheck fails loudly; there is no silent fallback.
- The request index is folded into path-grid axis 2. B4 uses 20 of the existing
  32 fp32 export rows (`4 * 5`) and adds no capture-time allocation.

## Real-event gate

Boot the gate arm with all of:

```text
FR13_FIXED32_BATCH_GDN_BYTE_AB=1
ENFORCE_EAGER=1
FR10_METRICS=1
FR13_RING_EXPORT=1
FR13_FLAGS_INKERNEL=1
FR13_SUBTREE_PARALLEL=1
FR13_SUBTREE_PARALLEL_SELFCHECK=0
```

The launcher creates
`/logs/fr13_fixed32_batch_gdn_byte_ab.enabled` and removes any stale event
arm. After the server is ready, immediately before submitting a real
SWE-Verified task, create:

```bash
printf 'swe_verified:%s\n' "$TASK_ID" \
  > "$LOG_DIR/fr13_fixed32_batch_gdn_byte_ab.real_event.arm"
```

For each GDN layer on that first nonzero real event, the runtime:

1. Snapshots every mutable buffer.
2. Runs the legacy two-launch route separately for each request and gathers
   legacy export nodes `(0, 1, 4, 9, 14)` into logical `[B, 5]` order.
3. Restores the pre-event bytes and runs the two-launch batched candidate.
4. Byte-compares output, K/V/A/B rings, logical export scratch, untouched
   export tail, staging flags, and invocation counter. Each mismatch records
   the first differing byte.
5. Restores and serves the complete legacy result in both pass and mismatch
   cases. A layer switches to the candidate only after a zero-byte diff; a
   mismatch remains legacy-served and retries on the next real event.

The live gate is complete only when its JSONL contains one zero-diff record
for each of the 48 unique GDN layer keys, all bound to the same real
SWE-Verified task marker. The follow-on B4 speed run must use a fresh normal
graph boot with the diagnostic gate disabled.

## CPU/static verification

```text
pytest -q tests/test_fr13_fixed32_gdn_batch_byte_ab.py \
  tests/test_fr13_fixed32_gdn_batch_launch.py \
  tests/test_fr13_fixed32_gdn_exact_io.py
.........s
9 passed, 1 skipped in 1.12s

pytest -q tests/test_fr13_fixed32_gdn_schedule.py \
  tests/test_fr13_fixed32_gdn_batch_launch.py \
  tests/test_fr13_fixed32_gdn_batch_byte_ab.py
.........
9 passed in 1.12s
```

The skipped test is the preexisting CUDA/Triton gate on this CPU-only pass.
The mocked dual-route test covers both zero-diff and an injected candidate
mismatch and asserts that legacy outputs, rings, scratch, flags, and counter
are restored and served.

