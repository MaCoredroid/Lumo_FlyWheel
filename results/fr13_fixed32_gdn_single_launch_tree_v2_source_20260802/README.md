# FR13 fixed32 GDN single-launch v2 source qualification

Status: **source-qualified only, default OFF, no GPU measurement**.

This is the corrected artifact for `fixed32_gdn_single_launch_tree_v2`. It does
not replace or rewrite the earlier v1 source-candidate artifact. The v2 route
has its own identity, live PASS schemas, and production selector; no v1 or
legacy batched-BV credential can authorize it.

## Kernel change

The logical fixed32 tree remains one five-node root path and eleven terminal
paths. One kernel launch now owns the full 32-node recurrence for each request,
value head, and `BV=8` value tile. It keeps the current root state and a branch
state live while interleaving each root node's terminal paths:

```text
node  0 -> paths 1,2  -> continue node 1
node  1 -> paths 3,4  -> continue node 4
node  4 -> paths 5,6  -> continue node 9
node  9 -> paths 7,8  -> continue node 14
node 14 -> paths 0,9,10
```

The candidate has one physical launch per layer, one physical program unit per
request/value-head/value-tile, zero state-export writes, and zero parent-export
reads. The equivalent legacy schedule remains explicitly reported as two
logical launches, twelve logical path programs, 82 logical padded slots, and a
logical critical path of 12. The candidate's physical recurrence critical path
is 32.

The source contract records a nominal register working set of 4,096 fp32 values
per CTA. With eight warps (256 threads), that is 16 fp32 values per thread, or
16,384 aggregate bytes per CTA before compiler temporaries and layout effects.
This is contract arithmetic, not a compiler allocation or spill claim.

## Qualification and production

B1 and B4 qualification capture and serve the incumbent fixed32 path route.
After an authenticated full-graph replay, the same process reruns incumbent and
candidate from the same baseline and requires raw-byte equality on:

```text
out, ring_k, ring_v, ring_a, ring_b, flags, counter
```

The candidate must also leave the pre-run export scratch byte-identical. The
gate restores the exact pre-run baseline in a `finally` path on pass, mismatch,
or candidate exception. Qualification never returns the shadow candidate.

B1 requires one real `swe_verified:<task_id>` marker and 48 layer records. B4
requires all four tasks in `config/fr13_fixed32/subset_b4_four.json`, each with
48 layer records. Only then can the v2 B1 or B4 PASS be emitted. Production
requires `FR13_FIXED32_GDN_SINGLE_LAUNCH_PRODUCTION=1` plus the matching v2 PASS,
whose identity binds mode, batch, source SHA-256, contract SHA-256, kernel name,
topology, authoritative surfaces, and physical launch/export counts.

## Observer contract

The runtime observer consumes `last_executed_gdn`, not the mere presence of a
candidate descriptor. Qualification therefore reports `fixed32_path`; qualified
production reports `fixed32_single_launch_tree`, one physical launch per layer,
and zero state-export writes. Legacy schedule counts are retained only with
explicit `logical_*` labels and `logical_fixed32_path_equivalent` semantics.

## Verification

- Focused single-launch and parent-group suite: 24 passed.
- Full fixed32 suite: 788 passed, 8 skipped, 3 environment failures.
- The three environment failures are separate from this change: two require the
  absent private `.venv/bin/python`; one requires the absent private no-symlink
  Hugging Face dataset cache.
- Python compilation and `git diff --check`: passed.
- The patcher's built-in observer self-test also fails on the same pre-existing
  conv-pregather fixture in the parent commit, before reaching GDN assertions;
  the unchanged parent source reproduces that failure.

No GPU command, synthetic probe, Triton compile, or performance timing was run
for this artifact. B1 and B4 real SWE-Verified raw-byte qualification, compiler
resource inspection, full-step TPS, phase breakdown, and hardware-floor
acceptance remain pending.
