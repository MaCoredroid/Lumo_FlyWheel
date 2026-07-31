# FR13 fixed32 batched GDN wide-BV source gate

This artifact binds the source implementation of the missing combined B2-B4
fixed32 batched-GDN plus wide-BV route. The implementation is default-off and
keeps the GDN path scan at exactly two physical kernel launches per layer for
BV16, BV32, BV64, and BV128. It does not contain a GPU or real SWE-Verified
result and makes no byte-parity, speed, acceptance, or hardware-floor claim.

## Source identity

- Base production-stack commit:
  `43d0398e513b6a1ffbf134dddc5dde0253e19995`
- Code commit: `df7494658ce160411ab7dfadffd6394895e0c61a`
- Branch: `agent/fixed32-b4-wide-bv`
- Remote: `origin/agent/fixed32-b4-wide-bv`

## Why the combination is valid

The fixed32 schedule has two dependency levels. The combined Triton kernel
folds request identity into grid axis 2, producing level grids `B` and `11*B`.
Changing B therefore changes the number of programs inside each launch, not
the number of launches. BV independently partitions the value dimension on
grid axis 1. GDN has no reduction across value lanes, so changing BV does not
alter request indexing, path dependencies, or launch count.

The counter writer remains one program per request on level 0. The flag writer
remains one program for the whole batched call. Output, compact state export,
and K/V/A/B rings are disjoint by request and value lane. The structural
contract rejects B outside 2-4, any topology other than the exact 32-row
physical tree, unsupported BV, BV larger than DIM_V, and BV that does not
divide DIM_V.

## Diagnostic contract

The combined diagnostic uses the batch-specific selector
`FR13_FIXED32_BATCH_GDN_BV_CANDIDATE=16|32|64|128` together with
`FR13_FIXED32_BATCH_GDN_BYTE_AB=1`. It is mutually exclusive with B1 path-BV,
batch production, and combined wide-BV production selectors.

For a nonzero real SWE-Verified event, every GDN layer:

1. snapshots all externally visible mutable surfaces;
2. runs the stock per-request BV8 route and snapshots its result;
3. restores the initial snapshot and runs the two-launch batched wide-BV route;
4. compares raw bytes for output, compact and untouched state export, K/V/A/B
   rings, flags, and invocation counter;
5. restores and serves the stock BV8 result, including after a mismatch.

Only after all 48 layers compare equal does the gate emit a source-bound v2
PASS record. Production requires the same source hash, mode, batch, physical
rows, candidate BV, all byte-surface names, and exactly two candidate launches
per layer. It raises on any drift and has no BV8 fallback.

The formal launcher permits this host-comparison path only as an exact4 B4
diagnostic with `MAX_NUM_SEQS=4`, `ENFORCE_EAGER=1`, and `FR10_METRICS=1`.
That diagnostic is explicitly not an acceptance or timing arm. With all new
selectors absent, existing B1 and B4 production pins remain unchanged; B1
continues to use stock BV8.

## Contemporaneous B1 risk

While this source artifact was being packaged, the separate live B1 path-BV64
gate reported a raw-byte mismatch on its first record's `export` surface. That
gate restored and served stock bytes and did not produce a PASS. This is
material evidence against assuming that BV64 is generally bit-exact.

- Evidence commit: `850355982`
- Evidence path:
  `results/fr13_fixed32_b1_gdn_bv64_byte_rejection_20260731/`
- Verdict: real SWE B1; GDN BV64 rejected; TAW remains unclassified.

It is not a result for this combined kernel: the B1 gate exercises the
per-request path kernel, while this candidate also changes request batching.
The B1 result therefore neither proves nor disproves B4 combined parity. It
does make the B4 exact4 diagnostic mandatory and leaves combined production
unarmed.

## Verification

- `bash -n scripts/fr13_launch_forked_fa2_tree_server.sh`: PASS
- Python compilation of kernel and new tests: PASS
- Ruff source check on the changed kernel and new test: PASS
- Focused GDN source suite: `63 passed`
- Broad fixed32 source-relevant suite: `589 passed, 8 skipped, 10 deselected`
- The 10 deselections are the known isolated-worktree `.venv` and local
  SWE-Verified dataset-cache prerequisite cases.
- GPU used: no
- Docker used: no
- Real SWE-Verified task run: no

The next valid evidence step is an exact4 B4 real SWE-Verified diagnostic at
BV64. A timing arm is not valid until the 48-layer v2 byte PASS exists.
