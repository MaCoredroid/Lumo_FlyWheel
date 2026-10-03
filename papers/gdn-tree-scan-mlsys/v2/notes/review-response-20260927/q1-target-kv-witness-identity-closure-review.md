# Target-KV witness identity closure

**PASS for the bounded source/CPU identity repair.** This closes the required identity and duplicate-layer-view findings in `q1-target-kv-witness-v1-source-review.md`; the original review and failing controls remain preserved. It does not approve a runtime bridge, gate, machine operation, or full-model qualification.

Reviewed immutable snapshot under `p0/monitor/review-response-20260927/target-kv-witness-identity-closure-review/snapshot/`:

- `tools/q1_target_kv_witness_v1.py`: SHA256 `311b047bcc1bc59809d89f71baf27dc0f178c065d20d48358e408332a081d0d8`, 10,558 bytes.
- `tools/tests/test_q1_target_kv_witness_v1.py`: SHA256 `28b9b1973ff53495d1705dc85e81c0138285e7b1b152e90c6e5b257c1f2f8278`, 7,406 bytes.

The new `identities()` check requires the complete signature, strict integer address/size fields, bf16 pointer/offset consistency, positive injective strides within storage, unique object IDs, and nonoverlapping occupied bounding intervals per device. Both pre/post sets are checked before raw audit; capture checks real signatures before copying bytes. Shared allocation backing disjoint views is accepted. Interleaved views with overlapping bounds are conservatively unsupported, as documented. The planner, raw-copy equation, and scope flags are unchanged.

I independently reran all three original false-pass probes; missing identity fields, impossible zero identity/stride, and one layer duplicated under sixteen names now refuse. The prior 420 pure-byte positive cases and eight corruption negatives still pass. Eight additional focused negatives refuse overlapping views with different object IDs, boolean object ID, undersized storage, negative/zero/internal-overlap strides, inconsistent pointer offset, and missing device. Two positive controls accept sixteen disjoint views of one allocation and an injective transposed layout. Reproducers and JSON outputs are saved beside the snapshot as `check_stdlib.py`, `STDLIB-CONTROLS.json`, `check_identity_closure.py`, and `IDENTITY-CONTROLS.json`.

I inspected the supplied connected Torch test changes: block size 64 is added for a total 420 cases; layer names now use the target `language_model.model.layers.*` prefix; malformed identity, duplicate/overlapping view, disjoint-shared-storage, and capture-before-read checks are included. These connected tests were not independently executed here because local Python has no Torch. This verdict rests on source inspection and the independent standard-library controls above, not an inferred device result.

The separately evolving registry/hooks bridge still needs its own exact target-layer binding review. This note closes the pure witness's previous identity issue without asserting which live registry tensors a future caller supplies. Coverage remains tree-suffix blocks plus one allocated neighbor on each side; MTP and whole-cache/full-Q1 claims remain outside scope. No implementation edits or machine operations were performed.
