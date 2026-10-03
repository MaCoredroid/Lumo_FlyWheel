# MTP KV byte oracle v1: bounded independent review

2026-09-29. **No concrete arithmetic/capture source blocker found for the declared pure-B1, no-extra-input-slot route.** Accepted as prospective CPU/source preparation only. The future v9 hook connection, runtime ownership checks, actual Torch capture and GPU operation are not qualified by this review.

Reviewed SHA-256:

| File | SHA-256 |
|---|---|
| `tools/q1_mtp_kv_publication_witness_v1.py` | `9239e35fa10395cd01e7e2a9a7c4a34b42fea753b1f0b064ac16909937ace976` |
| `tools/tests/test_q1_mtp_kv_publication_witness_v1.py` | `ad85e645f159d0ae0982f9a99c58e7d41df411dd723da664046f69792786b6d9` |
| `tools/q1_target_kv_witness_v1.py` | `311b047bcc1bc59809d89f71baf27dc0f178c065d20d48358e408332a081d0d8` |
| `tools/q1_hidden_publication_bridge_v1.py` | `00d30bc67e34027b09eae5a0386017ffc7f8766d0a7399218abb5920a15622b8` |
| `tools/q1_hidden_publication_witness_v1.py` | `92ac01369e8e337716183172ec8f64ec266b6601bca21384abef834361f2cd18` |

The exact source map and production hashes are preserved in `q1-mtp-kv-publication-source-map.md` (SHA `221f0b556e6a9606019de13b7cbfc34887afba782e12ae38344a01aa3fe0d240`).

## What matches production

`plan` checks the fixed parent chain, physical node mask, exact accepted path including its neutral tail, accepted length, B1 query `[0,32]`, injective request blocks and capacity. It derives each of the 32 consumed slots independently from the supplied logical block table and prefix, with no target permutation. It maps accepted node `a_j` from flat `P+a_j` to `P+j+1`.

This matches the real fixed1 wrapper's `dst_pi=None` delegation and the shared remapper's source/destination equations. `audit` constructs all expected copies from immutable BEFORE bytes, so overlapping source/destination chains are handled simultaneously. Including a self-copy for a contiguous accepted node is byte-equivalent to the production inactive mask. Root-only L=0 verifies unchanged bytes across the covered region. All captured non-destination rows, root, inactive depths 1–16 and neighbor-block rows remain part of the full after-byte equality check.

`capture` reads one `[2,N,block_size,4,256]` BF16 cache using explicit slot-major ordering and stores both K/V planes. The reused signature validator checks injective strides, pointer/storage offset and bounds. The reducer authenticates before/after SHA-256 and exact raw length, requires stable cache identity/owner/restoration/index metadata, re-derives the plan, and does not trust a recorded pass flag. Its reported limitations correctly exclude whole-cache and MTP numerical qualification.

## Caller assumptions and remaining connection boundary

The inert graph-tail rule is supported by the actual source: generated Runner 5493–5495 fills unused slot entries with -1, while Eagle 538–554 copies common metadata's map and pads any additional drafter tail. Supplying the **actual slice** `_slot_mapping_buffer[:slot_mapping_size]` is essential; supplying the entire preallocated buffer could include unrelated stale rows and falsely refuse. A 32-entry unpadded slice is supported, as is an observed padded slice with all remaining entries -1.

The exact query `[0,32]` is appropriate for this pure B1 route with no extra input slots. The bridge must check `needs_extra_input_slots is False`, actual B1 request ownership, and real `batch_indices is None`; the oracle deliberately does not generalize these modes. Eagle's extra-slot branch can replace the query/slot metadata and must remain unsupported here rather than being silently coerced into the B1 oracle.

The supplied `blocks` must contain exactly the actual allocated CPU block-table entries (`num_blocks_per_row`), not padded device-table columns. The planner bounds integer block IDs but does not itself prove request ownership or distinguish a supplied block-zero entry from an allocated block. The live bridge must bind the request/group, active CPU/device table row, actual physical block-size units, registry/cache/payload identities, and restoration metadata as specified in the source map. Unused table columns/null-block padding are not additional allocated request blocks. These are connection obligations, not evidence produced by the pure record auditor.

Likewise, passing `expected_owner=record['owner']` is suitable only for the supplied synthetic fixture test. A live reducer must supply the independently expected owner and fixture inputs and bind the record to actual seams. Before capture must occur after the first MTP forward; immediate after capture occurs before production completion flags/payload clearing. Failure-record retention and fail-stop behavior belong to that connection and are not implemented or assessed by this pure helper.

## Independent CPU checks

Ran `PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s papers/gdn-tree-scan-mlsys/v2/experiments/review-response-20260927/tools/tests -p test_q1_mtp_kv_publication_witness_v1.py -v`: **5 methods passed**, including 16 positive root/branch × 64/1024 × four boundary-offset cases, inverse-permutation refusal, rehashed wrong-row/root/neighbor corruption, owner/index/restoration/cache-identity failures, alias/path/padding refusals.

These tests use synthetic raw bytes. They do not execute the capture function's Torch path or the production remapper. Source review independently checked the capture indexing and byte equations against the pinned caller/kernel; no CUDA, container, model, remote, or implementation/gate mutation occurred.
