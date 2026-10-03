# MTP KV v9 live connection: bounded independent review

2026-09-29. **No concrete blocker found in the reviewed hook/patcher delta for its declared B1, target metadata of 32 actual rows, no-extra-input-slot route.** This closes source-level connection review, not GPU qualification, launcher/job admission, or MTP numerical correctness. The unchanged pure-oracle review remains separate.

Reviewed SHA-256:

| Source | SHA-256 |
|---|---|
| `tools/q1_candidate_hooks_v9.py` | `905e35898a504357b8a96f2315395e0bec37d5b2509d2ae0a0709b827271317c` |
| `tools/q1_patch_candidate_v9.py` | `93894ce2758c8283f60dd90f64b364fd1efd6f310ab63ff3f5e25814928ab202` |
| `tools/q1_mtp_kv_publication_witness_v1.py` | `9239e35fa10395cd01e7e2a9a7c4a34b42fea753b1f0b064ac16909937ace976` |
| Reused live owner `tools/q1_hidden_publication_live_v1.py` | `e634c349434f29cbc5d5045fdd2098144a56871a4a66721a1ff85dfd0bab3eae` |

## Connection checks

- **Original operation preserved.** In-memory application to all three pinned generated sources compiles. Removing only the two MTP observer lines and reverting the v9 marker/import reproduces the v8 render byte-for-byte. AST inspection finds one unchanged `_fr13_mtp_kv1` call with identical arguments, immediately between the new before/after calls. The seams follow the actual first MTP forward and precede the production completion flags and payload clearing.
- **Owner and state binding.** `_mtp_operands` (hooks 663–693) joins the same case, stored runner, Eagle drafter and payload through the live hidden binding. `owner('forward_after')` deliberately requires the still-live payload, pending event and unset MTP completion fields at both adjacent seams. The cache must be the actual singleton registry object, the runner's cached MTP object and the payload's object. Shape/BF16, independent group lookup, drafter group ID, actual kernel/table/cache block units and resolved metadata-builder units are checked. MTP/target storage intervals must be disjoint under the existing conservative signature validator.
- **Restored map and independent table.** The actual active CPU request table is read using `num_blocks_per_row`; the common metadata's active device row must match. Query-start object identity, restored metadata slot-map object identity, cleared runner permutation stash, group/spans and consumed drafter-buffer pointer/stride are checked. The flat-map oracle then checks all 32 rows against independently derived logical slots and validates accepted paths/lens. No target inverse permutation is introduced.
- **Padding scope is coherent.** The helper receives the actual `_slot_mapping_buffer[:slot_mapping_size]` slice supplied to production. It supports a larger drafter graph slice with an inert -1 tail. The explicit `metadata.num_actual_tokens==32` condition is the inherited live hidden-producer scope; it does not force the separate padded `num_input_tokens` to equal 32. Extra input slots or a different target metadata shape are deliberately refused. This is not general mixed-batch or reshaped-metadata support.
- **Failure evidence and final join.** The before callback installs a partial record before fallible ownership/plan/capture work (699–708). The after callback retains that record and the captured after bytes before byte comparison (714–723). Both mark the process unusable and raise. The proposal exception scope contains errors from the real operation as well. The case sealer includes `mtp_publication`; `_problems` rejects missing/incomplete results. At the later real event seal, the MTP record is audited again against the independently reconstructed event owner, and a pending live capture refuses (742–743).

No claim follows that reading equal CPU/device table rows alone proves independent runtime ownership: the surrounding exact request/group/live registry checks supply that connection. Active blocks must come from the observed allocation count, not padded table columns. The code reads that bounded slice. Full-cache, null-block-outside-coverage, untested shape and numerical MTP claims remain excluded.

## Independent controls and limits

The companion `q1-mtp-kv-v9-connection-independent-audit.json` has SHA-256 `aaf641e692f31547a2e5d9224a1db44f8af755247bdf8f34271fc098fedc7526`. It records three byte-exact inverse-render checks, compilation, unchanged/adjacent production-call AST, and two actual callback AST controls with explicitly injected operand failures. Those controls retained the partial record or prior BEFORE evidence and raised `ProcessUnusable`; they did not substitute for the parent's connected Torch tests.

Rendered hashes: runner `1eb4cb97a1d44066e78735ec7917172904ea9cbdc8f470ac9d79299b4307f2f5`; rejection sampler `dc2eca29f1329fbdd56bcd75e3cd6f8a30535d504d3efe32ecb7bf56341d43e2`; Eagle `ff6c97fe5f3db9ce14361d3960d2c6716516ccb0fbdb41f37a29e7445a7330a0`.

The newly present `test_q1_mtp_v9_connected.py` was read: it connects actual hooks/capture with the extracted production Torch remap body, while explicitly excluding its CUDA/static validator, and exercises owner/map/units/restoration failures and corrupted after bytes. It was not run by this reviewer because this local Python lacks Torch. No GPU, model, Docker, remote, gate or implementation action was performed. First execution with the frozen model remains a runtime evidence boundary; source review does not predict that result.
