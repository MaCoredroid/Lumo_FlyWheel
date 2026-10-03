# Convolution publication observer v1: bounded source review

Disposition: byte semantics, bounded guard coverage, and the new v4 callback/O1 binding pass the source and saved-byte controls below. **One small strict-identity correction remains before treating the new witness as a fully validated offline record.** No GPU execution, source edits, launch review, gate approval, or qualification was performed. This review does not change the separate hydration-v4 acceptance.

## F1: reject impossible identity metadata

`q1_conv_publication_witness_v1.py:36` accepts any string as a device label. `plan()` validates positive `object_id` fields but never checks that repeated object IDs represent the same tensor identity. Independently reproduced from a valid 48-layer/16-bank plan: replacing every device label with `not-a-device`, or assigning the same object ID to two sources with different pointers, or two distinct bank classes with different pointers, each still returns a valid 16-group plan. These are malformed offline identities, not evidence that a live signature capture or numerical result was wrong.

Minimal repair: accept the same canonical CPU/CUDA device syntax used by the existing KV witness; build one identity map across banks and sources and reject a repeated object ID unless all recorded identity fields match. Legitimate repeated references to the same bank must remain valid, as must distinct tensor objects that are legitimate exact bank aliases. Add a valid same-object alias control and the three malformed controls above. This is metadata integrity hardening; do not change the byte oracle, source maps, numerical rules, or experiment scope.

## Accepted semantic checks

The ancestor-derived last-three source map equals the actual pinned `_fr13_fixed32_treeconv_expected_state_src` function evaluated independently through AST. All 28 declared paths satisfy it, including root-only handling. The source stage is row-major `[source_row, channel]`; persistent banks are `[row, channel, history]`. The kernel's leaf selection, SSI column-zero destination, live three columns, and zero-tail behavior agree with the observer (production `fr10_gdn_tree_kernel.py:6966–7050`). The current source accepts and preserves the actual 67-row B1 allocation; it checks the full captured source, including the inactive tail and positive-zero row 35. The earlier draft's 36-row restriction is already corrected.

Planning forms 16 exact physical alias classes of width three, rejects duplicate destinations within a class and partially overlapping distinct bank/source spans, and guards the union of running rows, null row, and immediate neighbors. The audit changes all three intended destination rows in each shared bank jointly before comparing the complete captured bytes; it does not mistakenly declare another legitimate layer's write to be corruption. Rows farther away are explicitly outside the claim. Shared attention storage does not require exempting attention writes here: the convolution witness brackets only this convolution operation, before recurrent replay and the later attention operation.

Capture performs signature reads and tensor-to-CPU copies; it has no in-place source/bank writes. The new hooks authenticate the actual preseed bank tuple, operand tensor identities/dtypes/shapes, target layer order, B1 capacity, and O0 running-row selection. Before/after operand metadata equality and raw source equality are enforced. Hook failures mark the process unusable; a missing witness fails the case seal. The patch inserts callbacks immediately before the one original convolution call and after it, immediately before the one original replay call. Removing the hook-marked lines reproduces the captured production source byte-for-byte; duplicate patching is refused.

The offline v3 auditor first authenticates the existing O1 state and validates the convolution witness, then `bind_o1()` joins every layer's physical destination to O1's transposed live-tap snapshot by row, pointer/offset/dtype/stride and actual materialized bytes. It does not rely only on matching digest summaries. The job and hooks bind the witness source hash. This is a source-only check of the new bridge; the wider future launch wiring and inherited helpers are outside this review.

## Independently executed controls

Standard-library execution of the actual pure observer and patcher (no Torch import) passed:

- A 48-layer/16-bank saved-byte fixture with real recorded bank strides, rows 53/54/55, 67-row source stages, and nonuniform per-channel two-byte data.
- O1 tap-major to physical DS-row equality for 2,949,120 live-tap bytes.
- Self-rehashed corruption of null/destination/guard bytes, active and inactive source bytes, and O1 materialized data all refused.
- Missing groups, changed operand metadata, rebound bank metadata, wrong source map, colliding destinations, and source aliasing refused.
- All 28 declared path plans and equality to the actual production source-map AST.
- Actual generated convolution/replay call bracketing, exact inverse source, and duplicate patch refusal.

The parent's separate Torch tests exercise real callbacks with synthetic CPU writes; they are not production Triton execution. I inspected those tests, but did not rerun Torch locally. No full-model or GPU qualification is inferred from either test set.

## Exact source snapshot

Files are retained under `p0/monitor/review-response-20260927/conv-publication-witness-v1-independent-review/current-snapshot/`; hashes/sizes are in `CURRENT-SNAPSHOT.json`. Pure controls and malformed-identity reproductions are preserved beside it.

- q1_conv_publication_witness_v1.py: `f6c62159852257ee2ff915209e1e46dd39eabb379cccb2a8fddcb054f01dcd23`
- q1_candidate_hooks_v4.py: `6ac2870ad178544fdf260237440802d27afbd6184b351e3877f532d9d86bf594`
- q1_patch_candidate_v4.py: `24c136e669033be4b06be1da52b5dafc98d5328e4f5ab3790f189d8f9b930302`
- q1_candidate_job_v4.py: `26c6a80471fada85d3e9555a8e0a980cd3bb0f74286bba4f4c45a4c411a47fc6`
- q1_candidate_raw_audit_v3.py: `7f78b7b69144a26471f9ba22725f2df566218ce7ead2d661d3859a7fb42c3f8b`
