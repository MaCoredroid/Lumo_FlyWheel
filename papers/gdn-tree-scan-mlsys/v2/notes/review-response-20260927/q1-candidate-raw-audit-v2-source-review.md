# Candidate raw auditor v2: witness-to-O1 binding review

**Required repair: bind the witness's request-block map and overlapping materialized bytes to O1.** The witness's internal byte-copy check and the O1 raw-state check are individually authenticated, but their common pointer metadata is currently insufficient to establish that they describe the same materialized cache content.

Reviewed immutable source snapshot: `p0/monitor/review-response-20260927/candidate-raw-audit-v2-independent-review/snapshot/`. Exact hashes:

- Auditor `q1_candidate_raw_audit_v2.py`: `eafb4183fa683a2456530a53a57210f6c0d94ccab5699150da9cb57867b4b9fb`.
- Witness `q1_target_kv_witness_v1.py`: `311b047bcc1bc59809d89f71baf27dc0f178c065d20d48358e408332a081d0d8` (previously accepted identity repair unchanged).
- Hooks `q1_candidate_hooks_v3.py`: `72a399fcddb80255d7eff9b650cc3deb88a1a48d4ee1f814721a3b297fb5bdf2`.
- Job builder `q1_candidate_job_v3.py`: `e72f32260a790dee192cad95a3c80bbd0c7b137a4fb6c57608ee22cff4e450fd`.
- Reused native reader: `d4502b7f47ab1ac6ed8cd295aa5565726fad486c4a1ff1bda6a13fe24e4ea770`.

## Concrete false passes

The auditor invokes `KW.audit()` at line 131, then lines 132–135 compare only `ptr`, `shape`, `stride`, `storage_offset`, and `dtype` with O1. It never compares `witness.inputs.blocks` with O1's `physical_block` fields or compares witness-after bytes with the raw O1 K/V objects.

I exercised the actual `audit_record()` and unchanged native reader using full-geometry synthetic raw objects, valid content hashes, case seals, job hash, native bindings, and a root-only case. The positive control passed. Both following altered controls also returned `raw_authenticated=true` and `target_kv_bytes_match=true`:

1. Set the witness's materialized root K value to bf16 `1.0` in both pre/post objects while the same logical O1 K value remains zero. A root-only publication legitimately copies no accepted drafts, so its internal byte-copy check passes. The two snapshots nevertheless disagree on the materialized root.
2. Change the witness's allocated block map to `[2,3]`, recomputing its slot map and oracle plan, while O1's materialized logical block still points to physical block `1`. Geometry and tensor addresses remain identical. The offline auditor accepts the conflicting request-block maps.

This does not allege that either inconsistency occurred on a GPU. It demonstrates that the prospective offline check can accept inconsistent evidence, including an accidental wrong request-block capture or misindexed raw snapshot, despite its existing identity checks.

## Minimal closure

For every attention layer and each O1 materialized logical block intersecting the witness's captured logical interval:

- Require O1's physical block to equal `witness.inputs.blocks[logical_index]`.
- Compare each materialized K and V row with the corresponding witness-after slot-major row. O1 stores K/V in separate block or valid-tail objects; the witness stores each slot as `[K,V,heads,head_dim]`. Derive offsets from the validated shapes and logical positions, not supplied result flags.
- Authenticate the O1 bytes used by this comparison and retain exact tail validity. Do not compare unmaterialized suffix rows against absent O1 data, or expand the claim to uncaptured cache regions.

Add a positive comparison that crosses a block boundary and includes a partial tail, plus the two altered controls above and K/V-plane swapping. The pointer/identity checks should remain as an additional condition. No comparator tolerance, numerical threshold, model setting, or experimental scope change is needed.

## Accepted parts of this bounded review

The job builder adds revision 3 plus exact hooks/patcher/registry/witness hashes and recomputes the existing canonical job seal. The offline job validator requires revision 3 and the witness source hash. Record identity, case/control, expected observation ID, caller-bound job hash, and native binding remain checked before the witness. The witness itself is mandatory; its saved audit flag is not trusted, and actual raw objects are reread and hash/length checked. Missing witness, foreign observation ID, missing raw file, corrupted raw content, wrong completion-counter delta, and bad pointer metadata all refused in independent CPU controls. Completion-counter delta is newly recomputed rather than merely reported.

The inspected hook seam obtains the request block table and checks actual registry-bound tensors before capture (`q1_candidate_hooks_v3.py:410`), records unique pre/post captures, and performs its own byte audit. `on_sealed()` separately captures O1 after the completion event (`:455`). Those live source checks do not replace the requested offline cross-record byte consistency check. This note does not approve patcher placement, launch integration, or a runtime result.

Evidence: `p0/monitor/review-response-20260927/candidate-raw-audit-v2-independent-review/{check_binding.py,BINDING-CONTROLS.json,SNAPSHOT.json}`. Controls exercise one synthetic per-record path through the actual auditor, not a complete 84-case job. No Torch, GPU, container, cache, gate, launcher, or implementation operation was performed. Parent-reported connected tensor/hook tests are separate evidence; they were not independently rerun in this local no-Torch environment. All prior accepted checks remain closed.
