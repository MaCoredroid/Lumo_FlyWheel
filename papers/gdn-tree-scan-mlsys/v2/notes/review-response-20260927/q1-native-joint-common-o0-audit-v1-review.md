# Native joint common-O0 offline audit: bounded review

**One required fix before treating this new raw auditor as accepted.** The new MTP identity join accepts a destination MTP cache that aliases a target attention cache, contrary to the already-accepted live union importer's disjointness premise. This is an offline evidence-validation gap; no real run or live importer failure was observed.

Reviewed source `tools/q1_native_joint_common_o0_audit_v1.py`: SHA-256 `528bcacd57ab3b2933360e1b4002aeaef85cb0805e384da1ba96555e3cfe1bf3`, 8,573 bytes. Source copies and dependency hashes are retained in `p0/monitor/review-response-20260927/native-joint-common-o0-audit-v1-review/SNAPSHOT.json`.

## F1: independently owned MTP cache may alias the target in an accepted raw record

The new block at lines 85–91 checks the MTP descriptor against its before/after exports, positive owner IDs, generation, block units and the target group's block table. It does not check whether the resulting MTP address range overlaps any target convolution, recurrent-state or attention tensor. The live `q1_joint_common_o0_import_v1.Transaction.apply` explicitly checks this before planning/copying, using the conservative byte spans from `q1_native_mtp_owner_v2.py:27–38`.

A positive connected synthetic record passes the complete new auditor, with actual source seal/object authentication, target `NC.observation`/snapshots, combined-source digest construction, and the accepted full MTP bootstrap/continuation auditor; no audit function is mocked. Starting from that record, the powered negative replaces every destination MTP cache storage descriptor and `identity_snapshot.mtp.kv` with the first target attention descriptor and recomputes the changed destination bootstrap metadata digest and outer seal. It leaves the source, target states, MTP payload bytes, numerical decisions, allocation maps and union digests unchanged. The auditor still returns `valid=True`.

Concrete collision: target `language_model.model.layers.11.self_attn.attn` and MTP both have pointer `8589934592`, shape `[2,8,64,4,256]`, stride `[65536,131072,1024,256,1]`, BF16 on `cuda:0`; they are the same occupied byte range. Reproduction: `additional_controls.py`, artifact `destination_mtp_aliases_target.json` (file SHA `ebe894a323e9121dc423d1ce814b9d2333721fa5283a7d0fce99b3841d54737d`), outcome in `ADDITIONAL-RESULTS.json`.

Smallest repair: reconstruct the conservative MTP byte span from its strict captured shape/stride/dtype/address metadata, and require it disjoint from all captured target conv/SSM/KV tensor spans under the same device-aware rule already used by the live primitive. Apply this to full captured tensor views, matching the live premise. Do not simply require different storage pointers: nonoverlapping views of the same allocation are permitted by the accepted primitive. Add a positive disjoint shared-allocation view and negative exact/partial overlaps; preserve the old target guard block and existing numerical criteria.

## Checks that passed

The old target guard block, from `receipt.identity_unchanged` through exact target alias classes, is AST-identical to `q1_native_common_o0_audit_v2.py` (14 top-level statements; proof in `AST-GUARDS.json`). It is not re-reviewed here.

The connected positive uses full 48-GDN/16-attention synthetic states, full-vocabulary target and MTP logits, real content-addressed bytes, a synthetic 84-case/3-source manifest, complete natural MTP source history and separated destination bootstrap/import/continuation records. It validates the new target-versus-union digest distinction. Seventeen initial malformed controls and two further raw-object controls correctly refuse: missing/changed MTP identity/generation/readback; target digest substituted for union digest; changed source binding or source-history join; absent/bootstrap/continuation/completion evidence; claimed hidden transplant; invalid source population/mapping/path; changed MTP raw decision; changed MTP physical map; and independently missing/corrupted destination MTP hidden bytes. Together: one expected positive, nineteen expected refusals, and the one F1 false acceptance. The original synthetic source and object stores remain preserved.

The source manifest's launch/run provenance, caller's fixed job/gate and future driver are outside this module's scope. The synthetic manifest tests the selected-case resolver, not a real three-source collection. No GPU/model/container/remote operation or gate change occurred; no qualification count advanced.
