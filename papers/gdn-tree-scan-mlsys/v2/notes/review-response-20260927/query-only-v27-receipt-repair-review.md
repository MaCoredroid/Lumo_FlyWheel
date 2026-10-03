# Query-only v2.7 receipt repair: bounded closure

**PASS for the identified receipt-evidence repair.** The four false admissions in `query-only-v27-draft-source-review.md` now refuse, while the unchanged producer-generated positive receipt passes. No additional defect was found within this bounded helper review. This accepts the repaired source behavior; it is not a live-query authorization or a final package-freeze approval.

Reviewed local immutable snapshot: `p0/monitor/review-response-20260927/query-only-v27-receipt-repair-reviewed-20260928T202339Z/`. Runner SHA256 **`db30c0db380adedcf32a461f36dc7b8242fa2e08c6cfe24390535387f195adfa`** (274332 bytes); test source **`9f332c2d476045ac2807386a2db8922989640ab9247570331b5bd302da715a7c`** (219929 bytes); unchanged consumer **`2230ba1dea4551b30d5efad88516a7aa6ad3487c61880b9bd0984d6ca2cc2eb9`**. This is the post-assertion-conversion source; an AST check confirms no `assert` statement occurs in the new helper. Future final freeze must bind the same bytes, or account for any later source delta.

The repaired `query_only_receipt_problems` requires exactly one start (3409). New `query_command_evidence_problems` (3451–3528) checks one create/start and the required ownership/cleanup command records, exact argv/CID/image/nonce, safe retained raw filenames, file-manifest membership, actual stream SHA/size, raw query-result parsing and equality to the recorded result, and ordered exact-CID cleanup/absence evidence. Its `require` failures are explicit `ValueError` paths, so Python optimization does not disable validation. The full verifier invokes it for the query-only profile (3650), preserving the separate unchanged consumer contract.

| Control on an actual injected producer SUCCESS | Full verifier result |
| --- | --- |
| Unaltered connected receipt | PASS |
| Missing sole query-start command | Refused: exactly one start required |
| Start argv targets another CID | Refused: start targets another CID |
| Missing `query.ownership` | Refused: required ownership absent |
| Saved query stdout file removed | Refused: raw source missing |

The unchanged consumer also refuses the missing-start case through the repaired verifier, rather than merely relying on the standalone helper result. Repeated bounded producer controls passed for SUCCESS, insufficient capacity, invalid query JSON, uncertain cleanup, failed creation, target identity drift, and dependency exception. They preserve zero advice/workers/slab, one attempt at most, and consumed failure semantics. Wrong effective config still refuses; unapproved/wrong-source authorization remains unconsumed with no query.

These were local injected CPU checks using the same explicit seams as the initial review: the unchanged absolute pinned-input loader is replaced, the author's fake host/command dependencies provide runtime evidence, and consumer subprocess execution is replaced by a direct exact-verifier adapter. No live command, query, proc scan, cache action, container or GPU was used. The parent’s Linux 91-control plus connected-wrapper rerun remains separate final-package evidence; this note does not claim to have independently rerun that entire suite.

The first local positive attempt used macOS `/var` temporary paths, whose real path is `/private/var`; the helper's strict expected cidfile argv rejected that alias. The recorded successful reproduction uses a canonical temporary directory inside the review snapshot. This is disclosed as a path-portability limit, not a request to change the already canonical production output path or broaden the repair.

`review_control.py`, `REVIEW-CONTROLS.json`, `REVIEW-CONTROLS.log` and `repair.diff` preserve the exact bounded reproduction. Earlier operational authorities stay consumed, the old failed evidence is preserved, and no gate or implementation was edited by this reviewer. The prior source closures remain closed.
