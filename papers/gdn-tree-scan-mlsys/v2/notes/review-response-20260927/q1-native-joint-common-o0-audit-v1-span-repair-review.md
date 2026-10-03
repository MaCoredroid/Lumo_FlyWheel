# Joint common-O0 span repair: bounded follow-up

Reviewed auditor SHA `5dc47f83f6dd3cf7ea4e0a30a8ca0986adfc2f07024f07e1fe0d61b7c2b7174d` (10,064 bytes). Original failure evidence and original review remain unchanged.

The new span check closes the original exact alias and a partial overlap. The retained connected positive and nineteen previous refusal controls still behave as expected. A new disjoint shared-allocation view passes. Ten malformed geometry/storage controls refuse (offset, pointer equation, storage size, negative/bool/float stride, zero/float shape, dtype and base). Together the new span suite has one expected positive, eleven expected refusals including partial overlap, and two remaining false acceptances.

**One same-boundary repair is still required:** `span()` returns the receipt-only `ident['device']` value without validation. The disjointness predicate accepts unequal labels without inspecting addresses. The original identical-cache repro passes again when only the MTP identity's device label is changed to `cuda:00` or `not-a-device`. Exported raw KV metadata contains no device field, so the existing export/identity equality does not authenticate this label. All other record identities, bytes, numerical results and seals are valid in the synthetic reproduction.

Require canonical device labels and the single device for target/MTP cache views supported by this native owner/import scope, then compare the same conservative intervals. Do not reject legitimate disjoint shared-allocation views. This is closure of the new overlap check, not a new numerical criterion or expanded experiment.

Evidence: `p0/monitor/review-response-20260927/native-joint-common-o0-audit-v1-repair-review/` retains the exact repaired source, copied predecessor control scripts, their new logs/results, `span_controls.py`, `SPAN-RESULTS.json`, and both resealed device-label reproductions. CPU only; no GPU/model/container/remote action and no gate approval.
