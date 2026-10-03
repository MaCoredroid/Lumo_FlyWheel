# Native joint common-O0 auditor: bounded repair closure

**PASS for the reviewed raw-audit source.** Both original F1 and its device-label residual are closed. The original failed source snapshots, resealed false-positive records and review notes remain preserved.

Accepted auditor `tools/q1_native_joint_common_o0_audit_v1.py`: SHA-256 `4f0703bdda80303eb5b520aead18469782da1fbf6e499465caee04f72e2d0aaa`. The latest delta from the span-repair snapshot is exactly the canonical `cuda:0` device requirement inside `span()`, matching this fixed TP1 native deployment. This changes an evidence-validity check, not model or numerical settings.

The auditor now reconstructs conservative occupied byte spans with strict dtype, positive integer shape, nonnegative integer stride, base/offset/pointer equation and storage bounds; requires the native MTP geometry; and refuses overlap between MTP and every target conv/SSM/KV tensor. Different or malformed device labels cannot bypass the check. A nonoverlapping MTP view in the same backing allocation remains accepted. These are the live union importer's existing span semantics. The prior target pointer/map/alias guard block remains unchanged, with its previously saved AST identity proof; all seven other dependencies in the original review snapshot still hash-match.

All **35 connected CPU checks** matched their expected outcomes: two positives (ordinary valid record and disjoint shared storage) and 33 refusals. The refused cases include the original exact-target alias, partial overlap, both device-label reproductions, malformed span geometry/addressing, missing/corrupt destination raw bytes, and retained source/union/history/owner/map omission and inconsistency controls. The raw target snapshots, full-vocabulary logits, full MTP source history and destination segments are actually read and authenticated from the retained synthetic payloads; the auditor functions are not mocked. The synthetic manifest is not evidence of a real three-source run.

Evidence is retained separately at `p0/monitor/review-response-20260927/native-joint-common-o0-audit-v1-device-closure/`: exact source, unchanged control scripts, raw positive/negative fixtures, three logs/results files and `REVIEW-SEAL.json`. Source and dependency hashes were rechecked when sealing.

Scope remains raw-record validation only. The caller still supplies independently authenticated run/job/source-manifest/gate provenance and must handle exceptions as refusal. This review performed no GPU/model/container/remote actions, opened no gate and advanced no qualification count.
